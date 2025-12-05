"""
Batch evaluation of PAC-Bayes bounds for all trained SubLoRA checkpoints.

This script finds all best_ckpt.pt files in a results directory and evaluates
bounds for each one, saving results alongside the checkpoint.

Usage:
    python experiments/eval_bounds_batch.py --results_dir results/hpc-out-d10000-d20000
    
    # With custom config
    python experiments/eval_bounds_batch.py \
        --results_dir results/hpc-out-d10000-d20000 \
        --config_file config/sublora_bounds.yaml \
        --dataset_dir data/openwebtext
        
    # Evaluate only specific experiments
    python experiments/eval_bounds_batch.py \
        --results_dir results/hpc-out-d10000-d20000 \
        --filter "d10000-uniform"
"""

import os
import sys
import glob
import argparse
import yaml
from pathlib import Path
from datetime import datetime


def find_checkpoints(results_dir, filter_pattern=None):
    """
    Find all best_ckpt.pt files in the results directory.
    For each experiment folder, find the most recent checkpoint.
    
    Returns list of tuples: (experiment_name, checkpoint_dir, checkpoint_path)
    """
    checkpoints = []
    results_path = Path(results_dir)
    
    # Find all sublora-d* experiment folders
    exp_folders = sorted(results_path.glob("sublora-d*"))
    
    for exp_folder in exp_folders:
        exp_name = exp_folder.name
        
        # Apply filter if specified
        if filter_pattern and filter_pattern not in exp_name:
            continue
        
        # Find all best_ckpt.pt files in this experiment
        ckpt_files = list(exp_folder.rglob("best_ckpt.pt"))
        
        if not ckpt_files:
            print(f"  [SKIP] {exp_name}: No best_ckpt.pt found")
            continue
        
        # If multiple checkpoints exist, find the most recent one
        # by looking at the parent directory names (date/time structure)
        if len(ckpt_files) > 1:
            # Sort by modification time to get most recent
            ckpt_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        best_ckpt = ckpt_files[0]
        ckpt_dir = best_ckpt.parent
        
        # Check if bounds already computed
        bounds_file = ckpt_dir / "bounds_levels11_iters0.yml"
        if bounds_file.exists():
            print(f"  [DONE] {exp_name}: Bounds already computed")
            continue
        
        checkpoints.append((exp_name, str(ckpt_dir), str(best_ckpt)))
    
    return checkpoints


def extract_config_from_checkpoint(checkpoint_path):
    """Extract intrinsic_dim and other config from checkpoint path or file."""
    import torch
    
    # Try to load checkpoint to get config
    try:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        config = ckpt.get('config', {})
        intrinsic_dim = config.get('sublora', {}).get('intrinsic_dim', None)
        
        # Also try to extract from path if not in config
        if intrinsic_dim is None:
            # Path like: .../id10000_lr0.0005_r4/...
            path_str = str(checkpoint_path)
            if 'id' in path_str:
                import re
                match = re.search(r'id(\d+)_', path_str)
                if match:
                    intrinsic_dim = int(match.group(1))
        
        return {
            'intrinsic_dim': intrinsic_dim,
            'config': config
        }
    except Exception as e:
        print(f"    Warning: Could not load checkpoint config: {e}")
        return {'intrinsic_dim': None, 'config': {}}


def run_bounds_evaluation(checkpoint_dir, config_file, dataset_dir, 
                          eot_indices_file, doc_lengths_file, intrinsic_dim=None):
    """Run bounds evaluation for a single checkpoint."""
    from fastargs import get_current_config, Param, Section
    from fastargs.decorators import param
    from sublora.sublora_pipeline import SubLoRA
    
    # Reset fastargs config for each run
    # This is a workaround since fastargs maintains global state
    import importlib
    import sublora.sublora_pipeline
    importlib.reload(sublora.sublora_pipeline)
    
    # Build command line args
    args = [
        sys.argv[0],
        f'--config-file={config_file}',
        f'--model.best_checkpoint_path={checkpoint_dir}',
        f'--data.dataset_dir={dataset_dir}',
        f'--data.openwebtext_train_eot_indices_file={eot_indices_file}',
        f'--data.empirical_document_length_distribution_file={doc_lengths_file}',
        '--login.wandb_log=False',  # Disable wandb for batch evaluation
    ]
    
    if intrinsic_dim is not None:
        args.append(f'--sublora.intrinsic_dim={intrinsic_dim}')
    
    # Override sys.argv for fastargs
    old_argv = sys.argv
    sys.argv = args
    
    try:
        config = get_current_config()
        from argparse import ArgumentParser
        parser = ArgumentParser(description="SubLoRA GPT-2 bound evaluation")
        config.augment_argparse(parser)
        config.collect_argparse_args(parser)
        config.validate(mode="stderr")
        
        yaml_config = {key[1]: value for key, value in config.content.items()}
        yaml_config["action"] = "eval_bounds"
        
        method = SubLoRA(yaml_config)
        method.get_bounds()
        
        return True
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        sys.argv = old_argv


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate PAC-Bayes bounds for all trained SubLoRA checkpoints"
    )
    parser.add_argument(
        '--results_dir', 
        type=str, 
        required=True,
        help='Directory containing sublora-d* experiment folders'
    )
    parser.add_argument(
        '--config_file',
        type=str,
        default='config/sublora_bounds.yaml',
        help='Base config file for bounds evaluation'
    )
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='data',
        help='Base directory for datasets (code appends openwebtext/)'
    )
    parser.add_argument(
        '--eot_indices_file',
        type=str,
        default=None,
        help='Path to eot_indices.npy (default: dataset_dir/openwebtext/eot_indices.npy)'
    )
    parser.add_argument(
        '--doc_lengths_file',
        type=str,
        default=None,
        help='Path to doc_lengths.npy (default: dataset_dir/openwebtext/doc_lengths.npy)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        default=None,
        help='Only evaluate experiments matching this pattern (e.g., "d10000-uniform")'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        help='Only list checkpoints without evaluating'
    )
    
    args = parser.parse_args()
    
    # Set default paths for eot_indices and doc_lengths
    # Note: These files are in dataset_dir/openwebtext/ alongside train.bin
    if args.eot_indices_file is None:
        args.eot_indices_file = os.path.join(args.dataset_dir, 'openwebtext', 'eot_indices.npy')
    if args.doc_lengths_file is None:
        args.doc_lengths_file = os.path.join(args.dataset_dir, 'openwebtext', 'doc_lengths.npy')
    
    # Validate paths
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.config_file):
        print(f"Error: Config file not found: {args.config_file}")
        sys.exit(1)
    
    print("=" * 60)
    print("SubLoRA Batch Bounds Evaluation")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Config file: {args.config_file}")
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Filter pattern: {args.filter or 'None (all experiments)'}")
    print()
    
    # Find all checkpoints
    print("Scanning for checkpoints...")
    checkpoints = find_checkpoints(args.results_dir, args.filter)
    
    if not checkpoints:
        print("\nNo checkpoints found to evaluate!")
        sys.exit(0)
    
    print(f"\nFound {len(checkpoints)} checkpoint(s) to evaluate:")
    for exp_name, ckpt_dir, ckpt_path in checkpoints:
        print(f"  - {exp_name}")
        print(f"    {ckpt_dir}")
    
    if args.dry_run:
        print("\n[DRY RUN] Exiting without evaluation.")
        sys.exit(0)
    
    # Check data files exist
    if not os.path.exists(args.eot_indices_file):
        print(f"\nError: EOT indices file not found: {args.eot_indices_file}")
        print("Please provide --eot_indices_file or ensure data is prepared.")
        sys.exit(1)
    
    if not os.path.exists(args.doc_lengths_file):
        print(f"\nError: Document lengths file not found: {args.doc_lengths_file}")
        print("Please provide --doc_lengths_file or ensure data is prepared.")
        sys.exit(1)
    
    # Evaluate each checkpoint
    print("\n" + "=" * 60)
    print("Starting bounds evaluation...")
    print("=" * 60)
    
    results = []
    for i, (exp_name, ckpt_dir, ckpt_path) in enumerate(checkpoints, 1):
        print(f"\n[{i}/{len(checkpoints)}] Evaluating: {exp_name}")
        print(f"    Checkpoint: {ckpt_dir}")
        
        # Extract config from checkpoint
        ckpt_config = extract_config_from_checkpoint(ckpt_path)
        intrinsic_dim = ckpt_config.get('intrinsic_dim')
        print(f"    Intrinsic dim: {intrinsic_dim}")
        
        start_time = datetime.now()
        success = run_bounds_evaluation(
            checkpoint_dir=ckpt_dir,
            config_file=args.config_file,
            dataset_dir=args.dataset_dir,
            eot_indices_file=args.eot_indices_file,
            doc_lengths_file=args.doc_lengths_file,
            intrinsic_dim=intrinsic_dim
        )
        elapsed = datetime.now() - start_time
        
        status = "SUCCESS" if success else "FAILED"
        print(f"    Status: {status} (took {elapsed})")
        results.append((exp_name, success, elapsed))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    successes = sum(1 for _, success, _ in results if success)
    failures = len(results) - successes
    
    print(f"Total: {len(results)}")
    print(f"Success: {successes}")
    print(f"Failed: {failures}")
    
    if failures > 0:
        print("\nFailed experiments:")
        for exp_name, success, _ in results:
            if not success:
                print(f"  - {exp_name}")


if __name__ == "__main__":
    main()
