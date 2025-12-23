from argparse import ArgumentParser
from fastargs import get_current_config
from sublora.sublora_pipeline import SubLoRA
import os
import yaml


def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description="SubLoRA GPT-2 bound evaluation")
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    if not quiet:
        config.summary()
    return config


def save_bound_decomposition(yaml_config, fp_loss_train, fp_loss_val,
                             quant_loss_train, quant_loss_val,
                             best_bound, complexity_term, prefix_message_len,
                             output_dir):
    """
    Save bound decomposition components to YAML file.

    Args:
        yaml_config: Configuration dictionary
        fp_loss_train: Full precision training loss (before quantization)
        fp_loss_val: Full precision validation loss
        quant_loss_train: Training loss after quantization
        quant_loss_val: Validation loss after quantization
        best_bound: Best BPD bound value
        complexity_term: Complexity term from bound computation
        prefix_message_len: Compression size in bits
        output_dir: Directory to save decomposition file
    """
    decomposition = {
        'fp_bpd_train': float(fp_loss_train),
        'fp_bpd_val': float(fp_loss_val),
        'quant_bpd_train': float(quant_loss_train),
        'quant_bpd_val': float(quant_loss_val),
        'quantization_loss': float(quant_loss_train - fp_loss_train),
        'complexity_term': float(complexity_term),
        'total_bound': float(best_bound),
        'compression_bits': float(prefix_message_len),
        'budget': yaml_config.get('sublora', {}).get('intrinsic_dim', None),
        'allocation_config': yaml_config.get('sublora', {}).get('allocation_config', None),
    }

    # Determine allocation type for filename
    alloc_config = decomposition['allocation_config']
    if alloc_config is None or alloc_config.get('mode') == 'uniform':
        alloc_name = 'uniform'
    elif alloc_config.get('mode') in ['learned', 'learned_gating']:
        alloc_name = 'learned'
    elif alloc_config.get('mode') == 'fixed_asymmetric':
        ratio = alloc_config.get('a_to_b_ratio', 1.0)
        if ratio > 1.5:
            alloc_name = 'a_heavy'
        elif ratio < 0.67:
            alloc_name = 'b_heavy'
        else:
            alloc_name = 'equal'
    else:
        alloc_name = alloc_config.get('mode', 'unknown')

    budget = decomposition['budget']
    filename = f'bound_decomposition_d{budget}_{alloc_name}.yml'
    filepath = os.path.join(output_dir, filename)

    with open(filepath, 'w') as f:
        yaml.safe_dump(decomposition, f, indent=2)

    print(f"\nSaved bound decomposition: {filepath}")
    print("\n=== Bound Decomposition ===")
    print(f"FP BPD (Train):      {decomposition['fp_bpd_train']:.4f}")
    print(f"Quantization Loss:   {decomposition['quantization_loss']:.4f}")
    print(f"Complexity Term:     {decomposition['complexity_term']:.4f}")
    print(f"Total Bound:         {decomposition['total_bound']:.4f}")
    print(f"Compression (bits):  {decomposition['compression_bits']:.1f}")
    print("===========================\n")

    return filepath


if __name__ == "__main__":
    yaml_config = make_config()
    yaml_config = {key[1]: value for key, value in yaml_config.content.items()}
    yaml_config["action"] = "eval_bounds"

    # If a checkpoint path is provided, prefer any allocation metadata saved
    # inside the checkpoint so evaluation constructs the model identically
    # to training. This preserves learned allocations and any explicit caps.
    ckpt_path = yaml_config.get('model', {}).get('best_checkpoint_path') or yaml_config.get('model.best_checkpoint_path')
    if ckpt_path:
        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location='cpu')
            alloc = ckpt.get('allocation_config', None)
            if alloc is None:
                # try inside saved training config
                cfg = ckpt.get('config', {})
                alloc = cfg.get('sublora', {}).get('allocation_config', None)
            if alloc is not None:
                yaml_config.setdefault('sublora', {})
                yaml_config['sublora']['allocation_config'] = alloc
                print(f"Using allocation_config from checkpoint: {alloc}")
        except Exception:
            pass

    method = SubLoRA(yaml_config)
    method.get_bounds()