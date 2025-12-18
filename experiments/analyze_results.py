#!/usr/bin/env python3
"""
Analysis script for Adaptive Subspace Allocation experiments.
Generates all plots and tables required for the ICML 2025 project proposal.

Usage:
    python experiments/analyze_results.py \
        --results_dir=out/adaptive_experiments \
        --output_dir=analysis_outputs
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import yaml
import glob
from pathlib import Path
from collections import defaultdict

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def load_checkpoint(ckpt_path):
    """Load model checkpoint."""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    return checkpoint


def find_bounds_yaml(exp_dir):
    """
    Find bounds YAML file in experiment directory.
    Handles the nested output structure from SubLoRA training.
    
    Output structure from get_bounds():
        bounds_levels{levels}_iters{max_quant_iters}.yml
        metrics_levels{levels}_iters{max_quant_iters}.yml
    
    HPC directory structure:
        sublora-d{dim}-{mode}-seed{seed}/
        └── out/SubLoRA_Pretrain/id{dim}_lr0.005_r4/{date}/{time}/
            └── bounds_levels*.yml
    """
    # Search patterns for bounds files (most specific to least specific)
    search_patterns = [
        # HPC nested structure: out/SubLoRA_Pretrain/id{dim}_lr0.005_r4/{date}/{time}/
        str(exp_dir / 'out' / 'SubLoRA_Pretrain' / 'id*' / '*' / '*' / 'bounds_levels*.yml'),
        # Alternative nested structure
        str(exp_dir / 'out' / 'SubLoRA_Pretrain' / '*' / '*' / '*' / 'bounds_levels*.yml'),
        str(exp_dir / 'out' / '*' / '*' / '*' / '*' / 'bounds_levels*.yml'),
        # Direct in out folder
        str(exp_dir / 'out' / 'bounds_levels*.yml'),
        # Direct in exp_dir
        str(exp_dir / 'bounds_levels*.yml'),
    ]
    
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            # Return the most recent file if multiple exist
            return sorted(matches)[-1]
    
    return None


def find_metrics_yaml(exp_dir):
    """
    Find metrics YAML file in experiment directory.
    """
    search_patterns = [
        # HPC nested structure
        str(exp_dir / 'out' / 'SubLoRA_Pretrain' / 'id*' / '*' / '*' / 'metrics_levels*.yml'),
        str(exp_dir / 'out' / 'SubLoRA_Pretrain' / '*' / '*' / '*' / 'metrics_levels*.yml'),
        str(exp_dir / 'out' / '*' / '*' / '*' / '*' / 'metrics_levels*.yml'),
        str(exp_dir / 'out' / 'metrics_levels*.yml'),
        str(exp_dir / 'metrics_levels*.yml'),
    ]
    
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]
    
    return None


def find_quant_checkpoint(exp_dir):
    """
    Find quant_ckpt_levels*.pt file in experiment directory.
    This contains prefix_message_len from quantization.
    """
    search_patterns = [
        # HPC nested structure
        str(exp_dir / 'out' / 'SubLoRA_Pretrain' / 'id*' / '*' / '*' / 'quant_ckpt_levels*.pt'),
        str(exp_dir / 'out' / 'SubLoRA_Pretrain' / '*' / '*' / '*' / 'quant_ckpt_levels*.pt'),
        str(exp_dir / 'out' / '*' / '*' / '*' / '*' / 'quant_ckpt_levels*.pt'),
        str(exp_dir / 'out' / 'quant_ckpt_levels*.pt'),
        str(exp_dir / 'quant_ckpt_levels*.pt'),
    ]
    
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]
    
    return None


def find_best_checkpoint(exp_dir):
    """
    Find best_ckpt.pt in experiment directory, handling nested structure.

    HPC directory structure:
        sublora-d{dim}-{mode}-seed{seed}/
        └── out/SubLoRA_Pretrain/id{dim}_lr0.005_r4/{date}/{time}/
            └── best_ckpt.pt
    """
    search_patterns = [
        # HPC nested structure
        str(exp_dir / 'out' / 'SubLoRA_Pretrain' / 'id*' / '*' / '*' / 'best_ckpt.pt'),
        str(exp_dir / 'out' / 'SubLoRA_Pretrain' / '*' / '*' / '*' / 'best_ckpt.pt'),
        str(exp_dir / 'out' / '*' / '*' / '*' / '*' / 'best_ckpt.pt'),
        str(exp_dir / 'out' / 'best_ckpt.pt'),
        str(exp_dir / 'best_ckpt.pt'),
    ]

    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]

    return None


def find_topk_indices_file(exp_dir):
    """
    Find top_k_indices_levels*.txt file in experiment directory.
    """
    search_patterns = [
        # HPC nested structure
        str(exp_dir / 'out' / 'SubLoRA_Pretrain' / 'id*' / '*' / '*' / 'top_k_indices_levels*.txt'),
        str(exp_dir / 'out' / 'SubLoRA_LearnedShared' / 'id*' / '*' / '*' / 'top_k_indices_levels*.txt'),
        str(exp_dir / 'out' / 'SubLoRA_Pretrain' / '*' / '*' / '*' / 'top_k_indices_levels*.txt'),
        str(exp_dir / 'out' / 'SubLoRA_LearnedShared' / '*' / '*' / '*' / 'top_k_indices_levels*.txt'),
        str(exp_dir / 'out' / '*' / '*' / '*' / '*' / 'top_k_indices_levels*.txt'),
        str(exp_dir / 'out' / 'top_k_indices_levels*.txt'),
        str(exp_dir / 'top_k_indices_levels*.txt'),
    ]

    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]

    return None


def extract_bounds_metrics(results_dir):
    """
    Extract bounds metrics from all experiment runs.
    
    Primary sources (PT files - more authoritative):
    - best_ckpt.pt: Contains best_val_loss, model config, training state
    - quant_ckpt_levels{X}_iters{Y}.pt: Contains prefix_message_len after quantization
    
    Secondary sources (YAML files - for computed bounds):
    - bounds_levels{X}_iters{Y}.yml: Contains computed PAC-Bayes bounds
    - metrics_levels{X}_iters{Y}.yml: Contains empirical metrics (bpd, accuracy)

    Returns:
        pd.DataFrame with columns: config, seed, budget, mode, ratio,
                                   kl_divergence, empirical_bpd, bound_value, val_loss
    """
    results = []

    # Expected directory structure: results_dir/sublora-d{budget}-{mode}-seed{seed}/
    for exp_dir in Path(results_dir).iterdir():
        if not exp_dir.is_dir():
            continue

        # Parse directory name
        name = exp_dir.name
        
        try:
            # Handle naming convention: sublora-d{dim}-{mode}-seed{seed}
            # Example: sublora-d1000-uniform-seed42, sublora-d1000-fixed-bheavy-seed42
            
            # Extract budget (intrinsic dimension)
            # Use regex to extract the full number after 'd' or 'dim'
            import re
            budget = None
            
            # Try to match '-d' followed by digits and then '-' (e.g., sublora-d10000-uniform)
            match = re.search(r'-d(\d+)-', name)
            if match:
                budget = int(match.group(1))
            else:
                # Try 'dim' followed by digits
                match = re.search(r'dim(\d+)', name)
                if match:
                    budget = int(match.group(1))
                else:
                    # Fallback: try extracting from parts
                    parts = name.replace('-', '_').split('_')
                    for p in parts:
                        if p.startswith('d') and p[1:].isdigit():
                            budget = int(p[1:])
                            break
            
            if budget is None:
                continue

            # Extract seed
            seed = None
            if 'seed' in name:
                seed_part = name.split('seed')[-1].split('-')[0].split('_')[0]
                seed = int(''.join(filter(str.isdigit, seed_part)))
            else:
                parts = name.replace('-', '_').split('_')
                for p in parts:
                    if p.isdigit():
                        seed = int(p)
                        break
            
            if seed is None:
                seed = 0  # Default seed

            # Extract mode and ratio
            name_lower = name.lower()
            if 'uniform' in name_lower:
                mode = 'uniform'
                ratio = 0.5
            elif 'learned_shared' in name_lower or 'learned-shared' in name_lower:
                mode = 'learned_shared'
                ratio = None  # adaptive with shared projectors
            elif 'learned' in name_lower:
                mode = 'learned'
                ratio = None  # adaptive
            elif 'bheavy' in name_lower:
                mode = 'fixed_bheavy'
                ratio = 0.8
            elif 'aheavy' in name_lower:
                mode = 'fixed_aheavy'
                ratio = 0.2
            elif 'equal' in name_lower:
                mode = 'fixed_equal'
                ratio = 0.5
            elif 'fixed' in name_lower:
                mode = 'fixed_equal'
                ratio = 0.5
            else:
                continue

            # Initialize metrics
            val_loss = np.nan
            train_loss = np.nan
            prefix_message_len = 0
            kl_divergence = np.nan
            bound_value = np.nan
            empirical_bpd = np.nan
            compressed_size_bits = 0
            intrinsic_dim_from_config = budget
            
            # === PRIMARY SOURCE 1: best_ckpt.pt ===
            best_ckpt_path = find_best_checkpoint(exp_dir)
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                try:
                    ckpt = torch.load(best_ckpt_path, map_location='cpu')
                    val_loss = ckpt.get('best_val_loss', np.nan)
                    if val_loss is not None:
                        val_loss = float(val_loss)
                    else:
                        val_loss = np.nan
                    
                    # Extract config info if available
                    config = ckpt.get('config', {})
                    if isinstance(config, dict):
                        intrinsic_dim_from_config = config.get('intrinsic_dim', budget)
                    
                    print(f"  Loaded best_ckpt.pt: val_loss={val_loss:.4f}" if not np.isnan(val_loss) else f"  Loaded best_ckpt.pt (no val_loss)")
                except Exception as e:
                    print(f"  Warning: Could not load best_ckpt.pt: {e}")
            
            # === PRIMARY SOURCE 2: quant_ckpt.pt (has prefix_message_len) ===
            quant_ckpt_path = find_quant_checkpoint(exp_dir)
            if quant_ckpt_path and os.path.exists(quant_ckpt_path):
                try:
                    quant_ckpt = torch.load(quant_ckpt_path, map_location='cpu')
                    prefix_message_len = quant_ckpt.get('prefix_message_len', 0)
                    if prefix_message_len:
                        prefix_message_len = float(prefix_message_len)
                        compressed_size_bits = prefix_message_len
                        # KL divergence = prefix_message_len * log(2) to convert bits to nats
                        kl_divergence = prefix_message_len * np.log(2)
                    print(f"  Loaded quant_ckpt.pt: prefix_message_len={prefix_message_len:.2f} bits")
                except Exception as e:
                    print(f"  Warning: Could not load quant_ckpt.pt: {e}")
            
            # === SECONDARY SOURCE: bounds YAML (computed PAC-Bayes bounds) ===
            bounds_file = find_bounds_yaml(exp_dir)
            if bounds_file and os.path.exists(bounds_file):
                try:
                    with open(bounds_file, 'r') as f:
                        bounds_data = yaml.safe_load(f)
                    
                    # Get prefix_message_len if not already loaded from PT
                    if prefix_message_len == 0:
                        prefix_message_len = bounds_data.get('prefix_message_len', 0)
                        compressed_size_bits = prefix_message_len
                    
                    # Get divergence (may include misc_extra_bits adjustment)
                    if np.isnan(kl_divergence):
                        kl_divergence = bounds_data.get('bpd_divergence', 
                                                        bounds_data.get('acc_divergence', 
                                                        prefix_message_len * np.log(2)))
                    
                    # Best PAC-Bayes bound value
                    bound_value = bounds_data.get('best_bpd_bound', np.nan)
                    
                    print(f"  Loaded bounds YAML: bound={bound_value:.4f}" if not np.isnan(bound_value) else "  Loaded bounds YAML")
                except Exception as e:
                    print(f"  Warning: Could not load bounds YAML: {e}")
            
            # === SECONDARY SOURCE: metrics YAML (empirical BPD) ===
            metrics_file = find_metrics_yaml(exp_dir)
            if metrics_file and os.path.exists(metrics_file):
                try:
                    with open(metrics_file, 'r') as f:
                        metrics_data = yaml.safe_load(f)
                    
                    # Get empirical BPD (use best alpha)
                    for key in ['bpd_alpha_0.01', 'bpd_alpha_0.005', 'bpd_alpha_0.05', 'bpd_alpha_0.1']:
                        if key in metrics_data and metrics_data[key]:
                            empirical_bpd = float(metrics_data[key])
                            break
                    
                    print(f"  Loaded metrics YAML: empirical_bpd={empirical_bpd:.4f}" if not np.isnan(empirical_bpd) else "  Loaded metrics YAML")
                except Exception as e:
                    print(f"  Warning: Could not load metrics YAML: {e}")
            
            # Use val_loss as empirical_bpd proxy if not available from bounds eval
            # (val_loss is cross-entropy in nats, BPD = loss / log(2))
            if np.isnan(empirical_bpd) and not np.isnan(val_loss):
                empirical_bpd = val_loss / np.log(2)
                print(f"  Using val_loss as BPD proxy: {empirical_bpd:.4f}")

            # Only add if we have at least checkpoint data
            if best_ckpt_path or bounds_file:
                results.append({
                    'config': name,
                    'seed': seed,
                    'budget': budget,
                    'mode': mode,
                    'ratio': ratio,
                    'val_loss': float(val_loss) if not np.isnan(val_loss) else np.nan,
                    'kl_divergence': float(kl_divergence) if not np.isnan(kl_divergence) else np.nan,
                    'empirical_bpd': float(empirical_bpd) if not np.isnan(empirical_bpd) else np.nan,
                    'bound_value': float(bound_value) if not np.isnan(bound_value) else np.nan,
                    'compressed_size_bits': float(compressed_size_bits),
                    'prefix_message_len': float(prefix_message_len),
                })
                print(f"  ✓ Added {name} to results")
            else:
                print(f"  ✗ Skipping {name}: no checkpoint or bounds data found")
                
        except Exception as e:
            print(f"Warning: Could not parse {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return pd.DataFrame(results)


def extract_learned_gating(results_dir, budget, seed):
    """
    Extract learned gating parameters (gamma values) for a specific run.

    Returns:
        np.array of shape (num_layers,) with gamma values per layer
    """
    # Find learned gating checkpoint - try multiple naming conventions
    possible_names = [
        f'sublora-d{budget}-learned-seed{seed}',
        f'd{budget}_learned_seed{seed}',
        f'sublora-dim{budget}-learned-seed{seed}',
    ]
    
    checkpoint_path = None
    for name in possible_names:
        exp_dir = Path(results_dir) / name
        if exp_dir.exists():
            checkpoint_path = find_best_checkpoint(exp_dir)
            if checkpoint_path:
                break
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found for learned gating d={budget}, seed={seed}")
        return None

    try:
        checkpoint = load_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return None

    # Extract gating parameters from state dict
    # Look for parameters like 'gating_params', 'allocation_logits', 'theta_split', etc.
    gating_params = {}
    state_dict = checkpoint.get('raw_model', checkpoint.get('model', {}))
    
    for key, value in state_dict.items():
        if any(gating_key in key.lower() for gating_key in ['gating_param', 'allocation_logit', 'theta_split']):
            try:
                # Parse layer index from key
                parts = key.split('.')
                for p in parts:
                    if p.isdigit():
                        layer_idx = int(p)
                        # Compute gamma = sigmoid(theta_split)
                        if isinstance(value, torch.Tensor):
                            gamma = torch.sigmoid(value).item() if value.numel() == 1 else torch.sigmoid(value).mean().item()
                        else:
                            gamma = 1 / (1 + np.exp(-value))  # sigmoid
                        gating_params[layer_idx] = gamma
                        break
            except (ValueError, IndexError):
                continue

    if not gating_params:
        print(f"Warning: No gating parameters found in checkpoint for d={budget}, seed={seed}")
        return None

    # Convert to array sorted by layer index
    num_layers = max(gating_params.keys()) + 1
    gammas = np.array([gating_params.get(i, 0.5) for i in range(num_layers)])

    return gammas


def plot_pareto_frontier(df, budget, output_dir):
    """
    Plot Pareto frontier: model complexity (KL) vs empirical risk (BPD).

    Args:
        df: DataFrame with bounds metrics
        budget: Subspace budget (1000 or 2000)
        output_dir: Directory to save plots
    """
    # Filter by budget
    df_budget = df[df['budget'] == budget].copy()

    # Average over seeds (dropna=False to include learned mode with ratio=None)
    df_avg = df_budget.groupby(['mode', 'ratio'], dropna=False).agg({
        'kl_divergence': ['mean', 'std'],
        'empirical_bpd': ['mean', 'std']
    }).reset_index()

    df_avg.columns = ['mode', 'ratio', 'kl_mean', 'kl_std', 'bpd_mean', 'bpd_std']

    # Define colors and markers
    mode_styles = {
        'uniform': {'color': 'black', 'marker': 's', 'label': 'Baseline (Uniform)'},
        'fixed_bheavy': {'color': 'blue', 'marker': '^', 'label': 'Fixed B-heavy (0.8)'},
        'fixed_equal': {'color': 'green', 'marker': 'o', 'label': 'Fixed Equal (0.5)'},
        'fixed_aheavy': {'color': 'red', 'marker': 'v', 'label': 'Fixed A-heavy (0.2)'},
        'learned': {'color': 'purple', 'marker': '*', 'label': 'Learned Gating', 'markersize': 15},
        'learned_shared': {'color': 'orange', 'marker': 'D', 'label': 'Learned Shared', 'markersize': 12},
    }

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    for mode, style in mode_styles.items():
        df_mode = df_avg[df_avg['mode'] == mode]

        if len(df_mode) == 0:
            continue

        ax.errorbar(
            df_mode['kl_mean'],
            df_mode['bpd_mean'],
            xerr=df_mode['kl_std'],
            yerr=df_mode['bpd_std'],
            color=style['color'],
            marker=style['marker'],
            markersize=style.get('markersize', 10),
            label=style['label'],
            capsize=5,
            capthick=2,
            linewidth=0,
            elinewidth=2
        )

    ax.set_xlabel('Model Complexity (KL Divergence)', fontsize=14)
    ax.set_ylabel('Empirical Risk (BPD)', fontsize=14)
    ax.set_title(f'Pareto Frontier: Complexity vs. Risk (d={budget})', fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Save
    output_path = Path(output_dir) / f'pareto_frontier_d{budget}.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved Pareto frontier plot: {output_path}")


def plot_learned_gating_heatmap(results_dir, budgets, seeds, output_dir):
    """
    Plot heatmap of learned gating parameters (gamma) across layers.

    Args:
        results_dir: Directory with experimental results
        budgets: List of budgets (e.g., [1000, 2000])
        seeds: List of seeds (e.g., [42, 123, 999])
        output_dir: Directory to save plots
    """
    # Collect gamma values
    gamma_data = defaultdict(list)

    for budget in budgets:
        for seed in seeds:
            gammas = extract_learned_gating(results_dir, budget, seed)
            if gammas is not None:
                gamma_data[budget].append(gammas)

    if not gamma_data:
        print("Warning: No learned gating data found")
        return

    # Create heatmap for each budget
    for budget, gamma_list in gamma_data.items():
        if not gamma_list:
            continue

        # Stack into matrix: rows=seeds, cols=layers
        gamma_matrix = np.vstack(gamma_list)
        num_layers = gamma_matrix.shape[1]

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 4))

        im = ax.imshow(gamma_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)

        # Set ticks
        ax.set_xticks(np.arange(num_layers))
        ax.set_xticklabels([f'Layer {i}' for i in range(num_layers)], rotation=45, ha='right')
        ax.set_yticks(np.arange(len(gamma_list)))
        ax.set_yticklabels([f'Seed {seeds[i]}' for i in range(len(gamma_list))])

        # Labels
        ax.set_xlabel('Transformer Layer', fontsize=14)
        ax.set_ylabel('Random Seed', fontsize=14)
        ax.set_title(f'Learned Gating Parameters γ_ℓ (d={budget})\nHigher γ → More allocation to B',
                     fontsize=16, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('γ_ℓ (Fraction to B)', rotation=270, labelpad=20, fontsize=12)

        # Annotate cells with values
        for i in range(len(gamma_list)):
            for j in range(num_layers):
                text = ax.text(j, i, f'{gamma_matrix[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)

        # Save
        output_path = Path(output_dir) / f'learned_gating_heatmap_d{budget}.png'
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved learned gating heatmap: {output_path}")


def plot_learned_gating_trend(results_dir, budgets, seeds, output_dir):
    """
    Plot trend of learned gamma values across layers (mean ± std over seeds).
    """
    # Collect gamma values
    gamma_data = defaultdict(list)

    for budget in budgets:
        for seed in seeds:
            gammas = extract_learned_gating(results_dir, budget, seed)
            if gammas is not None:
                gamma_data[budget].append(gammas)

    if not gamma_data:
        print("Warning: No learned gating data found")
        return

    # Create line plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Dynamic color mapping based on available budgets
    color_cycle = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    budget_list = sorted(gamma_data.keys())
    colors = {b: color_cycle[i % len(color_cycle)] for i, b in enumerate(budget_list)}

    for budget, gamma_list in gamma_data.items():
        if not gamma_list:
            continue

        gamma_matrix = np.vstack(gamma_list)
        num_layers = gamma_matrix.shape[1]

        # Compute mean and std
        gamma_mean = gamma_matrix.mean(axis=0)
        gamma_std = gamma_matrix.std(axis=0)

        # Plot
        layers = np.arange(num_layers)
        ax.plot(layers, gamma_mean,
                color=colors.get(budget, 'black'),
                marker='o',
                linewidth=2,
                markersize=8,
                label=f'd={budget}')
        ax.fill_between(layers,
                        gamma_mean - gamma_std,
                        gamma_mean + gamma_std,
                        color=colors.get(budget, 'black'),
                        alpha=0.2)

    # Reference line at 0.5
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, label='Equal split (γ=0.5)')

    ax.set_xlabel('Transformer Layer', fontsize=14)
    ax.set_ylabel('γ_ℓ (Fraction allocated to B)', fontsize=14)
    ax.set_title('Learned Allocation Patterns Across Layers', fontsize=16, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xticks(np.arange(num_layers))
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)

    # Save
    output_path = Path(output_dir) / 'learned_gating_trend.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved learned gating trend: {output_path}")


def plot_gating_time_series(results_dir, output_dir):
    """
    Search for `gating_trace.csv` files under `results_dir` and plot per-layer
    gamma time series (one subplot per transformer layer).
    """
    results_dir = Path(results_dir)
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        # Find any gating_trace.csv under this experiment directory
        gating_files = list(exp_dir.rglob('gating_trace.csv'))
        if not gating_files:
            continue
        for gf in gating_files:
            try:
                df = pd.read_csv(gf)
            except Exception as e:
                print(f"  Warning: could not read {gf}: {e}")
                continue

            # Identify gamma columns (exclude iter and gating_scale)
            gamma_cols = [c for c in df.columns if c.startswith('gamma_layer_') or c == 'gamma_misc']
            if not gamma_cols:
                continue

            num_layers = len(gamma_cols)
            # Create subplots vertically stacked
            fig, axes = plt.subplots(num_layers, 1, figsize=(10, 2.2 * num_layers), sharex=True)
            if num_layers == 1:
                axes = [axes]

            for i, col in enumerate(gamma_cols):
                axes[i].plot(df['iter'], df[col], linewidth=1.5)
                axes[i].set_ylabel(col)
                axes[i].set_ylim(0, 1)
                axes[i].grid(True, alpha=0.3)

            axes[-1].set_xlabel('Iteration')
            fig.suptitle(f'Gating γ over time: {exp_dir.name} ({gf.parent.name})')
            out_path = Path(output_dir) / f'gating_trace_{exp_dir.name}_{gf.parent.name}.png'
            plt.tight_layout()
            plt.savefig(out_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved gating trace plot: {out_path}")


def plot_compression_comparison(df, budget, output_dir):
    """
    Create multi-panel plot comparing compression methods (similar to SubLoRA paper Figure).

    Left panel: Complexity vs BPD scatter
    Middle panel: Complexity vs Top-1 Error
    Right panel: Compression extent vs Bound/Risk
    """
    df_budget = df[df['budget'] == budget].copy()
    df_avg = df_budget.groupby(['mode', 'ratio'], dropna=False).agg({
        'kl_divergence': ['mean', 'std'],
        'empirical_bpd': ['mean', 'std'],
        'compressed_size_bits': ['mean', 'std'],
        'bound_value': ['mean', 'std']
    }).reset_index()

    df_avg.columns = ['mode', 'ratio', 'kl_mean', 'kl_std',
                      'bpd_mean', 'bpd_std', 'size_mean', 'size_std',
                      'bound_mean', 'bound_std']

    # Mode styles
    mode_styles = {
        'uniform': {'color': '#888888', 'marker': 's', 'label': 'Uniform (Baseline)', 'size': 100},
        'fixed_bheavy': {'color': '#4A90E2', 'marker': '^', 'label': 'Fixed B-heavy', 'size': 100},
        'fixed_equal': {'color': '#50C878', 'marker': 'o', 'label': 'Fixed Equal', 'size': 100},
        'fixed_aheavy': {'color': '#E74C3C', 'marker': 'v', 'label': 'Fixed A-heavy', 'size': 100},
        'learned': {'color': '#9B59B6', 'marker': '*', 'label': 'Learned Gating', 'size': 200},
        'learned_shared': {'color': '#F39C12', 'marker': 'D', 'label': 'Learned Shared', 'size': 150},
    }

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: Complexity vs BPD
    ax1 = axes[0]
    for mode, style in mode_styles.items():
        df_mode = df_avg[df_avg['mode'] == mode]
        if len(df_mode) == 0:
            continue
        ax1.scatter(df_mode['kl_mean'], df_mode['bpd_mean'],
                   c=style['color'], marker=style['marker'], s=style['size'],
                   label=style['label'], alpha=0.8, edgecolors='black', linewidth=1.5)

    ax1.set_xlabel('Complexity', fontsize=13, fontweight='bold')
    ax1.set_ylabel('BPD (Train)', fontsize=13, fontweight='bold')
    ax1.set_title('Complexity vs Empirical Risk', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=10, framealpha=0.9)

    # Panel 2: Complexity vs Bound/Error (using bound as proxy for error)
    ax2 = axes[1]
    for mode, style in mode_styles.items():
        df_mode = df_avg[df_avg['mode'] == mode]
        if len(df_mode) == 0:
            continue
        # Use bound value as generalization metric
        ax2.scatter(df_mode['kl_mean'], df_mode['bound_mean'],
                   c=style['color'], marker=style['marker'], s=style['size'],
                   alpha=0.8, edgecolors='black', linewidth=1.5)

    ax2.set_xlabel('Complexity', fontsize=13, fontweight='bold')
    ax2.set_ylabel('PAC-Bayes Bound', fontsize=13, fontweight='bold')
    ax2.set_title('Complexity vs Generalization Bound', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Compression extent vs Bound and Risk
    ax3 = axes[2]

    # Convert size to KB
    df_avg['size_kb'] = df_avg['size_mean'] / 8000

    # Plot bound
    for mode, style in mode_styles.items():
        df_mode = df_avg[df_avg['mode'] == mode]
        if len(df_mode) == 0:
            continue
        ax3.plot(df_mode['size_kb'], df_mode['bound_mean'],
                color=style['color'], marker=style['marker'], markersize=10,
                linewidth=2, alpha=0.7, label=f"{style['label']} (Bound)")

    # Plot empirical risk as dashed lines
    for mode, style in mode_styles.items():
        df_mode = df_avg[df_avg['mode'] == mode]
        if len(df_mode) == 0:
            continue
        ax3.plot(df_mode['size_kb'], df_mode['bpd_mean'],
                color=style['color'], marker=style['marker'], markersize=8,
                linewidth=2, linestyle='--', alpha=0.5, label=f"{style['label']} (Risk)")

    ax3.set_xlabel('Extent of Compression\n(model size KB)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Bits Per Dimension', fontsize=13, fontweight='bold')
    ax3.set_title('Compression vs Performance', fontsize=14, fontweight='bold')
    ax3.set_xscale('log')
    ax3.grid(True, alpha=0.3, which='both')
    ax3.legend(loc='best', fontsize=8, framealpha=0.9, ncol=2)

    plt.tight_layout()
    output_path = Path(output_dir) / f'compression_comparison_d{budget}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved compression comparison plot: {output_path}")


def plot_allocation_comparison_grid(df, output_dir):
    """
    Create 2x2 grid comparing different allocation strategies across metrics.
    """
    # Average over seeds (dropna=False to include learned mode with ratio=None)
    df_avg = df.groupby(['budget', 'mode', 'ratio'], dropna=False).agg({
        'kl_divergence': ['mean', 'std'],
        'empirical_bpd': ['mean', 'std'],
        'compressed_size_bits': ['mean', 'std'],
        'bound_value': ['mean', 'std']
    }).reset_index()

    df_avg.columns = ['budget', 'mode', 'ratio', 'kl_mean', 'kl_std',
                      'bpd_mean', 'bpd_std', 'size_mean', 'size_std',
                      'bound_mean', 'bound_std']

    # Get unique budgets from data (dynamically)
    budgets = sorted(df_avg['budget'].unique())
    
    # Mode styles
    mode_colors = {
        'uniform': '#888888',
        'fixed_bheavy': '#4A90E2',
        'fixed_equal': '#50C878',
        'fixed_aheavy': '#E74C3C',
        'learned': '#9B59B6',
        'learned_shared': '#F39C12',
    }

    mode_labels = {
        'uniform': 'Uniform',
        'fixed_bheavy': 'B-heavy (0.8)',
        'fixed_equal': 'Equal (0.5)',
        'fixed_aheavy': 'A-heavy (0.2)',
        'learned': 'Learned',
        'learned_shared': 'Learned Shared',
    }

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: BPD comparison (bar chart)
    ax1 = axes[0, 0]
    for i, budget in enumerate(budgets):
        df_budget = df_avg[df_avg['budget'] == budget]
        x_pos = np.arange(len(df_budget))
        offset = 0.2 if i == 0 else -0.2

        colors = [mode_colors.get(mode, '#888888') for mode in df_budget['mode']]
        bars = ax1.bar(x_pos + offset, df_budget['bpd_mean'], 0.35,
                      yerr=df_budget['bpd_std'], capsize=5,
                      label=f'd={budget}', alpha=0.8, color=colors,
                      edgecolor='black', linewidth=1.2)

    ax1.set_ylabel('Bits Per Dimension (BPD)', fontsize=12, fontweight='bold')
    ax1.set_title('Empirical Risk Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(np.arange(len(df_budget)))
    ax1.set_xticklabels([mode_labels.get(m, m) for m in df_budget['mode']], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Bound comparison (bar chart)
    ax2 = axes[0, 1]
    for i, budget in enumerate(budgets):
        df_budget = df_avg[df_avg['budget'] == budget]
        x_pos = np.arange(len(df_budget))
        offset = 0.2 if i == 0 else -0.2

        colors = [mode_colors.get(mode, '#888888') for mode in df_budget['mode']]
        bars = ax2.bar(x_pos + offset, df_budget['bound_mean'], 0.35,
                      yerr=df_budget['bound_std'], capsize=5,
                      label=f'd={budget}', alpha=0.8, color=colors,
                      edgecolor='black', linewidth=1.2)

    ax2.set_ylabel('PAC-Bayes Bound', fontsize=12, fontweight='bold')
    ax2.set_title('Generalization Bound Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(np.arange(len(df_budget)))
    ax2.set_xticklabels([mode_labels.get(m, m) for m in df_budget['mode']], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Compression size (bar chart)
    ax3 = axes[1, 0]
    for i, budget in enumerate(budgets):
        df_budget = df_avg[df_avg['budget'] == budget]
        x_pos = np.arange(len(df_budget))
        offset = 0.2 if i == 0 else -0.2

        colors = [mode_colors.get(mode, '#888888') for mode in df_budget['mode']]
        bars = ax3.bar(x_pos + offset, df_budget['size_mean']/8000, 0.35,
                      yerr=df_budget['size_std']/8000, capsize=5,
                      label=f'd={budget}', alpha=0.8, color=colors,
                      edgecolor='black', linewidth=1.2)

    ax3.set_ylabel('Compressed Size (KB)', fontsize=12, fontweight='bold')
    ax3.set_title('Model Compression Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(np.arange(len(df_budget)))
    ax3.set_xticklabels([mode_labels.get(m, m) for m in df_budget['mode']], rotation=45, ha='right')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Relative improvement over baseline
    ax4 = axes[1, 1]
    for i, budget in enumerate(budgets):
        df_budget = df_avg[df_avg['budget'] == budget].copy()
        baseline_rows = df_budget[df_budget['mode'] == 'uniform']
        if len(baseline_rows) == 0:
            continue
        baseline_bpd = baseline_rows['bpd_mean'].values[0]

        df_budget['improvement'] = (baseline_bpd - df_budget['bpd_mean']) / baseline_bpd * 100

        colors = [mode_colors.get(mode, '#888888') for mode in df_budget['mode']]
        x_pos = np.arange(len(df_budget))
        offset = 0.2 if i == 0 else -0.2

        bars = ax4.bar(x_pos + offset, df_budget['improvement'], 0.35,
                      label=f'd={budget}', alpha=0.8, color=colors,
                      edgecolor='black', linewidth=1.2)

    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_ylabel('Relative BPD Improvement (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Performance vs Baseline (Uniform)', fontsize=14, fontweight='bold')
    ax4.set_xticks(np.arange(len(df_budget)))
    ax4.set_xticklabels([mode_labels.get(m, m) for m in df_budget['mode']], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_dir) / 'allocation_comparison_grid.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved allocation comparison grid: {output_path}")


def generate_summary_table(df, output_dir):
    """
    Generate summary table with all experimental results.
    """
    # Average over seeds (dropna=False to include learned mode with ratio=None)
    summary = df.groupby(['budget', 'mode', 'ratio'], dropna=False).agg({
        'empirical_bpd': ['mean', 'std'],
        'kl_divergence': ['mean', 'std'],
        'bound_value': ['mean', 'std'],
        'compressed_size_bits': ['mean', 'std']
    }).reset_index()

    # Flatten column names
    summary.columns = [
        'budget', 'mode', 'ratio',
        'bpd_mean', 'bpd_std',
        'kl_mean', 'kl_std',
        'bound_mean', 'bound_std',
        'size_mean', 'size_std'
    ]

    # Format for display
    summary['BPD'] = summary.apply(lambda x: f"{x['bpd_mean']:.4f} ± {x['bpd_std']:.4f}", axis=1)
    summary['KL'] = summary.apply(lambda x: f"{x['kl_mean']:.2f} ± {x['kl_std']:.2f}", axis=1)
    summary['Bound'] = summary.apply(lambda x: f"{x['bound_mean']:.4f} ± {x['bound_std']:.4f}", axis=1)
    summary['Size (KB)'] = summary.apply(lambda x: f"{x['size_mean']/8000:.1f} ± {x['size_std']/8000:.1f}", axis=1)

    # Select columns for final table
    final_table = summary[['budget', 'mode', 'ratio', 'BPD', 'KL', 'Bound', 'Size (KB)']]

    # Save as CSV
    output_path = Path(output_dir) / 'summary_table.csv'
    final_table.to_csv(output_path, index=False)
    print(f"Saved summary table: {output_path}")

    # Save as LaTeX
    latex_path = Path(output_dir) / 'summary_table.tex'
    with open(latex_path, 'w') as f:
        f.write(final_table.to_latex(index=False))
    print(f"Saved LaTeX table: {latex_path}")

    return final_table


def plot_topk_histogram(results_dir, budget, seed, output_dir, k_values=[100, 500, 1000]):
    """
    Histogram showing distribution of prediction ranks (top-k indices).
    Shows where the correct token ranked in the model's predictions (0=top prediction).

    NOTE: This measures prediction quality, NOT subspace dimension utilization.
    Lower ranks = better predictions (rank 0 = perfect top-1 prediction).

    Args:
        results_dir: Directory with experimental results
        budget: Intrinsic dimension (e.g., 10000, 20000)
        seed: Random seed
        output_dir: Directory to save plots
        k_values: List of k values to plot (e.g., [100, 500, 1000])
    """
    # Find experiment directory
    possible_names = [
        f'sublora-d{budget}-learned-seed{seed}',
        f'd{budget}_learned_seed{seed}',
        f'sublora-dim{budget}-learned-seed{seed}',
    ]

    topk_file = None
    for name in possible_names:
        exp_dir = Path(results_dir) / name
        if exp_dir.exists():
            topk_file = find_topk_indices_file(exp_dir)
            if topk_file:
                break

    if not topk_file or not os.path.exists(topk_file):
        print(f"Warning: top_k_indices file not found for d={budget}, seed={seed}")
        return

    # Load top-k indices
    try:
        with open(topk_file, 'r') as f:
            content = f.read().strip()
            # Parse Python list format
            top_k_indices = eval(content)
            top_k_indices = np.array(top_k_indices)
    except Exception as e:
        print(f"Warning: Could not load top_k_indices file: {e}")
        return

    # Create figure with subplots for different k values
    fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 5))
    if len(k_values) == 1:
        axes = [axes]

    for ax, k in zip(axes, k_values):
        if k > len(top_k_indices):
            k = len(top_k_indices)

        indices_subset = top_k_indices[:k]

        # Create histogram
        num_bins = min(50, budget // 200)  # Adaptive number of bins
        ax.hist(indices_subset, bins=num_bins, edgecolor='black', alpha=0.7, color='steelblue')

        # Add vertical line at mean
        mean_idx = np.mean(indices_subset)
        ax.axvline(mean_idx, color='red', linestyle='--', linewidth=2, label=f'Mean Rank: {mean_idx:.0f}')

        ax.set_xlabel('Prediction Rank (0=top prediction)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Top-{k} Rank Distribution', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Prediction Rank Distribution (d={budget}, seed={seed})', fontsize=15, fontweight='bold')
    plt.tight_layout()

    output_path = Path(output_dir) / f'prediction_rank_histogram_d{budget}_seed{seed}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved prediction rank histogram: {output_path}")


def plot_magnitude_distribution(results_dir, budget, seed, output_dir):
    """
    Plot sorted magnitudes of subspace_params to show effective dimensionality.

    Args:
        results_dir: Directory with experimental results
        budget: Intrinsic dimension
        seed: Random seed
        output_dir: Directory to save plots
    """
    # Find experiment directory and checkpoint
    possible_names = [
        f'sublora-d{budget}-learned-seed{seed}',
        f'd{budget}_learned_seed{seed}',
        f'sublora-dim{budget}-learned-seed{seed}',
    ]

    checkpoint_path = None
    topk_file = None
    for name in possible_names:
        exp_dir = Path(results_dir) / name
        if exp_dir.exists():
            checkpoint_path = find_best_checkpoint(exp_dir)
            topk_file = find_topk_indices_file(exp_dir)
            if checkpoint_path and topk_file:
                break

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found for d={budget}, seed={seed}")
        return

    # Load checkpoint
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        # Extract subspace_params
        if 'subspace_params' in checkpoint:
            subspace_params = checkpoint['subspace_params']
        elif 'raw_model' in checkpoint and 'subspace_params' in checkpoint['raw_model']:
            subspace_params = checkpoint['raw_model']['subspace_params']
        elif 'model' in checkpoint:
            # Try to find subspace_params in model state dict
            state_dict = checkpoint['model']
            subspace_params = state_dict.get('subspace_params', None)
        else:
            print(f"Warning: Could not find subspace_params in checkpoint")
            return

        if subspace_params is None:
            print(f"Warning: subspace_params is None")
            return

        # Convert to numpy
        if isinstance(subspace_params, torch.Tensor):
            subspace_params = subspace_params.cpu().numpy()

    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return

    # Compute magnitudes and sort
    magnitudes = np.abs(subspace_params)
    sorted_magnitudes = np.sort(magnitudes)[::-1]  # Descending order

    # Load top-k indices for reference
    top_k_threshold = None
    if topk_file and os.path.exists(topk_file):
        try:
            with open(topk_file, 'r') as f:
                content = f.read().strip()
                top_k_indices = eval(content)
                if len(top_k_indices) > 0:
                    # Get magnitude of the k-th largest element
                    k = len(top_k_indices)
                    top_k_threshold = sorted_magnitudes[min(k, len(sorted_magnitudes)-1)]
        except:
            pass

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Full sorted magnitudes (log scale)
    indices = np.arange(1, len(sorted_magnitudes) + 1)
    ax1.plot(indices, sorted_magnitudes, linewidth=1.5, color='steelblue')
    ax1.set_xlabel('Rank (sorted by magnitude)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Magnitude |θ_i|', fontsize=12, fontweight='bold')
    ax1.set_title('Sorted Subspace Parameter Magnitudes', fontsize=13, fontweight='bold')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')

    if top_k_threshold is not None:
        ax1.axhline(top_k_threshold, color='red', linestyle='--', linewidth=2,
                    label=f'Top-{k} threshold')
        ax1.legend(fontsize=10)

    # Plot 2: Cumulative percentage of total magnitude
    cumsum_magnitudes = np.cumsum(sorted_magnitudes)
    total_magnitude = cumsum_magnitudes[-1]
    cumsum_percent = 100 * cumsum_magnitudes / total_magnitude

    ax2.plot(indices, cumsum_percent, linewidth=2, color='darkgreen')
    ax2.set_xlabel('Rank (sorted by magnitude)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Cumulative % of Total Magnitude', fontsize=12, fontweight='bold')
    ax2.set_title('Effective Dimensionality', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Add reference lines
    for pct in [50, 90, 95, 99]:
        idx_pct = np.searchsorted(cumsum_percent, pct)
        ax2.axhline(pct, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(idx_pct, color='gray', linestyle=':', alpha=0.5)
        ax2.text(idx_pct, pct + 2, f'{idx_pct} dims', fontsize=9, ha='center')

    fig.suptitle(f'Subspace Parameter Magnitude Analysis (d={budget}, seed={seed})',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    output_path = Path(output_dir) / f'magnitude_distribution_d{budget}_seed{seed}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved magnitude distribution: {output_path}")


def plot_sparsity_patterns(results_dir, budget, seed, output_dir, percentile_threshold=95):
    """
    Binary heatmap showing which indices are 'active' (above threshold).
    Reveals structured sparsity patterns.

    Args:
        results_dir: Directory with experimental results
        budget: Intrinsic dimension
        seed: Random seed
        output_dir: Directory to save plots
        percentile_threshold: Percentile threshold for active indices (default: 95)
    """
    # Find checkpoint
    possible_names = [
        f'sublora-d{budget}-learned-seed{seed}',
        f'd{budget}_learned_seed{seed}',
        f'sublora-dim{budget}-learned-seed{seed}',
    ]

    checkpoint_path = None
    for name in possible_names:
        exp_dir = Path(results_dir) / name
        if exp_dir.exists():
            checkpoint_path = find_best_checkpoint(exp_dir)
            if checkpoint_path:
                break

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint not found for d={budget}, seed={seed}")
        return

    # Load subspace_params
    try:
        checkpoint = load_checkpoint(checkpoint_path)
        if 'subspace_params' in checkpoint:
            subspace_params = checkpoint['subspace_params']
        elif 'raw_model' in checkpoint and 'subspace_params' in checkpoint['raw_model']:
            subspace_params = checkpoint['raw_model']['subspace_params']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
            subspace_params = state_dict.get('subspace_params', None)
        else:
            print(f"Warning: Could not find subspace_params")
            return

        if isinstance(subspace_params, torch.Tensor):
            subspace_params = subspace_params.cpu().numpy()
    except Exception as e:
        print(f"Warning: Could not load checkpoint: {e}")
        return

    # Compute threshold
    magnitudes = np.abs(subspace_params)
    threshold = np.percentile(magnitudes, percentile_threshold)

    # Create binary pattern
    active_indices = magnitudes > threshold

    # Reshape into 2D for visualization (sqrt grid)
    grid_size = int(np.ceil(np.sqrt(len(subspace_params))))
    padded_length = grid_size ** 2

    # Pad with zeros if needed
    if len(active_indices) < padded_length:
        active_indices_padded = np.zeros(padded_length, dtype=bool)
        active_indices_padded[:len(active_indices)] = active_indices
    else:
        active_indices_padded = active_indices[:padded_length]

    pattern_2d = active_indices_padded.reshape(grid_size, grid_size)

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Binary sparsity pattern
    im1 = ax1.imshow(pattern_2d, cmap='RdYlGn', aspect='auto', interpolation='nearest')
    ax1.set_title(f'Sparsity Pattern (>{percentile_threshold}th percentile)',
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Index (reshaped)', fontsize=11)
    ax1.set_ylabel('Index (reshaped)', fontsize=11)

    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Active (1) / Inactive (0)', rotation=270, labelpad=20)

    # Plot 2: Magnitude heatmap (log scale)
    magnitudes_2d = np.zeros(padded_length)
    magnitudes_2d[:len(magnitudes)] = magnitudes
    if len(magnitudes) < padded_length:
        magnitudes_2d[len(magnitudes):] = np.nan
    magnitudes_2d = magnitudes_2d.reshape(grid_size, grid_size)

    im2 = ax2.imshow(np.log10(magnitudes_2d + 1e-10), cmap='viridis', aspect='auto', interpolation='nearest')
    ax2.set_title('Log Magnitude Heatmap', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Index (reshaped)', fontsize=11)
    ax2.set_ylabel('Index (reshaped)', fontsize=11)

    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('log₁₀(|θ_i|)', rotation=270, labelpad=20)

    # Add statistics
    sparsity = 100 * np.sum(active_indices) / len(active_indices)
    fig.suptitle(f'Sparsity Analysis (d={budget}, seed={seed})\n'
                 f'Active indices: {np.sum(active_indices)} / {len(active_indices)} ({sparsity:.1f}%)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = Path(output_dir) / f'sparsity_pattern_d{budget}_seed{seed}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved sparsity pattern: {output_path}")


def plot_index_stability_across_seeds(results_dir, budget, seeds, output_dir, k=1000):
    """
    Analyze which indices are consistently in top-k across different seeds.
    Shows stable vs unstable dimensions.

    Args:
        results_dir: Directory with experimental results
        budget: Intrinsic dimension
        seeds: List of random seeds
        output_dir: Directory to save plots
        k: Number of top indices to consider
    """
    # Collect top-k indices for each seed
    all_topk_indices = []
    valid_seeds = []

    for seed in seeds:
        possible_names = [
            f'sublora-d{budget}-learned-seed{seed}',
            f'd{budget}_learned_seed{seed}',
            f'sublora-dim{budget}-learned-seed{seed}',
        ]

        topk_file = None
        for name in possible_names:
            exp_dir = Path(results_dir) / name
            if exp_dir.exists():
                topk_file = find_topk_indices_file(exp_dir)
                if topk_file:
                    break

        if not topk_file or not os.path.exists(topk_file):
            print(f"Warning: top_k_indices file not found for d={budget}, seed={seed}")
            continue

        try:
            with open(topk_file, 'r') as f:
                content = f.read().strip()
                top_k_indices = eval(content)
                top_k_indices = np.array(top_k_indices[:k])
                all_topk_indices.append(set(top_k_indices))
                valid_seeds.append(seed)
        except Exception as e:
            print(f"Warning: Could not load top_k_indices for seed {seed}: {e}")
            continue

    if len(all_topk_indices) < 2:
        print(f"Warning: Need at least 2 seeds for stability analysis, found {len(all_topk_indices)}")
        return

    # Compute overlap statistics
    # Count how many seeds each index appears in
    all_indices = set()
    for indices_set in all_topk_indices:
        all_indices.update(indices_set)

    index_counts = {idx: 0 for idx in all_indices}
    for indices_set in all_topk_indices:
        for idx in indices_set:
            index_counts[idx] += 1

    # Create histogram of stability (how many seeds each index appears in)
    stability_counts = np.array(list(index_counts.values()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Histogram of stability
    ax1.hist(stability_counts, bins=np.arange(0.5, len(valid_seeds) + 1.5, 1),
             edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Number of Seeds Index Appears In', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Number of Indices', fontsize=12, fontweight='bold')
    ax1.set_title(f'Index Stability Across {len(valid_seeds)} Seeds', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add statistics
    n_stable = np.sum(stability_counts == len(valid_seeds))
    n_unstable = np.sum(stability_counts == 1)
    ax1.text(0.95, 0.95, f'Stable (all seeds): {n_stable}\nUnstable (1 seed): {n_unstable}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Pairwise Jaccard similarity between seeds
    n_seeds = len(all_topk_indices)
    jaccard_matrix = np.zeros((n_seeds, n_seeds))

    for i in range(n_seeds):
        for j in range(n_seeds):
            if i == j:
                jaccard_matrix[i, j] = 1.0
            else:
                intersection = len(all_topk_indices[i] & all_topk_indices[j])
                union = len(all_topk_indices[i] | all_topk_indices[j])
                jaccard_matrix[i, j] = intersection / union if union > 0 else 0

    im = ax2.imshow(jaccard_matrix, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    ax2.set_xticks(np.arange(n_seeds))
    ax2.set_yticks(np.arange(n_seeds))
    ax2.set_xticklabels([f'Seed {s}' for s in valid_seeds], rotation=45, ha='right')
    ax2.set_yticklabels([f'Seed {s}' for s in valid_seeds])
    ax2.set_title('Pairwise Jaccard Similarity\n(Top-k Index Overlap)', fontsize=13, fontweight='bold')

    # Add text annotations
    for i in range(n_seeds):
        for j in range(n_seeds):
            ax2.text(j, i, f'{jaccard_matrix[i, j]:.2f}',
                    ha="center", va="center", color="black", fontsize=10)

    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Jaccard Similarity', rotation=270, labelpad=20)

    fig.suptitle(f'Top-{k} Index Stability Analysis (d={budget})', fontsize=15, fontweight='bold')
    plt.tight_layout()

    output_path = Path(output_dir) / f'index_stability_d{budget}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved index stability plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze adaptive subspace allocation experiments')
    parser.add_argument('--results_dir', type=str, default='out/adaptive_experiments',
                       help='Directory with experimental results')
    parser.add_argument('--output_dir', type=str, default='analysis_outputs',
                       help='Directory to save analysis outputs')
    parser.add_argument('--budgets', type=int, nargs='+', default=[10000, 20000],
                       help='Subspace budgets to analyze')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123],
                       help='Random seeds used in experiments')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("ANALYZING ADAPTIVE SUBSPACE ALLOCATION EXPERIMENTS")
    print("=" * 80)

    # Extract metrics
    print("\n[1/8] Extracting bounds metrics from experiments...")
    df = extract_bounds_metrics(args.results_dir)

    if df.empty:
        print("ERROR: No experimental results found. Please ensure bounds evaluation has been run.")
        return

    print(f"Found {len(df)} experimental runs")
    print(f"Configurations: {df['mode'].nunique()}")
    print(f"Budgets: {sorted(df['budget'].unique())}")

    # Generate Pareto frontiers
    print("\n[2/8] Generating Pareto frontier plots...")
    for budget in args.budgets:
        plot_pareto_frontier(df, budget, args.output_dir)

    # Generate compression comparison plots (SubLoRA-style)
    print("\n[3/8] Generating compression comparison plots...")
    for budget in args.budgets:
        plot_compression_comparison(df, budget, args.output_dir)

    # Generate allocation comparison grid
    print("\n[4/8] Generating allocation comparison grid...")
    plot_allocation_comparison_grid(df, args.output_dir)

    # Generate learned gating heatmaps
    print("\n[5/8] Generating learned gating heatmaps...")
    plot_learned_gating_heatmap(args.results_dir, args.budgets, args.seeds, args.output_dir)

    # Generate learned gating trends
    print("\n[6/8] Generating learned gating trend plots...")
    plot_learned_gating_trend(args.results_dir, args.budgets, args.seeds, args.output_dir)

    # Generate gating time-series plots if available
    print("\n[6.5/8] Generating gating time-series plots (per-iteration)...")
    plot_gating_time_series(args.results_dir, args.output_dir)

    # Generate prediction rank visualizations
    print("\n[7/12] Generating prediction rank histograms...")
    for budget in args.budgets:
        for seed in args.seeds:
            plot_topk_histogram(args.results_dir, budget, seed, args.output_dir, k_values=[100, 500, 1000])

    print("\n[8/12] Generating magnitude distribution plots...")
    for budget in args.budgets:
        for seed in args.seeds:
            plot_magnitude_distribution(args.results_dir, budget, seed, args.output_dir)

    print("\n[9/12] Generating sparsity pattern plots...")
    for budget in args.budgets:
        for seed in args.seeds:
            plot_sparsity_patterns(args.results_dir, budget, seed, args.output_dir, percentile_threshold=95)

    print("\n[10/12] Generating index stability analysis...")
    for budget in args.budgets:
        plot_index_stability_across_seeds(args.results_dir, budget, args.seeds, args.output_dir, k=1000)

    # Generate summary table
    print("\n[11/12] Generating summary tables...")
    summary_table = generate_summary_table(df, args.output_dir)

    print("\n[12/12] Summary Statistics:")
    print(summary_table.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE! Outputs saved to: {args.output_dir}")
    print("=" * 80)
    print("\n📊 Generated Visualizations:")
    print(f"\n  Gating Analysis:")
    print(f"    • learned_gating_heatmap_d*.png      - Per-layer γ heatmap")
    print(f"    • learned_gating_trend.png           - γ trends across layers")
    print(f"    • gating_trace_*.png                 - Per-iteration γ evolution")
    print(f"\n  Complexity & Bounds:")
    print(f"    • pareto_frontier_d*.png             - Complexity vs Risk")
    print(f"    • compression_comparison_d*.png      - 3-panel SubLoRA-style")
    print(f"    • allocation_comparison_grid.png     - 2x2 grid comparing methods")
    print(f"\n  Prediction Quality Analysis:")
    print(f"    • prediction_rank_histogram_d*_seed*.png - Prediction rank distributions (0=top-1)")
    print(f"    • magnitude_distribution_d*_seed*.png    - Sorted magnitudes & effective dim")
    print(f"    • sparsity_pattern_d*_seed*.png          - Binary sparsity patterns")
    print(f"    • index_stability_d*.png             - Cross-seed consistency")
    print(f"\n📋 Generated Tables:")
    print(f"    • summary_table.csv                  - Complete numerical results")
    print(f"    • summary_table.tex                  - LaTeX formatted table")

    # Count total files
    n_budgets = len(args.budgets)
    n_seeds = len(args.seeds)
    n_gating_heatmaps = n_budgets
    n_pareto = n_budgets
    n_compression = n_budgets
    n_topk_hist = n_budgets * n_seeds
    n_magnitude = n_budgets * n_seeds
    n_sparsity = n_budgets * n_seeds
    n_stability = n_budgets
    n_gating_trace = n_budgets * n_seeds  # approximate

    total_plots = (n_gating_heatmaps + 1 + n_pareto + n_compression + 1 +
                   n_topk_hist + n_magnitude + n_sparsity + n_stability + n_gating_trace)

    print(f"\n✅ Estimated total: {total_plots}+ plots + 2 tables ready for your paper!")


if __name__ == '__main__':
    main()