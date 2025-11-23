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


def extract_bounds_metrics(results_dir):
    """
    Extract bounds metrics from all experiment runs.

    Returns:
        pd.DataFrame with columns: config, seed, budget, mode, ratio,
                                   kl_divergence, empirical_bpd, bound_value
    """
    results = []

    # Expected directory structure: results_dir/d{budget}_{mode}_{detail}_seed{seed}/
    for exp_dir in Path(results_dir).iterdir():
        if not exp_dir.is_dir():
            continue

        # Parse directory name
        name = exp_dir.name
        parts = name.split('_')

        try:
            # Extract budget
            if name.startswith('d1000'):
                budget = 1000
            elif name.startswith('d2000'):
                budget = 2000
            else:
                continue

            # Extract seed
            seed = int(parts[-1].replace('seed', ''))

            # Extract mode
            if 'uniform' in name:
                mode = 'uniform'
                ratio = 0.5
            elif 'learned' in name:
                mode = 'learned'
                ratio = None  # adaptive
            elif 'bheavy' in name:
                mode = 'fixed_bheavy'
                ratio = 0.8
            elif 'aheavy' in name:
                mode = 'fixed_aheavy'
                ratio = 0.2
            elif 'equal' in name:
                mode = 'fixed_equal'
                ratio = 0.5
            else:
                continue

            # Load metrics (assuming they're saved during bounds evaluation)
            metrics_file = exp_dir / 'bounds_metrics.pt'
            if metrics_file.exists():
                metrics = torch.load(metrics_file)

                results.append({
                    'config': name,
                    'seed': seed,
                    'budget': budget,
                    'mode': mode,
                    'ratio': ratio,
                    'kl_divergence': metrics.get('kl_divergence', np.nan),
                    'empirical_bpd': metrics.get('empirical_bpd', np.nan),
                    'bound_value': metrics.get('bound_value', np.nan),
                    'compressed_size_bits': metrics.get('compressed_size_bits', np.nan),
                })
        except Exception as e:
            print(f"Warning: Could not parse {name}: {e}")
            continue

    return pd.DataFrame(results)


def extract_learned_gating(results_dir, budget, seed):
    """
    Extract learned gating parameters (gamma values) for a specific run.

    Returns:
        np.array of shape (num_layers,) with gamma values per layer
    """
    # Find learned gating checkpoint
    checkpoint_path = Path(results_dir) / f'd{budget}_learned_seed{seed}' / 'best_ckpt.pt'

    if not checkpoint_path.exists():
        print(f"Warning: Checkpoint not found: {checkpoint_path}")
        return None

    checkpoint = load_checkpoint(checkpoint_path)

    # Extract gating parameters from state dict
    gating_params = {}
    for key, value in checkpoint['raw_model'].items():
        if 'gating_params' in key:
            # Parse layer index from key like 'gating_params.0'
            layer_idx = int(key.split('.')[-1])
            # Compute gamma = sigmoid(theta_split)
            gamma = torch.sigmoid(value).item()
            gating_params[layer_idx] = gamma

    if not gating_params:
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

    # Average over seeds
    df_avg = df_budget.groupby(['mode', 'ratio']).agg({
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

    colors = {1000: 'blue', 2000: 'red'}

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


def plot_compression_comparison(df, budget, output_dir):
    """
    Create multi-panel plot comparing compression methods (similar to SubLoRA paper Figure).

    Left panel: Complexity vs BPD scatter
    Middle panel: Complexity vs Top-1 Error
    Right panel: Compression extent vs Bound/Risk
    """
    df_budget = df[df['budget'] == budget].copy()
    df_avg = df_budget.groupby(['mode', 'ratio']).agg({
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
    # Average over seeds
    df_avg = df.groupby(['budget', 'mode', 'ratio']).agg({
        'kl_divergence': ['mean', 'std'],
        'empirical_bpd': ['mean', 'std'],
        'compressed_size_bits': ['mean', 'std'],
        'bound_value': ['mean', 'std']
    }).reset_index()

    df_avg.columns = ['budget', 'mode', 'ratio', 'kl_mean', 'kl_std',
                      'bpd_mean', 'bpd_std', 'size_mean', 'size_std',
                      'bound_mean', 'bound_std']

    # Mode styles
    mode_colors = {
        'uniform': '#888888',
        'fixed_bheavy': '#4A90E2',
        'fixed_equal': '#50C878',
        'fixed_aheavy': '#E74C3C',
        'learned': '#9B59B6',
    }

    mode_labels = {
        'uniform': 'Uniform',
        'fixed_bheavy': 'B-heavy (0.8)',
        'fixed_equal': 'Equal (0.5)',
        'fixed_aheavy': 'A-heavy (0.2)',
        'learned': 'Learned',
    }

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: BPD comparison (bar chart)
    ax1 = axes[0, 0]
    for budget in [1000, 2000]:
        df_budget = df_avg[df_avg['budget'] == budget]
        x_pos = np.arange(len(df_budget))
        offset = 0.2 if budget == 1000 else -0.2

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
    for budget in [1000, 2000]:
        df_budget = df_avg[df_avg['budget'] == budget]
        x_pos = np.arange(len(df_budget))
        offset = 0.2 if budget == 1000 else -0.2

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
    for budget in [1000, 2000]:
        df_budget = df_avg[df_avg['budget'] == budget]
        x_pos = np.arange(len(df_budget))
        offset = 0.2 if budget == 1000 else -0.2

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
    for budget in [1000, 2000]:
        df_budget = df_avg[df_avg['budget'] == budget].copy()
        baseline_bpd = df_budget[df_budget['mode'] == 'uniform']['bpd_mean'].values[0]

        df_budget['improvement'] = (baseline_bpd - df_budget['bpd_mean']) / baseline_bpd * 100

        colors = [mode_colors.get(mode, '#888888') for mode in df_budget['mode']]
        x_pos = np.arange(len(df_budget))
        offset = 0.2 if budget == 1000 else -0.2

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
    # Average over seeds
    summary = df.groupby(['budget', 'mode', 'ratio']).agg({
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


def main():
    parser = argparse.ArgumentParser(description='Analyze adaptive subspace allocation experiments')
    parser.add_argument('--results_dir', type=str, default='out/adaptive_experiments',
                       help='Directory with experimental results')
    parser.add_argument('--output_dir', type=str, default='analysis_outputs',
                       help='Directory to save analysis outputs')
    parser.add_argument('--budgets', type=int, nargs='+', default=[1000, 2000],
                       help='Subspace budgets to analyze')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 999],
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

    # Generate summary table
    print("\n[7/8] Generating summary tables...")
    summary_table = generate_summary_table(df, args.output_dir)

    print("\n[8/8] Summary Statistics:")
    print(summary_table.to_string(index=False))

    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE! Outputs saved to: {args.output_dir}")
    print("=" * 80)
    print("\n📊 Generated Visualizations:")
    print(f"  1. pareto_frontier_d1000.png           - Complexity vs Risk (d=1000)")
    print(f"  2. pareto_frontier_d2000.png           - Complexity vs Risk (d=2000)")
    print(f"  3. compression_comparison_d1000.png    - 3-panel SubLoRA-style (d=1000)")
    print(f"  4. compression_comparison_d2000.png    - 3-panel SubLoRA-style (d=2000)")
    print(f"  5. allocation_comparison_grid.png      - 2x2 grid comparing all methods")
    print(f"  6. learned_gating_heatmap_d1000.png    - Per-layer γ heatmap (d=1000)")
    print(f"  7. learned_gating_heatmap_d2000.png    - Per-layer γ heatmap (d=2000)")
    print(f"  8. learned_gating_trend.png            - γ trends across layers")
    print(f"\n📋 Generated Tables:")
    print(f"  9. summary_table.csv                   - Complete numerical results")
    print(f" 10. summary_table.tex                   - LaTeX formatted table")
    print(f"\n✅ Total: 10 output files ready for your paper!")


if __name__ == '__main__':
    main()