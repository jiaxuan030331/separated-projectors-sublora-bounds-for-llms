# Analysis Guide: Bounds & Learned Gating

This guide covers analyzing experimental results after training and bounds evaluation are complete.

## Quick Reference

```
┌─────────────────────────────────────────────────────────────┐
│  Input: 30 experiment folders with bounds results           │
│  Output: 10 visualization files + summary tables            │
│  Time: ~5 minutes (CPU only)                                │
└─────────────────────────────────────────────────────────────┘
```

---

## Overview

The analysis pipeline (`experiments/analyze_results.py`) performs:

1. **Metrics Extraction** - Reads bounds/metrics from all experiments
2. **Pareto Frontier Plots** - Complexity vs. Risk trade-offs
3. **Compression Comparison** - SubLoRA-style 3-panel plots
4. **Allocation Comparison** - Grid comparing all methods
5. **Learned Gating Analysis** - Heatmaps and trends of γ values
6. **Summary Tables** - CSV and LaTeX formats

---

## Data Flow

The analysis script handles the **nested directory structure** from HPC training:

```
Experiment Folders (HPC structure)              Analysis Script
──────────────────────────────────              ───────────────
sublora-d1000-uniform-seed42/
└── out/
    └── SubLoRA_Pretrain/
        └── id1000_lr0.005_r4/
            └── 2025-12-02/
                └── 02-54/                      
                    ├── best_ckpt.pt ──────────┐
                    ├── quant_ckpt_levels*.pt ─┼──→ extract_bounds_metrics()
                    ├── bounds_levels*.yml ────┤        │
                    └── metrics_levels*.yml ───┘        │
                                                        ↓
sublora-d1000-learned-seed42/                  ┌─────────────────────┐
└── out/                                       │ DataFrame with:     │
    └── SubLoRA_Pretrain/                      │ - val_loss          │
        └── id1000_lr0.005_r4/                 │ - kl_divergence     │
            └── {date}/{time}/                 │ - empirical_bpd     │
                ├── best_ckpt.pt ──────────┬───│ - bound_value       │
                │   (contains gating_params)│  │ - compressed_size   │
                └── ...                     │  │ - prefix_msg_len    │
                                            │  └─────────────────────┘
                   extract_learned_gating() ←┘
                            │
                            ↓
                   ┌─────────────────────┐
                   │ Gating γ values:    │
                   │ [γ_0, γ_1, ... γ_11]│
                   │ (12 layers)         │
                   └─────────────────────┘
```

**Note**: The helper functions (`find_best_checkpoint`, `find_quant_checkpoint`, `find_bounds_yaml`, `find_metrics_yaml`) automatically search through the nested `SubLoRA_Pretrain/id{dim}_lr0.005_r4/{date}/{time}/` structure to find the most recent files.

---

## Running the Analysis

### Basic Usage (Local experiments)

```bash
python experiments/analyze_results.py \
    --results_dir=out/adaptive_experiments \
    --output_dir=analysis_outputs \
    --budgets 1000 2000 \
    --seeds 42 123 999
```

### With Downloaded HPC Results

```powershell
# After downloading from HPC to sublora-experiments/hpc_out/
python experiments/analyze_results.py `
    --results_dir=sublora-experiments/hpc_out `
    --output_dir=analysis_outputs `
    --budgets 1000 2000 `
    --seeds 42 123 999
```

---

## Metrics Extraction

### Data Sources (Priority Order)

The `extract_bounds_metrics()` function reads from multiple sources:

| Priority | Source File | Key Metrics |
|----------|-------------|-------------|
| 1 (Primary) | `best_ckpt.pt` | `best_val_loss`, `config` |
| 2 (Primary) | `quant_ckpt_levels*.pt` | `prefix_message_len` |
| 3 (Secondary) | `bounds_levels*.yml` | `best_bpd_bound`, `bpd_divergence` |
| 4 (Secondary) | `metrics_levels*.yml` | `bpd_alpha_*`, `top_k_acc` |

### Metrics Computed

| Metric | Formula | Description |
|--------|---------|-------------|
| `val_loss` | From checkpoint | Best validation cross-entropy (nats) |
| `prefix_message_len` | From quant checkpoint | Compressed model size (bits) |
| `kl_divergence` | `prefix_message_len × log(2)` | KL(Q‖P) in nats |
| `empirical_bpd` | From metrics YAML or `val_loss / log(2)` | Bits per dimension |
| `bound_value` | `best_bpd_bound` from bounds YAML | PAC-Bayes generalization bound |

### Experiment Naming Convention

The script parses experiment folder names to extract configuration:

```
sublora-d{budget}-{mode}-seed{seed}
         │         │        │
         │         │        └─── Random seed (42, 123, 999)
         │         └──────────── Mode (uniform, fixed-bheavy, learned, etc.)
         └────────────────────── Budget/intrinsic_dim (1000, 2000)
```

**Recognized Modes**:
- `uniform` → Baseline (d_A = d_B = d/2)
- `fixed-bheavy` or `bheavy` → B-heavy (ratio=0.8)
- `fixed-equal` or `equal` → Equal (ratio=0.5)  
- `fixed-aheavy` or `aheavy` → A-heavy (ratio=0.2)
- `learned` → Learned gating

---

## Learned Gating Analysis

### Where Gating Parameters Are Stored

During training with `allocation_mode=learned`, the `StructuredIDModule` creates learnable gating parameters:

```python
# In sublora/nn/projectors.py (StructuredIDModule.__init__)
self.gating_params = nn.ParameterDict()
for layer_idx in range(self.num_layers):
    # θ_split initialized to 0, so γ = sigmoid(0) = 0.5
    self.gating_params[str(layer_idx)] = nn.Parameter(torch.zeros(1))
```

These are saved in `best_ckpt.pt` under the model state dict with keys like:
- `intrinsic_model.gating_params.0`
- `intrinsic_model.gating_params.1`
- ... through `intrinsic_model.gating_params.11` (for 12 layers)

### Extracting Gating Parameters

The `extract_learned_gating()` function:

```python
def extract_learned_gating(results_dir, budget, seed):
    """
    Extract learned gating parameters (gamma values) for a specific run.
    
    Returns:
        np.array of shape (num_layers,) with γ values per layer
        where γ_l = sigmoid(θ_split_l)
    """
```

**What it does**:
1. Finds checkpoint for `sublora-d{budget}-learned-seed{seed}`
2. Loads model state dict
3. Searches for keys containing `gating_param`, `allocation_logit`, or `theta_split`
4. Applies sigmoid to convert logits to γ values: $\gamma_\ell = \sigma(\theta_\ell)$
5. Returns array of shape `(num_layers,)`

### Gating Parameter Interpretation

| γ Value | Meaning | Allocation |
|---------|---------|------------|
| γ = 0.0 | All to A | d_A = d, d_B = 0 |
| γ = 0.5 | Equal split | d_A = d/2, d_B = d/2 |
| γ = 1.0 | All to B | d_A = 0, d_B = d |

---

## Generated Visualizations

### 1. Pareto Frontier Plots (`pareto_frontier_d{budget}.png`)

**Purpose**: Show complexity-performance trade-offs across allocation strategies.

**Axes**:
- X: KL Divergence (model complexity)
- Y: Empirical BPD (risk/performance)

**Points**: Each allocation strategy (mean ± std over 3 seeds)

**Interpretation**:
- Points closer to origin (lower-left) are better
- Pareto-optimal points dominate others
- Compare learned vs fixed strategies

### 2. Compression Comparison (`compression_comparison_d{budget}.png`)

**Purpose**: 3-panel SubLoRA-style visualization (similar to paper Figure 16)

**Panels**:
1. **Left**: Complexity vs BPD scatter
2. **Middle**: Complexity vs PAC-Bayes Bound
3. **Right**: Compression extent (KB, log scale) vs Performance

### 3. Allocation Comparison Grid (`allocation_comparison_grid.png`)

**Purpose**: 2×2 grid comparing all methods head-to-head

**Subplots**:
- **Top-Left**: Bar chart of empirical BPD by method
- **Top-Right**: Bar chart of PAC-Bayes bounds by method
- **Bottom-Left**: Bar chart of compressed model size
- **Bottom-Right**: Relative improvement over baseline (%)

### 4. Learned Gating Heatmaps (`learned_gating_heatmap_d{budget}.png`)

**Purpose**: Visualize per-layer allocation patterns learned during training

**Structure**:
- **Rows**: Random seeds (42, 123, 999)
- **Columns**: Transformer layers (0-11)
- **Color**: γ_ℓ value (blue=A-heavy, red=B-heavy, white=equal)

**Key Questions Answered**:
- Do different seeds learn similar patterns?
- Are there consistent layer-wise preferences?
- Do early/late layers prefer different allocations?

### 5. Learned Gating Trend (`learned_gating_trend.png`)

**Purpose**: Show how γ evolves across layer depth

**Structure**:
- **X-axis**: Layer depth (0-11)
- **Y-axis**: γ_ℓ (mean ± std over seeds)
- **Lines**: Separate for d=1000 and d=2000
- **Reference**: Dashed line at γ=0.5 (equal split)

**Key Questions Answered**:
- Does γ increase or decrease with depth?
- Is the pattern consistent across budget sizes?
- Is there a "sweet spot" layer range?

### 6. Summary Tables

**`summary_table.csv`**: Complete numerical results for all configurations

| Column | Description |
|--------|-------------|
| budget | Intrinsic dimension (1000, 2000) |
| mode | Allocation strategy |
| ratio | Fixed ratio (for fixed mode) |
| bpd_mean/std | Empirical BPD statistics |
| kl_mean/std | KL divergence statistics |
| bound_mean/std | PAC-Bayes bound statistics |
| size_mean/std | Compressed size (KB) statistics |

**`summary_table.tex`**: LaTeX-formatted table ready for paper

---

## Analysis Workflow

```
[1/8] Extracting bounds metrics from experiments...
      └─→ Reads all experiment folders
      └─→ Creates DataFrame with 30 rows (10 configs × 3 seeds)

[2/8] Generating Pareto frontier plots...
      └─→ pareto_frontier_d1000.png
      └─→ pareto_frontier_d2000.png

[3/8] Generating compression comparison plots...
      └─→ compression_comparison_d1000.png
      └─→ compression_comparison_d2000.png

[4/8] Generating allocation comparison grid...
      └─→ allocation_comparison_grid.png

[5/8] Generating learned gating heatmaps...
      └─→ Loads gating params from learned checkpoints
      └─→ learned_gating_heatmap_d1000.png
      └─→ learned_gating_heatmap_d2000.png

[6/8] Generating learned gating trend plots...
      └─→ learned_gating_trend.png

[7/8] Generating summary tables...
      └─→ summary_table.csv
      └─→ summary_table.tex

[8/8] Summary Statistics
      └─→ Prints final table to console
```

---

## Expected Output Structure

```
analysis_outputs/
├── pareto_frontier_d1000.png           # Complexity vs Risk (d=1000)
├── pareto_frontier_d2000.png           # Complexity vs Risk (d=2000)
├── compression_comparison_d1000.png    # 3-panel SubLoRA-style (d=1000)
├── compression_comparison_d2000.png    # 3-panel SubLoRA-style (d=2000)
├── allocation_comparison_grid.png      # 2x2 grid comparing all methods
├── learned_gating_heatmap_d1000.png    # Per-layer γ heatmap (d=1000)
├── learned_gating_heatmap_d2000.png    # Per-layer γ heatmap (d=2000)
├── learned_gating_trend.png            # γ trends across layers
├── summary_table.csv                   # Complete numerical results
└── summary_table.tex                   # LaTeX formatted table
```

---

## Hypothesis Testing

### Hypothesis 1: Asymmetric Allocation Improves Trade-offs

**Check**: Do fixed B-heavy or A-heavy points dominate equal split on Pareto frontier?

**Where to look**: `pareto_frontier_d{budget}.png`

**Expected**: B-heavy (blue triangles) should be closer to origin than equal (green circles)

### Hypothesis 2: Learned Gating Discovers Patterns

**Check**: Do learned γ_ℓ values differ significantly from 0.5 across layers?

**Where to look**: `learned_gating_heatmap_d{budget}.png`

**Expected**: Consistent patterns across seeds (not random noise)

### Hypothesis 3: Layer-Depth Correlation

**Check**: Does γ_ℓ increase or decrease with layer depth?

**Where to look**: `learned_gating_trend.png`

**Expected**: Clear trend (e.g., early layers prefer A, late layers prefer B)

---

## Troubleshooting

### "No experimental results found"

Check that experiment folders exist and have the expected structure:
```bash
ls out/adaptive_experiments/*/out/bounds_levels*.yml
```

### "Warning: No learned gating data found"

The gating parameters may have different key names. Check what keys exist:
```python
import torch
ckpt = torch.load('path/to/best_ckpt.pt', map_location='cpu')
gating_keys = [k for k in ckpt['raw_model'].keys() if 'gating' in k.lower()]
print(gating_keys)
```

### Empty Heatmaps

Ensure experiments were run with `allocation_mode=learned`:
```bash
grep -r "allocation_mode" out/adaptive_experiments/*/config/
```

### Missing Metrics

If metrics YAML files are missing, the script falls back to checkpoint data:
- `val_loss` from `best_ckpt.pt` is used as BPD proxy
- `prefix_message_len` from `quant_ckpt.pt` provides compression size

---

## Advanced Usage

### Custom Budgets/Seeds

```bash
python experiments/analyze_results.py \
    --results_dir=out/custom_experiments \
    --budgets 500 1000 1500 2000 \
    --seeds 1 2 3 4 5
```

### Programmatic Access

```python
from experiments.analyze_results import (
    extract_bounds_metrics,
    extract_learned_gating,
    plot_pareto_frontier
)

# Load all metrics
df = extract_bounds_metrics('out/adaptive_experiments')

# Get gating for specific run
gammas = extract_learned_gating('out/adaptive_experiments', budget=1000, seed=42)
print(f"Layer-wise gammas: {gammas}")

# Filter and analyze
learned_df = df[df['mode'] == 'learned']
print(f"Learned mode average BPD: {learned_df['empirical_bpd'].mean():.4f}")
```

### Comparing Specific Configurations

```python
import pandas as pd

df = extract_bounds_metrics('out/adaptive_experiments')

# Compare uniform vs learned for d=1000
comparison = df[df['budget'] == 1000].groupby('mode').agg({
    'empirical_bpd': ['mean', 'std'],
    'bound_value': ['mean', 'std'],
    'kl_divergence': ['mean', 'std']
})

print(comparison)
```

---

## Paper-Ready Figures

The generated plots use:
- **Font size**: 12pt (configurable in script)
- **Figure size**: 10×7 inches (configurable)
- **Style**: Seaborn whitegrid
- **Format**: PNG (300 DPI)

To generate PDF/EPS for publication:
```python
# Modify in analyze_results.py
plt.savefig(output_path, format='pdf', bbox_inches='tight')
```

---

## References

- **SubLoRA Paper**: Lotfi et al., "Non-Vacuous Generalization Bounds for Large Language Models"
- **PAC-Bayes Bounds**: Catoni (2007), "PAC-Bayesian Supervised Classification"
- **Analysis Script**: `experiments/analyze_results.py`
