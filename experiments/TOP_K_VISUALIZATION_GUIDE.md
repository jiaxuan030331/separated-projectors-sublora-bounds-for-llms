# Top-K Indices Visualization Guide

## Overview

Four new visualization functions have been added to `analyze_results.py` to analyze the top-k indices from the `top_k_indices_levels11_iters100.txt` files.

---

## New Visualization Functions

### 1. **`plot_topk_histogram()`**
**Purpose**: Visualize the distribution of top-k indices across the subspace.

**What it shows**:
- Histogram showing which regions of the d-dimensional space are most utilized
- Multiple subplots for different k values (default: 100, 500, 1000)
- Mean index position as reference line

**Output**: `topk_histogram_d{budget}_seed{seed}.png`

**Interpretation**:
- Uniform distribution → All dimensions equally important
- Skewed distribution → Certain regions more critical
- Clustered indices → Structured importance patterns

---

### 2. **`plot_magnitude_distribution()`**
**Purpose**: Show the effective dimensionality of the learned subspace.

**What it shows**:
- **Left panel**: Sorted parameter magnitudes (log scale)
  - Shows how quickly magnitude drops off
  - Red line marks top-k threshold
- **Right panel**: Cumulative percentage of total magnitude
  - Reference lines at 50%, 90%, 95%, 99%
  - Shows how many dimensions capture X% of total magnitude

**Output**: `magnitude_distribution_d{budget}_seed{seed}.png`

**Interpretation**:
- Steep drop-off → Low effective dimensionality (sparse representation)
- Gradual decay → Higher effective dimensionality
- E.g., "500 dimensions capture 95% of total magnitude" → Effective dim ≈ 500

---

### 3. **`plot_sparsity_patterns()`**
**Purpose**: Reveal structured sparsity in the learned subspace.

**What it shows**:
- **Left panel**: Binary heatmap showing active/inactive indices
  - Reshaped as 2D grid for visualization
  - Green = active (above 95th percentile)
  - Red = inactive
- **Right panel**: Log magnitude heatmap
  - Continuous color scale showing magnitude values

**Output**: `sparsity_pattern_d{budget}_seed{seed}.png`

**Interpretation**:
- Random patterns → No structure, distributed sparsity
- Block patterns → Structured allocation to parameter groups
- Sparsity percentage shown in title
- Reveals if learned gating creates structured activation

---

### 4. **`plot_index_stability_across_seeds()`**
**Purpose**: Analyze consistency of important dimensions across different random seeds.

**What it shows**:
- **Left panel**: Histogram of index stability
  - How many seeds each index appears in
  - Statistics on stable (all seeds) vs unstable (1 seed) indices
- **Right panel**: Jaccard similarity matrix
  - Pairwise overlap between top-k sets across seeds
  - Higher values → more consistent important dimensions

**Output**: `index_stability_d{budget}.png`

**Interpretation**:
- High Jaccard similarity (>0.7) → Stable, reproducible important dimensions
- Low similarity (<0.3) → Training is highly seed-dependent
- Many stable indices → Core dimensions consistently important
- Many unstable indices → Exploration varies across seeds

---

## Usage

### Basic Usage (with defaults)
```bash
python experiments/analyze_results.py \
    --results_dir=out/adaptive_experiments \
    --output_dir=analysis_outputs
```

**Default parameters**:
- Budgets: `[10000, 20000]`
- Seeds: `[42, 123]`

### Custom Parameters
```bash
python experiments/analyze_results.py \
    --results_dir=out/adaptive_experiments \
    --output_dir=analysis_outputs \
    --budgets 10000 20000 \
    --seeds 42 123
```

---

## Expected Outputs

Running the analysis script will now generate:

### Per-Experiment Outputs (d × seed combinations)
For each budget and seed:
1. `topk_histogram_d{budget}_seed{seed}.png` - 3-panel histogram
2. `magnitude_distribution_d{budget}_seed{seed}.png` - 2-panel magnitude analysis
3. `sparsity_pattern_d{budget}_seed{seed}.png` - 2-panel sparsity visualization

### Aggregate Outputs (per budget)
For each budget (across all seeds):
4. `index_stability_d{budget}.png` - Cross-seed stability analysis

**Total new plots** (for default d=10000,20000 and seeds=42,123):
- Top-k histograms: 2 budgets × 2 seeds = 4 plots
- Magnitude distributions: 2 budgets × 2 seeds = 4 plots
- Sparsity patterns: 2 budgets × 2 seeds = 4 plots
- Index stability: 2 budgets = 2 plots
- **Total: 14 new plots**

---

## Integration with Existing Pipeline

The new functions are seamlessly integrated into the existing analysis workflow:

**Execution order**:
1. [1/12] Extract bounds metrics
2. [2/12] Pareto frontiers
3. [3/12] Compression comparison
4. [4/12] Allocation comparison grid
5. [5/12] Learned gating heatmaps
6. [6/12] Learned gating trends
7. [6.5/12] Gating time-series
8. **[7/12] Top-k histograms** ← NEW
9. **[8/12] Magnitude distributions** ← NEW
10. **[9/12] Sparsity patterns** ← NEW
11. **[10/12] Index stability** ← NEW
12. [11/12] Summary tables
13. [12/12] Display statistics

---

## Technical Details

### Function Signatures

```python
def plot_topk_histogram(results_dir, budget, seed, output_dir, k_values=[100, 500, 1000]):
    """
    Args:
        results_dir: Directory with experimental results
        budget: Intrinsic dimension (e.g., 10000, 20000)
        seed: Random seed
        output_dir: Directory to save plots
        k_values: List of k values to plot
    """

def plot_magnitude_distribution(results_dir, budget, seed, output_dir):
    """
    Loads subspace_params from best_ckpt.pt and analyzes magnitude distribution.
    """

def plot_sparsity_patterns(results_dir, budget, seed, output_dir, percentile_threshold=95):
    """
    Args:
        percentile_threshold: Threshold for marking indices as 'active' (default: 95)
    """

def plot_index_stability_across_seeds(results_dir, budget, seeds, output_dir, k=1000):
    """
    Args:
        seeds: List of random seeds to compare
        k: Number of top indices to consider (default: 1000)
    """
```

### Data Sources

1. **Top-k indices**: `top_k_indices_levels11_iters100.txt`
   - Python list format (use `eval()` to parse)
   - Indices ranked by importance after quantization

2. **Subspace parameters**: `best_ckpt.pt`
   - Checkpoint contains `subspace_params` tensor
   - Shape: (d,) where d is intrinsic dimension

3. **File search**: Handles multiple directory structures
   - `SubLoRA_Pretrain/id{dim}_lr0.005_r4/{date}/{time}/`
   - `SubLoRA_LearnedShared/id{dim}_lr0.0005_r4/{date}/{time}/`

---

## Example Interpretations

### Example 1: Effective Dimensionality
```
Magnitude distribution shows:
- 500 dims capture 90% of magnitude
- 1000 dims capture 95% of magnitude
- 2000 dims capture 99% of magnitude

Interpretation: Effective dimensionality is ~1000 out of d=10000
→ 90% compression while retaining 95% of "signal"
```

### Example 2: Stability Analysis
```
Jaccard similarity matrix:
  Seed 42 vs Seed 123: 0.68

Interpretation: 68% overlap in top-1000 indices
→ Core important dimensions are consistent
→ Training is reasonably stable despite random initialization
```

### Example 3: Sparsity Pattern
```
Sparsity analysis shows:
- Active indices: 1523 / 10000 (15.2%)
- Pattern: Clustered blocks in heatmap

Interpretation: Learned gating creates structured sparsity
→ Certain parameter groups heavily utilized
→ Not random sparsity, but intentional allocation
```

---

## Troubleshooting

### Issue: "top_k_indices file not found"
**Solution**: Ensure bounds evaluation has been run with quantization enabled.

### Issue: "Could not find subspace_params in checkpoint"
**Solution**: Verify checkpoint is from the correct training run. Try different checkpoint file paths.

### Issue: "Need at least 2 seeds for stability analysis"
**Solution**: Run experiments with multiple seeds or adjust `--seeds` parameter.

### Issue: Plots look strange or empty
**Solution**:
- Check that data files are not corrupted
- Verify file sizes match expected values (~62 MB for top_k_indices)
- Ensure matplotlib and seaborn are properly installed

---

## Customization Options

### Adjust k values for histogram
```python
plot_topk_histogram(..., k_values=[50, 200, 500, 2000])
```

### Change sparsity threshold
```python
plot_sparsity_patterns(..., percentile_threshold=99)  # Stricter threshold
```

### Modify stability analysis k
```python
plot_index_stability_across_seeds(..., k=500)  # Analyze top-500
```

---

## Paper-Ready Outputs

All plots use:
- **High resolution**: 300 DPI
- **Professional styling**: Seaborn whitegrid theme
- **Clear labels**: Bold axis labels, informative titles
- **Color schemes**:
  - Steelblue for distributions
  - Red/green for binary patterns
  - Viridis/YlOrRd for continuous heatmaps
- **Grid lines**: For easy reading
- **Statistics**: Embedded text annotations with key metrics

---

## Future Extensions

Potential additions (not yet implemented):
1. **Layer-wise index density**: Which indices are used by each transformer layer?
2. **Progressive top-k analysis**: How does top-k change during training?
3. **Index co-occurrence network**: Which indices are activated together?
4. **Comparison with uniform baseline**: How does learned allocation compare to uniform?

---

## Summary

The new visualizations provide comprehensive analysis of which dimensions in the intrinsic subspace are most important, revealing:
1. **Distribution**: Where important indices are located
2. **Dimensionality**: How many dimensions truly matter
3. **Structure**: Whether sparsity is structured or random
4. **Stability**: Whether important dimensions are consistent across training runs

These insights are critical for understanding how learned gating allocates the subspace budget across different parameter groups.
