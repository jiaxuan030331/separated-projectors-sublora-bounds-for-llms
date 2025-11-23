# Complete Experimental Workflow - Quick Reference

## Overview

This document provides a **complete end-to-end workflow** for running all experiments and generating all visualizations described in your ICML 2025 project proposal.

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Train 30 Models                                    │
│  Time: ~135 GPU-hours total (4.5 hours × 30 runs)          │
│  Output: Checkpoints in out/adaptive_experiments/          │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Evaluate Bounds (30 evaluations)                  │
│  Time: ~22.5 GPU-hours total (45 min × 30 runs)            │
│  Output: bounds_metrics.pt in each checkpoint dir          │
└─────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Generate Visualizations                            │
│  Time: ~5 minutes (CPU)                                     │
│  Output: 7 plots/tables in analysis_outputs/               │
└─────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Instructions

### Prerequisites

1. **Data Preparation:**
   ```bash
   python data/openwebtext/prepare.py
   ```
   Creates `data/openwebtext/train.bin` and `data/openwebtext/val.bin`

2. **Environment Setup:**
   ```bash
   conda env create -f environment.yml -n sublora
   conda activate sublora
   pip install -e .
   ```

---

### Step 1: Train All 30 Models

**Option A: Batch Script (Windows)**
```bash
# Edit paths in the script first
experiments\run_adaptive_experiments.bat
```

**Option B: Individual Commands**
See `EXPERIMENT_COMMANDS.md` for all 30 commands

**Option C: Multi-GPU (Recommended for speed)**
```bash
# Example: 2 GPUs
torchrun --standalone --nproc_per_node=2 experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_uniform_seed42 \
    --login.wandb_run_name=d1000_uniform_s42 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=uniform
```

**Expected Training Outputs per Model:**
```
out/adaptive_experiments/d1000_uniform_seed42/
├── best_ckpt.pt                              # Best checkpoint
├── ckpt_at_random_initialization.pt          # Initial weights
├── forward_ckpt_at_random_initialization.pt  # Forward model init
├── trainable_initparams.pt                   # LoRA init params
└── names.pt                                  # Parameter names
```

---

### Step 2: Evaluate Bounds for All Models

You need to run `eval_bounds.py` for **each of the 30 trained models**.

**Example Loop Script (create as `run_all_bounds.sh`):**
```bash
#!/bin/bash
DATA_DIR="data/openwebtext"
RESULTS_DIR="out/adaptive_experiments"
EOT_FILE="data/openwebtext/eot_indices.npy"
DOC_LENGTHS="data/openwebtext/doc_lengths.npy"

# Loop through all checkpoint directories
for checkpoint_dir in $RESULTS_DIR/*/; do
    if [ -f "$checkpoint_dir/best_ckpt.pt" ]; then
        echo "Evaluating bounds for: $checkpoint_dir"

        python experiments/eval_bounds.py \
            --config-file=config/sublora_bounds.yaml \
            --data.dataset_dir=$DATA_DIR \
            --model.best_checkpoint_path=$checkpoint_dir \
            --bounds.bound_type=document_level \
            --data.openwebtext_train_eot_indices_file=$EOT_FILE \
            --data.empirical_document_length_distribution_file=$DOC_LENGTHS
    fi
done
```

**IMPORTANT:** The bounds evaluation script needs to **save metrics** to `bounds_metrics.pt`. You may need to modify `experiments/eval_bounds.py` to ensure it saves:

```python
# Add at the end of eval_bounds.py
metrics = {
    'kl_divergence': kl_value,
    'empirical_bpd': empirical_risk,
    'bound_value': bound_value,
    'compressed_size_bits': compressed_bits
}
torch.save(metrics, os.path.join(checkpoint_path, 'bounds_metrics.pt'))
```

---

### Step 3: Generate All Visualizations

After bounds evaluation completes for all 30 models:

```bash
python experiments/analyze_results.py \
    --results_dir=out/adaptive_experiments \
    --output_dir=analysis_outputs \
    --budgets 1000 2000 \
    --seeds 42 123 999
```

**Generated Outputs:**
```
analysis_outputs/
├── pareto_frontier_d1000.png              # Figure: Complexity vs Risk (d=1000)
├── pareto_frontier_d2000.png              # Figure: Complexity vs Risk (d=2000)
├── compression_comparison_d1000.png       # Figure: 3-panel SubLoRA-style plot (d=1000)
├── compression_comparison_d2000.png       # Figure: 3-panel SubLoRA-style plot (d=2000)
├── allocation_comparison_grid.png         # Figure: 2x2 grid comparing all methods
├── learned_gating_heatmap_d1000.png       # Figure: Per-layer γ heatmap (d=1000)
├── learned_gating_heatmap_d2000.png       # Figure: Per-layer γ heatmap (d=2000)
├── learned_gating_trend.png               # Figure: γ trends across layers
├── summary_table.csv                      # Table: All numerical results
└── summary_table.tex                      # Table: LaTeX format for paper
```

**Total: 10 output files (8 visualizations + 2 tables)**

---

## Visual Outputs Explained

### 1. Pareto Frontier Plots (2 plots)
- **X-axis:** KL divergence (model complexity)
- **Y-axis:** Empirical BPD (risk)
- **Points:** Each allocation strategy (uniform, fixed 0.2/0.5/0.8, learned)
- **Error bars:** Standard deviation over 3 seeds
- **Purpose:** Show if asymmetric allocation achieves better compression-performance trade-offs

### 2. Compression Comparison Plots (2 plots) - SubLoRA-Style
Similar to the attached image with **3 panels**:
- **Left Panel:** Complexity vs BPD scatter plot
- **Middle Panel:** Complexity vs PAC-Bayes Bound (generalization)
- **Right Panel:** Compression extent (KB, log scale) vs Performance
  - Solid lines: Bound values
  - Dashed lines: Empirical risk
- **Purpose:** Comprehensive view like SubLoRA paper Figure 16

### 3. Allocation Comparison Grid (1 plot) - 2×2 Multi-Metric View
Four subplots comparing all methods:
- **Top-Left:** Bar chart of empirical BPD by method
- **Top-Right:** Bar chart of PAC-Bayes bounds by method
- **Bottom-Left:** Bar chart of compressed model size by method
- **Bottom-Right:** Relative improvement over baseline (%)
- **Purpose:** Direct comparison across all metrics

### 4. Learned Gating Heatmaps (2 plots)
- **Rows:** Random seeds (42, 123, 999)
- **Columns:** Transformer layers (0-11)
- **Color:** γ_ℓ value (0=all to A, 1=all to B)
- **Cell values:** Numeric γ_ℓ annotations
- **Purpose:** Visualize learned allocation patterns per layer

### 5. Learned Gating Trend Plot (1 plot)
- **X-axis:** Layer depth (0-11)
- **Y-axis:** γ_ℓ (mean ± std over seeds)
- **Lines:** d=1000 (blue), d=2000 (red)
- **Reference:** Dashed line at γ=0.5 (equal split)
- **Purpose:** Show if early vs late layers prefer different allocations

### 6. Summary Tables (2 files)
- **Columns:** Budget, Mode, Ratio, BPD, KL, Bound, Size
- **Rows:** All 10 configurations
- **Format:** CSV (for analysis) + LaTeX (for paper)
- **Purpose:** Comprehensive numerical results

---

## Expected Results (From Proposal)

### Hypothesis 1: Asymmetric Allocation Improves Trade-offs
✅ **Check:** Do fixed B-heavy or A-heavy points dominate equal split on Pareto frontier?

### Hypothesis 2: Learned Gating Discovers Patterns
✅ **Check:** Do learned γ_ℓ values differ significantly from 0.5 across layers?

### Hypothesis 3: Layer-Depth Correlation
✅ **Check:** Does γ_ℓ increase or decrease with layer depth in trend plot?

---

## Compute Resources

### Estimated Time (Single NVIDIA A100 80GB)
- **Training:** 4.5 hours/model × 30 = **135 GPU-hours** (~5.6 days)
- **Bounds Eval:** 45 min/model × 30 = **22.5 GPU-hours** (~1 day)
- **Analysis:** 5 minutes (CPU)
- **Total:** ~6.6 GPU-days

### Parallelization (2× A100)
- **Training:** ~3 GPU-days (15 models per GPU)
- **Bounds Eval:** ~0.5 GPU-days
- **Total:** ~3.5 GPU-days

### Storage Requirements
- **OpenWebText:** ~38 GB
- **Checkpoints:** 30 × ~500 MB = ~15 GB
- **Total:** ~55 GB

---

## Troubleshooting

### Issue: Bounds evaluation fails with missing metrics
**Fix:** Modify `experiments/eval_bounds.py` to save `bounds_metrics.pt`:
```python
# Add at end of evaluation
import torch
import os
metrics = {
    'kl_divergence': kl_divergence_value,
    'empirical_bpd': empirical_bpd_value,
    'bound_value': pac_bayes_bound_value,
    'compressed_size_bits': compressed_size
}
save_path = os.path.join(checkpoint_dir, 'bounds_metrics.pt')
torch.save(metrics, save_path)
print(f"Saved metrics to {save_path}")
```

### Issue: Analysis script finds no data
**Fix:** Ensure all 30 models have `bounds_metrics.pt` in their directories:
```bash
find out/adaptive_experiments -name "bounds_metrics.pt"
# Should show 30 files
```

### Issue: Learned gating heatmap is empty
**Fix:** Check that `StructuredIDModule` with `mode='learned'` was used:
```python
# Verify checkpoint has gating_params
checkpoint = torch.load('path/to/best_ckpt.pt')
gating_keys = [k for k in checkpoint['raw_model'].keys() if 'gating' in k]
print(gating_keys)  # Should show gating_params.0, gating_params.1, etc.
```

---

## Quick Commands Reference

### Single Model Training (Example)
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/test_run \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=learned
```

### Single Model Bounds Evaluation
```bash
python experiments/eval_bounds.py \
    --config-file=config/sublora_bounds.yaml \
    --data.dataset_dir=data/openwebtext \
    --model.best_checkpoint_path=out/test_run \
    --bounds.bound_type=document_level
```

### Generate Visualizations
```bash
python experiments/analyze_results.py \
    --results_dir=out/adaptive_experiments \
    --output_dir=analysis_outputs
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `EXPERIMENT_COMMANDS.md` | All 30 individual training commands |
| `experiments/run_adaptive_experiments.bat` | Windows batch script for all 30 runs |
| `experiments/analyze_results.py` | Visualization generation script |
| `config/sublora_train.yaml` | Training configuration |
| `config/sublora_bounds.yaml` | Bounds evaluation configuration |
| `sublora/nn/projectors.py` | Core implementation (StructuredIDModule) |

---

## Citation for Generated Results

When using these experiments in your paper:

```bibtex
@inproceedings{yourname2025adaptive,
  title={Adaptive Subspace Allocation for Compressed Neural Network Fine-Tuning},
  author={Huang, Jiaxuan and Jing, Rui and Son, Sunny},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```

---

## Summary Checklist

- [ ] Data prepared (OpenWebText downloaded and tokenized)
- [ ] Environment set up (conda env with all dependencies)
- [ ] 30 models trained (checkpoints saved)
- [ ] 30 bounds evaluations completed (metrics saved)
- [ ] 7 visualizations generated (plots and tables)
- [ ] Results analyzed and ready for paper

**Good luck with your experiments!** 🚀