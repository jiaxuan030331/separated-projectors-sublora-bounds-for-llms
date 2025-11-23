# Adaptive Subspace Allocation Experiments - CLI Commands

## Overview

This document provides all CLI commands for the **30 training runs** described in the ICML 2025 project proposal: "Adaptive Subspace Allocation for Compressed Neural Network Fine-Tuning."

**Experimental Setup:**
- **Model:** GPT-2 Small (124M parameters, 12 layers)
- **LoRA Configuration:** rank r=8 on query and value projections
- **Dataset:** OpenWebText (~9B tokens)
- **Subspace Budgets:** d ∈ {1000, 2000}
- **Allocation Strategies:** uniform, fixed (0.2, 0.5, 0.8), learned
- **Random Seeds:** {42, 123, 999} for each configuration
- **Total Runs:** 10 configurations × 3 seeds = 30 training runs

---

## Quick Start

### Windows (Batch Script)
```bash
# Edit paths in the script
experiments\run_adaptive_experiments.bat
```

### Linux/Mac (Shell Script)
```bash
# Make executable and run
chmod +x experiments/run_adaptive_experiments.sh
./experiments/run_adaptive_experiments.sh
```

### Individual Commands
See sections below for all 30 individual commands.

---

## Configuration Parameters

### Key Command-Line Arguments

| Argument | Description | Values |
|----------|-------------|--------|
| `--sublora.intrinsic_dim` | Subspace dimensionality | 1000, 2000 |
| `--sublora.allocation_mode` | Allocation strategy | uniform, fixed, learned |
| `--sublora.allocation_ratio` | d_B / (d_A + d_B) for fixed mode | 0.2, 0.5, 0.8 |
| `--login.out_dir` | Output directory for checkpoints | Custom path |
| `--login.wandb_run_name` | W&B run identifier | Custom name |
| `--data.dataset_dir` | Path to OpenWebText data | data/openwebtext |

### Allocation Modes Explained

1. **uniform**: Original SubLoRA (baseline)
   - Single projection for all parameters
   - No A/B separation
   - Uses `IDModule`

2. **fixed**: Fixed asymmetric allocation
   - Separate projections for A and B matrices
   - `ratio` determines d_B / (d_A + d_B)
   - Example: ratio=0.8 → d_B=800, d_A=200 (for d=1000)

3. **learned**: Adaptive per-layer gating
   - Learnable sigmoid-gated allocation parameters per layer
   - Soft masking for differentiable split
   - Discovers optimal allocation during training

---

## Experiment Commands

### BUDGET d=1000 (15 runs)

#### 1. Baseline (Uniform) - 3 seeds

**Seed 42:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_uniform_seed42 \
    --login.wandb_run_name=d1000_uniform_s42 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=uniform \
    --sublora.allocation_ratio=0.5
```

**Seed 123:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_uniform_seed123 \
    --login.wandb_run_name=d1000_uniform_s123 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=uniform \
    --sublora.allocation_ratio=0.5
```

**Seed 999:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_uniform_seed999 \
    --login.wandb_run_name=d1000_uniform_s999 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=uniform \
    --sublora.allocation_ratio=0.5
```

---

#### 2. Fixed B-heavy (ratio=0.8, d_B=800, d_A=200) - 3 seeds

**Seed 42:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_fixed_bheavy_seed42 \
    --login.wandb_run_name=d1000_bheavy_s42 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.8
```

**Seed 123:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_fixed_bheavy_seed123 \
    --login.wandb_run_name=d1000_bheavy_s123 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.8
```

**Seed 999:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_fixed_bheavy_seed999 \
    --login.wandb_run_name=d1000_bheavy_s999 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.8
```

---

#### 3. Fixed Equal (ratio=0.5, d_B=500, d_A=500) - 3 seeds

**Seed 42:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_fixed_equal_seed42 \
    --login.wandb_run_name=d1000_equal_s42 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.5
```

**Seed 123:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_fixed_equal_seed123 \
    --login.wandb_run_name=d1000_equal_s123 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.5
```

**Seed 999:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_fixed_equal_seed999 \
    --login.wandb_run_name=d1000_equal_s999 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.5
```

---

#### 4. Fixed A-heavy (ratio=0.2, d_B=200, d_A=800) - 3 seeds

**Seed 42:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_fixed_aheavy_seed42 \
    --login.wandb_run_name=d1000_aheavy_s42 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.2
```

**Seed 123:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_fixed_aheavy_seed123 \
    --login.wandb_run_name=d1000_aheavy_s123 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.2
```

**Seed 999:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_fixed_aheavy_seed999 \
    --login.wandb_run_name=d1000_aheavy_s999 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.2
```

---

#### 5. Learned Gating (Adaptive) - 3 seeds

**Seed 42:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_learned_seed42 \
    --login.wandb_run_name=d1000_learned_s42 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=learned \
    --sublora.allocation_ratio=0.5
```

**Seed 123:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_learned_seed123 \
    --login.wandb_run_name=d1000_learned_s123 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=learned \
    --sublora.allocation_ratio=0.5
```

**Seed 999:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_learned_seed999 \
    --login.wandb_run_name=d1000_learned_s999 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=learned \
    --sublora.allocation_ratio=0.5
```

---

### BUDGET d=2000 (15 runs)

#### 6. Baseline (Uniform) - 3 seeds

**Seed 42:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_uniform_seed42 \
    --login.wandb_run_name=d2000_uniform_s42 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=uniform \
    --sublora.allocation_ratio=0.5
```

**Seed 123:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_uniform_seed123 \
    --login.wandb_run_name=d2000_uniform_s123 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=uniform \
    --sublora.allocation_ratio=0.5
```

**Seed 999:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_uniform_seed999 \
    --login.wandb_run_name=d2000_uniform_s999 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=uniform \
    --sublora.allocation_ratio=0.5
```

---

#### 7. Fixed B-heavy (ratio=0.8, d_B=1600, d_A=400) - 3 seeds

**Seed 42:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_fixed_bheavy_seed42 \
    --login.wandb_run_name=d2000_bheavy_s42 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.8
```

**Seed 123:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_fixed_bheavy_seed123 \
    --login.wandb_run_name=d2000_bheavy_s123 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.8
```

**Seed 999:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_fixed_bheavy_seed999 \
    --login.wandb_run_name=d2000_bheavy_s999 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.8
```

---

#### 8. Fixed Equal (ratio=0.5, d_B=1000, d_A=1000) - 3 seeds

**Seed 42:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_fixed_equal_seed42 \
    --login.wandb_run_name=d2000_equal_s42 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.5
```

**Seed 123:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_fixed_equal_seed123 \
    --login.wandb_run_name=d2000_equal_s123 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.5
```

**Seed 999:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_fixed_equal_seed999 \
    --login.wandb_run_name=d2000_equal_s999 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.5
```

---

#### 9. Fixed A-heavy (ratio=0.2, d_B=400, d_A=1600) - 3 seeds

**Seed 42:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_fixed_aheavy_seed42 \
    --login.wandb_run_name=d2000_aheavy_s42 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.2
```

**Seed 123:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_fixed_aheavy_seed123 \
    --login.wandb_run_name=d2000_aheavy_s123 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.2
```

**Seed 999:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_fixed_aheavy_seed999 \
    --login.wandb_run_name=d2000_aheavy_s999 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=fixed \
    --sublora.allocation_ratio=0.2
```

---

#### 10. Learned Gating (Adaptive) - 3 seeds

**Seed 42:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_learned_seed42 \
    --login.wandb_run_name=d2000_learned_s42 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=learned \
    --sublora.allocation_ratio=0.5
```

**Seed 123:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_learned_seed123 \
    --login.wandb_run_name=d2000_learned_s123 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=learned \
    --sublora.allocation_ratio=0.5
```

**Seed 999:**
```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d2000_learned_seed999 \
    --login.wandb_run_name=d2000_learned_s999 \
    --sublora.intrinsic_dim=2000 \
    --sublora.allocation_mode=learned \
    --sublora.allocation_ratio=0.5
```

---

## Multi-GPU Training (Recommended)

For faster training with 2 GPUs:

```bash
torchrun --standalone --nproc_per_node=2 experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data/openwebtext \
    --login.out_dir=out/adaptive_experiments/d1000_uniform_seed42 \
    --login.wandb_run_name=d1000_uniform_s42 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=uniform \
    --sublora.allocation_ratio=0.5
```

**Adjust `--nproc_per_node` based on available GPUs.**

---

## Post-Training Analysis Pipeline

After training all 30 models, follow this 3-step pipeline:

### Step 1: Bounds Evaluation (For Each Model)

Compute PAC-Bayes bounds for each trained model:

```bash
python experiments/eval_bounds.py \
    --config-file=config/sublora_bounds.yaml \
    --data.dataset_dir=data/openwebtext \
    --model.best_checkpoint_path=out/adaptive_experiments/d1000_uniform_seed42 \
    --bounds.bound_type=document_level \
    --data.openwebtext_train_eot_indices_file=data/openwebtext/eot_indices.npy \
    --data.empirical_document_length_distribution_file=data/openwebtext/doc_lengths.npy
```

**Note:** Requires EOT indices and document length distribution files.

**Run for all 30 configurations** - this will generate `bounds_metrics.pt` in each output directory.

### Step 2: Generate Analysis & Visualizations

After bounds evaluation completes for all models, run the analysis script:

```bash
python experiments/analyze_results.py \
    --results_dir=out/adaptive_experiments \
    --output_dir=analysis_outputs \
    --budgets 1000 2000 \
    --seeds 42 123 999
```

This generates all plots and tables from your proposal:

**Generated Outputs:**
1. `pareto_frontier_d1000.png` - Complexity vs. Risk plot (d=1000)
2. `pareto_frontier_d2000.png` - Complexity vs. Risk plot (d=2000)
3. `learned_gating_heatmap_d1000.png` - Per-layer γ_ℓ values (d=1000)
4. `learned_gating_heatmap_d2000.png` - Per-layer γ_ℓ values (d=2000)
5. `learned_gating_trend.png` - γ_ℓ trends across layers
6. `summary_table.csv` - Complete results table
7. `summary_table.tex` - LaTeX-formatted table for paper

### Step 3: Optional - Extract Training Curves from W&B

If you logged to Weights & Biases, download training curves:

```python
import wandb

api = wandb.Api()
runs = api.runs("YOUR_USERNAME/SubLoRA_Pretrain")

for run in runs:
    history = run.history()
    # Extract training loss, validation loss, etc.
    # Save to CSV or plot
```

---

## Expected Outputs

### Training Outputs
Each run will create:
- `best_ckpt.pt`: Best model checkpoint
- `ckpt_at_random_initialization.pt`: Initial random weights
- `trainable_initparams.pt`: Initial trainable parameters
- `names.pt`: Parameter names
- W&B logs: Training curves, metrics

### Bounds Evaluation Outputs
- Compressed model size (bits)
- KL divergence term
- Empirical risk (BPD)
- PAC-Bayes bound value
- Pareto frontier data

---

## Troubleshooting

### Out of Memory
- Reduce `data.batch_size` (default: 8)
- Increase `training.gradient_accumulation_steps` (default: 40)
- Use smaller budget (d=1000 instead of d=2000)

### Slow Training
- Use multi-GPU training with `torchrun`
- Enable PyTorch 2.0 compilation: `system.compile=True`
- Use mixed precision: `system.dtype=bfloat16`

### Missing Data
- Ensure OpenWebText is downloaded: `python data/openwebtext/prepare.py`
- Check paths in commands match your directory structure

---

## Summary Table

| Config | Budget | Mode | Ratio | d_A | d_B | Seeds | Total Runs |
|--------|--------|------|-------|-----|-----|-------|------------|
| 1 | 1000 | uniform | 0.5 | - | - | 42, 123, 999 | 3 |
| 2 | 1000 | fixed | 0.8 | 200 | 800 | 42, 123, 999 | 3 |
| 3 | 1000 | fixed | 0.5 | 500 | 500 | 42, 123, 999 | 3 |
| 4 | 1000 | fixed | 0.2 | 800 | 200 | 42, 123, 999 | 3 |
| 5 | 1000 | learned | 0.5 | adaptive | adaptive | 42, 123, 999 | 3 |
| 6 | 2000 | uniform | 0.5 | - | - | 42, 123, 999 | 3 |
| 7 | 2000 | fixed | 0.8 | 400 | 1600 | 42, 123, 999 | 3 |
| 8 | 2000 | fixed | 0.5 | 1000 | 1000 | 42, 123, 999 | 3 |
| 9 | 2000 | fixed | 0.2 | 1600 | 400 | 42, 123, 999 | 3 |
| 10 | 2000 | learned | 0.5 | adaptive | adaptive | 42, 123, 999 | 3 |
| **Total** | | | | | | | **30** |

---

## References

- **Project Proposal:** ICML 2025 submission
- **SubLoRA Paper:** Lotfi et al., "Non-Vacuous Generalization Bounds for Large Language Models"
- **LoRA+:** Hayou et al., "LoRA+: Efficient Low Rank Adaptation with Differential Learning Rates"