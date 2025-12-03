# NYU HPC Bounds Evaluation Guide

This guide covers running **PAC-Bayes bounds evaluation** on NYU HPC Cloud Bursting after training is complete.

## Quick Reference

| Stage | Time per Run | Total (30 runs) | GPU |
|-------|-------------|-----------------|-----|
| Bounds Evaluation | ~45 min | ~22.5 hours | A100 |

---

## Overview

The bounds evaluation phase:
1. **Quantizes** the trained model parameters (reduces bits)
2. **Samples** 10,000 documents from training data
3. **Computes** PAC-Bayes generalization bounds

```
┌─────────────────────────────────────────────────────────────┐
│  Input: best_ckpt.pt (trained checkpoint)                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 1: Quantization                                       │
│  - K-means clustering on parameters                         │
│  - Reduces bits needed to encode model                      │
│  Output: quant_ckpt_levels{X}_iters{Y}.pt                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 2: Document Sampling                                  │
│  - Sample 10,000 documents from OpenWebText                 │
│  - Compute per-document BPD and accuracy                    │
│  Output: ix_levels{X}.txt, top_k_indices_levels{X}.txt      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  Step 3: Bound Computation                                  │
│  - PAC-Bayes bound with subsampling                         │
│  - Multiple alpha values for BPD bounds                     │
│  Output: bounds_levels{X}_iters{Y}.yml                      │
│          metrics_levels{X}_iters{Y}.yml                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

Training creates a **nested directory structure** with timestamps:

```
/scratch/${HPC_USER}/sublora-experiments/
└── sublora-d{dim}-{mode}-seed{seed}/           # e.g., sublora-d1000-uniform-seed42
    ├── config/
    │   └── sublora_train.yaml
    ├── logs/
    │   ├── <training_job_id>.out
    │   └── <training_job_id>.err
    └── out/
        └── SubLoRA_Pretrain/
            └── id{dim}_lr0.005_r4/             # e.g., id1000_lr0.005_r4
                └── {date}/                      # e.g., 2025-12-02
                    └── {time}/                  # e.g., 02-54
                        ├── best_ckpt.pt         ← INPUT: trained checkpoint
                        ├── ckpt_at_random_initialization.pt
                        ├── trainable_initparams.pt
                        ├── names.pt
                        │
                        │ After bounds evaluation:
                        ├── quant_ckpt_levels11_iters100.pt   ← OUTPUT
                        ├── bounds_levels11_iters100.yml      ← OUTPUT
                        ├── metrics_levels11_iters100.yml     ← OUTPUT
                        ├── ix_levels11_iters100.txt
                        ├── top_k_indices_levels11_iters100.txt
                        ├── selected_prob_scores_levels11_iters100.txt
                        └── percentile_vec_levels11_iters100.txt
```

---

## HPC_USER Configuration

The bounds evaluation scripts support the `HPC_USER` variable to use another user's pre-configured environment and data. See [HPC_TRAINING.md](./HPC_TRAINING.md#2a-using-another-users-setup-hpc_user-configuration) for full details.

```bash
# Submit bounds jobs using another user's setup
HPC_USER=sons01 ./experiments/submit_bounds_jobs.sh

# Or pass via sbatch --export
sbatch --job-name=bounds-test \
       --export=CHECKPOINT_DIR=/scratch/sons01/...,HPC_USER=sons01 \
       experiments/run_bounds_job.slurm
```

**Important**: The `CHECKPOINT_DIR` for bounds evaluation should point to the **innermost timestamp folder** containing `best_ckpt.pt`, NOT the top-level experiment folder.

---

## Prerequisites

Before running bounds evaluation:

1. **Training must be complete**: Ensure `best_ckpt.pt` exists in the timestamp folder
2. **Data files required** (in `/scratch/$USER/sublora-data/`):
   - `train.bin` - Training data
   - `val.bin` - Validation data  
   - `eot_indices.npy` - End-of-text token positions
   - `doc_lengths.npy` - Document length distribution

---

## Submitting Bounds Evaluation Jobs

### Option 1: Submit All 30 Jobs (Recommended)

Use the `submit_bounds_jobs.sh` script which automatically finds the latest checkpoint for each experiment:

```bash
cd /scratch/$USER/sublora-repo/experiments
chmod +x submit_bounds_jobs.sh
./submit_bounds_jobs.sh
```

The script:
1. Iterates through all `sublora-d*` experiment folders
2. Finds the most recent timestamp folder with `best_ckpt.pt`
3. Skips experiments where bounds are already computed
4. Submits SLURM jobs for remaining experiments

### Option 2: Submit Individual Jobs

First, find the checkpoint path:
```bash
# Find the latest checkpoint for a specific experiment
exp_name="sublora-d1000-uniform-seed42"
d_num=1000
base_path="/scratch/$USER/sublora-experiments/${exp_name}/out/SubLoRA_Pretrain/id${d_num}_lr0.005_r4"
ckpt_dir=$(ls -d ${base_path}/*/* 2>/dev/null | sort -r | head -1)
echo "Checkpoint dir: ${ckpt_dir}"
```

Then submit:
```bash
sbatch --job-name=bounds-sublora-d1000-uniform-seed42 \
       --export=CHECKPOINT_DIR=${ckpt_dir} \
       /scratch/$USER/sublora-repo/experiments/run_bounds_job.slurm
```

### Example Commands for All Experiments

```bash
# d=1000 experiments
sbatch --job-name=bounds-d1000-uniform-s42 \
    --export=CHECKPOINT_DIR=/scratch/$USER/sublora-experiments/sublora-d1000-uniform-seed42/out/SubLoRA_Pretrain/id1000_lr0.005_r4/2025-12-02/02-54 \
    /scratch/$USER/sublora-repo/experiments/run_bounds_job.slurm

# Repeat for other experiments, adjusting the timestamp path as needed
```

---

## Local Testing (Windows)

To test bounds evaluation locally before running on HPC:

```powershell
# Quick test with minimal samples (for debugging)
python experiments/eval_bounds.py `
    --config-file=config/sublora_bounds.yaml `
    --data.dataset_dir=data/openwebtext `
    --data.openwebtext_train_eot_indices_file=data/openwebtext/eot_indices.npy `
    --data.empirical_document_length_distribution_file=data/openwebtext/doc_lengths.npy `
    --model.best_checkpoint_path="sublora-experiments/hpc_out/sublora-d1000-fixed-bheavy-seed42/out/SubLoRA_Pretrain/id1000_lr0.005_r4/2025-12-02/02-54" `
    --bounds.bound_samples=10 `
    --bounds.max_quant_iters=5 `
    --bounds.levels=3 `
    --login.wandb_log=False
```

**Note**: For downloaded HPC results, the checkpoint path includes the full nested structure.

---

## Output Files

After bounds evaluation, files are written to the **same directory** as `best_ckpt.pt`:

```
.../id1000_lr0.005_r4/2025-12-02/02-54/
├── best_ckpt.pt                               # From training
├── quant_ckpt_levels11_iters100.pt            # Quantized checkpoint
├── bounds_levels11_iters100.yml               # PAC-Bayes bounds
├── metrics_levels11_iters100.yml              # Empirical metrics
├── ix_levels11_iters100.txt                   # Sampled document indices
├── top_k_indices_levels11_iters100.txt        # Top-k predictions
├── selected_prob_scores_levels11_iters100.txt
└── percentile_vec_levels11_iters100.txt
```

### Key Output Files Explained

#### `quant_ckpt_levels11_iters100.pt`
PyTorch checkpoint after quantization containing:
- `raw_model`: Quantized model state dict
- `prefix_message_len`: **Compressed model size in bits** (KL divergence proxy)
- `config`: Original training config

#### `bounds_levels11_iters100.yml`
YAML file with computed bounds:
```yaml
prefix_message_len: 12345.67          # Bits to encode model
bpd_divergence: 8567.89               # KL divergence (nats)
acc_divergence: 8567.89               # KL divergence for accuracy bounds
best_bpd_bound: 4.523                 # Best PAC-Bayes BPD bound
bound_bpd_alpha_0.01: 4.523           # BPD bound at α=0.01
bound_bpd_alpha_0.001: 4.678          # BPD bound at α=0.001
bound_top_1_acc: 0.234                # Top-1 accuracy bound
bound_top_5_acc: 0.567                # Top-5 accuracy bound
...
```

#### `metrics_levels11_iters100.yml`
YAML file with empirical metrics:
```yaml
n_train: 10000                        # Number of documents sampled
bpd_alpha_0.01: 3.456                 # Empirical BPD at α=0.01
bpd_alpha_0.001: 3.489                # Empirical BPD at α=0.001
top_1_acc: 0.432                      # Top-1 token accuracy
top_5_acc: 0.712                      # Top-5 token accuracy
top_10_acc: 0.823                     # Top-10 token accuracy
...
```

---

## Monitoring Jobs

```bash
# Check job queue
squeue -u $USER

# Watch bounds evaluation progress (find the log file first)
exp_name="sublora-d1000-uniform-seed42"
tail -f /scratch/$USER/sublora-experiments/${exp_name}/logs/*.out

# Check which experiments have bounds computed
for exp in /scratch/$USER/sublora-experiments/sublora-d*/; do
    exp_name=$(basename "$exp")
    d_num=$(echo "${exp_name}" | grep -oP '(?<=-d)\d+')
    base_path="${exp}/out/SubLoRA_Pretrain/id${d_num}_lr0.005_r4"
    ckpt_dir=$(ls -d ${base_path}/*/* 2>/dev/null | sort -r | head -1)
    if [ -f "${ckpt_dir}/bounds_levels11_iters100.yml" ]; then
        echo "✓ ${exp_name}"
    else
        echo "✗ ${exp_name} (missing bounds)"
    fi
done

# Count completed bounds evaluations
find /scratch/$USER/sublora-experiments -name "bounds_levels11_iters100.yml" | wc -l
# Should show 30 when all complete
```

---

## Configuration Reference

### Bounds Config (`config/sublora_bounds.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bounds.levels` | 11 | Quantization levels |
| `bounds.max_quant_iters` | 100 | K-means iterations |
| `bounds.use_kmeans` | True | Use K-means quantization |
| `bounds.bound_samples` | 10000 | Documents to sample |
| `bounds.bound_type` | document_level | Bound granularity |
| `bounds.misc_extra_bits` | 0 | Extra bits for misc params |
| `bounds.sliding_window_size` | 512 | Sliding window for BPD |

### Bound Types

- **`document_level`**: Sample complete documents, more accurate but slower
- **`sequence_level`**: Sample fixed-length sequences, faster but less accurate

---

## Downloading Results

### From Local Machine (Windows PowerShell)

```powershell
# Download all bounds files
scp -rp sons01@greene-dtn.hpc.nyu.edu:/scratch/sons01/sublora-experiments/*/out/*levels*.yml .\bounds_results\

# Download quantized checkpoints (for analysis)
scp -rp sons01@greene-dtn.hpc.nyu.edu:/scratch/sons01/sublora-experiments/*/out/quant_ckpt*.pt .\bounds_results\

# Download everything
scp -rp sons01@greene-dtn.hpc.nyu.edu:/scratch/sons01/sublora-experiments .\
```

### Using rsync (Selective Download)

```bash
# Download only bounds-related files
rsync -avz --progress \
    --include="*/" \
    --include="bounds_*.yml" \
    --include="metrics_*.yml" \
    --include="quant_ckpt_*.pt" \
    --include="best_ckpt.pt" \
    --exclude="*" \
    sons01@greene-dtn.hpc.nyu.edu:/scratch/sons01/sublora-experiments/ \
    ./experiments/
```

---

## Troubleshooting

### Bounds Job Fails with OOM

Reduce batch size in `config/sublora_bounds.yaml`:
```yaml
bounds:
  eval_batch_size: 4  # Reduce from 8 to 4
```

### "No checkpoint found"

Ensure the checkpoint path points to the directory containing `best_ckpt.pt`, not the file itself:
```bash
# Correct
--model.best_checkpoint_path=/scratch/$USER/sublora-experiments/sublora-d1000-uniform-seed42/out

# Wrong
--model.best_checkpoint_path=/scratch/$USER/sublora-experiments/sublora-d1000-uniform-seed42/out/best_ckpt.pt
```

### Missing EOT/Doc Length Files

Generate them from the training data:
```bash
# Inside singularity container
python data/openwebtext/prepare_manual.py \
    --input_dir=/scratch/$USER/sublora-data \
    --output_dir=/scratch/$USER/sublora-data
```

### Job Preempted

Jobs are automatically requeued with `--requeue`. The bounds evaluation will restart from the beginning (no checkpointing for bounds eval).

---

## GPU Hour Budget

| Stage | Per Run | 30 Runs | Notes |
|-------|---------|---------|-------|
| Quantization | ~5 min | ~2.5 hrs | K-means on parameters |
| Document Sampling | ~35 min | ~17.5 hrs | 10k documents |
| Bound Computation | ~5 min | ~2.5 hrs | PAC-Bayes calculation |
| **Total** | **~45 min** | **~22.5 hrs** | |

With 300 GPU hours budget:
- Training: ~135 hours
- Bounds: ~23 hours
- **Remaining**: ~142 hours (buffer for retries/debugging)

---

## Next Steps

After bounds evaluation is complete for all 30 experiments:

1. **Verify all bounds files exist**:
   ```bash
   ls /scratch/$USER/sublora-experiments/*/out/bounds_levels11_iters100.yml | wc -l
   # Should show: 30
   ```

2. **Download results to local machine**

3. **Run analysis** (see [ANALYSIS.md](ANALYSIS.md)):
   ```bash
   python experiments/analyze_results.py \
       --results_dir=./experiments \
       --output_dir=./analysis_outputs
   ```
