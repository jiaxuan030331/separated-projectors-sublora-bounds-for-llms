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

## Prerequisites

Before running bounds evaluation:

1. **Training must be complete**: Ensure `best_ckpt.pt` exists in each experiment folder
2. **Data files required**:
   - `train.bin` - Training data
   - `val.bin` - Validation data  
   - `eot_indices.npy` - End-of-text token positions
   - `doc_lengths.npy` - Document length distribution

---

## SLURM Job Script

Create `/scratch/$USER/sublora-repo/experiments/run_bounds_job.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=sublora-bounds
#SBATCH --account=ds_ga_1006-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/scratch/%u/sublora-experiments/%x/logs/%j.out
#SBATCH --error=/scratch/%u/sublora-experiments/%x/logs/%j.err
#SBATCH --requeue

# === Environment Variables (passed via --export) ===
# CHECKPOINT_DIR: Full path to experiment directory with best_ckpt.pt

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "=========================================="

# Set paths
REPO_DIR=/scratch/$USER/sublora-repo
DATA_DIR=/scratch/$USER/sublora-data
OVERLAY=/scratch/$USER/sublora_env.ext3

# Create logs directory
mkdir -p ${CHECKPOINT_DIR}/logs

# Set wandb API key if available
if [ -f /scratch/$USER/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat /scratch/$USER/.wandb_api_key)
fi

# Run bounds evaluation
singularity exec --nv \
    --overlay ${OVERLAY}:ro \
    --bind /scratch:/scratch \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate sublora
        cd ${REPO_DIR}
        
        python experiments/eval_bounds.py \
            --config-file=config/sublora_bounds.yaml \
            --data.dataset_dir=${DATA_DIR} \
            --model.best_checkpoint_path=${CHECKPOINT_DIR}/out \
            --bounds.bound_type=document_level \
            --bounds.levels=11 \
            --bounds.bound_samples=10000 \
            --bounds.max_quant_iters=100 \
            --bounds.use_kmeans=True \
            --data.openwebtext_train_eot_indices_file=${DATA_DIR}/eot_indices.npy \
            --data.empirical_document_length_distribution_file=${DATA_DIR}/doc_lengths.npy
    "

echo "=========================================="
echo "Completed: $(date)"
echo "=========================================="
```

---

## Submitting Bounds Evaluation Jobs

### Option 1: Submit All 30 Jobs

Create `submit_bounds_jobs.sh`:

```bash
#!/bin/bash
# Submit bounds evaluation for all completed training runs

EXPERIMENTS_DIR=/scratch/$USER/sublora-experiments
SLURM_SCRIPT=/scratch/$USER/sublora-repo/experiments/run_bounds_job.slurm

for exp_dir in ${EXPERIMENTS_DIR}/sublora-d*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename $exp_dir)
        
        # Check if training is complete (best_ckpt.pt exists)
        if [ -f "$exp_dir/out/best_ckpt.pt" ]; then
            # Check if bounds already computed
            if [ ! -f "$exp_dir/out/bounds_levels11_iters100.yml" ]; then
                echo "Submitting bounds evaluation for: $exp_name"
                sbatch --job-name=bounds-${exp_name} \
                       --export=CHECKPOINT_DIR=${exp_dir} \
                       ${SLURM_SCRIPT}
            else
                echo "Skipping $exp_name (bounds already computed)"
            fi
        else
            echo "Skipping $exp_name (training not complete)"
        fi
    fi
done
```

Run with:
```bash
chmod +x submit_bounds_jobs.sh
./submit_bounds_jobs.sh
```

### Option 2: Submit Individual Jobs

```bash
# Example: Submit bounds for d=1000 uniform seed=42
sbatch --job-name=bounds-sublora-d1000-uniform-seed42 \
       --export=CHECKPOINT_DIR=/scratch/$USER/sublora-experiments/sublora-d1000-uniform-seed42 \
       /scratch/$USER/sublora-repo/experiments/run_bounds_job.slurm

# Example: Submit bounds for d=2000 learned seed=999
sbatch --job-name=bounds-sublora-d2000-learned-seed999 \
       --export=CHECKPOINT_DIR=/scratch/$USER/sublora-experiments/sublora-d2000-learned-seed999 \
       /scratch/$USER/sublora-repo/experiments/run_bounds_job.slurm
```

---

## Output Files

After bounds evaluation, each experiment folder will contain:

```
/scratch/$USER/sublora-experiments/sublora-d1000-uniform-seed42/
├── out/
│   ├── best_ckpt.pt                           # From training
│   ├── quant_ckpt_levels11_iters100.pt        # Quantized checkpoint
│   ├── bounds_levels11_iters100.yml           # PAC-Bayes bounds
│   ├── metrics_levels11_iters100.yml          # Empirical metrics
│   ├── ix_levels11_iters100.txt               # Sampled document indices
│   ├── top_k_indices_levels11_iters100.txt    # Top-k predictions
│   ├── selected_prob_scores_levels11_iters100.txt
│   └── percentile_vec_levels11_iters100.txt
└── logs/
    ├── <training_job_id>.out
    ├── <training_job_id>.err
    ├── <bounds_job_id>.out                    # Bounds evaluation log
    └── <bounds_job_id>.err
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

# Watch bounds evaluation progress
tail -f /scratch/$USER/sublora-experiments/sublora-d1000-uniform-seed42/logs/<job_id>.out

# Check which experiments have bounds computed
ls /scratch/$USER/sublora-experiments/*/out/bounds_levels11_iters100.yml | wc -l
# Should show 30 when all complete

# List experiments missing bounds
for d in /scratch/$USER/sublora-experiments/sublora-d*/; do
    if [ ! -f "$d/out/bounds_levels11_iters100.yml" ]; then
        echo "Missing bounds: $(basename $d)"
    fi
done
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
