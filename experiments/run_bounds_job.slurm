#!/bin/bash
#SBATCH --job-name=sublora-bounds
#SBATCH --account=ds_ga_1006-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --requeue

# Note: We don't use #SBATCH --output/--error here because we want logs
# in the same folder as the checkpoint (nested timestamp folder).
# Logs will be redirected manually below.

# ============================================================
# HPC_USER Configuration
# ============================================================
# HPC_USER specifies whose /scratch space to use for:
#   - sublora-repo (code)
#   - sublora-data (dataset)
#   - sublora_env.ext3 (conda environment overlay)
#
# Pass via: --export=CHECKPOINT_DIR=...,HPC_USER=netid
# Defaults to current user if not specified.
# ============================================================
HPC_USER=${HPC_USER:-${USER:-$(whoami)}}

# === Environment Variables (passed via --export) ===
# CHECKPOINT_DIR: Full path to the directory containing best_ckpt.pt
#                 e.g., /scratch/user/sublora-experiments/sublora-d1000-uniform-seed42/out/SubLoRA_Pretrain/id1000_lr0.005_r4/2025-12-02/02-54
#
# Directory structure from training:
#   sublora-experiments/
#   └── sublora-d{dim}-{mode}-seed{seed}/
#       ├── config/
#       ├── logs/
#       └── out/
#           └── SubLoRA_Pretrain/
#               └── id{dim}_lr0.005_r4/
#                   └── {date}/
#                       └── {time}/
#                           ├── best_ckpt.pt          ← Input checkpoint
#                           ├── quant_ckpt_levels*.pt ← Output: quantized checkpoint
#                           ├── bounds_levels*.yml    ← Output: PAC-Bayes bounds
#                           ├── metrics_levels*.yml   ← Output: empirical metrics
#                           ├── bounds_eval.out       ← Output: SLURM stdout log
#                           └── bounds_eval.err       ← Output: SLURM stderr log

# Redirect stdout and stderr to checkpoint directory
exec > "${CHECKPOINT_DIR}/bounds_eval_${SLURM_JOB_ID}.out" 2> "${CHECKPOINT_DIR}/bounds_eval_${SLURM_JOB_ID}.err"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started: $(date)"
echo "Checkpoint Dir: ${CHECKPOINT_DIR}"
echo "=========================================="

# Set paths - uses HPC_USER's scratch space
REPO_DIR=/scratch/${HPC_USER}/sublora-repo
DATA_DIR=/scratch/${HPC_USER}/sublora-data
OVERLAY=/scratch/${HPC_USER}/sublora_env.ext3

# Verify checkpoint exists
if [ ! -f "${CHECKPOINT_DIR}/best_ckpt.pt" ]; then
    echo "ERROR: best_ckpt.pt not found in ${CHECKPOINT_DIR}"
    exit 1
fi

# Set wandb API key if available (from HPC_USER's scratch)
if [ -f /scratch/${HPC_USER}/.wandb_api_key ]; then
    export WANDB_API_KEY=$(cat /scratch/${HPC_USER}/.wandb_api_key)
fi

# Run bounds evaluation
# Note: --model.best_checkpoint_path should point to the directory containing best_ckpt.pt
#       Bounds output files will be written to the same directory
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
            --model.best_checkpoint_path=${CHECKPOINT_DIR} \
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
echo "Output files in: ${CHECKPOINT_DIR}"
echo "=========================================="