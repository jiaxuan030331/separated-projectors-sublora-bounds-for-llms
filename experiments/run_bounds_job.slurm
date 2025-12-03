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