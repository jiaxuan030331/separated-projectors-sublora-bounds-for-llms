#!/bin/bash
# Submit all 30 SubLoRA training jobs to NYU HPC Cloud Bursting
# Each job gets its own folder: sublora-d{dim}-{mode}-seed{seed}

# ============================================================
# HPC_USER Configuration
# ============================================================
# HPC_USER specifies whose /scratch space to use for:
#   - sublora-repo (code)
#   - sublora-data (dataset) 
#   - sublora_env.ext3 (conda environment overlay)
#   - sublora-experiments (output)
#
# Usage:
#   ./submit_hpc_jobs.sh                              # Uses defaults
#   HPC_USER=netid ./submit_hpc_jobs.sh               # Uses specified user's setup
#   SEEDS="42 123" ./submit_hpc_jobs.sh               # Custom seeds
#   SEEDS="42 123" HPC_USER=netid ./submit_hpc_jobs.sh  # Both custom
# ============================================================
HPC_USER=${HPC_USER:-$USER}

# Seeds for experiments (space-separated, override via environment variable)
# Default: 42 123 999 (3 seeds × 10 configs = 30 experiments)
SEEDS_STR=${SEEDS:-"42 123"}
read -ra SEEDS <<< "$SEEDS_STR"

ACCOUNT="ds_ga_1006-2025fa"
PARTITION="c12m85-a100-1"  # A100 GPU partition
TIME="12:00:00"            # 12 hours per job (adjust as needed)

# Base directories - uses HPC_USER's scratch space
REPO_DIR="/scratch/${HPC_USER}/sublora-repo"
SLURM_SCRIPT="$REPO_DIR/experiments/run_single_job.slurm"


DIMS=${DIMS:-"20000 50000"}
read -ra DIMS_ARR <<< "$DIMS"

LOW_DIM=${DIMS_ARR[0]}
HIGH_DIM=${DIMS_ARR[1]}

# Array of configurations: "intrinsic_dim,allocation_mode,allocation_ratio,mode_name"
CONFIGS=(
    # lower d experiments
    "${LOW_DIM},uniform,0.5,uniform"
    "${LOW_DIM},fixed,0.8,fixed-bheavy"
    "${LOW_DIM},fixed,0.5,fixed-equal"
    "${LOW_DIM},fixed,0.2,fixed-aheavy"
    "${LOW_DIM},learned,0.5,learned"
    # higher d experiments
    "${HIGH_DIM},uniform,0.5,uniform"
    "${HIGH_DIM},fixed,0.8,fixed-bheavy"
    "${HIGH_DIM},fixed,0.5,fixed-equal"
    "${HIGH_DIM},fixed,0.2,fixed-aheavy"
    "${HIGH_DIM},learned,0.5,learned"
)

NUM_CONFIGS=${#CONFIGS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_CONFIGS * NUM_SEEDS))

echo "============================================"
echo "Submitting $TOTAL_JOBS SubLoRA experiments"
echo "  Configs: $NUM_CONFIGS"
echo "  Seeds: ${SEEDS[*]}"
echo "  HPC_USER: $HPC_USER"
echo "============================================"

# Submit jobs for each configuration × seed combination
for config in "${CONFIGS[@]}"; do
    IFS=',' read -r DIM MODE RATIO MODE_NAME <<< "$config"
    
    for SEED in "${SEEDS[@]}"; do
        # Job name format: sublora-d1000-uniform-seed42
        JOB_NAME="sublora-d${DIM}-${MODE_NAME}-seed${SEED}"
        
        echo "Submitting: $JOB_NAME"
        
        sbatch --job-name=$JOB_NAME \
               --export=DIM=$DIM,MODE=$MODE,RATIO=$RATIO,SEED=$SEED,HPC_USER=$HPC_USER \
               $SLURM_SCRIPT
        
        # Small delay to avoid overwhelming scheduler
        sleep 1
    done
done

echo "============================================"
echo "Submitted $TOTAL_JOBS jobs total"
echo "HPC_USER: $HPC_USER"
echo "Seeds: ${SEEDS[*]}"
echo "Monitor with: squeue -u \$USER"
echo "Experiment folders: /scratch/${HPC_USER}/sublora-experiments/"
echo "============================================"
