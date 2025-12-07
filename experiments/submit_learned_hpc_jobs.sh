#!/bin/bash
# Submit SubLoRA learned-mode training jobs to NYU HPC Cloud Bursting
# Each job gets its own folder: sublora-d{dim}-learned-seed{seed}

HPC_USER=${HPC_USER:-$USER}

# Seeds for experiments (space-separated, override via environment variable)
SEEDS_STR=${SEEDS:-"42 123"}
read -ra SEEDS <<< "$SEEDS_STR"

ACCOUNT="ds_ga_1006-2025fa"
PARTITION="c12m85-a100-1"
TIME="12:00:00"

REPO_DIR="/scratch/${HPC_USER}/sublora-repo"
SLURM_SCRIPT="$REPO_DIR/experiments/run_single_learned_job.slurm"

DIMS=${DIMS:-"10000 20000"}
read -ra DIMS_ARR <<< "$DIMS"

LOW_DIM=${DIMS_ARR[0]}
HIGH_DIM=${DIMS_ARR[1]}

# Only include learned-mode configs
CONFIGS=(
    "${LOW_DIM},learned,0.5,learned"
    "${HIGH_DIM},learned,0.5,learned"
)

NUM_CONFIGS=${#CONFIGS[@]}
NUM_SEEDS=${#SEEDS[@]}
TOTAL_JOBS=$((NUM_CONFIGS * NUM_SEEDS))

echo "============================================"
echo "Submitting $TOTAL_JOBS learned SubLoRA experiments"
echo "  Configs: $NUM_CONFIGS"
echo "  Seeds: ${SEEDS[*]}"
echo "  HPC_USER: $HPC_USER"
echo "============================================"

for config in "${CONFIGS[@]}"; do
    IFS=',' read -r DIM MODE RATIO MODE_NAME <<< "$config"
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="sublora-d${DIM}-${MODE_NAME}-seed${SEED}"
        echo "Submitting: $JOB_NAME"
        sbatch --job-name=$JOB_NAME \
               --export=DIM=$DIM,MODE=$MODE,RATIO=$RATIO,SEED=$SEED,HPC_USER=$HPC_USER \
               $SLURM_SCRIPT
        sleep 1
    done
done

echo "============================================"
echo "Submitted $TOTAL_JOBS jobs total"
echo "Experiment folders: /scratch/${HPC_USER}/sublora-experiments/"
echo "============================================"
