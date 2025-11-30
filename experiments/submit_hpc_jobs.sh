#!/bin/bash
# Submit all 30 SubLoRA training jobs to NYU HPC Cloud Bursting
# Each job gets its own folder: sublora-d{dim}-{mode}-seed{seed}

ACCOUNT="ds_ga_1006-2025fa"
PARTITION="c12m85-a100-1"  # A100 GPU partition
TIME="08:00:00"            # 8 hours per job (adjust as needed)

# Base directories
REPO_DIR="/scratch/$USER/sublora-repo"
SLURM_SCRIPT="$REPO_DIR/experiments/run_single_job.slurm"

# Array of configurations: "intrinsic_dim,allocation_mode,allocation_ratio,mode_name"
CONFIGS=(
    # d=1000 experiments
    "1000,uniform,0.5,uniform"
    "1000,fixed,0.8,fixed-bheavy"
    "1000,fixed,0.5,fixed-equal"
    "1000,fixed,0.2,fixed-aheavy"
    "1000,learned,0.5,learned"
    # d=2000 experiments
    "2000,uniform,0.5,uniform"
    "2000,fixed,0.8,fixed-bheavy"
    "2000,fixed,0.5,fixed-equal"
    "2000,fixed,0.2,fixed-aheavy"
    "2000,learned,0.5,learned"
)

SEEDS=(42 123 999)

echo "============================================"
echo "Submitting 30 SubLoRA experiments"
echo "============================================"

# Submit jobs for each configuration × seed combination
for config in "${CONFIGS[@]}"; do
    IFS=',' read -r DIM MODE RATIO MODE_NAME <<< "$config"
    
    for SEED in "${SEEDS[@]}"; do
        # Job name format: sublora-d1000-uniform-seed42
        JOB_NAME="sublora-d${DIM}-${MODE_NAME}-seed${SEED}"
        
        echo "Submitting: $JOB_NAME"
        
        sbatch --job-name=$JOB_NAME \
               --export=DIM=$DIM,MODE=$MODE,RATIO=$RATIO,SEED=$SEED \
               $SLURM_SCRIPT
        
        # Small delay to avoid overwhelming scheduler
        sleep 1
    done
done

echo "============================================"
echo "Submitted 30 jobs total"
echo "Monitor with: squeue -u \$USER"
echo "Experiment folders: /scratch/\$USER/sublora-experiments/"
echo "============================================"
