#!/bin/bash
#SBATCH --job-name=sublora_train
#SBATCH --account=ds_ga_1006-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=/scratch/%u/sublora-logs/%x_%j.out
#SBATCH --error=/scratch/%u/sublora-logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

# ============================================================
# SubLoRA Training Job Template for NYU HPC Cloud Bursting
# ============================================================
# Each experiment gets its own folder:
#   /scratch/<NetID>/sublora-experiments/sublora-d1000-uniform-seed42/
#
# Usage: 
#   sbatch --job-name=sublora-d1000-uniform-seed42 \
#          --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=42 \
#          run_single_job.slurm
# ============================================================

# Ensure USER is set (sometimes not available on compute nodes)
USER=${USER:-$(whoami)}

# Default values (override via --export)
DIM=${DIM:-1000}
MODE=${MODE:-uniform}
RATIO=${RATIO:-0.5}
SEED=${SEED:-42}
WANDB_DISABLED=${WANDB_DISABLED:-false}  # Set to 'true' to disable wandb logging

# Build experiment name based on mode
if [[ "$MODE" == "fixed" ]]; then
    if [[ "$RATIO" == "0.8" ]]; then
        MODE_NAME="fixed-bheavy"
    elif [[ "$RATIO" == "0.5" ]]; then
        MODE_NAME="fixed-equal"
    elif [[ "$RATIO" == "0.2" ]]; then
        MODE_NAME="fixed-aheavy"
    else
        MODE_NAME="fixed-r${RATIO}"
    fi
else
    MODE_NAME="$MODE"
fi

# Experiment folder name: sublora-d1000-uniform-seed42
EXP_NAME="sublora-d${DIM}-${MODE_NAME}-seed${SEED}"

# Directories - each experiment gets its own folder
BASE_DIR="/scratch/$USER/sublora-experiments"
EXP_DIR="$BASE_DIR/$EXP_NAME"
DATA_DIR="/scratch/$USER/sublora-data"  # Shared data directory
OUT_DIR="$EXP_DIR/out"

# Create experiment directory structure
mkdir -p $EXP_DIR/{out,logs,config}
mkdir -p $DATA_DIR

# Copy config files to experiment folder (first run only)
if [ ! -f "$EXP_DIR/config/sublora_train.yaml" ]; then
    cp /scratch/$USER/sublora-repo/config/*.yaml $EXP_DIR/config/ 2>/dev/null || true
fi

echo "============================================"
echo "Experiment: $EXP_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Config: dim=$DIM, mode=$MODE, ratio=$RATIO, seed=$SEED"
echo "Experiment Dir: $EXP_DIR"
echo "Output Dir: $OUT_DIR"
echo "Start time: $(date)"
echo "============================================"

# Check for checkpoint to resume from
RESUME_FLAG=""
if [ -f "$OUT_DIR/best_ckpt.pt" ]; then
    echo "Found existing checkpoint, will resume training..."
    RESUME_FLAG="--model.init_from=resume"
fi

# Run training with Singularity + Conda
# Note: /scratch is auto-mounted on burst nodes, no --bind needed
singularity exec --nv \
    --overlay /scratch/$USER/sublora_env.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        source /ext3/miniconda3/etc/profile.d/conda.sh
        conda activate sublora
        
        cd /scratch/$USER/sublora-repo
        
        # Ensure output directory exists and is writable
        mkdir -p $OUT_DIR
        
        # ============================================================
        # WANDB AUTHENTICATION
        # ============================================================
        # Option 1: Set your API key here (not recommended for shared scripts)
        # export WANDB_API_KEY='your-api-key-here'
        
        # Option 2: Read from a secure file (recommended)
        # Create this file once: echo 'your-api-key' > /scratch/\$USER/.wandb_api_key && chmod 600 /scratch/\$USER/.wandb_api_key
        if [ -f /scratch/$USER/.wandb_api_key ]; then
            export WANDB_API_KEY=\$(cat /scratch/$USER/.wandb_api_key)
        fi
        
        # Set wandb directories to writable locations (inside experiment's out folder)
        export WANDB_DIR=$OUT_DIR
        export WANDB_CACHE_DIR=$OUT_DIR/.wandb_cache
        export WANDB_CONFIG_DIR=$OUT_DIR/.wandb_config
        export WANDB_DATA_DIR=$OUT_DIR/.wandb_data
        export WANDB_SILENT=true
        mkdir -p $OUT_DIR/.wandb_cache $OUT_DIR/.wandb_config $OUT_DIR/.wandb_data
        
        # Optionally disable wandb
        if [ \"$WANDB_DISABLED\" = \"true\" ]; then
            export WANDB_MODE=disabled
            WANDB_FLAG=\"--login.wandb_log=False\"
        else
            WANDB_FLAG=\"\"
        fi
        
        python experiments/train.py \
            --config-file=config/sublora_train.yaml \
            --data.dataset_dir=$DATA_DIR \
            --login.out_dir=$OUT_DIR \
            --login.wandb_run_name=$EXP_NAME \
            --sublora.intrinsic_dim=$DIM \
            --sublora.allocation_mode=$MODE \
            --sublora.allocation_ratio=$RATIO \
            --system.seed=$SEED \
            --system.compile=True \
            --training.max_iters=10000 \
            \$WANDB_FLAG \
            $RESUME_FLAG
    "

EXIT_CODE=$?
echo "============================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
