#!/bin/bash
#SBATCH --job-name=sublora_train
#SBATCH --account=ds_ga_1006-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --open-mode=append
#SBATCH --requeue

# ============================================================
# SubLoRA Training Job Template for NYU HPC Cloud Bursting
# ============================================================
# Each experiment gets its own folder:
#   /scratch/<HPC_USER>/sublora-experiments/sublora-d1000-uniform-seed42/
#
# Usage: 
#   sbatch --job-name=sublora-d1000-uniform-seed42 \
#          --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=42,HPC_USER=netid \
#          run_single_job.slurm
#
# HPC_USER: The NetID whose /scratch space contains the data and environment.
#           Defaults to current user if not specified.
# ============================================================

# ============================================================
# HPC_USER Configuration
# ============================================================
# HPC_USER specifies whose /scratch space to use for:
#   - sublora-repo (code)
#   - sublora-data (dataset)
#   - sublora_env.ext3 (conda environment overlay)
#   - sublora-experiments (output)
#
# This allows running jobs using another user's pre-configured environment.
# Can be set via: --export=HPC_USER=netid or environment variable
# ============================================================
HPC_USER=${HPC_USER:-${USER:-$(whoami)}}

# Default values (override via --export)
DIM=${DIM:-10000}
MODE=${MODE:-uniform}
RATIO=${RATIO:-0.5}
SEED=${SEED:-42}
WANDB_DISABLED=${WANDB_DISABLED:-false}  # Set to 'true' to disable wandb logging

# ============================================================
# Training hyperparameters - matching paper defaults
# ============================================================
MAX_ITERS=${MAX_ITERS:-10000}
# Warmup: 5% of total training iterations
WARMUP_ITERS=$((MAX_ITERS / 20))
# LR decay: over entire training (paper uses lr_decay_iters = max_iters)
LR_DECAY_ITERS=$MAX_ITERS
# Base LR and min LR (min = learning_rate/10 per Chinchilla, paper default)
BASE_LR="5e-4"
MIN_LR="5e-5"

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
# Uses HPC_USER's scratch space for all paths
BASE_DIR="/scratch/${HPC_USER}/sublora-experiments"
EXP_DIR="$BASE_DIR/$EXP_NAME"
DATA_DIR="/scratch/${HPC_USER}/sublora-data"  # Shared data directory
REPO_DIR="/scratch/${HPC_USER}/sublora-repo"  # Code repository
OVERLAY="/scratch/${HPC_USER}/sublora_env.ext3"  # Conda environment overlay
OUT_DIR="$EXP_DIR/out"
LOG_DIR="$EXP_DIR/logs"

# Create experiment directory structure
mkdir -p $EXP_DIR/{out,logs,config}
mkdir -p $DATA_DIR
mkdir -p $LOG_DIR

# Redirect SLURM output to experiment's log directory
exec > "${LOG_DIR}/${SLURM_JOB_ID}.out" 2> "${LOG_DIR}/${SLURM_JOB_ID}.err"

# Copy config files to experiment folder (first run only)
if [ ! -f "$EXP_DIR/config/sublora_train.yaml" ]; then
    cp ${REPO_DIR}/config/*.yaml $EXP_DIR/config/ 2>/dev/null || true
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
    --overlay ${OVERLAY}:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        source /ext3/miniconda3/etc/profile.d/conda.sh
        conda activate sublora
        
        cd ${REPO_DIR}
        
        # Ensure output directory exists and is writable
        mkdir -p $OUT_DIR
        
        # ============================================================
        # WANDB AUTHENTICATION
        # ============================================================
        # Option 1: Set your API key here (not recommended for shared scripts)
        # export WANDB_API_KEY='your-api-key-here'
        
        # Option 2: Read from a secure file (recommended)
        # Create this file once: echo 'your-api-key' > /scratch/<HPC_USER>/.wandb_api_key && chmod 600 /scratch/<HPC_USER>/.wandb_api_key
        if [ -f /scratch/${HPC_USER}/.wandb_api_key ]; then
            export WANDB_API_KEY=\$(cat /scratch/${HPC_USER}/.wandb_api_key)
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
            --system.compile=False \
            --data.batch_size=12 \
            --training.gradient_accumulation_steps=5 \
            --training.eval_interval=100 \
            --training.log_interval=10 \
            --training.max_iters=$MAX_ITERS \
            --optimizer.learning_rate=$BASE_LR \
            --learning_rate.warmup_iters=$WARMUP_ITERS \
            --learning_rate.lr_decay_iters=$LR_DECAY_ITERS \
            --learning_rate.min_lr=$MIN_LR \
            \$WANDB_FLAG \
            $RESUME_FLAG
    "

EXIT_CODE=$?
echo "============================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
