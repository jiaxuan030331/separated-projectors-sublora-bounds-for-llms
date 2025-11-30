#!/bin/bash
#SBATCH --job-name=sublora_train
#SBATCH --account=ds_ga_1006-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=/scratch/%u/sublora/logs/%x_%j.out
#SBATCH --error=/scratch/%u/sublora/logs/%x_%j.err
#SBATCH --requeue

# ============================================================
# SubLoRA Training Job Template for NYU HPC Cloud Bursting
# ============================================================
# Usage: 
#   sbatch --job-name=d1000_uniform_seed42 \
#          --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=42 \
#          run_single_job.slurm
# ============================================================

# Default values (override via --export)
DIM=${DIM:-1000}
MODE=${MODE:-uniform}
RATIO=${RATIO:-0.5}
SEED=${SEED:-42}

# Directories
WORK_DIR="/scratch/$USER/sublora"
DATA_DIR="$WORK_DIR/data"
RUN_NAME="d${DIM}_${MODE}"
[[ "$MODE" == "fixed" && "$RATIO" == "0.8" ]] && RUN_NAME="d${DIM}_fixed_bheavy"
[[ "$MODE" == "fixed" && "$RATIO" == "0.5" ]] && RUN_NAME="d${DIM}_fixed_equal"
[[ "$MODE" == "fixed" && "$RATIO" == "0.2" ]] && RUN_NAME="d${DIM}_fixed_aheavy"
OUT_DIR="$WORK_DIR/out/adaptive_experiments/${RUN_NAME}_seed${SEED}"

# Create directories
mkdir -p $OUT_DIR
mkdir -p $WORK_DIR/logs

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Config: dim=$DIM, mode=$MODE, ratio=$RATIO, seed=$SEED"
echo "Output: $OUT_DIR"
echo "Start time: $(date)"
echo "============================================"

# Check for checkpoint to resume from
RESUME_FLAG=""
if [ -f "$OUT_DIR/best_ckpt.pt" ]; then
    echo "Found existing checkpoint, will resume training..."
    RESUME_FLAG="--model.init_from=resume"
fi

# Run training with Singularity + Conda
singularity exec --nv \
    --overlay /scratch/$USER/sublora_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate sublora
        
        cd $WORK_DIR
        
        python experiments/train.py \
            --config-file=config/sublora_train.yaml \
            --data.dataset_dir=$DATA_DIR \
            --login.out_dir=$OUT_DIR \
            --login.wandb_run_name=${RUN_NAME}_s${SEED} \
            --sublora.intrinsic_dim=$DIM \
            --sublora.allocation_mode=$MODE \
            --sublora.allocation_ratio=$RATIO \
            --system.seed=$SEED \
            --system.compile=True \
            --training.max_iters=10000 \
            $RESUME_FLAG
    "

EXIT_CODE=$?
echo "============================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
