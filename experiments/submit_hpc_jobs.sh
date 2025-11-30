#!/bin/bash
# Submit all 30 SubLoRA training jobs to NYU HPC Cloud Bursting
# Each job gets its own output folder based on hyperparameter configuration

ACCOUNT="ds_ga_1006-2025fa"
PARTITION="c12m85-a100-1"  # A100 GPU partition
TIME="08:00:00"            # 8 hours per job (adjust as needed)

# Base directories
WORK_DIR="/scratch/$USER/sublora"
DATA_DIR="$WORK_DIR/data"
OUT_DIR="$WORK_DIR/out/adaptive_experiments"

# Array of configurations: "intrinsic_dim,allocation_mode,allocation_ratio,run_name"
CONFIGS=(
    # d=1000 experiments
    "1000,uniform,0.5,d1000_uniform"
    "1000,fixed,0.8,d1000_fixed_bheavy"
    "1000,fixed,0.5,d1000_fixed_equal"
    "1000,fixed,0.2,d1000_fixed_aheavy"
    "1000,learned,0.5,d1000_learned"
    # d=2000 experiments
    "2000,uniform,0.5,d2000_uniform"
    "2000,fixed,0.8,d2000_fixed_bheavy"
    "2000,fixed,0.5,d2000_fixed_equal"
    "2000,fixed,0.2,d2000_fixed_aheavy"
    "2000,learned,0.5,d2000_learned"
)

SEEDS=(42 123 999)

# Submit jobs for each configuration × seed combination
for config in "${CONFIGS[@]}"; do
    IFS=',' read -r DIM MODE RATIO NAME <<< "$config"
    
    for SEED in "${SEEDS[@]}"; do
        JOB_NAME="${NAME}_seed${SEED}"
        JOB_OUT_DIR="${OUT_DIR}/${JOB_NAME}"
        
        echo "Submitting job: $JOB_NAME"
        
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:1
#SBATCH --time=$TIME
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --output=$WORK_DIR/logs/${JOB_NAME}_%j.out
#SBATCH --error=$WORK_DIR/logs/${JOB_NAME}_%j.err
#SBATCH --requeue

# Create output directories
mkdir -p $JOB_OUT_DIR
mkdir -p $WORK_DIR/logs

# Load singularity and run with conda overlay
singularity exec --nv \\
    --overlay /scratch/$USER/sublora_env.ext3:ro \\
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \\
    /bin/bash -c "
        source /ext3/env.sh
        conda activate sublora
        
        cd $WORK_DIR
        
        python experiments/train.py \\
            --config-file=config/sublora_train.yaml \\
            --data.dataset_dir=$DATA_DIR \\
            --login.out_dir=$JOB_OUT_DIR \\
            --login.wandb_run_name=${NAME}_s${SEED} \\
            --sublora.intrinsic_dim=$DIM \\
            --sublora.allocation_mode=$MODE \\
            --sublora.allocation_ratio=$RATIO \\
            --system.seed=$SEED \\
            --system.compile=True \\
            --training.max_iters=10000
    "
EOF
        
        # Small delay to avoid overwhelming scheduler
        sleep 1
    done
done

echo "Submitted 30 jobs total"
echo "Monitor with: squeue -u \$USER"
