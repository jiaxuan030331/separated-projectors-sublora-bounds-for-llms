#!/bin/bash
# Submit bounds evaluation for all completed training runs

# ============================================================
# HPC_USER Configuration
# ============================================================
# HPC_USER specifies whose /scratch space to use for:
#   - sublora-repo (code)
#   - sublora-data (dataset)
#   - sublora_env.ext3 (conda environment overlay)
#   - sublora-experiments (experiments to evaluate)
#
# Usage:
#   ./submit_bounds_jobs.sh                    # Uses current user
#   HPC_USER=netid ./submit_bounds_jobs.sh     # Uses specified user's setup
#   SEEDS="42 123" ./submit_bounds_jobs.sh     # Custom seeds
#   DIMS="10000 20000" ./submit_bounds_jobs.sh # Custom dimensions
# ============================================================
HPC_USER=${HPC_USER:-$USER}

# Seeds for experiments (space-separated, override via environment variable)
SEEDS_STR=${SEEDS:-"42 123"}
read -ra SEEDS <<< "$SEEDS_STR"

# Intrinsic dimensions (space-separated, override via environment variable)
DIMS=${DIMS:-"10000 20000"}
read -ra DIMS_ARR <<< "$DIMS"

LOW_DIM=${DIMS_ARR[0]}
HIGH_DIM=${DIMS_ARR[1]}

ACCOUNT="ds_ga_1006-2025fa"
PARTITION="c12m85-a100-1"
TIME=${TIME:-"02:00:00"}

# Paths - uses HPC_USER's scratch space
REPO_DIR="/scratch/${HPC_USER}/sublora-repo"
EXPERIMENTS_DIR=/scratch/${HPC_USER}/sublora-experiments
SLURM_SCRIPT=/scratch/${HPC_USER}/sublora-repo/experiments/run_bounds_job.slurm

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

num_configs=${#CONFIGS[@]}
num_seeds=${#SEEDS[@]}
total_jobs=$((num_configs * num_seeds))

echo "============================================"
echo "Submitting ${total_jobs} SubLoRA bounds evaluations"
echo "Using HPC_USER: ${HPC_USER}"
echo "Seeds: ${SEEDS[*]}"
echo "Dimensions: ${DIMS}"
echo "============================================"

for exp_dir in ${EXPERIMENTS_DIR}/sublora-d*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        # Extract the number after -d
        d_num=$(echo "${exp_name}" | grep -oP '(?<=-d)\d+')
        # Build the path to the date directories
        # Try both lr0.0005 (current) and lr0.005 (paper default) patterns
        base_path="${exp_dir}/out/SubLoRA_Pretrain/id${d_num}_lr0.0005_r4"
        if [ ! -d "${base_path}" ]; then
            base_path="${exp_dir}/out/SubLoRA_Pretrain/id${d_num}_lr0.005_r4"
        fi
        # Get most recent date folder
        exp_ckpt_date_dir=$(ls -d "${base_path}"/*/ 2>/dev/null | sort -r | head -1)
        # Get most recent time folder within that date
        exp_ckpt_time_dir=$(ls -d "${exp_ckpt_date_dir}"*/ 2>/dev/null | sort -r | head -1)
        # Remove trailing slash for consistent path handling
        exp_ckpt_time_dir="${exp_ckpt_time_dir%/}"
        # Check if training is complete (best_ckpt.pt exists)
        if [ -f "${exp_ckpt_time_dir}/best_ckpt.pt" ]; then
            # Check if bounds already computed (bounds files are in same dir as best_ckpt.pt)
            if [ ! -f "${exp_ckpt_time_dir}/bounds_levels11_iters100.yml" ]; then
                echo "Submitting bounds evaluation for: $exp_name"
                echo "  Checkpoint dir: ${exp_ckpt_time_dir}"
                sbatch --job-name=bounds-${exp_name} \
                       --export=CHECKPOINT_DIR=${exp_ckpt_time_dir},HPC_USER=${HPC_USER} \
                       ${SLURM_SCRIPT}
            else
                echo "Skipping $exp_name (bounds already computed)"
            fi
        else
            echo "Skipping $exp_name (training not complete)"
            echo "  Expected: ${exp_ckpt_time_dir}/best_ckpt.pt"
        fi
    fi
done