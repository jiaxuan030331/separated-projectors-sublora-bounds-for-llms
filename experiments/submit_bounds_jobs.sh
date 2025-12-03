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
# ============================================================
HPC_USER=${HPC_USER:-$USER}

ACCOUNT="ds_ga_1006-2025fa"
PARTITION="c12m85-a100-1"
TIME="01:00:00"

# Paths - uses HPC_USER's scratch space
REPO_DIR="/scratch/${HPC_USER}/sublora-repo"
EXPERIMENTS_DIR=/scratch/${HPC_USER}/sublora-experiments
SLURM_SCRIPT=/scratch/${HPC_USER}/sublora-repo/experiments/run_bounds_job.slurm

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

SEEDS=(42,123,999)

echo "============================================"
echo "Submitting 30 SubLoRA evaluations"
echo "============================================"

for exp_dir in ${EXPERIMENTS_DIR}/sublora-d*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        # Extract the number after -d
        d_num=$(echo "${exp_name}" | grep -oP '(?<=-d)\d+')
        # Build the path to the date directories
        base_path="${exp_dir}/out/SubLoRA_Pretrain/id${d_num}_lr0.005_r4"
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