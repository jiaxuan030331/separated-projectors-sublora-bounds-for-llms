#!/bin/bash
# Submit bounds evaluation for all completed training runs

EXPERIMENTS_DIR=/scratch/$USER/sublora-experiments
SLURM_SCRIPT=/scratch/$USER/sublora-repo/experiments/run_bounds_job.slurm

for exp_dir in ${EXPERIMENTS_DIR}/sublora-d*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename $exp_dir)
        
        # Check if training is complete (best_ckpt.pt exists)
        if [ -f "$exp_dir/out/best_ckpt.pt" ]; then
            # Check if bounds already computed
            if [ ! -f "$exp_dir/out/bounds_levels11_iters100.yml" ]; then
                echo "Submitting bounds evaluation for: $exp_name"
                sbatch --job-name=bounds-${exp_name} \
                       --export=CHECKPOINT_DIR=${exp_dir} \
                       ${SLURM_SCRIPT}
            else
                echo "Skipping $exp_name (bounds already computed)"
            fi
        else
            echo "Skipping $exp_name (training not complete)"
        fi
    fi
done