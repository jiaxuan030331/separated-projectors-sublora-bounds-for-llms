# NYU HPC Cloud Bursting Setup Guide

## Quick Reference

| Resource | Value |
|----------|-------|
| Account | `ds_ga_1006-2025fa` |
| GPU Budget | 300 hours (18,000 minutes) |
| Best GPU Partition | `c12m85-a100-1` (A100) |
| Alternative | `g2-standard-12` (L4 GPU) |
| OOD Portal | https://ood-burst-001.hpc.nyu.edu |

---

## Initial Setup (One-Time)

### 1. Access Cloud Bursting

```bash
# From your local machine, SSH to Greene
ssh <NetID>@greene.hpc.nyu.edu

# From Greene, SSH to burst
ssh burst
```

### 2. Create Project Directory Structure

```bash
# Create base directories
# - sublora-repo: contains the code (shared across all experiments)
# - sublora-data: contains training data (shared across all experiments)  
# - sublora-experiments: contains individual experiment folders

mkdir -p /scratch/$USER/sublora-repo
mkdir -p /scratch/$USER/sublora-data
mkdir -p /scratch/$USER/sublora-experiments
```

### 2b. Upload Code from Local Machine

Git is not available on burst nodes. Upload your local repo directly from Windows PowerShell:

```powershell
# From your local Windows machine, upload the entire repo
scp -rp "C:\Users\sunny\OneDrive\Documents\GitHub\separated-projectors-sublora-bounds-for-llms\*" sons01@greene-dtn.hpc.nyu.edu:/scratch/sons01/sublora-repo/
```

**Alternative**: Clone from Greene first, then copy to burst:
```bash
# On Greene (not burst) - git is available there
ssh sons01@greene.hpc.nyu.edu
cd /scratch/sons01
git clone https://github.com/jiaxuan030331/separated-projectors-sublora-bounds-for-llms.git sublora-repo

# Then from burst, copy it over
ssh burst
scp -rp greene-dtn:/scratch/sons01/sublora-repo /scratch/sons01/
```

### Directory Structure Overview

```
/scratch/<NetID>/
├── sublora-repo/                    # Shared code repository
│   ├── experiments/
│   ├── sublora/
│   ├── config/
│   └── ...
├── sublora-data/                    # Shared training data
│   ├── train.bin
│   ├── val.bin
│   ├── eot_indices.npy
│   └── doc_lengths.npy
├── sublora_env.ext3                 # Conda environment overlay
└── sublora-experiments/             # Individual experiment folders
    ├── sublora-d1000-uniform-seed42/
    │   ├── out/
    │   │   └── best_ckpt.pt
    │   ├── logs/
    │   │   ├── <job_id>.out
    │   │   └── <job_id>.err
    │   └── config/
    ├── sublora-d1000-uniform-seed123/
    ├── sublora-d1000-fixed-bheavy-seed42/
    ├── sublora-d1000-learned-seed42/
    ├── sublora-d2000-uniform-seed42/
    └── ... (30 experiment folders total)
```

### 3. Set Up Conda Environment with Singularity Overlay

```bash
# Copy overlay template (15GB should be enough)
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz /scratch/$USER/sublora_env.ext3.gz
gunzip /scratch/$USER/sublora_env.ext3.gz

# Start interactive session for setup
srun --account=ds_ga_1006-2025fa --partition=interactive --time=02:00:00 --pty /bin/bash

# Install conda in overlay (--bind mounts /scratch inside the container)
singularity exec --overlay /scratch/$USER/sublora_env.ext3:rw \
    --bind /scratch:/scratch \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash

# Inside singularity container:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /ext3/miniconda3
rm Miniconda3-latest-Linux-x86_64.sh

# Create activation script
cat > /ext3/env.sh << 'EOF'
#!/bin/bash
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
EOF

source /ext3/env.sh

# Create sublora environment from environment.yml
cd /scratch/$USER/sublora-repo
conda env create -f environment.yml
conda activate sublora

# Install sublora package in editable mode
pip install -e .

# Exit singularity
exit
```

### 4. Transfer Data

```bash
# From burst node, copy OpenWebText data from Greene
scp -rp greene-dtn:/scratch/<NetID>/sublora/data/* /scratch/$USER/sublora-data/

# Or if preparing fresh:
singularity exec --overlay /scratch/$USER/sublora_env.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate sublora
        cd /scratch/$USER/sublora-repo
        python data/openwebtext/prepare.py --output_dir=/scratch/$USER/sublora-data
    "
```

---

## Running Experiments

Each experiment runs in its own folder with naming convention:
```
sublora-d{dim}-{mode}-seed{seed}
```

Examples:
- `sublora-d1000-uniform-seed42`
- `sublora-d1000-fixed-bheavy-seed42`
- `sublora-d2000-learned-seed999`

### Option 1: Submit All 30 Jobs at Once

```bash
cd /scratch/$USER/sublora-repo
chmod +x experiments/submit_hpc_jobs.sh
./experiments/submit_hpc_jobs.sh
```

### Option 2: Submit Individual Jobs

Each job creates its own experiment folder automatically:

```bash
# d=1000, uniform, seed=42 → creates sublora-d1000-uniform-seed42/
sbatch --job-name=sublora-d1000-uniform-seed42 \
       --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=42 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# d=1000, fixed B-heavy, seed=42 → creates sublora-d1000-fixed-bheavy-seed42/
sbatch --job-name=sublora-d1000-fixed-bheavy-seed42 \
       --export=DIM=1000,MODE=fixed,RATIO=0.8,SEED=42 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# d=1000, fixed A-heavy, seed=123 → creates sublora-d1000-fixed-aheavy-seed123/
sbatch --job-name=sublora-d1000-fixed-aheavy-seed123 \
       --export=DIM=1000,MODE=fixed,RATIO=0.2,SEED=123 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# d=1000, learned, seed=42 → creates sublora-d1000-learned-seed42/
sbatch --job-name=sublora-d1000-learned-seed42 \
       --export=DIM=1000,MODE=learned,RATIO=0.5,SEED=42 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# d=2000, uniform, seed=999 → creates sublora-d2000-uniform-seed999/
sbatch --job-name=sublora-d2000-uniform-seed999 \
       --export=DIM=2000,MODE=uniform,RATIO=0.5,SEED=999 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm
```

### Complete List of All 30 Experiment Commands

**Note:** If you get "DOS line breaks" error, first run:
```bash
sed -i 's/\r$//' /scratch/$USER/sublora-repo/experiments/run_single_job.slurm
sed -i 's/\r$//' /scratch/$USER/sublora-repo/experiments/submit_hpc_jobs.sh
```

```bash
# ============== d=1000 Experiments ==============

# Uniform (baseline)
sbatch --job-name=sublora-d1000-uniform-seed42 \
    --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d1000-uniform-seed123 \
    --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d1000-uniform-seed999 \
    --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=999 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed B-heavy (ratio=0.8)
sbatch --job-name=sublora-d1000-fixed-bheavy-seed42 \
    --export=DIM=1000,MODE=fixed,RATIO=0.8,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d1000-fixed-bheavy-seed123 \
    --export=DIM=1000,MODE=fixed,RATIO=0.8,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d1000-fixed-bheavy-seed999 \
    --export=DIM=1000,MODE=fixed,RATIO=0.8,SEED=999 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed Equal (ratio=0.5)
sbatch --job-name=sublora-d1000-fixed-equal-seed42 \
    --export=DIM=1000,MODE=fixed,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d1000-fixed-equal-seed123 \
    --export=DIM=1000,MODE=fixed,RATIO=0.5,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d1000-fixed-equal-seed999 \
    --export=DIM=1000,MODE=fixed,RATIO=0.5,SEED=999 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed A-heavy (ratio=0.2)
sbatch --job-name=sublora-d1000-fixed-aheavy-seed42 \
    --export=DIM=1000,MODE=fixed,RATIO=0.2,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d1000-fixed-aheavy-seed123 \
    --export=DIM=1000,MODE=fixed,RATIO=0.2,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d1000-fixed-aheavy-seed999 \
    --export=DIM=1000,MODE=fixed,RATIO=0.2,SEED=999 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Learned
sbatch --job-name=sublora-d1000-learned-seed42 \
    --export=DIM=1000,MODE=learned,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d1000-learned-seed123 \
    --export=DIM=1000,MODE=learned,RATIO=0.5,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d1000-learned-seed999 \
    --export=DIM=1000,MODE=learned,RATIO=0.5,SEED=999 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# ============== d=2000 Experiments ==============

# Uniform (baseline)
sbatch --job-name=sublora-d2000-uniform-seed42 \
    --export=DIM=2000,MODE=uniform,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d2000-uniform-seed123 \
    --export=DIM=2000,MODE=uniform,RATIO=0.5,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d2000-uniform-seed999 \
    --export=DIM=2000,MODE=uniform,RATIO=0.5,SEED=999 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed B-heavy (ratio=0.8)
sbatch --job-name=sublora-d2000-fixed-bheavy-seed42 \
    --export=DIM=2000,MODE=fixed,RATIO=0.8,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d2000-fixed-bheavy-seed123 \
    --export=DIM=2000,MODE=fixed,RATIO=0.8,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d2000-fixed-bheavy-seed999 \
    --export=DIM=2000,MODE=fixed,RATIO=0.8,SEED=999 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed Equal (ratio=0.5)
sbatch --job-name=sublora-d2000-fixed-equal-seed42 \
    --export=DIM=2000,MODE=fixed,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d2000-fixed-equal-seed123 \
    --export=DIM=2000,MODE=fixed,RATIO=0.5,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d2000-fixed-equal-seed999 \
    --export=DIM=2000,MODE=fixed,RATIO=0.5,SEED=999 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed A-heavy (ratio=0.2)
sbatch --job-name=sublora-d2000-fixed-aheavy-seed42 \
    --export=DIM=2000,MODE=fixed,RATIO=0.2,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d2000-fixed-aheavy-seed123 \
    --export=DIM=2000,MODE=fixed,RATIO=0.2,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d2000-fixed-aheavy-seed999 \
    --export=DIM=2000,MODE=fixed,RATIO=0.2,SEED=999 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Learned
sbatch --job-name=sublora-d2000-learned-seed42 \
    --export=DIM=2000,MODE=learned,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d2000-learned-seed123 \
    --export=DIM=2000,MODE=learned,RATIO=0.5,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d2000-learned-seed999 \
    --export=DIM=2000,MODE=learned,RATIO=0.5,SEED=999 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm
```

### Option 3: Interactive GPU Session

```bash
# Get an A100 for 4 hours
srun --account=ds_ga_1006-2025fa --partition=c12m85-a100-1 --gres=gpu --time=04:00:00 --pty /bin/bash

# Inside the session:
singularity exec --nv --overlay /scratch/$USER/sublora_env.ext3:ro \
    /scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash

source /ext3/env.sh
conda activate sublora
cd /scratch/$USER/sublora-repo

# Run training (output goes to experiment-specific folder)
EXP_DIR=/scratch/$USER/sublora-experiments/sublora-d1000-uniform-seed42
mkdir -p $EXP_DIR/out

python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=/scratch/$USER/sublora-data \
    --login.out_dir=$EXP_DIR/out \
    --login.wandb_run_name=sublora-d1000-uniform-seed42 \
    --sublora.intrinsic_dim=1000 \
    --sublora.allocation_mode=uniform \
    --system.seed=42 \
    --training.max_iters=10000
```

---

## Monitoring Jobs

```bash
# Check your job queue
squeue -u $USER

# Check job details
scontrol show job <JOB_ID>

# View job output in real-time (logs are in each experiment folder)
tail -f /scratch/$USER/sublora-experiments/sublora-d1000-uniform-seed42/logs/<job_id>.out

# List all experiment folders
ls -la /scratch/$USER/sublora-experiments/

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

---

## Checkpointing & Resume

The `--requeue` flag automatically requeues jobs if preempted by GCP. The training script checks for `best_ckpt.pt` and resumes automatically.

To manually resume a job (just resubmit the same command):
```bash
# Resubmit - it auto-detects checkpoint in the experiment folder
sbatch --job-name=sublora-d1000-uniform-seed42 \
       --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=42 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm
```

---

## GPU Hour Budgeting

| Partition | GPU | Est. Time/Run | Hours for 30 Runs |
|-----------|-----|---------------|-------------------|
| c12m85-a100-1 | A100 | ~4.5 hrs | ~135 hrs |
| g2-standard-12 | L4 | ~8 hrs | ~240 hrs |

**Recommendation**: Use A100 partition (`c12m85-a100-1`) for faster training. With 300 GPU hours, you can complete all 30 experiments (~135 hrs) plus bounds evaluation (~23 hrs).

---

## Output Folder Structure

Each experiment gets its own self-contained folder:

```
/scratch/$USER/sublora-experiments/
├── sublora-d1000-uniform-seed42/
│   ├── out/
│   │   ├── best_ckpt.pt
│   │   ├── ckpt_at_random_initialization.pt
│   │   ├── trainable_initparams.pt
│   │   └── names.pt
│   ├── logs/
│   │   ├── 12345678.out
│   │   └── 12345678.err
│   └── config/
│       └── sublora_train.yaml
├── sublora-d1000-uniform-seed123/
├── sublora-d1000-uniform-seed999/
├── sublora-d1000-fixed-bheavy-seed42/
├── sublora-d1000-fixed-bheavy-seed123/
├── sublora-d1000-fixed-bheavy-seed999/
├── sublora-d1000-fixed-equal-seed42/
├── sublora-d1000-fixed-aheavy-seed42/
├── sublora-d1000-learned-seed42/
├── sublora-d2000-uniform-seed42/
├── sublora-d2000-fixed-bheavy-seed42/
├── sublora-d2000-learned-seed999/
└── ... (30 folders total)
```

---

## Downloading Results

### Download Specific Experiment
```bash
# From local machine (Windows PowerShell)
scp -rp <NetID>@greene-dtn.hpc.nyu.edu:/scratch/<NetID>/sublora-experiments/sublora-d1000-uniform-seed42 .\experiments\

# Download just the checkpoint
scp <NetID>@greene-dtn.hpc.nyu.edu:/scratch/<NetID>/sublora-experiments/sublora-d1000-uniform-seed42/out/best_ckpt.pt .\checkpoints\
```

### Download All Experiments
```bash
scp -rp <NetID>@greene-dtn.hpc.nyu.edu:/scratch/<NetID>/sublora-experiments .\
```

### Download Only Checkpoints (rsync)
```bash
rsync -avz --progress \
    --include="*/" \
    --include="best_ckpt.pt" \
    --include="bounds_metrics.pt" \
    --exclude="*" \
    <NetID>@greene-dtn.hpc.nyu.edu:/scratch/<NetID>/sublora-experiments/ \
    ./experiments/
```

---

## Troubleshooting

### Job Preempted
- Jobs are automatically requeued with `--requeue`
- Check logs for "PREEMPT" message
- Training resumes from last checkpoint

### Out of Memory
- A100 has 40GB VRAM - should be sufficient
- If OOM, reduce batch size in config

### CUDA Version Mismatch
- Use matching Singularity image: `/scratch/work/public/singularity/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`
- Ensure PyTorch was installed with `cu121`

### Cannot Connect to wandb
- Pre-login on interactive node: `wandb login`
- Or set `--login.wandb_log=False` to disable
