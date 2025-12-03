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

## ⚠️ IMPORTANT: Understanding the Filesystem Architecture

NYU HPC has **TWO separate systems** with **separate `/scratch/` storage**:

### Greene vs Burst Architecture

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        GREENE (NYU On-Premise)                                ║
║                                                                               ║
║   ┌────────────────┐         ┌────────────────┐                               ║
║   │  Login Nodes   │         │   log-burst    │                               ║
║   │ (greene.hpc)   │────────>│  (ssh burst)   │                               ║
║   └────────────────┘         └───────┬────────┘                               ║
║          │                           │                                        ║
║          │                           │              ┌───────────────────────┐ ║
║          │                           │              │   GREENE /scratch/    │ ║
║          └───────────────────────────┼─────────────>│   (On-Premise NFS)    │ ║
║                                      │              │                       │ ║
║                          Both see ───┘              └───────────────────────┘ ║
║                          this storage                         ▲               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
                                       │
                                       │ sbatch submits jobs
                                       │ to Google Cloud
                                       ▼
╔═══════════════════════════════════════════════════════════════════════════════╗
║                        BURST (Google Cloud)                                   ║
║                                                                               ║
║   ┌────────────────┐         ┌────────────────┐                               ║
║   │   OOD Burst    │         │ Compute Nodes  │                               ║
║   │ (ood-burst-    │         │  (your jobs)   │                               ║
║   │  001.hpc.nyu)  │         └───────┬────────┘                               ║
║   └───────┬────────┘                 │                                        ║
║           │                          │              ┌───────────────────────┐ ║
║           │                          │              │   BURST /scratch/     │ ║
║           └──────────────────────────┼─────────────>│   (Google Cloud)      │ ║
║                                      │              │      SEPARATE!        │ ║
║                          Both see ───┘              └───────────────────────┘ ║
║                          THIS storage                                         ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

### Which `/scratch/` do you see?

| Access Method | Which `/scratch/` |
|---------------|-------------------|
| `ssh greene.hpc.nyu.edu` | Greene's `/scratch/` |
| `ssh greene` → `ssh burst` (terminal) | **Greene's `/scratch/`** ⚠️ |
| OOD Greene (`ood.hpc.nyu.edu`, `ood-4.hpc...`) | Greene's `/scratch/` |
| **OOD Burst** (`ood-burst-001.hpc.nyu.edu`) | **Burst's `/scratch/`** ✅ |
| **Burst compute nodes** (running jobs) | **Burst's `/scratch/`** ✅ |

### Key Implications

1. **`ssh burst` is misleading**: Even after running `ssh burst` from Greene, you're on a gateway node that still sees Greene's filesystem. You are NOT on burst's cloud storage yet.

2. **Jobs write to Burst's `/scratch/`**: When your SLURM jobs run, they execute on Google Cloud compute nodes and write to burst's cloud-based `/scratch/`.

3. **To see job output files**: Use **OOD Burst** (`ood-burst-001.hpc.nyu.edu`) or start an interactive job:
   ```bash
   # From log-burst (after ssh burst)
   srun --account=ds_ga_1006-2025fa --partition=interactive --time=01:00:00 --pty /bin/bash
   # Now you're on an actual cloud node and can see burst's /scratch/
   ```

4. **File transfers between systems**: Use `scp` with `greene-dtn`:
   ```bash
   # From burst compute node or OOD Burst terminal
   scp greene-dtn:/scratch/$USER/file.txt /scratch/$USER/
   ```

5. **Recommended workflow**: Use **OOD Burst** (`ood-burst-001.hpc.nyu.edu`) for all burst work - file editing, job submission, and viewing output. This avoids confusion between the two filesystems.

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

### 2a. Using Another User's Setup (HPC_USER Configuration)

The scripts support running jobs using a different user's pre-configured environment via the `HPC_USER` variable. This is useful when:
- A team member has already set up the environment/data
- You want to share resources across multiple users
- You're debugging another user's experiments

**What HPC_USER controls:**
```
/scratch/${HPC_USER}/
├── sublora-repo/           # Code repository
├── sublora-data/           # Training data (train.bin, val.bin, etc.)
├── sublora_env.ext3        # Conda environment overlay
├── sublora-experiments/    # Experiment outputs
└── .wandb_api_key          # Optional: WandB API key
```

**Usage:**
```bash
# Method 1: Set environment variable before running submit script
HPC_USER=sons01 ./experiments/submit_hpc_jobs.sh

# Method 2: Export for session
export HPC_USER=sons01
./experiments/submit_hpc_jobs.sh
./experiments/submit_bounds_jobs.sh

# Method 3: Pass directly via sbatch --export
sbatch --job-name=test --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=42,HPC_USER=sons01 \
       experiments/run_single_job.slurm
```

**Note:** You must have read access to the other user's `/scratch/` directory. Contact the user to set permissions:
```bash
# Run as the owner (sons01) to grant access
chmod -R o+rX /scratch/sons01/sublora-repo
chmod -R o+rX /scratch/sons01/sublora-data
chmod o+r /scratch/sons01/sublora_env.ext3
chmod -R o+rwX /scratch/sons01/sublora-experiments  # Write access for outputs
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
    ├── sublora-d10000-uniform-seed42/
    │   ├── out/
    │   │   └── best_ckpt.pt
    │   ├── logs/
    │   │   ├── <job_id>.out
    │   │   └── <job_id>.err
    │   └── config/
    ├── sublora-d10000-uniform-seed123/
    ├── sublora-d10000-fixed-bheavy-seed42/
    ├── sublora-d10000-learned-seed42/
    ├── sublora-d20000-uniform-seed42/
    └── ... (experiment folders by config)
```

### 3. Set Up Conda Environment with Singularity Overlay

```bash
# Copy overlay template (15GB should be enough)
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz /scratch/$USER/sublora_env.ext3.gz
gunzip /scratch/$USER/sublora_env.ext3.gz

# Start interactive session for setup (IMPORTANT)
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

### 5. Set Up Weights & Biases (wandb) Authentication

To enable wandb logging, you need to store your API key securely:

```bash
# Get your API key from https://wandb.ai/authorize
# Then create a secure key file on burst:
echo 'your-wandb-api-key-here' > /scratch/$USER/.wandb_api_key
chmod 600 /scratch/$USER/.wandb_api_key
```

The SLURM script will automatically read this file and set `WANDB_API_KEY`.

**Alternative: Disable wandb** if you don't need logging:
```bash
# Add WANDB_DISABLED=true to your sbatch --export
sbatch --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=42,WANDB_DISABLED=true ...
```

---

## Running Experiments

Each experiment runs in its own folder with naming convention:
```
sublora-d{dim}-{mode}-seed{seed}
```

Examples:
- `sublora-d10000-uniform-seed42`
- `sublora-d10000-fixed-bheavy-seed42`
- `sublora-d20000-learned-seed42`

### Environment Variables

The **submit script** (`submit_hpc_jobs.sh`) supports these environment variables:
- `HPC_USER` - NetID whose `/scratch` space to use (default: `$USER`)
- `SEEDS` - Space-separated seeds (default: `"42 123"`)
- `DIMS` - Space-separated intrinsic dimensions (default: `"10000 20000"`)

The **job script** (`run_single_job.slurm`) supports these environment variables:
- `DIM` - Intrinsic dimension (e.g., 10000, 20000, 50000)
- `MODE` - Allocation mode (uniform, fixed, learned)
- `RATIO` - Allocation ratio for fixed mode (0.2, 0.5, 0.8)
- `SEED` - Random seed (e.g., 42, 123, 999)
- `HPC_USER` - NetID whose `/scratch` space to use (default: `$USER`)
- `WANDB_DISABLED` - Set to `true` to disable wandb logging (default: `false`)

### Option 1: Submit Jobs with Script

```bash
cd /scratch/$USER/sublora-repo
chmod +x experiments/submit_hpc_jobs.sh

# Default: DIMS="10000 20000", SEEDS="42 123" (20 jobs)
./experiments/submit_hpc_jobs.sh

# Custom seeds (single seed = 10 jobs)
SEEDS="42" ./experiments/submit_hpc_jobs.sh

# Custom dimensions
DIMS="5000 10000" ./experiments/submit_hpc_jobs.sh

# Use another user's setup
HPC_USER=sons01 ./experiments/submit_hpc_jobs.sh

# All custom
DIMS="10000" SEEDS="42 123 999" HPC_USER=sons01 ./experiments/submit_hpc_jobs.sh
```

### Option 2: Submit Individual Jobs

Each job creates its own experiment folder automatically:

```bash
# d=10000, uniform, seed=42 → creates sublora-d10000-uniform-seed42/
sbatch --job-name=sublora-d10000-uniform-seed42 \
       --export=DIM=10000,MODE=uniform,RATIO=0.5,SEED=42 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# d=10000, fixed B-heavy, seed=42 → creates sublora-d10000-fixed-bheavy-seed42/
sbatch --job-name=sublora-d10000-fixed-bheavy-seed42 \
       --export=DIM=10000,MODE=fixed,RATIO=0.8,SEED=42 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Using another user's environment
sbatch --job-name=sublora-d10000-uniform-seed42 \
       --export=DIM=10000,MODE=uniform,RATIO=0.5,SEED=42,HPC_USER=sons01 \
       /scratch/sons01/sublora-repo/experiments/run_single_job.slurm

# d=10000, fixed A-heavy, seed=123 → creates sublora-d10000-fixed-aheavy-seed123/
sbatch --job-name=sublora-d10000-fixed-aheavy-seed123 \
       --export=DIM=10000,MODE=fixed,RATIO=0.2,SEED=123 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# d=20000, learned, seed=42 → creates sublora-d20000-learned-seed42/
sbatch --job-name=sublora-d20000-learned-seed42 \
       --export=DIM=20000,MODE=learned,RATIO=0.5,SEED=42 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# d=20000, uniform, seed=999 → creates sublora-d20000-uniform-seed999/
sbatch --job-name=sublora-d20000-uniform-seed999 \
       --export=DIM=20000,MODE=uniform,RATIO=0.5,SEED=999 \
       /scratch/$USER/sublora-repo/experiments/run_single_job.slurm
```

### Example Commands for d=10000 and d=20000

**Note:** If you get "DOS line breaks" error, first run:
```bash
sed -i 's/\r$//' /scratch/$USER/sublora-repo/experiments/run_single_job.slurm
sed -i 's/\r$//' /scratch/$USER/sublora-repo/experiments/submit_hpc_jobs.sh
```

```bash
# ============== d=10000 Experiments ==============

# Uniform (baseline)
sbatch --job-name=sublora-d10000-uniform-seed42 \
    --export=DIM=10000,MODE=uniform,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

sbatch --job-name=sublora-d10000-uniform-seed123 \
    --export=DIM=10000,MODE=uniform,RATIO=0.5,SEED=123 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed B-heavy (ratio=0.8)
sbatch --job-name=sublora-d10000-fixed-bheavy-seed42 \
    --export=DIM=10000,MODE=fixed,RATIO=0.8,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed Equal (ratio=0.5)
sbatch --job-name=sublora-d10000-fixed-equal-seed42 \
    --export=DIM=10000,MODE=fixed,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed A-heavy (ratio=0.2)
sbatch --job-name=sublora-d10000-fixed-aheavy-seed42 \
    --export=DIM=10000,MODE=fixed,RATIO=0.2,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Learned
sbatch --job-name=sublora-d10000-learned-seed42 \
    --export=DIM=10000,MODE=learned,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# ============== d=20000 Experiments ==============

# Uniform (baseline)
sbatch --job-name=sublora-d20000-uniform-seed42 \
    --export=DIM=20000,MODE=uniform,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed B-heavy (ratio=0.8)
sbatch --job-name=sublora-d20000-fixed-bheavy-seed42 \
    --export=DIM=20000,MODE=fixed,RATIO=0.8,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed Equal (ratio=0.5)
sbatch --job-name=sublora-d20000-fixed-equal-seed42 \
    --export=DIM=20000,MODE=fixed,RATIO=0.5,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Fixed A-heavy (ratio=0.2)
sbatch --job-name=sublora-d20000-fixed-aheavy-seed42 \
    --export=DIM=20000,MODE=fixed,RATIO=0.2,SEED=42 \
    /scratch/$USER/sublora-repo/experiments/run_single_job.slurm

# Learned
sbatch --job-name=sublora-d20000-learned-seed42 \
    --export=DIM=20000,MODE=learned,RATIO=0.5,SEED=42 \
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
scp <NetID>@greene-dtn.hpc.nyu.edu:/scratch/<NetID>/sublora-experiments/sublora-d10000-uniform-seed42/out/best_ckpt.pt .\checkpoints\
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
