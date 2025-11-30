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
# Create main project folder
mkdir -p /scratch/$USER/sublora
cd /scratch/$USER/sublora

# Create subdirectories
mkdir -p data out/adaptive_experiments logs config

# Clone your repo (or transfer files)
git clone https://github.com/jiaxuan030331/separated-projectors-sublora-bounds-for-llms.git repo
# Copy necessary files
cp -r repo/experiments repo/sublora repo/config repo/setup.py repo/setup.cfg .
```

### 3. Set Up Conda Environment with Singularity Overlay

```bash
# Copy overlay template (15GB should be enough)
cp /share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz /scratch/$USER/sublora_env.ext3.gz
gunzip /scratch/$USER/sublora_env.ext3.gz

# Start interactive session for setup
srun --account=ds_ga_1006-2025fa --partition=interactive --time=02:00:00 --pty /bin/bash

# Install conda in overlay
singularity exec --overlay /scratch/$USER/sublora_env.ext3:rw \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash

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

# Create sublora environment
conda create -n sublora python=3.10 -y
conda activate sublora

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers datasets tiktoken wandb numpy tqdm pyyaml

# Install sublora package
cd /scratch/$USER/sublora
pip install -e .

# Exit singularity
exit
```

### 4. Transfer Data

```bash
# From burst node, copy OpenWebText data from Greene
scp -rp greene-dtn:/scratch/<NetID>/sublora/data/* /scratch/$USER/sublora/data/

# Or if preparing fresh:
singularity exec --overlay /scratch/$USER/sublora_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate sublora
        cd /scratch/$USER/sublora
        python data/openwebtext/prepare.py
    "
```

---

## Running Experiments

### Option 1: Submit All 30 Jobs at Once

```bash
cd /scratch/$USER/sublora
chmod +x experiments/submit_hpc_jobs.sh
./experiments/submit_hpc_jobs.sh
```

### Option 2: Submit Individual Jobs

```bash
# d=1000, uniform, seed=42
sbatch --job-name=d1000_uniform_seed42 \
       --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=42 \
       experiments/run_single_job.slurm

# d=1000, fixed B-heavy, seed=42
sbatch --job-name=d1000_bheavy_seed42 \
       --export=DIM=1000,MODE=fixed,RATIO=0.8,SEED=42 \
       experiments/run_single_job.slurm

# d=1000, learned, seed=42
sbatch --job-name=d1000_learned_seed42 \
       --export=DIM=1000,MODE=learned,RATIO=0.5,SEED=42 \
       experiments/run_single_job.slurm
```

### Option 3: Interactive GPU Session

```bash
# Get an A100 for 4 hours
srun --account=ds_ga_1006-2025fa --partition=c12m85-a100-1 --gres=gpu --time=04:00:00 --pty /bin/bash

# Inside the session:
singularity exec --nv --overlay /scratch/$USER/sublora_env.ext3:ro \
    /share/apps/images/cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif /bin/bash

source /ext3/env.sh
conda activate sublora
cd /scratch/$USER/sublora

# Run training
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=data \
    --login.out_dir=out/adaptive_experiments/d1000_uniform_seed42 \
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

# View job output in real-time
tail -f /scratch/$USER/sublora/logs/<job_name>_<job_id>.out

# Cancel a job
scancel <JOB_ID>

# Cancel all your jobs
scancel -u $USER
```

---

## Checkpointing & Resume

The `--requeue` flag automatically requeues jobs if preempted by GCP. The training script checks for `best_ckpt.pt` and resumes automatically.

To manually resume a job:
```bash
sbatch --job-name=d1000_uniform_seed42 \
       --export=DIM=1000,MODE=uniform,RATIO=0.5,SEED=42 \
       experiments/run_single_job.slurm
# The script auto-detects the checkpoint and adds --model.init_from=resume
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

After all jobs complete:
```
/scratch/$USER/sublora/out/adaptive_experiments/
├── d1000_uniform_seed42/
│   ├── best_ckpt.pt
│   ├── ckpt_at_random_initialization.pt
│   └── ...
├── d1000_uniform_seed123/
├── d1000_uniform_seed999/
├── d1000_fixed_bheavy_seed42/
├── d1000_fixed_bheavy_seed123/
├── d1000_fixed_bheavy_seed999/
├── d1000_fixed_equal_seed42/
...
├── d2000_learned_seed999/
└── (30 folders total)
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
- Use matching Singularity image: `cuda12.1.1-cudnn8.9.0-devel-ubuntu22.04.2.sif`
- Ensure PyTorch was installed with `cu121`

### Cannot Connect to wandb
- Pre-login on interactive node: `wandb login`
- Or set `--login.wandb_log=False` to disable
