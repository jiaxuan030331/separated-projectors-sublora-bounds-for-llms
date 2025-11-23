# CLI Commands for Adaptive Subspace Allocation Experiments

## Environment Setup

First, activate your conda environment:

```bash
conda activate sublora
```

---

## Quick Start: Run All Experiments

### Option 1: Using the Batch Script (Windows)

```bash
# Edit paths in run_adaptive_experiments.bat first, then:
experiments\run_adaptive_experiments.bat
```

### Option 2: Using PowerShell (Windows)

```powershell
# Set your paths
$DATA_DIR = "C:\path\to\data\openwebtext"
$OUT_DIR = "C:\path\to\output\adaptive_experiments"

# Run all 10 configurations (see individual commands below)
```

---

## Individual Experiment Commands

### Budget d=1000 (5 Configurations)

#### 1. Baseline (Uniform Allocation)
```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_uniform ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=uniform
```

#### 2. Fixed B-heavy (d_B = 0.8d, d_A = 0.2d)
```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_b_heavy ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=fixed ^
    --sublora.allocation_ratio=0.8
```

#### 3. Fixed Equal (d_B = 0.5d, d_A = 0.5d)
```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_equal ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=fixed ^
    --sublora.allocation_ratio=0.5
```

#### 4. Fixed A-heavy (d_B = 0.2d, d_A = 0.8d)
```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_a_heavy ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=fixed ^
    --sublora.allocation_ratio=0.2
```

#### 5. Learned Gating (Adaptive Per-Layer)
```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_learned ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=learned
```

---

### Budget d=2000 (5 Configurations)

#### 6. Baseline (Uniform Allocation)
```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d2000_uniform ^
    --sublora.intrinsic_dim=2000 ^
    --sublora.allocation_mode=uniform
```

#### 7. Fixed B-heavy (d_B = 0.8d, d_A = 0.2d)
```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d2000_b_heavy ^
    --sublora.intrinsic_dim=2000 ^
    --sublora.allocation_mode=fixed ^
    --sublora.allocation_ratio=0.8
```

#### 8. Fixed Equal (d_B = 0.5d, d_A = 0.5d)
```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d2000_equal ^
    --sublora.intrinsic_dim=2000 ^
    --sublora.allocation_mode=fixed ^
    --sublora.allocation_ratio=0.5
```

#### 9. Fixed A-heavy (d_B = 0.2d, d_A = 0.8d)
```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d2000_a_heavy ^
    --sublora.intrinsic_dim=2000 ^
    --sublora.allocation_mode=fixed ^
    --sublora.allocation_ratio=0.2
```

#### 10. Learned Gating (Adaptive Per-Layer)
```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d2000_learned ^
    --sublora.intrinsic_dim=2000 ^
    --sublora.allocation_mode=learned
```

---

## Multi-Seed Experiments (3 Seeds per Configuration)

For statistical robustness, run each configuration with 3 different random seeds:

### Example: d=1000 Learned Gating with 3 Seeds

```bash
# Seed 1
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_learned_seed1 ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=learned ^
    --login.wandb_run_name=d1000_learned_seed1

# Seed 2
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_learned_seed2 ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=learned ^
    --login.wandb_run_name=d1000_learned_seed2

# Seed 3
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_learned_seed3 ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=learned ^
    --login.wandb_run_name=d1000_learned_seed3
```

**Note**: The random seed is controlled by the `seed` parameter in `create_intrinsic_model()`, which is currently hardcoded to 137. To use different seeds, you would need to add a `--sublora.seed` parameter to the config system or modify the code.

---

## Multi-GPU Training

For faster training on multiple GPUs:

### Single Node, Multiple GPUs (e.g., 2 GPUs)

```bash
torchrun --standalone --nproc_per_node=2 experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_learned ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=learned
```

### Example: 4 GPUs

```bash
torchrun --standalone --nproc_per_node=4 experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_learned ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=learned
```

---

## Quick Test Runs (Reduced Iterations)

For debugging or quick tests, reduce the number of iterations:

```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\test ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=learned ^
    --training.max_iters=100 ^
    --training.eval_interval=10
```

---

## Bound Evaluation Commands

After training, compute generalization bounds:

### Single Model Evaluation

```bash
python experiments/eval_bounds.py ^
    --config-file=config/sublora_bounds.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --model.best_checkpoint_path=C:\path\to\output\d1000_learned\ckpt.pt ^
    --bounds.bound_type=document_level ^
    --data.openwebtext_train_eot_indices_file=C:\path\to\eot_indices.npy ^
    --data.empirical_document_length_distribution_file=C:\path\to\doc_lengths.npy
```

### Sequence-Level Bound

```bash
python experiments/eval_bounds.py ^
    --config-file=config/sublora_bounds.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --model.best_checkpoint_path=C:\path\to\output\d1000_learned\ckpt.pt ^
    --bounds.bound_type=sequence_level
```

---

## Parameter Reference

### Core Training Parameters

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `--config-file` | Path to YAML config | - | `config/sublora_train.yaml` |
| `--data.dataset_dir` | Path to processed data | TOADD | `C:\data\openwebtext` |
| `--login.out_dir` | Output directory | TOADD | `C:\output\experiment1` |
| `--login.wandb_log` | Enable W&B logging | True | `True`, `False` |
| `--login.wandb_run_name` | W&B run name | training | `d1000_learned_seed1` |

### SubLoRA Allocation Parameters

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `--sublora.intrinsic_dim` | Subspace dimension d | 50000 | `1000`, `2000` |
| `--sublora.allocation_mode` | Allocation strategy | uniform | `uniform`, `fixed`, `learned` |
| `--sublora.allocation_ratio` | d_B/(d_A+d_B) for fixed | 0.5 | `0.2`, `0.5`, `0.8` |
| `--sublora.use_lora` | Enable LoRA | True | `True`, `False` |
| `--sublora.attention_linear_lora_r` | LoRA rank (attention) | 4 | `2`, `4`, `8` |
| `--sublora.lora_alpha` | LoRA scaling factor | 32 | `16`, `32`, `64` |

### Training Hyperparameters

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `--training.max_iters` | Total training iterations | 600000 | `100`, `10000`, `600000` |
| `--training.gradient_accumulation_steps` | Gradient accumulation | 40 | `1`, `20`, `40` |
| `--training.eval_interval` | Evaluation frequency | 10 | `10`, `100`, `1000` |
| `--data.batch_size` | Batch size per GPU | 8 | `4`, `8`, `16` |
| `--data.block_size` | Sequence length | 1024 | `512`, `1024`, `2048` |
| `--optimizer.learning_rate` | Learning rate | 5e-3 | `1e-3`, `5e-3`, `1e-2` |
| `--optimizer.weight_decay` | Weight decay | 1e-2 | `0`, `1e-2`, `1e-1` |

### Model Architecture

| Parameter | Description | Default | Example Values |
|-----------|-------------|---------|----------------|
| `--model.n_layer` | Number of layers | 12 | `6`, `12`, `24` |
| `--model.n_head` | Number of heads | 12 | `8`, `12`, `16` |
| `--model.n_embd` | Embedding dimension | 768 | `512`, `768`, `1024` |

---

## Parallel Execution Strategy

For efficient experimentation, you can run multiple configurations in parallel on different GPUs:

### PowerShell Script for Parallel Runs (2 GPUs)

```powershell
# GPU 0: Run d=1000 experiments
Start-Process -NoNewWindow cmd -ArgumentList "/c conda activate sublora && set CUDA_VISIBLE_DEVICES=0 && python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=C:\data\openwebtext --login.out_dir=C:\output\d1000_uniform --sublora.intrinsic_dim=1000 --sublora.allocation_mode=uniform"

# GPU 1: Run d=2000 experiments
Start-Process -NoNewWindow cmd -ArgumentList "/c conda activate sublora && set CUDA_VISIBLE_DEVICES=1 && python experiments/train.py --config-file=config/sublora_train.yaml --data.dataset_dir=C:\data\openwebtext --login.out_dir=C:\output\d2000_uniform --sublora.intrinsic_dim=2000 --sublora.allocation_mode=uniform"
```

### Sequential Execution (One After Another)

```bash
# Run all 10 configs sequentially
for /L %%i in (1,1,10) do (
    echo Running configuration %%i...
    REM Add your command here
)
```

---

## Data Preparation

Before running experiments, prepare the dataset:

```bash
conda activate sublora
python data/openwebtext/prepare.py
```

This will:
- Download OpenWebText (~54GB)
- Tokenize with GPT-2 BPE tokenizer
- Create `train.bin` (~17GB) and `val.bin` (~8.5MB)
- Save to `data/openwebtext/`

---

## Monitoring and Logging

### View Training Progress

```bash
# Real-time log watching (if logs saved to file)
tail -f C:\path\to\output\d1000_learned\train.log

# Or use Weights & Biases dashboard
# https://wandb.ai/your-project/SubLoRA_Pretrain
```

### Check GPU Usage

```bash
# Monitor GPU utilization
nvidia-smi -l 1

# Detailed GPU stats
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv -l 1
```

---

## Resuming Interrupted Training

If training is interrupted, resume from checkpoint:

```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_learned ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=learned ^
    --model.init_from=resume
```

**Note**: Checkpoints are automatically saved in the `out_dir` based on `always_save_checkpoint` setting.

---

## Custom Allocation Ratios

To experiment with ratios not in the proposal (e.g., 0.3, 0.7):

```bash
# d_B = 0.3d, d_A = 0.7d
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_custom_0.3 ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=fixed ^
    --sublora.allocation_ratio=0.3

# d_B = 0.7d, d_A = 0.3d
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\d1000_custom_0.7 ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=fixed ^
    --sublora.allocation_ratio=0.7
```

---

## Debugging Commands

### Check Configuration Loading

```bash
python experiments/train.py ^
    --config-file=config/sublora_train.yaml ^
    --data.dataset_dir=C:\path\to\data\openwebtext ^
    --login.out_dir=C:\path\to\output\debug ^
    --sublora.intrinsic_dim=1000 ^
    --sublora.allocation_mode=learned ^
    --training.max_iters=1
```

### Verify Data Loading

```bash
python -c "from sublora.utils import get_batch; import torch; train_data = torch.tensor(open('data/openwebtext/train.bin', 'rb').read()[:1000000]); X, Y = get_batch(train_data, 1024, 2, 'cpu'); print(f'Batch shapes: X={X.shape}, Y={Y.shape}')"
```

### Test Model Creation

```bash
python -c "from sublora.nn.create_model import get_model; allocation_config={'mode':'learned','ratio':0.5}; model, *_ = get_model(12,12,768,False,0.0,False,False,False,True,32,0.1,True,4,4,True,1000,1024,'data/openwebtext','out/test','scratch',True,'cpu',None,allocation_config); print(model)"
```

---

## Summary Table: All 10 Configurations

| # | Budget | Mode | Ratio | d_A | d_B | Command Suffix |
|---|--------|------|-------|-----|-----|----------------|
| 1 | 1000 | uniform | - | 500 | 500 | `--sublora.intrinsic_dim=1000 --sublora.allocation_mode=uniform` |
| 2 | 1000 | fixed | 0.8 | 200 | 800 | `--sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.8` |
| 3 | 1000 | fixed | 0.5 | 500 | 500 | `--sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.5` |
| 4 | 1000 | fixed | 0.2 | 800 | 200 | `--sublora.intrinsic_dim=1000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.2` |
| 5 | 1000 | learned | - | Per-layer | Per-layer | `--sublora.intrinsic_dim=1000 --sublora.allocation_mode=learned` |
| 6 | 2000 | uniform | - | 1000 | 1000 | `--sublora.intrinsic_dim=2000 --sublora.allocation_mode=uniform` |
| 7 | 2000 | fixed | 0.8 | 400 | 1600 | `--sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.8` |
| 8 | 2000 | fixed | 0.5 | 1000 | 1000 | `--sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.5` |
| 9 | 2000 | fixed | 0.2 | 1600 | 400 | `--sublora.intrinsic_dim=2000 --sublora.allocation_mode=fixed --sublora.allocation_ratio=0.2` |
| 10 | 2000 | learned | - | Per-layer | Per-layer | `--sublora.intrinsic_dim=2000 --sublora.allocation_mode=learned` |

---

## Expected Runtime

Based on the proposal's compute estimates:
- **Per run**: ~3 GPU-hours on A100 80GB
- **Total (10 configs × 3 seeds)**: ~90 GPU-hours ≈ 4 GPU-days
- **With 2 GPUs in parallel**: ~2 days wall-clock time

Adjust `--training.max_iters` proportionally for faster testing.