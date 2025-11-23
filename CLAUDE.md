# Non-Vacuous Generalization Bounds for Large Language Models

## Repository Overview

This repository implements the paper "Non-Vacuous Generalization Bounds for Large Language Models" by Lotfi et al. (ICML 2024). The code provides the first non-vacuous generalization bounds for language model pretraining on next-token prediction, demonstrating mathematically that LLMs can generalize beyond their training data.

## Key Contributions

1. **Novel bounds for unbounded NLL objective**: Bounds tailored for the continuous bits-per-dimension loss used in LLM evaluation
2. **Subsampling bounds**: Efficient bound evaluation (45 minutes on 1 GPU vs 3 days on 8 GPUs)
3. **SubLoRA compression**: Combination of LoRA and linear subspace training for strong nonlinear compression
4. **Scalability**: Non-vacuous bounds for models up to 800M+ parameters
5. **Generalization insights**: Larger models achieve better compression and bounds

---

## Repository Structure

### Core Package (`sublora/`)

The main package containing all implementation logic.

#### `sublora/nn/` - Neural Network Components

- **`model.py`**: GPT-2 implementation with LoRA support
  - `GPT` class: Main GPT-2 model with configurable LoRA layers
  - `GPTConfig`: Configuration dataclass for model hyperparameters
  - `CausalSelfAttention`: Multi-head attention with optional LoRA projections
  - `MLP`: Feed-forward network layers
  - `Block`: Transformer block (attention + MLP)
  - Support for rotary positional embeddings (RoPE)
  - Integration with `loralib` for low-rank adaptation

- **`projectors.py`**: Intrinsic dimensionality reduction operators
  - `IDModule`: Main wrapper for intrinsic dimension training
  - `LazyRandom`: Random projection operator
  - `LazyRandomQR`: QR-decomposed random projection
  - `RoundedDoubleKron`: Double Kronecker product projector
  - `RoundedDoubleKronQR`: QR-decomposed Kronecker projector
  - `FiLMLazyRandom`: Filter-based random projection
  - `FastfoodOperator`: Fast Hadamard transform-based projector
  - `SparseOperator`: Sparse random projection
  - `QuantizingWrapper`: Wrapper for quantized model inference
  - `create_intrinsic_model()`: Factory function to create intrinsic dimensionality models

- **`linear_operator_base.py`**: Base classes for lazy linear operators
  - Abstract classes for memory-efficient matrix operations
  - Support for Kronecker products, concatenation, permutations

- **`create_model.py`**: Model creation utilities
  - `get_model()`: Factory function to instantiate models with specified configurations

#### `sublora/bounds/` - Generalization Bounds

- **`compute_bounds.py`**: Core bound computation functions
  - `llm_subsampling_bound()`: Main subsampling-based bound for LLMs
  - `pac_bayes_bound()`: Standard PAC-Bayes bound
  - `pac_bayes_bound_opt()`: Optimized PAC-Bayes bound with gamma optimization
  - `compute_convexity_bound()`: Convexity-based bound
  - `compute_mcallester_bound()`: McAllester's PAC-Bayes bound
  - `compute_catoni_bound()`: Catoni's PAC-Bayes bound

- **`quantize_fns.py`**: Weight quantization for compression
  - `finetune_quantization()`: Quantization-aware training
  - `get_kmeans_symbols_and_codebook()`: K-means clustering for codebook generation
  - `get_random_symbols_and_codebook()`: Random initialization for codebook
  - `Quantize`: Custom autograd function for differentiable quantization
  - `QuantizingWrapper`: Neural network wrapper for quantized inference

- **`bound_utils.py`**: Utilities for bound computation
  - `quantize_model()`: Apply quantization to model weights
  - `compute_bound_scores()`: Evaluate metrics for bound computation
  - `compute_bound_metrics()`: Calculate final bound values

- **`compression_summary.py`**: Compression analysis utilities
  - Functions to summarize model compression statistics

#### `sublora/` - Main Pipeline

- **`sublora_pipeline.py`**: Main training and evaluation pipeline
  - `SubLoRA` class: Orchestrates training, evaluation, and bound computation
  - Handles distributed training with DDP
  - Implements training loop with learning rate scheduling
  - Manages checkpoint saving and loading
  - Integrates with Weights & Biases for logging

- **`utils.py`**: Utility functions
  - `get_lr()`: Learning rate scheduling (cosine decay with warmup)
  - `get_batch()`: Data loading and batching
  - `sample_single_document()`: Document-level sampling
  - `sample_nonoverlapping_sequences()`: Non-overlapping sequence sampling
  - `get_model_config()`: Configuration extraction

---

### Experiments (`experiments/`)

Scripts for running experiments and evaluations.

- **`train.py`**: Main training script
  - Loads configuration from YAML files
  - Initializes `SubLoRA` pipeline
  - Runs pretraining with SubLoRA parametrization
  - Usage: See "Replication Guide" below

- **`eval_bounds.py`**: Bound evaluation script
  - Loads pretrained checkpoints
  - Performs weight quantization
  - Computes generalization bounds
  - Supports document-level and sequence-level bounds

- **`run_glue_no_trainer.py`**: GLUE finetuning script
  - Finetunes GPT-2 on GLUE tasks (QQP, CoLA)
  - Supports SubLoRA finetuning in intrinsic dimension subspace
  - Can start from pretrained or random initialization

- **`sample.py`**: Text generation script
  - Generates samples from trained models
  - Useful for qualitative evaluation

---

### Configuration (`config/`)

YAML configuration files for experiments.

- **`sublora_train.yaml`**: Training configuration
  - Model architecture: 12 layers, 12 heads, 768 embedding dim (GPT-2 small)
  - LoRA settings: rank 4 for attention, alpha 32, dropout 0.1
  - Intrinsic dimension: 50,000
  - Optimizer: AdamW with lr=5e-3, weight decay=1e-2
  - Training: 600k iterations, batch size 8, gradient accumulation 40
  - Data: 1024 token sequences

- **`sublora_bounds.yaml`**: Bound evaluation configuration
  - Quantization: 11 levels, learning rate 5e-5
  - Bound computation: 10k samples, document-level or sequence-level
  - Sliding window: 100 tokens for long documents
  - Extra bits for hyperparameter search: 7

---

### Data (`data/`)

Data preparation scripts.

- **`openwebtext/prepare.py`**: OpenWebText dataset preparation
  - Downloads OpenWebText from HuggingFace (~54GB, 8M documents)
  - Tokenizes with GPT-2 BPE tokenizer (tiktoken)
  - Creates train/validation splits (0.05% validation)
  - Saves as binary files (train.bin ~17GB, val.bin ~8.5MB)
  - Adds EOT tokens to separate documents

- **`openwebtext/readme.md`**: Dataset documentation

---

### Installation Files

- **`setup.py`**: Package installation script
  - Installs `sublora` package
  - Excludes experiment and config directories from package

- **`environment.yml`**: Conda environment specification
  - Python 3.8.17
  - PyTorch 2.0.1 with CUDA 11
  - Transformers 4.31.0
  - LoRA library (loralib 0.1.1)
  - PEFT 0.4.0
  - Datasets, evaluate, accelerate
  - Scientific computing: numpy, scipy, scikit-learn
  - Logging: wandb, neptune

- **`README.md`**: Project documentation
  - Paper citation
  - Setup instructions
  - Experiment commands

---

## Replication Guide

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml -n sublora

# Activate environment
conda activate sublora

# Install sublora package
pip install -e .
```

### 2. Data Preparation

Download and preprocess the OpenWebText dataset:

```bash
python data/openwebtext/prepare.py
```

This will create:
- `data/openwebtext/train.bin` (~17GB, ~9B tokens)
- `data/openwebtext/val.bin` (~8.5MB, ~4M tokens)

**Additional Requirements for Bounds:**

To compute document-level bounds, you need:
1. **EOT indices file**: Numpy array of end-of-text token positions marking document boundaries
2. **Document length distribution file**: Numpy array of document lengths in the training data

These files are referenced in `sublora_bounds.yaml` and must be generated from the training data.

### 3. Model Pretraining

#### Single GPU Training

```bash
python experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=/path/to/data/openwebtext \
    --login.out_dir=/path/to/output
```

#### Multi-GPU Training (Recommended)

For 4 GPUs within a single node:

```bash
torchrun --standalone --nproc_per_node=4 experiments/train.py \
    --config-file=config/sublora_train.yaml \
    --data.dataset_dir=/path/to/data/openwebtext \
    --login.out_dir=/path/to/output
```

**Training Configuration Details:**
- **Architecture**: GPT-2 small (124M base parameters)
- **SubLoRA parametrization**:
  - Intrinsic dimension: 50,000
  - LoRA rank: 4 (attention layers)
  - LoRA alpha: 32
- **Optimization**:
  - Learning rate: 5e-3 with cosine decay
  - Batch size: 8 × 40 gradient accumulation = 320 effective
  - Training iterations: 600k
- **Expected output**: Checkpoints saved to `out_dir`

### 4. Generalization Bounds Computation

After training, compute bounds on the pretrained model:

```bash
python experiments/eval_bounds.py \
    --config-file=config/sublora_bounds.yaml \
    --data.dataset_dir=/path/to/data/openwebtext \
    --model.best_checkpoint_path=/path/to/best_checkpoint.pt \
    --bounds.bound_type=document_level \
    --data.openwebtext_train_eot_indices_file=/path/to/eot_indices.npy \
    --data.empirical_document_length_distribution_file=/path/to/doc_lengths.npy
```

**Bound Computation Steps:**
1. Load pretrained SubLoRA model
2. Quantize weights using arithmetic coding (11 levels)
3. Compute compressed model size
4. Evaluate subsampled empirical risk (10k samples)
5. Calculate generalization bound

**Bound Types:**
- `document_level`: Bounds treating each document as a sample
- `sequence_level`: Bounds treating fixed-length sequences as samples

**Output**: Bound values, compression statistics, and empirical risk metrics

### 5. GLUE Finetuning (Optional)

Finetune GPT-2 with SubLoRA on GLUE tasks:

```bash
python experiments/run_glue_no_trainer.py \
    --model_name_or_path=gpt2 \
    --task_name=qqp \
    --max_length=128 \
    --pad_to_max_length=True \
    --per_device_train_batch_size=32 \
    --learning_rate=2e-5 \
    --num_train_epochs=5 \
    --output_dir=/path/to/output \
    --cache_dir=/path/to/cache \
    --intrinsic_dim=1000 \
    --load_pretrained_model=1
```

**Parameters:**
- `task_name`: GLUE task (qqp, cola, etc.)
- `intrinsic_dim`: Subspace dimensionality for finetuning
- `load_pretrained_model`: 0 (random init) or 1 (pretrained GPT-2)

---

## Key Concepts

### SubLoRA Parametrization

SubLoRA combines two compression techniques:

1. **LoRA (Low-Rank Adaptation)**:
   - Decomposes weight updates as: W = W₀ + BA
   - B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d,k)
   - Applied to attention Q, V projections and language model head

2. **Linear Subspace Training**:
   - Constrains all trainable parameters to d-dimensional subspace
   - θ = θ₀ + P·v, where P ∈ ℝ^(D×d), v ∈ ℝ^d
   - P is a random projection (LazyRandom, Kronecker, etc.)
   - Only v is optimized during training

**Combined Effect**:
- Strong nonlinear compression of model
- Enables evaluation of generalization bounds
- Models with ~124M parameters compressed to ~50k effective parameters

### Quantization for Compression

The bound computation requires quantizing model weights:

1. **Codebook Learning**:
   - K-means or random initialization
   - Typically 11 quantization levels
   - Codebook optimized via quantization-aware training

2. **Arithmetic Coding**:
   - Compresses quantized weights losslessly
   - Exploits symbol frequency distribution
   - Final compression measured in bits

3. **Compression Size**:
   - Used as KL divergence term in PAC-Bayes bounds
   - Smaller compression → tighter bounds

### Bound Evaluation

The main bound formula:

```
Generalization Error ≤ Training Error + Complexity Term
```

Where:
- **Training Error**: Evaluated on subsample of data (10k samples)
- **Complexity Term**: Function of:
  - Compressed model size (divergence)
  - Dataset size
  - Sample size
  - Confidence parameter (δ)

**Subsampling**:
- Avoids evaluating on entire dataset (9B tokens)
- Theoretically sound via refined analysis
- Makes evaluation tractable

---

## Understanding the Code Flow

### Training Flow

1. `experiments/train.py` loads config from `sublora_train.yaml`
2. Initializes `SubLoRA` class with configuration
3. `SubLoRA.train()` method:
   - Creates GPT-2 model via `get_model()`
   - Wraps model with `IDModule` for subspace training
   - Sets up distributed training (DDP) if multi-GPU
   - Runs training loop with gradient accumulation
   - Evaluates on validation set periodically
   - Saves checkpoints

### Bound Evaluation Flow

1. `experiments/eval_bounds.py` loads config from `sublora_bounds.yaml`
2. Initializes `SubLoRA` class with bound configuration
3. `SubLoRA.get_bounds()` method:
   - Loads pretrained checkpoint
   - Performs quantization via `quantize_model()`
   - Computes compression size using arithmetic coding
   - Samples data for empirical risk evaluation
   - Computes bound using `llm_subsampling_bound()`
   - Logs results

---

## Customization Options

### Model Architecture

Modify `config/sublora_train.yaml`:

```yaml
model:
  n_layer: 12      # Number of transformer layers
  n_head: 12       # Number of attention heads
  n_embd: 768      # Embedding dimension
```

For GPT-2 variants:
- Small: 12 layers, 12 heads, 768 dim (124M params)
- Medium: 24 layers, 16 heads, 1024 dim (350M params)
- Large: 36 layers, 20 heads, 1280 dim (774M params)
- XL: 48 layers, 25 heads, 1600 dim (1.5B params)

### SubLoRA Configuration

Adjust compression level:

```yaml
sublora:
  intrinsic_dim: 50000           # Subspace dimension (lower = more compression)
  attention_linear_lora_r: 4     # LoRA rank for attention
  linear_head_lora_r: 4          # LoRA rank for LM head
  lora_alpha: 32                 # LoRA scaling factor
```

### Projection Operators

Change in `sublora/nn/projectors.py` via `create_intrinsic_model()`:

- `dense`: Standard random projection
- `sparse`: Sparse random projection
- `fastfood`: Fast Hadamard transform
- `rdkron`: Rounded double Kronecker product
- `rdkronqr`: QR-decomposed Kronecker product
- `film`: Filter-based projection
- `filmrdkronqr`: Combined filter + Kronecker

### Quantization Settings

Modify `config/sublora_bounds.yaml`:

```yaml
bounds:
  levels: 11              # Number of quantization levels (must be odd)
  quant_lr: 5e-5          # Learning rate for quantization-aware training
  max_quant_iters: 0      # QAT iterations (0 = no finetuning)
  use_kmeans: False       # K-means vs random initialization
```

---

## Expected Results

Based on the paper, you should expect:

1. **Training Loss**:
   - GPT-2 small with SubLoRA: ~3.3-3.4 bits per dimension
   - Comparable to full model training

2. **Compression**:
   - Base model: ~124M parameters
   - SubLoRA compressed: ~50k parameters
   - Quantized size: ~few hundred KB

3. **Generalization Bounds**:
   - Non-vacuous (< 1.0) bounds on test error
   - Tighter bounds for larger models
   - Document-level bounds tighter than sequence-level

4. **Bound Evaluation Time**:
   - Single GPU: ~45 minutes
   - Full dataset evaluation would take ~3 days on 8 GPUs

---

## Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Reduce `data.batch_size` in config
   - Increase `training.gradient_accumulation_steps`
   - Use smaller model architecture

2. **Slow Training**:
   - Use multiple GPUs with `torchrun`
   - Enable `system.compile=True` for PyTorch 2.0 compilation
   - Check `system.dtype` is set to `bfloat16` or `float16`

3. **Bound Evaluation Fails**:
   - Ensure EOT indices and document length files are provided
   - Check checkpoint path is correct
   - Verify quantization settings are valid

4. **NaN Loss**:
   - Reduce learning rate
   - Check gradient clipping is enabled
   - Verify data preprocessing is correct

---

## Citation

If you use this code, please cite:

```bibtex
@article{lotfi2023non,
  title={Non-vacuous generalization bounds for large language models},
  author={Lotfi, Sanae and Finzi, Marc and Kuang, Yilun and Rudner, Tim GJ and Goldblum, Micah and Wilson, Andrew Gordon},
  journal={arXiv preprint arXiv:2312.17173},
  year={2023}
}
```

---

## License

Apache 2.0

---

## Additional Resources

- **Paper**: https://arxiv.org/abs/2312.17173
- **NanoGPT**: https://github.com/karpathy/nanoGPT (base implementation)
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **PAC-Bayes Theory**: Background on generalization bounds
- **OpenWebText**: https://huggingface.co/datasets/openwebtext

---

## Summary

This repository provides a complete implementation for:
1. Training GPT-2 models with SubLoRA compression
2. Computing the first non-vacuous generalization bounds for LLMs
3. Demonstrating that larger models achieve better compression and generalization

The key innovation is combining LoRA with linear subspace training to achieve strong nonlinear compression, enabling tractable bound evaluation while maintaining model performance.
