"""
Create a small dummy dataset for quick testing.
This avoids the 54GB download and long processing time.

Usage: python create_dummy_dataset.py
"""
import os
import numpy as np
import tiktoken

print("=" * 80)
print("CREATING DUMMY DATASET FOR TESTING")
print("=" * 80)

# Sample texts - realistic but small
sample_texts = [
    "The field of machine learning has revolutionized artificial intelligence.",
    "Deep neural networks can learn complex patterns from large datasets.",
    "Natural language processing enables computers to understand human language.",
    "Transformers have become the dominant architecture for language models.",
    "Generalization bounds provide theoretical guarantees for model performance.",
    "Low-rank adaptation allows efficient fine-tuning of large language models.",
    "SubLoRA combines LoRA with subspace training for better compression.",
    "PAC-Bayes theory gives non-vacuous bounds for neural networks.",
    "The attention mechanism allows models to focus on relevant information.",
    "Gradient descent optimizes model parameters to minimize loss functions.",
    "Regularization techniques prevent overfitting on training data.",
    "Cross-validation helps estimate model performance on unseen data.",
    "Batch normalization stabilizes training of deep neural networks.",
    "Dropout randomly deactivates neurons during training for better generalization.",
    "The learning rate controls the step size in gradient descent.",
    "Momentum accumulates gradients to accelerate convergence.",
    "Adam optimizer adapts learning rates for each parameter.",
    "Backpropagation computes gradients through the computational graph.",
    "Convolutional layers extract spatial features from images.",
    "Recurrent networks process sequential data with hidden states.",
] * 500  # Repeat 500 times to get ~10k tokens

print(f"Sample texts: {len(sample_texts)} documents")

# Initialize GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

# Tokenize and create train data
train_tokens = []
doc_lengths = []
eot_indices = []
current_idx = 0

for text in sample_texts:
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)  # Add end-of-text token (50256)
    train_tokens.extend(ids)
    
    doc_len = len(ids)
    doc_lengths.append(doc_len)
    current_idx += doc_len
    eot_indices.append(current_idx - 1) # Index of the EOT token

# Create validation set (10% of train)
val_tokens = train_tokens[:len(train_tokens)//10]

print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens: {len(val_tokens):,}")

# Save as binary files
output_dir = os.path.dirname(__file__)

# Save train.bin
train_arr = np.array(train_tokens, dtype=np.uint16)
train_path = os.path.join(output_dir, 'train.bin')
train_arr.tofile(train_path)

# Save val.bin
val_arr = np.array(val_tokens, dtype=np.uint16)
val_path = os.path.join(output_dir, 'val.bin')
val_arr.tofile(val_path)

# Save auxiliary files for bounds evaluation
eot_indices_arr = np.array(eot_indices, dtype=np.int32) # or int64 depending on size
eot_path = os.path.join(output_dir, 'eot_indices.npy')
np.save(eot_path, eot_indices_arr)

doc_lengths_arr = np.array(doc_lengths, dtype=np.int32)
doc_len_path = os.path.join(output_dir, 'doc_lengths.npy')
np.save(doc_len_path, doc_lengths_arr)

# Verify files
train_size_kb = os.path.getsize(train_path) / 1024
val_size_kb = os.path.getsize(val_path) / 1024

print("\n" + "=" * 80)
print("SUCCESS! Dummy dataset created:")
print("=" * 80)
print(f"  train.bin: {len(train_tokens):,} tokens ({train_size_kb:.1f} KB)")
print(f"  val.bin: {len(val_tokens):,} tokens ({val_size_kb:.1f} KB)")
print(f"  eot_indices.npy: {len(eot_indices):,} entries")
print(f"  doc_lengths.npy: {len(doc_lengths):,} entries")
print(f"\nLocation: {output_dir}")
print("\n" + "=" * 80)
print("TO USE THIS DATASET:")
print("=" * 80)
print("Run training with: --data.dataset_dir=data")
print("\nExample:")
print("  python experiments/train.py \\")
print("      --config-file=config/sublora_train.yaml \\")
print("      --data.dataset_dir=data \\")
print("      --login.out_dir=out/test_run \\")
print("      --sublora.intrinsic_dim=1000 \\")
print("      --sublora.allocation_mode=learned \\")
print("      --training.max_iters=100")
print("\n" + "=" * 80)
print("NOTE: This is a DUMMY dataset for testing only!")
print("For real experiments, use the full OpenWebText dataset.")
print("=" * 80)
