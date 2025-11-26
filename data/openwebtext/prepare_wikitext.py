"""
Alternative dataset preparation using WikiText-103 (Windows-compatible).
WikiText is much smaller (~500MB) but works reliably on Windows.

Usage: python prepare_wikitext.py
"""
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

print("=" * 80)
print("PREPARING WIKITEXT-103 DATASET (WINDOWS-COMPATIBLE)")
print("=" * 80)

# WikiText-103 is much more reliable on Windows
num_proc = 1  # Single process for Windows compatibility

if __name__ == '__main__':
    print("Loading WikiText-103 dataset...")
    print("This is smaller than OpenWebText but works on Windows!")

    # Load WikiText-103 (more reliable than OpenWebText)
    dataset = load_dataset("wikitext", "wikitext-103-v1")

    print(f"\nDataset loaded:")
    print(f"  Train: {len(dataset['train']):,} documents")
    print(f"  Validation: {len(dataset['validation']):,} documents")
    print(f"  Test: {len(dataset['test']):,} documents")

    # Use train and validation splits
    split_dataset = {
        'train': dataset['train'],
        'val': dataset['validation']
    }

    # Tokenize with GPT-2 BPE
    enc = tiktoken.get_encoding("gpt2")

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize
    print("\nTokenizing dataset...")
    tokenized = {}
    for split_name, dset in split_dataset.items():
        print(f"  Processing {split_name}...")
        tokenized[split_name] = dset.map(
            process,
            remove_columns=['text'],
            desc=f"Tokenizing {split_name}",
            num_proc=num_proc,
        )

    # Write binary files
    print("\nWriting binary files...")
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        dtype = np.uint16

        print(f"\n  Creating {split}.bin ({arr_len:,} tokens)...")
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 100  # Fewer batches for smaller dataset

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'  Writing {split}.bin'):
            batch = dset.shard(
                num_shards=total_batches,
                index=batch_idx,
                contiguous=True
            ).with_format('numpy')

            if len(batch['ids']) > 0:
                arr_batch = np.concatenate(batch['ids'])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
        arr.flush()

        # Verify
        file_size_mb = os.path.getsize(filename) / (1024 * 1024)
        print(f"    ✓ {split}.bin created: {file_size_mb:.1f} MB")

    print("\n" + "=" * 80)
    print("SUCCESS! WikiText-103 dataset prepared.")
    print("=" * 80)
    print(f"\nFiles created in: {os.path.dirname(__file__)}")
    print("\nYou can now run training with:")
    print("  --data.dataset_dir=data")
