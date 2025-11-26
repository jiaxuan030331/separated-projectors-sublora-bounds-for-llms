# Windows-compatible version of prepare.py for openwebtext dataset
# Fixes: OSError [Errno 22] Invalid argument due to Windows path issues

import os
from tqdm import tqdm
import numpy as np
import tiktoken

# Set a shorter cache directory to avoid Windows path length issues
os.environ['HF_DATASETS_CACHE'] = 'C:/hf_cache'

from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
# On Windows, use fewer workers to avoid multiprocessing issues
num_proc = 1  # Changed from 8 to 1 for Windows compatibility

# number of workers in load_dataset() call
num_proc_load_dataset = 1  # Changed from num_proc for Windows

if __name__ == '__main__':
    print("=" * 80)
    print("WINDOWS-COMPATIBLE OPENWEBTEXT PREPARATION")
    print("=" * 80)
    print(f"Cache directory: {os.environ.get('HF_DATASETS_CACHE', 'default')}")
    print(f"Workers: {num_proc}")
    print()

    try:
        # Load dataset with Windows-friendly settings
        print("Downloading OpenWebText dataset (this may take a while)...")
        print("Note: Download is ~54GB, processing takes additional time")

        dataset = load_dataset(
            "openwebtext",
            num_proc=num_proc_load_dataset,
            # Use streaming to avoid some Windows path issues
            # streaming=False  # Keep as regular download
        )

        # owt by default only contains the 'train' split, so create a test split
        print("\nCreating train/val split...")
        split_dataset = dataset["train"].train_test_split(
            test_size=0.0005,
            seed=2357,
            shuffle=True
        )
        split_dataset['val'] = split_dataset.pop('test')

        print(f"\nDataset split complete:")
        print(f"  Train: {len(split_dataset['train']):,} documents")
        print(f"  Val: {len(split_dataset['val']):,} documents")

        # Tokenize with GPT-2 BPE
        print("\nInitializing tokenizer...")
        enc = tiktoken.get_encoding("gpt2")

        def process(example):
            ids = enc.encode_ordinary(example['text'])
            ids.append(enc.eot_token)  # 50256 for gpt2
            out = {'ids': ids, 'len': len(ids)}
            return out

        # tokenize the dataset
        print("\nTokenizing dataset...")
        tokenized = split_dataset.map(
            process,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=num_proc,
        )

        # concatenate all the ids in each dataset into one large file
        print("\nWriting binary files...")
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'], dtype=np.uint64)
            filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
            dtype = np.uint16

            print(f"\n  Creating {split}.bin ({arr_len:,} tokens)...")
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024

            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'  Writing {filename}'):
                batch = dset.shard(
                    num_shards=total_batches,
                    index=batch_idx,
                    contiguous=True
                ).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()

            # Verify file was created
            file_size_mb = os.path.getsize(filename) / (1024 * 1024)
            print(f"    ✓ {split}.bin created: {file_size_mb:.1f} MB")

        print("\n" + "=" * 80)
        print("SUCCESS! Dataset preparation complete.")
        print("=" * 80)
        print(f"\nFiles created in: {os.path.dirname(__file__)}")
        print("  - train.bin: ~17GB, ~9B tokens")
        print("  - val.bin: ~8.5MB, ~4M tokens")
        print("\nYou can now run training with:")
        print("  --data.dataset_dir=data")

    except Exception as e:
        print("\n" + "=" * 80)
        print("ERROR DURING DATASET PREPARATION")
        print("=" * 80)
        print(f"\nError: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure you have ~70GB free disk space")
        print("2. Check your internet connection")
        print("3. If download fails repeatedly, try:")
        print("   - Using a VPN")
        print("   - Manually downloading from: https://huggingface.co/datasets/openwebtext")
        print("4. For quick testing, consider creating a small dummy dataset instead")
        raise
