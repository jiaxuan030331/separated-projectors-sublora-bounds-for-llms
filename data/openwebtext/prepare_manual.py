import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
import glob
import tarfile

# Configuration
MANUAL_DATA_DIR = r"C:\Users\sunny\Downloads\openwebtext_manual" # Change this to where you extracted the files
OUTPUT_DIR = os.path.dirname(__file__)

def extract_inner_archives(data_dir):
    """
    OpenWebText comes as tars containing xz-compressed tars.
    This function finds those inner archives (often named *_data) and extracts them.
    """
    print(f"Scanning {data_dir} for nested archives...")
    # Look for files that look like the inner archives (often have 'data' in name or .xz extension)
    # We exclude .txt files and directories
    candidates = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".txt") or file.endswith(".bin") or file.endswith(".npy"):
                continue
            # Check if it's an archive we want to extract
            # It should have 'data' in the name, AND NOT be inside an already extracted folder
            if "subset" in file and "data" in file and "_extracted" not in root:
                candidates.append(os.path.join(root, file))

    if not candidates:
        print("No nested archives found. Assuming files are already extracted.")
        return

    print(f"Found {len(candidates)} nested archives. Extracting...")
    
    for archive_path in tqdm(candidates, desc="Extracting nested archives"):
        # Create a folder for this archive's contents to keep things organized
        output_subdir = archive_path + "_extracted"
        
        # Skip if already extracted
        if os.path.exists(output_subdir) and os.listdir(output_subdir):
            continue
            
        try:
            os.makedirs(output_subdir, exist_ok=True)
            # Open as tar (Python handles .tar.xz automatically if lzma is available)
            with tarfile.open(archive_path, "r:*") as tar:
                tar.extractall(path=output_subdir)
        except Exception as e:
            print(f"Could not extract {archive_path} as tar: {e}")
            # Clean up empty dir
            if os.path.exists(output_subdir) and not os.listdir(output_subdir):
                os.rmdir(output_subdir)

def prepare_manual_dataset(data_dir):
    # Step 0: Handle nested extraction
    extract_inner_archives(data_dir)

    print(f"Scanning for text files in {data_dir}...")
    # Find all .txt files recursively
    txt_files = glob.glob(os.path.join(data_dir, "**/*.txt"), recursive=True)
    
    if not txt_files:
        print(f"No .txt files found in {data_dir}. Please ensure the archives are extracted.")
        return

    print(f"Found {len(txt_files)} text files.")
    
    # Load text files into a dataset
    print("Loading files into dataset...")
    # Passing 8 million file paths directly to load_dataset can cause issues on Windows
    # Instead, we'll point it to the directory and let it find the files, or use a generator
    
    # Option 1: Point to the directory (simpler, but might hit the same issue if it expands internally)
    # dataset = load_dataset("text", data_dir=data_dir, sample_by="line")
    
    # Option 2: Use a generator (most robust for 8M files)
    def gen():
        for file_path in txt_files:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                if text.strip(): # Skip empty files
                    yield {'text': text}

    from datasets import Dataset
    dataset = Dataset.from_generator(gen)
    
    # The dataset returned by from_generator is not a DatasetDict, so we wrap it or split it directly
    # It returns a Dataset object which corresponds to the 'train' split usually
    
    # Create train/val split
    print("Creating train/val split...")
    split_dataset = dataset.train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    # Tokenize
    print("Initializing tokenizer...")
    enc = tiktoken.get_encoding("gpt2")

    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        out = {'ids': ids, 'len': len(ids)}
        return out

    print("Tokenizing...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing",
        num_proc=1, # Single process for Windows safety
    )

    # Save to binary files
    print("Saving binary files...")
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(OUTPUT_DIR, f'{split}.bin')
        dtype = np.uint16
        
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024 if arr_len > 1024 else 1

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Saved {filename}")

    # Also generate the auxiliary files needed for bounds evaluation
    print("Generating auxiliary files for bounds evaluation...")
    
    # 1. Generate eot_indices.npy (indices of End-Of-Text tokens)
    # We need to scan the train.bin to find all 50256 tokens
    train_bin_path = os.path.join(OUTPUT_DIR, 'train.bin')
    train_data = np.memmap(train_bin_path, dtype=np.uint16, mode='r')
    
    # Find indices where token is EOT (50256)
    eot_token = enc.eot_token
    eot_indices = np.where(train_data == eot_token)[0]
    
    eot_save_path = os.path.join(OUTPUT_DIR, 'eot_indices.npy')
    np.save(eot_save_path, eot_indices)
    print(f"Saved {eot_save_path} with {len(eot_indices)} documents")

    # 2. Generate doc_lengths.npy
    # Calculate lengths based on EOT positions
    # First doc length is eot_indices[0] + 1 (since 0-indexed)
    # Subsequent lengths are diffs between indices
    
    if len(eot_indices) > 0:
        doc_lengths = np.zeros(len(eot_indices), dtype=np.int32)
        doc_lengths[0] = eot_indices[0] + 1
        doc_lengths[1:] = np.diff(eot_indices)
        
        doc_len_save_path = os.path.join(OUTPUT_DIR, 'doc_lengths.npy')
        np.save(doc_len_save_path, doc_lengths)
        print(f"Saved {doc_len_save_path}")

if __name__ == "__main__":
    prepare_manual_dataset(MANUAL_DATA_DIR)
