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
    
    # Since datasets library is failing on Windows with LocalFileSystem caching issues,
    # we will implement a simple custom dataset class that mimics the behavior we need.
    # This bypasses the datasets library for the loading/splitting phase entirely.
    
    import random
    random.seed(2357)
    
    # Shuffle the file list directly
    random.shuffle(txt_files)
    
    # Calculate split index
    split_idx = int(len(txt_files) * (1 - 0.0005))
    train_files = txt_files[:split_idx]
    val_files = txt_files[split_idx:]
    
    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")
    
    # Tokenize
    print("Initializing tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    eot_token = enc.eot_token

    def process_files_to_bin(files, split_name):
        arr_len = 0
        # First pass: calculate total length (optional, but good for progress bar)
        # Actually, for speed, let's just write to a temporary list or file and then memmap
        # Or better: write directly to a binary file using standard file I/O, then convert to memmap if needed
        # But to match the original format, we need a single contiguous binary file.
        
        # Let's use a dynamic approach: write chunks to disk, then merge? 
        # Or just iterate and write to a growing file.
        
        filename = os.path.join(OUTPUT_DIR, f'{split_name}.bin')
        print(f"Processing {split_name} split to {filename}...")
        
        # We'll write directly to the file as uint16
        # This is much more memory efficient than building a huge list
        token_count = 0
        
        with open(filename, 'wb') as f:
            for file_path in tqdm(files, desc=f"Tokenizing {split_name}"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as txt_f:
                        text = txt_f.read()
                        if not text.strip(): continue
                        
                        ids = enc.encode_ordinary(text)
                        ids.append(eot_token)
                        
                        # Convert to uint16 and write bytes
                        # Ensure we don't exceed uint16 range (GPT2 vocab is ~50k, so it fits)
                        arr = np.array(ids, dtype=np.uint16)
                        f.write(arr.tobytes())
                        token_count += len(ids)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    
        print(f"Saved {filename} with {token_count} tokens.")
        return token_count

    # Process splits
    process_files_to_bin(train_files, 'train')
    process_files_to_bin(val_files, 'val')

    # Also generate the auxiliary files needed for bounds evaluation
    print("Generating auxiliary files for bounds evaluation...")

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
