import os
import requests
from tqdm import tqdm

import tarfile
import subprocess

# Configuration
# We will download to a folder in your Downloads directory
DEST_DIR = r"C:\Users\sunny\Downloads\openwebtext_manual"
# You can add more subsets here if you want more data (00 to 20)
# Generating list for all subsets 00 through 20
SUBSETS_TO_DOWNLOAD = [f"urlsf_subset{i:02d}.tar" for i in range(21)]
BASE_URL = "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/"

def extract_file(filename, dest_dir):
    local_path = os.path.join(dest_dir, filename)
    print(f"Extracting {filename}...")
    
    # Method 1: Try system 'tar' command (Windows 10+ includes this)
    # This is often more robust than Python's tarfile on Windows
    try:
        cmd = ["tar", "-xf", local_path, "-C", dest_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Extracted {filename} (using system tar)")
            return
        else:
            # If tar fails, it might be because it's not installed or some other error
            # We don't print the error yet, we try the fallback
            pass
    except Exception:
        pass

    # Method 2: Python tarfile (Fallback)
    try:
        with tarfile.open(local_path, "r") as tar:
            tar.extractall(path=dest_dir)
        print(f"Extracted {filename} (using tarfile)")
    except Exception as e:
        print(f"Error extracting {filename}: {e}")
        print("  Tip: You can try extracting this file manually with 7-Zip if this persists.")

def download_file(filename, dest_dir):
    url = BASE_URL + filename
    local_path = os.path.join(dest_dir, filename)
    
    os.makedirs(dest_dir, exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(local_path):
        print(f"File {filename} already exists. Skipping download.")
        extract_file(filename, dest_dir)
        return

    print(f"Downloading {filename}...")
    print(f"Source: {url}")
    print(f"Destination: {local_path}")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192 
        
        progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
        
        if total_size != 0 and progress_bar.n != total_size:
            print("ERROR: Download incomplete")
        else:
            print("Download complete!")
            extract_file(filename, dest_dir)
            
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

if __name__ == "__main__":
    print(f"Starting download and extraction of {len(SUBSETS_TO_DOWNLOAD)} file(s)...")
    for subset in SUBSETS_TO_DOWNLOAD:
        download_file(subset, DEST_DIR)
    
    print("\n" + "="*50)
    print("PROCESS COMPLETE")
    print("="*50)
    print(f"1. Files are in: {DEST_DIR}")
    print("2. Run: python data/openwebtext/prepare_manual.py")
