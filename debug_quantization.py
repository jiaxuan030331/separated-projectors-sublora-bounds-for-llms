"""
Debug script to understand the quantization issue in learned_shared mode.

This script loads a checkpoint and analyzes the subspace_params to understand
why the quantization loss is higher than expected.
"""
import torch
import numpy as np
import sys
import argparse

def analyze_checkpoint(checkpoint_path):
    """Load checkpoint and analyze subspace_params."""
    print(f"Loading checkpoint from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get the state dict
    state_dict = checkpoint.get('raw_model', checkpoint)

    print("\n=== Checkpoint Keys ===")
    for key in sorted(state_dict.keys()):
        tensor = state_dict[key]
        if isinstance(tensor, torch.Tensor):
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}, numel={tensor.numel()}")

    # Check if subspace_params exists
    if 'subspace_params' in state_dict:
        subspace_params = state_dict['subspace_params']
        print(f"\n=== Subspace Params Analysis ===")
        print(f"  Shape: {subspace_params.shape}")
        print(f"  Numel: {subspace_params.numel()}")
        print(f"  Dtype: {subspace_params.dtype}")
        print(f"  Min: {subspace_params.min().item():.6f}")
        print(f"  Max: {subspace_params.max().item():.6f}")
        print(f"  Mean: {subspace_params.mean().item():.6f}")
        print(f"  Std: {subspace_params.std().item():.6f}")

        # Check if this looks like it was duplicated
        vec = subspace_params.numpy()
        unique_values = np.unique(vec)
        print(f"  Unique values: {len(unique_values)}")

    else:
        print("\n ERROR: No subspace_params found in checkpoint!")

    # Check for gating params (indicates learned mode)
    gating_keys = [k for k in state_dict.keys() if 'gating_params' in k]
    if gating_keys:
        print(f"\n=== Gating Params (Learned Mode) ===")
        print(f"  Found {len(gating_keys)} gating parameters:")
        for key in sorted(gating_keys):
            tensor = state_dict[key]
            print(f"    {key}: {tensor.item():.6f}")

    # Get config
    config = checkpoint.get('config', checkpoint.get('model_args', {}))
    print(f"\n=== Config ===")
    if isinstance(config, dict):
        for key in ['intrinsic_dim', 'allocation_mode', 'allocation_ratio', 'n_layer']:
            if key in config or (isinstance(config.get('sublora'), dict) and key in config['sublora']):
                sublora_config = config.get('sublora', config)
                value = sublora_config.get(key, 'N/A')
                print(f"  {key}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Debug quantization issue')
    parser.add_argument('checkpoint_path', type=str, help='Path to checkpoint file')
    args = parser.parse_args()

    analyze_checkpoint(args.checkpoint_path)
