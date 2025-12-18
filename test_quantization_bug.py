"""
Test script to reproduce the quantization bug.

This creates a simple StructuredIDModule and tests the quantization flow.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '.')

from sublora.bounds import quantize_fns as quantize

class SimpleStructuredIDModule(nn.Module):
    """Simplified version of StructuredIDModule for testing."""
    def __init__(self, d=20000, num_layers=12, has_misc=True):
        super().__init__()
        self.d = d
        self.mode = 'learned'
        self.allocation_config = {'mode': 'learned'}
        self.layers = list(range(num_layers))
        self.has_misc = has_misc

        # Single shared subspace_params
        self.subspace_params = nn.Parameter(torch.randn(d))

        # Per-layer gating params
        self.gating_params = nn.ParameterDict()
        for i in self.layers:
            self.gating_params[str(i)] = nn.Parameter(torch.randn(1))
        if self.has_misc:
            self.gating_params['misc'] = nn.Parameter(torch.randn(1))

        print(f"Created SimpleStructuredIDModule:")
        print(f"  d={d}")
        print(f"  num_layers={num_layers}")
        print(f"  has_misc={has_misc}")
        print(f"  subspace_params.shape={self.subspace_params.shape}")
        print(f"  num_gating_params={len(self.gating_params)}")

def test_quantization(d=20000, levels=11, use_kmeans=False):
    """Test the quantization flow."""
    print(f"\n{'='*60}")
    print(f"Testing quantization with d={d}, levels={levels}")
    print(f"{'='*60}\n")

    # Create model
    model = SimpleStructuredIDModule(d=d, num_layers=12, has_misc=True)

    # Simulate what happens in quantize_model
    print("\n--- Extracting subspace_params ---")
    vector = model.subspace_params.cpu().data.numpy()
    print(f"vector.shape: {vector.shape}")
    print(f"vector.size: {vector.size}")
    print(f"len(vector): {len(vector)}")

    # Check if this is correct
    assert len(vector) == d, f"ERROR: Expected len(vector)={d}, got {len(vector)}"
    print("✓ Vector size is correct")

    # Quantize
    print("\n--- Quantizing ---")
    quantized_vec, message_len = quantize.quantize_vector(vector, levels=levels, use_kmeans=use_kmeans)

    print(f"\nQuantization results:")
    print(f"  quantized_vec.shape: {quantized_vec.shape}")
    print(f"  message_len: {message_len:.2f} bits ({message_len/1e6:.3f} Mbits)")

    # Calculate expected message length
    # For d=20000 params with levels=11, we expect roughly:
    #   - Entropy ~ 3-4 bits per param (depending on distribution)
    #   - Codebook: 11 * 16 bits = 176 bits
    #   - Probabilities: 11 * ceil(log2(20000)) ~ 11 * 15 = 165 bits
    #   - Total: ~3.5 * 20000 + 176 + 165 ~ 70k-80k bits
    expected_bits_per_param = 4.0  # conservative estimate
    expected_message_len = d * expected_bits_per_param + 11 * 16 + 11 * np.ceil(np.log2(d))

    print(f"\nExpected message_len: ~{expected_message_len:.2f} bits ({expected_message_len/1e6:.3f} Mbits)")
    print(f"Actual / Expected ratio: {message_len / expected_message_len:.2f}x")

    # Check for 13x inflation (bug indicator)
    num_sharing_groups = 13  # 12 layers + 1 misc
    if message_len > expected_message_len * (num_sharing_groups * 0.8):
        print(f"\n⚠️  WARNING: message_len seems {num_sharing_groups}x too high!")
        print(f"   This suggests the subspace might be counted {num_sharing_groups} times")
        print(f"   Corrected message_len: {message_len / num_sharing_groups:.2f} bits")
    else:
        print(f"\n✓ message_len seems reasonable")

    # Also check what state_dict contains
    print("\n--- Checking state_dict ---")
    state_dict = model.state_dict()
    print(f"state_dict keys: {list(state_dict.keys())}")
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters in state_dict: {total_params}")
    print(f"  subspace_params: {state_dict['subspace_params'].numel()}")
    print(f"  gating_params: {sum(state_dict[k].numel() for k in state_dict if 'gating' in k)}")

    return message_len, expected_message_len

if __name__ == "__main__":
    # Test with d=20000 (from your example)
    message_len, expected = test_quantization(d=20000, levels=11, use_kmeans=False)

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Actual message_len: {message_len:.2f} bits ({message_len/1e6:.3f} Mbits)")
    print(f"Expected message_len: {expected:.2f} bits ({expected/1e6:.3f} Mbits)")
    print(f"Ratio: {message_len/expected:.2f}x")

    if message_len / expected > 10:
        print("\n❌ BUG CONFIRMED: message_len is ~13x too high!")
    else:
        print("\n✓ No bug detected in this test")
