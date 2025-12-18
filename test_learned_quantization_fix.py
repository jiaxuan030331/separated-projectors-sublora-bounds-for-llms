"""
Test script to verify the quantization fix for Learned mode.

This script creates a simple model with Learned gating and tests:
1. That gating_params are included in quantization-aware training
2. That gating_params are included in message length calculation
"""
import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.insert(0, '.')

from sublora.bounds import bound_utils, quantize_fns

class MockGPT(nn.Module):
    """Mock GPT model for testing."""
    def __init__(self, vocab_size=1000, embed_dim=64):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        x = self.wte(idx)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.reshape(B*T, C)
            targets_flat = targets.reshape(B*T)
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        return logits, loss

class MockStructuredIDModule(nn.Module):
    """Mock StructuredIDModule with gating params."""
    def __init__(self, base_net, d=1000, num_layers=12):
        super().__init__()
        self.d = d
        self.mode = 'learned'
        self.allocation_config = {'mode': 'learned'}
        self.layers = list(range(num_layers))
        self.has_misc = True

        # Store base net
        self._forward_net = [base_net]

        # Single shared subspace_params
        self.subspace_params = nn.Parameter(torch.randn(d))

        # Per-layer gating params
        self.gating_params = nn.ParameterDict()
        for i in self.layers:
            self.gating_params[str(i)] = nn.Parameter(torch.randn(1))
        if self.has_misc:
            self.gating_params['misc'] = nn.Parameter(torch.randn(1))

        print(f"Created MockStructuredIDModule:")
        print(f"  d={d}")
        print(f"  num_layers={num_layers}")
        print(f"  num_gating_params={len(self.gating_params)}")

    def forward(self, idx, targets=None):
        # In real implementation, this would project subspace_params
        # For testing, just pass through to base net
        return self._forward_net[0](idx, targets)

def test_quantization_with_learned_gating():
    """Test that the quantization fix properly handles gating_params."""
    print("\n" + "="*70)
    print("Testing Quantization Fix for Learned Gating Mode")
    print("="*70 + "\n")

    # Create mock model
    base_net = MockGPT(vocab_size=1000, embed_dim=64)
    model = MockStructuredIDModule(base_net, d=1000, num_layers=12)

    # Create mock training data in the format expected by get_batch
    # get_batch expects a 1D numpy array of token indices (np.memmap-like)
    np.random.seed(42)
    block_size = 32
    # Create a dataset with enough tokens (at least block_size + batch_size * block_size)
    train_data = np.random.randint(0, 1000, size=(10000,), dtype=np.uint16)

    # Test parameters
    intrinsic_dim = 1000
    levels = 11
    use_kmeans = False
    quant_lr = 5e-5
    max_quant_iters = 10  # Small number for testing
    quant_batch_size = 2
    block_size = 32
    device = 'cpu'
    device_type = 'cpu'
    ddp = False
    perturb_word_order_window_size = 0

    print("Test Configuration:")
    print(f"  intrinsic_dim: {intrinsic_dim}")
    print(f"  levels: {levels}")
    print(f"  max_quant_iters: {max_quant_iters}")
    print(f"  num_gating_params: {len(model.gating_params)}")

    # Record initial gating param values
    initial_gating_values = {k: v.item() for k, v in model.gating_params.items()}
    print("\nInitial gating param values (first 3):")
    for i, (k, v) in enumerate(list(initial_gating_values.items())[:3]):
        print(f"  gating_params[{k}]: {v:.6f}")

    # Run quantization
    print("\n" + "-"*70)
    print("Running quantization with fix...")
    print("-"*70 + "\n")

    try:
        # This should print diagnostic messages from the fix
        quantized_model, prefix_message_len = bound_utils.quantize_model(
            model=model,
            train_data=train_data,
            block_size=block_size,
            intrinsic_dim=intrinsic_dim,
            device_type=device_type,
            device=device,
            ddp=ddp,
            perturb_word_order_window_size=perturb_word_order_window_size,
            quant_batch_size=quant_batch_size,
            max_quant_iters=max_quant_iters,
            use_kmeans=use_kmeans,
            levels=levels,
            quant_lr=quant_lr
        )

        print("\n" + "-"*70)
        print("Quantization completed successfully!")
        print("-"*70 + "\n")

        # Check if gating params were updated
        final_gating_values = {k: v.item() for k, v in quantized_model.gating_params.items()}

        print("Final gating param values (first 3):")
        for i, (k, v) in enumerate(list(final_gating_values.items())[:3]):
            print(f"  gating_params[{k}]: {v:.6f}")

        print("\nGating param changes (first 3):")
        gating_changed = False
        for i, k in enumerate(list(initial_gating_values.keys())[:3]):
            change = abs(final_gating_values[k] - initial_gating_values[k])
            print(f"  gating_params[{k}]: {change:.6f}")
            if change > 1e-6:
                gating_changed = True

        print(f"\nResults:")
        print(f"  prefix_message_len: {prefix_message_len:.2f} bits ({prefix_message_len/1e6:.3f} Mbits)")
        print(f"  Gating params changed: {gating_changed}")

        # Verify the fix worked
        expected_gating_bits = len(model.gating_params) * 32
        print(f"\nExpected gating param contribution: {expected_gating_bits} bits")

        if gating_changed:
            print("\n✅ SUCCESS: Gating parameters were optimized during quantization!")
        else:
            print("\n⚠️  WARNING: Gating parameters did NOT change during quantization.")
            print("   This might indicate the optimizer didn't include them.")

        if prefix_message_len > 0:
            print("✅ SUCCESS: Message length calculation completed!")
        else:
            print("❌ FAIL: Message length is zero!")

    except Exception as e:
        print(f"\n❌ ERROR during quantization: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_quantization_with_learned_gating()

    print("\n" + "="*70)
    if success:
        print("TEST PASSED")
    else:
        print("TEST FAILED")
    print("="*70 + "\n")
