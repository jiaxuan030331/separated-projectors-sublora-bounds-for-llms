#!/usr/bin/env python
"""
Test script to verify the learned_shared mode works correctly.
This mode shares the same d-dimensional subspace across all layers,
but each layer has its own learnable gating parameter.
"""

import torch
import sys
sys.path.insert(0, '/workspace/separated-projectors-sublora-bounds-for-llms')

from sublora.nn.model import GPT, GPTConfig
from sublora.nn.projectors import create_intrinsic_model, StructuredIDModule

def test_learned_shared_mode():
    print("=" * 60)
    print("Testing learned_shared mode")
    print("=" * 60)
    
    # Create a small GPT model for testing
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=4,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        use_lora=True,
        attention_linear_use_lora=True,
        attention_linear_lora_r=4,
        linear_head_lora_r=4,
        linear_head_enable_lora=True,
        lora_alpha=32,
        lora_dropout=0.0,
    )
    
    base_model = GPT(config)
    
    # Test learned_shared mode
    print("\n--- Creating model with learned_shared mode ---")
    allocation_config = {
        'mode': 'learned_shared',
        'ratio': 0.5,
        'gating_scale': 10.0,
    }
    
    intrinsic_dim = 1000
    
    model = create_intrinsic_model(
        base_model,
        intrinsic_mode='dense',
        intrinsic_dim=intrinsic_dim,
        seed=42,
        allocation_config=allocation_config,
    )
    
    print(f"\nModel type: {type(model)}")
    print(f"Mode: {model.mode}")
    print(f"Intrinsic dim (d): {model.d}")
    print(f"subspace_params shape: {model.subspace_params.shape}")
    
    # Check projectors
    print(f"\nProjectors created: {list(model.projectors.keys())}")
    
    # Check that all projectors expect input of size d (not d_alloc)
    print("\n--- Verifying projector input dimensions ---")
    for name, proj in model.projectors.items():
        try:
            proj_d = proj.shape[-1]
        except:
            proj_d = getattr(proj, 'P', None)
            if proj_d is not None and hasattr(proj_d, 'shape'):
                proj_d = proj_d.shape[1]
            else:
                proj_d = "unknown"
        print(f"  {name}: input_dim = {proj_d}")
        if proj_d != "unknown":
            assert proj_d == intrinsic_dim, f"Expected {intrinsic_dim}, got {proj_d}"
    
    # Check gating params
    print(f"\nGating params: {list(model.gating_params.keys())}")
    for name, param in model.gating_params.items():
        print(f"  {name}: shape={param.shape}, value={param.item():.4f}")
    
    # Test forward pass
    print("\n--- Testing forward pass ---")
    model.eval()
    x = torch.randint(0, 100, (2, 32))  # batch_size=2, seq_len=32
    
    with torch.no_grad():
        logits, loss = model(x, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    
    # Test backward pass
    print("\n--- Testing backward pass ---")
    model.train()
    logits, loss = model(x, x)
    loss.backward()
    
    # Check that gradients flow to subspace_params and gating_params
    print(f"subspace_params grad norm: {model.subspace_params.grad.norm().item():.6f}")
    for name, param in model.gating_params.items():
        if param.grad is not None:
            print(f"gating_params[{name}] grad: {param.grad.item():.6f}")
    
    print("\n" + "=" * 60)
    print("learned_shared mode test PASSED!")
    print("=" * 60)
    
    return model


def compare_modes():
    """Compare learned vs learned_shared modes"""
    print("\n" + "=" * 60)
    print("Comparing learned vs learned_shared modes")
    print("=" * 60)
    
    config = GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=4,
        n_head=4,
        n_embd=64,
        dropout=0.0,
        bias=False,
        use_lora=True,
        attention_linear_use_lora=True,
        attention_linear_lora_r=4,
        linear_head_lora_r=4,
        linear_head_enable_lora=True,
        lora_alpha=32,
        lora_dropout=0.0,
    )
    
    intrinsic_dim = 1000
    
    # Create learned mode model
    print("\n--- learned mode ---")
    base_model_1 = GPT(config)
    model_learned = create_intrinsic_model(
        base_model_1,
        intrinsic_mode='dense',
        intrinsic_dim=intrinsic_dim,
        seed=42,
        allocation_config={'mode': 'learned', 'gating_scale': 10.0},
    )
    
    print(f"Mode: {model_learned.mode}")
    print(f"d_alloc_order (per-layer allocation): {model_learned.d_alloc_order}")
    print(f"Sum of allocations: {sum(model_learned.d_alloc_order)}")
    
    # Check projector dimensions
    for name, proj in model_learned.projectors.items():
        try:
            proj_d = proj.shape[-1]
        except:
            proj_d = "unknown"
        print(f"  Projector {name}: input_dim = {proj_d}")
    
    # Create learned_shared mode model
    print("\n--- learned_shared mode ---")
    base_model_2 = GPT(config)
    model_shared = create_intrinsic_model(
        base_model_2,
        intrinsic_mode='dense',
        intrinsic_dim=intrinsic_dim,
        seed=42,
        allocation_config={'mode': 'learned_shared', 'gating_scale': 10.0},
    )
    
    print(f"Mode: {model_shared.mode}")
    print(f"All projectors use full d = {model_shared.d}")
    
    # Check projector dimensions
    for name, proj in model_shared.projectors.items():
        try:
            proj_d = proj.shape[-1]
        except:
            proj_d = "unknown"
        print(f"  Projector {name}: input_dim = {proj_d}")
    
    print("\n--- Key Difference ---")
    print("learned: subspace_params is SLICED per layer (each layer uses d_alloc dims)")
    print("learned_shared: ALL layers use the FULL subspace_params (d dims)")
    print("\nBoth modes have:")
    print("  - Per-layer gating parameters g_ℓ → γ_ℓ = sigmoid(g_ℓ)")
    print("  - Soft mask m_ℓ(i) = sigmoid(scale * (γ_ℓ * d - i))")
    print("  - A/B split: α_B = α * m, α_A = α * (1-m)")
    
    print("\n" + "=" * 60)
    print("Comparison complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_learned_shared_mode()
    compare_modes()
