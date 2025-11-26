
import sys
from unittest.mock import MagicMock
import torch
import torch.nn as nn

# Mock loralib
class MockLoraLinear(nn.Linear):
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0., **kwargs):
        super().__init__(in_features, out_features, **kwargs)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))

sys.modules['loralib'] = MagicMock()
sys.modules['loralib'].Linear = MockLoraLinear
sys.modules['loralib'].MergedLinear = MockLoraLinear # Simplified for now

import re
from sublora.nn.model import GPT, GPTConfig
from sublora.nn.projectors import StructuredIDModule, LazyRandom
from functools import partial

def test_structure():
    # Create a dummy config and model
    config = GPTConfig(
        n_layer=2, 
        n_head=4, 
        n_embd=128, 
        block_size=128,
        vocab_size=1000,
        use_lora=True,
        attention_linear_use_lora=True,
        attention_linear_lora_r=4
    )
    model = GPT(config)
    
    print("Model Parameter Names:")
    for n, p in model.named_parameters():
        print(n)
        
    print("\nTesting StructuredIDModule initialization...")
    
    allocation_config = {'mode': 'learned', 'ratio': 0.5}
    
    # Mock projector factory
    projector_factory = partial(LazyRandom, seed=42)
    
    # Initialize wrapper
    wrapped_model = StructuredIDModule(
        model, 
        projector_factory, 
        dimension=100, 
        allocation_config=allocation_config
    )
    
    print("\nWrapped Model Parameters:")
    for n, p in wrapped_model.named_parameters():
        print(f"{n}: {p.shape} (Requires Grad: {p.requires_grad})")

    print("\nGating Params keys:", wrapped_model.gating_params.keys())
    
    # Check regex logic
    print("\nRegex Check:")
    layers = set()
    for n, p in model.named_parameters():
        match = re.search(r'\.h\.(\d+)\.', n)
        if match:
            print(f"Matched: {n} -> Layer {match.group(1)}")
            layers.add(int(match.group(1)))
        else:
            print(f"No match: {n}")
            
    print(f"Layers found: {layers}")

if __name__ == "__main__":
    test_structure()
