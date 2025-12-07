#!/usr/bin/env python3
"""Quick smoke test for StructuredIDModule learned gating and allocation.

This constructs a tiny toy network with `transformer.h.<i>` layers so the
layer-detection regex in the project code matches. It then instantiates
`StructuredIDModule` with `mode='learned'`, prints allocation maps,
steps the gating annealer, runs a forward pass, and prints gate probs
and alpha norms. Finally it constructs the optimizer via
`configure_optimizers` to show gating LR grouping.
"""
import os
import sys
import torch
import torch.nn as nn
from functools import partial

# Ensure repo root is on sys.path so `import sublora` works when running this script
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sublora.nn.projectors import StructuredIDModule, LazyRandom


class ToyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # names include 'lora_A' and 'lora_B' so _group_parameters picks them up
        self.lora_A = nn.Linear(4, 4)
        self.lora_B = nn.Linear(4, 4)

    def forward(self, x):
        # not used directly by StructuredIDModule, but keep for completeness
        return self.lora_A(x) + self.lora_B(x)


class ToyNet(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        # Put layers under `transformer.h` so names contain `.h.<i>.` as expected
        self.transformer = nn.Module()
        self.transformer.h = nn.ModuleList([ToyLayer() for _ in range(n_layers)])
        self.head = nn.Linear(4, 4)

    def forward(self, x):
        out = x
        for layer in self.transformer.h:
            out = layer(out)
        out = self.head(out)
        return out


def main():
    torch.manual_seed(0)

    net = ToyNet(n_layers=3)

    d = 24
    allocation_config = {
        'mode': 'learned',
        'gating_fraction': 0.25,
        'gating_init_std': 0.01,
        'gating_lr_mult': 5.0,
        'gating_anneal': {'start': 1.0, 'end': 8.0, 'total_steps': 5, 'mode': 'linear', 'auto_step': False},
    }

    module = StructuredIDModule(net, partial(LazyRandom, seed=137), dimension=d, allocation_config=allocation_config)

    print("--- Allocation ---")
    print("d_alloc_map:", module.d_alloc_map)
    print("d_alloc_order:", module.d_alloc_order)
    print("nominal d_per_layer:", module.d_per_layer)
    print("initial gating_scale:", module.get_gating_scale())

    print("\n--- Gating params (raw logits) ---")
    for k, v in module.gating_params.items():
        print(k, float(v.detach().cpu().item()))

    print("\n--- Annealing steps ---")
    for step in range(6):
        s = module.step_gating_anneal()
        print(f"step {step}: gating_scale={s}")

    print("\n--- Gate probs before forward ---")
    for k, v in module.gating_params.items():
        print(k, float(torch.sigmoid(v).item()))

    # Run a forward pass with a small random batch
    x = torch.randn(2, 4)
    out = module(x)
    print("\nForward OK. Output shape:", out.shape)

    # Print alpha norms per group (use d_alloc_order cumulative offsets)
    print("\n--- Alpha norms per group ---")
    offsets = []
    cur = 0
    for dval in module.d_alloc_order:
        offsets.append(cur)
        cur += int(dval)

    keys = list(module.d_alloc_map.keys())
    for i, key in enumerate(keys):
        start = offsets[i]
        end = start + int(module.d_alloc_order[i])
        alpha = module.subspace_params[start:end]
        print(f"{key}: slice [{start}:{end}] d={int(module.d_alloc_order[i])} norm={float(alpha.norm().item()):.6f} mean={float(alpha.mean().item()):.6f}")

    # Build optimizer to show gating param grouping
    optim = module.configure_optimizers(weight_decay=0.01, learning_rate=1e-3, betas=(0.9, 0.999), device_type='cpu', correct_bias=True, adam_epislon=1e-8, no_decay_bias=True)
    print("\n--- Optimizer groups ---")
    print("num groups:", len(optim.param_groups))
    for i, g in enumerate(optim.param_groups):
        lr = g.get('lr', 1e-3)
        wd = g.get('weight_decay', None)
        pcount = sum(p.numel() for p in g['params'])
        print(f"group {i}: lr={lr}, weight_decay={wd}, params={pcount}")

    print("\nSmoke test completed successfully.")


if __name__ == '__main__':
    main()
