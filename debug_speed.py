#!/usr/bin/env python
"""Quick diagnostic script to verify projection speed on GPU."""
import torch
import time
import sys

# Add sublora to path
sys.path.insert(0, '.')

from sublora.nn.projectors import RoundedDoubleKronQR, LazyRandom, LazyRandomQR
from sublora.nn.linear_operator_base import LazyKron, LazyDirectSum, LazyPerm, Lazy

def benchmark(name, proj, d, device, warmup=3, runs=10):
    """Benchmark a projector."""
    v = torch.randn(d, device=device, dtype=torch.bfloat16)
    
    # Warmup (important - this is where caching should happen)
    for _ in range(warmup):
        _ = proj @ v
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs):
        result = proj @ v
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / runs * 1000
    
    print(f"{name}: {elapsed:.2f} ms/iter, result on {result.device}")
    return elapsed

def main():
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device('cuda')
    print(f"Testing on: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Model dimensions (GPT-2 style)
    D = 300_000  # ~300K trainable params (typical for small GPT)
    d = 1000     # intrinsic dimension
    
    params = [torch.randn(100, 100) for _ in range(30)]  # dummy params
    names = [f"layer.{i}.weight" for i in range(30)]
    
    print(f"Testing projection: D={D:,}, d={d}")
    print("-" * 50)
    
    # Test RoundedDoubleKronQR
    print("\n1. RoundedDoubleKronQR (the main projector used in training):")
    proj = RoundedDoubleKronQR(D, d, params, names, order=2, seed=42)
    t1 = benchmark("  RoundedDoubleKronQR", proj, d, device)
    
    # Test again to verify caching works
    print("\n2. Same projector, second run (should be cached):")
    t2 = benchmark("  RoundedDoubleKronQR (cached)", proj, d, device)
    
    # Test LazyRandom for comparison
    print("\n3. LazyRandom (baseline comparison):")
    proj_lazy = LazyRandom(D, d, params, names, seed=42)
    t3 = benchmark("  LazyRandom", proj_lazy, d, device)
    
    # Test LazyRandomQR
    print("\n4. Single LazyRandomQR (component of RoundedDoubleKronQR):")
    proj_qr = LazyRandomQR(547, 31, params, names, seed=42)
    t4 = benchmark("  LazyRandomQR(547,31)", proj_qr, 31, device, runs=100)
    
    print("\n" + "=" * 50)
    print("DIAGNOSIS:")
    if t1 > 100:
        print(f"❌ SLOW: RoundedDoubleKronQR taking {t1:.0f}ms (should be <5ms)")
        print("   The fix was NOT applied correctly!")
        print("   Check that projectors.py and linear_operator_base.py are updated.")
    elif t1 > 10:
        print(f"⚠️ MODERATE: RoundedDoubleKronQR taking {t1:.0f}ms")
        print("   Some caching may not be working.")
    else:
        print(f"✓ FAST: RoundedDoubleKronQR taking {t1:.1f}ms")
        print("   Caching is working correctly!")
    
    if t2 > t1 * 1.5:
        print(f"⚠️ Caching might not be working (2nd run slower than 1st)")
    
    print("\n" + "=" * 50)
    print("Expected values on A100:")
    print("  RoundedDoubleKronQR: < 5ms")
    print("  LazyRandom: < 10ms") 
    print("  LazyRandomQR(547,31): < 0.1ms")

if __name__ == "__main__":
    main()
