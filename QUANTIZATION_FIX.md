# Quantization Bug Fix for Learned Gating Mode

## Problem Summary

The Learned gating mode exhibited significantly higher quantization loss (~1.83 BPD) compared to other allocation modes (Uniform, Fixed, A-heavy, B-heavy which had ~0.1-0.35 BPD quantization loss). This represents a 5-18x performance degradation after quantization.

## Root Cause Analysis

The issue had two components:

### 1. Gating Parameters Not Optimized During Quantization-Aware Training

**Location**: [sublora/bounds/bound_utils.py:183](sublora/bounds/bound_utils.py#L183) (original code)

**Problem**: When performing quantization-aware training (`max_quant_iters > 0`), the optimizer only included:
```python
optim = SGD([qw.subspace_params, qw.centroids], lr=quant_lr, momentum=0.9)
```

This meant that:
- The `subspace_params` (shared intrinsic dimension vector) was being optimized to work with quantization
- The `gating_params` (per-layer learned allocation parameters) were **NOT** being updated

Since the `gating_params` were learned to work with the *original* (unquantized) subspace values, after quantization they became misaligned with the new quantized subspace values. This caused the model to perform poorly after quantization in Learned mode.

**Why this only affected Learned mode**:
- Uniform/Fixed/B-heavy/A-heavy modes don't have `gating_params`
- Only Learned mode uses per-layer learnable gating parameters that control how the shared subspace is allocated

### 2. Gating Parameters Not Included in Compression Bound

**Location**: [sublora/bounds/bound_utils.py:268](sublora/bounds/bound_utils.py#L268) (original code)

**Problem**: The message length calculation only included the quantized `subspace_params`:
```python
prefix_message_len = message_len + 2 * np.log2(message_len)
```

For a PAC-Bayes bound, we must account for ALL information needed to specify the model. The `gating_params` (13 float32 values = 416 bits for 12 layers + 1 misc) were not included in the bound computation.

While 416 bits is small compared to the subspace message length (~50k-250k bits), it should still be accounted for to maintain theoretical soundness.

## The Fix

### Fix 1: Include Gating Parameters in Quantization-Aware Training

**File**: [sublora/bounds/bound_utils.py](sublora/bounds/bound_utils.py#L183-L193)

**Change**: Modified the optimizer to include gating parameters when present:
```python
# Build list of parameters to optimize
params_to_optimize = [qw.subspace_params, qw.centroids]

# If model has gating_params (learned mode), include them
if is_learned_shared and hasattr(qw._forward_net[0], 'gating_params'):
    for gating_param in qw._forward_net[0].gating_params.values():
        params_to_optimize.append(gating_param)
    print(f"[QuantizeFix] Including {len(qw._forward_net[0].gating_params)} gating params in quantization-aware training")

optim = SGD(params_to_optimize, lr=quant_lr, momentum=0.9)
```

**Effect**: During quantization-aware training, the gating parameters are now adjusted to work optimally with the quantized subspace, reducing the quantization loss.

### Fix 2: Include Gating Parameters in Message Length

**File**: [sublora/bounds/bound_utils.py](sublora/bounds/bound_utils.py#L256-L289)

**Change**: Added gating parameter bit cost to the total message length:
```python
# Initialize gating message length
gating_message_len = 0

# After quantizing subspace_params, check if we need to account for gating_params
if intrinsic_dim > 0:
    module = model.module if ddp else model
    module.subspace_params.data = torch.tensor(quantized_vec).float().to(device)

    if is_learned_shared and hasattr(module, 'gating_params'):
        num_gating_params = len(module.gating_params)
        # Each gating param is a scalar float32 (32 bits)
        gating_message_len = num_gating_params * 32
        print(f"[QuantizeFix] Including {num_gating_params} gating params in message length")
        print(f"[QuantizeFix]   Gating params: {gating_message_len} bits")

# Include gating params in the total message length
total_message_len = message_len + gating_message_len
prefix_message_len = total_message_len + 2 * np.log2(total_message_len)
```

**Effect**: The PAC-Bayes bound now correctly accounts for all model parameters, maintaining theoretical soundness.

## Expected Outcome

After this fix, the Learned gating mode should have:

1. **Reduced Quantization Loss**: The quantization loss should decrease from ~1.83 BPD to a level comparable with other modes (~0.1-0.35 BPD), because the gating parameters are now adapted to work with the quantized subspace.

2. **Correct Compression Bound**: The complexity term in the bound will increase slightly (by 416 bits for 13 gating params), but this is negligible compared to the subspace compression (~50k-250k bits) and represents the true cost of the model.

3. **Better Overall Bound**: Even though the complexity term increases slightly, the reduced empirical risk (due to lower quantization loss) should result in a better overall generalization bound for Learned mode.

## Testing

To verify the fix works:

1. Run quantization with `max_quant_iters > 0` on a Learned mode checkpoint
2. Check the output logs for:
   - `[QuantizeFix] Including X gating params in quantization-aware training`
   - `[QuantizeFix] Including X gating params in message length`
3. Compare the "Quantization Loss" before and after the fix - it should be significantly lower
4. Verify the bound computation includes the gating parameter cost

## Technical Notes

### Why Not Quantize Gating Parameters?

We chose to keep gating parameters as full-precision float32 and optimize them during quantization-aware training, rather than quantizing them, because:

1. **Small number**: Only 13 parameters (12 layers + 1 misc)
2. **Tiny cost**: 13 * 32 = 416 bits is negligible compared to subspace (~50k-250k bits)
3. **Better performance**: Full-precision gating parameters can adapt more precisely to the quantized subspace
4. **Simpler implementation**: No need to handle mixed quantization levels

### Why This Bug Was Hard to Detect

1. **Mode-specific**: Only affected Learned mode, other modes worked fine
2. **Silent failure**: The quantization code ran without errors, but produced poor results
3. **No explicit iteration**: The code didn't iterate "once per layer" - instead, it had a single shared subspace but forgot to update the per-layer gating parameters
4. **Misleading symptoms**: The high quantization loss suggested "overcounting," but the real issue was parameter misalignment

## Files Modified

- [sublora/bounds/bound_utils.py](sublora/bounds/bound_utils.py):
  - Lines 183-193: Include gating_params in quantization-aware training optimizer
  - Lines 256-289: Include gating_params in message length calculation
