# Position IDs Fix - Implementation Summary

## Problem Identified

**Root Cause**: CAB and H2O were generating corrupted outputs (gibberish) because position IDs were not being tracked after KV cache eviction.

**Symptoms**:
- CAB outputs: "To be in) in) 1-111", "The0. 33. 0. 1. Constraints"
- H2O outputs: "To9555500000,0000 ;"
- F1 scores: 0% (complete failure)

**Why This Happened**:
1. During prefill: Model processes 1000 tokens at positions [0...999]
2. After 90% eviction: KV cache shrinks to 100 tokens (indices [900...999])
3. During generation: Model was called WITHOUT `position_ids` parameter
4. Model assumed: New tokens are at position = cache_length = 100
5. Reality: New tokens should be at position = 1000, 1001, 1002...
6. Result: RoPE embeddings completely misaligned → gibberish

## Solution Implemented

### Files Modified

**[experiments/longbench_qa/runner.py](experiments/longbench_qa/runner.py)**

#### Change 1: Track Current Position (line 584)
```python
# Track current position for RoPE after eviction
current_position = inputs['input_ids'].shape[1]
```

#### Change 2: Pass Position IDs to Model (lines 626-628)
```python
# Pass position_ids to maintain correct RoPE after eviction
device = next_token.device
position_ids = torch.tensor([[current_position]], device=device)
```

#### Change 3: Include Position IDs in Forward Pass (line 634)
```python
outputs = self.model(
    input_ids=next_token,
    past_key_values=past_key_values,
    position_ids=position_ids,  # ← NEW: Maintain correct positions
    use_cache=True,
    return_dict=True,
    output_attentions=need_attention,
)
```

#### Change 4: Increment Position After Each Token (line 641)
```python
# Increment position for next token (regardless of eviction)
current_position += 1
```

## How This Fixes the Bug

**Before Fix**:
```
Prefill: [Token 0] [Token 1] ... [Token 999]
         Position 0 → 999

Evict to 10%: Keep tokens [900...999]
              Cache size = 100

Generate token 1000:
  position_ids = MISSING
  → Model assumes position = 100 (cache length)
  → RoPE uses cos/sin for position 100
  → Attention queries wrong positions
  → GIBBERISH OUTPUT
```

**After Fix**:
```
Prefill: [Token 0] [Token 1] ... [Token 999]
         Position 0 → 999

Evict to 10%: Keep tokens [900...999]
              Cache size = 100
              current_position = 1000 (tracked!)

Generate token 1000:
  position_ids = [1000] ✓
  → Model uses correct position 1000
  → RoPE uses cos/sin for position 1000
  → Attention aligns correctly
  → CLEAN OUTPUT
```

## Testing

### Option 1: Quick Validation Script

Run the validation script to test CAB/H2O on a single sample:

```bash
cd ~/FRC
git pull origin main
python test_position_ids_fix.py
```

**Expected Output**:
- CAB: Clean answer (not gibberish)
- H2O: Clean answer (not gibberish)
- Status: "✓ Position IDs fix is working!"

### Option 2: Full Benchmark Test

Test with 5 samples each:

```bash
cd ~/FRC
git pull origin main

# Test CAB
python -m experiments.longbench_qa.driver \
  --model Qwen/Qwen2.5-7B-Instruct \
  --methods cab \
  --datasets hotpotqa \
  --sparsity 0.9 \
  --max-samples 5

# Test H2O
python -m experiments.longbench_qa.driver \
  --model Qwen/Qwen2.5-7B-Instruct \
  --methods h2o \
  --datasets hotpotqa \
  --sparsity 0.9 \
  --max-samples 5
```

**Expected Results**:
- CAB F1: ~30-50% (was 0% before fix)
- H2O F1: ~30-50% (was 0% before fix)
- Predictions should be readable text, not gibberish

### Option 3: Full Validation

Once quick tests pass, run full benchmarks:

```bash
# All methods on HotPotQA (200 samples)
python -m experiments.longbench_qa.driver \
  --model Qwen/Qwen2.5-7B-Instruct \
  --methods dense cab h2o \
  --datasets hotpotqa \
  --sparsity 0.9

# All methods on NarrativeQA (200 samples)
python -m experiments.longbench_qa.driver \
  --model Qwen/Qwen2.5-7B-Instruct \
  --methods dense cab h2o \
  --datasets narrativeqa \
  --sparsity 0.9
```

## Commits

1. **[319479e](https://github.com/Js-Hwang1/FRC-CAB-/commit/319479e)**: Fix: Track position IDs after KV cache eviction
2. **[aad845f](https://github.com/Js-Hwang1/FRC-CAB-/commit/aad845f)**: Test: Add validation script for position IDs fix

## Related Documents

- [fix_position_ids.md](fix_position_ids.md): Detailed technical explanation of the bug
- [test_position_ids_fix.py](test_position_ids_fix.py): Quick validation script
- [debug_cab.py](debug_cab.py): Debug script used to identify the bug

## Status

✅ **Fix Implemented**: Position IDs tracking added to `_sparse_generate` method
⏳ **Testing Pending**: Waiting for remote server access to validate fix
📊 **Expected Outcome**: CAB/H2O should produce clean outputs with F1 > 0%

## Next Steps

1. Pull latest code: `git pull origin main`
2. Run quick validation: `python test_position_ids_fix.py`
3. If validation passes, run full benchmarks
4. Compare results: Dense vs CAB vs H2O on HotPotQA and NarrativeQA
