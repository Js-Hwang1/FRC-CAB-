# Generation Benchmark Root Cause Analysis

## Problem

CAB and H2O showing **identical perplexity** at every sparsity level:

```
Method   Sparsity   Perplexity
--------------------------------
dense    0.0        8.77
cab      0.0        79.29
h2o      0.0        79.29  <- Same as CAB!
cab      0.5        82.66
h2o      0.5        82.66  <- Same as CAB!
cab      0.9        212.67
h2o      0.9        212.67 <- Same as CAB!
```

## Root Cause

Looking at `experiments/generation_benchmark/driver.py:402-406`:

```python
if method != 'dense' and input_ids.shape[1] > int(1024 * keep_ratio):
    # For methods with eviction, truncate to simulate cache pruning
    # Keep most recent tokens
    max_len = int(1024 * keep_ratio)
    input_ids = input_ids[:, -max_len:]  # SAME FOR BOTH!
```

**Both CAB and H2O use the exact same truncation logic!**

The benchmark is:
1. Using `use_cache=False` (no KV cache)
2. For CAB/H2O: Just truncating to keep recent tokens (identical strategy!)
3. For Dense: Using SDPA attention, full sequences

So it's comparing:
- **Dense**: SDPA attention on full sequences
- **CAB**: Flash Attention on truncated sequences (keep recent)
- **H2O**: Flash Attention on truncated sequences (keep recent)

This explains:
- ✅ CAB == H2O (same truncation)
- ✅ CAB/H2O at 0% != Dense at 0% (Flash vs SDPA numerical differences)

## What Should Happen

### H2O Eviction Policy
- 20% recent tokens
- 80% highest cumulative attention scores
- Two components

### CAB Eviction Policy
- 30% local (recent)
- 20% bridges (connectors)
- 50% important (high attention)
- Three components

These should produce DIFFERENT results!

## Fix Required

The `evaluate_perplexity()` function needs to:

1. **Get cumulative importance scores** from Flash Attention
2. **Apply actual eviction policies** to select which tokens to keep:
   - H2O: Select based on recency (20%) + importance (80%)
   - CAB: Use `ThreeComponentEvictionPolicy.select_indices()`
3. **Simulate eviction** by only using selected token positions

Even though we use `use_cache=False` for evaluation simplicity, we can still:
- Compute attention scores during context processing
- Use eviction policies to select which token indices to keep
- Truncate input_ids to only include selected tokens
- This simulates what the evicted cache would contain

## Implementation Plan

### Step 1: Add H2O eviction function
Create `cab_attention/eviction/h2o.py`:

```python
def h2o_select_indices(
    cache_len: int,
    keep_size: int,
    importance_scores: torch.Tensor,
    local_ratio: float = 0.2,
) -> torch.Tensor:
    """H2O eviction: keep recent + high importance."""
    local_size = max(1, int(keep_size * local_ratio))
    important_size = keep_size - local_size

    # Local tokens
    local_indices = torch.arange(cache_len - local_size, cache_len)

    # Important tokens (exclude already selected)
    mask = torch.ones(cache_len, dtype=torch.bool)
    mask[local_indices] = False
    candidate_scores = importance_scores.clone()
    candidate_scores[~mask] = -float('inf')
    important_indices = candidate_scores.topk(important_size).indices

    # Combine and sort
    keep_indices = torch.cat([local_indices, important_indices])
    keep_indices = torch.sort(keep_indices)[0]
    return keep_indices
```

### Step 2: Modify `evaluate_perplexity()`

Replace lines 402-406 with:

```python
if method != 'dense' and input_ids.shape[1] > int(1024 * keep_ratio):
    keep_size = int(1024 * keep_ratio)
    cache_len = input_ids.shape[1]

    # Get importance scores from Flash Attention
    importance_scores = self._get_cumulative_scores()

    if method == 'h2o':
        # H2O: 20% recent + 80% important
        from cab_attention.eviction.h2o import h2o_select_indices
        keep_indices = h2o_select_indices(
            cache_len, keep_size, importance_scores, local_ratio=0.2
        )
    elif method == 'cab':
        # CAB: 30% local + 20% bridges + 50% important
        from cab_attention.eviction.policy import ThreeComponentEvictionPolicy, EvictionConfig
        policy = ThreeComponentEvictionPolicy(EvictionConfig())
        keep_indices, _ = policy.select_indices(
            cache_len, keep_size, importance_scores
        )

    # Only keep selected tokens
    input_ids = input_ids[:, keep_indices]
```

### Step 3: Add method to get cumulative scores

```python
def _get_cumulative_scores(self) -> torch.Tensor:
    """Get cumulative attention scores from Flash Attention modules."""
    if not self.use_flash_attention:
        return None

    # Get scores from first layer (they're accumulated across all layers)
    first_layer = self.model.model.layers[0].self_attn
    if hasattr(first_layer, 'cumulative_scores'):
        return first_layer.cumulative_scores.clone()
    return None
```

## Expected Results After Fix

With proper eviction policies:
- Dense at 0%: ~8.8 (baseline)
- CAB at 0%: ~8.8 (no eviction, should match Dense if Flash Attention is correct)
- H2O at 0%: ~8.8 (no eviction, should match Dense)
- CAB at 50%: ~10-12 (moderate degradation)
- H2O at 50%: ~11-13 (worse than CAB)
- CAB at 90%: ~15-20 (significant degradation, but bridges help)
- H2O at 90%: ~20-25 (worse than CAB, no bridge tokens)

**CAB should outperform H2O** at high sparsity because bridge tokens preserve semantic connections.

## Why This Matters

Without this fix, the benchmark is meaningless - it's just comparing Flash Attention vs SDPA on truncated sequences, not actually evaluating eviction policies.
