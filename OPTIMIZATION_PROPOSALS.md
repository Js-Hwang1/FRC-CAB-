# CAB Performance Optimization Proposals

**Goal:** Reduce average eviction time from ~50ms to <10ms (5x speedup)

**Current Bottleneck:** Multiple passes over importance scores + topk operations

---

## Benchmark Analysis

### Current Performance (32 layers, N=512)
```
CAB Eviction:
  Average: 49.47ms
  Maximum: 144.07ms
  Target:  <10ms

Breakdown (estimated):
  - Median computation:     ~5ms
  - Distance calculation:   ~3ms
  - Bridge topk:           ~15ms
  - Importance topk:       ~15ms
  - Index filtering:        ~5ms
  - Multi-layer pruning:    ~7ms
```

### Target Performance
- **<10ms per eviction** for production viability
- Requires ~5x overall speedup

---

## Proposal 1: Fused Single-Pass Selection (High Impact)

### Problem
Current implementation makes **4 separate passes** over data:
1. Compute median on candidates
2. Compute distance from median (full array)
3. Topk for bridges
4. Topk for importance

### Solution
Fuse into **2-pass algorithm**:

```python
def select_indices_fused(self, cache_len, keep_size, importance_scores, device):
    """Fused two-pass eviction - 2-3x faster."""

    # Budgets
    local_budget = int(keep_size * self.config.local_ratio)
    bridge_budget = int(keep_size * self.config.bridge_ratio)
    importance_budget = keep_size - local_budget - bridge_budget

    # PASS 1: Local selection + masking (O(1) - just indexing)
    local_indices = torch.arange(cache_len - local_budget, cache_len, device=device)

    # Create candidate mask (exclude local)
    candidate_mask = torch.ones(cache_len, dtype=torch.bool, device=device)
    candidate_mask[local_indices] = False
    candidate_indices = torch.where(candidate_mask)[0]
    candidate_scores = importance_scores[candidate_indices]

    # PASS 2: Joint selection of importance + bridges using percentiles
    # Instead of median, use percentile ranges for stability
    num_candidates = len(candidate_scores)

    if num_candidates > 0:
        # Argsort once (O(N log N))
        sorted_idx = torch.argsort(candidate_scores, descending=True)

        # Top K = importance (already sorted)
        importance_idx = sorted_idx[:importance_budget]

        # Middle K = bridges (around median)
        # Select from middle third of distribution
        start = num_candidates // 3
        end = start + bridge_budget
        bridge_idx = sorted_idx[start:min(end, num_candidates)]

        # Map back to original indices
        importance_indices = candidate_indices[importance_idx]
        bridge_indices = candidate_indices[bridge_idx]
    else:
        importance_indices = torch.tensor([], dtype=torch.long, device=device)
        bridge_indices = torch.tensor([], dtype=torch.long, device=device)

    # Combine and sort once
    keep_indices = torch.cat([local_indices, importance_indices, bridge_indices])
    keep_indices = keep_indices.unique().sort().values

    return keep_indices
```

**Expected Speedup:** 2-3x (reduces passes from 4 to 2, single argsort)

**Complexity:** Still O(N log N), but with better constants

---

## Proposal 2: Adaptive Eviction Interval (Medium Impact)

### Problem
Current eviction interval is **fixed at 10 tokens**, leading to frequent evictions even when not needed.

### Solution
**Adaptive interval** based on cache utilization:

```python
def _compute_eviction_interval(self, cache_len, max_size):
    """Dynamically adjust eviction frequency."""
    utilization = cache_len / max_size

    if utilization < 0.8:
        return 20  # Evict less frequently
    elif utilization < 0.95:
        return 10  # Normal frequency
    else:
        return 5   # Evict more frequently when near capacity
```

**Benefits:**
- Fewer evictions when cache has room
- More evictions when approaching limit
- Reduces overhead by ~30-40% on average

**Expected Speedup:** 1.3-1.4x (fewer evictions overall)

---

## Proposal 3: Quantized Importance Scores (Low Impact, High Safety)

### Problem
Float32 scores have unnecessary precision for ranking.

### Solution
**Quantize to int16** for faster comparisons:

```python
def update(self, attention_weights):
    """Track with quantized scores."""
    position_scores = attention_weights.sum(dim=(0, 1, 2))

    # Quantize to int16 (0-32767 range)
    scale = 32767.0 / (position_scores.max() + 1e-8)
    position_scores_int = (position_scores * scale).to(torch.int16)

    if self.cumulative_scores is None:
        self.cumulative_scores = position_scores_int.to(torch.int32)
    else:
        self.cumulative_scores[:len(position_scores)] += position_scores_int.to(torch.int32)
```

**Benefits:**
- 2x memory bandwidth improvement
- Faster sorting/comparison on integers
- No loss of ranking accuracy

**Expected Speedup:** 1.2x (memory bandwidth limited operations)

---

## Proposal 4: In-Place Cache Pruning (Medium Impact)

### Problem
Current pruning creates **copies** of KV cache:
```python
self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_indices, :]
```

### Solution
**In-place gather** operation:

```python
def _evict_inplace(self, keep_indices):
    """Prune cache in-place - avoids copies."""
    for layer_idx in range(len(self.key_cache)):
        if self.key_cache[layer_idx] is not None:
            # In-place gather
            torch.index_select(
                self.key_cache[layer_idx],
                dim=2,
                index=keep_indices,
                out=self.key_cache[layer_idx][:, :, :len(keep_indices), :]
            )
            # Truncate to new size
            self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, :len(keep_indices), :]

            # Same for values
            torch.index_select(
                self.value_cache[layer_idx],
                dim=2,
                index=keep_indices,
                out=self.value_cache[layer_idx][:, :, :len(keep_indices), :]
            )
            self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, :len(keep_indices), :]
```

**Expected Speedup:** 1.5x (reduces memory allocation overhead)

**Caveat:** Need to benchmark - PyTorch might optimize the original already

---

## Proposal 5: Parallel Layer Pruning (Low Impact)

### Problem
Layers pruned sequentially:
```python
for layer_idx in range(len(self.key_cache)):
    # Prune layer_idx
```

### Solution
**Parallelize across layers** using torch.multiprocessing or batched operations:

```python
def _evict_parallel(self, keep_indices):
    """Prune all layers in parallel."""
    # Stack all layers
    all_keys = torch.stack([k for k in self.key_cache if k is not None])
    all_values = torch.stack([v for v in self.value_cache if v is not None])

    # Single batched index_select
    all_keys = all_keys[:, :, :, keep_indices, :]
    all_values = all_values[:, :, :, keep_indices, :]

    # Unstack
    for i, (k, v) in enumerate(zip(all_keys, all_values)):
        self.key_cache[i] = k
        self.value_cache[i] = v
```

**Expected Speedup:** 1.2x (better GPU utilization)

**Caveat:** May increase peak memory usage

---

## Proposal 6: Top-K Heap Optimization (Medium Impact)

### Problem
PyTorch `topk` is general-purpose, may have overhead for our use case.

### Solution
**Custom heap-based selection** for small K:

```python
def fast_topk_small(scores, k, largest=True):
    """Optimized topk for small k (<< N)."""
    if k > 100:
        return torch.topk(scores, k, largest=largest)

    # Use heap for small k (O(N log k) vs O(N log N))
    import heapq

    if largest:
        # Maintain min-heap of top-k values
        heap = []
        for idx, score in enumerate(scores):
            if len(heap) < k:
                heapq.heappush(heap, (score.item(), idx))
            elif score > heap[0][0]:
                heapq.heapreplace(heap, (score.item(), idx))

        indices = torch.tensor([idx for _, idx in heap], device=scores.device)
    else:
        # Maintain max-heap of bottom-k values
        heap = []
        for idx, score in enumerate(scores):
            if len(heap) < k:
                heapq.heappush(heap, (-score.item(), idx))
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, (-score.item(), idx))

        indices = torch.tensor([idx for _, idx in heap], device=scores.device)

    return None, indices
```

**Expected Speedup:** 1.5x for small k (typical case: k=10-50)

**Caveat:** Pure Python heap might be slower than optimized PyTorch - need to benchmark

---

## Proposal 7: Cached Median with Moving Window (Low Impact)

### Problem
Recompute median from scratch every eviction.

### Solution
**Track running statistics** and update incrementally:

```python
class RunningMedian:
    """Approximate median tracking with minimal overhead."""

    def __init__(self, percentile=0.5, decay=0.95):
        self.estimate = None
        self.decay = decay
        self.percentile = percentile

    def update(self, values):
        """Update median estimate."""
        current_median = values.median()

        if self.estimate is None:
            self.estimate = current_median
        else:
            # Exponential moving average
            self.estimate = self.decay * self.estimate + (1 - self.decay) * current_median

        return self.estimate
```

**Expected Speedup:** 1.1x (saves median computation)

**Caveat:** Approximate - may reduce eviction quality slightly

---

## Combined Impact Estimate

### Conservative Estimate (Low-Risk Changes)
Combining Proposals 1, 2, and 4:
- Fused selection: 2x
- Adaptive interval: 1.3x
- In-place pruning: 1.2x

**Total: ~3x speedup → 50ms → 17ms** (still above target)

### Aggressive Estimate (All Proposals)
If all optimizations compound:
- Fused selection: 2x
- Adaptive interval: 1.3x
- Quantized scores: 1.2x
- In-place pruning: 1.5x
- Parallel layers: 1.2x
- Custom topk: 1.5x

**Total: ~7x speedup → 50ms → 7ms** ✓ (meets <10ms target)

---

## Implementation Priority

### Phase 1: Low-Hanging Fruit (1-2 days)
1. ✅ **Proposal 2: Adaptive eviction interval** - Simple, safe, 30% improvement
2. ✅ **Proposal 1: Fused selection** - Moderate complexity, 2x improvement

**Expected:** 50ms → 20ms

### Phase 2: Memory Optimizations (2-3 days)
3. ✅ **Proposal 3: Quantized scores** - Low risk, memory bandwidth improvement
4. ✅ **Proposal 4: In-place pruning** - Need careful testing, 1.5x potential

**Expected:** 20ms → 10ms ✓ (target met)

### Phase 3: Advanced (Optional, 3-5 days)
5. **Proposal 6: Custom topk** - Benchmark first, may not help
6. **Proposal 5: Parallel layers** - Complex, test memory usage
7. **Proposal 7: Cached median** - Risky (quality vs speed tradeoff)

**Expected:** 10ms → 5-7ms (bonus)

---

## Benchmarking Plan

For each optimization:

```python
# Benchmark template
def benchmark_optimization(method_name, num_trials=100):
    """Measure speedup of optimization."""
    cache = CABCache(max_cache_size=512, sparsity=0.9)

    # Warm up
    for _ in range(10):
        # ... simulate eviction

    # Baseline
    times_before = []
    for _ in range(num_trials):
        start = time.time()
        cache._evict()
        times_before.append(time.time() - start)

    # Apply optimization
    # ... enable optimization

    # After
    times_after = []
    for _ in range(num_trials):
        start = time.time()
        cache._evict()
        times_after.append(time.time() - start)

    speedup = np.mean(times_before) / np.mean(times_after)
    print(f"{method_name}: {speedup:.2f}x speedup")
```

---

## Risk Assessment

### Low Risk (Implement First)
- ✅ Adaptive eviction interval
- ✅ Fused selection (maintains same logic)
- ✅ Quantized scores (integers preserve ranking)

### Medium Risk (Test Thoroughly)
- ⚠️ In-place pruning (potential memory bugs)
- ⚠️ Parallel layer pruning (memory spike risk)

### High Risk (Research Phase)
- ❌ Custom topk (may be slower than PyTorch)
- ❌ Cached median (may hurt eviction quality)

---

## Success Metrics

### Performance
- ✅ Average eviction time <10ms
- ✅ 95th percentile <20ms
- ✅ No degradation in cache hit rate

### Quality
- ✅ F1 score maintained (within 1% of baseline)
- ✅ No regression on HotpotQA or NarrativeQA
- ✅ Same complexity guarantees (O(N log K))

### Robustness
- ✅ All unit tests pass
- ✅ No OOM errors under stress test
- ✅ Stable across different N (128-8192)

---

## Recommendation

**Start with Phase 1 optimizations:**
1. Adaptive eviction interval (1-2 hours to implement)
2. Fused selection algorithm (4-6 hours to implement)

**Expected result:** 50ms → 20ms (2.5x improvement)

**If target not met, proceed to Phase 2:**
3. Quantized scores (2-3 hours)
4. In-place pruning (4-6 hours + testing)

**Expected cumulative:** 50ms → 10ms ✓

This conservative approach minimizes risk while achieving the <10ms target.
