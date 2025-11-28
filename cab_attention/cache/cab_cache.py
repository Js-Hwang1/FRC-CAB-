"""
CAB KV Cache: Three-Component Eviction
=======================================

Efficient KV cache with dynamic eviction using:
1. Local context (recent tokens for fluency)
2. Bridge tokens (low FRC connectors for reasoning chains)
3. Important tokens (high cumulative attention, H2O-style)
"""

import torch
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

from ..scoring import ImportanceTracker, FRCTracker
from ..eviction import ThreeComponentEvictionPolicy, EvictionConfig


@dataclass
class CABCacheConfig:
    """Configuration for CAB cache."""
    max_cache_size: int = 4096      # Maximum cache size
    sparsity: float = 0.9           # Target sparsity (0.9 = keep 10%)

    # Component ratios
    local_ratio: float = 0.3        # 30% for local context
    bridge_ratio: float = 0.2       # 20% for bridges
    importance_ratio: float = 0.5   # 50% for importance

    # Update intervals
    eviction_interval: int = 10     # Evict every K tokens (amortization)
    frc_update_interval: int = 10   # Recompute FRC every K evictions

    # Triton optimization
    use_triton: bool = True         # Use Triton kernels if available

    # Device
    device: str = 'cuda'


class CABCache:
    """
    CAB KV Cache with three-component eviction policy.

    Compatible with HuggingFace transformers Cache interface.

    Usage:
        cache = CABCache(max_cache_size=4096, sparsity=0.9)

        outputs = model.generate(
            input_ids=input_ids,
            past_key_values=cache,
            max_new_tokens=512,
        )
    """

    def __init__(
        self,
        max_cache_size: int = 4096,
        sparsity: float = 0.9,
        local_ratio: float = 0.3,
        bridge_ratio: float = 0.2,
        importance_ratio: float = 0.5,
        eviction_interval: int = 10,
        frc_update_interval: int = 10,
        use_triton: bool = True,
        device: str = 'cuda',
    ):
        """
        Initialize CAB cache.

        Args:
            max_cache_size: Maximum number of tokens to cache
            sparsity: Target sparsity (0.9 = keep 10% of tokens)
            local_ratio: Fraction of budget for local context
            bridge_ratio: Fraction of budget for bridge tokens
            importance_ratio: Fraction of budget for important tokens
            eviction_interval: Evict every K tokens (amortization)
            frc_update_interval: Recompute FRC every K evictions
            use_triton: Use Triton kernels if available
            device: Device for cache
        """
        self.config = CABCacheConfig(
            max_cache_size=max_cache_size,
            sparsity=sparsity,
            local_ratio=local_ratio,
            bridge_ratio=bridge_ratio,
            importance_ratio=importance_ratio,
            eviction_interval=eviction_interval,
            frc_update_interval=frc_update_interval,
            use_triton=use_triton and device == 'cuda',
            device=device,
        )

        # Cache storage (per layer)
        self.key_cache = []    # List of [B, H, N, D] per layer
        self.value_cache = []  # List of [B, H, N, D] per layer

        # Scoring trackers
        self.importance_tracker = ImportanceTracker(device=device)
        self.frc_tracker = FRCTracker(device=device, use_triton=self.config.use_triton)

        # Eviction policy
        eviction_config = EvictionConfig(
            local_ratio=local_ratio,
            bridge_ratio=bridge_ratio,
            importance_ratio=importance_ratio,
        )
        self.eviction_policy = ThreeComponentEvictionPolicy(eviction_config)

        # State tracking
        self.tokens_since_last_eviction = 0
        self.evictions_since_frc_update = 0
        self.total_evictions = 0

        # Statistics
        self.stats = {
            'total_tokens_processed': 0,
            'total_evictions': 0,
            'total_tokens_evicted': 0,
            'avg_cache_size': 0.0,
        }

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states.

        Args:
            key_states: [B, H, 1, D] new key states (single token)
            value_states: [B, H, 1, D] new value states
            layer_idx: Layer index
            attention_weights: [B, H, 1, N] attention weights (optional, for tracking)

        Returns:
            keys: [B, H, N, D] updated key cache
            values: [B, H, N, D] updated value cache
        """
        # Ensure layer exists
        while len(self.key_cache) <= layer_idx:
            self.key_cache.append(None)
            self.value_cache.append(None)

        # Append new states
        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        # Update importance tracker (only for first layer to save compute)
        if layer_idx == 0 and attention_weights is not None:
            self.importance_tracker.update(attention_weights)

        # Update stats
        self.stats['total_tokens_processed'] += 1
        self.tokens_since_last_eviction += 1

        # Check if eviction needed
        cache_len = self.key_cache[layer_idx].shape[2]
        eviction_threshold = int(self.config.max_cache_size * 1.1)  # 10% buffer

        should_evict = (
            cache_len > eviction_threshold or
            self.tokens_since_last_eviction >= self.config.eviction_interval
        )

        if should_evict and cache_len > self.config.max_cache_size * (1 - self.config.sparsity):
            self._evict()

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def _evict(self):
        """Evict tokens from cache using three-component policy."""
        if len(self.key_cache) == 0 or self.key_cache[0] is None:
            return

        # Get minimum cache length across all layers
        # This ensures indices are valid for all layers (handles async updates)
        cache_len = min(
            self.key_cache[i].shape[2]
            for i in range(len(self.key_cache))
            if self.key_cache[i] is not None
        )

        # Compute target size
        keep_ratio = 1.0 - self.config.sparsity
        keep_size = int(self.config.max_cache_size * keep_ratio * 0.9)  # Target 90% of budget

        if cache_len <= keep_size:
            return  # No eviction needed

        # Update FRC scores if needed
        force_frc_update = (self.evictions_since_frc_update >= self.config.frc_update_interval)

        if force_frc_update or self.frc_tracker.frc_scores is None:
            # Compute FRC from first layer keys
            self.frc_tracker.compute_from_keys(
                self.key_cache[0],
                force_update=True
            )
            self.evictions_since_frc_update = 0

        # Select indices to keep
        keep_indices, diagnostics = self.eviction_policy.select_indices(
            cache_len=cache_len,
            keep_size=keep_size,
            importance_scores=self.importance_tracker.get_scores(),
            frc_scores=self.frc_tracker.get_scores(),
            device=self.config.device,
        )

        # Prune all layers
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] is not None:
                self.key_cache[layer_idx] = self.key_cache[layer_idx][:, :, keep_indices, :]
                self.value_cache[layer_idx] = self.value_cache[layer_idx][:, :, keep_indices, :]

        # Prune trackers
        self.importance_tracker.prune(keep_indices)
        self.frc_tracker.prune(keep_indices)

        # Update state
        self.tokens_since_last_eviction = 0
        self.evictions_since_frc_update += 1
        self.total_evictions += 1

        # Update stats
        tokens_evicted = cache_len - len(keep_indices)
        self.stats['total_evictions'] += 1
        self.stats['total_tokens_evicted'] += tokens_evicted
        self.stats['avg_cache_size'] = (
            (self.stats['avg_cache_size'] * (self.total_evictions - 1) + len(keep_indices)) /
            self.total_evictions
        )

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Get current sequence length (HuggingFace Cache interface)."""
        if layer_idx >= len(self.key_cache) or self.key_cache[layer_idx] is None:
            return 0
        return self.key_cache[layer_idx].shape[2]

    def get_max_length(self) -> int:
        """Get maximum cache length (HuggingFace Cache interface)."""
        return self.config.max_cache_size

    def reset(self):
        """Reset cache to empty state."""
        self.key_cache = []
        self.value_cache = []
        self.importance_tracker.reset()
        self.frc_tracker.reset()
        self.tokens_since_last_eviction = 0
        self.evictions_since_frc_update = 0
        self.total_evictions = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_size = self.get_seq_length(0) if len(self.key_cache) > 0 else 0

        return {
            **self.stats,
            'current_cache_size': current_size,
            'max_cache_size': self.config.max_cache_size,
            'effective_sparsity': 1.0 - (current_size / self.config.max_cache_size) if self.config.max_cache_size > 0 else 0.0,
        }

    def __len__(self) -> int:
        """Return current cache length."""
        return self.get_seq_length(0)

    def __repr__(self) -> str:
        return (
            f"CABCache("
            f"size={self.get_seq_length(0)}/{self.config.max_cache_size}, "
            f"sparsity={self.config.sparsity:.0%}, "
            f"evictions={self.total_evictions})"
        )


if __name__ == "__main__":
    # Test CAB cache
    print("Testing CABCache...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create cache
    cache = CABCache(
        max_cache_size=100,
        sparsity=0.9,
        local_ratio=0.3,
        bridge_ratio=0.2,
        importance_ratio=0.5,
        eviction_interval=5,
        device=device,
    )

    print(f"\nCache config:")
    print(f"  Max size: {cache.config.max_cache_size}")
    print(f"  Target sparsity: {cache.config.sparsity:.0%}")
    print(f"  Local ratio: {cache.config.local_ratio:.0%}")
    print(f"  Bridge ratio: {cache.config.bridge_ratio:.0%}")
    print(f"  Importance ratio: {cache.config.importance_ratio:.0%}")

    # Simulate generation
    B, H, D = 1, 8, 64
    num_layers = 4

    print(f"\nSimulating generation with {num_layers} layers...")

    for step in range(200):
        # Generate random key/value for each layer
        for layer_idx in range(num_layers):
            key_state = torch.randn(B, H, 1, D, device=device)
            value_state = torch.randn(B, H, 1, D, device=device)

            # Simulate attention (if first layer)
            attention = None
            if layer_idx == 0 and cache.get_seq_length(0) > 0:
                cache_len = cache.get_seq_length(0)
                attention = torch.rand(B, H, 1, cache_len, device=device)
                attention = attention / attention.sum(dim=-1, keepdim=True)

            # Update cache
            keys, values = cache.update(key_state, value_state, layer_idx, attention)

        # Print progress
        if (step + 1) % 50 == 0:
            print(f"  Step {step + 1}: Cache size = {len(cache)}, Evictions = {cache.total_evictions}")

    # Final stats
    print(f"\nFinal statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"\n{cache}")
