"""
Importance Tracking for KV Cache
=================================

Implements H2O-style cumulative attention tracking to identify
"heavy hitter" tokens that receive high attention.

Reference: Zhang et al., 2023 - "H2O: Heavy-Hitter Oracle for Efficient
           Generative Inference of Large Language Models"
           arXiv:2306.14048
"""

import torch
from typing import Optional


class ImportanceTracker:
    """
    Track cumulative attention scores for H2O-style importance.

    Maintains running sum of attention received by each cached token.
    Higher scores = more important tokens (frequently attended to).
    """

    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.cumulative_scores = None
        self.total_queries = 0

    def reset(self):
        """Reset tracker state."""
        self.cumulative_scores = None
        self.total_queries = 0

    def update(self, attention_weights: torch.Tensor):
        """
        Update importance scores with new attention weights.

        Args:
            attention_weights: [B, H, N_q, N_kv] attention weights
                              or [B, H, 1, N_kv] for single query

        Updates cumulative scores for each key position.
        """
        # Sum across batch, heads, and queries to get per-key importance
        # [B, H, N_q, N_kv] -> [N_kv]
        position_scores = attention_weights.sum(dim=(0, 1, 2))

        if self.cumulative_scores is None:
            # Initialize
            self.cumulative_scores = position_scores
        else:
            # Accumulate
            # Handle case where cache grew (new positions added)
            current_len = len(position_scores)
            cached_len = len(self.cumulative_scores)

            if current_len > cached_len:
                # Cache grew, extend cumulative scores
                padding = torch.zeros(
                    current_len - cached_len,
                    device=self.device,
                    dtype=self.cumulative_scores.dtype
                )
                self.cumulative_scores = torch.cat([self.cumulative_scores, padding])

            # Add new scores
            self.cumulative_scores[:current_len] += position_scores

        self.total_queries += attention_weights.shape[2]

    def get_top_k_indices(
        self,
        k: int,
        exclude_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get top-k most important positions.

        Args:
            k: Number of positions to select
            exclude_indices: Indices to exclude from selection (e.g., already kept)

        Returns:
            indices: [k] indices of top-k important positions
        """
        if self.cumulative_scores is None:
            # No tracking yet, return most recent
            cache_len = 0
            return torch.arange(
                max(0, cache_len - k), cache_len,
                device=self.device
            )

        scores = self.cumulative_scores.clone()

        # Exclude already-selected indices
        if exclude_indices is not None and len(exclude_indices) > 0:
            scores[exclude_indices] = -float('inf')

        # Select top-k
        k = min(k, len(scores))
        _, indices = torch.topk(scores, k=k, largest=True)

        return indices

    def get_scores(self) -> Optional[torch.Tensor]:
        """Get current cumulative importance scores."""
        return self.cumulative_scores

    def prune(self, keep_indices: torch.Tensor):
        """
        Prune importance scores to match pruned cache.

        Args:
            keep_indices: Indices of positions to keep
        """
        if self.cumulative_scores is not None:
            # Filter keep_indices to only include valid indices for current size
            current_len = len(self.cumulative_scores)
            valid_mask = keep_indices < current_len
            valid_indices = keep_indices[valid_mask]

            if len(valid_indices) > 0:
                self.cumulative_scores = self.cumulative_scores[valid_indices]
            else:
                # No valid indices, reset
                self.cumulative_scores = None

    def __len__(self) -> int:
        """Return number of tracked positions."""
        return 0 if self.cumulative_scores is None else len(self.cumulative_scores)


if __name__ == "__main__":
    # Test importance tracking
    print("Testing ImportanceTracker...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    tracker = ImportanceTracker(device=device)

    # Simulate generation
    B, H = 1, 8
    cache_len = 0

    for step in range(100):
        # Add new token to cache
        cache_len += 1

        # Simulate attention (current query attends to all cached keys)
        attention = torch.rand(B, H, 1, cache_len, device=device)
        attention = attention / attention.sum(dim=-1, keepdim=True)

        # Make some positions more important
        if cache_len > 10:
            # Position 5 and 10 receive high attention
            attention[:, :, :, 5] *= 5.0
            if cache_len > 20:
                attention[:, :, :, 20] *= 10.0

        attention = attention / attention.sum(dim=-1, keepdim=True)

        # Update tracker
        tracker.update(attention)

    # Get top-10 important positions
    top_10 = tracker.get_top_k_indices(k=10)

    print(f"\nAfter 100 generation steps:")
    print(f"Cache length: {len(tracker)}")
    print(f"Top-10 important positions: {top_10.cpu().numpy()}")
    print(f"  (Expected: positions 5, 20 should be in top-10)")

    scores = tracker.get_scores()
    print(f"\nTop-10 scores:")
    for idx in top_10[:10]:
        print(f"  Position {idx.item()}: {scores[idx].item():.4f}")

    # Test pruning
    print(f"\nTesting pruning...")
    keep_indices = torch.arange(0, 50, device=device)
    tracker.prune(keep_indices)
    print(f"After pruning to first 50 positions: {len(tracker)}")
