"""
FRC Scoring for Bridge Detection
==================================

Uses Forman-Ricci Curvature to identify bridge tokens (low FRC)
that connect important contexts.
"""

import torch
import torch.nn.functional as F
from typing import Optional

from ..kernels.frc_triton import compute_frc


class FRCTracker:
    """
    Track and compute FRC scores for cached tokens.

    Lower FRC = bridge/bottleneck (keep!)
    Higher FRC = redundant/well-connected (can prune)
    """

    def __init__(self, device: str = 'cuda', use_triton: bool = True):
        self.device = device
        self.use_triton = use_triton
        self.frc_scores = None
        self.last_update_len = 0
        self.update_interval = 10  # Only recompute every N steps

    def reset(self):
        """Reset tracker state."""
        self.frc_scores = None
        self.last_update_len = 0

    def compute_from_keys(
        self,
        keys: torch.Tensor,
        force_update: bool = False,
    ) -> torch.Tensor:
        """
        Compute FRC scores from cached keys.

        Args:
            keys: [B, H, N, D] cached key states
            force_update: Force recomputation even if within update interval

        Returns:
            frc_scores: [N] FRC scores for each position
        """
        B, H, N, D = keys.shape

        # Check if we need to update
        if not force_update:
            if self.frc_scores is not None:
                # Only update every K steps to amortize cost
                if N - self.last_update_len < self.update_interval:
                    # Extend with zeros for new positions
                    if N > len(self.frc_scores):
                        padding = torch.zeros(
                            N - len(self.frc_scores),
                            device=self.device,
                            dtype=self.frc_scores.dtype
                        )
                        self.frc_scores = torch.cat([self.frc_scores, padding])
                    return self.frc_scores

        # Compute attention-like similarity matrix from keys
        # This approximates the attention graph structure
        # [B, H, N, D] -> [N, N] similarity matrix

        # Aggregate across batch and heads
        keys_agg = keys.mean(dim=(0, 1))  # [N, D]

        # Normalize keys
        keys_norm = F.normalize(keys_agg, dim=-1)

        # Compute similarity matrix (approximation of attention)
        attention = torch.mm(keys_norm, keys_norm.t())  # [N, N]

        # Ensure non-negative (attention weights are non-negative)
        attention = F.relu(attention)

        # Normalize to sum to 1 (like attention)
        row_sums = attention.sum(dim=1, keepdim=True) + 1e-8
        attention = attention / row_sums

        # Compute FRC using Triton kernel
        frc_scores = compute_frc(attention, use_triton=self.use_triton)

        # Cache result
        self.frc_scores = frc_scores
        self.last_update_len = N

        return frc_scores

    def compute_from_attention(
        self,
        attention: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute FRC directly from attention weights.

        Args:
            attention: [B, H, N_q, N_kv] or [N, N] attention weights

        Returns:
            frc_scores: [N_kv] FRC scores
        """
        if attention.dim() == 4:
            # Aggregate attention across batch, heads, queries
            B, H, N_q, N_kv = attention.shape
            attention_agg = attention.mean(dim=(0, 1))  # [N_q, N_kv]

            # Average across queries to get [N_kv, N_kv] matrix
            # This approximates the key-key relationship
            attention_symmetric = (attention_agg.T + attention_agg) / 2.0

            # Take square submatrix [N_kv, N_kv]
            N = min(attention_symmetric.shape)
            attention_matrix = attention_symmetric[:N, :N]
        else:
            # Already [N, N]
            attention_matrix = attention

        # Ensure on correct device
        attention_matrix = attention_matrix.to(self.device)

        # Compute FRC
        frc_scores = compute_frc(attention_matrix, use_triton=self.use_triton)

        # Cache result
        self.frc_scores = frc_scores
        self.last_update_len = len(frc_scores)

        return frc_scores

    def get_bottom_k_indices(
        self,
        k: int,
        exclude_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get bottom-k FRC positions (bridges/bottlenecks).

        Args:
            k: Number of positions to select
            exclude_indices: Indices to exclude from selection

        Returns:
            indices: [k] indices of bottom-k FRC positions (bridges)
        """
        if self.frc_scores is None:
            # No FRC computed yet, return empty
            return torch.tensor([], dtype=torch.long, device=self.device)

        scores = self.frc_scores.clone()

        # Exclude already-selected indices
        if exclude_indices is not None and len(exclude_indices) > 0:
            scores[exclude_indices] = float('inf')

        # Select bottom-k (LOWEST FRC = bridges)
        k = min(k, len(scores))
        if k == 0:
            return torch.tensor([], dtype=torch.long, device=self.device)

        _, indices = torch.topk(scores, k=k, largest=False)  # LOWEST!

        return indices

    def get_scores(self) -> Optional[torch.Tensor]:
        """Get current FRC scores."""
        return self.frc_scores

    def prune(self, keep_indices: torch.Tensor):
        """
        Prune FRC scores to match pruned cache.

        Args:
            keep_indices: Indices of positions to keep
        """
        if self.frc_scores is not None:
            # Filter keep_indices to only include valid indices for current size
            current_len = len(self.frc_scores)
            valid_mask = keep_indices < current_len
            valid_indices = keep_indices[valid_mask]

            if len(valid_indices) > 0:
                self.frc_scores = self.frc_scores[valid_indices]
                self.last_update_len = len(self.frc_scores)
            else:
                # No valid indices, reset
                self.frc_scores = None
                self.last_update_len = 0

    def __len__(self) -> int:
        """Return number of tracked positions."""
        return 0 if self.frc_scores is None else len(self.frc_scores)


if __name__ == "__main__":
    # Test FRC tracking
    print("Testing FRCTracker...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    tracker = FRCTracker(device=device, use_triton=(device == 'cuda'))

    # Create synthetic key cache
    B, H, N, D = 1, 8, 100, 64
    keys = torch.randn(B, H, N, D, device=device)

    # Compute FRC
    import time
    start = time.time()
    frc_scores = tracker.compute_from_keys(keys)
    elapsed = (time.time() - start) * 1000

    print(f"\nFRC computation: {elapsed:.2f}ms for {N} tokens")
    print(f"FRC scores shape: {frc_scores.shape}")
    print(f"  Mean: {frc_scores.mean():.4f}")
    print(f"  Std: {frc_scores.std():.4f}")
    print(f"  Min: {frc_scores.min():.4f} (strongest bridge)")
    print(f"  Max: {frc_scores.max():.4f} (most redundant)")

    # Get bottom-10 (bridges)
    bottom_10 = tracker.get_bottom_k_indices(k=10)
    print(f"\nBottom-10 FRC positions (bridges): {bottom_10.cpu().numpy()}")
    print(f"Their FRC scores:")
    for idx in bottom_10[:5]:
        print(f"  Position {idx.item()}: {frc_scores[idx].item():.4f}")

    # Test update interval (should not recompute)
    start = time.time()
    frc_scores_2 = tracker.compute_from_keys(keys, force_update=False)
    elapsed_2 = (time.time() - start) * 1000
    print(f"\nCached lookup: {elapsed_2:.2f}ms (should be ~0ms)")

    # Test force update
    start = time.time()
    frc_scores_3 = tracker.compute_from_keys(keys, force_update=True)
    elapsed_3 = (time.time() - start) * 1000
    print(f"Forced update: {elapsed_3:.2f}ms (should be ~{elapsed:.2f}ms)")
