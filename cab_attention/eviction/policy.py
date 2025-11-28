"""
Three-Component Eviction Policy
================================

Combines:
1. Local context (recent tokens)
2. Bridge tokens (low FRC connectors)
3. Important tokens (high cumulative attention)
"""

import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class EvictionConfig:
    """Configuration for eviction policy."""
    local_ratio: float = 0.3       # 30% for local context
    bridge_ratio: float = 0.2      # 20% for bridges
    importance_ratio: float = 0.5  # 50% for important tokens

    def __post_init__(self):
        # Validate ratios sum to 1.0
        total = self.local_ratio + self.bridge_ratio + self.importance_ratio
        assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"


class ThreeComponentEvictionPolicy:
    """
    Eviction policy using three components:
    - Local: Keep most recent K tokens
    - Bridges: Keep tokens with lowest FRC (connectors)
    - Importance: Keep tokens with highest cumulative attention (H2O-style)
    """

    def __init__(self, config: EvictionConfig):
        self.config = config

    def select_indices(
        self,
        cache_len: int,
        keep_size: int,
        importance_scores: Optional[torch.Tensor],
        frc_scores: Optional[torch.Tensor],
        device: str = 'cuda',
    ) -> Tuple[torch.Tensor, dict]:
        """
        Select which indices to keep using three-component strategy.

        Args:
            cache_len: Current cache length
            keep_size: Total number of tokens to keep
            importance_scores: [cache_len] H2O-style cumulative attention
            frc_scores: [cache_len] FRC scores (lower = bridge)
            device: Device for tensor operations

        Returns:
            keep_indices: [keep_size] indices of tokens to keep (sorted)
            diagnostics: Dict with selection statistics
        """
        # Compute budget for each component
        local_budget = int(keep_size * self.config.local_ratio)
        bridge_budget = int(keep_size * self.config.bridge_ratio)
        importance_budget = keep_size - local_budget - bridge_budget

        # Ensure budgets are valid
        local_budget = min(local_budget, cache_len)
        bridge_budget = min(bridge_budget, cache_len)
        importance_budget = min(importance_budget, cache_len)

        # Component 1: Local context (most recent tokens)
        local_indices = torch.arange(
            cache_len - local_budget, cache_len,
            device=device, dtype=torch.long
        )

        # Create mask of already-selected indices
        selected_mask = torch.zeros(cache_len, dtype=torch.bool, device=device)
        selected_mask[local_indices] = True

        # Component 2: Important tokens (highest cumulative attention)
        if importance_scores is not None and importance_budget > 0:
            # Ensure importance_scores matches cache_len
            if len(importance_scores) != cache_len:
                # Pad or truncate to match cache_len
                if len(importance_scores) < cache_len:
                    # Pad with zeros
                    padding = torch.zeros(
                        cache_len - len(importance_scores),
                        device=device,
                        dtype=importance_scores.dtype
                    )
                    importance_scores = torch.cat([importance_scores, padding])
                else:
                    # Truncate
                    importance_scores = importance_scores[:cache_len]

            # Mask out already-selected indices
            candidate_scores = importance_scores.clone()
            candidate_scores[selected_mask] = -float('inf')

            # Select top-k remaining
            k = min(importance_budget, (~selected_mask).sum().item())
            if k > 0:
                _, importance_indices = torch.topk(
                    candidate_scores, k=k, largest=True
                )
            else:
                importance_indices = torch.tensor([], dtype=torch.long, device=device)
        else:
            # Fallback: keep tokens before local window
            start = max(0, cache_len - local_budget - importance_budget)
            end = cache_len - local_budget
            importance_indices = torch.arange(start, end, device=device, dtype=torch.long)

        # Update selected mask
        if len(importance_indices) > 0:
            selected_mask[importance_indices] = True

        # Component 3: Bridge tokens (LOWEST FRC)
        if frc_scores is not None and bridge_budget > 0:
            # Ensure frc_scores matches cache_len
            if len(frc_scores) != cache_len:
                # Pad or truncate to match cache_len
                if len(frc_scores) < cache_len:
                    # Pad with high values (so they won't be selected as bridges)
                    padding = torch.full(
                        (cache_len - len(frc_scores),),
                        float('inf'),
                        device=device,
                        dtype=frc_scores.dtype
                    )
                    frc_scores = torch.cat([frc_scores, padding])
                else:
                    # Truncate
                    frc_scores = frc_scores[:cache_len]

            # Mask out already-selected indices
            candidate_frc = frc_scores.clone()
            candidate_frc[selected_mask] = float('inf')

            # Select bottom-k remaining (LOWEST FRC = bridges)
            k = min(bridge_budget, (~selected_mask).sum().item())
            if k > 0:
                _, bridge_indices = torch.topk(
                    candidate_frc, k=k, largest=False  # LOWEST!
                )
            else:
                bridge_indices = torch.tensor([], dtype=torch.long, device=device)
        else:
            # No FRC available, select randomly from remaining
            remaining_indices = torch.where(~selected_mask)[0]
            k = min(bridge_budget, len(remaining_indices))
            if k > 0:
                perm = torch.randperm(len(remaining_indices), device=device)[:k]
                bridge_indices = remaining_indices[perm]
            else:
                bridge_indices = torch.tensor([], dtype=torch.long, device=device)

        # Combine all selected indices
        keep_indices = torch.cat([
            local_indices,
            importance_indices,
            bridge_indices,
        ])

        # Remove duplicates and sort
        keep_indices = keep_indices.unique().sort().values

        # Diagnostics
        diagnostics = {
            'cache_len': cache_len,
            'keep_size': len(keep_indices),
            'local_count': len(local_indices),
            'importance_count': len(importance_indices),
            'bridge_count': len(bridge_indices),
            'local_ratio_actual': len(local_indices) / len(keep_indices) if len(keep_indices) > 0 else 0,
            'importance_ratio_actual': len(importance_indices) / len(keep_indices) if len(keep_indices) > 0 else 0,
            'bridge_ratio_actual': len(bridge_indices) / len(keep_indices) if len(keep_indices) > 0 else 0,
        }

        return keep_indices, diagnostics


if __name__ == "__main__":
    # Test eviction policy
    print("Testing ThreeComponentEvictionPolicy...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create policy
    config = EvictionConfig(
        local_ratio=0.3,
        bridge_ratio=0.2,
        importance_ratio=0.5,
    )
    policy = ThreeComponentEvictionPolicy(config)

    # Simulate cache
    cache_len = 1000
    keep_size = 100  # 90% sparsity

    # Synthetic scores
    importance_scores = torch.rand(cache_len, device=device)
    # Make some positions very important
    importance_scores[100] = 10.0
    importance_scores[500] = 15.0

    # FRC scores (some positions are bridges with low FRC)
    frc_scores = torch.randn(cache_len, device=device)
    # Make some positions strong bridges
    frc_scores[250] = -5.0  # Strong bridge
    frc_scores[750] = -3.0  # Strong bridge

    # Select indices
    keep_indices, diagnostics = policy.select_indices(
        cache_len=cache_len,
        keep_size=keep_size,
        importance_scores=importance_scores,
        frc_scores=frc_scores,
        device=device,
    )

    print(f"\nEviction results:")
    print(f"  Cache length: {diagnostics['cache_len']}")
    print(f"  Keep size: {diagnostics['keep_size']}")
    print(f"  Local tokens: {diagnostics['local_count']} ({diagnostics['local_ratio_actual']:.2%})")
    print(f"  Important tokens: {diagnostics['importance_count']} ({diagnostics['importance_ratio_actual']:.2%})")
    print(f"  Bridge tokens: {diagnostics['bridge_count']} ({diagnostics['bridge_ratio_actual']:.2%})")

    # Check if important positions were kept
    print(f"\nImportant position 100 kept: {100 in keep_indices}")
    print(f"Important position 500 kept: {500 in keep_indices}")
    print(f"Bridge position 250 kept: {250 in keep_indices}")
    print(f"Bridge position 750 kept: {750 in keep_indices}")

    # Check local positions
    local_start = cache_len - int(keep_size * config.local_ratio)
    print(f"\nLocal window: [{local_start}, {cache_len})")
    print(f"All local positions kept: {all(i in keep_indices for i in range(local_start, cache_len))}")

    # Verify indices are sorted
    print(f"\nIndices are sorted: {torch.all(keep_indices[1:] > keep_indices[:-1]).item()}")
    print(f"No duplicates: {len(keep_indices) == len(keep_indices.unique())}")
