"""
H2O (Heavy-Hitter Oracle) Eviction Policy
==========================================

Two-component eviction strategy:
1. Local context (recent tokens) - typically 20%
2. Important tokens (high cumulative attention) - typically 80%

Reference: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference" (2023)
"""

import torch
from typing import Tuple


def h2o_select_indices(
    cache_len: int,
    keep_size: int,
    importance_scores: torch.Tensor,
    local_ratio: float = 0.2,
    device: str = 'cuda',
) -> Tuple[torch.Tensor, dict]:
    """
    Select indices to keep using H2O two-component strategy.

    Args:
        cache_len: Current cache length
        keep_size: Total number of tokens to keep
        importance_scores: [cache_len] cumulative attention scores (H2O heavy-hitter metric)
        local_ratio: Fraction of keep_size to allocate to recent tokens (default: 0.2 = 20%)
        device: Device for tensor operations

    Returns:
        keep_indices: [keep_size] indices of tokens to keep (sorted)
        diagnostics: Dict with selection statistics
    """
    # Compute budget for each component
    local_size = max(1, int(keep_size * local_ratio))
    important_size = keep_size - local_size

    # Ensure budgets don't exceed cache length
    local_size = min(local_size, cache_len)
    important_size = min(important_size, cache_len - local_size)

    # Component 1: Local context (most recent tokens)
    local_indices = torch.arange(
        cache_len - local_size, cache_len,
        device=device, dtype=torch.long
    )

    # Create mask for already-selected tokens
    selected_mask = torch.zeros(cache_len, dtype=torch.bool, device=device)
    selected_mask[local_indices] = True

    # Component 2: Important tokens (highest cumulative attention)
    if importance_scores is not None and important_size > 0:
        # Ensure importance_scores matches cache_len
        if len(importance_scores) != cache_len:
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

        # Mask out already-selected local tokens
        candidate_scores = importance_scores.clone()
        candidate_scores[selected_mask] = -float('inf')

        # Select top important_size tokens by score
        important_indices = candidate_scores.topk(important_size).indices
        selected_mask[important_indices] = True
    else:
        important_indices = torch.tensor([], device=device, dtype=torch.long)

    # Combine and sort indices
    keep_indices = torch.cat([local_indices, important_indices])
    keep_indices = torch.sort(keep_indices)[0]

    # Diagnostics
    diagnostics = {
        'local_count': len(local_indices),
        'important_count': len(important_indices),
        'total_kept': len(keep_indices),
        'cache_len': cache_len,
        'target_size': keep_size,
    }

    return keep_indices, diagnostics
