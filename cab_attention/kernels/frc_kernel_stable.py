"""
Task 1.2: Stabilized Forman-Ricci Curvature Computation

KEY IMPROVEMENTS FOR ICML SUBMISSION:
1. Normalized affinity matrix A ∈ [0, 1] for numerical stability
2. Temperature-aware scaling to prevent explosion
3. Gradient-stable operations (avoid NaN/Inf)
4. Physics-grounded: FRC ∝ (Direct Connection - Redundancy)

PHYSICS INTUITION:
- HIGH FRC (positive): Redundant clique → low unique information → PRUNE
- LOW FRC (negative): Sparse bridge → high unique information → KEEP
"""

import torch
import torch.nn.functional as F


def compute_block_frc_stable(
    q_coarse: torch.Tensor,
    k_coarse: torch.Tensor,
    temperature: float = 1.0,
    lambda_redundancy: float = 0.5,
    normalization: str = 'row',  # 'row', 'minmax', or 'none'
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stabilized Forman-Ricci Curvature on coarse block graph.

    CRITICAL CHANGES FOR STABILITY:
    1. Normalize affinity matrix to [0, 1] to prevent explosion
    2. Use temperature scaling properly
    3. Ensure all operations are gradient-stable
    4. Physics-grounded formula: FRC = Direct - λ × Redundancy

    Args:
        q_coarse: Coarse query [B, H, M, D]
        k_coarse: Coarse key [B, H, M, D]
        temperature: Scaling factor (default 1.0, typically 1/sqrt(D))
        lambda_redundancy: Weight for triangle penalty (default 0.5)
        normalization: How to normalize affinity ('row', 'minmax', 'none')
        eps: Small constant for numerical stability

    Returns:
        frc_scores: Curvature scores [B, H, M, M] (LOW = bridge = KEEP)
        affinity: Normalized affinity matrix [B, H, M, M]
        redundancy: Triangle counts [B, H, M, M]

    PHYSICS:
        FRC = A - λ × (A @ A)

        Where:
        - A[i,j]: Direct connection strength (normalized to [0,1])
        - A @ A: Redundancy through 2-hop paths
        - λ: Controls redundancy penalty

        Interpretation:
        - FRC > 0: Direct > Redundancy → Unique connection → KEEP
        - FRC < 0: Direct < Redundancy → Redundant path → PRUNE (if magnitude-based)

        But we INVERT for bridge finding:
        - We want LOW FRC (negative) because those are bridges!
        - A bridge has LOW direct strength but UNIQUE path (low redundancy relative to its weight)
    """
    B, H, M, D = q_coarse.shape
    assert k_coarse.shape == (B, H, M, D), f"Shape mismatch: {k_coarse.shape} != {q_coarse.shape}"

    # ======================================================================
    # Step 1: Compute Raw Affinity Matrix
    # ======================================================================
    # Standard attention: Q @ K^T / sqrt(D)
    scale = temperature / (D ** 0.5)
    raw_affinity = torch.matmul(q_coarse, k_coarse.transpose(-2, -1)) * scale  # [B, H, M, M]

    # ======================================================================
    # Step 2: Normalize Affinity to [0, 1] (CRITICAL FOR STABILITY)
    # ======================================================================
    if normalization == 'row':
        # Row normalization: Each row sums to 1 (like softmax but preserves sparsity)
        # A[i,j] = exp(raw[i,j]) / sum_j exp(raw[i,j])
        # BUT we want to preserve low-magnitude bridges, so use ReLU + normalize
        A_positive = F.relu(raw_affinity)  # Ensure non-negative
        row_sums = A_positive.sum(dim=-1, keepdim=True) + eps  # [B, H, M, 1]
        A = A_positive / row_sums  # [B, H, M, M] ∈ [0, 1]

    elif normalization == 'minmax':
        # Min-max normalization: A ∈ [0, 1] globally per head
        # A[i,j] = (raw[i,j] - min) / (max - min)
        min_val = raw_affinity.view(B, H, -1).min(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B, H, 1, 1]
        max_val = raw_affinity.view(B, H, -1).max(dim=-1, keepdim=True)[0].unsqueeze(-1)  # [B, H, 1, 1]
        A = (raw_affinity - min_val) / (max_val - min_val + eps)  # [B, H, M, M] ∈ [0, 1]

    elif normalization == 'softmax':
        # Standard softmax (most stable but loses absolute magnitude info)
        A = F.softmax(raw_affinity, dim=-1)  # [B, H, M, M] ∈ [0, 1]

    else:  # normalization == 'none'
        # No normalization (original unstable version)
        # WARNING: Only use for debugging/comparison
        A = F.relu(raw_affinity)
        # Clip to prevent explosion
        A = torch.clamp(A, 0, 1.0)

    # ======================================================================
    # Step 3: Compute Redundancy (Triangle Count)
    # ======================================================================
    # Redundancy[i,j] = sum_k A[i,k] * A[k,j]
    # This counts 2-hop paths from i to j through intermediate node k
    # High redundancy → information can flow through alternative paths → PRUNE
    redundancy = torch.matmul(A, A)  # [B, H, M, M]

    # Normalize redundancy to same scale as A for fair comparison
    # Since A ∈ [0, 1], redundancy ∈ [0, M] theoretically
    # We normalize by M to get redundancy ∈ [0, 1]
    redundancy = redundancy / (M + eps)

    # ======================================================================
    # Step 4: Forman-Ricci Curvature (Simplified Physics-Grounded Formula)
    # ======================================================================
    # Original FRC formula: F = 4*A - S_i - S_j + 3*T
    # This is unstable because coefficients don't account for normalization
    #
    # Our STABLE formula: FRC = A - λ × Redundancy
    #
    # Interpretation:
    #   FRC > 0: Direct > Redundancy → Unique high-value path → KEEP
    #   FRC < 0: Direct < Redundancy → Redundant low-value path → PRUNE
    #
    # For BRIDGE FINDING (our goal):
    #   Bridges have LOW absolute weight but UNIQUE path
    #   → We select LOWEST FRC (most negative or least positive)
    #   → These are edges that don't have good alternative paths

    frc_scores = A - lambda_redundancy * redundancy  # [B, H, M, M]

    # ======================================================================
    # Step 5: Gradient Stability Check
    # ======================================================================
    # Ensure no NaN/Inf
    frc_scores = torch.where(
        torch.isfinite(frc_scores),
        frc_scores,
        torch.zeros_like(frc_scores)
    )

    return frc_scores, A, redundancy


def compute_block_frc_v3_high(
    q_coarse: torch.Tensor,
    k_coarse: torch.Tensor,
    temperature: float = 1.0,
    lambda_redundancy: float = 0.5,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    CAB V3 Version: Select HIGH FRC (the breakthrough from debugging!)

    This version selects blocks with HIGH FRC scores, which represents:
    - High direct attention
    - Low redundancy (unique information)

    This is the OPPOSITE of bridge finding but works better empirically.

    Physics Interpretation:
    - HIGH FRC → Strong unique connection → Important for task
    - LOW FRC → Weak or redundant connection → Can be pruned
    """
    frc_scores, A, redundancy = compute_block_frc_stable(
        q_coarse, k_coarse, temperature, lambda_redundancy,
        normalization='row', eps=eps
    )

    # For CAB V3, we want HIGH FRC
    # Return as-is (downstream selection uses largest=True)
    return frc_scores, A, redundancy


def generate_block_mask_from_frc_stable(
    frc_scores: torch.Tensor,
    sparsity: float = 0.95,
    select_high: bool = True,  # True for CAB V3, False for bridge finding
    keep_diagonal: bool = True,
    causal: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Stabilized block mask generation from FRC scores.

    Args:
        frc_scores: FRC scores [B, H, M, M]
        sparsity: Fraction to prune (0.95 = keep 5%)
        select_high: If True, keep HIGH FRC (CAB V3); if False, keep LOW FRC (bridges)
        keep_diagonal: Always keep diagonal (local attention)
        causal: Enforce causal masking
        eps: Numerical stability constant

    Returns:
        block_mask: Binary mask [B, H, M, M] (1 = KEEP, 0 = PRUNE)
    """
    B, H, M, _ = frc_scores.shape

    # Calculate number of blocks to KEEP per query
    k = max(1, int(M * (1.0 - sparsity)))

    # Ensure k doesn't exceed M
    k = min(k, M)

    # Select blocks
    if select_high:
        # CAB V3: Select HIGH FRC (strong unique connections)
        _, top_indices = torch.topk(frc_scores, k, dim=-1, largest=True, sorted=False)
    else:
        # Bridge finding: Select LOW FRC (weak unique connections)
        _, top_indices = torch.topk(frc_scores, k, dim=-1, largest=False, sorted=False)

    # Create mask
    mask = torch.zeros_like(frc_scores, dtype=torch.bool)

    # Scatter operation to set selected indices to True
    batch_idx = torch.arange(B, device=frc_scores.device).view(B, 1, 1, 1).expand(B, H, M, k)
    head_idx = torch.arange(H, device=frc_scores.device).view(1, H, 1, 1).expand(B, H, M, k)
    query_idx = torch.arange(M, device=frc_scores.device).view(1, 1, M, 1).expand(B, H, M, k)

    mask[batch_idx, head_idx, query_idx, top_indices] = True

    # Always keep diagonal (local attention)
    if keep_diagonal:
        diag_mask = torch.eye(M, device=frc_scores.device, dtype=torch.bool)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        mask = mask | diag_mask

    # Causal masking
    if causal:
        causal_mask = torch.tril(torch.ones(M, M, device=frc_scores.device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        mask = mask & causal_mask

    return mask


def validate_frc_stability(frc_scores: torch.Tensor, affinity: torch.Tensor, redundancy: torch.Tensor) -> dict:
    """
    Validate numerical stability of FRC computation.

    Returns dict with diagnostics.
    """
    diagnostics = {}

    # Check for NaN/Inf
    diagnostics['has_nan'] = torch.isnan(frc_scores).any().item()
    diagnostics['has_inf'] = torch.isinf(frc_scores).any().item()

    # Value ranges
    diagnostics['frc_min'] = frc_scores.min().item()
    diagnostics['frc_max'] = frc_scores.max().item()
    diagnostics['frc_mean'] = frc_scores.mean().item()
    diagnostics['frc_std'] = frc_scores.std().item()

    diagnostics['affinity_min'] = affinity.min().item()
    diagnostics['affinity_max'] = affinity.max().item()
    diagnostics['affinity_mean'] = affinity.mean().item()

    diagnostics['redundancy_min'] = redundancy.min().item()
    diagnostics['redundancy_max'] = redundancy.max().item()
    diagnostics['redundancy_mean'] = redundancy.mean().item()

    # Check if affinity is normalized
    diagnostics['affinity_in_01'] = (affinity.min() >= -1e-6) and (affinity.max() <= 1.0 + 1e-6)

    # Gradient check (if requires_grad)
    if frc_scores.requires_grad:
        diagnostics['has_grad'] = True
        # Try backward pass
        try:
            loss = frc_scores.sum()
            loss.backward()
            diagnostics['grad_stable'] = True
        except:
            diagnostics['grad_stable'] = False
    else:
        diagnostics['has_grad'] = False
        diagnostics['grad_stable'] = None

    return diagnostics
