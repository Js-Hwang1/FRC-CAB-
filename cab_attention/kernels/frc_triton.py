"""
Triton Kernels for Forman-Ricci Curvature Computation
======================================================

Efficient GPU kernels for computing FRC on attention graphs.

FRC Formula (Forman 1999, Sreejith et al. 2016):
    F(i) = 4*w_i - 2*S_i + 3*T_i

Where:
    - w_i: node weight (sum of incoming/outgoing edges)
    - S_i: node strength (degree)
    - T_i: triangles involving node i
"""

import torch
import triton
import triton.language as tl


@triton.jit
def compute_node_strengths_kernel(
    # Inputs
    attention_ptr,      # [N, N] attention matrix
    N,
    # Outputs
    strengths_ptr,      # [N] output node strengths
    # Meta
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute node strengths: S_i = sum_j w_ij

    Each program computes strength for one node.
    """
    # Node index
    i = tl.program_id(0)

    if i >= N:
        return

    # Load row of attention matrix
    row_offset = i * N
    strength = 0.0

    # Process in blocks
    for block_start in range(0, N, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, N)
        block_size = block_end - block_start

        # Load block of attention weights
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N

        weights = tl.load(
            attention_ptr + row_offset + offsets,
            mask=mask,
            other=0.0
        )

        # Sum
        strength += tl.sum(weights)

    # Store result
    tl.store(strengths_ptr + i, strength)


@triton.jit
def compute_triangles_kernel(
    # Inputs
    attention_ptr,      # [N, N] attention matrix
    N,
    # Outputs
    triangles_ptr,      # [N] output triangle counts
    # Meta
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute triangles: T_i = sum_j sum_k min(w_ij, w_jk)

    This counts triangles (i -> j -> k) where node i is involved.
    Uses matrix multiplication: (A @ A)[i, i] approximation.
    """
    # Node index
    i = tl.program_id(0)

    if i >= N:
        return

    # For node i, compute sum_j sum_k w_ij * w_jk
    # This is (A^2)[i, k] summed over k
    triangle_count = 0.0

    # Load row i of attention
    row_i_offset = i * N

    # Iterate over intermediate nodes j
    for j in range(N):
        # w_ij
        w_ij = tl.load(attention_ptr + row_i_offset + j)

        # Load row j (for w_jk)
        row_j_offset = j * N

        # Process in blocks
        for block_start in range(0, N, BLOCK_SIZE):
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < N

            # Load w_jk
            w_jk = tl.load(
                attention_ptr + row_j_offset + offsets,
                mask=mask,
                other=0.0
            )

            # Add w_ij * w_jk (weighted triangles)
            # If w_ij is 0, product will be 0 anyway
            triangle_count += tl.sum(w_ij * w_jk)

    # Store result
    tl.store(triangles_ptr + i, triangle_count)


@triton.jit
def compute_frc_kernel(
    # Inputs
    attention_ptr,      # [N, N] attention matrix
    strengths_ptr,      # [N] node strengths
    triangles_ptr,      # [N] triangle counts
    N,
    # Outputs
    frc_ptr,            # [N] output FRC scores
    # Meta
    BLOCK_SIZE: tl.constexpr,
):
    """
    Compute FRC: F(i) = 4*w_i - 2*S_i + 3*T_i

    Where:
        w_i = sum_j w_ij (node weight, same as strength for undirected)
        S_i = strengths[i]
        T_i = triangles[i]
    """
    # Node index
    i = tl.program_id(0)

    if i >= N:
        return

    # Load strength and triangles
    S_i = tl.load(strengths_ptr + i)
    T_i = tl.load(triangles_ptr + i)

    # For directed graphs, w_i ≈ S_i (node strength)
    # For simplicity, use w_i = S_i
    w_i = S_i

    # FRC formula: F(i) = 4*w_i - 2*S_i + 3*T_i
    frc = 4.0 * w_i - 2.0 * S_i + 3.0 * T_i

    # Store result
    tl.store(frc_ptr + i, frc)


# Python wrapper functions
def compute_node_strengths_triton(attention: torch.Tensor) -> torch.Tensor:
    """
    Compute node strengths using Triton kernel.

    Args:
        attention: [N, N] attention matrix

    Returns:
        strengths: [N] node strengths
    """
    assert attention.is_cuda, "Input must be on CUDA"
    assert attention.dim() == 2, "Input must be 2D"

    N = attention.shape[0]
    strengths = torch.empty(N, dtype=attention.dtype, device=attention.device)

    BLOCK_SIZE = 128
    grid = (N,)

    compute_node_strengths_kernel[grid](
        attention,
        N,
        strengths,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return strengths


def compute_triangles_triton(attention: torch.Tensor) -> torch.Tensor:
    """
    Compute triangle counts using Triton kernel.

    Args:
        attention: [N, N] attention matrix

    Returns:
        triangles: [N] triangle counts
    """
    assert attention.is_cuda, "Input must be on CUDA"
    assert attention.dim() == 2, "Input must be 2D"

    N = attention.shape[0]
    triangles = torch.empty(N, dtype=attention.dtype, device=attention.device)

    BLOCK_SIZE = 64
    grid = (N,)

    compute_triangles_kernel[grid](
        attention,
        N,
        triangles,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return triangles


def compute_frc_triton(attention: torch.Tensor) -> torch.Tensor:
    """
    Compute Forman-Ricci Curvature using Triton kernels.

    Args:
        attention: [N, N] attention matrix (can be sparse or dense)

    Returns:
        frc_scores: [N] FRC scores for each node

    Note:
        Lower FRC = bridge/bottleneck (keep these!)
        Higher FRC = redundant/well-connected (can prune)
    """
    assert attention.is_cuda, "Input must be on CUDA"
    assert attention.dim() == 2, "Input must be 2D"

    N = attention.shape[0]

    # Step 1: Compute node strengths
    strengths = compute_node_strengths_triton(attention)

    # Step 2: Compute triangles
    triangles = compute_triangles_triton(attention)

    # Step 3: Compute FRC
    frc_scores = torch.empty(N, dtype=attention.dtype, device=attention.device)

    BLOCK_SIZE = 128
    grid = (N,)

    compute_frc_kernel[grid](
        attention,
        strengths,
        triangles,
        N,
        frc_scores,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return frc_scores


# PyTorch fallback for CPU or debugging
def compute_frc_pytorch(attention: torch.Tensor) -> torch.Tensor:
    """
    PyTorch fallback for FRC computation (CPU-compatible).

    Args:
        attention: [N, N] attention matrix

    Returns:
        frc_scores: [N] FRC scores
    """
    N = attention.shape[0]

    # Node strengths: sum of row (outgoing edges)
    strengths = attention.sum(dim=1)  # [N]

    # Triangles: (A @ A).diagonal()
    # This approximates sum_j sum_k w_ij * w_jk
    attention_squared = torch.matmul(attention, attention)
    triangles = attention_squared.sum(dim=1)  # [N]

    # FRC formula
    w = strengths  # Node weight ≈ strength for directed graphs
    frc_scores = 4.0 * w - 2.0 * strengths + 3.0 * triangles

    return frc_scores


def compute_frc(attention: torch.Tensor, use_triton: bool = True) -> torch.Tensor:
    """
    Compute Forman-Ricci Curvature with automatic backend selection.

    Args:
        attention: [N, N] attention matrix
        use_triton: Use Triton kernel if available and on CUDA

    Returns:
        frc_scores: [N] FRC scores
    """
    if use_triton and attention.is_cuda:
        try:
            return compute_frc_triton(attention)
        except Exception as e:
            print(f"Warning: Triton kernel failed ({e}), falling back to PyTorch")
            return compute_frc_pytorch(attention)
    else:
        return compute_frc_pytorch(attention)


if __name__ == "__main__":
    # Test FRC computation
    print("Testing FRC kernels...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create synthetic attention matrix
    N = 512
    attention = torch.rand(N, N, device=device)
    attention = attention / attention.sum(dim=1, keepdim=True)  # Normalize

    # Compute FRC
    import time

    if device == 'cuda':
        # Triton version
        torch.cuda.synchronize()
        start = time.time()
        frc_triton = compute_frc_triton(attention)
        torch.cuda.synchronize()
        triton_time = (time.time() - start) * 1000

        print(f"Triton FRC: {triton_time:.2f}ms")
        print(f"  Mean: {frc_triton.mean():.4f}")
        print(f"  Std: {frc_triton.std():.4f}")
        print(f"  Min: {frc_triton.min():.4f}")
        print(f"  Max: {frc_triton.max():.4f}")

    # PyTorch version (reference)
    start = time.time()
    frc_pytorch = compute_frc_pytorch(attention)
    pytorch_time = (time.time() - start) * 1000

    print(f"\nPyTorch FRC: {pytorch_time:.2f}ms")
    print(f"  Mean: {frc_pytorch.mean():.4f}")
    print(f"  Std: {frc_pytorch.std():.4f}")
    print(f"  Min: {frc_pytorch.min():.4f}")
    print(f"  Max: {frc_pytorch.max():.4f}")

    if device == 'cuda':
        print(f"\nSpeedup: {pytorch_time / triton_time:.2f}x")

        # Verify correctness
        error = (frc_triton - frc_pytorch).abs().max()
        print(f"Max error: {error:.6f}")
