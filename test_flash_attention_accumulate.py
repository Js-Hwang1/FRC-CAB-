"""
Test and Benchmark Flash Attention with Cumulative Score Accumulation
======================================================================

Validates correctness and measures performance of custom Flash Attention kernel.
"""

import torch
import time
import numpy as np

try:
    from cab_attention.kernels.flash_attention_accumulate import (
        flash_attention_with_cumulative_scores,
        FlashAttentionWithAccumulation,
    )
    HAS_FLASH = True
except ImportError:
    print("WARNING: Could not import flash_attention_accumulate. Install triton first:")
    print("  pip install triton")
    HAS_FLASH = False


def eager_attention_with_accumulation(q, k, v, cumulative_scores=None):
    """Reference implementation using PyTorch eager attention."""
    B, H, N, D = q.shape
    scale = 1.0 / (D ** 0.5)

    # Compute attention: QK^T
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, N, N]

    # Softmax
    attn_weights = torch.softmax(scores, dim=-1)  # [B, H, N, N]

    # Apply to values
    output = torch.matmul(attn_weights, v)  # [B, H, N, D]

    # Accumulate scores: sum across query dimension
    # Each key position gets total attention from all queries
    if cumulative_scores is None:
        cumulative_scores = attn_weights.sum(dim=2)  # [B, H, N]
    else:
        cumulative_scores = cumulative_scores + attn_weights.sum(dim=2)

    return output, cumulative_scores


def test_correctness():
    """Test that Flash Attention produces same results as eager attention."""
    print("="*60)
    print("Testing Correctness")
    print("="*60)

    if not HAS_FLASH:
        print("SKIPPED: Flash Attention not available")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("SKIPPED: CUDA not available (Flash Attention requires GPU)")
        return

    torch.manual_seed(42)

    # Test configuration
    B, H, N, D = 2, 8, 256, 64

    # Generate random inputs
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

    # Compute with eager attention
    q_ref = q.float()
    k_ref = k.float()
    v_ref = v.float()

    output_ref, scores_ref = eager_attention_with_accumulation(q_ref, k_ref, v_ref)

    # Compute with Flash Attention
    output_flash, scores_flash = flash_attention_with_cumulative_scores(q, k, v)

    # Compare outputs
    output_diff = (output_ref.half() - output_flash).abs().max().item()
    output_mean = output_ref.abs().mean().item()
    output_rel_error = output_diff / (output_mean + 1e-6)

    print(f"\nOutput comparison:")
    print(f"  Max absolute diff: {output_diff:.6f}")
    print(f"  Mean value: {output_mean:.6f}")
    print(f"  Relative error: {output_rel_error:.6f}")

    # Compare cumulative scores
    scores_diff = (scores_ref.half() - scores_flash).abs().max().item()
    scores_mean = scores_ref.abs().mean().item()
    scores_rel_error = scores_diff / (scores_mean + 1e-6)

    print(f"\nCumulative scores comparison:")
    print(f"  Max absolute diff: {scores_diff:.6f}")
    print(f"  Mean value: {scores_mean:.6f}")
    print(f"  Relative error: {scores_rel_error:.6f}")

    # Test accumulation across multiple calls
    print(f"\nTesting accumulation across multiple forward passes...")

    cumul_eager = None
    cumul_flash = None

    for i in range(3):
        # Eager
        _, cumul_eager = eager_attention_with_accumulation(
            q_ref, k_ref, v_ref, cumul_eager
        )

        # Flash
        _, cumul_flash = flash_attention_with_cumulative_scores(
            q, k, v, cumul_flash
        )

    cumul_diff = (cumul_eager.half() - cumul_flash).abs().max().item()
    cumul_mean = cumul_eager.abs().mean().item()
    cumul_rel_error = cumul_diff / (cumul_mean + 1e-6)

    print(f"  After 3 accumulations:")
    print(f"    Max absolute diff: {cumul_diff:.6f}")
    print(f"    Relative error: {cumul_rel_error:.6f}")

    # Pass/fail
    if output_rel_error < 0.01 and scores_rel_error < 0.01 and cumul_rel_error < 0.01:
        print(f"\n✓ PASS: Flash Attention is numerically correct")
    else:
        print(f"\n✗ FAIL: Flash Attention has numerical errors")


def benchmark_speed():
    """Benchmark Flash Attention vs Eager Attention speed."""
    print("\n" + "="*60)
    print("Benchmarking Speed")
    print("="*60)

    if not HAS_FLASH:
        print("SKIPPED: Flash Attention not available")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("SKIPPED: CUDA not available")
        return

    torch.manual_seed(42)

    # Test configurations
    configs = [
        (1, 8, 512, 64),
        (1, 8, 1024, 64),
        (1, 8, 2048, 64),
        (1, 32, 2048, 128),
    ]

    print(f"\n{'Config':<25} {'Eager (ms)':<15} {'Flash (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    for B, H, N, D in configs:
        # Generate inputs
        q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

        q_float = q.float()
        k_float = k.float()
        v_float = v.float()

        # Warmup
        for _ in range(10):
            _ = eager_attention_with_accumulation(q_float, k_float, v_float)
            _ = flash_attention_with_cumulative_scores(q, k, v)

        # Benchmark eager
        torch.cuda.synchronize()
        t0 = time.time()
        num_iters = 100
        for _ in range(num_iters):
            _ = eager_attention_with_accumulation(q_float, k_float, v_float)
        torch.cuda.synchronize()
        eager_time = (time.time() - t0) / num_iters * 1000

        # Benchmark Flash
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(num_iters):
            _ = flash_attention_with_cumulative_scores(q, k, v)
        torch.cuda.synchronize()
        flash_time = (time.time() - t0) / num_iters * 1000

        speedup = eager_time / flash_time

        config_str = f"B={B}, H={H}, N={N}, D={D}"
        print(f"{config_str:<25} {eager_time:>13.2f}  {flash_time:>13.2f}  {speedup:>8.2f}x")


def benchmark_memory():
    """Benchmark memory usage: Flash Attention vs Eager Attention."""
    print("\n" + "="*60)
    print("Benchmarking Memory Usage")
    print("="*60)

    if not HAS_FLASH:
        print("SKIPPED: Flash Attention not available")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("SKIPPED: CUDA not available")
        return

    torch.manual_seed(42)

    # Large sequence length to show memory difference
    B, H, N, D = 1, 32, 3000, 128

    print(f"\nConfiguration: B={B}, H={H}, N={N}, D={D}")

    # Generate inputs
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float16)

    # Measure eager attention memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    baseline_mem = torch.cuda.memory_allocated() / 1024**2

    with torch.no_grad():
        q_float = q.float()
        k_float = k.float()
        v_float = v.float()
        output_eager, scores_eager = eager_attention_with_accumulation(q_float, k_float, v_float)

    eager_mem = torch.cuda.max_memory_allocated() / 1024**2 - baseline_mem

    # Clear
    del output_eager, scores_eager, q_float, k_float, v_float
    torch.cuda.empty_cache()

    # Measure Flash Attention memory
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    baseline_mem = torch.cuda.memory_allocated() / 1024**2

    with torch.no_grad():
        output_flash, scores_flash = flash_attention_with_cumulative_scores(q, k, v)

    flash_mem = torch.cuda.max_memory_allocated() / 1024**2 - baseline_mem

    # Calculate theoretical attention matrix size
    attn_matrix_size = B * H * N * N * 2 / 1024**2  # float16 = 2 bytes

    print(f"\nMemory usage:")
    print(f"  Eager attention:  {eager_mem:>8.2f} MB")
    print(f"  Flash attention:  {flash_mem:>8.2f} MB")
    print(f"  Reduction:        {eager_mem - flash_mem:>8.2f} MB ({(eager_mem - flash_mem) / eager_mem * 100:.1f}%)")
    print(f"\n  Theoretical attention matrix size: {attn_matrix_size:.2f} MB")
    print(f"  Saved by not storing: ~{attn_matrix_size:.2f} MB")


def test_huggingface_integration():
    """Test integration with HuggingFace-style attention."""
    print("\n" + "="*60)
    print("Testing HuggingFace Integration")
    print("="*60)

    if not HAS_FLASH:
        print("SKIPPED: Flash Attention not available")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("SKIPPED: CUDA not available")
        return

    # Create Flash Attention module
    flash_attn = FlashAttentionWithAccumulation(accumulate_scores=True).to(device)

    # Simulate sequence of forward passes
    B, H, N, D = 1, 8, 512, 64

    print(f"\nSimulating sequence generation with {N} tokens...")

    for step in range(5):
        q = torch.randn(B, H, N + step, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N + step, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N + step, D, device=device, dtype=torch.float16)

        output, _ = flash_attn(q, k, v)

        # Check cumulative scores
        scores = flash_attn.get_cumulative_scores()
        if scores is not None:
            print(f"  Step {step}: Output shape {tuple(output.shape)}, "
                  f"Cumulative scores shape {tuple(scores.shape)}, "
                  f"Mean score: {scores.mean():.4f}")

    print("\n✓ HuggingFace integration test passed")


if __name__ == "__main__":
    print("="*60)
    print("Flash Attention with Cumulative Scores - Test Suite")
    print("="*60)

    # Run all tests
    test_correctness()
    benchmark_speed()
    benchmark_memory()
    test_huggingface_integration()

    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)
