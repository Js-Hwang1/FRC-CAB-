"""
Deep dive debug: Compare SDPA vs Flash Attention at tensor level.
This tests a single attention computation to isolate the kernel bug.
"""

import torch
import torch.nn.functional as F
from cab_attention.kernels.flash_attention_accumulate import flash_attention_with_cumulative_scores

def test_flash_kernel_directly():
    """Test Flash Attention kernel against SDPA on same inputs."""

    print("="*80)
    print("FLASH ATTENTION KERNEL DEBUG")
    print("="*80)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # Create test tensors (simulating a single attention layer)
    B, H, N, D = 1, 8, 16, 64  # Small sizes for debugging
    device = 'cuda'
    dtype = torch.float16

    print(f"\nTest configuration:")
    print(f"  Batch size: {B}")
    print(f"  Num heads: {H}")
    print(f"  Sequence length: {N}")
    print(f"  Head dim: {D}")

    # Create random Q, K, V
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)

    scale = 1.0 / (D ** 0.5)

    print(f"\n{'='*80}")
    print("TEST 1: Self-attention (N_q == N_k)")
    print(f"{'='*80}")

    # Compute with SDPA
    print("\nComputing with SDPA...")
    with torch.no_grad():
        sdpa_output = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False, scale=scale
        )

    # Compute with Flash Attention
    print("Computing with Flash Attention...")
    with torch.no_grad():
        flash_output, cumulative = flash_attention_with_cumulative_scores(
            q, k, v, cumulative_scores=None, scale=scale
        )

    # Compare outputs
    max_diff = (sdpa_output - flash_output).abs().max().item()
    mean_diff = (sdpa_output - flash_output).abs().mean().item()

    print(f"\nOutput comparison:")
    print(f"  SDPA output range: [{sdpa_output.min():.4f}, {sdpa_output.max():.4f}]")
    print(f"  Flash output range: [{flash_output.min():.4f}, {flash_output.max():.4f}]")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Mean difference: {mean_diff:.6f}")

    # Check if outputs match (within numerical precision)
    tolerance = 1e-2  # Float16 has limited precision
    if max_diff < tolerance:
        print(f"  ✅ Outputs match (max diff {max_diff:.6f} < {tolerance})")
    else:
        print(f"  ❌ Outputs differ significantly (max diff {max_diff:.6f} >= {tolerance})")

    print(f"\n{'='*80}")
    print("TEST 2: KV cache scenario (N_q < N_k)")
    print(f"{'='*80}")

    # Simulate generation: Q has 1 token, K/V have N tokens (cached)
    N_q, N_k = 1, 16
    q_gen = torch.randn(B, H, N_q, D, device=device, dtype=dtype)
    k_cache = torch.randn(B, H, N_k, D, device=device, dtype=dtype)
    v_cache = torch.randn(B, H, N_k, D, device=device, dtype=dtype)

    print(f"\n  Q shape: {list(q_gen.shape)} (new token)")
    print(f"  K shape: {list(k_cache.shape)} (cached)")
    print(f"  V shape: {list(v_cache.shape)} (cached)")

    # Compute with SDPA
    print("\nComputing with SDPA...")
    with torch.no_grad():
        sdpa_gen_output = F.scaled_dot_product_attention(
            q_gen, k_cache, v_cache, dropout_p=0.0, is_causal=False, scale=scale
        )

    # Compute with Flash Attention
    print("Computing with Flash Attention...")
    with torch.no_grad():
        flash_gen_output, cumulative_gen = flash_attention_with_cumulative_scores(
            q_gen, k_cache, v_cache, cumulative_scores=None, scale=scale
        )

    # Compare outputs
    max_diff_gen = (sdpa_gen_output - flash_gen_output).abs().max().item()
    mean_diff_gen = (sdpa_gen_output - flash_gen_output).abs().mean().item()

    print(f"\nOutput comparison (KV cache scenario):")
    print(f"  SDPA output range: [{sdpa_gen_output.min():.4f}, {sdpa_gen_output.max():.4f}]")
    print(f"  Flash output range: [{flash_gen_output.min():.4f}, {flash_gen_output.max():.4f}]")
    print(f"  Max difference: {max_diff_gen:.6f}")
    print(f"  Mean difference: {mean_diff_gen:.6f}")

    if max_diff_gen < tolerance:
        print(f"  ✅ Outputs match (max diff {max_diff_gen:.6f} < {tolerance})")
    else:
        print(f"  ❌ Outputs differ significantly (max diff {max_diff_gen:.6f} >= {tolerance})")

    print(f"\n{'='*80}")
    print("TEST 3: Larger sequence (stress test)")
    print(f"{'='*80}")

    # Test with larger sequence (closer to real usage)
    N_large = 512
    q_large = torch.randn(B, H, N_large, D, device=device, dtype=dtype)
    k_large = torch.randn(B, H, N_large, D, device=device, dtype=dtype)
    v_large = torch.randn(B, H, N_large, D, device=device, dtype=dtype)

    print(f"\n  Sequence length: {N_large}")

    # Compute with SDPA
    print("Computing with SDPA...")
    with torch.no_grad():
        sdpa_large_output = F.scaled_dot_product_attention(
            q_large, k_large, v_large, dropout_p=0.0, is_causal=False, scale=scale
        )

    # Compute with Flash Attention
    print("Computing with Flash Attention...")
    with torch.no_grad():
        flash_large_output, _ = flash_attention_with_cumulative_scores(
            q_large, k_large, v_large, cumulative_scores=None, scale=scale
        )

    # Compare outputs
    max_diff_large = (sdpa_large_output - flash_large_output).abs().max().item()
    mean_diff_large = (sdpa_large_output - flash_large_output).abs().mean().item()

    print(f"\nOutput comparison (large sequence):")
    print(f"  Max difference: {max_diff_large:.6f}")
    print(f"  Mean difference: {mean_diff_large:.6f}")

    if max_diff_large < tolerance:
        print(f"  ✅ Outputs match (max diff {max_diff_large:.6f} < {tolerance})")
    else:
        print(f"  ❌ Outputs differ significantly (max diff {max_diff_large:.6f} >= {tolerance})")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    all_pass = all([
        max_diff < tolerance,
        max_diff_gen < tolerance,
        max_diff_large < tolerance,
    ])

    if all_pass:
        print("✅ All tests passed! Flash Attention kernel is correct.")
        print("The bug must be in the integration code (forward wrapper).")
    else:
        print("❌ Flash Attention kernel produces incorrect outputs.")
        print("The bug is in the Triton kernel implementation.")

    return all_pass

if __name__ == "__main__":
    try:
        result = test_flash_kernel_directly()
        exit(0 if result else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
