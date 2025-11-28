#!/usr/bin/env python3
"""
Quick test to verify the two critical fixes:
1. Triton kernel (no continue statement)
2. Eviction policy (length mismatch handling)
"""

import torch

def test_triton_kernel():
    """Test that Triton kernel works now."""
    print("Testing Triton FRC kernel fix...")
    from cab_attention.kernels.frc_triton import compute_frc_triton

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device != 'cuda':
        print("  Skipping (CUDA not available)")
        return True

    N = 256
    attention = torch.rand(N, N, device=device)
    attention = attention / attention.sum(dim=1, keepdim=True)

    try:
        frc_scores = compute_frc_triton(attention)
        print(f"  ✓ Triton kernel working! FRC shape: {frc_scores.shape}")
        return True
    except Exception as e:
        print(f"  ✗ Triton kernel failed: {e}")
        return False


def test_eviction_policy():
    """Test that eviction policy handles length mismatches."""
    print("\nTesting eviction policy length handling...")
    from cab_attention.eviction import ThreeComponentEvictionPolicy, EvictionConfig

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = EvictionConfig(local_ratio=0.3, bridge_ratio=0.2, importance_ratio=0.5)
    policy = ThreeComponentEvictionPolicy(config)

    # Test case: importance_scores shorter than cache_len
    cache_len = 100
    keep_size = 10

    # Simulate stale importance scores (length mismatch)
    importance_scores = torch.rand(90, device=device)  # 90 < 100
    frc_scores = torch.randn(95, device=device)       # 95 < 100

    try:
        keep_indices, diagnostics = policy.select_indices(
            cache_len=cache_len,
            keep_size=keep_size,
            importance_scores=importance_scores,
            frc_scores=frc_scores,
            device=device,
        )
        print(f"  ✓ Eviction policy working! Selected {len(keep_indices)} indices")
        print(f"    Local: {diagnostics['local_count']}, "
              f"Importance: {diagnostics['importance_count']}, "
              f"Bridges: {diagnostics['bridge_count']}")
        return True
    except Exception as e:
        print(f"  ✗ Eviction policy failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cab_cache():
    """Test that CAB cache works end-to-end."""
    print("\nTesting CAB cache with fixes...")
    from cab_attention import CABCache

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cache = CABCache(
        max_cache_size=50,
        sparsity=0.9,
        eviction_interval=5,
        device=device,
    )

    B, H, D = 1, 8, 64

    try:
        # Simulate 100 generation steps
        for step in range(100):
            for layer_idx in range(2):
                key_state = torch.randn(B, H, 1, D, device=device)
                value_state = torch.randn(B, H, 1, D, device=device)

                attention = None
                if layer_idx == 0 and cache.get_seq_length(0) > 0:
                    cache_len = cache.get_seq_length(0)
                    attention = torch.rand(B, H, 1, cache_len, device=device)
                    attention = attention / attention.sum(dim=-1, keepdim=True)

                keys, values = cache.update(key_state, value_state, layer_idx, attention)

        stats = cache.get_stats()
        print(f"  ✓ CAB cache working!")
        print(f"    Final size: {stats['current_cache_size']}/{stats['max_cache_size']}")
        print(f"    Total evictions: {stats['total_evictions']}")
        return True
    except Exception as e:
        print(f"  ✗ CAB cache failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("Testing Critical Fixes")
    print("=" * 60)

    results = {
        'Triton kernel': test_triton_kernel(),
        'Eviction policy': test_eviction_policy(),
        'CAB cache': test_cab_cache(),
    }

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n{passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
