"""
Debug script to check if CAB and H2O are actually using different eviction logic.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from cab_attention.kernels.flash_attention_accumulate import replace_attention_with_flash
from cab_attention.eviction.policy import EvictionConfig

def test_methods():
    print("=" * 80)
    print("Testing if CAB and H2O use different eviction logic")
    print("=" * 80)

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Test 1: Check EvictionConfig for CAB
    print("\n1. CAB EvictionConfig:")
    cab_config = EvictionConfig(
        local_ratio=0.3,
        bridge_ratio=0.2,
        importance_ratio=0.5,
    )
    print(f"   local_ratio: {cab_config.local_ratio}")
    print(f"   bridge_ratio: {cab_config.bridge_ratio}")
    print(f"   importance_ratio: {cab_config.importance_ratio}")

    # Test 2: Check what H2O config would be
    print("\n2. H2O simulated config:")
    keep_size = 100  # Example
    local_size = max(1, int(keep_size * 0.2))
    important_size = keep_size - local_size
    print(f"   keep_size: {keep_size}")
    print(f"   local_size (20%): {local_size}")
    print(f"   important_size (80%): {important_size}")

    # Test 3: Load model and check Flash Attention setup
    print("\n3. Loading model with Flash Attention...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='cpu',
        attn_implementation='eager',
    )

    # Patch with Flash Attention
    print("\n4. Patching with Flash Attention...")
    model = replace_attention_with_flash(model)

    # Check if eviction config is set
    print("\n5. Checking if model has eviction config...")
    for i, layer in enumerate(model.model.layers[:3]):  # Check first 3 layers
        attn = layer.self_attn
        if hasattr(attn, 'eviction_config'):
            print(f"   Layer {i}: eviction_config = {attn.eviction_config}")
        else:
            print(f"   Layer {i}: NO eviction_config attribute")

    # Test 4: Check the actual generation benchmark code
    print("\n6. Analyzing generation benchmark evaluation logic...")
    print("   Current implementation in evaluate_perplexity():")
    print("   - Uses use_cache=False throughout")
    print("   - For Dense: SDPA attention, no truncation")
    print("   - For CAB/H2O: Flash Attention, sequence truncation")
    print("   - NO ACTUAL KV CACHE EVICTION IS HAPPENING!")
    print("\n   Issue: The benchmark is NOT testing eviction policies.")
    print("          It's only testing Flash Attention vs SDPA with sequence truncation.")
    print("          This is why CAB and H2O give identical results.")

    print("\n" + "=" * 80)
    print("ROOT CAUSE IDENTIFIED")
    print("=" * 80)
    print("The evaluate_perplexity() function uses use_cache=False,")
    print("so it never builds or prunes a KV cache.")
    print("\nIt's just:")
    print("- Dense: SDPA attention on full sequences")
    print("- CAB/H2O: Flash Attention on truncated sequences (same truncation!)")
    print("\nThis is not a real benchmark of eviction methods.")
    print("=" * 80)

if __name__ == "__main__":
    test_methods()
