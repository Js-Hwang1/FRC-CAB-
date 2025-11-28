"""
Debug script to check if cumulative scores are being extracted correctly.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_cumulative_scores():
    """Test if Flash Attention cumulative scores are being tracked."""

    print("="*80)
    print("CUMULATIVE SCORES DEBUG")
    print("="*80)

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Load model with eager attention
    print(f"\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Patch with Flash Attention
    print("Patching with Flash Attention...")
    from cab_attention.kernels.flash_attention_accumulate import replace_attention_with_flash, get_all_cumulative_scores
    model = replace_attention_with_flash(model)

    # Test input
    prompt = "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"\nInput: {prompt}")
    print(f"Input length: {inputs['input_ids'].shape[1]} tokens")

    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False, return_dict=True)

    # Check cumulative scores
    print("\nChecking cumulative scores...")
    all_scores = get_all_cumulative_scores(model)

    if not all_scores:
        print("❌ No cumulative scores found!")
        print("Flash Attention may not be accumulating scores correctly.")
        return False

    print(f"✓ Found scores from {len(all_scores)} layers")

    # Check a few layers
    for i, (layer_name, scores) in enumerate(list(all_scores.items())[:3]):
        print(f"\n  Layer {i} ({layer_name}):")
        print(f"    Shape: {scores.shape}")
        print(f"    Dtype: {scores.dtype}")
        print(f"    Range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"    Sum: {scores.sum():.4f}")
        print(f"    Non-zero: {(scores > 0).sum().item()}/{scores.numel()}")

    # Aggregate scores
    print("\nAggregating scores across layers...")
    aggregated = None
    for layer_name, scores in all_scores.items():
        layer_contrib = scores.sum(dim=(0, 1))  # [B, H, N] -> [N]

        if aggregated is None:
            aggregated = layer_contrib
        else:
            aggregated += layer_contrib

    print(f"  Aggregated shape: {aggregated.shape}")
    print(f"  Aggregated range: [{aggregated.min():.4f}, {aggregated.max():.4f}]")
    print(f"  Aggregated sum: {aggregated.sum():.4f}")

    # Check if scores are reasonable
    expected_length = inputs['input_ids'].shape[1]
    actual_length = aggregated.shape[0]

    if expected_length != actual_length:
        print(f"\n⚠️  Length mismatch: expected {expected_length}, got {actual_length}")
    else:
        print(f"\n✓ Length matches input: {actual_length}")

    if aggregated.sum() == 0:
        print("❌ All scores are zero! Flash Attention is not accumulating correctly.")
        return False
    else:
        print(f"✓ Scores are non-zero (sum={aggregated.sum():.4f})")

    print("\n" + "="*80)
    print("✅ CUMULATIVE SCORES WORKING")
    print("="*80)
    return True

if __name__ == "__main__":
    try:
        success = test_cumulative_scores()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
