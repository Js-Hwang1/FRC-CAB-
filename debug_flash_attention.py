"""
Debug script to identify Flash Attention bug.
Compares SDPA vs Flash Attention outputs and tensor shapes.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def test_flash_attention_shapes():
    """Test to identify shape mismatches or corruption in Flash Attention."""

    print("="*80)
    print("FLASH ATTENTION DEBUG TEST")
    print("="*80)

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Load model with eager attention (no Flash)
    print(f"\nLoading {model_name} with eager attention...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Test input
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    print(f"\n{'='*80}")
    print("TEST 1: Check position_embeddings format")
    print(f"{'='*80}")

    # Patch first attention layer to inspect position_embeddings
    original_forward = model.model.layers[0].self_attn.forward

    captured_args = {}

    def capture_forward(*args, **kwargs):
        captured_args['args'] = args
        captured_args['kwargs'] = kwargs
        return original_forward(*args, **kwargs)

    model.model.layers[0].self_attn.forward = capture_forward

    # Run a single forward pass
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    # Check what was passed
    print("\nInspecting attention forward arguments:")
    if captured_args:
        print(f"  Number of positional args: {len(captured_args.get('args', []))}")
        print(f"  Keyword args: {list(captured_args.get('kwargs', {}).keys())}")

        # Check position_embeddings
        if 'position_embeddings' in captured_args['kwargs']:
            pos_emb = captured_args['kwargs']['position_embeddings']
            print(f"\n  position_embeddings type: {type(pos_emb)}")
            if isinstance(pos_emb, tuple):
                print(f"  position_embeddings length: {len(pos_emb)}")
                for i, tensor in enumerate(pos_emb):
                    if isinstance(tensor, torch.Tensor):
                        print(f"    Element {i} shape: {tensor.shape}, dtype: {tensor.dtype}")
                    else:
                        print(f"    Element {i} type: {type(tensor)}")

        # Check hidden_states
        if len(captured_args.get('args', [])) > 0:
            hidden_states = captured_args['args'][0]
            print(f"\n  hidden_states shape: {hidden_states.shape}")

    print(f"\n{'='*80}")
    print("TEST 2: Compare SDPA vs Flash Attention outputs")
    print(f"{'='*80}")

    # Generate with SDPA (baseline)
    print("\nGenerating with SDPA...")
    model.model.layers[0].self_attn.forward = original_forward  # Restore original
    with torch.no_grad():
        sdpa_output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
        )
    sdpa_text = tokenizer.decode(sdpa_output[0], skip_special_tokens=True)
    print(f"SDPA result: {sdpa_text}")

    # Now patch with Flash Attention
    print("\nPatching with Flash Attention...")
    from cab_attention.kernels.flash_attention_accumulate import replace_attention_with_flash

    # Reload model to reset state
    del model
    torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",  # Start with eager, then patch
        trust_remote_code=True,
    )

    # Apply Flash Attention patch
    model = replace_attention_with_flash(model)

    # Generate with Flash Attention
    print("Generating with Flash Attention...")
    with torch.no_grad():
        flash_output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
        )
    flash_text = tokenizer.decode(flash_output[0], skip_special_tokens=True)
    print(f"Flash result: {flash_text}")

    print(f"\n{'='*80}")
    print("COMPARISON")
    print(f"{'='*80}")
    print(f"SDPA:  {sdpa_text}")
    print(f"Flash: {flash_text}")
    print(f"Match: {sdpa_text == flash_text}")

    if sdpa_text != flash_text:
        print("\n❌ Flash Attention produces different output!")
        print("This confirms the bug exists.")
    else:
        print("\n✅ Flash Attention produces same output as SDPA!")
        print("Bug may be fixed or test case insufficient.")

    print(f"\n{'='*80}")
    print("TEST 3: Check RoPE application")
    print(f"{'='*80}")

    # Test RoPE function directly
    from cab_attention.kernels.flash_attention_accumulate import apply_rotary_pos_emb, rotate_half

    # Create dummy tensors
    B, H, L, D = 1, 8, 5, 64
    q = torch.randn(B, H, L, D, device='cuda', dtype=torch.float16)
    k = torch.randn(B, H, L, D, device='cuda', dtype=torch.float16)

    # Try different cos/sin shapes to see which works
    shapes_to_try = [
        (B, L, D),      # Expected by current implementation
        (B, 1, L, D),   # Already has head dimension
        (1, L, D),      # No batch dimension
        (L, D),         # Neither batch nor head dimension
    ]

    for shape in shapes_to_try:
        try:
            cos = torch.randn(*shape, device='cuda', dtype=torch.float16)
            sin = torch.randn(*shape, device='cuda', dtype=torch.float16)

            q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
            print(f"  cos/sin shape {shape}: ✓ (output shape: {q_rot.shape})")
        except Exception as e:
            print(f"  cos/sin shape {shape}: ✗ ({type(e).__name__}: {str(e)[:50]})")

if __name__ == "__main__":
    test_flash_attention_shapes()
