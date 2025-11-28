"""
Debug CAB Corruption
====================

Investigate why CAB generates corrupted outputs like:
- "To be in) in) 1-111"
- "The0. 33. 0. 1. Constraints"
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys

print("="*60)
print("CAB Corruption Debug")
print("="*60)

# Load model with eager attention (required for CAB)
print("\n1. Loading model with eager attention...")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda",
    trust_remote_code=True,
    attn_implementation='eager',
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)

# Patch with Flash Attention
print("\n2. Patching with custom Flash Attention...")
from cab_attention.kernels.flash_attention_accumulate import replace_attention_with_flash
model = replace_attention_with_flash(model)
model.eval()

# Load sample
print("\n3. Loading HotPotQA sample...")
with open('/root/FRC/experiments/longbench_qa/data/hotpotqa/longbench_raw/hotpotqa.jsonl', 'r') as f:
    sample = json.loads(f.readline())

context = sample['context']
question = sample['input']
answer = sample['answers'][0]

print(f"   Question: {question}")
print(f"   Expected: {answer}")

# Build prompt
prompt = f"""<|im_start|>system
You are a helpful assistant. Answer questions concisely based on the given context.<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}

Answer with only the answer, nothing else.<|im_end|>
<|im_start|>assistant
"""

# Tokenize
inputs = tokenizer(prompt, return_tensors='pt', truncation=False).to('cuda')
print(f"\n4. Input tokens: {inputs['input_ids'].shape[1]}")

# Test 1: Dense generation (no eviction)
print("\n5. Testing DENSE (no eviction)...")
with torch.no_grad():
    outputs_dense = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
generated_dense = tokenizer.decode(outputs_dense[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"   Dense output: '{generated_dense}'")

# Check cumulative scores
first_layer = model.model.layers[0].self_attn
if hasattr(first_layer, 'cumulative_scores') and first_layer.cumulative_scores is not None:
    scores = first_layer.cumulative_scores
    print(f"   Cumulative scores shape: {scores.shape}")
    print(f"   Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"   Top 10 token scores: {scores[0, 0, :10].cpu().numpy()}")
else:
    print(f"   ✗ No cumulative scores!")

# Clear scores for next test
for layer in model.model.layers:
    if hasattr(layer.self_attn, 'cumulative_scores'):
        layer.self_attn.cumulative_scores = None

# Test 2: CAB with eviction
print("\n6. Testing CAB with 90% sparsity...")

# Manual generation with KV cache eviction
from cab_attention.eviction.policy import CABEvictionPolicy

eviction_policy = CABEvictionPolicy(
    sparsity=0.9,
    magnitude_ratio=0.5,
)

# Initial prefill
print("   Running prefill...")
with torch.no_grad():
    prefill_outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs.get('attention_mask'),
        use_cache=True,
    )

past_key_values = prefill_outputs.past_key_values
generated_tokens = []

print(f"   Initial KV cache size: {past_key_values[0][0].shape[2]} tokens")

# Check cumulative scores after prefill
if hasattr(first_layer, 'cumulative_scores') and first_layer.cumulative_scores is not None:
    scores_prefill = first_layer.cumulative_scores
    print(f"   Cumulative scores after prefill: {scores_prefill.shape}")
    print(f"   Score stats: min={scores_prefill.min():.3f}, max={scores_prefill.max():.3f}, mean={scores_prefill.mean():.3f}")

    # Show top tokens by score
    top_indices = torch.topk(scores_prefill[0, 0], k=10).indices
    print(f"   Top 10 token indices by score: {top_indices.cpu().numpy()}")

    # Show bottom tokens (will be evicted)
    bottom_indices = torch.topk(scores_prefill[0, 0], k=10, largest=False).indices
    print(f"   Bottom 10 token indices (to evict): {bottom_indices.cpu().numpy()}")
else:
    print(f"   ✗ No cumulative scores after prefill!")
    sys.exit(1)

# Generate tokens with eviction
for step in range(min(10, 50)):
    # Apply eviction
    if past_key_values[0][0].shape[2] > 100:  # Only evict if cache is large enough
        print(f"\n   Step {step}: Applying eviction...")
        print(f"      Before eviction: {past_key_values[0][0].shape[2]} tokens")

        past_key_values = eviction_policy.evict(
            model=model,
            past_key_values=past_key_values,
        )

        print(f"      After eviction: {past_key_values[0][0].shape[2]} tokens")

        # Check if eviction corrupted anything
        if past_key_values[0][0].shape[2] == 0:
            print(f"      ✗ ERROR: All tokens evicted!")
            break

    # Get next token
    next_token_logits = prefill_outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    generated_tokens.append(next_token.item())

    # Decode token
    token_str = tokenizer.decode([next_token.item()], skip_special_tokens=False)
    print(f"      Generated token {step}: '{token_str}' (id={next_token.item()})")

    # Check for EOS
    if next_token.item() == tokenizer.eos_token_id:
        print(f"      EOS generated, stopping")
        break

    # Forward pass for next token
    with torch.no_grad():
        prefill_outputs = model(
            input_ids=next_token,
            attention_mask=None,
            past_key_values=past_key_values,
            use_cache=True,
        )
    past_key_values = prefill_outputs.past_key_values

# Decode CAB output
cab_output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(f"\n7. CAB output: '{cab_output}'")

print("\n" + "="*60)
print("Analysis:")
print("="*60)
print(f"Dense:  '{generated_dense}'")
print(f"CAB:    '{cab_output}'")
print("\nIf CAB output is corrupted, check:")
print("1. Are cumulative scores being tracked correctly?")
print("2. Are the right tokens being evicted?")
print("3. Is position encoding preserved after eviction?")
print("4. Are KV cache indices correct after eviction?")
print("="*60)
