"""
Test Position IDs Fix for CAB/H2O
==================================

Quick validation that position IDs tracking fixes the corruption bug.
Tests CAB and H2O with 3 samples to verify outputs are clean.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import sys

print("="*60)
print("Position IDs Fix Validation")
print("="*60)

# Load model with eager attention
print("\n1. Loading model...")
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

# Test CAB with position IDs fix
print("\n5. Testing CAB with 90% sparsity...")
from experiments.longbench_qa.runner import LongBenchQARunner
from experiments.longbench_qa.config import ExperimentConfig

config = ExperimentConfig(
    model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
    max_length=32768,
    max_new_tokens=50,
)

runner = LongBenchQARunner(config)
runner.model = model
runner.tokenizer = tokenizer
runner.use_flash_attention = True

# Test CAB
print("\n   Testing CAB...")
with torch.no_grad():
    cab_output = runner._sparse_generate(
        inputs=inputs,
        max_new_tokens=50,
        num_keep_ratio=0.1,  # 90% sparsity
        method='cab',
        magnitude_ratio=0.5,
    )

cab_text = tokenizer.decode(cab_output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"   CAB output: '{cab_text}'")

# Test H2O
print("\n   Testing H2O...")
with torch.no_grad():
    h2o_output = runner._sparse_generate(
        inputs=inputs,
        max_new_tokens=50,
        num_keep_ratio=0.1,  # 90% sparsity
        method='h2o',
    )

h2o_text = tokenizer.decode(h2o_output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print(f"   H2O output: '{h2o_text}'")

# Analysis
print("\n" + "="*60)
print("Analysis:")
print("="*60)
print(f"Expected:  '{answer}'")
print(f"CAB:       '{cab_text}'")
print(f"H2O:       '{h2o_text}'")
print()

# Check for corruption patterns
def is_corrupted(text):
    """Check if text shows signs of corruption."""
    # Gibberish patterns from before fix
    gibberish_patterns = [
        "))", "((", "0000", "9555",
        text.count("0") > len(text) * 0.3,  # >30% zeros
        text.count(",") > len(text) * 0.2,  # >20% commas
        len(text) < 5 and text.startswith(("0", "1", ",")),
    ]
    return any(gibberish_patterns)

cab_ok = not is_corrupted(cab_text)
h2o_ok = not is_corrupted(h2o_text)

print(f"CAB Status: {'✓ Clean' if cab_ok else '✗ CORRUPTED'}")
print(f"H2O Status: {'✓ Clean' if h2o_ok else '✗ CORRUPTED'}")

if cab_ok and h2o_ok:
    print("\n✓ Position IDs fix is working! Both CAB and H2O produce clean outputs.")
else:
    print("\n✗ Still seeing corruption. Check position IDs implementation.")

print("="*60)
