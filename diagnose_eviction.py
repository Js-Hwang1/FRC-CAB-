"""
Diagnostic: Inspect which tokens are being kept vs evicted.
Shows top/bottom tokens by cumulative attention score.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

def diagnose_eviction():
    """Diagnose what tokens are kept vs evicted."""

    print("="*80)
    print("EVICTION DIAGNOSTIC")
    print("="*80)

    model_name = "Qwen/Qwen2.5-7B-Instruct"

    # Load model
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

    # Sample QA prompt
    context = """The wine region of McLaren Vale is located in South Australia, approximately 35 km south of Adelaide.
The region is known for its Mediterranean climate and premium Shiraz wine production. McLaren Vale has a
long history dating back to the 1830s when viticulture first began in the area."""

    question = "Mawson is an electoral district that includes the wine region around which town 55 km south of Adelaide?"

    # Format prompt (simple QA format)
    prompt = f"""Context: {context}

Question: {question}

Answer:"""

    print(f"\nContext length: {len(context)} chars")
    print(f"Question: {question}")
    print(f"Expected Answer: McLaren Vale")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_ids = inputs['input_ids'][0]
    tokens = [tokenizer.decode([t]) for t in input_ids]

    print(f"\nTotal tokens: {len(tokens)}")

    # Forward pass to get cumulative scores
    print("\nRunning forward pass...")
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False, return_dict=True)

    # Get cumulative scores
    all_scores = get_all_cumulative_scores(model)

    if not all_scores:
        print("❌ No cumulative scores found!")
        return False

    # Aggregate scores
    aggregated = None
    for layer_name, scores in all_scores.items():
        layer_contrib = scores.sum(dim=(0, 1))  # [B, H, N] -> [N]
        if aggregated is None:
            aggregated = layer_contrib
        else:
            aggregated += layer_contrib

    scores_np = aggregated.cpu().numpy()

    print(f"\nCumulative Scores:")
    print(f"  Range: [{scores_np.min():.2f}, {scores_np.max():.2f}]")
    print(f"  Mean: {scores_np.mean():.2f}")
    print(f"  Std: {scores_np.std():.2f}")

    # Analyze top and bottom scored tokens
    sorted_indices = aggregated.argsort(descending=True)

    print(f"\n{'='*80}")
    print("TOP 20 TOKENS (Highest cumulative attention - WILL BE KEPT)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Score':<12} {'Token':<20} {'Context'}")
    print("-"*80)

    for rank, idx in enumerate(sorted_indices[:20].tolist(), 1):
        score = scores_np[idx]
        token = tokens[idx]
        # Show context (3 tokens before and after)
        start = max(0, idx-2)
        end = min(len(tokens), idx+3)
        context_tokens = tokens[start:end]
        context_str = ''.join(context_tokens)
        print(f"{rank:<6} {score:<12.2f} {token:<20} ...{context_str[:50]}...")

    print(f"\n{'='*80}")
    print("BOTTOM 20 TOKENS (Lowest cumulative attention - WILL BE EVICTED)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Score':<12} {'Token':<20} {'Context'}")
    print("-"*80)

    for rank, idx in enumerate(sorted_indices[-20:].tolist(), 1):
        score = scores_np[idx]
        token = tokens[idx]
        # Show context
        start = max(0, idx-2)
        end = min(len(tokens), idx+3)
        context_tokens = tokens[start:end]
        context_str = ''.join(context_tokens)
        print(f"{rank:<6} {score:<12.2f} {token:<20} ...{context_str[:50]}...")

    # Check if answer tokens have high scores
    print(f"\n{'='*80}")
    print("ANSWER TOKEN ANALYSIS")
    print(f"{'='*80}")

    answer_keywords = ["McLaren", "Vale", "Mc", "Laren"]
    print(f"\nSearching for answer-related tokens: {answer_keywords}")

    for keyword in answer_keywords:
        for idx, token in enumerate(tokens):
            if keyword.lower() in token.lower():
                score = scores_np[idx]
                rank = (aggregated > score).sum().item() + 1
                percentile = (rank / len(tokens)) * 100
                status = "✓ KEEP" if rank <= len(tokens) * 0.1 else "✗ EVICT"
                print(f"  Token '{token}' at position {idx}: score={score:.2f}, rank={rank}/{len(tokens)} ({percentile:.1f}%ile) [{status}]")

    # Check if question tokens have high scores
    print(f"\n{'='*80}")
    print("QUESTION TOKEN ANALYSIS")
    print(f"{'='*80}")

    question_keywords = ["Mawson", "electoral", "district", "wine", "region", "55", "km", "south", "Adelaide"]
    print(f"\nSearching for question tokens...")

    for keyword in question_keywords:
        for idx, token in enumerate(tokens):
            if keyword.lower() in token.lower():
                score = scores_np[idx]
                rank = (aggregated > score).sum().item() + 1
                percentile = (rank / len(tokens)) * 100
                status = "✓ KEEP" if rank <= len(tokens) * 0.1 else "✗ EVICT"
                print(f"  Token '{token}' (keyword: {keyword}): score={score:.2f}, rank={rank}/{len(tokens)} ({percentile:.1f}%ile) [{status}]")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")

    # Count how many answer tokens would be kept at 90% sparsity (keep 10%)
    keep_size = int(len(tokens) * 0.1)
    keep_indices = sorted_indices[:keep_size]

    answer_kept = 0
    answer_total = 0
    for keyword in answer_keywords:
        for idx, token in enumerate(tokens):
            if keyword.lower() in token.lower():
                answer_total += 1
                if idx in keep_indices:
                    answer_kept += 1

    print(f"\nAt 90% sparsity (keep {keep_size}/{len(tokens)} tokens):")
    print(f"  Answer tokens found: {answer_total}")
    print(f"  Answer tokens kept: {answer_kept}/{answer_total}")

    if answer_kept == 0:
        print("\n❌ PROBLEM: No answer tokens are being kept!")
        print("Cumulative attention scores are not identifying answer-relevant information.")
    elif answer_kept < answer_total:
        print(f"\n⚠️  WARNING: Only {answer_kept}/{answer_total} answer tokens kept.")
    else:
        print("\n✓ All answer tokens are being kept.")

    return answer_kept > 0

if __name__ == "__main__":
    try:
        success = diagnose_eviction()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
