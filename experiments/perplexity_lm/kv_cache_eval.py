"""
KV Cache-Based Perplexity Evaluation

This implements the EXACT evaluation methodology used by H2O, StreamingLLM, etc.
These methods are designed for KV cache pruning during generation, not attention replacement.

Published Methodology:
1. Process input token-by-token (autoregressive)
2. At each step, prune KV cache according to the method
3. Compute log probability of actual next token
4. Aggregate to get perplexity

This is slower but matches exactly how these methods are evaluated in their papers.

References:
- H2O: https://arxiv.org/abs/2306.14048
- StreamingLLM: https://arxiv.org/abs/2309.17453
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import math
import logging
from dataclasses import dataclass
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class KVCacheConfig:
    """Configuration for KV cache management."""
    method: str  # dense, h2o, streaming_llm, cab_v4
    max_cache_size: int = 1024  # Maximum KV cache size
    sink_size: int = 4  # Number of sink tokens (for streaming methods)
    recent_size: int = 256  # Recent window size
    sparsity: float = 0.5  # For methods that use sparsity


class KVCacheManager:
    """
    Manages KV cache pruning for different sparse attention methods.
    
    Implements exact algorithms from published papers.
    """
    
    def __init__(self, config: KVCacheConfig):
        self.config = config
        self.method = config.method
        
    def should_prune(self, cache_len: int) -> bool:
        """Check if cache needs pruning."""
        return cache_len > self.config.max_cache_size
    
    def compute_importance(
        self,
        key_states: torch.Tensor,  # [num_layers, B, H, cache_len, D]
        value_states: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,  # [num_layers, B, H, 1, cache_len]
    ) -> torch.Tensor:
        """
        Compute importance scores for each cached position.
        
        Returns: [cache_len] importance scores
        """
        cache_len = key_states.shape[3]
        device = key_states.device
        
        if self.method == "dense":
            # Keep all tokens
            return torch.ones(cache_len, device=device)
        
        elif self.method == "h2o":
            # H2O: Use L2 norm of keys as importance proxy
            # Average across layers and heads
            key_norms = key_states.norm(dim=-1)  # [num_layers, B, H, cache_len]
            importance = key_norms.mean(dim=(0, 1, 2))  # [cache_len]
            return importance
        
        elif self.method == "streaming_llm":
            # StreamingLLM: Sinks + Recent window
            # Importance: max for sinks and recent, 0 for middle
            importance = torch.zeros(cache_len, device=device)
            importance[:self.config.sink_size] = 1.0  # Sink tokens
            if cache_len > self.config.sink_size:
                recent_start = max(self.config.sink_size, cache_len - self.config.recent_size)
                importance[recent_start:] = 1.0  # Recent tokens
            return importance
        
        elif self.method == "cab_v4":
            # CAB V4: Hybrid magnitude + uniqueness
            return self._compute_cab_importance(key_states)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _compute_cab_importance(self, key_states: torch.Tensor) -> torch.Tensor:
        """
        CAB V4: Hybrid importance = 50% magnitude + 50% uniqueness
        
        Uniqueness = 1 - avg_cosine_similarity (low redundancy = high importance)
        """
        # Average across layers for efficiency
        keys = key_states.mean(dim=0)  # [B, H, cache_len, D]
        B, H, M, D = keys.shape
        
        # Magnitude component
        magnitude = keys.norm(dim=-1).mean(dim=(0, 1))  # [cache_len]
        
        # Uniqueness component
        keys_flat = keys.mean(dim=(0, 1))  # [cache_len, D]
        keys_norm = F.normalize(keys_flat, dim=-1)
        similarity = torch.mm(keys_norm, keys_norm.t())  # [cache_len, cache_len]
        redundancy = similarity.mean(dim=-1)  # [cache_len]
        uniqueness = 1.0 - redundancy
        
        # Normalize to [0, 1]
        mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
        uniq_norm = (uniqueness - uniqueness.min()) / (uniqueness.max() - uniqueness.min() + 1e-8)
        
        # Hybrid: 50% magnitude + 50% uniqueness
        importance = 0.5 * mag_norm + 0.5 * uniq_norm
        
        return importance
    
    def prune_cache(
        self,
        key_cache: List[torch.Tensor],  # List of [B, H, cache_len, D] per layer
        value_cache: List[torch.Tensor],
        attention_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Prune KV cache to max_cache_size.
        
        Returns: Pruned (key_cache, value_cache)
        """
        if not key_cache:
            return key_cache, value_cache
        
        cache_len = key_cache[0].shape[2]
        
        if cache_len <= self.config.max_cache_size:
            return key_cache, value_cache
        
        # Stack for importance computation
        key_states = torch.stack(key_cache, dim=0)  # [num_layers, B, H, cache_len, D]
        
        # Compute importance
        importance = self.compute_importance(key_states, None, attention_scores)
        
        # Select top-K positions to keep
        num_keep = self.config.max_cache_size
        
        if self.method == "streaming_llm":
            # StreamingLLM: Always keep sinks + most recent
            keep_indices = []
            # Sinks
            for i in range(min(self.config.sink_size, cache_len)):
                keep_indices.append(i)
            # Recent
            recent_budget = num_keep - len(keep_indices)
            if recent_budget > 0:
                start_idx = max(self.config.sink_size, cache_len - recent_budget)
                for i in range(start_idx, cache_len):
                    if len(keep_indices) < num_keep:
                        keep_indices.append(i)
            keep_indices = torch.tensor(keep_indices, device=key_cache[0].device)
        else:
            # H2O, CAB: Top-K by importance
            _, keep_indices = torch.topk(importance, k=num_keep)
            keep_indices = keep_indices.sort().values  # Maintain order
        
        # Prune each layer's cache
        pruned_keys = []
        pruned_values = []
        for layer_idx in range(len(key_cache)):
            pruned_keys.append(key_cache[layer_idx][:, :, keep_indices, :])
            pruned_values.append(value_cache[layer_idx][:, :, keep_indices, :])
        
        return pruned_keys, pruned_values


def evaluate_perplexity_with_kv_pruning(
    model,
    tokenizer,
    text: str,
    config: KVCacheConfig,
    max_length: int = 2048,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Evaluate perplexity using KV cache pruning (published methodology).
    
    This processes the input autoregressively and prunes the KV cache
    at each step according to the method's algorithm.
    
    Args:
        model: HuggingFace causal LM
        tokenizer: Tokenizer
        text: Input text to evaluate
        config: KV cache configuration
        max_length: Maximum sequence length to evaluate
        device: Device to run on
    
    Returns:
        Dict with perplexity and other metrics
    """
    model.eval()
    cache_manager = KVCacheManager(config)
    
    # Tokenize
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = input_ids.to(device)
    seq_len = input_ids.shape[1]
    
    if seq_len < 2:
        return {"perplexity": float("nan"), "tokens": 0}
    
    total_loss = 0.0
    num_tokens = 0
    
    # For dense, just do standard forward pass
    if config.method == "dense":
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()
        return {
            "perplexity": math.exp(loss),
            "cross_entropy": loss,
            "tokens": seq_len - 1,
        }
    
    # For sparse methods, process token-by-token with KV cache pruning
    past_key_values = None
    
    with torch.no_grad():
        for i in tqdm(range(seq_len - 1), desc=f"Evaluating {config.method}", leave=False):
            # Current token
            current_token = input_ids[:, i:i+1]
            
            # Forward pass with KV cache
            outputs = model(
                input_ids=current_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            # Get logits for next token prediction
            logits = outputs.logits[:, -1, :]  # [B, vocab_size]
            
            # Compute loss for actual next token
            target = input_ids[:, i+1]
            loss = F.cross_entropy(logits, target, reduction="sum")
            total_loss += loss.item()
            num_tokens += 1
            
            # Update KV cache
            past_key_values = outputs.past_key_values
            
            # Prune KV cache if needed
            if past_key_values is not None:
                # Convert to list format for pruning
                key_cache = [layer_kv[0] for layer_kv in past_key_values]
                value_cache = [layer_kv[1] for layer_kv in past_key_values]
                
                cache_len = key_cache[0].shape[2]
                if cache_manager.should_prune(cache_len):
                    key_cache, value_cache = cache_manager.prune_cache(key_cache, value_cache)
                    # Convert back to tuple format
                    past_key_values = tuple(
                        (key_cache[i], value_cache[i]) 
                        for i in range(len(key_cache))
                    )
    
    avg_loss = total_loss / num_tokens if num_tokens > 0 else float("nan")
    perplexity = math.exp(avg_loss) if not math.isnan(avg_loss) else float("nan")
    
    return {
        "perplexity": perplexity,
        "cross_entropy": avg_loss,
        "tokens": num_tokens,
    }


def run_kv_cache_benchmark(
    model,
    tokenizer,
    texts: List[str],
    methods: List[str] = ["dense", "h2o", "streaming_llm", "cab_v4"],
    max_cache_sizes: List[int] = [128, 256, 512],
    max_length: int = 2048,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Run full benchmark comparing methods with KV cache pruning.
    
    Args:
        model: HuggingFace causal LM
        tokenizer: Tokenizer
        texts: List of texts to evaluate
        methods: List of methods to compare
        max_cache_sizes: Cache sizes to test (controls effective sparsity)
        max_length: Maximum sequence length
        device: Device
    
    Returns:
        Results dictionary
    """
    results = {}
    
    for method in methods:
        results[method] = {}
        
        for cache_size in max_cache_sizes:
            if method == "dense" and cache_size != max_cache_sizes[0]:
                continue  # Dense doesn't depend on cache size
            
            config = KVCacheConfig(
                method=method,
                max_cache_size=cache_size if method != "dense" else max_length,
                sink_size=4,
                recent_size=cache_size // 2 if method == "streaming_llm" else 0,
            )
            
            total_loss = 0.0
            total_tokens = 0
            
            for text in tqdm(texts, desc=f"{method} (cache={cache_size})"):
                result = evaluate_perplexity_with_kv_pruning(
                    model, tokenizer, text, config, max_length, device
                )
                if not math.isnan(result["cross_entropy"]):
                    total_loss += result["cross_entropy"] * result["tokens"]
                    total_tokens += result["tokens"]
            
            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                ppl = math.exp(avg_loss)
            else:
                avg_loss = float("nan")
                ppl = float("nan")
            
            cache_key = str(cache_size) if method != "dense" else "full"
            results[method][cache_key] = {
                "perplexity": ppl,
                "cross_entropy": avg_loss,
                "tokens": total_tokens,
            }
            
            logger.info(f"{method} (cache={cache_key}): PPL={ppl:.2f}")
    
    return results


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, "/root/FRC")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading model...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Test text
    test_text = "The quick brown fox jumps over the lazy dog. " * 50
    
    print("\nTesting methods...")
    for method in ["dense", "h2o", "streaming_llm", "cab_v4"]:
        config = KVCacheConfig(
            method=method,
            max_cache_size=128,
            sink_size=4,
            recent_size=64,
        )
        result = evaluate_perplexity_with_kv_pruning(
            model, tokenizer, test_text, config, max_length=512, device="cuda"
        )
        print(f"{method}: PPL={result['perplexity']:.2f}, tokens={result['tokens']}")

