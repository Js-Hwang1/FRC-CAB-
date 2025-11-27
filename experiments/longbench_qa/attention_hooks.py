"""
Attention Hooks for Sparse Attention Methods

This module provides hooks to actually apply sparse attention during inference.
It intercepts the attention computation and applies the selected method's masking.

Key approaches:
1. KV Cache Pruning - Remove tokens from the cache (like real H2O)
2. Attention Weight Modification - Mask attention weights during computation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SparseAttentionConfig:
    """Configuration for sparse attention."""
    method: str = "dense"  # dense, h2o, cab_v4
    sparsity: float = 0.9
    block_size: int = 64
    magnitude_ratio: float = 0.5  # For CAB V4
    lambda_redundancy: float = 0.3


class KVCachePruner:
    """
    Prunes KV cache to keep only important tokens.
    
    This is how H2O and similar methods work in practice:
    - Compute importance scores for each token
    - Keep top-k tokens in the cache
    - Continue generation with pruned cache
    """
    
    def __init__(self, config: SparseAttentionConfig):
        self.config = config
        self.importance_scores = {}  # Per-layer importance
    
    def compute_importance_h2o(
        self,
        attention_weights: torch.Tensor,  # [B, H, N, N]
    ) -> torch.Tensor:
        """
        H2O importance: cumulative attention received by each key.
        
        Returns: [B, H, N] importance scores
        """
        # Sum attention each key receives from all queries
        importance = attention_weights.sum(dim=2)  # [B, H, N]
        return importance
    
    def compute_importance_cab_v4(
        self,
        attention_weights: torch.Tensor,  # [B, H, N, N]
        query: torch.Tensor,  # [B, H, N, D]
        key: torch.Tensor,  # [B, H, N, D]
    ) -> torch.Tensor:
        """
        CAB V4 importance: hybrid of magnitude + FRC.
        
        Returns: [B, H, N] importance scores
        """
        B, H, N, D = query.shape
        
        # Magnitude component (like H2O)
        magnitude_importance = attention_weights.sum(dim=2)  # [B, H, N]
        
        # FRC component (simplified for token-level)
        # Compute pairwise similarity
        q_norm = F.normalize(query, dim=-1)
        k_norm = F.normalize(key, dim=-1)
        similarity = torch.matmul(q_norm, k_norm.transpose(-2, -1))  # [B, H, N, N]
        
        # Redundancy: how similar is each token to others?
        # High similarity to many tokens = redundant
        redundancy = similarity.mean(dim=-1)  # [B, H, N]
        
        # Uniqueness: inverse of redundancy
        uniqueness = 1.0 - redundancy
        
        # Combine: magnitude + uniqueness
        ratio = self.config.magnitude_ratio
        importance = ratio * magnitude_importance + (1 - ratio) * uniqueness
        
        return importance
    
    def get_keep_mask(
        self,
        importance: torch.Tensor,  # [B, H, N]
        num_keep: int,
    ) -> torch.Tensor:
        """
        Get mask for tokens to keep.
        
        Returns: [B, H, N] boolean mask (True = keep)
        """
        B, H, N = importance.shape
        
        # Get top-k indices
        _, top_indices = torch.topk(importance, k=min(num_keep, N), dim=-1)
        
        # Create mask
        mask = torch.zeros(B, H, N, dtype=torch.bool, device=importance.device)
        mask.scatter_(2, top_indices, True)
        
        return mask


class AttentionHook:
    """
    Hook that modifies attention weights during forward pass.
    
    Registers as a forward hook on attention layers.
    """
    
    def __init__(self, config: SparseAttentionConfig, layer_idx: int = 0):
        self.config = config
        self.layer_idx = layer_idx
        self.pruner = KVCachePruner(config)
        self._enabled = True
    
    def __call__(
        self,
        module: nn.Module,
        args: tuple,
        kwargs: dict,
        output: Any,
    ) -> Any:
        """
        Forward hook that modifies attention output.
        
        Note: This is called AFTER the attention computation.
        For true sparse attention, we'd need to modify the attention
        computation itself, not just the output.
        """
        if not self._enabled or self.config.method == "dense":
            return output
        
        # For most transformer implementations, we can't easily
        # modify attention weights post-hoc. Instead, we'll use
        # a different approach: KV cache pruning.
        
        return output
    
    def enable(self):
        self._enabled = True
    
    def disable(self):
        self._enabled = False


class SparseAttentionWrapper:
    """
    Wrapper that applies sparse attention to a HuggingFace model.
    
    This modifies the model's generate() method to apply sparse attention.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: SparseAttentionConfig,
    ):
        self.model = model
        self.config = config
        self.hooks = []
        self.pruner = KVCachePruner(config)
        
        # Try to identify attention layers
        self.attention_layers = self._find_attention_layers()
        logger.info(f"Found {len(self.attention_layers)} attention layers")
    
    def _find_attention_layers(self) -> List[nn.Module]:
        """Find attention layers in the model."""
        attention_layers = []
        
        for name, module in self.model.named_modules():
            # Common attention layer names
            if any(x in name.lower() for x in ['attention', 'attn', 'self_attn']):
                if hasattr(module, 'forward'):
                    attention_layers.append((name, module))
        
        return attention_layers
    
    def apply_sparse_attention_to_cache(
        self,
        past_key_values: Tuple,
        attention_mask: torch.Tensor,
        method: str = "h2o",
    ) -> Tuple:
        """
        Apply sparse attention by pruning the KV cache.
        
        This is the practical way to implement sparse attention during generation:
        - After each generation step, prune the KV cache
        - Keep only the most important tokens
        
        Args:
            past_key_values: Tuple of (key, value) for each layer
            attention_mask: Current attention mask
            method: Sparse attention method
        
        Returns:
            Pruned past_key_values
        """
        if method == "dense" or self.config.sparsity == 0:
            return past_key_values
        
        if past_key_values is None:
            return past_key_values
        
        pruned_kvs = []
        num_keep = int((1 - self.config.sparsity) * past_key_values[0][0].shape[2])
        num_keep = max(num_keep, 4)  # Keep at least 4 tokens
        
        for layer_idx, (key, value) in enumerate(past_key_values):
            # key, value: [B, H, N, D]
            B, H, N, D = key.shape
            
            if N <= num_keep:
                pruned_kvs.append((key, value))
                continue
            
            # Compute importance based on method
            # For simplicity, use L2 norm of key vectors as importance
            importance = key.norm(dim=-1)  # [B, H, N]
            
            if method == "cab_v4":
                # Add uniqueness component
                k_norm = F.normalize(key, dim=-1)
                similarity = torch.matmul(k_norm, k_norm.transpose(-2, -1))
                redundancy = similarity.mean(dim=-1)
                uniqueness = 1.0 - redundancy
                
                ratio = self.config.magnitude_ratio
                importance = ratio * importance + (1 - ratio) * uniqueness
            
            # Get top-k indices
            _, top_indices = torch.topk(importance, k=num_keep, dim=-1)  # [B, H, num_keep]
            
            # Sort indices to maintain order
            top_indices, _ = torch.sort(top_indices, dim=-1)
            
            # Gather pruned keys and values
            top_indices_expanded = top_indices.unsqueeze(-1).expand(-1, -1, -1, D)
            pruned_key = torch.gather(key, 2, top_indices_expanded)
            pruned_value = torch.gather(value, 2, top_indices_expanded)
            
            pruned_kvs.append((pruned_key, pruned_value))
        
        return tuple(pruned_kvs)


def create_sparse_generate_fn(
    model: nn.Module,
    tokenizer: Any,
    config: SparseAttentionConfig,
) -> Callable:
    """
    Create a generate function that applies sparse attention.
    
    This wraps the model's generate() to apply KV cache pruning.
    """
    wrapper = SparseAttentionWrapper(model, config)
    
    def sparse_generate(
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        **kwargs,
    ) -> torch.Tensor:
        """Generate with sparse attention via KV cache pruning."""
        
        if config.method == "dense":
            # Just use normal generation
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                **kwargs,
            )
        
        # For sparse methods, we generate token by token with cache pruning
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initial forward pass to get KV cache
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        
        # Prune KV cache
        past_key_values = wrapper.apply_sparse_attention_to_cache(
            past_key_values,
            attention_mask,
            method=config.method,
        )
        
        # Generate tokens one by one
        generated_ids = input_ids
        
        for _ in range(max_new_tokens):
            # Get next token
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check for EOS
            if (next_token == tokenizer.eos_token_id).all():
                break
            
            # Update attention mask
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                ], dim=-1)
            
            # Forward pass with pruned cache
            # Adjust attention mask for pruned cache size
            cache_len = past_key_values[0][0].shape[2] if past_key_values else 0
            adjusted_mask = torch.ones((batch_size, cache_len + 1), device=device)
            
            with torch.no_grad():
                outputs = model(
                    input_ids=next_token,
                    attention_mask=adjusted_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Prune cache periodically (every N tokens)
            if generated_ids.shape[1] % 10 == 0:
                past_key_values = wrapper.apply_sparse_attention_to_cache(
                    past_key_values,
                    attention_mask,
                    method=config.method,
                )
        
        return generated_ids
    
    return sparse_generate


# Simpler approach: Modify attention scores directly
def apply_sparse_attention_mask(
    attention_scores: torch.Tensor,  # [B, H, N, N]
    method: str,
    sparsity: float,
    query: Optional[torch.Tensor] = None,
    key: Optional[torch.Tensor] = None,
    magnitude_ratio: float = 0.5,
) -> torch.Tensor:
    """
    Apply sparse attention mask to attention scores.
    
    Args:
        attention_scores: Raw attention scores (before softmax)
        method: "dense", "h2o", "cab_v4", "random"
        sparsity: Fraction of attention to mask out
        query, key: For CAB V4 FRC computation
        magnitude_ratio: For CAB V4 hybrid
    
    Returns:
        Masked attention scores
    """
    if method == "dense" or sparsity == 0:
        return attention_scores
    
    B, H, N, _ = attention_scores.shape
    
    # Number of positions to keep per query
    num_keep = max(1, int(N * (1 - sparsity)))
    
    if method == "h2o":
        # Keep tokens with highest attention scores
        # Sum over queries to get key importance
        importance = attention_scores.sum(dim=2)  # [B, H, N]
        _, top_indices = torch.topk(importance, k=num_keep, dim=-1)
        
        # Create mask
        mask = torch.full_like(attention_scores, float('-inf'))
        for b in range(B):
            for h in range(H):
                mask[b, h, :, top_indices[b, h]] = 0
        
        return attention_scores + mask
    
    elif method == "cab_v4":
        # Hybrid: magnitude + uniqueness
        magnitude_importance = attention_scores.sum(dim=2)  # [B, H, N]
        
        # Compute uniqueness from attention patterns
        attn_softmax = F.softmax(attention_scores, dim=-1)
        # Tokens that get diverse attention are unique
        entropy = -(attn_softmax * (attn_softmax + 1e-8).log()).sum(dim=2)
        uniqueness = entropy / entropy.max()  # Normalize
        
        # Combine
        importance = magnitude_ratio * magnitude_importance + (1 - magnitude_ratio) * uniqueness
        _, top_indices = torch.topk(importance, k=num_keep, dim=-1)
        
        # Create mask
        mask = torch.full_like(attention_scores, float('-inf'))
        for b in range(B):
            for h in range(H):
                mask[b, h, :, top_indices[b, h]] = 0
        
        return attention_scores + mask
    
    elif method == "random":
        # Random selection
        mask = torch.full_like(attention_scores, float('-inf'))
        for b in range(B):
            for h in range(H):
                perm = torch.randperm(N, device=attention_scores.device)[:num_keep]
                mask[b, h, :, perm] = 0
        
        return attention_scores + mask
    
    else:
        return attention_scores

