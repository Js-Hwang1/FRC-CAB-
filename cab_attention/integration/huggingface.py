"""
HuggingFace Transformers Integration
=====================================

Provides hooks and utilities to integrate CABCache with HuggingFace models.

Supported models:
- LLaMA / LLaMA-2 / LLaMA-3
- Mistral
- Qwen / Qwen2
- Other models with standard attention interface
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Callable
from contextlib import contextmanager
import warnings

from ..cache.cab_cache import CABCache
from ..cache.h2o_cache import H2OCache


# Registry of attention hooks
_ATTENTION_HOOKS: Dict[int, List[torch.utils.hooks.RemovableHandle]] = {}


def _get_attention_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Find all attention layers in a HuggingFace model.
    
    Supports various model architectures by looking for common patterns.
    """
    attention_layers = []
    
    for name, module in model.named_modules():
        # Check for common attention layer names
        if any(pattern in name.lower() for pattern in ['self_attn', 'attention', 'attn']):
            # Verify it's an actual attention layer (has q_proj or similar)
            if hasattr(module, 'q_proj') or hasattr(module, 'query') or hasattr(module, 'qkv_proj'):
                attention_layers.append((name, module))
    
    return attention_layers


def _create_attention_hook(
    cache: CABCache,
    layer_idx: int,
) -> Callable:
    """
    Create a forward hook that captures attention weights and updates cache.
    
    This hook intercepts the attention layer's output and extracts attention weights
    for importance tracking.
    """
    def hook(module, args, output):
        # HuggingFace attention layers return (hidden_states, attention_weights, past_key_value)
        # or just (hidden_states,) if output_attentions=False
        
        if isinstance(output, tuple) and len(output) >= 2:
            attention_weights = output[1]  # [B, H, N_q, N_kv]
            
            if attention_weights is not None and layer_idx == 0:
                # Only track for first layer to save compute
                cache.importance_tracker.update(attention_weights.detach())
        
        return output
    
    return hook


def apply_cab_to_model(
    model: nn.Module,
    cache: CABCache,
    output_attentions: bool = True,
) -> List[torch.utils.hooks.RemovableHandle]:
    """
    Apply CAB cache to a HuggingFace model.
    
    This registers forward hooks on attention layers to capture attention weights
    for importance tracking.
    
    Args:
        model: HuggingFace model
        cache: CABCache instance
        output_attentions: Whether to enable attention output (needed for tracking)
        
    Returns:
        List of hook handles (for removal later)
    """
    model_id = id(model)
    
    # Remove existing hooks if any
    if model_id in _ATTENTION_HOOKS:
        remove_cab_from_model(model)
    
    # Enable attention output if needed
    if output_attentions and hasattr(model.config, 'output_attentions'):
        model.config.output_attentions = True
    
    # Find attention layers
    attention_layers = _get_attention_layers(model)
    
    if not attention_layers:
        warnings.warn(
            "No attention layers found in model. "
            "CAB will work but without attention-based importance tracking."
        )
        return []
    
    # Register hooks
    handles = []
    for layer_idx, (name, module) in enumerate(attention_layers):
        hook = _create_attention_hook(cache, layer_idx)
        handle = module.register_forward_hook(hook)
        handles.append(handle)
    
    _ATTENTION_HOOKS[model_id] = handles
    
    return handles


def remove_cab_from_model(model: nn.Module):
    """
    Remove CAB hooks from a HuggingFace model.
    
    Args:
        model: HuggingFace model with CAB hooks
    """
    model_id = id(model)
    
    if model_id in _ATTENTION_HOOKS:
        for handle in _ATTENTION_HOOKS[model_id]:
            handle.remove()
        del _ATTENTION_HOOKS[model_id]


@contextmanager
def cab_context(model: nn.Module, cache: CABCache):
    """
    Context manager for using CAB with a model.
    
    Usage:
        cache = CABCache(max_cache_size=4096, sparsity=0.9)
        with cab_context(model, cache):
            outputs = model.generate(input_ids, max_new_tokens=512)
    """
    handles = apply_cab_to_model(model, cache)
    try:
        yield cache
    finally:
        for handle in handles:
            handle.remove()


def generate_with_cab(
    model: nn.Module,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    cache: Optional[CABCache] = None,
    sparsity: float = 0.9,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    do_sample: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Generate text using CAB cache for efficient KV management.
    
    This is the main entry point for generation with CAB.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        input_ids: [B, N] input token ids
        max_new_tokens: Maximum tokens to generate
        cache: Optional CABCache (created if not provided)
        sparsity: Target sparsity if creating cache
        temperature: Sampling temperature
        top_k: Top-k sampling
        top_p: Top-p (nucleus) sampling
        do_sample: Whether to sample (False = greedy)
        **kwargs: Additional arguments for CABCache
        
    Returns:
        generated_ids: [B, N + max_new_tokens] generated token ids
        stats: Dictionary with generation statistics
    """
    device = input_ids.device
    B, prompt_len = input_ids.shape
    
    # Create cache if not provided
    if cache is None:
        max_cache_size = kwargs.pop('max_cache_size', prompt_len + max_new_tokens)
        cache = CABCache(
            max_cache_size=max_cache_size,
            sparsity=sparsity,
            device=str(device),
            **kwargs,
        )
    
    # Initial forward pass to populate cache
    model.eval()
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        # Process prompt
        for i in range(prompt_len):
            token = input_ids[:, i:i+1]
            
            outputs = model(
                input_ids=token,
                use_cache=False,  # We manage our own cache
                output_attentions=True,
                return_dict=True,
            )
            
            # Extract key/value from the model's internal computation
            # This requires accessing the model's hidden states
            # For now, we'll use a simpler approach with past_key_values
        
        # Actually, let's use a more robust approach: token-by-token with cache updates
        # Reset and use proper cache management
        cache.reset()
        
        # Forward pass with our cache
        past_key_values = None
        
        for i in range(prompt_len):
            token = input_ids[:, i:i+1]
            
            outputs = model(
                input_ids=token,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            
            past_key_values = outputs.past_key_values
            
            # Update our cache with attention weights
            if outputs.attentions is not None:
                # Average attention across layers for importance tracking
                attn_avg = torch.stack(outputs.attentions, dim=0).mean(dim=0)
                cache.importance_tracker.update(attn_avg.detach())
        
        # Convert HF cache to our format
        if past_key_values is not None:
            for layer_idx, (k, v) in enumerate(past_key_values):
                while len(cache.key_cache) <= layer_idx:
                    cache.key_cache.append(None)
                    cache.value_cache.append(None)
                cache.key_cache[layer_idx] = k
                cache.value_cache[layer_idx] = v
        
        # Compute initial FRC
        if len(cache.key_cache) > 0 and cache.key_cache[0] is not None:
            cache.frc_tracker.compute_from_keys(cache.key_cache[0], force_update=True)
        
        # Generation loop
        next_token_logits = outputs.logits[:, -1, :]
        
        for step in range(max_new_tokens):
            # Sample or greedy
            if do_sample:
                # Apply temperature
                logits = next_token_logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float('-inf')
                
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check EOS
            if tokenizer.eos_token_id is not None:
                if (next_token == tokenizer.eos_token_id).all():
                    break
            
            # Forward pass for next token
            # Convert our cache back to HF format
            hf_past = tuple(
                (cache.key_cache[i], cache.value_cache[i])
                for i in range(len(cache.key_cache))
                if cache.key_cache[i] is not None
            )
            
            outputs = model(
                input_ids=next_token,
                past_key_values=hf_past if len(hf_past) > 0 else None,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
            
            # Update cache
            new_past = outputs.past_key_values
            if new_past is not None:
                for layer_idx, (k, v) in enumerate(new_past):
                    # Update our cache (this triggers eviction if needed)
                    new_k = k[:, :, -1:, :]  # Just the new token
                    new_v = v[:, :, -1:, :]
                    
                    # Get attention for importance
                    attn = None
                    if outputs.attentions is not None and layer_idx < len(outputs.attentions):
                        attn = outputs.attentions[layer_idx]
                    
                    cache.update(new_k, new_v, layer_idx, attn)
            
            next_token_logits = outputs.logits[:, -1, :]
    
    # Collect stats
    stats = cache.get_stats()
    stats['generated_tokens'] = generated_ids.shape[1] - prompt_len
    stats['prompt_length'] = prompt_len
    
    return generated_ids, stats


class CABGenerationMixin:
    """
    Mixin class that adds CAB generation capabilities to HuggingFace models.
    
    Usage:
        from transformers import AutoModelForCausalLM
        from cab_attention.integration import CABGenerationMixin
        
        class MyModel(CABGenerationMixin, AutoModelForCausalLM):
            pass
        
        model = MyModel.from_pretrained("...")
        outputs = model.generate_with_cab(input_ids, sparsity=0.9)
    """
    
    def generate_with_cab(
        self,
        input_ids: torch.Tensor,
        tokenizer=None,
        max_new_tokens: int = 128,
        sparsity: float = 0.9,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Generate using CAB cache."""
        if tokenizer is None:
            tokenizer = getattr(self, 'tokenizer', None)
            if tokenizer is None:
                raise ValueError("Tokenizer required for generation")
        
        return generate_with_cab(
            model=self,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            sparsity=sparsity,
            **kwargs,
        )


if __name__ == "__main__":
    print("Testing HuggingFace integration...")
    
    # This requires transformers to be installed
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        
        # Load a small model for testing
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"\nLoading {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
            device_map='auto' if device == 'cuda' else None,
        )
        
        if device == 'cpu':
            model = model.to(device)
        
        # Test generation with CAB
        prompt = "The quick brown fox"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        
        print(f"\nPrompt: {prompt}")
        print(f"Input shape: {input_ids.shape}")
        
        # Create CAB cache
        cache = CABCache(
            max_cache_size=256,
            sparsity=0.9,
            local_ratio=0.3,
            bridge_ratio=0.2,
            importance_ratio=0.5,
            device=device,
        )
        
        # Generate
        print("\nGenerating with CAB...")
        generated_ids, stats = generate_with_cab(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=50,
            cache=cache,
        )
        
        # Decode
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"\nGenerated: {generated_text}")
        
        print(f"\nStats:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n✓ Integration test passed!")
        
    except ImportError as e:
        print(f"Skipping test (missing dependency): {e}")

