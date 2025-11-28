"""
HuggingFace Integration for CAB Attention
==========================================

Provides seamless integration with HuggingFace transformers models.

Usage:
    from cab_attention import CABCache
    from cab_attention.integration import apply_cab_to_model, remove_cab_from_model

    # Apply CAB cache to model
    cache = CABCache(max_cache_size=4096, sparsity=0.9)
    apply_cab_to_model(model, cache)

    # Generate with CAB
    outputs = model.generate(input_ids, max_new_tokens=512)

    # Remove hooks when done
    remove_cab_from_model(model)
"""

from .huggingface import (
    apply_cab_to_model,
    remove_cab_from_model,
    CABGenerationMixin,
    generate_with_cab,
)

__all__ = [
    "apply_cab_to_model",
    "remove_cab_from_model", 
    "CABGenerationMixin",
    "generate_with_cab",
]

