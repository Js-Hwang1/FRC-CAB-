"""
CAB: Curvature-Aware Block-Sparse Attention
=============================================

Efficient KV cache eviction using three components:
- Local context (recent tokens)
- Bridge tokens (low Forman-Ricci curvature connectors)
- Important tokens (high cumulative attention, H2O-style)

Usage:
    from cab_attention import CABCache
    from cab_attention.integration import generate_with_cab

    # Simple usage
    cache = CABCache(max_cache_size=4096, sparsity=0.9)
    generated_ids, stats = generate_with_cab(
        model=model,
        tokenizer=tokenizer,
        input_ids=input_ids,
        cache=cache,
        max_new_tokens=512,
    )

    # Or with custom ratios
    cache = CABCache(
        max_cache_size=4096,
        sparsity=0.9,
        local_ratio=0.3,      # 30% recent tokens
        bridge_ratio=0.2,      # 20% bridge tokens (low FRC)
        importance_ratio=0.5,  # 50% important tokens (H2O-style)
    )
"""

__version__ = "5.0.0"

from .cache.cab_cache import CABCache
from .cache.h2o_cache import H2OCache

# Lazy import for integration (requires transformers)
def __getattr__(name):
    if name == "generate_with_cab":
        from .integration.huggingface import generate_with_cab
        return generate_with_cab
    elif name == "apply_cab_to_model":
        from .integration.huggingface import apply_cab_to_model
        return apply_cab_to_model
    elif name == "remove_cab_from_model":
        from .integration.huggingface import remove_cab_from_model
        return remove_cab_from_model
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "CABCache",
    "H2OCache",
    "generate_with_cab",
    "apply_cab_to_model",
    "remove_cab_from_model",
]
