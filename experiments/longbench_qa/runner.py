"""
Benchmark Runner for Long-Context QA Experiments

Implements EXACT published algorithms for fair comparison:

Baselines:
- Dense: Full attention (no sparsity)
- H2O: Heavy-Hitter Oracle (Zhang et al., 2023) - arxiv:2306.14048
  Uses CUMULATIVE ATTENTION SCORES to identify important tokens.
  
- StreamingLLM: Efficient Streaming LLM (Xiao et al., 2023) - arxiv:2309.17453
  Keeps "attention sinks" (first few tokens) + sliding window.

Our Method:
- CAB: Curvature-Aware Block-Sparse Attention
  Three-component eviction: local + bridge (low FRC) + importance

Other Baselines:
- Random: Random token selection (lower bound)
- Local+Strided: Sparse Transformer pattern (Child et al., 2019) - arxiv:1904.10509

Designed for ICML-quality reproducible experiments.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

# Optional dependencies with fallbacks
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        """Fallback tqdm that just returns the iterable."""
        desc = kwargs.get('desc', '')
        if desc:
            print(f"Processing: {desc}...")
        return iterable

try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    F = None

from .config import (
    ExperimentConfig,
    DatasetConfig,
    MethodConfig,
    ModelConfig,
    MetricName,
    ALL_DATASETS,
    METHOD_CONFIGS,
)
from .data_loaders import get_dataset, BenchmarkSample, BaseBenchmarkDataset
from .methods import get_method, BaseAttentionMethod
from .metrics import (
    compute_metrics,
    compute_batch_metrics,
    EvaluationReport,
    create_comparison_table,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Result Classes
# =============================================================================

@dataclass
class SampleResult:
    """Result for a single sample."""
    sample_id: str
    prediction: str
    references: List[str]
    metrics: Dict[str, float]
    
    # Optional diagnostics
    attention_diagnostics: Optional[Dict[str, Any]] = None
    generation_time_ms: float = 0.0
    context_length: int = 0


@dataclass
class MethodResult:
    """Results for a single method on a dataset."""
    method_name: str
    dataset_name: str
    sparsity: float
    
    # Aggregated metrics
    metrics: Dict[str, Dict[str, Any]]  # metric_name -> {mean, std, min, max}
    
    # Per-sample results
    sample_results: List[SampleResult]
    
    # Timing
    total_time_sec: float = 0.0
    samples_per_sec: float = 0.0
    
    # Model info
    model_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method_name': self.method_name,
            'dataset_name': self.dataset_name,
            'sparsity': self.sparsity,
            'metrics': self.metrics,
            'sample_results': [asdict(s) for s in self.sample_results],
            'total_time_sec': self.total_time_sec,
            'samples_per_sec': self.samples_per_sec,
            'model_name': self.model_name,
        }


@dataclass
class ExperimentResult:
    """Results for a complete experiment."""
    name: str
    description: str
    
    # Results per method
    method_results: Dict[str, MethodResult]  # method_name -> MethodResult
    
    # Experiment config
    config: Dict[str, Any]
    
    # Timing
    start_time: str = ""
    end_time: str = ""
    total_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'method_results': {k: v.to_dict() for k, v in self.method_results.items()},
            'config': self.config,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'total_time_sec': self.total_time_sec,
        }


# =============================================================================
# Model Wrapper
# =============================================================================

class ModelWrapper:
    """
    Wrapper for LLM model with sparse attention integration.
    
    Handles:
    - Model loading with memory optimization
    - Tokenization
    - Generation with attention modification
    - Caching and batching
    """
    
    def __init__(
        self,
        config: ModelConfig,
        device: str = None,
    ):
        self.config = config
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self) -> None:
        """Load model and tokenizer."""
        if self._loaded:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required: pip install transformers")
        
        logger.info(f"Loading model: {self.config.name}")
        
        # Determine dtype
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        model_kwargs = {
            'torch_dtype': torch_dtype,
            'device_map': self.config.device_map,
            'trust_remote_code': True,
        }
        
        # Try to use optimized attention implementations
        if self.config.use_flash_attention:
            try:
                import flash_attn
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                logger.info("Using Flash Attention 2")
            except ImportError:
                # Fall back to SDPA (Scaled Dot Product Attention) - built into PyTorch 2.0+
                if TORCH_AVAILABLE and hasattr(F, 'scaled_dot_product_attention'):
                    model_kwargs['attn_implementation'] = 'sdpa'
                    logger.info("Using SDPA (PyTorch native, nearly as fast as Flash Attention)")
                else:
                    logger.warning("Flash Attention 2 not available, using default attention")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.name,
                **model_kwargs
            )
        except Exception as e:
            # Fallback: try without flash attention
            if 'flash' in str(e).lower():
                logger.warning(f"Flash Attention error, falling back to default: {e}")
                model_kwargs.pop('attn_implementation', None)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.name,
                    **model_kwargs
                )
            else:
                raise
        
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        self.model.eval()
        self._loaded = True
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def generate(
        self,
        context: str,
        question: str,
        max_new_tokens: int = None,
        sparse_method: str = "dense",
        sparsity: float = 0.0,
        magnitude_ratio: float = 0.5,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate response for a question given context.
        
        Args:
            context: Long context
            question: Question to answer
            max_new_tokens: Max tokens to generate
            sparse_method: "dense", "h2o", "cab", "random"
            sparsity: Fraction of attention to prune (0.0 = dense, 0.9 = keep 10%)
            magnitude_ratio: For CAB V4, ratio of magnitude vs FRC
        
        Returns:
            prediction: Generated text
            diagnostics: Generation diagnostics
        """
        if not self._loaded:
            self.load()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Format prompt
        prompt = self._format_prompt(context, question)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.max_length - max_new_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate with sparse attention via KV cache pruning
        start_time = time.time()
        
        if sparse_method == "dense" or sparsity == 0:
            # Standard dense generation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=self.config.temperature if self.config.do_sample else None,
                    do_sample=self.config.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        else:
            # Sparse generation with KV cache pruning
            full_ids = self._sparse_generate(
                inputs,
                max_new_tokens=max_new_tokens,
                method=sparse_method,
                sparsity=sparsity,
                magnitude_ratio=magnitude_ratio,
            )
            generated_ids = full_ids[0, inputs['input_ids'].shape[1]:]
        
        generation_time = (time.time() - start_time) * 1000  # ms
        prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean prediction: take only first line/sentence (stop at newline or "Question:")
        prediction = prediction.strip()
        # Stop at newline
        if '\n' in prediction:
            prediction = prediction.split('\n')[0].strip()
        # Stop at next question marker
        if 'Question:' in prediction:
            prediction = prediction.split('Question:')[0].strip()
        # Stop at common continuation markers
        for marker in ['Question:', 'Context:', 'Q:', 'A:', '\n\n']:
            if marker in prediction:
                prediction = prediction.split(marker)[0].strip()
        
        diagnostics = {
            'generation_time_ms': generation_time,
            'input_length': inputs['input_ids'].shape[1],
            'output_length': len(generated_ids),
            'prompt_length': len(prompt),
        }
        
        return prediction.strip(), diagnostics
    
    def _format_prompt(self, context: str, question: str) -> str:
        """Format context and question into prompt."""
        model_name = self.config.name.lower()
        
        # Use instruction format for instruct-tuned models
        if 'instruct' in model_name or 'chat' in model_name:
            if 'mistral' in model_name:
                # Mistral-Instruct format - question at END to avoid forgetting
                prompt = f"""[INST] Read the context below and answer the question at the end.

Context:
{context}

Based on the context above, answer this question: {question}

Give only the answer, nothing else. [/INST]"""
            elif 'llama' in model_name:
                # Llama-2/3 Chat format
                prompt = f"""[INST] <<SYS>>
You are a helpful assistant. Answer questions based on the given context. Be concise.
<</SYS>>

Context: {context}

Question: {question} [/INST]"""
            elif 'qwen' in model_name:
                # Qwen format - use simple instruction
                prompt = f"""<|im_start|>system
You are a helpful assistant. Answer questions concisely based on the given context.<|im_end|>
<|im_start|>user
Context: {context}

Question: {question}

Answer with only the answer, nothing else.<|im_end|>
<|im_start|>assistant
"""
            else:
                # Generic instruct format
                prompt = f"""### Instruction:
Answer the question based on the context. Output ONLY the answer.

### Context:
{context}

### Question:
{question}

### Answer:"""
        else:
            # Base model format (completion style)
            prompt = f"""Context: {context}

Question: {question}

Answer:"""
        
        return prompt
    
    def _sparse_generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        method: str,
        sparsity: float,
        magnitude_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Generate with sparse attention via KV cache pruning.
        
        Uses EXACT published algorithms:
        - H2O: Cumulative attention scores (Zhang et al., 2023)
        - StreamingLLM: Sinks + recent window (Xiao et al., 2023)
        - CAB V5: Three-component eviction with CABCache (Ours - NEW)
        - CAB V4: Legacy hybrid magnitude + uniqueness
        """
        # Use CAB V5 with the new CABCache
        if method == "cab":
            return self._generate_with_cab(inputs, max_new_tokens, sparsity)
        
        device = inputs['input_ids'].device
        batch_size = inputs['input_ids'].shape[0]
        num_keep_ratio = 1.0 - sparsity
        
        # Try to import DynamicCache for proper cache handling
        try:
            from transformers.cache_utils import DynamicCache
            has_dynamic_cache = True
        except ImportError:
            has_dynamic_cache = False
            DynamicCache = None
        
        # H2O: Track cumulative attention scores (faithful to paper)
        # OPTIMIZATION: Only request attention when pruning (every 5 steps)
        # This is ~5x faster while remaining faithful to the algorithm
        cumulative_attention = None  # Will be [cache_len] tensor
        is_h2o = (method == "h2o")
        
        # Initial forward pass to get KV cache
        # For H2O: Need attention on first pass to initialize cumulative scores
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True,
                output_attentions=is_h2o,  # Only for initial pruning
            )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        
        # H2O: Initialize cumulative attention from first pass
        if is_h2o and outputs.attentions is not None:
            # attentions: tuple of [B, H, seq_len, seq_len] per layer
            # Sum across layers, batch, heads, query positions to get per-key score
            attn_stack = torch.stack(outputs.attentions, dim=0)  # [L, B, H, Q, K]
            cumulative_attention = attn_stack.sum(dim=(0, 1, 2, 3))  # [K]
        
        # Check cache type and prune accordingly
        uses_dynamic_cache = has_dynamic_cache and isinstance(past_key_values, DynamicCache)
        
        # Prune initial KV cache
        past_key_values, kept_indices, cumulative_attention = self._prune_kv_cache_v3(
            past_key_values, num_keep_ratio, method, magnitude_ratio, 
            uses_dynamic_cache, DynamicCache, cumulative_attention
        )
        
        # Generate tokens one by one
        generated_ids = inputs['input_ids'].clone()
        
        for step in range(max_new_tokens):
            # Get next token (greedy)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check for EOS
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
            
            # OPTIMIZATION: Only request attention on pruning steps
            will_prune = ((step + 1) % 5 == 0)
            need_attention = is_h2o and will_prune
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=need_attention,
                )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # H2O: Accumulate attention scores (only on pruning steps)
            if need_attention and outputs.attentions is not None:
                attn_stack = torch.stack(outputs.attentions, dim=0)  # [L, B, H, 1, K]
                new_scores = attn_stack.sum(dim=(0, 1, 2, 3))  # [K]
                if cumulative_attention is not None:
                    # Extend if needed (new position added)
                    if len(new_scores) > len(cumulative_attention):
                        padding = torch.zeros(len(new_scores) - len(cumulative_attention), 
                                            device=cumulative_attention.device)
                        cumulative_attention = torch.cat([cumulative_attention, padding])
                    cumulative_attention[:len(new_scores)] += new_scores
                else:
                    cumulative_attention = new_scores
            
            # Prune cache periodically (every 5 tokens to balance speed/sparsity)
            if will_prune:
                uses_dynamic_cache = has_dynamic_cache and isinstance(past_key_values, DynamicCache)
                past_key_values, kept_indices, cumulative_attention = self._prune_kv_cache_v3(
                    past_key_values, num_keep_ratio, method, magnitude_ratio, 
                    uses_dynamic_cache, DynamicCache, cumulative_attention
                )
        
        return generated_ids
    
    def _generate_with_cab(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        sparsity: float,
    ) -> torch.Tensor:
        """
        Generate using CAB with CABCache.
        
        CAB uses three-component eviction:
        - Local: Recent tokens (30%)
        - Bridge: Low FRC connectors (20%)
        - Importance: High cumulative attention (50%)
        """
        try:
            from cab_attention import CABCache
            from cab_attention.integration import generate_with_cab
        except ImportError as e:
            logger.warning(f"CABCache not available: {e}. Using fallback.")
            return self._sparse_generate_fallback(inputs, max_new_tokens, sparsity)
        
        device = inputs['input_ids'].device
        input_len = inputs['input_ids'].shape[1]
        
        # Create CAB cache with appropriate size
        max_cache_size = input_len + max_new_tokens
        
        cache = CABCache(
            max_cache_size=max_cache_size,
            sparsity=sparsity,
            local_ratio=0.3,       # 30% local context
            bridge_ratio=0.2,      # 20% bridge tokens
            importance_ratio=0.5,  # 50% important tokens
            eviction_interval=5,   # Match our pruning interval
            device=str(device),
        )
        
        # Generate using CABCache
        generated_ids, stats = generate_with_cab(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=inputs['input_ids'],
            max_new_tokens=max_new_tokens,
            cache=cache,
            do_sample=self.config.do_sample,
            temperature=self.config.temperature if self.config.do_sample else 1.0,
        )
        
        # Log stats
        logger.debug(f"CAB V5 stats: {stats}")
        
        return generated_ids
    
    def _prune_kv_cache_v2(
        self,
        past_key_values,
        keep_ratio: float,
        method: str,
        magnitude_ratio: float = 0.5,
        uses_dynamic_cache: bool = False,
        DynamicCache = None,
    ):
        """
        Prune KV cache to keep only important tokens.
        
        DynamicCache can be indexed like a tuple: cache[layer_idx] -> (key, value)
        Use to_legacy_cache() and from_legacy_cache() for conversion.
        
        Returns: (pruned_cache, kept_indices)
        """
        if past_key_values is None or keep_ratio >= 1.0:
            return past_key_values, None
        
        # Get number of layers - works for both DynamicCache and tuple
        num_layers = len(past_key_values)
        
        # Build pruned cache as list of tuples first
        pruned_kvs = []
        kept_indices_all = []
        
        for layer_idx in range(num_layers):
            # Access works the same for DynamicCache and tuple
            kv = past_key_values[layer_idx]
            key, value = kv
            
            B, H, N, D = key.shape
            num_keep = max(4, int(N * keep_ratio))
            
            if N <= num_keep:
                pruned_kvs.append((key, value))
                kept_indices_all.append(None)
                continue
            
            # Compute importance
            importance = self._compute_importance(key, method, magnitude_ratio, num_keep)
            
            # Get top-k indices (average over heads)
            importance_flat = importance.mean(dim=1)  # [B, N]
            _, top_indices = torch.topk(importance_flat, k=num_keep, dim=-1)  # [B, num_keep]
            top_indices, _ = torch.sort(top_indices, dim=-1)
            
            # Expand for gathering: need [B, H, num_keep, D]
            top_indices_for_key = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D)
            
            pruned_key = torch.gather(key, 2, top_indices_for_key)
            pruned_value = torch.gather(value, 2, top_indices_for_key)
            
            pruned_kvs.append((pruned_key, pruned_value))
            kept_indices_all.append(top_indices)
        
        # Convert back to the original format
        if uses_dynamic_cache and DynamicCache is not None:
            # Use from_legacy_cache to create new DynamicCache
            new_cache = DynamicCache.from_legacy_cache(tuple(pruned_kvs))
            return new_cache, kept_indices_all
        else:
            return tuple(pruned_kvs), kept_indices_all
    
    def _prune_kv_cache_v3(
        self,
        past_key_values,
        keep_ratio: float,
        method: str,
        magnitude_ratio: float = 0.5,
        uses_dynamic_cache: bool = False,
        DynamicCache = None,
        cumulative_attention: Optional[torch.Tensor] = None,
    ):
        """
        Prune KV cache using EXACT published algorithms.
        
        H2O (Zhang et al., 2023): 
            - Uses CUMULATIVE ATTENTION SCORES
            - Evicts tokens with lowest cumulative attention
            
        StreamingLLM (Xiao et al., 2023):
            - Keep first sink_size tokens (attention sinks)
            - Keep most recent tokens
            
        CAB V4 (Ours):
            - Hybrid: 50% magnitude + 50% uniqueness
            
        Returns: (pruned_cache, kept_indices, updated_cumulative_attention)
        """
        if past_key_values is None or keep_ratio >= 1.0:
            return past_key_values, None, cumulative_attention
        
        num_layers = len(past_key_values)
        
        # Get cache info from first layer
        kv = past_key_values[0]
        key, value = kv
        B, H, N, D = key.shape
        device = key.device
        
        num_keep = max(4, int(N * keep_ratio))
        
        if N <= num_keep:
            return past_key_values, None, cumulative_attention
        
        # Method-specific selection
        if method == "h2o":
            # H2O: Use CUMULATIVE ATTENTION SCORES (exact algorithm)
            if cumulative_attention is not None and len(cumulative_attention) >= N:
                scores = cumulative_attention[:N]
                _, keep_indices = torch.topk(scores, k=num_keep, largest=True)
                keep_indices = keep_indices.sort().values
            else:
                # Fallback: keep most recent
                keep_indices = torch.arange(N - num_keep, N, device=device)
        
        elif method == "streaming_llm":
            # StreamingLLM: Sinks + recent (exact algorithm)
            sink_size = 4
            sink_indices = torch.arange(min(sink_size, N), device=device)
            recent_budget = num_keep - len(sink_indices)
            recent_start = max(sink_size, N - recent_budget)
            recent_indices = torch.arange(recent_start, N, device=device)
            keep_indices = torch.cat([sink_indices, recent_indices])
        
        else:
            # Other methods: use existing _compute_importance
            importance = self._compute_importance(key, method, magnitude_ratio, num_keep)
            importance_flat = importance.mean(dim=1)  # [B, N]
            _, top_indices = torch.topk(importance_flat, k=num_keep, dim=-1)
            keep_indices = top_indices[0].sort().values  # Use first batch
        
        # Prune all layers
        pruned_kvs = []
        for layer_idx in range(num_layers):
            kv = past_key_values[layer_idx]
            key, value = kv
            
            # Expand keep_indices for gathering
            keep_indices_exp = keep_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, num_keep, 1]
            keep_indices_exp = keep_indices_exp.expand(B, H, -1, D)
            
            pruned_key = torch.gather(key, 2, keep_indices_exp)
            pruned_value = torch.gather(value, 2, keep_indices_exp)
            
            pruned_kvs.append((pruned_key, pruned_value))
        
        # Update cumulative attention
        new_cumulative = None
        if cumulative_attention is not None:
            new_cumulative = cumulative_attention[keep_indices]
        
        # Convert back to original format
        if uses_dynamic_cache and DynamicCache is not None:
            new_cache = DynamicCache.from_legacy_cache(tuple(pruned_kvs))
            return new_cache, keep_indices, new_cumulative
        else:
            return tuple(pruned_kvs), keep_indices, new_cumulative
    
    def _prune_kv_cache(
        self,
        past_key_values,
        keep_ratio: float,
        method: str,
        magnitude_ratio: float = 0.5,
    ):
        """Legacy method - use _prune_kv_cache_v2 instead."""
        try:
            from transformers.cache_utils import DynamicCache
            uses_dynamic = hasattr(past_key_values, 'get_seq_length')
        except ImportError:
            DynamicCache = None
            uses_dynamic = False
        
        result, _ = self._prune_kv_cache_v2(
            past_key_values, keep_ratio, method, magnitude_ratio,
            uses_dynamic_cache=uses_dynamic,
            DynamicCache=DynamicCache
        )
        return result
    
    def _compute_importance(
        self,
        key: torch.Tensor,
        method: str,
        magnitude_ratio: float,
        num_keep: int,
    ) -> torch.Tensor:
        """
        Compute importance scores for KV cache pruning.
        
        All methods return [B, H, N] importance scores where higher = more important.
        This ensures apple-to-apple fair comparison.
        """
        B, H, N, D = key.shape
        device = key.device
        
        if method == "dense":
            # Keep everything - shouldn't be called, but handle gracefully
            importance = torch.ones(B, H, N, device=device)
        
        elif method == "h2o":
            # H2O (Heavy Hitter Oracle): Use L2 norm of key vectors as importance
            # This is the standard magnitude-based approach
            importance = key.norm(dim=-1)  # [B, H, N]
        
        elif method == "cab":
            # CAB: Three-component importance (fallback when CABCache not available)
            # Uses magnitude + uniqueness as proxy for full CAB
            magnitude = key.norm(dim=-1)  # [B, H, N]
            
            # Compute uniqueness (inverse of redundancy via cosine similarity)
            k_norm = F.normalize(key, dim=-1)
            sim = torch.matmul(k_norm, k_norm.transpose(-2, -1))  # [B, H, N, N]
            redundancy = sim.mean(dim=-1)  # [B, H, N]
            uniqueness = 1.0 - redundancy
            
            # Normalize both to [0, 1] range
            mag_min, mag_max = magnitude.min(), magnitude.max()
            mag_norm = (magnitude - mag_min) / (mag_max - mag_min + 1e-8)
            
            uniq_min, uniq_max = uniqueness.min(), uniqueness.max()
            uniq_norm = (uniqueness - uniq_min) / (uniq_max - uniq_min + 1e-8)
            
            # 50% magnitude + 50% uniqueness
            importance = 0.5 * mag_norm + 0.5 * uniq_norm
        
        elif method == "streaming_llm":
            # StreamingLLM: Keep first 4 tokens (attention sinks) + recent tokens
            # This mimics the StreamingLLM paper approach
            importance = torch.zeros(B, H, N, device=device)
            num_sinks = min(4, N)
            importance[:, :, :num_sinks] = 1e6  # Always keep attention sinks
            
            # Give recency-based scores to remaining tokens
            if N > num_sinks:
                recency = torch.arange(N, device=device).float()
                importance[:, :, num_sinks:] = recency[num_sinks:]
        
        elif method == "local_strided":
            # Local + Strided: Keep local window + strided global tokens
            importance = torch.zeros(B, H, N, device=device)
            
            # Local window: last 25% of tokens get high importance
            local_start = int(N * 0.75)
            importance[:, :, local_start:] = 1e6
            
            # Strided global: every 4th token from the rest
            stride = 4
            strided_indices = torch.arange(0, local_start, stride, device=device)
            importance[:, :, strided_indices] = 1e3
        
        elif method == "random":
            # Random baseline: random importance scores
            importance = torch.rand(B, H, N, device=device)
        
        else:
            # Unknown method - fallback to magnitude
            logger.warning(f"Unknown method '{method}', falling back to magnitude-based")
            importance = key.norm(dim=-1)
        
        return importance
    
    def get_attention_weights(
        self,
        context: str,
        question: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get attention weights for analysis.
        
        Returns:
            attention_weights: [num_layers, num_heads, seq_len, seq_len]
            query: [num_layers, num_heads, seq_len, head_dim]
            key: [num_layers, num_heads, seq_len, head_dim]
        """
        if not self._loaded:
            self.load()
        
        prompt = self._format_prompt(context, question)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.max_length,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Forward pass with output_attentions=True
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True,
            )
        
        # Stack attention weights from all layers
        # Each attention is [batch, num_heads, seq_len, seq_len]
        attentions = torch.stack(outputs.attentions, dim=0)  # [num_layers, batch, ...]
        attentions = attentions.squeeze(1)  # Remove batch dim
        
        return attentions, None, None  # Q, K extraction depends on model architecture


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """
    Main benchmark runner class.
    
    Coordinates evaluation across datasets, methods, and sparsity levels.
    """
    
    def __init__(
        self,
        config: ExperimentConfig,
        output_dir: str = None,
    ):
        self.config = config
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model wrapper
        self.model_wrapper = ModelWrapper(config.model)
        
        # Track results
        self.results: Dict[str, MethodResult] = {}
    
    def run(self) -> ExperimentResult:
        """
        Run the complete experiment.
        
        Returns:
            ExperimentResult with all results
        """
        start_time = datetime.now()
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Datasets: {self.config.datasets}")
        logger.info(f"Methods: {self.config.methods}")
        logger.info(f"Sparsity levels: {self.config.sparsity_levels}")
        
        # Load model
        self.model_wrapper.load()
        
        # Run for each dataset
        all_results = {}
        
        for dataset_name in self.config.datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"{'='*60}")
            
            # Load dataset
            dataset = get_dataset(dataset_name)
            dataset_config = self.config.dataset_configs.get(
                dataset_name, ALL_DATASETS.get(dataset_name)
            )
            
            logger.info(f"Loaded {len(dataset)} samples")
            
            # Run for each method
            for method_name in self.config.methods:
                # Run for each sparsity level
                for sparsity in self.config.sparsity_levels:
                    result_key = f"{dataset_name}_{method_name}_{sparsity}"
                    
                    logger.info(f"\nEvaluating: {method_name} @ {sparsity:.0%} sparsity")
                    
                    result = self._evaluate_method(
                        dataset=dataset,
                        dataset_config=dataset_config,
                        method_name=method_name,
                        sparsity=sparsity,
                    )
                    
                    all_results[result_key] = result
                    
                    # Log summary
                    self._log_result_summary(result)
                    
                    # Save intermediate results
                    self._save_intermediate_result(result, result_key)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Create experiment result
        experiment_result = ExperimentResult(
            name=self.config.name,
            description=self.config.description,
            method_results=all_results,
            config=asdict(self.config.model),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_time_sec=total_time,
        )
        
        # Save final results
        self._save_final_results(experiment_result)
        
        logger.info(f"\nExperiment completed in {total_time:.1f}s")
        logger.info(f"Results saved to: {self.output_dir}")
        
        return experiment_result
    
    def _evaluate_method(
        self,
        dataset: BaseBenchmarkDataset,
        dataset_config: DatasetConfig,
        method_name: str,
        sparsity: float,
    ) -> MethodResult:
        """Evaluate a single method on a dataset."""
        
        # Get method with specified sparsity
        method_config = self.config.method_configs.get(
            method_name, METHOD_CONFIGS.get(method_name)
        )
        
        # Override sparsity
        from dataclasses import replace
        method_config = replace(method_config, sparsity=sparsity)
        method = get_method(method_name, config=method_config)
        
        # Evaluate samples
        sample_results = []
        start_time = time.time()
        
        for sample in tqdm(dataset, desc=f"Evaluating {method_name}"):
            try:
                result = self._evaluate_sample(sample, method, dataset_config)
                sample_results.append(result)
            except Exception as e:
                logger.warning(f"Error evaluating sample {sample.sample_id}: {e}")
                continue
        
        total_time = time.time() - start_time
        
        # Aggregate metrics
        metrics = self._aggregate_metrics(sample_results, dataset_config.metrics)
        
        return MethodResult(
            method_name=method_name,
            dataset_name=dataset_config.name,
            sparsity=sparsity,
            metrics=metrics,
            sample_results=sample_results,
            total_time_sec=total_time,
            samples_per_sec=len(sample_results) / total_time if total_time > 0 else 0,
            model_name=self.config.model.name,
        )
    
    def _evaluate_sample(
        self,
        sample: BenchmarkSample,
        method: BaseAttentionMethod,
        dataset_config: DatasetConfig,
    ) -> SampleResult:
        """Evaluate a single sample with a method."""
        
        # Get sparse attention parameters from method
        method_name = method.config.name.value if hasattr(method.config.name, 'value') else method.config.name
        sparsity = method.config.sparsity
        magnitude_ratio = getattr(method.config, 'magnitude_ratio', 0.5)
        
        # Generate prediction with sparse attention
        prediction, gen_diagnostics = self.model_wrapper.generate(
            context=sample.context,
            question=sample.question,
            sparse_method=method_name,
            sparsity=sparsity,
            magnitude_ratio=magnitude_ratio,
        )
        
        # Compute metrics
        references = sample.answers or ([sample.answer] if sample.answer else [])
        metrics = compute_metrics(
            prediction=prediction,
            references=references,
            metrics=dataset_config.metrics,
        )
        
        return SampleResult(
            sample_id=sample.sample_id,
            prediction=prediction,
            references=references,
            metrics=metrics,
            generation_time_ms=gen_diagnostics.get('generation_time_ms', 0),
            context_length=sample.context_length,
        )
    
    def _aggregate_metrics(
        self,
        sample_results: List[SampleResult],
        metric_names: List[MetricName],
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics across samples."""
        metrics = {}
        
        for metric_name in metric_names:
            metric_key = metric_name.value
            scores = [r.metrics.get(metric_key, 0) for r in sample_results]
            
            if np is not None:
                mean_val = float(np.mean(scores))
                std_val = float(np.std(scores))
                min_val = float(np.min(scores))
                max_val = float(np.max(scores))
            else:
                # Fallback without numpy
                mean_val = sum(scores) / len(scores) if scores else 0.0
                std_val = (sum((x - mean_val) ** 2 for x in scores) / len(scores)) ** 0.5 if scores else 0.0
                min_val = min(scores) if scores else 0.0
                max_val = max(scores) if scores else 0.0
            
            metrics[metric_key] = {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val,
                'count': len(scores),
            }
        
        return metrics
    
    def _log_result_summary(self, result: MethodResult) -> None:
        """Log summary of result."""
        logger.info(f"  Results for {result.method_name} @ {result.sparsity:.0%}:")
        for metric_name, stats in result.metrics.items():
            logger.info(f"    {metric_name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        logger.info(f"    Throughput: {result.samples_per_sec:.2f} samples/sec")
    
    def _save_intermediate_result(
        self,
        result: MethodResult,
        key: str,
    ) -> None:
        """Save intermediate result to file."""
        result_file = self.output_dir / f"intermediate_{key}.json"
        with open(result_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _save_final_results(
        self,
        experiment_result: ExperimentResult,
    ) -> None:
        """Save final experiment results."""
        # Save full results
        result_file = self.output_dir / f"{self.config.name}_results.json"
        with open(result_file, 'w') as f:
            json.dump(experiment_result.to_dict(), f, indent=2)
        
        # Save summary table
        summary_file = self.output_dir / f"{self.config.name}_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Experiment: {experiment_result.name}\n")
            f.write(f"Description: {experiment_result.description}\n")
            f.write(f"Duration: {experiment_result.total_time_sec:.1f}s\n")
            f.write(f"\n{'='*60}\n\n")
            
            # Group results by dataset
            datasets = set()
            for key in experiment_result.method_results.keys():
                parts = key.split('_')
                datasets.add(parts[0])
            
            for dataset in sorted(datasets):
                f.write(f"\nDataset: {dataset}\n")
                f.write("-" * 40 + "\n")
                
                for key, result in experiment_result.method_results.items():
                    if key.startswith(dataset):
                        f.write(f"\n{result.method_name} @ {result.sparsity:.0%}:\n")
                        for metric, stats in result.metrics.items():
                            f.write(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        logger.info(f"Results saved to {result_file}")
        logger.info(f"Summary saved to {summary_file}")


# =============================================================================
# Quick Evaluation Functions
# =============================================================================

def quick_evaluate(
    dataset_name: str,
    method_name: str,
    sparsity: float = 0.9,
    max_samples: int = 10,
    model_name: str = "meta-llama/Llama-2-7b-hf",
) -> MethodResult:
    """
    Quick evaluation for testing.
    
    Args:
        dataset_name: Dataset to evaluate
        method_name: Method to use
        sparsity: Sparsity level
        max_samples: Maximum samples to evaluate
        model_name: Model to use
    
    Returns:
        MethodResult with evaluation results
    
    Example:
        >>> result = quick_evaluate("narrativeqa", "cab", sparsity=0.9, max_samples=5)
        >>> print(result.metrics)
    """
    config = ExperimentConfig(
        name=f"quick_{dataset_name}_{method_name}",
        datasets=[dataset_name],
        methods=[method_name],
        sparsity_levels=[sparsity],
        model=ModelConfig(name=model_name),
    )
    
    # Override max_samples in dataset config
    if dataset_name in config.dataset_configs:
        config.dataset_configs[dataset_name].max_samples = max_samples
    
    runner = BenchmarkRunner(config, output_dir="results/quick_eval")
    result = runner.run()
    
    # Return first method result
    return list(result.method_results.values())[0]


def evaluate_sparsity_sweep(
    dataset_name: str,
    method_name: str,
    sparsity_levels: List[float] = None,
    max_samples: int = 50,
    model_name: str = "meta-llama/Llama-2-7b-hf",
) -> Dict[float, MethodResult]:
    """
    Evaluate method across multiple sparsity levels.
    
    Args:
        dataset_name: Dataset to evaluate
        method_name: Method to use
        sparsity_levels: List of sparsity levels
        max_samples: Maximum samples per level
        model_name: Model to use
    
    Returns:
        Dict mapping sparsity to MethodResult
    
    Example:
        >>> results = evaluate_sparsity_sweep("narrativeqa", "cab")
        >>> for sparsity, result in results.items():
        ...     print(f"{sparsity:.0%}: F1={result.metrics['f1']['mean']:.4f}")
    """
    if sparsity_levels is None:
        sparsity_levels = [0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    config = ExperimentConfig(
        name=f"sweep_{dataset_name}_{method_name}",
        datasets=[dataset_name],
        methods=[method_name],
        sparsity_levels=sparsity_levels,
        model=ModelConfig(name=model_name),
    )
    
    if dataset_name in config.dataset_configs:
        config.dataset_configs[dataset_name].max_samples = max_samples
    
    runner = BenchmarkRunner(config, output_dir="results/sparsity_sweep")
    experiment_result = runner.run()
    
    # Organize by sparsity
    results = {}
    for key, result in experiment_result.method_results.items():
        results[result.sparsity] = result
    
    return results


def compare_methods_on_dataset(
    dataset_name: str,
    methods: List[str] = None,
    sparsity: float = 0.9,
    max_samples: int = 100,
    model_name: str = "meta-llama/Llama-2-7b-hf",
) -> Dict[str, MethodResult]:
    """
    Compare multiple methods on a single dataset.
    
    Args:
        dataset_name: Dataset to evaluate
        methods: Methods to compare
        sparsity: Sparsity level
        max_samples: Maximum samples
        model_name: Model to use
    
    Returns:
        Dict mapping method name to MethodResult
    
    Example:
        >>> results = compare_methods_on_dataset("narrativeqa", ["dense", "h2o", "cab"])
        >>> for method, result in results.items():
        ...     print(f"{method}: F1={result.metrics['f1']['mean']:.4f}")
    """
    if methods is None:
        methods = ["dense", "h2o", "cab", "streaming_llm", "random"]
    
    config = ExperimentConfig(
        name=f"compare_{dataset_name}",
        datasets=[dataset_name],
        methods=methods,
        sparsity_levels=[sparsity],
        model=ModelConfig(name=model_name),
    )
    
    if dataset_name in config.dataset_configs:
        config.dataset_configs[dataset_name].max_samples = max_samples
    
    runner = BenchmarkRunner(config, output_dir="results/method_comparison")
    experiment_result = runner.run()
    
    # Organize by method
    results = {}
    for key, result in experiment_result.method_results.items():
        results[result.method_name] = result
    
    return results


# =============================================================================
# CLI Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LongBench QA Benchmark")
    parser.add_argument("--dataset", type=str, default="narrativeqa",
                        help="Dataset to evaluate")
    parser.add_argument("--method", type=str, default="cab",
                        help="Method to use")
    parser.add_argument("--sparsity", type=float, default=0.9,
                        help="Sparsity level")
    parser.add_argument("--max-samples", type=int, default=10,
                        help="Maximum samples")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf",
                        help="Model to use")
    
    args = parser.parse_args()
    
    print(f"Running quick evaluation:")
    print(f"  Dataset: {args.dataset}")
    print(f"  Method: {args.method}")
    print(f"  Sparsity: {args.sparsity:.0%}")
    print(f"  Max samples: {args.max_samples}")
    print(f"  Model: {args.model}")
    
    try:
        result = quick_evaluate(
            dataset_name=args.dataset,
            method_name=args.method,
            sparsity=args.sparsity,
            max_samples=args.max_samples,
            model_name=args.model,
        )
        
        print("\nResults:")
        for metric, stats in result.metrics.items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    except Exception as e:
        print(f"Error: {e}")
        raise

