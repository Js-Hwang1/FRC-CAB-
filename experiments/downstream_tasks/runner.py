"""
Benchmark Runner for Downstream Tasks

Orchestrates the full evaluation pipeline with proper sparse attention handling:
1. Load model and tokenizer
2. Load datasets
3. Apply sparse attention methods via KV cache pruning
4. Generate predictions
5. Compute metrics
6. Aggregate and save results

All methods use the same underlying mechanism for fair comparison.
"""

import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
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
    TaskType,
    ALL_DATASETS,
    METHOD_CONFIGS,
)
from .data_loaders import get_dataset, BenchmarkSample, BaseBenchmarkDataset
from .metrics import compute_metrics, compute_batch_metrics


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
    
    generation_time_ms: float = 0.0
    input_length: int = 0
    task_type: str = ""


@dataclass
class MethodResult:
    """Results for a single method on a dataset."""
    method_name: str
    dataset_name: str
    sparsity: float
    
    metrics: Dict[str, Dict[str, Any]]
    sample_results: List[SampleResult]
    
    total_time_sec: float = 0.0
    samples_per_sec: float = 0.0
    model_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
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
    method_results: Dict[str, MethodResult]
    config: Dict[str, Any]
    
    start_time: str = ""
    end_time: str = ""
    total_time_sec: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
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
# Model Wrapper with Unified Sparse Attention
# =============================================================================

class ModelWrapper:
    """
    Wrapper for LLM with unified sparse attention handling.
    
    Ensures fair apple-to-apple comparison between methods by using
    the same KV cache pruning mechanism with different importance functions.
    """
    
    def __init__(self, config: ModelConfig, device: str = None):
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
            raise ImportError("transformers required: pip install transformers")
        
        logger.info(f"Loading model: {self.config.name}")
        
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.name,
            trust_remote_code=True,
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_kwargs = {
            'torch_dtype': torch_dtype,
            'device_map': self.config.device_map,
            'trust_remote_code': True,
        }
        
        # Try optimized attention
        if self.config.use_flash_attention:
            try:
                import flash_attn
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                logger.info("Using Flash Attention 2")
            except ImportError:
                if TORCH_AVAILABLE and hasattr(F, 'scaled_dot_product_attention'):
                    model_kwargs['attn_implementation'] = 'sdpa'
                    logger.info("Using SDPA")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.name,
            **model_kwargs
        )
        
        self.model.eval()
        self._loaded = True
        logger.info(f"Model loaded on {self.device}")
    
    def generate(
        self,
        input_text: str,
        context: str = "",
        task_type: TaskType = TaskType.SUMMARIZATION,
        instruction: str = "",
        max_new_tokens: int = None,
        method: str = "dense",
        sparsity: float = 0.0,
        magnitude_ratio: float = 0.5,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate with sparse attention.
        
        Uses unified KV cache pruning for all sparse methods.
        """
        if not self._loaded:
            self.load()
        
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        
        # Format prompt based on task
        prompt = self._format_prompt(input_text, context, task_type, instruction)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=self.config.max_length - max_new_tokens,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        start_time = time.time()
        
        if method == "dense" or sparsity == 0:
            # Dense attention
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=self.config.do_sample,
                    temperature=self.config.temperature if self.config.do_sample else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            generated_ids = outputs[0, inputs['input_ids'].shape[1]:]
        else:
            # Sparse attention via KV cache pruning
            full_ids = self._sparse_generate(
                inputs, max_new_tokens, method, sparsity, magnitude_ratio
            )
            generated_ids = full_ids[0, inputs['input_ids'].shape[1]:]
        
        generation_time = (time.time() - start_time) * 1000
        
        # Decode
        prediction = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        prediction = self._clean_prediction(prediction, task_type)
        
        diagnostics = {
            'generation_time_ms': generation_time,
            'input_length': inputs['input_ids'].shape[1],
            'output_length': len(generated_ids),
        }
        
        return prediction, diagnostics
    
    def _format_prompt(
        self,
        input_text: str,
        context: str,
        task_type: TaskType,
        instruction: str,
    ) -> str:
        """Format prompt for different task types."""
        model_name = self.config.name.lower()
        
        # Build content based on task
        if task_type == TaskType.SUMMARIZATION:
            content = f"{instruction}\n\n{input_text}" if instruction else input_text
        elif task_type == TaskType.OPEN_DOMAIN_QA:
            if context:
                content = f"Context: {context}\n\nQuestion: {input_text}"
            else:
                content = f"Question: {input_text}"
        elif task_type == TaskType.DIALOGUE:
            content = f"{instruction}\n\nDialogue:\n{input_text}" if instruction else input_text
        elif task_type == TaskType.CODE:
            content = f"{instruction}\n\n{input_text}" if instruction else input_text
        else:
            content = input_text
        
        # Apply model-specific formatting
        if 'qwen' in model_name:
            prompt = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{content}

Answer concisely.<|im_end|>
<|im_start|>assistant
"""
        elif 'mistral' in model_name or 'mixtral' in model_name:
            prompt = f"[INST] {content}\n\nProvide a concise answer. [/INST]"
        elif 'llama' in model_name:
            prompt = f"[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n\n{content} [/INST]"
        else:
            prompt = f"{content}\n\nAnswer:"
        
        return prompt
    
    def _clean_prediction(self, prediction: str, task_type: TaskType) -> str:
        """Clean model output based on task type."""
        prediction = prediction.strip()
        
        # Stop at common markers
        for marker in ['\n\n', 'Question:', 'Context:', '<|im_end|>', '[INST]', '</s>']:
            if marker in prediction:
                prediction = prediction.split(marker)[0].strip()
        
        # Task-specific cleaning
        if task_type == TaskType.CODE:
            # Keep code as-is mostly
            pass
        elif task_type == TaskType.DIALOGUE:
            # Keep JSON-like output
            pass
        else:
            # For text tasks, take first paragraph
            if '\n' in prediction:
                prediction = prediction.split('\n')[0].strip()
        
        return prediction
    
    def _sparse_generate(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        method: str,
        sparsity: float,
        magnitude_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Generate with KV cache pruning."""
        keep_ratio = 1.0 - sparsity
        
        # Import DynamicCache
        try:
            from transformers.cache_utils import DynamicCache
            has_dynamic_cache = True
        except ImportError:
            has_dynamic_cache = False
            DynamicCache = None
        
        # Initial forward pass
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True,
            )
        
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        
        uses_dynamic_cache = has_dynamic_cache and isinstance(past_key_values, DynamicCache)
        
        # Prune initial cache
        past_key_values = self._prune_kv_cache(
            past_key_values, keep_ratio, method, magnitude_ratio,
            uses_dynamic_cache, DynamicCache
        )
        
        # Generate
        generated_ids = inputs['input_ids'].clone()
        
        for step in range(max_new_tokens):
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check EOS
            if self.tokenizer.eos_token_id is not None:
                if (next_token == self.tokenizer.eos_token_id).all():
                    break
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
            
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            
            # Prune periodically
            if (step + 1) % 5 == 0:
                uses_dynamic_cache = has_dynamic_cache and isinstance(past_key_values, DynamicCache)
                past_key_values = self._prune_kv_cache(
                    past_key_values, keep_ratio, method, magnitude_ratio,
                    uses_dynamic_cache, DynamicCache
                )
        
        return generated_ids
    
    def _prune_kv_cache(
        self,
        past_key_values,
        keep_ratio: float,
        method: str,
        magnitude_ratio: float = 0.5,
        uses_dynamic_cache: bool = False,
        DynamicCache = None,
    ):
        """Prune KV cache with unified importance scoring."""
        if past_key_values is None or keep_ratio >= 1.0:
            return past_key_values
        
        num_layers = len(past_key_values)
        pruned_kvs = []
        
        for layer_idx in range(num_layers):
            kv = past_key_values[layer_idx]
            key, value = kv
            
            B, H, N, D = key.shape
            num_keep = max(4, int(N * keep_ratio))
            
            if N <= num_keep:
                pruned_kvs.append((key, value))
                continue
            
            # Compute importance using unified method
            importance = self._compute_importance(key, method, magnitude_ratio, num_keep)
            
            # Select top-k
            importance_flat = importance.mean(dim=1)  # [B, N]
            _, top_indices = torch.topk(importance_flat, k=num_keep, dim=-1)
            top_indices, _ = torch.sort(top_indices, dim=-1)
            
            # Gather
            top_indices_exp = top_indices.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, D)
            pruned_key = torch.gather(key, 2, top_indices_exp)
            pruned_value = torch.gather(value, 2, top_indices_exp)
            
            pruned_kvs.append((pruned_key, pruned_value))
        
        # Convert back
        if uses_dynamic_cache and DynamicCache is not None:
            return DynamicCache.from_legacy_cache(tuple(pruned_kvs))
        else:
            return tuple(pruned_kvs)
    
    def _compute_importance(
        self,
        key: torch.Tensor,
        method: str,
        magnitude_ratio: float,
        num_keep: int,
    ) -> torch.Tensor:
        """
        Unified importance computation for all methods.
        
        This is the core of the fair comparison - all methods use the same
        mechanism with different importance functions.
        """
        B, H, N, D = key.shape
        device = key.device
        
        if method == "dense":
            return torch.ones(B, H, N, device=device)
        
        elif method == "h2o":
            # H2O: Pure magnitude (L2 norm)
            return key.norm(dim=-1)
        
        elif method == "cab_v3":
            # CAB V3: Pure FRC-based (uniqueness)
            k_norm = F.normalize(key, dim=-1)
            sim = torch.matmul(k_norm, k_norm.transpose(-2, -1))
            redundancy = sim.mean(dim=-1)
            return 1.0 - redundancy
        
        elif method == "cab_v4":
            # CAB V4: Hybrid magnitude + uniqueness
            magnitude = key.norm(dim=-1)
            
            k_norm = F.normalize(key, dim=-1)
            sim = torch.matmul(k_norm, k_norm.transpose(-2, -1))
            redundancy = sim.mean(dim=-1)
            uniqueness = 1.0 - redundancy
            
            # Normalize
            mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
            uniq_norm = (uniqueness - uniqueness.min()) / (uniqueness.max() - uniqueness.min() + 1e-8)
            
            return magnitude_ratio * mag_norm + (1 - magnitude_ratio) * uniq_norm
        
        elif method == "streaming_llm":
            importance = torch.zeros(B, H, N, device=device)
            importance[:, :, :4] = 1e6  # Attention sinks
            if N > 4:
                importance[:, :, 4:] = torch.arange(N - 4, device=device).float()
            return importance
        
        elif method == "local_strided":
            importance = torch.zeros(B, H, N, device=device)
            local_start = int(N * 0.75)
            importance[:, :, local_start:] = 1e6
            strided = torch.arange(0, local_start, 4, device=device)
            importance[:, :, strided] = 1e3
            return importance
        
        else:  # random
            return torch.rand(B, H, N, device=device)


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self, config: ExperimentConfig, output_dir: str = None):
        self.config = config
        self.output_dir = Path(output_dir or config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_wrapper = ModelWrapper(config.model)
        self.results: Dict[str, MethodResult] = {}
    
    def run(self) -> ExperimentResult:
        """Run complete experiment."""
        start_time = datetime.now()
        logger.info(f"Starting experiment: {self.config.name}")
        
        self.model_wrapper.load()
        
        all_results = {}
        
        for dataset_name in self.config.datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Dataset: {dataset_name}")
            logger.info(f"{'='*60}")
            
            dataset = get_dataset(dataset_name)
            dataset_config = self.config.dataset_configs.get(
                dataset_name, ALL_DATASETS.get(dataset_name)
            )
            
            logger.info(f"Loaded {len(dataset)} samples")
            
            for method_name in self.config.methods:
                for sparsity in self.config.sparsity_levels:
                    result_key = f"{dataset_name}_{method_name}_{sparsity}"
                    
                    logger.info(f"\nEvaluating: {method_name} @ {sparsity:.0%} sparsity")
                    
                    result = self._evaluate_method(
                        dataset, dataset_config, method_name, sparsity
                    )
                    
                    all_results[result_key] = result
                    self._log_summary(result)
                    self._save_intermediate(result, result_key)
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        experiment_result = ExperimentResult(
            name=self.config.name,
            description=self.config.description,
            method_results=all_results,
            config=asdict(self.config.model),
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_time_sec=total_time,
        )
        
        self._save_final(experiment_result)
        logger.info(f"\nCompleted in {total_time:.1f}s")
        
        return experiment_result
    
    def _evaluate_method(
        self,
        dataset: BaseBenchmarkDataset,
        dataset_config: DatasetConfig,
        method_name: str,
        sparsity: float,
    ) -> MethodResult:
        """Evaluate a method."""
        method_config = self.config.method_configs.get(
            method_name, METHOD_CONFIGS.get(method_name)
        )
        
        sample_results = []
        start_time = time.time()
        
        for sample in tqdm(dataset, desc=f"Evaluating {method_name}"):
            try:
                result = self._evaluate_sample(
                    sample, dataset_config, method_name, sparsity,
                    getattr(method_config, 'magnitude_ratio', 0.5)
                )
                sample_results.append(result)
            except Exception as e:
                logger.warning(f"Error on {sample.sample_id}: {e}")
                continue
        
        total_time = time.time() - start_time
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
        dataset_config: DatasetConfig,
        method_name: str,
        sparsity: float,
        magnitude_ratio: float,
    ) -> SampleResult:
        """Evaluate a single sample."""
        prediction, diagnostics = self.model_wrapper.generate(
            input_text=sample.input_text,
            context=sample.context,
            task_type=dataset_config.task_type,
            instruction=dataset_config.instruction,
            method=method_name,
            sparsity=sparsity if method_name != "dense" else 0.0,
            magnitude_ratio=magnitude_ratio,
        )
        
        metrics = compute_metrics(
            prediction=prediction,
            references=sample.targets,
            metrics=dataset_config.metrics,
        )
        
        return SampleResult(
            sample_id=sample.sample_id,
            prediction=prediction,
            references=sample.targets,
            metrics=metrics,
            generation_time_ms=diagnostics.get('generation_time_ms', 0),
            input_length=diagnostics.get('input_length', 0),
            task_type=dataset_config.task_type.value,
        )
    
    def _aggregate_metrics(
        self,
        sample_results: List[SampleResult],
        metric_names: List[MetricName],
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate metrics."""
        metrics = {}
        
        for metric_name in metric_names:
            key = metric_name.value
            scores = [r.metrics.get(key, 0) for r in sample_results]
            
            if np is not None:
                metrics[key] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'min': float(np.min(scores)),
                    'max': float(np.max(scores)),
                    'count': len(scores),
                }
            else:
                mean_val = sum(scores) / len(scores) if scores else 0
                std_val = (sum((x - mean_val) ** 2 for x in scores) / len(scores)) ** 0.5 if scores else 0
                metrics[key] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min(scores) if scores else 0,
                    'max': max(scores) if scores else 0,
                    'count': len(scores),
                }
        
        return metrics
    
    def _log_summary(self, result: MethodResult) -> None:
        """Log result summary."""
        logger.info(f"  {result.method_name} @ {result.sparsity:.0%}:")
        for metric, stats in result.metrics.items():
            logger.info(f"    {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    def _save_intermediate(self, result: MethodResult, key: str) -> None:
        """Save intermediate result."""
        path = self.output_dir / f"intermediate_{key}.json"
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def _save_final(self, result: ExperimentResult) -> None:
        """Save final results."""
        path = self.output_dir / f"{self.config.name}_results.json"
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Summary
        summary_path = self.output_dir / f"{self.config.name}_summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Experiment: {result.name}\n")
            f.write(f"Duration: {result.total_time_sec:.1f}s\n\n")
            
            for key, method_result in result.method_results.items():
                f.write(f"\n{key}:\n")
                for metric, stats in method_result.metrics.items():
                    f.write(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}\n")
        
        logger.info(f"Saved to {path}")

