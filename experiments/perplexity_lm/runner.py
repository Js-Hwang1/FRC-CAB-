"""
Benchmark Runner for Language Model Perplexity Evaluation

Orchestrates:
- Model loading with sparse attention methods
- Dataset iteration
- Perplexity computation
- Context length scaling analysis
- Sparsity trade-off curves
- Result aggregation and saving

ICML Publication-Quality Implementation
"""

import torch
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict
import math
from tqdm import tqdm

from .config import (
    ExperimentConfig, ModelConfig, DatasetConfig, MethodConfig,
    ContextLengthSweepConfig, SparsitySweepConfig,
    DATASET_CONFIGS, METHOD_CONFIGS, MethodName,
)
from .data_loaders import (
    create_perplexity_dataset, create_dataloader, get_dataset_stats,
)
from .metrics import (
    PerplexityEvaluator, PerplexityResult, format_perplexity_result,
    compute_perplexity_statistics, aggregate_results,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Sparse Perplexity Evaluator
# =============================================================================

class SparsePerplexityEvaluator:
    """
    Perplexity evaluator with sparse attention via attention masking.
    
    For fair comparison, all methods use the same evaluation approach:
    1. Compute attention scores for the full sequence
    2. Create sparse attention mask based on importance scores
    3. Re-compute forward pass with masked attention
    
    This ensures apple-to-apple comparison between methods.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device,
        max_length: int = 4096,
        method: str = "dense",
        sparsity: float = 0.0,
        magnitude_ratio: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.method = method
        self.sparsity = sparsity
        self.magnitude_ratio = magnitude_ratio
        
        self.model.eval()
    
    def _compute_importance(
        self,
        hidden_states: torch.Tensor,
        method: str,
        magnitude_ratio: float,
    ) -> torch.Tensor:
        """
        Compute importance scores for each position.
        
        Args:
            hidden_states: [B, N, D] hidden states
            method: Importance method
            magnitude_ratio: For CAB V4
        
        Returns:
            importance: [B, N] importance scores
        """
        B, N, D = hidden_states.shape
        device = hidden_states.device
        
        if method == "dense":
            return torch.ones(B, N, device=device)
        
        elif method == "h2o":
            # H2O: L2 norm of hidden states
            return hidden_states.norm(dim=-1)  # [B, N]
        
        elif method == "cab_v3":
            # CAB V3: Pure uniqueness (inverse redundancy)
            h_norm = torch.nn.functional.normalize(hidden_states, dim=-1)
            sim = torch.matmul(h_norm, h_norm.transpose(-2, -1))  # [B, N, N]
            redundancy = sim.mean(dim=-1)  # [B, N]
            return 1.0 - redundancy
        
        elif method == "cab_v4":
            # CAB V4: Hybrid magnitude + uniqueness
            magnitude = hidden_states.norm(dim=-1)  # [B, N]
            
            h_norm = torch.nn.functional.normalize(hidden_states, dim=-1)
            sim = torch.matmul(h_norm, h_norm.transpose(-2, -1))
            redundancy = sim.mean(dim=-1)
            uniqueness = 1.0 - redundancy
            
            # Normalize to [0, 1]
            mag_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
            uniq_norm = (uniqueness - uniqueness.min()) / (uniqueness.max() - uniqueness.min() + 1e-8)
            
            return magnitude_ratio * mag_norm + (1 - magnitude_ratio) * uniq_norm
        
        elif method == "streaming_llm":
            # StreamingLLM: attention sinks + recency
            importance = torch.zeros(B, N, device=device)
            importance[:, :4] = 1e6  # Attention sinks
            importance[:, 4:] = torch.arange(N - 4, device=device).float().unsqueeze(0)
            return importance
        
        elif method == "local_strided":
            # Local window + strided global
            importance = torch.zeros(B, N, device=device)
            local_start = int(N * 0.75)
            importance[:, local_start:] = 1e6
            strided = torch.arange(0, local_start, 4, device=device)
            importance[:, strided] = 1e3
            return importance
        
        else:  # random
            return torch.rand(B, N, device=device)
    
    def _create_sparse_attention_mask(
        self,
        importance: torch.Tensor,
        sparsity: float,
    ) -> torch.Tensor:
        """
        Create causal sparse attention mask.
        
        Args:
            importance: [B, N] importance scores
            sparsity: Fraction to mask (0 = dense, 0.9 = keep 10%)
        
        Returns:
            mask: [B, 1, N, N] attention mask (0 = attend, -inf = mask)
        """
        B, N = importance.shape
        device = importance.device
        
        # Base causal mask
        causal = torch.triu(torch.ones(N, N, device=device), diagonal=1).bool()
        
        if sparsity == 0:
            # Dense: just causal mask
            mask = torch.zeros(B, 1, N, N, device=device)
            mask[:, :, causal] = float('-inf')
            return mask
        
        # For each query position, keep top-k keys
        keep_ratio = 1.0 - sparsity
        
        # Create sparse mask
        mask = torch.full((B, N, N), float('-inf'), device=device)
        
        for i in range(N):
            # For position i, can attend to positions 0..i (causal)
            if i == 0:
                mask[:, i, 0] = 0
                continue
            
            # Get importance of positions 0..i
            pos_importance = importance[:, :i+1]  # [B, i+1]
            
            # Keep top-k
            num_keep = max(1, int((i + 1) * keep_ratio))
            _, top_idx = torch.topk(pos_importance, k=num_keep, dim=-1)  # [B, num_keep]
            
            # Set mask
            for b in range(B):
                mask[b, i, top_idx[b]] = 0
        
        return mask.unsqueeze(1)  # [B, 1, N, N]
    
    @torch.no_grad()
    def evaluate_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> Tuple[float, float, int]:
        """
        Evaluate perplexity on a batch with sparse attention.
        """
        B, N = input_ids.shape
        input_ids = input_ids.to(self.device)
        
        if self.method == "dense" or self.sparsity == 0:
            # Standard dense evaluation
            outputs = self.model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss.item()
            num_tokens = N - 1
            return math.exp(loss), loss, num_tokens
        
        # For sparse methods, we need to evaluate with modified attention
        # Get hidden states from first layer to compute importance
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            return_dict=True,
        )
        
        # Use first layer hidden states for importance
        hidden_states = outputs.hidden_states[1]  # [B, N, D]
        
        # Compute importance scores
        importance = self._compute_importance(
            hidden_states, self.method, self.magnitude_ratio
        )
        
        # Create sparse attention mask
        sparse_mask = self._create_sparse_attention_mask(importance, self.sparsity)
        
        # Forward pass with sparse attention mask
        # Note: This requires the model to accept attention_mask in the right format
        # For models that don't support 4D attention masks, we fall back to token dropping
        
        try:
            # Try with 4D attention mask
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=sparse_mask,
                labels=input_ids,
                return_dict=True,
            )
            loss = outputs.loss.item()
        except (TypeError, RuntimeError):
            # Fall back to top-k token selection (drop masked tokens)
            keep_ratio = 1.0 - self.sparsity
            num_keep = max(4, int(N * keep_ratio))
            
            # Select top-k positions based on importance
            _, keep_idx = torch.topk(importance, k=num_keep, dim=-1)
            keep_idx, _ = torch.sort(keep_idx, dim=-1)
            
            # Gather tokens
            kept_ids = torch.gather(input_ids, 1, keep_idx)
            
            # Evaluate on kept tokens
            outputs = self.model(input_ids=kept_ids, labels=kept_ids)
            loss = outputs.loss.item()
        
        num_tokens = N - 1
        perplexity = math.exp(loss)
        
        return perplexity, loss, num_tokens
    
    @torch.no_grad()
    def evaluate_dataset(
        self,
        dataloader,
        max_samples: int = None,
        verbose: bool = True,
    ) -> PerplexityResult:
        """
        Evaluate perplexity on entire dataset.
        """
        total_nll = 0.0
        total_tokens = 0
        num_samples = 0
        per_sample_ppl = []
        per_sample_ce = []
        
        for batch_idx, batch in enumerate(dataloader):
            if max_samples and num_samples >= max_samples:
                break
            
            input_ids = batch['input_ids']
            attention_mask = batch.get('attention_mask')
            
            B = input_ids.size(0)
            
            for i in range(B):
                if max_samples and num_samples >= max_samples:
                    break
                
                sample_ids = input_ids[i:i+1]
                
                ppl, ce, n_tokens = self.evaluate_batch(sample_ids)
                
                if not math.isnan(ce):
                    total_nll += ce * n_tokens
                    total_tokens += n_tokens
                    per_sample_ppl.append(ppl)
                    per_sample_ce.append(ce)
                
                num_samples += 1
            
            if verbose and (batch_idx + 1) % 10 == 0:
                current_ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('nan')
                logger.info(f"Processed {num_samples} samples, current PPL: {current_ppl:.2f}")
        
        if total_tokens == 0:
            return PerplexityResult(
                perplexity=float('nan'),
                cross_entropy=float('nan'),
                bits_per_token=float('nan'),
                num_tokens=0,
                num_samples=num_samples,
            )
        
        avg_ce = total_nll / total_tokens
        final_ppl = math.exp(avg_ce)
        bpt = avg_ce / math.log(2)
        
        return PerplexityResult(
            perplexity=final_ppl,
            cross_entropy=avg_ce,
            bits_per_token=bpt,
            num_tokens=total_tokens,
            num_samples=num_samples,
            per_sample_ppl=per_sample_ppl,
            per_sample_ce=per_sample_ce,
        )


# =============================================================================
# Benchmark Runner
# =============================================================================

class PerplexityBenchmarkRunner:
    """
    Main benchmark runner for perplexity evaluation.
    
    Supports:
    - Multiple datasets (WikiText-103, C4, PG-19)
    - Multiple methods (Dense, H2O, CAB V4, etc.)
    - Context length scaling analysis
    - Sparsity trade-off curves
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup logging
        self._setup_logging()
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components (lazy loading)
        self.model = None
        self.tokenizer = None
        self.datasets = {}
        self.results = {}
    
    def _setup_logging(self) -> None:
        """Configure logging."""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
    
    def load_model(self) -> None:
        """Load model and tokenizer."""
        if self.model is not None:
            return
        
        logger.info(f"Loading model: {self.config.model.name}")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError("transformers required: pip install transformers")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            trust_remote_code=True,
        )
        
        # Set padding token if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        dtype = getattr(torch, self.config.model.torch_dtype, torch.float16)
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name,
                torch_dtype=dtype,
                device_map=self.config.model.device_map,
                trust_remote_code=True,
                attn_implementation="flash_attention_2" if self.config.model.use_flash_attention else None,
            )
        except Exception as e:
            logger.warning(f"Failed to load with flash attention: {e}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model.name,
                torch_dtype=dtype,
                device_map=self.config.model.device_map,
                trust_remote_code=True,
            )
        
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
    
    def load_datasets(self) -> None:
        """Load all configured datasets."""
        if self.datasets:
            return
        
        self.load_model()  # Ensure tokenizer is available
        
        for ds_name in self.config.datasets:
            logger.info(f"Loading dataset: {ds_name}")
            
            ds_config = self.config.dataset_configs.get(ds_name)
            if ds_config is None:
                ds_config = DATASET_CONFIGS[ds_name]
            
            self.datasets[ds_name] = create_perplexity_dataset(
                config=ds_config,
                tokenizer=self.tokenizer,
                max_length=self.config.model.max_length,
            )
            
            stats = get_dataset_stats(self.datasets[ds_name])
            logger.info(f"  {ds_name}: {stats}")
    
    def run_single_evaluation(
        self,
        dataset_name: str,
        method_name: str,
        sparsity: float,
        context_length: int,
        max_samples: Optional[int] = None,
    ) -> PerplexityResult:
        """
        Run perplexity evaluation for a single configuration.
        
        Args:
            dataset_name: Name of dataset
            method_name: Name of attention method
            sparsity: Target sparsity level
            context_length: Maximum context length
            max_samples: Optional sample limit
        
        Returns:
            PerplexityResult
        """
        self.load_model()
        self.load_datasets()
        
        dataset = self.datasets[dataset_name]
        
        # Create dataloader with context length limit
        # Recreate dataset with adjusted max_length if needed
        if context_length != self.config.model.max_length:
            ds_config = self.config.dataset_configs.get(dataset_name)
            if ds_config is None:
                ds_config = DATASET_CONFIGS[dataset_name]
            
            dataset = create_perplexity_dataset(
                config=ds_config,
                tokenizer=self.tokenizer,
                max_length=context_length,
            )
        
        dataloader = create_dataloader(
            dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        
        logger.info(f"Evaluating: {dataset_name} | {method_name} | sparsity={sparsity} | ctx={context_length}")
        
        # Get method config for magnitude_ratio
        method_config = self.config.method_configs.get(method_name)
        if method_config is None:
            method_config = METHOD_CONFIGS.get(method_name)
        magnitude_ratio = getattr(method_config, 'magnitude_ratio', 0.5) if method_config else 0.5
        
        # Create evaluator with sparse attention support
        evaluator = SparsePerplexityEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_length=context_length,
            method=method_name,
            sparsity=sparsity,
            magnitude_ratio=magnitude_ratio,
        )
        
        # Run evaluation
        start_time = time.time()
        result = evaluator.evaluate_dataset(
            dataloader,
            max_samples=max_samples,
            verbose=True,
        )
        elapsed = time.time() - start_time
        
        logger.info(f"  Result: {format_perplexity_result(result)} ({elapsed:.1f}s)")
        
        return result
    
    def run_context_length_sweep(
        self,
        dataset_name: str,
        method_name: str,
        sparsity: float = 0.9,
    ) -> Dict[int, PerplexityResult]:
        """
        Run perplexity evaluation across different context lengths.
        
        Args:
            dataset_name: Dataset to evaluate on
            method_name: Attention method
            sparsity: Fixed sparsity level
        
        Returns:
            Dict mapping context_length -> PerplexityResult
        """
        sweep_config = self.config.context_length_sweep
        if not sweep_config.enabled:
            return {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Context Length Sweep: {dataset_name} | {method_name}")
        logger.info(f"Context lengths: {sweep_config.context_lengths}")
        logger.info(f"{'='*60}")
        
        results = {}
        
        for ctx_len in sweep_config.context_lengths:
            try:
                result = self.run_single_evaluation(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    sparsity=sparsity,
                    context_length=ctx_len,
                )
                results[ctx_len] = result
            except Exception as e:
                logger.error(f"Error at context_length={ctx_len}: {e}")
                results[ctx_len] = PerplexityResult(
                    perplexity=float('nan'),
                    cross_entropy=float('nan'),
                    bits_per_token=float('nan'),
                )
        
        return results
    
    def run_sparsity_sweep(
        self,
        dataset_name: str,
        method_name: str,
        context_length: int = 4096,
    ) -> Dict[float, PerplexityResult]:
        """
        Run perplexity evaluation across different sparsity levels.
        
        Args:
            dataset_name: Dataset to evaluate on
            method_name: Attention method
            context_length: Fixed context length
        
        Returns:
            Dict mapping sparsity -> PerplexityResult
        """
        sweep_config = self.config.sparsity_sweep
        if not sweep_config.enabled:
            return {}
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Sparsity Sweep: {dataset_name} | {method_name}")
        logger.info(f"Sparsity levels: {sweep_config.sparsity_levels}")
        logger.info(f"{'='*60}")
        
        results = {}
        
        for sparsity in sweep_config.sparsity_levels:
            try:
                result = self.run_single_evaluation(
                    dataset_name=dataset_name,
                    method_name=method_name,
                    sparsity=sparsity,
                    context_length=context_length,
                )
                results[sparsity] = result
            except Exception as e:
                logger.error(f"Error at sparsity={sparsity}: {e}")
                results[sparsity] = PerplexityResult(
                    perplexity=float('nan'),
                    cross_entropy=float('nan'),
                    bits_per_token=float('nan'),
                )
        
        return results
    
    def run_full_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark as configured.
        
        Returns:
            Dict with all results organized by dataset/method/sweep
        """
        logger.info("\n" + "="*70)
        logger.info("PERPLEXITY BENCHMARK")
        logger.info("="*70)
        logger.info(f"Experiment: {self.config.name}")
        logger.info(f"Datasets: {self.config.datasets}")
        logger.info(f"Methods: {self.config.methods}")
        logger.info("="*70 + "\n")
        
        all_results = {
            'config': {
                'name': self.config.name,
                'description': self.config.description,
                'model': self.config.model.name,
                'datasets': self.config.datasets,
                'methods': self.config.methods,
                'timestamp': datetime.now().isoformat(),
            },
            'results': {},
        }
        
        for dataset_name in self.config.datasets:
            all_results['results'][dataset_name] = {}
            
            for method_name in self.config.methods:
                method_results = {}
                
                # Run context length sweep
                if self.config.context_length_sweep.enabled:
                    ctx_results = self.run_context_length_sweep(
                        dataset_name=dataset_name,
                        method_name=method_name,
                        sparsity=self.config.context_length_sweep.fixed_sparsity,
                    )
                    method_results['context_length_sweep'] = {
                        str(k): v.to_dict() for k, v in ctx_results.items()
                    }
                
                # Run sparsity sweep
                if self.config.sparsity_sweep.enabled:
                    sparsity_results = self.run_sparsity_sweep(
                        dataset_name=dataset_name,
                        method_name=method_name,
                        context_length=self.config.sparsity_sweep.fixed_context_length,
                    )
                    method_results['sparsity_sweep'] = {
                        str(k): v.to_dict() for k, v in sparsity_results.items()
                    }
                
                all_results['results'][dataset_name][method_name] = method_results
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.config.name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {filepath}")
        
        # Also save a latest symlink
        latest_path = self.output_dir / f"{self.config.name}_latest.json"
        if latest_path.exists():
            latest_path.unlink()
        with open(latest_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_context_scaling(
    results: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Analyze how perplexity scales with context length.
    
    Returns analysis for plotting Figure 4 (Context Length Scaling).
    """
    analysis = {
        'dataset': dataset_name,
        'methods': {},
    }
    
    dataset_results = results.get('results', {}).get(dataset_name, {})
    
    for method_name, method_results in dataset_results.items():
        sweep = method_results.get('context_length_sweep', {})
        
        if not sweep:
            continue
        
        # Extract data points
        context_lengths = []
        perplexities = []
        
        for ctx_len_str, ppl_result in sorted(sweep.items(), key=lambda x: int(x[0])):
            ctx_len = int(ctx_len_str)
            ppl = ppl_result.get('perplexity', float('nan'))
            
            if not math.isnan(ppl):
                context_lengths.append(ctx_len)
                perplexities.append(ppl)
        
        analysis['methods'][method_name] = {
            'context_lengths': context_lengths,
            'perplexities': perplexities,
            'ppl_decrease': perplexities[0] - perplexities[-1] if len(perplexities) >= 2 else 0,
        }
    
    return analysis


def analyze_sparsity_tradeoff(
    results: Dict[str, Any],
    dataset_name: str,
) -> Dict[str, Any]:
    """
    Analyze perplexity vs sparsity trade-off.
    
    Returns analysis for plotting Figure 8 (Pareto Frontier).
    """
    analysis = {
        'dataset': dataset_name,
        'methods': {},
    }
    
    dataset_results = results.get('results', {}).get(dataset_name, {})
    
    for method_name, method_results in dataset_results.items():
        sweep = method_results.get('sparsity_sweep', {})
        
        if not sweep:
            continue
        
        # Extract data points
        sparsity_levels = []
        perplexities = []
        
        for sparsity_str, ppl_result in sorted(sweep.items(), key=lambda x: float(x[0])):
            sparsity = float(sparsity_str)
            ppl = ppl_result.get('perplexity', float('nan'))
            
            if not math.isnan(ppl):
                sparsity_levels.append(sparsity)
                perplexities.append(ppl)
        
        # Find dense baseline (sparsity=0)
        dense_ppl = None
        for s, p in zip(sparsity_levels, perplexities):
            if s == 0:
                dense_ppl = p
                break
        
        # Compute relative degradation
        relative_degradation = []
        for ppl in perplexities:
            if dense_ppl and dense_ppl > 0:
                rel_deg = (ppl - dense_ppl) / dense_ppl * 100
            else:
                rel_deg = float('nan')
            relative_degradation.append(rel_deg)
        
        analysis['methods'][method_name] = {
            'sparsity_levels': sparsity_levels,
            'perplexities': perplexities,
            'relative_degradation': relative_degradation,
            'dense_ppl': dense_ppl,
        }
    
    return analysis


def generate_summary_table(
    results: Dict[str, Any],
) -> str:
    """
    Generate markdown summary table for paper.
    
    Returns Table 3 style format.
    """
    lines = [
        "| Dataset | Method | Sparsity | Perplexity | Δ PPL (%) |",
        "|---------|--------|----------|------------|-----------|",
    ]
    
    for dataset_name, dataset_results in results.get('results', {}).items():
        for method_name, method_results in dataset_results.items():
            # Use sparsity sweep at 90% for comparison
            sweep = method_results.get('sparsity_sweep', {})
            
            if '0.0' in sweep and '0.9' in sweep:
                dense_ppl = sweep['0.0'].get('perplexity', float('nan'))
                sparse_ppl = sweep['0.9'].get('perplexity', float('nan'))
                
                if not math.isnan(dense_ppl) and not math.isnan(sparse_ppl):
                    delta = (sparse_ppl - dense_ppl) / dense_ppl * 100
                    lines.append(
                        f"| {dataset_name} | {method_name} | 0.9 | {sparse_ppl:.2f} | {delta:+.1f}% |"
                    )
    
    return "\n".join(lines)


# =============================================================================
# Entry Point
# =============================================================================

def run_benchmark(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Main entry point for running the benchmark.
    
    Args:
        config: Experiment configuration
    
    Returns:
        Dict with all results
    """
    runner = PerplexityBenchmarkRunner(config)
    return runner.run_full_benchmark()


if __name__ == "__main__":
    # Quick test
    from .config import create_quick_test
    
    config = create_quick_test()
    results = run_benchmark(config)
    
    print("\n" + "="*60)
    print("BENCHMARK COMPLETE")
    print("="*60)
    print(json.dumps(results['config'], indent=2))

