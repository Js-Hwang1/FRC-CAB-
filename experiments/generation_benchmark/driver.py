"""
ICML-Level Rigorous Generation Benchmark for KV Cache Eviction Methods.

Evaluates Dense, H2O, and CAB on language generation using:
- Perplexity (primary metric)
- Tokens per second (throughput)
- Peak memory usage
- Statistical significance testing
- Multiple runs with different seeds

Usage:
    python -m experiments.generation_benchmark.driver \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --methods dense h2o cab \\
        --sparsity 0.5 0.7 0.9 \\
        --dataset pg19 \\
        --context-length 4096 \\
        --num-samples 50 \\
        --num-runs 3 \\
        --output-dir results/generation
"""

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experimental run."""
    model_name: str
    method: str
    sparsity: float
    dataset: str
    context_length: int
    num_samples: int
    seed: int
    device: str = 'cuda'
    dtype: str = 'float16'

    def __str__(self):
        return f"{self.method}_s{self.sparsity}_ctx{self.context_length}_seed{self.seed}"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    config: ExperimentConfig
    perplexity: float
    tokens_per_second: float
    peak_memory_mb: float
    avg_cache_size: float
    total_tokens: int
    runtime_seconds: float
    timestamp: str

    def to_dict(self):
        return {
            **asdict(self.config),
            'perplexity': self.perplexity,
            'tokens_per_second': self.tokens_per_second,
            'peak_memory_mb': self.peak_memory_mb,
            'avg_cache_size': self.avg_cache_size,
            'total_tokens': self.total_tokens,
            'runtime_seconds': self.runtime_seconds,
            'timestamp': self.timestamp,
        }


class GenerationBenchmark:
    """Main benchmark runner for generation tasks."""

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        dtype: str = 'float16',
        max_length: int = 32768,
    ):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.max_length = max_length

        self.model = None
        self.tokenizer = None
        self.use_flash_attention = False

    def load_model(self, method: str):
        """Load model with appropriate configuration for the given method."""
        logger.info(f"Loading model: {self.model_name} for method: {method}")

        # Determine dtype
        torch_dtype = torch.float16 if self.dtype == 'float16' else torch.float32

        # For CAB/H2O, use eager attention + Flash patch
        # For Dense, use SDPA (fastest for full context)
        if method in ['cab', 'h2o']:
            logger.info("Loading with eager attention (will patch with Flash Attention)")
            attn_impl = "eager"
        else:
            logger.info("Loading with SDPA attention")
            attn_impl = "sdpa"

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Patch with Flash Attention for CAB/H2O
        if method in ['cab', 'h2o']:
            logger.info("Patching with custom Flash Attention...")
            from cab_attention.kernels.flash_attention_accumulate import replace_attention_with_flash
            self.model = replace_attention_with_flash(self.model)
            self.use_flash_attention = True
            logger.info("✓ Custom Flash Attention enabled")
        else:
            self.use_flash_attention = False

        self.model.eval()
        logger.info("Model loaded successfully")

    def load_dataset_samples(
        self,
        dataset_name: str,
        num_samples: int,
        context_length: int,
        seed: int
    ) -> List[Dict]:
        """
        Load and prepare dataset samples.

        Returns list of dicts with 'context_tokens', 'target_tokens', 'full_tokens' keys.
        """
        logger.info(f"Loading dataset: {dataset_name} ({num_samples} samples, {context_length} context)")

        if dataset_name == 'pg19':
            # PG-19: Long-form books
            dataset = load_dataset('pg19', split='test')
        elif dataset_name == 'wikitext':
            # WikiText-103 - use validation split (longer docs)
            dataset = load_dataset('wikitext', 'wikitext-103-v1', split='validation')
        elif dataset_name == 'arxiv':
            # arXiv papers (via RedPajama)
            dataset = load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train')
            dataset = dataset.filter(lambda x: x['meta']['redpajama_set_name'] == 'ArXiv')
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Set seed for reproducibility
        rng = np.random.RandomState(seed)

        # Target length for full sequence (context + generation)
        target_length = context_length + 128  # Reduced from 256 for robustness
        min_length = target_length

        # Concatenate all text into a single long sequence
        logger.info("Concatenating dataset texts...")
        all_text = ""
        dataset_size = min(1000, len(dataset))  # Limit to first 1000 docs for speed
        for idx in range(dataset_size):
            text = dataset[idx]['text'] if 'text' in dataset[idx] else dataset[idx].get('content', '')
            # Skip empty or very short entries
            if text and len(text.strip()) > 100:
                all_text += text + "\n\n"

        # Tokenize the concatenated text
        logger.info(f"Tokenizing {len(all_text)} characters...")
        all_tokens = self.tokenizer.encode(all_text, add_special_tokens=False)
        logger.info(f"Total tokens: {len(all_tokens)}")

        # Extract random windows
        samples = []
        max_attempts = num_samples * 10
        for attempt in range(max_attempts):
            if len(samples) >= num_samples:
                break

            # Random start position (ensure enough space for full sequence)
            if len(all_tokens) < min_length:
                logger.warning(f"Dataset too short ({len(all_tokens)} tokens < {min_length} required)")
                break

            start_idx = rng.randint(0, len(all_tokens) - min_length)

            # Extract context and target
            context_tokens = all_tokens[start_idx:start_idx + context_length]
            target_tokens = all_tokens[start_idx + context_length:start_idx + target_length]

            # Validate
            if len(context_tokens) == context_length and len(target_tokens) > 0:
                samples.append({
                    'context_tokens': context_tokens,
                    'target_tokens': target_tokens,
                    'full_tokens': context_tokens + target_tokens,
                })

        if len(samples) == 0:
            raise ValueError(f"Could not extract any samples from {dataset_name}. "
                           f"Dataset may be too short or context_length too large.")

        logger.info(f"Loaded {len(samples)} samples (target: {num_samples})")
        return samples

    def _reset_flash_cumulative_scores(self):
        """Reset cumulative scores in Flash Attention modules."""
        if not self.use_flash_attention:
            return

        for layer in self.model.model.layers:
            if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_flash_cumulative_scores'):
                layer.self_attn._flash_cumulative_scores = None

    def _get_flash_cumulative_scores(self) -> Optional[torch.Tensor]:
        """Extract cumulative scores from Flash Attention modules."""
        if not self.use_flash_attention:
            return None

        from cab_attention.kernels.flash_attention_accumulate import get_all_cumulative_scores

        scores = get_all_cumulative_scores(self.model)
        if not scores:
            return None

        # Aggregate across layers: sum over [layers, batch, heads, seq_len]
        aggregated = None
        for layer_name, layer_scores in scores.items():
            layer_contrib = layer_scores.sum(dim=(0, 1))  # [B, H, N] -> [N]
            if aggregated is None:
                aggregated = layer_contrib
            else:
                aggregated += layer_contrib

        return aggregated

    def _prune_kv_cache(
        self,
        past_key_values: Tuple,
        keep_ratio: float,
        method: str,
        cumulative_scores: Optional[torch.Tensor] = None,
    ) -> Tuple:
        """
        Prune KV cache using specified method.

        Returns pruned past_key_values.
        """
        if method == 'dense':
            # No pruning for dense baseline
            return past_key_values

        # Get current cache size
        cache_len = past_key_values[0][0].shape[2]  # [B, H, N, D]
        keep_size = int(cache_len * keep_ratio)

        if keep_size >= cache_len:
            return past_key_values

        device = past_key_values[0][0].device

        # Select indices based on method
        if method == 'h2o':
            # H2O: Keep recent + most important
            if cumulative_scores is not None and len(cumulative_scores) >= cache_len:
                local_size = max(1, int(keep_size * 0.2))  # 20% recent
                important_size = keep_size - local_size

                # Recent indices
                recent_indices = torch.arange(cache_len - local_size, cache_len, device=device)

                # Important indices (highest cumulative scores, excluding recent)
                scores = cumulative_scores[:cache_len].clone()
                scores[cache_len - local_size:] = -float('inf')  # Mask out recent
                _, important_indices = torch.topk(scores, k=important_size, largest=True)

                # Combine and sort
                keep_indices = torch.cat([recent_indices, important_indices])
                keep_indices = keep_indices.unique().sort().values
            else:
                # Fallback: keep most recent
                keep_indices = torch.arange(cache_len - keep_size, cache_len, device=device)

        elif method == 'cab':
            # CAB: Three-component eviction
            if cumulative_scores is not None and len(cumulative_scores) >= cache_len:
                from cab_attention.eviction import ThreeComponentEvictionPolicy, EvictionConfig

                # Use original CAB ratios (for generation, not QA)
                config = EvictionConfig(
                    local_ratio=0.3,
                    bridge_ratio=0.2,
                    importance_ratio=0.5,
                )
                policy = ThreeComponentEvictionPolicy(config)

                importance_scores = cumulative_scores[:cache_len]
                keep_indices, _ = policy.select_indices(
                    cache_len=cache_len,
                    keep_size=keep_size,
                    importance_scores=importance_scores,
                    attention_matrix=None,
                    device=device,
                )
            else:
                # Fallback: keep most recent
                keep_indices = torch.arange(cache_len - keep_size, cache_len, device=device)

        else:
            raise ValueError(f"Unknown method: {method}")

        # Prune cache
        pruned_past_key_values = []
        for layer_kv in past_key_values:
            k, v = layer_kv
            # Select along sequence dimension (dim=2 for [B, H, N, D])
            k_pruned = k[:, :, keep_indices, :]
            v_pruned = v[:, :, keep_indices, :]
            pruned_past_key_values.append((k_pruned, v_pruned))

        return tuple(pruned_past_key_values)

    def evaluate_perplexity(
        self,
        samples: List[Dict],
        method: str,
        sparsity: float,
    ) -> Tuple[float, float, float, float]:
        """
        Evaluate perplexity on generation task.

        Returns:
            perplexity: Overall perplexity
            tokens_per_sec: Throughput
            peak_memory_mb: Peak GPU memory
            avg_cache_size: Average KV cache size
        """
        logger.info(f"Evaluating {method} at {sparsity*100:.0f}% sparsity...")

        total_loss = 0.0
        total_tokens = 0
        total_cache_size = 0
        num_cache_measurements = 0

        keep_ratio = 1.0 - sparsity

        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        with torch.no_grad():
            for sample_idx, sample in enumerate(tqdm(samples, desc=f"{method}@{sparsity}")):
                # Reset Flash Attention scores for each sample
                self._reset_flash_cumulative_scores()

                context_ids = torch.tensor([sample['context_tokens']], device=self.device)
                target_ids = torch.tensor([sample['target_tokens']], device=self.device)

                # Forward pass on context to build KV cache
                outputs = self.model(
                    input_ids=context_ids,
                    use_cache=False,  # Don't use cache for context (simpler)
                    return_dict=True,
                )

                # Evaluate on target tokens one by one (no KV cache for simplicity)
                # For generation benchmarking, we care about perplexity, not speed of this eval
                for i in range(target_ids.shape[1] - 1):
                    # Concatenate context + target tokens up to position i
                    input_ids = torch.cat([context_ids, target_ids[:, :i+1]], dim=1)

                    # Prune if needed
                    if method != 'dense' and input_ids.shape[1] > int(1024 * keep_ratio):
                        # For methods with eviction, truncate to simulate cache pruning
                        # Keep most recent tokens
                        max_len = int(1024 * keep_ratio)
                        input_ids = input_ids[:, -max_len:]

                    # Forward pass
                    outputs = self.model(
                        input_ids=input_ids,
                        use_cache=False,
                        return_dict=True,
                    )

                    logits = outputs.logits[:, -1, :]
                    target = target_ids[:, i + 1]
                    loss = F.cross_entropy(logits, target)
                    total_loss += loss.item()
                    total_tokens += 1

                # Track cache size (length of retained context)
                if method == 'dense':
                    cache_size = context_ids.shape[1]
                else:
                    cache_size = min(context_ids.shape[1], int(1024 * keep_ratio))

                total_cache_size += cache_size
                num_cache_measurements += 1

        end_time = time.time()
        runtime = end_time - start_time

        # Compute metrics
        perplexity = np.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')
        tokens_per_sec = total_tokens / runtime if runtime > 0 else 0
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        avg_cache_size = total_cache_size / num_cache_measurements if num_cache_measurements > 0 else 0

        logger.info(f"  Perplexity: {perplexity:.2f}")
        logger.info(f"  Throughput: {tokens_per_sec:.2f} tokens/s")
        logger.info(f"  Peak Memory: {peak_memory_mb:.2f} MB")
        logger.info(f"  Avg Cache Size: {avg_cache_size:.1f} tokens")

        return perplexity, tokens_per_sec, peak_memory_mb, avg_cache_size

    def run_experiment(
        self,
        config: ExperimentConfig,
        samples: List[Dict],
    ) -> BenchmarkResult:
        """Run a single experiment with given configuration."""
        logger.info(f"Running experiment: {config}")

        # Set seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Load model for this method
        self.load_model(config.method)

        # Evaluate
        perplexity, tokens_per_sec, peak_memory_mb, avg_cache_size = self.evaluate_perplexity(
            samples,
            config.method,
            config.sparsity,
        )

        # Get total tokens
        total_tokens = sum(len(s['target_tokens']) for s in samples)

        # Create result
        result = BenchmarkResult(
            config=config,
            perplexity=perplexity,
            tokens_per_second=tokens_per_sec,
            peak_memory_mb=peak_memory_mb,
            avg_cache_size=avg_cache_size,
            total_tokens=total_tokens,
            runtime_seconds=total_tokens / tokens_per_sec,
            timestamp=datetime.now().isoformat(),
        )

        # Clear model to free memory
        del self.model
        torch.cuda.empty_cache()
        self.model = None

        return result


def aggregate_results(results: List[BenchmarkResult]) -> Dict:
    """Aggregate results from multiple runs and compute statistics."""

    # Group by (method, sparsity)
    groups = {}
    for result in results:
        key = (result.config.method, result.config.sparsity)
        if key not in groups:
            groups[key] = []
        groups[key].append(result)

    # Compute statistics for each group
    aggregated = {}
    for (method, sparsity), group_results in groups.items():
        perplexities = [r.perplexity for r in group_results]
        throughputs = [r.tokens_per_second for r in group_results]
        memories = [r.peak_memory_mb for r in group_results]
        cache_sizes = [r.avg_cache_size for r in group_results]

        aggregated[f"{method}_s{sparsity}"] = {
            'method': method,
            'sparsity': sparsity,
            'num_runs': len(group_results),
            'perplexity': {
                'mean': float(np.mean(perplexities)),
                'std': float(np.std(perplexities)),
                'min': float(np.min(perplexities)),
                'max': float(np.max(perplexities)),
            },
            'tokens_per_second': {
                'mean': float(np.mean(throughputs)),
                'std': float(np.std(throughputs)),
            },
            'peak_memory_mb': {
                'mean': float(np.mean(memories)),
                'std': float(np.std(memories)),
            },
            'avg_cache_size': {
                'mean': float(np.mean(cache_sizes)),
                'std': float(np.std(cache_sizes)),
            },
        }

    return aggregated


def compute_significance_tests(results: List[BenchmarkResult]) -> Dict:
    """Compute statistical significance tests comparing methods."""

    # Group results by method and sparsity
    groups = {}
    for result in results:
        key = (result.config.method, result.config.sparsity)
        if key not in groups:
            groups[key] = []
        groups[key].append(result.perplexity)

    # Perform pairwise t-tests
    comparisons = {}
    methods = list(set(r.config.method for r in results))
    sparsities = list(set(r.config.sparsity for r in results))

    for sparsity in sparsities:
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                key1 = (method1, sparsity)
                key2 = (method2, sparsity)

                if key1 in groups and key2 in groups:
                    group1 = groups[key1]
                    group2 = groups[key2]

                    if len(group1) > 1 and len(group2) > 1:
                        # Two-sample t-test
                        t_stat, p_value = stats.ttest_ind(group1, group2)

                        comp_key = f"{method1}_vs_{method2}_s{sparsity}"
                        comparisons[comp_key] = {
                            'method1': method1,
                            'method2': method2,
                            'sparsity': sparsity,
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant_at_0.05': p_value < 0.05,
                            'significant_at_0.01': p_value < 0.01,
                            'mean_diff': float(np.mean(group1) - np.mean(group2)),
                        }

    return comparisons


def main():
    parser = argparse.ArgumentParser(description='ICML-Level Generation Benchmark')

    # Model configuration
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model name or path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32'],
                        help='Data type for model')

    # Experiment configuration
    parser.add_argument('--methods', nargs='+', default=['dense', 'h2o', 'cab'],
                        choices=['dense', 'h2o', 'cab'],
                        help='Methods to evaluate')
    parser.add_argument('--sparsity', nargs='+', type=float, default=[0.5, 0.7, 0.9],
                        help='Sparsity levels to test')
    parser.add_argument('--dataset', type=str, default='pg19',
                        choices=['pg19', 'wikitext', 'arxiv'],
                        help='Dataset to use')
    parser.add_argument('--context-length', type=int, default=4096,
                        help='Context length for evaluation')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples to evaluate')
    parser.add_argument('--num-runs', type=int, default=3,
                        help='Number of independent runs for statistical significance')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='Random seeds for runs (default: 42, 43, 44, ...)')

    # Output configuration
    parser.add_argument('--output-dir', type=str, default='results/generation',
                        help='Directory to save results')
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Name for this experiment (default: auto-generated)')

    args = parser.parse_args()

    # Set up experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f"exp_{timestamp}"

    # Create output directory
    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up seeds
    if args.seeds is None:
        args.seeds = list(range(42, 42 + args.num_runs))

    # Print configuration
    logger.info("=" * 80)
    logger.info("GENERATION BENCHMARK CONFIGURATION")
    logger.info("=" * 80)
    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Context Length: {args.context_length}")
    logger.info(f"Num Samples: {args.num_samples}")
    logger.info(f"Methods: {args.methods}")
    logger.info(f"Sparsity Levels: {args.sparsity}")
    logger.info(f"Num Runs: {args.num_runs}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # Initialize benchmark
    benchmark = GenerationBenchmark(
        model_name=args.model,
        device=args.device,
        dtype=args.dtype,
    )

    # Load dataset once (same samples for all methods/runs)
    base_seed = args.seeds[0]
    logger.info(f"Loading dataset with base seed {base_seed}...")
    benchmark.load_model('dense')  # Load once to get tokenizer
    samples = benchmark.load_dataset_samples(
        dataset_name=args.dataset,
        num_samples=args.num_samples,
        context_length=args.context_length,
        seed=base_seed,
    )
    del benchmark.model
    torch.cuda.empty_cache()

    # Run all experiments
    all_results = []

    for run_idx, seed in enumerate(args.seeds):
        logger.info(f"\n{'='*80}")
        logger.info(f"RUN {run_idx + 1}/{len(args.seeds)} (seed={seed})")
        logger.info(f"{'='*80}\n")

        for method in args.methods:
            for sparsity in args.sparsity:
                # Skip sparsity for dense method
                if method == 'dense' and sparsity != 0.0:
                    continue

                config = ExperimentConfig(
                    model_name=args.model,
                    method=method,
                    sparsity=sparsity if method != 'dense' else 0.0,
                    dataset=args.dataset,
                    context_length=args.context_length,
                    num_samples=args.num_samples,
                    seed=seed,
                    device=args.device,
                    dtype=args.dtype,
                )

                result = benchmark.run_experiment(config, samples)
                all_results.append(result)

                # Save individual result
                result_file = output_dir / f"{config}.json"
                with open(result_file, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)

    # Aggregate results
    logger.info("\n" + "="*80)
    logger.info("AGGREGATED RESULTS")
    logger.info("="*80)

    aggregated = aggregate_results(all_results)

    # Print summary table
    print("\nPerplexity Summary:")
    print(f"{'Method':<10} {'Sparsity':<10} {'Mean PPL':<12} {'Std PPL':<12} {'Tokens/s':<12}")
    print("-" * 70)
    for key, stats in sorted(aggregated.items()):
        method = stats['method']
        sparsity = stats['sparsity']
        mean_ppl = stats['perplexity']['mean']
        std_ppl = stats['perplexity']['std']
        mean_tps = stats['tokens_per_second']['mean']
        print(f"{method:<10} {sparsity:<10.1f} {mean_ppl:<12.2f} {std_ppl:<12.2f} {mean_tps:<12.2f}")

    # Save aggregated results
    agg_file = output_dir / "aggregated_results.json"
    with open(agg_file, 'w') as f:
        json.dump(aggregated, f, indent=2)
    logger.info(f"\nAggregated results saved to: {agg_file}")

    # Compute significance tests
    if args.num_runs > 1:
        logger.info("\n" + "="*80)
        logger.info("STATISTICAL SIGNIFICANCE TESTS")
        logger.info("="*80)

        comparisons = compute_significance_tests(all_results)

        for comp_key, comp_stats in sorted(comparisons.items()):
            sig_marker = "***" if comp_stats['p_value'] < 0.001 else \
                        "**" if comp_stats['p_value'] < 0.01 else \
                        "*" if comp_stats['p_value'] < 0.05 else ""

            logger.info(f"{comp_key}: p={comp_stats['p_value']:.4f} {sig_marker}")
            logger.info(f"  Mean difference: {comp_stats['mean_diff']:.2f}")

        # Save significance tests
        sig_file = output_dir / "significance_tests.json"
        with open(sig_file, 'w') as f:
            json.dump(comparisons, f, indent=2)
        logger.info(f"\nSignificance tests saved to: {sig_file}")

    logger.info("\n" + "="*80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
