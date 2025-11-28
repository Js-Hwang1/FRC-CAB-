#!/usr/bin/env python3
"""
LongBench QA Benchmark Driver (ICML 2025)

Comprehensive CLI for running long-context QA benchmarks.

USAGE:
------
# Quick test with defaults
python driver.py --quick-test

# Run specific experiment
python driver.py --datasets narrativeqa qasper --methods cab h2o --sparsity 0.9 0.95

# Run full benchmark suite
python driver.py --preset full

# Run with config file
python driver.py --config configs/icml_benchmark.yaml

# Sparsity sweep
python driver.py --sweep --datasets narrativeqa --methods cab

# Compare all methods
python driver.py --compare-all --datasets narrativeqa --sparsity 0.9

PRESETS:
--------
- quick:    Small test (10 samples, 1 dataset, 1 method)
- standard: Medium test (100 samples, 3 datasets, 4 methods)
- full:     Complete ICML benchmark (all datasets, all methods, all sparsity levels)
- ablation: Ablation studies (sparsity sweeps, lambda sweeps)

OUTPUT:
-------
Results are saved to experiments/longbench_qa/results/<experiment_name>/
- <experiment>_results.json: Full results with per-sample data
- <experiment>_summary.txt: Human-readable summary
- <experiment>_comparison.png: Visualization (if matplotlib available)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.longbench_qa.config import (
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    MethodConfig,
    BenchmarkConfig,
    ALL_DATASETS,
    METHOD_CONFIGS,
    LONGBENCH_DATASETS,
    SCROLLS_DATASETS,
    INFINITEBENCH_DATASETS,
    ZEROSCROLLS_DATASETS,
    create_full_benchmark,
)
from experiments.longbench_qa.runner import (
    BenchmarkRunner,
    quick_evaluate,
    evaluate_sparsity_sweep,
    compare_methods_on_dataset,
)


# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS = {
    'quick': {
        'description': 'Quick test with minimal samples',
        'datasets': ['narrativeqa'],
        'methods': ['dense', 'cab'],
        'sparsity_levels': [0.9],
        'max_samples': 10,
    },
    'standard': {
        'description': 'Standard benchmark with moderate samples',
        'datasets': ['narrativeqa', 'qasper', 'hotpotqa'],
        'methods': ['dense', 'h2o', 'cab', 'streaming_llm'],
        'sparsity_levels': [0.9, 0.95],
        'max_samples': 100,
    },
    'full': {
        'description': 'Full ICML benchmark (all datasets, methods, sparsity levels)',
        'datasets': list(LONGBENCH_DATASETS.keys())[:8],  # Main LongBench QA tasks
        'methods': ['dense', 'h2o', 'cab', 'cab', 'streaming_llm', 'local_strided', 'random'],
        'sparsity_levels': [0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
        'max_samples': None,  # Use all samples
    },
    'longbench_qa': {
        'description': 'LongBench QA tasks only',
        'datasets': ['narrativeqa', 'qasper', 'multifieldqa_en', 'hotpotqa', '2wikimqa', 'musique'],
        'methods': ['dense', 'h2o', 'cab', 'streaming_llm'],
        'sparsity_levels': [0.9, 0.95],
        'max_samples': 200,
    },
    'longbench_sum': {
        'description': 'LongBench summarization tasks',
        'datasets': ['gov_report', 'qmsum', 'multi_news'],
        'methods': ['dense', 'h2o', 'cab'],
        'sparsity_levels': [0.9, 0.95],
        'max_samples': 100,
    },
    'scrolls': {
        'description': 'SCROLLS benchmark',
        'datasets': ['quality', 'qasper_scrolls', 'narrativeqa_scrolls', 'summ_screen_fd'],
        'methods': ['dense', 'h2o', 'cab', 'streaming_llm'],
        'sparsity_levels': [0.9, 0.95],
        'max_samples': 100,
    },
    'infinitebench': {
        'description': 'InfiniteBench extreme long-context (128K+)',
        'datasets': ['passkey', 'number_string', 'kv_retrieval'],
        'methods': ['dense', 'h2o', 'cab'],
        'sparsity_levels': [0.95, 0.99],
        'max_samples': 50,
    },
    'zeroscrolls': {
        'description': 'ZeroSCROLLS zero-shot benchmark',
        'datasets': ['quality_zero', 'qasper_zero', 'narrativeqa_zero'],
        'methods': ['dense', 'h2o', 'cab'],
        'sparsity_levels': [0.9],
        'max_samples': 100,
    },
    'ablation_sparsity': {
        'description': 'Sparsity ablation study',
        'datasets': ['narrativeqa', 'qasper'],
        'methods': ['cab'],
        'sparsity_levels': [0.0, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.99],
        'max_samples': 100,
    },
    'ablation_ratio': {
        'description': 'Magnitude ratio ablation for CAB V4',
        'datasets': ['narrativeqa'],
        'methods': ['cab'],  # Will be run with different magnitude_ratio values
        'sparsity_levels': [0.9],
        'max_samples': 100,
        'magnitude_ratios': [0.0, 0.25, 0.5, 0.75, 1.0],  # Custom parameter
    },
}


# =============================================================================
# CLI Argument Parser
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser."""
    parser = argparse.ArgumentParser(
        description='LongBench QA Benchmark Driver for CAB-Attention (ICML 2025)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Experiment identification
    parser.add_argument(
        '--name', '-n',
        type=str,
        default=None,
        help='Experiment name (default: auto-generated)',
    )
    parser.add_argument(
        '--description', '-d',
        type=str,
        default='',
        help='Experiment description',
    )
    
    # Preset configurations
    parser.add_argument(
        '--preset', '-p',
        type=str,
        choices=list(PRESETS.keys()),
        help='Use a preset configuration',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to YAML/JSON config file',
    )
    
    # Quick modes
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test (same as --preset quick)',
    )
    parser.add_argument(
        '--sweep',
        action='store_true',
        help='Run sparsity sweep',
    )
    parser.add_argument(
        '--compare-all',
        action='store_true',
        help='Compare all methods on specified datasets',
    )
    
    # Datasets
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        default=None,
        help='Datasets to evaluate (space-separated)',
    )
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='List all available datasets and exit',
    )
    
    # Methods
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=None,
        help='Methods to evaluate (space-separated)',
    )
    parser.add_argument(
        '--list-methods',
        action='store_true',
        help='List all available methods and exit',
    )
    
    # Sparsity settings
    parser.add_argument(
        '--sparsity',
        type=float,
        nargs='+',
        default=None,
        help='Sparsity levels to evaluate (space-separated)',
    )
    
    # CAB-specific settings
    parser.add_argument(
        '--magnitude-ratio',
        type=float,
        default=0.5,
        help='CAB V4 magnitude ratio (0=pure FRC, 1=pure magnitude)',
    )
    parser.add_argument(
        '--lambda-redundancy',
        type=float,
        default=0.3,
        help='FRC lambda redundancy parameter',
    )
    parser.add_argument(
        '--block-size',
        type=int,
        default=64,
        help='Block size for block-sparse attention',
    )
    
    # Model settings
    parser.add_argument(
        '--model',
        type=str,
        default='meta-llama/Llama-2-7b-hf',
        help='HuggingFace model name',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=32768,  # Support full long contexts (~30K tokens for NarrativeQA)
        help='Maximum context length in tokens',
    )
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=256,
        help='Maximum tokens to generate',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        choices=['float16', 'bfloat16', 'float32'],
        default='float16',
        help='Model dtype',
    )
    
    # Evaluation settings
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum samples per dataset (None = use all)',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for evaluation',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility',
    )
    
    # Output settings
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='results',
        help='Output directory for results',
    )
    parser.add_argument(
        '--save-predictions',
        action='store_true',
        default=True,
        help='Save per-sample predictions',
    )
    parser.add_argument(
        '--save-attention',
        action='store_true',
        default=False,
        help='Save attention patterns (memory-intensive)',
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='count',
        default=0,
        help='Increase verbosity (-v, -vv, -vvv)',
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output',
    )
    
    # W&B integration
    parser.add_argument(
        '--wandb',
        action='store_true',
        help='Enable Weights & Biases logging',
    )
    parser.add_argument(
        '--wandb-project',
        type=str,
        default='cab-attention-benchmark',
        help='W&B project name',
    )
    
    # Misc
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print configuration and exit without running',
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from a previous experiment directory',
    )
    
    return parser


# =============================================================================
# Configuration Building
# =============================================================================

def build_config(args: argparse.Namespace) -> ExperimentConfig:
    """Build experiment configuration from CLI arguments."""
    
    # Start with defaults
    datasets = ['narrativeqa']
    methods = ['dense', 'cab']
    sparsity_levels = [0.9]
    max_samples = 100
    
    # Apply preset if specified
    if args.preset or args.quick_test:
        preset_name = args.preset or 'quick'
        preset = PRESETS[preset_name]
        datasets = preset['datasets']
        methods = preset['methods']
        sparsity_levels = preset['sparsity_levels']
        max_samples = preset.get('max_samples')
    
    # Apply config file if specified
    if args.config:
        config_path = Path(args.config)
        if config_path.suffix == '.yaml':
            import yaml
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
        else:
            with open(config_path) as f:
                config_data = json.load(f)
        
        datasets = config_data.get('datasets', datasets)
        methods = config_data.get('methods', methods)
        sparsity_levels = config_data.get('sparsity_levels', sparsity_levels)
        max_samples = config_data.get('max_samples', max_samples)
    
    # Override with CLI arguments
    if args.datasets:
        datasets = args.datasets
    if args.methods:
        methods = args.methods
    if args.sparsity:
        sparsity_levels = args.sparsity
    if args.max_samples:
        max_samples = args.max_samples
    
    # Handle special modes
    if args.compare_all:
        methods = list(METHOD_CONFIGS.keys())
    
    if args.sweep:
        sparsity_levels = [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
    
    # Generate experiment name
    if args.name:
        name = args.name
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        name = f"exp_{timestamp}"
    
    # Build model config
    model_config = ModelConfig(
        name=args.model,
        torch_dtype=args.dtype,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )
    
    # Build method configs with custom parameters
    method_configs = {}
    for method_name in methods:
        if method_name in METHOD_CONFIGS:
            config = METHOD_CONFIGS[method_name]
            # Apply custom CAB parameters
            if 'cab' in method_name:
                from dataclasses import replace
                config = replace(
                    config,
                    magnitude_ratio=args.magnitude_ratio,
                    lambda_redundancy=args.lambda_redundancy,
                    block_size=args.block_size,
                )
            method_configs[method_name] = config
    
    # Build dataset configs
    dataset_configs = {}
    for ds_name in datasets:
        if ds_name in ALL_DATASETS:
            config = ALL_DATASETS[ds_name]
            if max_samples:
                from dataclasses import replace
                config = replace(config, max_samples=max_samples)
            dataset_configs[ds_name] = config
    
    # Build experiment config
    return ExperimentConfig(
        name=name,
        description=args.description or f"Benchmark: {', '.join(datasets[:3])}...",
        seed=args.seed,
        model=model_config,
        datasets=datasets,
        dataset_configs=dataset_configs,
        methods=methods,
        method_configs=method_configs,
        sparsity_levels=sparsity_levels,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        save_attention_patterns=args.save_attention,
        wandb_project=args.wandb_project if args.wandb else None,
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def list_datasets():
    """Print all available datasets."""
    print("\n" + "=" * 70)
    print("AVAILABLE DATASETS")
    print("=" * 70)
    
    for family_name, family_datasets in [
        ("LongBench", LONGBENCH_DATASETS),
        ("SCROLLS", SCROLLS_DATASETS),
        ("InfiniteBench", INFINITEBENCH_DATASETS),
        ("ZeroSCROLLS", ZEROSCROLLS_DATASETS),
    ]:
        print(f"\n{family_name}")
        print("-" * 40)
        for name, config in family_datasets.items():
            metrics = [m.value for m in config.metrics[:2]]
            print(f"  {name:25} | {config.task_type.value:15} | {metrics}")
    
    print()


def list_methods():
    """Print all available methods."""
    print("\n" + "=" * 70)
    print("AVAILABLE METHODS")
    print("=" * 70)
    
    for name, config in METHOD_CONFIGS.items():
        desc = {
            'dense': 'Full attention (oracle upper bound)',
            'h2o': 'Heavy-Hitter Oracle - magnitude-based selection',
            'cab': 'Pure FRC - topology-based selection',
            'cab': 'Hybrid - magnitude + FRC (RECOMMENDED)',
            'streaming_llm': 'Attention sinks + recent window',
            'local_strided': 'Local window + strided patterns',
            'random': 'Random selection baseline',
        }.get(name, 'No description')
        
        print(f"  {name:15} | {desc}")
    
    print()


def print_config(config: ExperimentConfig):
    """Print experiment configuration."""
    print("\n" + "=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)
    
    print(f"\nName: {config.name}")
    print(f"Description: {config.description}")
    print(f"\nModel: {config.model.name}")
    print(f"Max Length: {config.model.max_length}")
    print(f"Dtype: {config.model.torch_dtype}")
    
    print(f"\nDatasets ({len(config.datasets)}):")
    for ds in config.datasets:
        max_samples = config.dataset_configs.get(ds, DatasetConfig(name=ds, family='longbench', task_type='qa')).max_samples
        print(f"  - {ds}" + (f" (max {max_samples} samples)" if max_samples else ""))
    
    print(f"\nMethods ({len(config.methods)}):")
    for method in config.methods:
        print(f"  - {method}")
    
    print(f"\nSparsity Levels: {config.sparsity_levels}")
    print(f"\nOutput Directory: {config.output_dir}")
    
    # Estimate total evaluations
    total_evals = len(config.datasets) * len(config.methods) * len(config.sparsity_levels)
    print(f"\nTotal Evaluations: {total_evals}")
    
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Set up logging
    import logging
    if args.quiet:
        logging.basicConfig(level=logging.ERROR)
    elif args.verbose >= 2:
        logging.basicConfig(level=logging.DEBUG)
    elif args.verbose >= 1:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    # Handle list commands
    if args.list_datasets:
        list_datasets()
        return 0
    
    if args.list_methods:
        list_methods()
        return 0
    
    # Build configuration
    config = build_config(args)
    
    # Print configuration
    if not args.quiet:
        print_config(config)
    
    # Dry run - just print config and exit
    if args.dry_run:
        print("Dry run complete. Use without --dry-run to execute.")
        return 0
    
    # Confirm if running large experiment
    total_evals = len(config.datasets) * len(config.methods) * len(config.sparsity_levels)
    if total_evals > 20 and not args.quiet:
        print(f"\nThis will run {total_evals} evaluations. Continue? [y/N] ", end='')
        response = input().strip().lower()
        if response != 'y':
            print("Aborted.")
            return 0
    
    # Run benchmark
    try:
        runner = BenchmarkRunner(
            config=config,
            output_dir=Path(config.output_dir) / config.name,
        )
        
        start_time = time.time()
        result = runner.run()
        elapsed = time.time() - start_time
        
        if not args.quiet:
            print("\n" + "=" * 70)
            print("EXPERIMENT COMPLETE")
            print("=" * 70)
            print(f"\nTotal Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
            print(f"Results saved to: {config.output_dir}/{config.name}/")
            
            # Print summary
            print("\nSummary:")
            print("-" * 40)
            for key, method_result in result.method_results.items():
                primary_metric = list(method_result.metrics.keys())[0]
                score = method_result.metrics[primary_metric]['mean']
                print(f"  {key}: {primary_metric}={score:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

