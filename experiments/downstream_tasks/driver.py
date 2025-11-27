#!/usr/bin/env python3
"""
Driver for Downstream Tasks Benchmark

CLI interface for running TODO 1.4 experiments:
- Document Summarization
- Open-Domain QA
- Dialogue State Tracking
- Code Understanding

Example usage:
    # Quick test
    python -m experiments.downstream_tasks.driver --quick-test
    
    # Summarization benchmark
    python -m experiments.downstream_tasks.driver --preset summarization
    
    # Custom run
    python -m experiments.downstream_tasks.driver \
        --datasets cnn_dailymail xsum \
        --methods dense h2o cab_v4 \
        --sparsity 0.9 \
        --max-samples 100
"""

import argparse
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from .config import (
    ExperimentConfig,
    ModelConfig,
    DatasetConfig,
    MethodConfig,
    ALL_DATASETS,
    METHOD_CONFIGS,
    EXPERIMENT_PRESETS,
)
from .runner import BenchmarkRunner


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Downstream Tasks Benchmark for CAB Attention",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python -m experiments.downstream_tasks.driver --quick-test

  # Full summarization benchmark
  python -m experiments.downstream_tasks.driver --preset summarization

  # Custom configuration
  python -m experiments.downstream_tasks.driver \\
      --model Qwen/Qwen2.5-7B-Instruct \\
      --datasets cnn_dailymail xsum \\
      --methods dense h2o cab_v4 \\
      --sparsity 0.5 0.7 0.9 \\
      --max-samples 100

  # Compare all methods on QA
  python -m experiments.downstream_tasks.driver \\
      --preset qa --compare-all
        """
    )
    
    # Experiment setup
    parser.add_argument('--name', type=str, default=None,
                        help='Experiment name')
    parser.add_argument('--description', type=str, default='',
                        help='Experiment description')
    
    # Presets
    parser.add_argument('--preset', type=str,
                        choices=list(EXPERIMENT_PRESETS.keys()),
                        help='Use a preset configuration')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test (10 samples)')
    parser.add_argument('--compare-all', action='store_true',
                        help='Compare all methods')
    
    # Datasets
    parser.add_argument('--datasets', type=str, nargs='+',
                        help='Datasets to evaluate')
    parser.add_argument('--list-datasets', action='store_true',
                        help='List available datasets')
    
    # Methods
    parser.add_argument('--methods', type=str, nargs='+',
                        help='Methods to evaluate')
    parser.add_argument('--list-methods', action='store_true',
                        help='List available methods')
    
    # Sparsity
    parser.add_argument('--sparsity', type=float, nargs='+',
                        default=[0.9],
                        help='Sparsity levels')
    parser.add_argument('--magnitude-ratio', type=float, default=0.5,
                        help='CAB V4 magnitude ratio')
    
    # Model
    parser.add_argument('--model', type=str,
                        default='Qwen/Qwen2.5-7B-Instruct',
                        help='Model to use')
    parser.add_argument('--max-length', type=int, default=8192,
                        help='Maximum sequence length')
    parser.add_argument('--max-new-tokens', type=int, default=256,
                        help='Maximum new tokens to generate')
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'])
    
    # Evaluation
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum samples per dataset')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output
    parser.add_argument('--output-dir', type=str,
                        default='results/downstream_tasks',
                        help='Output directory')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save individual predictions')
    
    # Logging
    parser.add_argument('--verbose', action='store_true',
                        help='Verbose logging')
    parser.add_argument('--quiet', action='store_true',
                        help='Minimal logging')
    
    return parser.parse_args()


def list_datasets():
    """Print available datasets."""
    print("\n" + "=" * 70)
    print("AVAILABLE DATASETS")
    print("=" * 70)
    
    task_groups = {}
    for name, config in ALL_DATASETS.items():
        task = config.task_type.value
        if task not in task_groups:
            task_groups[task] = []
        task_groups[task].append((name, config))
    
    for task, datasets in task_groups.items():
        print(f"\n{task.upper().replace('_', ' ')}")
        print("-" * 40)
        for name, config in datasets:
            metrics = ', '.join(m.value for m in config.metrics)
            print(f"  {name:25s} | {metrics}")
    
    print()


def list_methods():
    """Print available methods."""
    print("\n" + "=" * 70)
    print("AVAILABLE METHODS")
    print("=" * 70 + "\n")
    
    for name, config in METHOD_CONFIGS.items():
        desc = ""
        if name == "dense":
            desc = "Full attention (oracle upper bound)"
        elif name == "h2o":
            desc = "Heavy Hitter Oracle - magnitude-based"
        elif name == "cab_v3":
            desc = "CAB V3 - pure FRC-based"
        elif name == "cab_v4":
            desc = "CAB V4 - hybrid magnitude + FRC"
        elif name == "streaming_llm":
            desc = "Attention sinks + recency"
        elif name == "local_strided":
            desc = "Local window + strided global"
        elif name == "random":
            desc = "Random baseline"
        
        print(f"  {name:20s} | {desc}")
    
    print()


def build_config(args) -> ExperimentConfig:
    """Build experiment config from args."""
    
    # Start with preset if specified
    if args.preset:
        preset = EXPERIMENT_PRESETS[args.preset].copy()
    elif args.quick_test:
        preset = EXPERIMENT_PRESETS['quick_test'].copy()
    else:
        preset = {}
    
    # Override with CLI args
    datasets = args.datasets or preset.get('datasets', ['cnn_dailymail'])
    methods = args.methods or preset.get('methods', ['dense', 'h2o', 'cab_v4'])
    sparsity_levels = args.sparsity or preset.get('sparsity_levels', [0.9])
    
    if args.compare_all:
        methods = list(METHOD_CONFIGS.keys())
    
    # Build model config
    model_config = ModelConfig(
        name=args.model,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        torch_dtype=args.dtype,
    )
    
    # Handle max_samples override
    dataset_configs = {}
    max_samples = args.max_samples or preset.get('max_samples')
    if max_samples:
        for ds_name in datasets:
            if ds_name in ALL_DATASETS:
                from dataclasses import replace
                config = replace(ALL_DATASETS[ds_name], max_samples=max_samples)
                dataset_configs[ds_name] = config
    
    # Method configs with magnitude_ratio
    method_configs = {}
    for method_name in methods:
        if method_name in METHOD_CONFIGS:
            from dataclasses import replace
            config = replace(
                METHOD_CONFIGS[method_name],
                magnitude_ratio=args.magnitude_ratio
            )
            method_configs[method_name] = config
    
    # Build experiment name
    name = args.name or f"downstream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return ExperimentConfig(
        name=name,
        description=args.description or f"Benchmark: {', '.join(datasets)}",
        datasets=datasets,
        methods=methods,
        sparsity_levels=sparsity_levels,
        model=model_config,
        output_dir=args.output_dir,
        save_predictions=args.save_predictions,
        dataset_configs=dataset_configs,
        method_configs=method_configs,
    )


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
        ds_config = config.dataset_configs.get(ds, ALL_DATASETS.get(ds))
        if ds_config:
            print(f"  - {ds} (max {ds_config.max_samples} samples)")
        else:
            print(f"  - {ds}")
    
    print(f"\nMethods ({len(config.methods)}):")
    for method in config.methods:
        print(f"  - {method}")
    
    print(f"\nSparsity Levels: {config.sparsity_levels}")
    print(f"\nOutput Directory: {config.output_dir}")
    
    # Total evaluations
    total = len(config.datasets) * len(config.methods) * len(config.sparsity_levels)
    print(f"\nTotal Evaluations: {total}")
    print("=" * 70 + "\n")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Handle list commands
    if args.list_datasets:
        list_datasets()
        return 0
    
    if args.list_methods:
        list_methods()
        return 0
    
    # Build config
    try:
        config = build_config(args)
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    
    # Print config
    print_config(config)
    
    # Run benchmark
    try:
        runner = BenchmarkRunner(config, output_dir=args.output_dir)
        result = runner.run()
        
        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        
        for key, method_result in result.method_results.items():
            print(f"\n{method_result.method_name} @ {method_result.sparsity:.0%}:")
            for metric, stats in method_result.metrics.items():
                print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

