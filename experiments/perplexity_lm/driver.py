#!/usr/bin/env python3
"""
CLI Driver for Perplexity Benchmark (TODO 1.3)

Usage:
    # Quick test
    python -m experiments.perplexity_lm.driver --quick-test
    
    # Full ICML benchmark
    python -m experiments.perplexity_lm.driver --config configs/icml_full.yaml
    
    # Custom configuration
    python -m experiments.perplexity_lm.driver \
        --datasets wikitext-103 c4 \
        --methods dense h2o cab \
        --model meta-llama/Llama-2-7b-hf \
        --sparsity-levels 0.0 0.9 0.95 \
        --context-lengths 512 1024 2048 4096
"""

import argparse
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.perplexity_lm.config import (
    ExperimentConfig, ModelConfig, BenchmarkConfig,
    ContextLengthSweepConfig, SparsitySweepConfig,
    create_quick_test, create_icml_benchmark,
    DATASET_CONFIGS, METHOD_CONFIGS,
)
from experiments.perplexity_lm.runner import (
    run_benchmark, PerplexityBenchmarkRunner,
    generate_summary_table, analyze_context_scaling, analyze_sparsity_tradeoff,
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Perplexity Benchmark for CAB-Attention (ICML 2025)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with TinyLlama
  python driver.py --quick-test
  
  # WikiText-103 with multiple methods
  python driver.py --datasets wikitext-103 --methods dense h2o cab
  
  # Full ICML benchmark
  python driver.py --icml-full
  
  # Load from config file
  python driver.py --config configs/my_experiment.yaml
        """
    )
    
    # Preset configurations
    preset_group = parser.add_argument_group("Preset Configurations")
    preset_group.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with TinyLlama on WikiText-2",
    )
    preset_group.add_argument(
        "--icml-full",
        action="store_true",
        help="Run full ICML benchmark (all datasets, all methods)",
    )
    preset_group.add_argument(
        "--config",
        type=str,
        help="Path to YAML or JSON configuration file",
    )
    
    # Dataset configuration
    data_group = parser.add_argument_group("Dataset Configuration")
    data_group.add_argument(
        "--datasets",
        nargs="+",
        default=["wikitext-103"],
        choices=list(DATASET_CONFIGS.keys()),
        help="Datasets to evaluate (default: wikitext-103)",
    )
    data_group.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for debugging)",
    )
    
    # Method configuration
    method_group = parser.add_argument_group("Method Configuration")
    method_group.add_argument(
        "--methods",
        nargs="+",
        default=["dense", "cab"],
        choices=list(METHOD_CONFIGS.keys()),
        help="Attention methods to compare (default: dense cab)",
    )
    
    # Model configuration
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name or path",
    )
    model_group.add_argument(
        "--max-length",
        type=int,
        default=4096,
        help="Maximum context length (default: 4096)",
    )
    model_group.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: float16)",
    )
    
    # Sweep configurations
    sweep_group = parser.add_argument_group("Sweep Configurations")
    sweep_group.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[512, 1024, 2048, 4096],
        help="Context lengths for scaling analysis",
    )
    sweep_group.add_argument(
        "--sparsity-levels",
        nargs="+",
        type=float,
        default=[0.0, 0.9],
        help="Sparsity levels for trade-off curves",
    )
    sweep_group.add_argument(
        "--no-context-sweep",
        action="store_true",
        help="Disable context length sweep",
    )
    sweep_group.add_argument(
        "--no-sparsity-sweep",
        action="store_true",
        help="Disable sparsity sweep",
    )
    
    # Output configuration
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument(
        "--output-dir",
        type=str,
        default="results/perplexity",
        help="Output directory for results",
    )
    output_group.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not specified)",
    )
    output_group.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    return parser.parse_args()


def create_config_from_args(args) -> ExperimentConfig:
    """Create experiment configuration from command-line arguments."""
    
    # Determine experiment name
    name = args.name
    if name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"perplexity_{timestamp}"
    
    # Create model config
    model_config = ModelConfig(
        name=args.model,
        max_length=args.max_length,
        torch_dtype=args.dtype,
    )
    
    # Create sweep configs
    ctx_sweep = ContextLengthSweepConfig(
        enabled=not args.no_context_sweep,
        context_lengths=args.context_lengths,
    )
    
    sparsity_sweep = SparsitySweepConfig(
        enabled=not args.no_sparsity_sweep,
        sparsity_levels=args.sparsity_levels,
    )
    
    # Create experiment config
    config = ExperimentConfig(
        name=name,
        description=f"Perplexity benchmark: {args.datasets}",
        model=model_config,
        datasets=args.datasets,
        methods=args.methods,
        context_length_sweep=ctx_sweep,
        sparsity_sweep=sparsity_sweep,
        output_dir=args.output_dir,
        log_level="DEBUG" if args.verbose else "INFO",
    )
    
    return config


def print_config_summary(config: ExperimentConfig) -> None:
    """Print configuration summary."""
    print("\n" + "="*70)
    print("EXPERIMENT CONFIGURATION")
    print("="*70)
    print(f"Name: {config.name}")
    print(f"Description: {config.description}")
    print(f"Model: {config.model.name}")
    print(f"Max Length: {config.model.max_length}")
    print(f"Dtype: {config.model.torch_dtype}")
    print(f"\nDatasets ({len(config.datasets)}):")
    for ds in config.datasets:
        print(f"  - {ds}")
    print(f"\nMethods ({len(config.methods)}):")
    for m in config.methods:
        print(f"  - {m}")
    print(f"\nContext Length Sweep: {config.context_length_sweep.enabled}")
    if config.context_length_sweep.enabled:
        print(f"  Lengths: {config.context_length_sweep.context_lengths}")
    print(f"\nSparsity Sweep: {config.sparsity_sweep.enabled}")
    if config.sparsity_sweep.enabled:
        print(f"  Levels: {config.sparsity_sweep.sparsity_levels}")
    print(f"\nOutput Directory: {config.output_dir}")
    print("="*70 + "\n")


def main():
    args = parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Determine configuration
    if args.quick_test:
        logger.info("Running quick test configuration...")
        config = create_quick_test()
    
    elif args.icml_full:
        logger.info("Running full ICML benchmark...")
        benchmark_config = create_icml_benchmark()
        # Run all experiments
        for exp_config in benchmark_config.experiments:
            print_config_summary(exp_config)
            results = run_benchmark(exp_config)
        return
    
    elif args.config:
        logger.info(f"Loading configuration from: {args.config}")
        config_path = Path(args.config)
        
        if config_path.suffix in ['.yaml', '.yml']:
            benchmark_config = BenchmarkConfig.from_yaml(args.config)
        elif config_path.suffix == '.json':
            benchmark_config = BenchmarkConfig.from_json(args.config)
        else:
            raise ValueError(f"Unknown config format: {config_path.suffix}")
        
        # Run all experiments in config
        for exp_config in benchmark_config.experiments:
            print_config_summary(exp_config)
            results = run_benchmark(exp_config)
        return
    
    else:
        # Create from command-line arguments
        config = create_config_from_args(args)
    
    # Print configuration
    print_config_summary(config)
    
    # Confirm before running (if not quick test)
    if not args.quick_test:
        response = input("Continue with this configuration? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Run benchmark
    results = run_benchmark(config)
    
    # Print summary
    print("\n" + "="*70)
    print("BENCHMARK COMPLETE")
    print("="*70)
    
    # Generate and print summary table
    print("\n## Summary Table\n")
    print(generate_summary_table(results))
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()

