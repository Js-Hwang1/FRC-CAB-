# Generation Benchmark: ICML-Level Rigorous Evaluation

Comprehensive benchmarking suite for KV cache eviction methods (Dense, H2O, CAB) on language generation tasks.

## Features

- **Metrics**: Perplexity, throughput (tokens/sec), peak memory, cache size
- **Statistical Rigor**: Multiple runs with different seeds, significance testing
- **Datasets**: PG-19, WikiText-103, arXiv
- **Methods**: Dense (baseline), H2O (importance-based), CAB (three-component)
- **Flash Attention**: Custom Triton kernels for O(N) memory cumulative scoring

## Quick Start

### Basic Usage

```bash
# Test on WikiText with 3 methods at 90% sparsity
python -m experiments.generation_benchmark.driver \
  --model Qwen/Qwen2.5-7B-Instruct \
  --methods dense h2o cab \
  --sparsity 0.9 \
  --dataset wikitext \
  --context-length 2048 \
  --num-samples 20 \
  --num-runs 3

# Full evaluation (3 sparsity levels, 50 samples, 3 runs)
python -m experiments.generation_benchmark.driver \
  --model Qwen/Qwen2.5-7B-Instruct \
  --methods dense h2o cab \
  --sparsity 0.5 0.7 0.9 \
  --dataset pg19 \
  --context-length 4096 \
  --num-samples 50 \
  --num-runs 3 \
  --output-dir results/generation
```

### Quick Test (Small Scale)

```bash
# Fast sanity check: 5 samples, 1 run
python -m experiments.generation_benchmark.driver \
  --methods dense cab \
  --sparsity 0.9 \
  --dataset wikitext \
  --context-length 1024 \
  --num-samples 5 \
  --num-runs 1
```

## Command-Line Arguments

### Model Configuration
- `--model`: Model name or path (default: `Qwen/Qwen2.5-7B-Instruct`)
- `--device`: Device (default: `cuda`)
- `--dtype`: Data type (`float16` or `float32`, default: `float16`)

### Experiment Configuration
- `--methods`: Methods to evaluate (choices: `dense`, `h2o`, `cab`)
- `--sparsity`: Sparsity levels (e.g., `0.5 0.7 0.9` for 50%, 70%, 90%)
- `--dataset`: Dataset (choices: `pg19`, `wikitext`, `arxiv`)
- `--context-length`: Context length (default: `4096`)
- `--num-samples`: Number of samples (default: `50`)
- `--num-runs`: Number of independent runs for significance testing (default: `3`)
- `--seeds`: Custom seeds for runs (default: `42, 43, 44, ...`)

### Output Configuration
- `--output-dir`: Output directory (default: `results/generation`)
- `--experiment-name`: Name for experiment (default: auto-generated timestamp)

## Output Structure

```
results/generation/<experiment_name>/
├── aggregated_results.json       # Summary statistics
├── significance_tests.json        # Pairwise t-tests
├── dense_s0.0_ctx4096_seed42.json
├── h2o_s0.9_ctx4096_seed42.json
├── cab_s0.9_ctx4096_seed42.json
└── ...
```

### Output Metrics

Each result contains:
- **Perplexity**: exp(average log-likelihood) - lower is better
- **Tokens per Second**: Throughput - higher is better
- **Peak Memory (MB)**: GPU memory - lower is better
- **Avg Cache Size**: Average KV cache length - smaller = more compression

## Experimental Design

### Why Perplexity?

Perplexity is the standard metric for evaluating language models on generation:
- **Definition**: exp(cross-entropy loss) on held-out text
- **Interpretation**: Lower perplexity = better language modeling
- **Advantage**: Captures both quality (correct predictions) and calibration

### Statistical Significance

- Multiple runs with different seeds ensure reproducibility
- Pairwise t-tests between methods with p-values
- Significance levels: *** (p<0.001), ** (p<0.01), * (p<0.05)

### Sparsity Levels

- **0.5 (50%)**: Moderate compression, should preserve quality
- **0.7 (70%)**: Aggressive compression, may show quality degradation
- **0.9 (90%)**: Extreme compression, tests limits of methods

## Example Output

```
Perplexity Summary:
Method     Sparsity   Mean PPL     Std PPL      Tokens/s
----------------------------------------------------------------------
dense      0.0        12.45        0.12         45.23
h2o        0.9        13.87        0.23         78.45
cab        0.9        13.21        0.18         76.92

Statistical Significance Tests:
h2o_vs_cab_s0.9: p=0.0234 *
  Mean difference: 0.66
dense_vs_h2o_s0.9: p=0.0001 ***
  Mean difference: -1.42
```

## Datasets

### PG-19
- **Source**: Project Gutenberg books
- **Content**: Long-form literature
- **Length**: Very long (10K+ tokens per book)
- **Use Case**: Tests extreme long-context handling

### WikiText-103
- **Source**: Wikipedia articles
- **Content**: Factual, encyclopedic text
- **Length**: Medium (few thousand tokens)
- **Use Case**: Standard language modeling benchmark

### arXiv
- **Source**: Scientific papers (via RedPajama)
- **Content**: Technical, formal writing
- **Length**: Long (full papers)
- **Use Case**: Domain-specific generation

## Implementation Details

### KV Cache Eviction

#### Dense (Baseline)
- No eviction - keeps full KV cache
- Best quality, highest memory

#### H2O (Heavy-Hitter Oracle)
- Evicts based on cumulative attention scores
- Config: 20% recent + 80% highest importance
- O(N) scoring via Flash Attention

#### CAB (Context-Aware Budget)
- Three-component eviction:
  - 30% recent (local context)
  - 20% bridges (median-importance connectors)
  - 50% important (high cumulative attention)
- O(N) bridge selection via median-based heuristic

### Flash Attention Integration

- Custom Triton kernels compute attention + cumulative scores in single pass
- O(N) memory vs O(N²) for naive attention matrices
- Scores accumulated across layers and heads
- Reset between samples to prevent contamination

## Troubleshooting

### Out of Memory

Reduce:
- `--context-length`: Try 2048 or 1024 instead of 4096
- `--num-samples`: Use fewer samples
- Use `--dtype float16` instead of float32

### Slow Evaluation

- Use fewer samples: `--num-samples 10`
- Shorter context: `--context-length 1024`
- Single run: `--num-runs 1`
- Smaller dataset: `--dataset wikitext` (faster than PG-19)

### Dataset Download Issues

Datasets are auto-downloaded via HuggingFace. Ensure:
- Internet connection
- Sufficient disk space (~10GB for PG-19)
- HuggingFace cache directory writable

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@software{frc_generation_benchmark,
  title={ICML-Level Generation Benchmark for KV Cache Eviction},
  author={FRC Team},
  year={2025},
  url={https://github.com/Js-Hwang1/FRC-CAB-}
}
```

## Related Work

- **H2O**: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference" (2023)
- **CAB**: Context-Aware Budget allocation for KV cache compression
- **Flash Attention**: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention" (2022)
