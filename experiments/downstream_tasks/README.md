# Downstream Tasks Benchmark (TODO 1.4)

This module implements comprehensive benchmarks for real-world downstream tasks, comparing CAB Attention against other sparse attention methods.

## Tasks Covered

### 1. Document Summarization (1.4.1)
- **CNN/DailyMail**: Extractive summarization of news articles
- **XSum**: Abstractive one-sentence summaries
- **Multi-News**: Multi-document summarization

### 2. Open-Domain QA (1.4.2)
- **Natural Questions**: Wikipedia-based QA
- **TriviaQA**: Long-context passage QA
- **SQuAD v2**: Reading comprehension with unanswerable questions

### 3. Dialogue State Tracking (1.4.3)
- **MultiWOZ**: Task-oriented dialogue state tracking

### 4. Code Understanding (1.4.4)
- **CodeXGLUE Code Summarization**: Generate docstrings from code
- **CodeXGLUE Code Completion**: Complete code lines
- **HumanEval**: Code generation with test cases

## Methods Compared

| Method | Description | Key Property |
|--------|-------------|--------------|
| Dense | Full attention | Oracle upper bound |
| H2O | Heavy Hitter Oracle | Magnitude-based selection |
| CAB V3 | Curvature-Aware Block | Pure FRC-based |
| CAB V4 | Hybrid CAB | 50% magnitude + 50% FRC |
| StreamingLLM | Attention sinks | First 4 + recent tokens |
| Local+Strided | Fixed patterns | Local window + global stride |
| Random | Random selection | Baseline |

## Quick Start

```bash
# Quick test (10 samples)
python -m experiments.downstream_tasks.driver --quick-test

# Summarization benchmark
python -m experiments.downstream_tasks.driver --preset summarization

# QA benchmark
python -m experiments.downstream_tasks.driver --preset qa

# Custom run
python -m experiments.downstream_tasks.driver \
    --model Qwen/Qwen2.5-7B-Instruct \
    --datasets cnn_dailymail xsum \
    --methods dense h2o cab_v4 \
    --sparsity 0.5 0.7 0.9 \
    --max-samples 100
```

## Fair Comparison (Apple-to-Apple)

All methods use the **same underlying mechanism**:
1. Forward pass to populate KV cache
2. Compute importance scores (method-specific)
3. Keep top-k tokens based on importance
4. Continue generation with pruned cache

The only difference is the importance function:
- **H2O**: `importance = ||key||_2` (L2 norm)
- **CAB V4**: `importance = α * magnitude + (1-α) * uniqueness`
- **StreamingLLM**: `importance[0:4] = ∞, importance[i] = i` (sinks + recency)

This ensures the comparison is fair and isolates the effect of the importance function.

## Configuration

### Presets

- `quick_test`: 10 samples, 3 methods, 1 sparsity
- `summarization`: CNN/DM, XSum, Multi-News
- `qa`: NQ, TriviaQA, SQuAD v2
- `code`: CodeXGLUE tasks
- `full_downstream`: All tasks and methods

### Custom YAML Config

```yaml
name: "my_experiment"
description: "Custom benchmark"

datasets:
  - cnn_dailymail
  - natural_questions

methods:
  - dense
  - h2o
  - cab_v4

sparsity_levels:
  - 0.7
  - 0.9

model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  max_length: 8192
  max_new_tokens: 256
  torch_dtype: "float16"

max_samples: 100
```

## Metrics

### Summarization
- ROUGE-1, ROUGE-2, ROUGE-L (F1 scores)

### QA
- F1 (token overlap)
- Exact Match

### Dialogue
- Joint Goal Accuracy (all slots correct)
- Slot Accuracy (per-slot accuracy)

### Code
- BLEU, CodeBLEU
- pass@k (for HumanEval)

## Output

Results are saved to `results/downstream_tasks/`:
- `{experiment_name}_results.json`: Full results with per-sample data
- `{experiment_name}_summary.txt`: Human-readable summary
- `intermediate_*.json`: Intermediate checkpoints

## Requirements

```bash
pip install transformers datasets torch tqdm numpy
```

Optional for better metrics:
```bash
pip install rouge-score bert-score
```

