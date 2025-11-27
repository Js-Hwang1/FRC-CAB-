"""
Downstream Tasks Benchmark for CAB Attention

This module implements TODO 1.4 from the ICML roadmap:
- Document Summarization (CNN/DailyMail, XSum)
- Open-Domain QA (Natural Questions, TriviaQA)
- Dialogue State Tracking (MultiWOZ)
- Code Understanding (CodeXGLUE)

All benchmarks use apple-to-apple fair comparisons between:
- Dense Attention (oracle upper bound)
- H2O (magnitude-based)
- CAB V4 (hybrid topology + magnitude)
- StreamingLLM (attention sinks + recency)
- Local + Strided (fixed patterns)
- Random (baseline)
"""

from .config import (
    TaskType,
    DatasetConfig,
    MethodConfig,
    ModelConfig,
    ExperimentConfig,
    ALL_DATASETS,
    METHOD_CONFIGS,
)

from .data_loaders import (
    get_dataset,
    BaseBenchmarkDataset,
    BenchmarkSample,
)

from .runner import (
    BenchmarkRunner,
    MethodResult,
    ExperimentResult,
)

__all__ = [
    # Config
    'TaskType',
    'DatasetConfig',
    'MethodConfig', 
    'ModelConfig',
    'ExperimentConfig',
    'ALL_DATASETS',
    'METHOD_CONFIGS',
    # Data
    'get_dataset',
    'BaseBenchmarkDataset',
    'BenchmarkSample',
    # Runner
    'BenchmarkRunner',
    'MethodResult',
    'ExperimentResult',
]

__version__ = '1.0.0'

