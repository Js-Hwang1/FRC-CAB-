"""
LongBench QA Benchmark Suite for CAB-Attention (ICML 2025)

This package provides comprehensive long-context question answering benchmarks
for evaluating sparse attention methods, including:

Datasets:
- LongBench: NarrativeQA, Qasper, MultiFieldQA, GovReport, QMSum, etc.
- SCROLLS: QuALITY, Qasper, NarrativeQA, SummScreenFD
- InfiniteBench: Passkey, Number, KV retrieval (128K+ context)
- ZeroSCROLLS: Zero-shot evaluation

Methods:
- Dense Attention (oracle upper bound)
- H2O (Heavy-Hitter Oracle) - magnitude-based
- CAB V4 (Curvature-Aware Block-Sparse) - our method
- StreamingLLM - attention sinks + recent tokens
- Local + Strided - fixed window patterns
- Random Selection - baseline

Usage:
    python driver.py --config configs/full_benchmark.yaml
    python driver.py --methods cab h2o --datasets longbench --sparsity 0.9
"""

__version__ = "0.1.0"
__author__ = "CAB-Attention Team"

# Core config classes (no heavy dependencies)
from .config import (
    BenchmarkConfig,
    DatasetConfig,
    MethodConfig,
    ExperimentConfig,
)

# Lazy imports for heavy dependencies (torch, transformers)
def get_dataset(*args, **kwargs):
    """Get dataset by name. Lazy import to avoid torch requirement at package import."""
    from .data_loaders import get_dataset as _get_dataset
    return _get_dataset(*args, **kwargs)

def get_method(*args, **kwargs):
    """Get method by name. Lazy import to avoid torch requirement at package import."""
    from .methods import get_method as _get_method
    return _get_method(*args, **kwargs)

def compute_metrics(*args, **kwargs):
    """Compute metrics. Lazy import."""
    from .metrics import compute_metrics as _compute_metrics
    return _compute_metrics(*args, **kwargs)

# Registry classes (lazy loaded)
class DatasetRegistry:
    """Proxy for DatasetRegistry with lazy loading."""
    @staticmethod
    def get_dataset_class(*args, **kwargs):
        from .data_loaders import DatasetRegistry as _DatasetRegistry
        return _DatasetRegistry.get_dataset_class(*args, **kwargs)
    
    @staticmethod
    def list_families():
        from .data_loaders import DatasetRegistry as _DatasetRegistry
        return _DatasetRegistry.list_families()
    
    @staticmethod
    def list_datasets(*args, **kwargs):
        from .data_loaders import DatasetRegistry as _DatasetRegistry
        return _DatasetRegistry.list_datasets(*args, **kwargs)

class MethodRegistry:
    """Proxy for MethodRegistry with lazy loading."""
    @staticmethod
    def get_method_class(*args, **kwargs):
        from .methods import MethodRegistry as _MethodRegistry
        return _MethodRegistry.get_method_class(*args, **kwargs)
    
    @staticmethod
    def list_methods():
        from .methods import MethodRegistry as _MethodRegistry
        return _MethodRegistry.list_methods()
    
    @staticmethod
    def create_method(*args, **kwargs):
        from .methods import MethodRegistry as _MethodRegistry
        return _MethodRegistry.create_method(*args, **kwargs)

class MetricRegistry:
    """Proxy for MetricRegistry with lazy loading."""
    @staticmethod
    def get_metric(*args, **kwargs):
        from .metrics import MetricRegistry as _MetricRegistry
        return _MetricRegistry.get_metric(*args, **kwargs)
    
    @staticmethod
    def list_metrics():
        from .metrics import MetricRegistry as _MetricRegistry
        return _MetricRegistry.list_metrics()

class BenchmarkRunner:
    """Proxy for BenchmarkRunner with lazy loading."""
    def __new__(cls, *args, **kwargs):
        from .runner import BenchmarkRunner as _BenchmarkRunner
        return _BenchmarkRunner(*args, **kwargs)

__all__ = [
    "BenchmarkConfig",
    "DatasetConfig", 
    "MethodConfig",
    "ExperimentConfig",
    "DatasetRegistry",
    "MethodRegistry",
    "MetricRegistry",
    "BenchmarkRunner",
    "get_dataset",
    "get_method",
    "compute_metrics",
]

