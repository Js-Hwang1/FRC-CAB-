"""
Configuration Classes for Language Model Perplexity Benchmark (TODO 1.3)

ICML Publication-Quality Benchmark for:
- WikiText-103 (standard benchmark)
- C4 (diverse web text)
- PG-19 (long book sequences)
- Perplexity vs context length scaling
- Perplexity vs sparsity trade-off curves

Designed for apple-to-apple comparison of sparse attention methods.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Literal
from enum import Enum
import json
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# =============================================================================
# Enums for Type Safety
# =============================================================================

class PerplexityDataset(str, Enum):
    """Supported perplexity evaluation datasets."""
    WIKITEXT_103 = "wikitext-103"       # Standard benchmark (small, clean)
    WIKITEXT_2 = "wikitext-2"           # Smaller version for quick tests
    C4 = "c4"                           # Diverse web text (large)
    PG19 = "pg19"                       # Long book sequences
    PILE = "pile"                       # The Pile (diverse)
    OPENWEBTEXT = "openwebtext"         # OpenWebText2


class MethodName(str, Enum):
    """Supported sparse attention methods."""
    DENSE = "dense"                     # Full attention (oracle)
    H2O = "h2o"                         # Heavy-Hitter Oracle
    CAB_V3 = "cab_v3"                   # Pure FRC (legacy)
    CAB_V4 = "cab_v4"                   # Hybrid magnitude + FRC (legacy)
    CAB_V5 = "cab_v5"                   # Three-component: local + bridge + importance (NEW)
    STREAMING_LLM = "streaming_llm"     # Attention sinks + recent
    LOCAL_STRIDED = "local_strided"     # Local window + strided
    RANDOM = "random"                   # Random selection baseline


# =============================================================================
# Dataset Configurations
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a perplexity evaluation dataset."""
    
    name: PerplexityDataset
    
    # Data loading
    hf_path: str                           # HuggingFace dataset path
    hf_subset: Optional[str] = None        # Dataset subset
    split: str = "test"                    # Evaluation split
    text_column: str = "text"              # Column containing text
    
    # Sampling
    max_samples: Optional[int] = None      # Limit samples (None = all)
    min_length: int = 512                  # Minimum sequence length
    max_length: int = 4096                 # Maximum sequence length (for chunking)
    
    # Stride for perplexity calculation (overlap between chunks)
    stride: Optional[int] = None           # None = use max_length (no overlap)
    
    # Cache settings
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = PerplexityDataset(self.name)
        if self.stride is None:
            self.stride = self.max_length


# Pre-defined dataset configurations
DATASET_CONFIGS = {
    "wikitext-103": DatasetConfig(
        name=PerplexityDataset.WIKITEXT_103,
        hf_path="wikitext",
        hf_subset="wikitext-103-raw-v1",
        split="test",
        text_column="text",
        min_length=512,
        max_length=4096,
    ),
    "wikitext-2": DatasetConfig(
        name=PerplexityDataset.WIKITEXT_2,
        hf_path="wikitext",
        hf_subset="wikitext-2-raw-v1",
        split="test",
        text_column="text",
        min_length=128,
        max_length=2048,
    ),
    "c4": DatasetConfig(
        name=PerplexityDataset.C4,
        hf_path="allenai/c4",
        hf_subset="en",
        split="validation",  # C4 test is huge
        text_column="text",
        max_samples=1000,   # Sample for efficiency
        min_length=512,
        max_length=4096,
    ),
    "pg19": DatasetConfig(
        name=PerplexityDataset.PG19,
        hf_path="pg19",
        split="test",
        text_column="text",
        max_samples=100,    # Books are very long
        min_length=1024,
        max_length=16384,   # Long context for books
    ),
}


# =============================================================================
# Method Configuration
# =============================================================================

@dataclass
class MethodConfig:
    """Configuration for a sparse attention method."""
    
    name: MethodName
    
    # Sparsity settings
    sparsity: float = 0.9                 # Fraction to PRUNE (0.9 = keep 10%)
    
    # Block settings
    block_size: int = 64
    
    # CAB-specific settings
    lambda_redundancy: float = 0.3
    magnitude_ratio: float = 0.5          # CAB V4: 0=pure FRC, 1=pure magnitude
    formula: str = "additive"
    normalization: str = "minmax"
    
    # StreamingLLM settings
    num_sink_tokens: int = 4
    window_size: int = 512
    
    # Local+Strided settings
    local_window: int = 256
    stride: int = 64
    
    # General settings
    causal: bool = True
    keep_diagonal: bool = True
    
    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = MethodName(self.name)


# Pre-defined method configurations
METHOD_CONFIGS = {
    "dense": MethodConfig(name=MethodName.DENSE, sparsity=0.0),
    "h2o": MethodConfig(name=MethodName.H2O, sparsity=0.9),
    "cab_v3": MethodConfig(name=MethodName.CAB_V3, sparsity=0.9, magnitude_ratio=0.0),
    "cab_v4": MethodConfig(name=MethodName.CAB_V4, sparsity=0.9, magnitude_ratio=0.5),
    "cab_v5": MethodConfig(name=MethodName.CAB_V5, sparsity=0.9),  # Three-component eviction
    "streaming_llm": MethodConfig(name=MethodName.STREAMING_LLM, sparsity=0.9),
    "local_strided": MethodConfig(name=MethodName.LOCAL_STRIDED, sparsity=0.9),
    "random": MethodConfig(name=MethodName.RANDOM, sparsity=0.9),
}


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for the base LLM model."""
    
    name: str = "meta-llama/Llama-2-7b-hf"
    revision: Optional[str] = None
    torch_dtype: str = "float16"
    device_map: str = "auto"
    
    # Tokenizer settings
    max_length: int = 4096
    
    # Memory optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    
    # For perplexity computation
    add_bos: bool = True                   # Add BOS token
    add_eos: bool = False                  # Add EOS token


# =============================================================================
# Experiment Configuration
# =============================================================================

@dataclass
class ContextLengthSweepConfig:
    """Configuration for context length scaling analysis."""
    
    enabled: bool = True
    context_lengths: List[int] = field(default_factory=lambda: [
        512, 1024, 2048, 4096, 8192, 16384
    ])
    fixed_sparsity: float = 0.9


@dataclass
class SparsitySweepConfig:
    """Configuration for sparsity trade-off curves."""
    
    enabled: bool = True
    sparsity_levels: List[float] = field(default_factory=lambda: [
        0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99
    ])
    fixed_context_length: int = 4096


@dataclass
class ExperimentConfig:
    """Full perplexity experiment configuration."""
    
    # Experiment metadata
    name: str = "perplexity_benchmark"
    description: str = "Language Model Perplexity Benchmark for CAB-Attention"
    seed: int = 42
    
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Datasets to evaluate
    datasets: List[str] = field(default_factory=lambda: ["wikitext-103"])
    dataset_configs: Dict[str, DatasetConfig] = field(default_factory=dict)
    
    # Methods to compare
    methods: List[str] = field(default_factory=lambda: ["dense", "h2o", "cab_v5"])
    method_configs: Dict[str, MethodConfig] = field(default_factory=dict)
    
    # Sweep configurations
    context_length_sweep: ContextLengthSweepConfig = field(default_factory=ContextLengthSweepConfig)
    sparsity_sweep: SparsitySweepConfig = field(default_factory=SparsitySweepConfig)
    
    # Evaluation settings
    batch_size: int = 1                    # Usually 1 for perplexity eval
    num_workers: int = 4
    
    # Output settings
    output_dir: str = "results/perplexity"
    save_per_sample: bool = False          # Save per-sample perplexities
    
    # Logging
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    
    def __post_init__(self):
        # Load default dataset configs
        for ds_name in self.datasets:
            if ds_name not in self.dataset_configs:
                if ds_name in DATASET_CONFIGS:
                    self.dataset_configs[ds_name] = DATASET_CONFIGS[ds_name]
                else:
                    raise ValueError(f"Unknown dataset: {ds_name}")
        
        # Load default method configs
        for method_name in self.methods:
            if method_name not in self.method_configs:
                if method_name in METHOD_CONFIGS:
                    self.method_configs[method_name] = METHOD_CONFIGS[method_name]
                else:
                    raise ValueError(f"Unknown method: {method_name}")


@dataclass
class BenchmarkConfig:
    """Top-level benchmark configuration."""
    
    experiments: List[ExperimentConfig] = field(default_factory=list)
    
    # Global settings
    global_seed: int = 42
    global_output_dir: str = "results/perplexity"
    
    # Hardware
    num_gpus: int = 1
    mixed_precision: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required: pip install pyyaml")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls._from_dict(data)
    
    @classmethod
    def from_json(cls, path: str) -> "BenchmarkConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        """Create config from dictionary."""
        experiments = []
        for exp_data in data.get('experiments', []):
            model_config = ModelConfig(**exp_data.pop('model', {}))
            
            # Handle sweep configs
            ctx_sweep = exp_data.pop('context_length_sweep', {})
            ctx_sweep_config = ContextLengthSweepConfig(**ctx_sweep)
            
            sparsity_sweep = exp_data.pop('sparsity_sweep', {})
            sparsity_sweep_config = SparsitySweepConfig(**sparsity_sweep)
            
            exp_config = ExperimentConfig(
                model=model_config,
                context_length_sweep=ctx_sweep_config,
                sparsity_sweep=sparsity_sweep_config,
                **exp_data
            )
            experiments.append(exp_config)
        
        return cls(
            experiments=experiments,
            global_seed=data.get('global_seed', 42),
            global_output_dir=data.get('global_output_dir', 'results/perplexity'),
            num_gpus=data.get('num_gpus', 1),
            mixed_precision=data.get('mixed_precision', True),
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required: pip install pyyaml")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self._to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._to_dict(), f, indent=2)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        result = asdict(self)
        # Convert enums to strings
        for exp in result.get('experiments', []):
            for ds in exp.get('dataset_configs', {}).values():
                if 'name' in ds and hasattr(ds['name'], 'value'):
                    ds['name'] = ds['name'].value
            for method in exp.get('method_configs', {}).values():
                if 'name' in method and hasattr(method['name'], 'value'):
                    method['name'] = method['name'].value
        return result


# =============================================================================
# Convenience Functions
# =============================================================================

def create_icml_benchmark() -> BenchmarkConfig:
    """Create full ICML benchmark configuration (TODO 1.3)."""
    
    # Experiment 1: Standard perplexity on WikiText-103
    exp_wikitext = ExperimentConfig(
        name="wikitext103_perplexity",
        description="Standard perplexity benchmark on WikiText-103",
        datasets=["wikitext-103"],
        methods=["dense", "h2o", "cab_v4", "cab_v3", "streaming_llm", "local_strided", "random"],
        context_length_sweep=ContextLengthSweepConfig(
            enabled=True,
            context_lengths=[512, 1024, 2048, 4096],
        ),
        sparsity_sweep=SparsitySweepConfig(
            enabled=True,
            sparsity_levels=[0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
        ),
    )
    
    # Experiment 2: C4 (diverse web text)
    exp_c4 = ExperimentConfig(
        name="c4_perplexity",
        description="Perplexity on diverse web text (C4)",
        datasets=["c4"],
        methods=["dense", "h2o", "cab_v4", "streaming_llm"],
        context_length_sweep=ContextLengthSweepConfig(
            enabled=True,
            context_lengths=[512, 1024, 2048, 4096],
        ),
        sparsity_sweep=SparsitySweepConfig(
            enabled=True,
            sparsity_levels=[0.0, 0.8, 0.9, 0.95],
        ),
    )
    
    # Experiment 3: PG-19 (long books)
    exp_pg19 = ExperimentConfig(
        name="pg19_perplexity",
        description="Long-context perplexity on books (PG-19)",
        datasets=["pg19"],
        methods=["dense", "h2o", "cab_v4", "streaming_llm"],
        context_length_sweep=ContextLengthSweepConfig(
            enabled=True,
            context_lengths=[1024, 2048, 4096, 8192, 16384],
        ),
        sparsity_sweep=SparsitySweepConfig(
            enabled=True,
            sparsity_levels=[0.0, 0.9, 0.95, 0.99],
        ),
    )
    
    return BenchmarkConfig(
        experiments=[exp_wikitext, exp_c4, exp_pg19],
        global_output_dir="results/perplexity/icml",
    )


def create_quick_test() -> ExperimentConfig:
    """Create quick test configuration."""
    return ExperimentConfig(
        name="quick_perplexity_test",
        description="Quick perplexity test for debugging",
        datasets=["wikitext-2"],
        methods=["dense", "h2o", "cab_v4"],  # Added h2o for comparison
        model=ModelConfig(
            name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            max_length=1024,
        ),
        context_length_sweep=ContextLengthSweepConfig(
            enabled=True,
            context_lengths=[512, 1024],
        ),
        sparsity_sweep=SparsitySweepConfig(
            enabled=True,
            sparsity_levels=[0.0, 0.5, 0.7, 0.9],  # More gradual sparsity
        ),
    )


def get_dataset_config(name: str) -> DatasetConfig:
    """Get dataset configuration by name."""
    if name in DATASET_CONFIGS:
        return DATASET_CONFIGS[name]
    raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_CONFIGS.keys())}")


def get_method_config(name: str) -> MethodConfig:
    """Get method configuration by name."""
    if name in METHOD_CONFIGS:
        return METHOD_CONFIGS[name]
    raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_CONFIGS.keys())}")

