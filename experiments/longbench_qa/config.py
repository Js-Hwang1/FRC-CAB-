"""
Configuration Classes for LongBench QA Benchmark

Provides comprehensive, type-safe configuration for:
- Datasets (LongBench, SCROLLS, InfiniteBench, ZeroSCROLLS)
- Methods (Dense, H2O, CAB V4, StreamingLLM, etc.)
- Experiments (sparsity sweeps, ablations, etc.)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
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

class DatasetFamily(str, Enum):
    """Supported dataset families."""
    LONGBENCH = "longbench"
    SCROLLS = "scrolls"
    INFINITEBENCH = "infinitebench"
    ZEROSCROLLS = "zeroscrolls"


class TaskType(str, Enum):
    """Task types for evaluation."""
    QA = "qa"                           # Question Answering
    SUMMARIZATION = "summarization"     # Summarization
    FEW_SHOT = "few_shot"               # Few-shot learning
    CODE = "code"                       # Code completion
    RETRIEVAL = "retrieval"             # Information retrieval
    MULTIPLE_CHOICE = "multiple_choice" # Multiple choice QA


class MethodName(str, Enum):
    """Supported sparse attention methods."""
    DENSE = "dense"                     # Full attention (oracle)
    H2O = "h2o"                         # Heavy-Hitter Oracle (arxiv:2306.14048)
    CAB = "cab"                         # Curvature-Aware Block-Sparse (Ours)
    STREAMING_LLM = "streaming_llm"     # Attention sinks + recent (arxiv:2309.17453)
    LOCAL_STRIDED = "local_strided"     # Sparse Transformer (arxiv:1904.10509)
    RANDOM = "random"                   # Random selection baseline


class MetricName(str, Enum):
    """Evaluation metrics."""
    F1 = "f1"
    EXACT_MATCH = "exact_match"
    ROUGE_1 = "rouge_1"
    ROUGE_2 = "rouge_2"
    ROUGE_L = "rouge_l"
    ACCURACY = "accuracy"
    BLEU = "bleu"
    RETRIEVAL_ACCURACY = "retrieval_accuracy"


# =============================================================================
# Dataset Configurations
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    
    name: str
    family: DatasetFamily
    task_type: TaskType
    
    # Data paths and loading
    subset: Optional[str] = None          # Dataset subset (e.g., "en" for MultiFieldQA)
    split: str = "test"                   # train/validation/test
    max_samples: Optional[int] = None     # Limit samples for quick testing
    
    # Context configuration
    max_context_length: int = 4096        # Maximum context length in tokens
    min_context_length: int = 512         # Minimum context length
    
    # Task-specific
    num_few_shot: int = 0                 # Number of few-shot examples
    
    # Metrics for this dataset
    metrics: List[MetricName] = field(default_factory=lambda: [MetricName.F1])
    
    # Cache settings
    cache_dir: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.family, str):
            self.family = DatasetFamily(self.family)
        if isinstance(self.task_type, str):
            self.task_type = TaskType(self.task_type)
        if self.metrics and isinstance(self.metrics[0], str):
            self.metrics = [MetricName(m) for m in self.metrics]


# Pre-defined dataset configurations for all TODO 1.2 tasks
LONGBENCH_DATASETS = {
    # QA Tasks
    "narrativeqa": DatasetConfig(
        name="narrativeqa",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.QA,
        max_context_length=16384,
        metrics=[MetricName.F1, MetricName.ROUGE_L],
    ),
    "qasper": DatasetConfig(
        name="qasper",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.QA,
        max_context_length=8192,
        metrics=[MetricName.F1],
    ),
    "multifieldqa_en": DatasetConfig(
        name="multifieldqa_en",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.QA,
        subset="en",
        max_context_length=8192,
        metrics=[MetricName.F1],
    ),
    "multifieldqa_zh": DatasetConfig(
        name="multifieldqa_zh",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.QA,
        subset="zh",
        max_context_length=8192,
        metrics=[MetricName.F1],
    ),
    "hotpotqa": DatasetConfig(
        name="hotpotqa",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.QA,
        max_context_length=8192,
        metrics=[MetricName.F1, MetricName.EXACT_MATCH],
    ),
    "2wikimqa": DatasetConfig(
        name="2wikimqa",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.QA,
        max_context_length=8192,
        metrics=[MetricName.F1],
    ),
    "musique": DatasetConfig(
        name="musique",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.QA,
        max_context_length=8192,
        metrics=[MetricName.F1],
    ),
    "dureader": DatasetConfig(
        name="dureader",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.QA,
        max_context_length=8192,
        metrics=[MetricName.F1, MetricName.ROUGE_L],
    ),
    
    # Summarization Tasks
    "gov_report": DatasetConfig(
        name="gov_report",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=16384,
        metrics=[MetricName.ROUGE_1, MetricName.ROUGE_2, MetricName.ROUGE_L],
    ),
    "qmsum": DatasetConfig(
        name="qmsum",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=16384,
        metrics=[MetricName.ROUGE_1, MetricName.ROUGE_2, MetricName.ROUGE_L],
    ),
    "multi_news": DatasetConfig(
        name="multi_news",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=8192,
        metrics=[MetricName.ROUGE_1, MetricName.ROUGE_2, MetricName.ROUGE_L],
    ),
    "vcsum": DatasetConfig(
        name="vcsum",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=8192,
        metrics=[MetricName.ROUGE_L],
    ),
    
    # Few-shot Learning
    "trec": DatasetConfig(
        name="trec",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.FEW_SHOT,
        max_context_length=4096,
        num_few_shot=5,
        metrics=[MetricName.ACCURACY],
    ),
    "triviaqa": DatasetConfig(
        name="triviaqa",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.FEW_SHOT,
        max_context_length=8192,
        num_few_shot=5,
        metrics=[MetricName.F1],
    ),
    "samsum": DatasetConfig(
        name="samsum",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.FEW_SHOT,
        max_context_length=4096,
        num_few_shot=5,
        metrics=[MetricName.ROUGE_1, MetricName.ROUGE_L],
    ),
    "lsht": DatasetConfig(
        name="lsht",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.FEW_SHOT,
        max_context_length=8192,
        num_few_shot=5,
        metrics=[MetricName.ACCURACY],
    ),
    
    # Code Completion
    "lcc": DatasetConfig(
        name="lcc",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.CODE,
        max_context_length=8192,
        metrics=[MetricName.EXACT_MATCH],
    ),
    "repobench-p": DatasetConfig(
        name="repobench-p",
        family=DatasetFamily.LONGBENCH,
        task_type=TaskType.CODE,
        max_context_length=8192,
        metrics=[MetricName.EXACT_MATCH],
    ),
}


SCROLLS_DATASETS = {
    "quality": DatasetConfig(
        name="quality",
        family=DatasetFamily.SCROLLS,
        task_type=TaskType.MULTIPLE_CHOICE,
        max_context_length=8192,
        metrics=[MetricName.ACCURACY],
    ),
    "qasper_scrolls": DatasetConfig(
        name="qasper",
        family=DatasetFamily.SCROLLS,
        task_type=TaskType.QA,
        max_context_length=8192,
        metrics=[MetricName.F1],
    ),
    "narrativeqa_scrolls": DatasetConfig(
        name="narrativeqa",
        family=DatasetFamily.SCROLLS,
        task_type=TaskType.QA,
        max_context_length=16384,
        metrics=[MetricName.F1, MetricName.ROUGE_L],
    ),
    "summ_screen_fd": DatasetConfig(
        name="summ_screen_fd",
        family=DatasetFamily.SCROLLS,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=8192,
        metrics=[MetricName.ROUGE_1, MetricName.ROUGE_L],
    ),
    "gov_report_scrolls": DatasetConfig(
        name="gov_report",
        family=DatasetFamily.SCROLLS,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=16384,
        metrics=[MetricName.ROUGE_1, MetricName.ROUGE_2, MetricName.ROUGE_L],
    ),
    "contract_nli": DatasetConfig(
        name="contract_nli",
        family=DatasetFamily.SCROLLS,
        task_type=TaskType.MULTIPLE_CHOICE,
        max_context_length=8192,
        metrics=[MetricName.ACCURACY],
    ),
}


INFINITEBENCH_DATASETS = {
    "passkey": DatasetConfig(
        name="passkey",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.RETRIEVAL,
        max_context_length=131072,  # 128K
        metrics=[MetricName.RETRIEVAL_ACCURACY],
    ),
    "number_string": DatasetConfig(
        name="number_string",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.RETRIEVAL,
        max_context_length=131072,
        metrics=[MetricName.RETRIEVAL_ACCURACY],
    ),
    "kv_retrieval": DatasetConfig(
        name="kv_retrieval",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.RETRIEVAL,
        max_context_length=131072,
        metrics=[MetricName.RETRIEVAL_ACCURACY],
    ),
    "longbook_qa_eng": DatasetConfig(
        name="longbook_qa_eng",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.QA,
        max_context_length=131072,
        metrics=[MetricName.F1],
    ),
    "longbook_sum_eng": DatasetConfig(
        name="longbook_sum_eng",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=131072,
        metrics=[MetricName.ROUGE_L],
    ),
    "longbook_choice_eng": DatasetConfig(
        name="longbook_choice_eng",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.MULTIPLE_CHOICE,
        max_context_length=131072,
        metrics=[MetricName.ACCURACY],
    ),
    "longdialogue_qa_eng": DatasetConfig(
        name="longdialogue_qa_eng",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.QA,
        max_context_length=131072,
        metrics=[MetricName.F1],
    ),
    "math_find": DatasetConfig(
        name="math_find",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.RETRIEVAL,
        max_context_length=131072,
        metrics=[MetricName.RETRIEVAL_ACCURACY],
    ),
    "math_calc": DatasetConfig(
        name="math_calc",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.QA,
        max_context_length=131072,
        metrics=[MetricName.EXACT_MATCH],
    ),
    "code_run": DatasetConfig(
        name="code_run",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.CODE,
        max_context_length=131072,
        metrics=[MetricName.EXACT_MATCH],
    ),
    "code_debug": DatasetConfig(
        name="code_debug",
        family=DatasetFamily.INFINITEBENCH,
        task_type=TaskType.CODE,
        max_context_length=131072,
        metrics=[MetricName.ACCURACY],
    ),
}


ZEROSCROLLS_DATASETS = {
    "quality_zero": DatasetConfig(
        name="quality",
        family=DatasetFamily.ZEROSCROLLS,
        task_type=TaskType.MULTIPLE_CHOICE,
        max_context_length=8192,
        num_few_shot=0,
        metrics=[MetricName.ACCURACY],
    ),
    "qasper_zero": DatasetConfig(
        name="qasper",
        family=DatasetFamily.ZEROSCROLLS,
        task_type=TaskType.QA,
        max_context_length=8192,
        num_few_shot=0,
        metrics=[MetricName.F1],
    ),
    "narrativeqa_zero": DatasetConfig(
        name="narrativeqa",
        family=DatasetFamily.ZEROSCROLLS,
        task_type=TaskType.QA,
        max_context_length=16384,
        num_few_shot=0,
        metrics=[MetricName.F1, MetricName.ROUGE_L],
    ),
    "summ_screen_fd_zero": DatasetConfig(
        name="summ_screen_fd",
        family=DatasetFamily.ZEROSCROLLS,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=8192,
        num_few_shot=0,
        metrics=[MetricName.ROUGE_L],
    ),
    "gov_report_zero": DatasetConfig(
        name="gov_report",
        family=DatasetFamily.ZEROSCROLLS,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=16384,
        num_few_shot=0,
        metrics=[MetricName.ROUGE_L],
    ),
    "squality": DatasetConfig(
        name="squality",
        family=DatasetFamily.ZEROSCROLLS,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=8192,
        num_few_shot=0,
        metrics=[MetricName.ROUGE_L],
    ),
    "book_sum_sort": DatasetConfig(
        name="book_sum_sort",
        family=DatasetFamily.ZEROSCROLLS,
        task_type=TaskType.SUMMARIZATION,
        max_context_length=16384,
        num_few_shot=0,
        metrics=[MetricName.ACCURACY],
    ),
}


# Combined registry
ALL_DATASETS = {
    **LONGBENCH_DATASETS,
    **SCROLLS_DATASETS,
    **INFINITEBENCH_DATASETS,
    **ZEROSCROLLS_DATASETS,
}


# =============================================================================
# Method Configurations
# =============================================================================

@dataclass
class MethodConfig:
    """Configuration for a sparse attention method."""
    
    name: MethodName
    
    # Sparsity settings
    sparsity: float = 0.9                 # Fraction to PRUNE (0.9 = keep 10%)
    
    # Block settings (for block-sparse methods)
    block_size: int = 64                  # Tokens per block
    
    # CAB-specific settings
    lambda_redundancy: float = 0.3        # FRC redundancy penalty
    magnitude_ratio: float = 0.5          # CAB V4: fraction for magnitude selection
    formula: str = "additive"             # FRC formula
    normalization: str = "minmax"         # Affinity normalization
    
    # StreamingLLM settings
    num_sink_tokens: int = 4              # Number of attention sink tokens
    window_size: int = 512                # Recent token window
    
    # Local+Strided settings
    local_window: int = 256               # Local attention window
    stride: int = 64                      # Strided attention stride
    
    # General settings
    causal: bool = True                   # Causal masking
    keep_diagonal: bool = True            # Keep diagonal blocks
    
    def __post_init__(self):
        if isinstance(self.name, str):
            self.name = MethodName(self.name)


# Pre-defined method configurations
METHOD_CONFIGS = {
    "dense": MethodConfig(
        name=MethodName.DENSE,
        sparsity=0.0,
    ),
    "h2o": MethodConfig(
        name=MethodName.H2O,
        sparsity=0.9,
    ),
    "cab": MethodConfig(
        name=MethodName.CAB,
        sparsity=0.9,
        # Three-component eviction: local + bridge + importance
    ),
    "streaming_llm": MethodConfig(
        name=MethodName.STREAMING_LLM,
        sparsity=0.9,
        num_sink_tokens=4,
        window_size=512,
    ),
    "local_strided": MethodConfig(
        name=MethodName.LOCAL_STRIDED,
        sparsity=0.9,
        local_window=256,
        stride=64,
    ),
    "random": MethodConfig(
        name=MethodName.RANDOM,
        sparsity=0.9,
    ),
}


# =============================================================================
# Experiment Configuration
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
    truncation_side: str = "left"         # Truncate from left for long contexts
    
    # Generation settings
    max_new_tokens: int = 256
    temperature: float = 0.0              # Greedy decoding
    do_sample: bool = False
    
    # Memory optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False


@dataclass
class ExperimentConfig:
    """Full experiment configuration."""
    
    # Experiment metadata
    name: str = "longbench_qa_benchmark"
    description: str = "Long-context QA benchmark for CAB-Attention"
    seed: int = 42
    
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Datasets to evaluate
    datasets: List[str] = field(default_factory=lambda: ["narrativeqa", "qasper"])
    dataset_configs: Dict[str, DatasetConfig] = field(default_factory=dict)
    
    # Methods to compare
    methods: List[str] = field(default_factory=lambda: ["dense", "h2o", "cab"])
    method_configs: Dict[str, MethodConfig] = field(default_factory=dict)
    
    # Sparsity sweep (for ablations)
    sparsity_levels: List[float] = field(default_factory=lambda: [0.9])
    
    # Evaluation settings
    batch_size: int = 1
    num_workers: int = 4
    
    # Output settings
    output_dir: str = "results"
    save_predictions: bool = True
    save_attention_patterns: bool = False  # Memory-intensive
    
    # Logging
    log_level: str = "INFO"
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    def __post_init__(self):
        # Load default dataset configs
        for ds_name in self.datasets:
            if ds_name not in self.dataset_configs:
                if ds_name in ALL_DATASETS:
                    self.dataset_configs[ds_name] = ALL_DATASETS[ds_name]
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
    """Top-level benchmark configuration with multiple experiments."""
    
    experiments: List[ExperimentConfig] = field(default_factory=list)
    
    # Global settings
    global_seed: int = 42
    global_output_dir: str = "results"
    
    # Hardware
    num_gpus: int = 1
    mixed_precision: bool = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "BenchmarkConfig":
        """Load configuration from YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
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
            exp_config = ExperimentConfig(model=model_config, **exp_data)
            experiments.append(exp_config)
        
        return cls(
            experiments=experiments,
            global_seed=data.get('global_seed', 42),
            global_output_dir=data.get('global_output_dir', 'results'),
            num_gpus=data.get('num_gpus', 1),
            mixed_precision=data.get('mixed_precision', True),
        )
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        if not YAML_AVAILABLE:
            raise ImportError("PyYAML required for YAML config files: pip install pyyaml")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self._to_dict(), f, default_flow_style=False)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self._to_dict(), f, indent=2)
    
    def _to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)


# =============================================================================
# Convenience Functions
# =============================================================================

def get_dataset_config(name: str) -> DatasetConfig:
    """Get dataset configuration by name."""
    if name in ALL_DATASETS:
        return ALL_DATASETS[name]
    raise ValueError(f"Unknown dataset: {name}. Available: {list(ALL_DATASETS.keys())}")


def get_method_config(name: str) -> MethodConfig:
    """Get method configuration by name."""
    if name in METHOD_CONFIGS:
        return METHOD_CONFIGS[name]
    raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_CONFIGS.keys())}")


def create_default_experiment(
    datasets: List[str] = None,
    methods: List[str] = None,
    sparsity_levels: List[float] = None,
    model_name: str = "meta-llama/Llama-2-7b-hf",
) -> ExperimentConfig:
    """Create default experiment configuration."""
    return ExperimentConfig(
        datasets=datasets or ["narrativeqa", "qasper"],
        methods=methods or ["dense", "h2o", "cab"],
        sparsity_levels=sparsity_levels or [0.9],
        model=ModelConfig(name=model_name),
    )


def create_full_benchmark() -> BenchmarkConfig:
    """Create full benchmark configuration for ICML paper."""
    
    # Experiment 1: LongBench QA Tasks
    exp_longbench_qa = ExperimentConfig(
        name="longbench_qa",
        description="LongBench QA benchmark",
        datasets=["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"],
        methods=["dense", "h2o", "cab", "streaming_llm", "local_strided", "random"],
        sparsity_levels=[0.5, 0.7, 0.8, 0.9, 0.95, 0.99],
    )
    
    # Experiment 2: LongBench Summarization
    exp_longbench_sum = ExperimentConfig(
        name="longbench_summarization",
        description="LongBench summarization benchmark",
        datasets=["gov_report", "qmsum", "multi_news"],
        methods=["dense", "h2o", "cab", "streaming_llm"],
        sparsity_levels=[0.9, 0.95],
    )
    
    # Experiment 3: SCROLLS
    exp_scrolls = ExperimentConfig(
        name="scrolls",
        description="SCROLLS benchmark",
        datasets=["quality", "qasper_scrolls", "narrativeqa_scrolls", "summ_screen_fd"],
        methods=["dense", "h2o", "cab", "streaming_llm"],
        sparsity_levels=[0.9, 0.95],
    )
    
    # Experiment 4: InfiniteBench (extreme long context)
    exp_infinitebench = ExperimentConfig(
        name="infinitebench",
        description="InfiniteBench extreme long-context benchmark (128K+)",
        datasets=["passkey", "number_string", "kv_retrieval"],
        methods=["dense", "h2o", "cab", "streaming_llm"],
        sparsity_levels=[0.95, 0.99],
    )
    
    # Experiment 5: ZeroSCROLLS
    exp_zeroscrolls = ExperimentConfig(
        name="zeroscrolls",
        description="ZeroSCROLLS zero-shot benchmark",
        datasets=["quality_zero", "qasper_zero", "narrativeqa_zero"],
        methods=["dense", "h2o", "cab"],
        sparsity_levels=[0.9],
    )
    
    return BenchmarkConfig(
        experiments=[
            exp_longbench_qa,
            exp_longbench_sum,
            exp_scrolls,
            exp_infinitebench,
            exp_zeroscrolls,
        ],
        global_output_dir="results/icml_benchmark",
    )

