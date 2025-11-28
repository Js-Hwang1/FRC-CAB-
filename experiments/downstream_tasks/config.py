"""
Configuration for Downstream Tasks Benchmark

Defines all datasets, methods, and experiment settings for TODO 1.4:
- Document Summarization
- Open-Domain QA  
- Dialogue State Tracking
- Code Understanding
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class TaskType(Enum):
    """Types of downstream tasks."""
    SUMMARIZATION = "summarization"
    OPEN_DOMAIN_QA = "open_domain_qa"
    DIALOGUE = "dialogue"
    CODE = "code"


class MetricName(Enum):
    """Available metrics."""
    # Common
    F1 = "f1"
    EXACT_MATCH = "exact_match"
    ACCURACY = "accuracy"
    
    # Summarization
    ROUGE_1 = "rouge_1"
    ROUGE_2 = "rouge_2"
    ROUGE_L = "rouge_l"
    BLEU = "bleu"
    BERTSCORE = "bertscore"
    
    # QA
    RETRIEVAL_ACCURACY = "retrieval_accuracy"
    
    # Dialogue
    JOINT_GOAL_ACCURACY = "joint_goal_accuracy"
    SLOT_ACCURACY = "slot_accuracy"
    
    # Code
    CODE_BLEU = "code_bleu"
    PASS_AT_K = "pass_at_k"


class MethodName(Enum):
    """Sparse attention methods."""
    DENSE = "dense"                     # Full attention (oracle)
    H2O = "h2o"                         # Heavy-Hitter Oracle (arxiv:2306.14048)
    CAB = "cab"                         # Curvature-Aware Block-Sparse (Ours)
    STREAMING_LLM = "streaming_llm"     # Attention sinks + recent (arxiv:2309.17453)
    LOCAL_STRIDED = "local_strided"     # Sparse Transformer (arxiv:1904.10509)
    RANDOM = "random"                   # Random selection baseline


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    task_type: TaskType
    metrics: List[MetricName]
    
    # HuggingFace dataset info
    hf_name: str = ""
    hf_subset: str = ""
    hf_split: str = "test"
    
    # Data fields
    input_field: str = "input"
    context_field: str = "context"
    target_field: str = "target"
    
    # Limits
    max_samples: int = 500
    max_context_length: int = 8192
    
    # Task-specific
    num_few_shot: int = 0
    instruction: str = ""


@dataclass
class MethodConfig:
    """Configuration for a sparse attention method."""
    name: MethodName
    sparsity: float = 0.9
    
    # CAB-specific
    magnitude_ratio: float = 0.5  # 0 = pure FRC, 1 = pure magnitude
    lambda_redundancy: float = 0.3
    block_size: int = 64
    
    # StreamingLLM-specific
    num_sink_tokens: int = 4
    
    # Local+Strided-specific
    local_window_ratio: float = 0.25
    stride: int = 4
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name.value if isinstance(self.name, Enum) else self.name,
            'sparsity': self.sparsity,
            'magnitude_ratio': self.magnitude_ratio,
            'lambda_redundancy': self.lambda_redundancy,
            'block_size': self.block_size,
            'num_sink_tokens': self.num_sink_tokens,
            'local_window_ratio': self.local_window_ratio,
            'stride': self.stride,
        }


@dataclass
class ModelConfig:
    """Configuration for the LLM."""
    name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_length: int = 8192
    max_new_tokens: int = 256
    torch_dtype: str = "float16"
    device_map: str = "auto"
    use_flash_attention: bool = True
    gradient_checkpointing: bool = False
    
    # Generation
    do_sample: bool = False
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 50


@dataclass
class ExperimentConfig:
    """Configuration for a complete experiment."""
    name: str
    description: str = ""
    
    # What to evaluate
    datasets: List[str] = field(default_factory=list)
    methods: List[str] = field(default_factory=list)
    sparsity_levels: List[float] = field(default_factory=lambda: [0.9])
    
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # Output
    output_dir: str = "results/downstream_tasks"
    save_predictions: bool = True
    save_attention: bool = False
    
    # Overrides
    dataset_configs: Dict[str, DatasetConfig] = field(default_factory=dict)
    method_configs: Dict[str, MethodConfig] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and fill in defaults."""
        if not self.datasets:
            self.datasets = ["cnn_dailymail"]
        
        if not self.methods:
            self.methods = ["dense", "h2o", "cab"]
        
        # Validate datasets
        for ds_name in self.datasets:
            if ds_name not in ALL_DATASETS:
                raise ValueError(f"Unknown dataset: {ds_name}")
        
        # Validate methods
        for method_name in self.methods:
            if method_name not in METHOD_CONFIGS:
                raise ValueError(f"Unknown method: {method_name}")
        
        # Fill in missing configs
        for ds_name in self.datasets:
            if ds_name not in self.dataset_configs:
                self.dataset_configs[ds_name] = ALL_DATASETS[ds_name]
        
        for method_name in self.methods:
            if method_name not in self.method_configs:
                self.method_configs[method_name] = METHOD_CONFIGS[method_name]


# =============================================================================
# Dataset Definitions
# =============================================================================

ALL_DATASETS: Dict[str, DatasetConfig] = {
    # =========================================================================
    # SUMMARIZATION (TODO 1.4.1)
    # =========================================================================
    
    # CNN/DailyMail - Extractive Summarization
    "cnn_dailymail": DatasetConfig(
        name="cnn_dailymail",
        task_type=TaskType.SUMMARIZATION,
        metrics=[MetricName.ROUGE_1, MetricName.ROUGE_2, MetricName.ROUGE_L],
        hf_name="cnn_dailymail",
        hf_subset="3.0.0",
        hf_split="test",
        input_field="article",
        target_field="highlights",
        max_samples=500,
        max_context_length=8192,
        instruction="Summarize the following article concisely:",
    ),
    
    # XSum - Abstractive Summarization
    "xsum": DatasetConfig(
        name="xsum",
        task_type=TaskType.SUMMARIZATION,
        metrics=[MetricName.ROUGE_1, MetricName.ROUGE_2, MetricName.ROUGE_L],
        hf_name="xsum",
        hf_split="test",
        input_field="document",
        target_field="summary",
        max_samples=500,
        max_context_length=4096,
        instruction="Write a one-sentence summary of this article:",
    ),
    
    # Multi-News - Multi-document Summarization
    "multi_news": DatasetConfig(
        name="multi_news",
        task_type=TaskType.SUMMARIZATION,
        metrics=[MetricName.ROUGE_1, MetricName.ROUGE_2, MetricName.ROUGE_L],
        hf_name="multi_news",
        hf_split="test",
        input_field="document",
        target_field="summary",
        max_samples=300,
        max_context_length=16384,
        instruction="Summarize these related news articles:",
    ),
    
    # =========================================================================
    # OPEN-DOMAIN QA (TODO 1.4.2)
    # =========================================================================
    
    # Natural Questions - Wikipedia QA
    "natural_questions": DatasetConfig(
        name="natural_questions",
        task_type=TaskType.OPEN_DOMAIN_QA,
        metrics=[MetricName.F1, MetricName.EXACT_MATCH],
        hf_name="natural_questions",
        hf_split="validation",
        input_field="question",
        context_field="document",
        target_field="answer",
        max_samples=500,
        max_context_length=8192,
        instruction="Answer the question based on the Wikipedia article:",
    ),
    
    # TriviaQA - Long Context QA
    "triviaqa": DatasetConfig(
        name="triviaqa",
        task_type=TaskType.OPEN_DOMAIN_QA,
        metrics=[MetricName.F1, MetricName.EXACT_MATCH],
        hf_name="trivia_qa",
        hf_subset="rc",
        hf_split="validation",
        input_field="question",
        context_field="search_results",
        target_field="answer",
        max_samples=500,
        max_context_length=16384,
        instruction="Answer the question using the provided search results:",
    ),
    
    # SQuAD 2.0 - Reading Comprehension
    "squad_v2": DatasetConfig(
        name="squad_v2",
        task_type=TaskType.OPEN_DOMAIN_QA,
        metrics=[MetricName.F1, MetricName.EXACT_MATCH],
        hf_name="squad_v2",
        hf_split="validation",
        input_field="question",
        context_field="context",
        target_field="answers",
        max_samples=500,
        max_context_length=4096,
        instruction="Answer the question based on the context. If unanswerable, say 'unanswerable':",
    ),
    
    # =========================================================================
    # DIALOGUE STATE TRACKING (TODO 1.4.3)
    # =========================================================================
    
    # MultiWOZ 2.4 - Task-Oriented Dialogue
    "multiwoz": DatasetConfig(
        name="multiwoz",
        task_type=TaskType.DIALOGUE,
        metrics=[MetricName.JOINT_GOAL_ACCURACY, MetricName.SLOT_ACCURACY],
        hf_name="multi_woz_v22",
        hf_split="test",
        input_field="turns",
        target_field="dialogue_state",
        max_samples=300,
        max_context_length=4096,
        instruction="Track the dialogue state (slots and values) based on the conversation:",
    ),
    
    # =========================================================================
    # CODE UNDERSTANDING (TODO 1.4.4)
    # =========================================================================
    
    # CodeXGLUE - Code Summarization
    "code_summarization": DatasetConfig(
        name="code_summarization",
        task_type=TaskType.CODE,
        metrics=[MetricName.BLEU, MetricName.ROUGE_L],
        hf_name="code_x_glue_ct_code_to_text",
        hf_subset="python",
        hf_split="test",
        input_field="code",
        target_field="docstring",
        max_samples=500,
        max_context_length=4096,
        instruction="Write a docstring describing what this Python function does:",
    ),
    
    # CodeXGLUE - Code Completion
    "code_completion": DatasetConfig(
        name="code_completion",
        task_type=TaskType.CODE,
        metrics=[MetricName.EXACT_MATCH, MetricName.BLEU],
        hf_name="code_x_glue_cc_code_completion_line",
        hf_subset="python",
        hf_split="test",
        input_field="input",
        target_field="gt",
        max_samples=500,
        max_context_length=4096,
        instruction="Complete the next line of code:",
    ),
    
    # HumanEval - Code Generation
    "humaneval": DatasetConfig(
        name="humaneval",
        task_type=TaskType.CODE,
        metrics=[MetricName.PASS_AT_K],
        hf_name="openai_humaneval",
        hf_split="test",
        input_field="prompt",
        target_field="canonical_solution",
        max_samples=164,  # Full dataset
        max_context_length=2048,
        instruction="",  # Prompt is self-contained
    ),
}


# =============================================================================
# Method Definitions
# =============================================================================

METHOD_CONFIGS: Dict[str, MethodConfig] = {
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
    ),
    
    "streaming_llm": MethodConfig(
        name=MethodName.STREAMING_LLM,
        sparsity=0.9,
        num_sink_tokens=4,
    ),
    
    "local_strided": MethodConfig(
        name=MethodName.LOCAL_STRIDED,
        sparsity=0.9,
        local_window_ratio=0.25,
        stride=4,
    ),
    
    "random": MethodConfig(
        name=MethodName.RANDOM,
        sparsity=0.9,
    ),
}


# =============================================================================
# Preset Configurations
# =============================================================================

EXPERIMENT_PRESETS: Dict[str, Dict[str, Any]] = {
    "quick_test": {
        "datasets": ["cnn_dailymail"],
        "methods": ["dense", "h2o", "cab"],
        "sparsity_levels": [0.9],
        "max_samples": 10,
    },
    
    "summarization": {
        "datasets": ["cnn_dailymail", "xsum", "multi_news"],
        "methods": ["dense", "h2o", "cab", "streaming_llm", "random"],
        "sparsity_levels": [0.5, 0.7, 0.9, 0.95],
    },
    
    "qa": {
        "datasets": ["natural_questions", "triviaqa", "squad_v2"],
        "methods": ["dense", "h2o", "cab", "streaming_llm", "random"],
        "sparsity_levels": [0.5, 0.7, 0.9, 0.95],
    },
    
    "code": {
        "datasets": ["code_summarization", "code_completion", "humaneval"],
        "methods": ["dense", "h2o", "cab", "streaming_llm"],
        "sparsity_levels": [0.5, 0.7, 0.9],
    },
    
    "full_downstream": {
        "datasets": list(ALL_DATASETS.keys()),
        "methods": ["dense", "h2o", "cab", "streaming_llm", "local_strided", "random"],
        "sparsity_levels": [0.5, 0.7, 0.8, 0.9, 0.95],
    },
}

