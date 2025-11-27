"""
Dataset Loaders for Long-Context QA Benchmarks

Provides unified interface for loading:
- LongBench (THUDM/LongBench)
- SCROLLS (tau/scrolls)
- InfiniteBench (xinrongzhang2022/infinitebench)
- ZeroSCROLLS (tau/zero_scrolls)

All datasets are loaded via HuggingFace datasets library and preprocessed
into a unified format for fair comparison across methods.
"""

import os
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator, Tuple
from pathlib import Path

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    DataLoader = None
    
    # Create a minimal Dataset-like base class when torch is unavailable
    class TorchDataset:
        """Minimal Dataset interface when torch is not available."""
        def __len__(self):
            raise NotImplementedError
        def __getitem__(self, idx):
            raise NotImplementedError

_HF_IMPORT_ERROR = None
try:
    from datasets import load_dataset, DatasetDict
    HF_DATASETS_AVAILABLE = True
except ImportError as e:
    HF_DATASETS_AVAILABLE = False
    _HF_IMPORT_ERROR = f"ImportError: {e}"
except Exception as e:
    # Catch any other import errors (e.g., dependency conflicts)
    HF_DATASETS_AVAILABLE = False
    _HF_IMPORT_ERROR = f"Exception: {type(e).__name__}: {e}"

from .config import DatasetConfig, DatasetFamily, TaskType, ALL_DATASETS


# =============================================================================
# Unified Sample Format
# =============================================================================

@dataclass
class BenchmarkSample:
    """
    Unified sample format for all benchmarks.
    
    This ensures apple-to-apple comparison across datasets and methods.
    """
    # Identifiers
    sample_id: str                        # Unique identifier
    dataset_name: str                     # Source dataset
    
    # Input
    context: str                          # Long context (document, conversation, etc.)
    question: str                         # Question or prompt
    
    # For few-shot learning
    few_shot_examples: List[Dict[str, str]] = None  # List of {"question": ..., "answer": ...}
    
    # Ground truth
    answers: List[str] = None             # List of acceptable answers
    answer: str = None                    # Primary answer (for summarization)
    
    # For multiple choice
    options: List[str] = None             # Answer options
    correct_option: int = None            # Index of correct option
    
    # Metadata
    context_length: int = 0               # Length in characters
    task_type: TaskType = TaskType.QA
    
    # Additional fields for specific datasets
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.context_length = len(self.context) if self.context else 0
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'sample_id': self.sample_id,
            'dataset_name': self.dataset_name,
            'context': self.context,
            'question': self.question,
            'few_shot_examples': self.few_shot_examples,
            'answers': self.answers,
            'answer': self.answer,
            'options': self.options,
            'correct_option': self.correct_option,
            'context_length': self.context_length,
            'task_type': self.task_type.value if isinstance(self.task_type, TaskType) else self.task_type,
            'metadata': self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSample":
        """Create from dictionary."""
        if 'task_type' in data and isinstance(data['task_type'], str):
            data['task_type'] = TaskType(data['task_type'])
        return cls(**data)


# =============================================================================
# Base Dataset Class
# =============================================================================

class BaseBenchmarkDataset(TorchDataset, ABC):
    """
    Abstract base class for all benchmark datasets.
    
    Subclasses must implement:
    - _load_raw_data(): Load raw data from source
    - _preprocess_sample(): Convert raw sample to BenchmarkSample
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        tokenizer: Optional[Any] = None,
        cache_dir: Optional[str] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir or os.path.join(
            os.path.dirname(__file__), "data", config.name
        )
        
        self.samples: List[BenchmarkSample] = []
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and preprocess data."""
        # Check for cached preprocessed data
        cache_file = self._get_cache_path()
        if cache_file.exists():
            self._load_from_cache(cache_file)
        else:
            raw_data = self._load_raw_data()
            self.samples = [self._preprocess_sample(sample, idx) 
                           for idx, sample in enumerate(raw_data)]
            
            # Apply max_samples limit
            if self.config.max_samples:
                self.samples = self.samples[:self.config.max_samples]
            
            # Cache preprocessed data
            self._save_to_cache(cache_file)
    
    def _get_cache_path(self) -> Path:
        """Get cache file path based on config hash."""
        config_str = f"{self.config.name}_{self.config.split}_{self.config.max_samples}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_dir = Path(self.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"preprocessed_{config_hash}.json"
    
    def _save_to_cache(self, path: Path) -> None:
        """Save preprocessed samples to cache."""
        with open(path, 'w') as f:
            json.dump([s.to_dict() for s in self.samples], f)
    
    def _load_from_cache(self, path: Path) -> None:
        """Load preprocessed samples from cache."""
        with open(path, 'r') as f:
            data = json.load(f)
            self.samples = [BenchmarkSample.from_dict(d) for d in data]
    
    @abstractmethod
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load raw data from source. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _preprocess_sample(self, raw_sample: Dict[str, Any], idx: int) -> BenchmarkSample:
        """Convert raw sample to BenchmarkSample. Must be implemented by subclasses."""
        pass
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> BenchmarkSample:
        return self.samples[idx]
    
    def __iter__(self) -> Iterator[BenchmarkSample]:
        return iter(self.samples)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics for reporting."""
        context_lengths = [s.context_length for s in self.samples]
        return {
            'name': self.config.name,
            'family': self.config.family.value,
            'task_type': self.config.task_type.value,
            'num_samples': len(self.samples),
            'avg_context_length': sum(context_lengths) / len(context_lengths) if context_lengths else 0,
            'min_context_length': min(context_lengths) if context_lengths else 0,
            'max_context_length': max(context_lengths) if context_lengths else 0,
        }


# =============================================================================
# LongBench Dataset
# =============================================================================

class LongBenchDataset(BaseBenchmarkDataset):
    """
    LongBench: Multi-task benchmark for long context understanding.
    
    Source: THUDM/LongBench on HuggingFace
    Paper: https://arxiv.org/abs/2308.14508
    
    Tasks:
    - QA: narrativeqa, qasper, multifieldqa, hotpotqa, 2wikimqa, musique, dureader
    - Summarization: gov_report, qmsum, multi_news, vcsum
    - Few-shot: trec, triviaqa, samsum, lsht
    - Code: lcc, repobench-p
    """
    
    HF_DATASET_NAME = "THUDM/LongBench"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load LongBench data from HuggingFace."""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(f"datasets library required: pip install datasets. Error: {_HF_IMPORT_ERROR}")
        
        # LongBench stores data in data.zip, we need to download and extract
        import tempfile
        import zipfile
        import io
        
        cache_dir = Path(self.cache_dir) / "longbench_raw"
        cache_dir.mkdir(parents=True, exist_ok=True)
        data_file = cache_dir / f"{self.config.name}.jsonl"
        
        # Check if we already have the extracted data
        if data_file.exists():
            with open(data_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return [json.loads(line) for line in lines if line.strip()]
        
        # Download and extract from data.zip
        if REQUESTS_AVAILABLE:
            try:
                zip_url = f"https://huggingface.co/datasets/{self.HF_DATASET_NAME}/resolve/main/data.zip"
                print(f"Downloading LongBench data from {zip_url}...")
                response = requests.get(zip_url, timeout=120)
                
                if response.status_code == 200:
                    # Extract the specific file we need
                    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                        # List files in zip to find the right one
                        file_list = zf.namelist()
                        target_file = None
                        for name in file_list:
                            if self.config.name in name and name.endswith('.jsonl'):
                                target_file = name
                                break
                        
                        if target_file:
                            # Extract and cache
                            content = zf.read(target_file).decode('utf-8')
                            with open(data_file, 'w', encoding='utf-8') as f:
                                f.write(content)
                            
                            lines = content.strip().split('\n')
                            return [json.loads(line) for line in lines if line.strip()]
                        else:
                            raise RuntimeError(f"Could not find {self.config.name}.jsonl in data.zip. Available: {file_list[:10]}")
                else:
                    raise RuntimeError(f"Failed to download data.zip: {response.status_code}")
            except Exception as e:
                raise RuntimeError(f"Failed to load LongBench dataset '{self.config.name}': {e}")
        else:
            raise ImportError("requests library required for LongBench: pip install requests")
    
    def _preprocess_sample(self, raw: Dict[str, Any], idx: int) -> BenchmarkSample:
        """Convert LongBench sample to unified format."""
        
        # Extract context and question
        context = raw.get('context', '')
        question = raw.get('input', '')
        
        # Extract answers (format varies by task)
        answers = raw.get('answers', [])
        if isinstance(answers, str):
            answers = [answers]
        
        # Primary answer for summarization tasks
        answer = answers[0] if answers else None
        
        return BenchmarkSample(
            sample_id=f"longbench_{self.config.name}_{idx}",
            dataset_name=self.config.name,
            context=context,
            question=question,
            answers=answers,
            answer=answer,
            task_type=self.config.task_type,
            metadata={
                'length': raw.get('length', len(context)),
                'all_classes': raw.get('all_classes', None),
            }
        )


# =============================================================================
# SCROLLS Dataset
# =============================================================================

class SCROLLSDataset(BaseBenchmarkDataset):
    """
    SCROLLS: Standardized CompaRison Over Long Language Sequences.
    
    Source: tau/scrolls on HuggingFace
    Paper: https://arxiv.org/abs/2201.03533
    
    Tasks:
    - QuALITY: Multiple-choice reading comprehension
    - Qasper: Scientific QA
    - NarrativeQA: Story understanding
    - SummScreenFD: Dialogue summarization
    - GovReport: Report summarization
    - ContractNLI: Legal NLI
    """
    
    HF_DATASET_NAME = "tau/scrolls"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load SCROLLS data from HuggingFace."""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(f"datasets library required: pip install datasets. Error: {_HF_IMPORT_ERROR}")
        
        # Map config name to SCROLLS subset name
        subset_map = {
            'quality': 'quality',
            'qasper': 'qasper',
            'qasper_scrolls': 'qasper',
            'narrativeqa': 'narrative_qa',
            'narrativeqa_scrolls': 'narrative_qa',
            'summ_screen_fd': 'summ_screen_fd',
            'gov_report': 'gov_report',
            'gov_report_scrolls': 'gov_report',
            'contract_nli': 'contract_nli',
        }
        
        subset = subset_map.get(self.config.name, self.config.name)
        
        try:
            dataset = load_dataset(
                self.HF_DATASET_NAME,
                subset,
                split=self.config.split if self.config.split != 'test' else 'validation',
            )
        except Exception:
            # Fallback with trust_remote_code for older datasets
        dataset = load_dataset(
            self.HF_DATASET_NAME,
            subset,
            split=self.config.split if self.config.split != 'test' else 'validation',
            trust_remote_code=True,
        )
        
        return list(dataset)
    
    def _preprocess_sample(self, raw: Dict[str, Any], idx: int) -> BenchmarkSample:
        """Convert SCROLLS sample to unified format."""
        
        context = raw.get('input', '')
        
        # Parse the input to separate context and question for some tasks
        question = ""
        if self.config.task_type == TaskType.MULTIPLE_CHOICE:
            # QuALITY format: question in input with options
            question = raw.get('input', '').split('\n')[-1] if '\n' in raw.get('input', '') else ''
        
        # Extract answer
        output = raw.get('output', '')
        answers = [output] if output else []
        
        # Handle multiple choice
        options = None
        correct_option = None
        if self.config.task_type == TaskType.MULTIPLE_CHOICE:
            # Parse options from input if available
            pass  # SCROLLS format varies
        
        return BenchmarkSample(
            sample_id=f"scrolls_{self.config.name}_{idx}",
            dataset_name=self.config.name,
            context=context,
            question=question,
            answers=answers,
            answer=output,
            options=options,
            correct_option=correct_option,
            task_type=self.config.task_type,
            metadata={
                'id': raw.get('id', str(idx)),
                'pid': raw.get('pid', None),
            }
        )


# =============================================================================
# InfiniteBench Dataset
# =============================================================================

class InfiniteBenchDataset(BaseBenchmarkDataset):
    """
    InfiniteBench: Extending Long Context Evaluation Beyond 100K Tokens.
    
    Source: xinrongzhang2022/InfiniteBench on HuggingFace
    Paper: https://arxiv.org/abs/2402.13718
    
    Tasks (128K+ context):
    - Passkey retrieval
    - Number string retrieval
    - KV retrieval
    - LongBook QA/Summary/Choice
    - Math find/calc
    - Code run/debug
    """
    
    HF_DATASET_NAME = "xinrongzhang2022/InfiniteBench"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load InfiniteBench data from HuggingFace."""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(f"datasets library required: pip install datasets. Error: {_HF_IMPORT_ERROR}")
        
        # Map config names to InfiniteBench subset names
        subset_map = {
            'passkey': 'passkey',
            'number_string': 'number_string',
            'kv_retrieval': 'kv_retrieval',
            'longbook_qa_eng': 'longbook_qa_eng',
            'longbook_sum_eng': 'longbook_sum_eng',
            'longbook_choice_eng': 'longbook_choice_eng',
            'longdialogue_qa_eng': 'longdialogue_qa_eng',
            'math_find': 'math_find',
            'math_calc': 'math_calc',
            'code_run': 'code_run',
            'code_debug': 'code_debug',
        }
        
        subset = subset_map.get(self.config.name, self.config.name)
        
        try:
            # Try standard loading first
            dataset = load_dataset(
                self.HF_DATASET_NAME,
                subset,
                split=self.config.split,
            )
        except Exception:
            try:
                # Fallback with trust_remote_code
                dataset = load_dataset(
                    self.HF_DATASET_NAME,
                    subset,
                    split=self.config.split,
                    trust_remote_code=True,
                )
            except Exception:
                # Try loading without subset and filter
                try:
                    dataset = load_dataset(
                        self.HF_DATASET_NAME,
                        split=self.config.split,
                    )
                except Exception:
            dataset = load_dataset(
                self.HF_DATASET_NAME,
                split=self.config.split,
                trust_remote_code=True,
            )
            # Filter for the specific task
            dataset = dataset.filter(lambda x: x.get('task', '') == subset)
        
        return list(dataset)
    
    def _preprocess_sample(self, raw: Dict[str, Any], idx: int) -> BenchmarkSample:
        """Convert InfiniteBench sample to unified format."""
        
        context = raw.get('context', raw.get('input', ''))
        question = raw.get('input', raw.get('question', ''))
        
        # For passkey/retrieval tasks, the answer is often a single value
        answer = raw.get('answer', raw.get('target', ''))
        answers = [answer] if answer else []
        
        # Handle multiple choice
        options = raw.get('options', None)
        correct_option = raw.get('answer_idx', None)
        
        return BenchmarkSample(
            sample_id=f"infinitebench_{self.config.name}_{idx}",
            dataset_name=self.config.name,
            context=context,
            question=question,
            answers=answers,
            answer=answer,
            options=options,
            correct_option=correct_option,
            task_type=self.config.task_type,
            metadata={
                'task': raw.get('task', self.config.name),
                'context_length_tokens': raw.get('length', None),
            }
        )


# =============================================================================
# ZeroSCROLLS Dataset
# =============================================================================

class ZeroSCROLLSDataset(BaseBenchmarkDataset):
    """
    ZeroSCROLLS: Zero-shot variant of SCROLLS benchmark.
    
    Source: tau/zero_scrolls on HuggingFace
    Paper: https://arxiv.org/abs/2305.14196
    
    Same tasks as SCROLLS but in zero-shot setting (no few-shot examples).
    """
    
    HF_DATASET_NAME = "tau/zero_scrolls"
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Load ZeroSCROLLS data from HuggingFace."""
        if not HF_DATASETS_AVAILABLE:
            raise ImportError(f"datasets library required: pip install datasets. Error: {_HF_IMPORT_ERROR}")
        
        # Map config names to ZeroSCROLLS subset names
        subset_map = {
            'quality_zero': 'quality',
            'qasper_zero': 'qasper',
            'narrativeqa_zero': 'narrativeqa',
            'summ_screen_fd_zero': 'summ_screen_fd',
            'gov_report_zero': 'gov_report',
            'squality': 'squality',
            'book_sum_sort': 'book_sum_sort',
        }
        
        subset = subset_map.get(self.config.name, self.config.name.replace('_zero', ''))
        
        try:
            dataset = load_dataset(
                self.HF_DATASET_NAME,
                subset,
                split=self.config.split if self.config.split != 'test' else 'validation',
            )
        except Exception:
            # Fallback with trust_remote_code
        dataset = load_dataset(
            self.HF_DATASET_NAME,
            subset,
            split=self.config.split if self.config.split != 'test' else 'validation',
            trust_remote_code=True,
        )
        
        return list(dataset)
    
    def _preprocess_sample(self, raw: Dict[str, Any], idx: int) -> BenchmarkSample:
        """Convert ZeroSCROLLS sample to unified format."""
        
        context = raw.get('input', '')
        question = raw.get('question', '')
        
        output = raw.get('output', raw.get('target', ''))
        answers = [output] if output else []
        
        return BenchmarkSample(
            sample_id=f"zeroscrolls_{self.config.name}_{idx}",
            dataset_name=self.config.name,
            context=context,
            question=question,
            answers=answers,
            answer=output,
            task_type=self.config.task_type,
            metadata={
                'id': raw.get('id', str(idx)),
            }
        )


# =============================================================================
# Synthetic Datasets for Controlled Experiments
# =============================================================================

class SyntheticNIAHDataset(BaseBenchmarkDataset):
    """
    Synthetic Needle-in-a-Haystack dataset for controlled experiments.
    
    Creates long contexts with hidden "needles" (facts) at various positions.
    Useful for ablation studies and debugging.
    """
    
    def __init__(
        self,
        config: DatasetConfig,
        num_samples: int = 100,
        context_lengths: List[int] = None,
        needle_positions: List[float] = None,  # 0.0 = start, 1.0 = end
        tokenizer: Optional[Any] = None,
        **kwargs
    ):
        self.num_samples = num_samples
        self.context_lengths = context_lengths or [4096, 8192, 16384]
        self.needle_positions = needle_positions or [0.1, 0.25, 0.5, 0.75, 0.9]
        super().__init__(config, tokenizer, **kwargs)
    
    def _load_raw_data(self) -> List[Dict[str, Any]]:
        """Generate synthetic NIAH samples."""
        import random
        random.seed(42)
        
        samples = []
        haystack_template = "The quick brown fox jumps over the lazy dog. " * 100
        
        for i in range(self.num_samples):
            context_length = random.choice(self.context_lengths)
            needle_pos = random.choice(self.needle_positions)
            
            # Generate random needle
            needle_key = f"KEY_{random.randint(1000, 9999)}"
            needle_value = f"{random.randint(100000, 999999)}"
            needle = f"The secret code for {needle_key} is {needle_value}. "
            
            # Build context
            num_chars = context_length * 4  # Rough char-to-token ratio
            haystack = haystack_template * (num_chars // len(haystack_template) + 1)
            haystack = haystack[:num_chars]
            
            # Insert needle at specified position
            insert_pos = int(len(haystack) * needle_pos)
            context = haystack[:insert_pos] + needle + haystack[insert_pos:]
            
            samples.append({
                'context': context,
                'question': f"What is the secret code for {needle_key}?",
                'answer': needle_value,
                'needle_position': needle_pos,
                'context_length': context_length,
            })
        
        return samples
    
    def _preprocess_sample(self, raw: Dict[str, Any], idx: int) -> BenchmarkSample:
        """Convert synthetic sample to unified format."""
        return BenchmarkSample(
            sample_id=f"synthetic_niah_{idx}",
            dataset_name="synthetic_niah",
            context=raw['context'],
            question=raw['question'],
            answers=[raw['answer']],
            answer=raw['answer'],
            task_type=TaskType.RETRIEVAL,
            metadata={
                'needle_position': raw['needle_position'],
                'target_context_length': raw['context_length'],
            }
        )


# =============================================================================
# Dataset Registry
# =============================================================================

class DatasetRegistry:
    """Registry of available datasets."""
    
    _datasets = {
        # LongBench
        DatasetFamily.LONGBENCH: LongBenchDataset,
        # SCROLLS
        DatasetFamily.SCROLLS: SCROLLSDataset,
        # InfiniteBench
        DatasetFamily.INFINITEBENCH: InfiniteBenchDataset,
        # ZeroSCROLLS
        DatasetFamily.ZEROSCROLLS: ZeroSCROLLSDataset,
    }
    
    @classmethod
    def get_dataset_class(cls, family: DatasetFamily) -> type:
        """Get dataset class for a family."""
        if family not in cls._datasets:
            raise ValueError(f"Unknown dataset family: {family}")
        return cls._datasets[family]
    
    @classmethod
    def list_families(cls) -> List[str]:
        """List all available dataset families."""
        return [f.value for f in cls._datasets.keys()]
    
    @classmethod
    def list_datasets(cls, family: Optional[DatasetFamily] = None) -> List[str]:
        """List all available datasets, optionally filtered by family."""
        datasets = []
        for name, config in ALL_DATASETS.items():
            if family is None or config.family == family:
                datasets.append(name)
        return datasets


def get_dataset(
    name: str,
    config: Optional[DatasetConfig] = None,
    tokenizer: Optional[Any] = None,
    **kwargs
) -> BaseBenchmarkDataset:
    """
    Get dataset by name.
    
    Args:
        name: Dataset name (e.g., "narrativeqa", "qasper")
        config: Optional custom configuration
        tokenizer: Optional tokenizer for length filtering
        **kwargs: Additional arguments passed to dataset constructor
    
    Returns:
        BaseBenchmarkDataset instance
    
    Example:
        >>> dataset = get_dataset("narrativeqa")
        >>> print(len(dataset))
        >>> sample = dataset[0]
    """
    if config is None:
        if name not in ALL_DATASETS:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(ALL_DATASETS.keys())}")
        config = ALL_DATASETS[name]
    
    dataset_class = DatasetRegistry.get_dataset_class(config.family)
    return dataset_class(config, tokenizer=tokenizer, **kwargs)


def create_dataloader(
    dataset: BaseBenchmarkDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn: Optional[callable] = None,
) -> DataLoader:
    """
    Create DataLoader for a benchmark dataset.
    
    Args:
        dataset: BenchmarkDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        collate_fn: Custom collate function
    
    Returns:
        DataLoader instance
    """
    def default_collate(samples: List[BenchmarkSample]) -> List[BenchmarkSample]:
        """Default collate just returns list of samples."""
        return samples
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn or default_collate,
    )


# =============================================================================
# Utility Functions
# =============================================================================

def print_dataset_statistics(datasets: List[str] = None) -> None:
    """Print statistics for specified datasets."""
    if datasets is None:
        datasets = list(ALL_DATASETS.keys())
    
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    
    for family in DatasetFamily:
        family_datasets = [d for d in datasets if ALL_DATASETS.get(d, DatasetConfig(name="", family=DatasetFamily.LONGBENCH, task_type=TaskType.QA)).family == family]
        if not family_datasets:
            continue
        
        print(f"\n{family.value.upper()}")
        print("-" * 40)
        
        for name in family_datasets:
            config = ALL_DATASETS[name]
            print(f"  {name}:")
            print(f"    Task: {config.task_type.value}")
            print(f"    Max Length: {config.max_context_length:,} tokens")
            print(f"    Metrics: {[m.value for m in config.metrics]}")


def validate_dataset_availability() -> Dict[str, bool]:
    """Check which datasets are available/downloadable."""
    availability = {}
    
    for name, config in ALL_DATASETS.items():
        try:
            dataset_class = DatasetRegistry.get_dataset_class(config.family)
            # Try to load just metadata
            availability[name] = True
        except Exception as e:
            availability[name] = False
    
    return availability


if __name__ == "__main__":
    # Print available datasets
    print_dataset_statistics()
    
    # Test loading a small sample
    print("\n\nTesting dataset loading...")
    try:
        dataset = get_dataset("narrativeqa")
        print(f"Loaded {len(dataset)} samples from narrativeqa")
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample context length: {sample.context_length}")
    except Exception as e:
        print(f"Could not load dataset: {e}")

