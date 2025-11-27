"""
Data Loaders for Downstream Tasks Benchmark

Implements loaders for all TODO 1.4 tasks:
- Document Summarization (CNN/DailyMail, XSum, Multi-News)
- Open-Domain QA (Natural Questions, TriviaQA, SQuAD v2)
- Dialogue State Tracking (MultiWOZ)
- Code Understanding (CodeXGLUE, HumanEval)
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterator
from pathlib import Path

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    TorchDataset = object

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False
    load_dataset = None

from .config import DatasetConfig, TaskType, ALL_DATASETS


logger = logging.getLogger(__name__)


# =============================================================================
# Sample Dataclass
# =============================================================================

@dataclass
class BenchmarkSample:
    """A single sample for downstream task evaluation."""
    sample_id: str
    
    # Input
    input_text: str  # Main input (question, document, code)
    context: str = ""  # Optional additional context
    
    # Output
    target: str = ""  # Expected output
    targets: List[str] = None  # Multiple valid outputs
    
    # Metadata
    task_type: str = ""
    dataset_name: str = ""
    context_length: int = 0
    
    def __post_init__(self):
        if self.targets is None:
            self.targets = [self.target] if self.target else []
        if not self.context_length:
            self.context_length = len(self.input_text) + len(self.context)


# =============================================================================
# Base Dataset Class
# =============================================================================

class BaseBenchmarkDataset(TorchDataset, ABC):
    """Base class for all downstream task datasets."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.samples: List[BenchmarkSample] = []
        self._loaded = False
    
    @abstractmethod
    def load(self) -> None:
        """Load the dataset."""
        pass
    
    def __len__(self) -> int:
        if not self._loaded:
            self.load()
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> BenchmarkSample:
        if not self._loaded:
            self.load()
        return self.samples[idx]
    
    def __iter__(self) -> Iterator[BenchmarkSample]:
        if not self._loaded:
            self.load()
        return iter(self.samples[:self.config.max_samples])


# =============================================================================
# Summarization Datasets
# =============================================================================

class CNNDailyMailDataset(BaseBenchmarkDataset):
    """CNN/DailyMail summarization dataset."""
    
    def load(self) -> None:
        if self._loaded:
            return
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading CNN/DailyMail dataset...")
        
        try:
            dataset = load_dataset(
                self.config.hf_name,
                self.config.hf_subset,
                split=self.config.hf_split,
            )
        except Exception as e:
            logger.error(f"Failed to load CNN/DailyMail: {e}")
            raise
        
        for idx, item in enumerate(dataset):
            if idx >= self.config.max_samples:
                break
            
            article = item.get(self.config.input_field, item.get("article", ""))
            highlights = item.get(self.config.target_field, item.get("highlights", ""))
            
            # Truncate if too long
            if len(article) > self.config.max_context_length * 4:
                article = article[:self.config.max_context_length * 4]
            
            sample = BenchmarkSample(
                sample_id=f"cnn_dailymail_{idx}",
                input_text=article,
                target=highlights,
                task_type="summarization",
                dataset_name="cnn_dailymail",
            )
            self.samples.append(sample)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} CNN/DailyMail samples")


class XSumDataset(BaseBenchmarkDataset):
    """XSum abstractive summarization dataset."""
    
    def load(self) -> None:
        if self._loaded:
            return
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading XSum dataset...")
        
        try:
            dataset = load_dataset(self.config.hf_name, split=self.config.hf_split)
        except Exception as e:
            logger.error(f"Failed to load XSum: {e}")
            raise
        
        for idx, item in enumerate(dataset):
            if idx >= self.config.max_samples:
                break
            
            document = item.get(self.config.input_field, item.get("document", ""))
            summary = item.get(self.config.target_field, item.get("summary", ""))
            
            sample = BenchmarkSample(
                sample_id=f"xsum_{idx}",
                input_text=document,
                target=summary,
                task_type="summarization",
                dataset_name="xsum",
            )
            self.samples.append(sample)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} XSum samples")


class MultiNewsDataset(BaseBenchmarkDataset):
    """Multi-News multi-document summarization dataset."""
    
    def load(self) -> None:
        if self._loaded:
            return
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading Multi-News dataset...")
        
        try:
            dataset = load_dataset(self.config.hf_name, split=self.config.hf_split)
        except Exception as e:
            logger.error(f"Failed to load Multi-News: {e}")
            raise
        
        for idx, item in enumerate(dataset):
            if idx >= self.config.max_samples:
                break
            
            document = item.get(self.config.input_field, item.get("document", ""))
            summary = item.get(self.config.target_field, item.get("summary", ""))
            
            # Multi-News has multiple documents separated by special tokens
            # Clean them up
            document = document.replace("|||||", "\n\n---\n\n")
            
            sample = BenchmarkSample(
                sample_id=f"multi_news_{idx}",
                input_text=document,
                target=summary,
                task_type="summarization",
                dataset_name="multi_news",
            )
            self.samples.append(sample)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} Multi-News samples")


# =============================================================================
# Open-Domain QA Datasets
# =============================================================================

class NaturalQuestionsDataset(BaseBenchmarkDataset):
    """Natural Questions dataset."""
    
    def load(self) -> None:
        if self._loaded:
            return
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading Natural Questions dataset...")
        
        try:
            # Natural Questions is large, use streaming
            dataset = load_dataset(
                self.config.hf_name,
                split=self.config.hf_split,
                streaming=True,
            )
        except Exception as e:
            logger.error(f"Failed to load Natural Questions: {e}")
            raise
        
        count = 0
        for item in dataset:
            if count >= self.config.max_samples:
                break
            
            question = item.get("question", {})
            if isinstance(question, dict):
                question_text = question.get("text", "")
            else:
                question_text = str(question)
            
            # Get document HTML/text
            document = item.get("document", {})
            if isinstance(document, dict):
                doc_text = document.get("html", document.get("text", ""))
            else:
                doc_text = str(document)
            
            # Get answers
            annotations = item.get("annotations", [])
            answers = []
            for ann in annotations:
                if isinstance(ann, dict):
                    short_answers = ann.get("short_answers", [])
                    for sa in short_answers:
                        if isinstance(sa, dict):
                            start = sa.get("start_byte", 0)
                            end = sa.get("end_byte", 0)
                            if start < end and end <= len(doc_text):
                                answers.append(doc_text[start:end])
                        elif isinstance(sa, str):
                            answers.append(sa)
            
            if not answers:
                continue  # Skip samples without answers
            
            sample = BenchmarkSample(
                sample_id=f"nq_{count}",
                input_text=question_text,
                context=doc_text[:self.config.max_context_length * 4],
                target=answers[0],
                targets=answers,
                task_type="open_domain_qa",
                dataset_name="natural_questions",
            )
            self.samples.append(sample)
            count += 1
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} Natural Questions samples")


class TriviaQADataset(BaseBenchmarkDataset):
    """TriviaQA dataset with long context."""
    
    def load(self) -> None:
        if self._loaded:
            return
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading TriviaQA dataset...")
        
        try:
            dataset = load_dataset(
                self.config.hf_name,
                self.config.hf_subset,
                split=self.config.hf_split,
            )
        except Exception as e:
            logger.error(f"Failed to load TriviaQA: {e}")
            raise
        
        for idx, item in enumerate(dataset):
            if idx >= self.config.max_samples:
                break
            
            question = item.get("question", "")
            
            # Get answer
            answer_dict = item.get("answer", {})
            if isinstance(answer_dict, dict):
                answer = answer_dict.get("value", "")
                aliases = answer_dict.get("aliases", [])
                answers = [answer] + aliases if answer else aliases
            else:
                answers = [str(answer_dict)]
            
            # Get context from search results or entity pages
            search_results = item.get("search_results", {})
            if isinstance(search_results, dict):
                search_contexts = search_results.get("search_context", [])
                context = "\n\n".join(search_contexts) if search_contexts else ""
            else:
                context = ""
            
            if not context:
                entity_pages = item.get("entity_pages", {})
                if isinstance(entity_pages, dict):
                    wiki_contexts = entity_pages.get("wiki_context", [])
                    context = "\n\n".join(wiki_contexts) if wiki_contexts else ""
            
            if not answers or not context:
                continue
            
            sample = BenchmarkSample(
                sample_id=f"triviaqa_{idx}",
                input_text=question,
                context=context[:self.config.max_context_length * 4],
                target=answers[0],
                targets=answers,
                task_type="open_domain_qa",
                dataset_name="triviaqa",
            )
            self.samples.append(sample)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} TriviaQA samples")


class SQuADv2Dataset(BaseBenchmarkDataset):
    """SQuAD 2.0 reading comprehension dataset."""
    
    def load(self) -> None:
        if self._loaded:
            return
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading SQuAD v2 dataset...")
        
        try:
            dataset = load_dataset(self.config.hf_name, split=self.config.hf_split)
        except Exception as e:
            logger.error(f"Failed to load SQuAD v2: {e}")
            raise
        
        for idx, item in enumerate(dataset):
            if idx >= self.config.max_samples:
                break
            
            question = item.get("question", "")
            context = item.get("context", "")
            
            answers_dict = item.get("answers", {})
            if isinstance(answers_dict, dict):
                answer_texts = answers_dict.get("text", [])
            else:
                answer_texts = []
            
            # Handle unanswerable questions
            if not answer_texts:
                answer_texts = ["unanswerable"]
            
            sample = BenchmarkSample(
                sample_id=f"squad_v2_{idx}",
                input_text=question,
                context=context,
                target=answer_texts[0],
                targets=answer_texts,
                task_type="open_domain_qa",
                dataset_name="squad_v2",
            )
            self.samples.append(sample)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} SQuAD v2 samples")


# =============================================================================
# Dialogue Datasets
# =============================================================================

class MultiWOZDataset(BaseBenchmarkDataset):
    """MultiWOZ dialogue state tracking dataset."""
    
    def load(self) -> None:
        if self._loaded:
            return
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading MultiWOZ dataset...")
        
        try:
            dataset = load_dataset(self.config.hf_name, split=self.config.hf_split)
        except Exception as e:
            logger.error(f"Failed to load MultiWOZ: {e}")
            raise
        
        for idx, item in enumerate(dataset):
            if idx >= self.config.max_samples:
                break
            
            # Format dialogue turns
            turns = item.get("turns", {})
            if isinstance(turns, dict):
                utterances = turns.get("utterance", [])
                speakers = turns.get("speaker", [])
            else:
                utterances = []
                speakers = []
            
            dialogue_text = ""
            for utt, spk in zip(utterances, speakers):
                role = "User" if spk == 0 else "System"
                dialogue_text += f"{role}: {utt}\n"
            
            # Get dialogue state
            dialogue_state = item.get("dialogue_state", {})
            state_str = json.dumps(dialogue_state) if dialogue_state else "{}"
            
            sample = BenchmarkSample(
                sample_id=f"multiwoz_{idx}",
                input_text=dialogue_text,
                target=state_str,
                task_type="dialogue",
                dataset_name="multiwoz",
            )
            self.samples.append(sample)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} MultiWOZ samples")


# =============================================================================
# Code Datasets
# =============================================================================

class CodeSummarizationDataset(BaseBenchmarkDataset):
    """CodeXGLUE code summarization dataset."""
    
    def load(self) -> None:
        if self._loaded:
            return
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading CodeXGLUE Code Summarization dataset...")
        
        try:
            dataset = load_dataset(
                self.config.hf_name,
                self.config.hf_subset,
                split=self.config.hf_split,
            )
        except Exception as e:
            logger.error(f"Failed to load CodeXGLUE: {e}")
            raise
        
        for idx, item in enumerate(dataset):
            if idx >= self.config.max_samples:
                break
            
            code = item.get(self.config.input_field, item.get("code", ""))
            docstring = item.get(self.config.target_field, item.get("docstring", ""))
            
            sample = BenchmarkSample(
                sample_id=f"code_sum_{idx}",
                input_text=code,
                target=docstring,
                task_type="code",
                dataset_name="code_summarization",
            )
            self.samples.append(sample)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} Code Summarization samples")


class CodeCompletionDataset(BaseBenchmarkDataset):
    """CodeXGLUE code completion dataset."""
    
    def load(self) -> None:
        if self._loaded:
            return
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading CodeXGLUE Code Completion dataset...")
        
        try:
            dataset = load_dataset(
                self.config.hf_name,
                self.config.hf_subset,
                split=self.config.hf_split,
            )
        except Exception as e:
            logger.error(f"Failed to load Code Completion dataset: {e}")
            raise
        
        for idx, item in enumerate(dataset):
            if idx >= self.config.max_samples:
                break
            
            input_code = item.get(self.config.input_field, item.get("input", ""))
            target = item.get(self.config.target_field, item.get("gt", ""))
            
            sample = BenchmarkSample(
                sample_id=f"code_comp_{idx}",
                input_text=input_code,
                target=target,
                task_type="code",
                dataset_name="code_completion",
            )
            self.samples.append(sample)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} Code Completion samples")


class HumanEvalDataset(BaseBenchmarkDataset):
    """OpenAI HumanEval code generation dataset."""
    
    def load(self) -> None:
        if self._loaded:
            return
        
        if not HF_DATASETS_AVAILABLE:
            raise ImportError("datasets library required: pip install datasets")
        
        logger.info(f"Loading HumanEval dataset...")
        
        try:
            dataset = load_dataset(self.config.hf_name, split=self.config.hf_split)
        except Exception as e:
            logger.error(f"Failed to load HumanEval: {e}")
            raise
        
        for idx, item in enumerate(dataset):
            if idx >= self.config.max_samples:
                break
            
            prompt = item.get("prompt", "")
            canonical_solution = item.get("canonical_solution", "")
            test = item.get("test", "")
            entry_point = item.get("entry_point", "")
            
            sample = BenchmarkSample(
                sample_id=item.get("task_id", f"humaneval_{idx}"),
                input_text=prompt,
                target=canonical_solution,
                context=f"# Test cases:\n{test}\n# Entry point: {entry_point}",
                task_type="code",
                dataset_name="humaneval",
            )
            self.samples.append(sample)
        
        self._loaded = True
        logger.info(f"Loaded {len(self.samples)} HumanEval samples")


# =============================================================================
# Dataset Factory
# =============================================================================

DATASET_CLASSES = {
    "cnn_dailymail": CNNDailyMailDataset,
    "xsum": XSumDataset,
    "multi_news": MultiNewsDataset,
    "natural_questions": NaturalQuestionsDataset,
    "triviaqa": TriviaQADataset,
    "squad_v2": SQuADv2Dataset,
    "multiwoz": MultiWOZDataset,
    "code_summarization": CodeSummarizationDataset,
    "code_completion": CodeCompletionDataset,
    "humaneval": HumanEvalDataset,
}


def get_dataset(name: str, config: DatasetConfig = None) -> BaseBenchmarkDataset:
    """
    Get a dataset by name.
    
    Args:
        name: Dataset name
        config: Optional config override
    
    Returns:
        Dataset instance
    """
    if name not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_CLASSES.keys())}")
    
    if config is None:
        if name not in ALL_DATASETS:
            raise ValueError(f"No default config for dataset: {name}")
        config = ALL_DATASETS[name]
    
    dataset_class = DATASET_CLASSES[name]
    return dataset_class(config)


def list_datasets() -> List[str]:
    """List available datasets."""
    return list(DATASET_CLASSES.keys())

