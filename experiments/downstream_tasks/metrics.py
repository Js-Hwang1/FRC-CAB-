"""
Metrics for Downstream Tasks Benchmark

Implements evaluation metrics for all TODO 1.4 tasks:
- Summarization: ROUGE-1/2/L, BLEU, BERTScore
- QA: F1, Exact Match
- Dialogue: Joint Goal Accuracy, Slot Accuracy
- Code: CodeBLEU, pass@k
"""

import re
import json
import string
import logging
from typing import List, Dict, Any, Optional, Union
from collections import Counter

try:
    import numpy as np
    NP_AVAILABLE = True
except ImportError:
    NP_AVAILABLE = False
    np = None

from .config import MetricName


logger = logging.getLogger(__name__)


# =============================================================================
# Text Normalization
# =============================================================================

def normalize_answer(s: str) -> str:
    """Normalize text for comparison."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> List[str]:
    """Tokenize string into words."""
    if not s:
        return []
    return normalize_answer(s).split()


# =============================================================================
# Basic Metrics
# =============================================================================

def compute_f1(prediction: str, reference: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)
    
    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)
    
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    
    return 2 * precision * recall / (precision + recall)


def compute_exact_match(prediction: str, reference: str) -> float:
    """Compute exact match score."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def compute_accuracy(prediction: str, reference: str) -> float:
    """Compute accuracy (same as exact match for single answers)."""
    return compute_exact_match(prediction, reference)


# =============================================================================
# ROUGE Metrics
# =============================================================================

def compute_rouge_n(prediction: str, reference: str, n: int = 1) -> Dict[str, float]:
    """Compute ROUGE-N score."""
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)
    
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    pred_ngrams = get_ngrams(pred_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)
    
    overlap = pred_ngrams & ref_ngrams
    overlap_count = sum(overlap.values())
    
    precision = overlap_count / sum(pred_ngrams.values()) if pred_ngrams else 0.0
    recall = overlap_count / sum(ref_ngrams.values()) if ref_ngrams else 0.0
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_rouge_l(prediction: str, reference: str) -> Dict[str, float]:
    """Compute ROUGE-L score using Longest Common Subsequence."""
    def lcs_length(x: List[str], y: List[str]) -> int:
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        
        # Use space-efficient LCS
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, [0] * (n + 1)
        
        return prev[n]
    
    pred_tokens = get_tokens(prediction)
    ref_tokens = get_tokens(reference)
    
    if not pred_tokens or not ref_tokens:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    lcs_len = lcs_length(pred_tokens, ref_tokens)
    
    precision = lcs_len / len(pred_tokens)
    recall = lcs_len / len(ref_tokens)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_rouge_1(prediction: str, reference: str) -> float:
    """Compute ROUGE-1 F1."""
    return compute_rouge_n(prediction, reference, n=1)['f1']


def compute_rouge_2(prediction: str, reference: str) -> float:
    """Compute ROUGE-2 F1."""
    return compute_rouge_n(prediction, reference, n=2)['f1']


def compute_rouge_l_f1(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1."""
    return compute_rouge_l(prediction, reference)['f1']


# =============================================================================
# BLEU Metrics
# =============================================================================

def compute_bleu(prediction: str, references: List[str], max_n: int = 4) -> float:
    """
    Compute sentence-level BLEU score.
    
    Uses smoothing for short sentences.
    """
    def get_ngrams(tokens: List[str], n: int) -> Counter:
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))
    
    pred_tokens = get_tokens(prediction)
    
    if not pred_tokens:
        return 0.0
    
    # Collect reference n-grams
    ref_ngrams_list = []
    ref_lengths = []
    for ref in references:
        ref_tokens = get_tokens(ref)
        ref_lengths.append(len(ref_tokens))
        ref_ngrams_list.append({
            n: get_ngrams(ref_tokens, n) for n in range(1, max_n + 1)
        })
    
    # Compute modified precision for each n
    precisions = []
    for n in range(1, min(max_n + 1, len(pred_tokens) + 1)):
        pred_ngrams = get_ngrams(pred_tokens, n)
        
        # Max count from any reference
        max_counts = Counter()
        for ref_ngrams in ref_ngrams_list:
            ref_n = ref_ngrams.get(n, Counter())
            for ngram in pred_ngrams:
                max_counts[ngram] = max(max_counts[ngram], ref_n[ngram])
        
        clipped = sum(min(count, max_counts[ngram]) 
                      for ngram, count in pred_ngrams.items())
        total = sum(pred_ngrams.values())
        
        # Add smoothing for zero counts
        if total == 0:
            precision = 0.0
        else:
            precision = (clipped + 1) / (total + 1)  # Add-1 smoothing
        
        precisions.append(precision)
    
    if not precisions:
        return 0.0
    
    # Geometric mean of precisions
    if NP_AVAILABLE:
        log_precisions = [np.log(p) if p > 0 else -100 for p in precisions]
        avg_log_precision = np.mean(log_precisions)
        geo_mean = np.exp(avg_log_precision)
    else:
        import math
        log_precisions = [math.log(p) if p > 0 else -100 for p in precisions]
        avg_log_precision = sum(log_precisions) / len(log_precisions)
        geo_mean = math.exp(avg_log_precision)
    
    # Brevity penalty
    c = len(pred_tokens)
    r = min(ref_lengths, key=lambda x: abs(x - c))  # Closest reference length
    
    if c >= r:
        bp = 1.0
    else:
        if NP_AVAILABLE:
            bp = np.exp(1 - r / c)
        else:
            import math
            bp = math.exp(1 - r / c)
    
    return bp * geo_mean


# =============================================================================
# Dialogue Metrics
# =============================================================================

def compute_joint_goal_accuracy(prediction: str, reference: str) -> float:
    """
    Compute joint goal accuracy for dialogue state tracking.
    
    Both prediction and reference should be JSON strings representing
    the dialogue state.
    """
    try:
        pred_state = json.loads(prediction)
        ref_state = json.loads(reference)
    except json.JSONDecodeError:
        # Fallback to string comparison
        return compute_exact_match(prediction, reference)
    
    # Check if all slots match exactly
    if pred_state == ref_state:
        return 1.0
    return 0.0


def compute_slot_accuracy(prediction: str, reference: str) -> float:
    """
    Compute slot-level accuracy for dialogue state tracking.
    
    Returns the fraction of slots that are correct.
    """
    try:
        pred_state = json.loads(prediction)
        ref_state = json.loads(reference)
    except json.JSONDecodeError:
        return compute_exact_match(prediction, reference)
    
    # Flatten nested dicts
    def flatten_dict(d: Dict, parent_key: str = '') -> Dict:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    pred_flat = flatten_dict(pred_state) if isinstance(pred_state, dict) else {}
    ref_flat = flatten_dict(ref_state) if isinstance(ref_state, dict) else {}
    
    all_slots = set(pred_flat.keys()) | set(ref_flat.keys())
    
    if not all_slots:
        return 1.0 if pred_state == ref_state else 0.0
    
    correct = sum(1 for slot in all_slots 
                  if pred_flat.get(slot) == ref_flat.get(slot))
    
    return correct / len(all_slots)


# =============================================================================
# Code Metrics
# =============================================================================

def compute_code_bleu(prediction: str, reference: str) -> float:
    """
    Compute CodeBLEU score (simplified version).
    
    Uses standard BLEU on code tokens.
    """
    # Simple tokenization for code
    def tokenize_code(code: str) -> List[str]:
        # Split on whitespace and common operators
        tokens = re.split(r'(\s+|[{}()\[\];,.])', code)
        return [t.strip() for t in tokens if t.strip()]
    
    pred_tokens = tokenize_code(prediction)
    ref_tokens = tokenize_code(reference)
    
    if not pred_tokens or not ref_tokens:
        return 0.0
    
    # Compute BLEU on code tokens
    return compute_bleu(
        ' '.join(pred_tokens),
        [' '.join(ref_tokens)],
        max_n=4
    )


def compute_pass_at_k(
    predictions: List[str],
    test_cases: str,
    entry_point: str,
    k: int = 1,
) -> float:
    """
    Compute pass@k for code generation.
    
    Note: This is a simplified version that doesn't actually execute code.
    For real evaluation, use the HumanEval execution harness.
    """
    # This would require code execution - return placeholder
    logger.warning("pass@k requires code execution. Returning 0.0")
    return 0.0


# =============================================================================
# Multi-Reference Metrics
# =============================================================================

def compute_max_over_references(
    prediction: str,
    references: List[str],
    metric_fn,
) -> float:
    """Compute max of metric over multiple references."""
    if not references:
        return 0.0
    
    scores = [metric_fn(prediction, ref) for ref in references]
    return max(scores)


# =============================================================================
# Main Metric Computation
# =============================================================================

METRIC_FUNCTIONS = {
    MetricName.F1: compute_f1,
    MetricName.EXACT_MATCH: compute_exact_match,
    MetricName.ACCURACY: compute_accuracy,
    MetricName.ROUGE_1: compute_rouge_1,
    MetricName.ROUGE_2: compute_rouge_2,
    MetricName.ROUGE_L: compute_rouge_l_f1,
    MetricName.BLEU: lambda p, r: compute_bleu(p, [r]),
    MetricName.JOINT_GOAL_ACCURACY: compute_joint_goal_accuracy,
    MetricName.SLOT_ACCURACY: compute_slot_accuracy,
    MetricName.CODE_BLEU: compute_code_bleu,
}


def compute_metrics(
    prediction: str,
    references: Union[str, List[str]],
    metrics: List[MetricName],
) -> Dict[str, float]:
    """
    Compute all specified metrics.
    
    Args:
        prediction: Model prediction
        references: Ground truth (single or multiple)
        metrics: List of metrics to compute
    
    Returns:
        Dict mapping metric name to score
    """
    if isinstance(references, str):
        references = [references]
    
    if not references:
        return {m.value: 0.0 for m in metrics}
    
    results = {}
    
    for metric in metrics:
        if metric not in METRIC_FUNCTIONS:
            logger.warning(f"Unknown metric: {metric}")
            results[metric.value] = 0.0
            continue
        
        metric_fn = METRIC_FUNCTIONS[metric]
        
        # Take max over references
        score = compute_max_over_references(prediction, references, metric_fn)
        results[metric.value] = score
    
    return results


def compute_batch_metrics(
    predictions: List[str],
    references_list: List[List[str]],
    metrics: List[MetricName],
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for a batch of predictions.
    
    Returns aggregated statistics (mean, std, min, max).
    """
    all_scores = {m.value: [] for m in metrics}
    
    for pred, refs in zip(predictions, references_list):
        scores = compute_metrics(pred, refs, metrics)
        for metric_name, score in scores.items():
            all_scores[metric_name].append(score)
    
    # Aggregate
    results = {}
    for metric_name, scores in all_scores.items():
        if not scores:
            results[metric_name] = {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
            continue
        
        if NP_AVAILABLE:
            results[metric_name] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'min': float(np.min(scores)),
                'max': float(np.max(scores)),
                'count': len(scores),
            }
        else:
            mean_val = sum(scores) / len(scores)
            std_val = (sum((x - mean_val) ** 2 for x in scores) / len(scores)) ** 0.5
            results[metric_name] = {
                'mean': mean_val,
                'std': std_val,
                'min': min(scores),
                'max': max(scores),
                'count': len(scores),
            }
    
    return results

