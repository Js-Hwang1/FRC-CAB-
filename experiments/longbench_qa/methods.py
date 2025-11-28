"""
Sparse Attention Methods for Long-Context Benchmarks

Implements all attention methods for fair apple-to-apple comparison:
- Dense Attention (oracle upper bound)
- H2O (Heavy-Hitter Oracle) - magnitude-based (arxiv:2306.14048)
- CAB (Curvature-Aware Block-Sparse) - OUR METHOD
- StreamingLLM (attention sinks + recent tokens) (arxiv:2309.17453)
- Local + Strided (fixed window patterns) (arxiv:1904.10509)
- Random Selection (baseline)

All methods operate on the same interface for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import math

from .config import MethodConfig, MethodName


# =============================================================================
# Base Attention Modifier Class
# =============================================================================

@dataclass
class AttentionOutput:
    """Output from attention computation."""
    output: torch.Tensor                  # [B, N, D] attention output
    attention_weights: Optional[torch.Tensor] = None  # [B, H, N, N] if saved
    block_mask: Optional[torch.Tensor] = None         # [B, H, M, M] block mask
    diagnostics: Optional[Dict[str, Any]] = None      # Additional info


class BaseAttentionMethod(ABC):
    """
    Abstract base class for all attention methods.
    
    All methods must implement:
    - modify_attention_weights(): Apply method-specific masking/selection
    - get_block_mask(): Get block-level mask for analysis
    
    This ensures apple-to-apple comparison across all methods.
    """
    
    def __init__(self, config: MethodConfig):
        self.config = config
        self.name = config.name.value
    
    @abstractmethod
    def modify_attention_weights(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply method-specific modification to attention weights.
        
        Args:
            attention_weights: Raw attention weights [B, H, N, N]
            query: Query tensor [B, H, N, D]
            key: Key tensor [B, H, N, D]
            layer_idx: Current layer index
        
        Returns:
            modified_weights: Modified attention weights [B, H, N, N]
            diagnostics: Dict with method-specific diagnostics
        """
        pass
    
    @abstractmethod
    def get_block_mask(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """
        Get block-level mask for visualization and analysis.
        
        Args:
            attention_weights: Attention weights [B, H, N, N]
            query: Query tensor [B, H, N, D]
            key: Key tensor [B, H, N, D]
            block_size: Tokens per block
        
        Returns:
            block_mask: Binary mask [B, H, M, M] where True = KEEP
        """
        pass
    
    def compute_effective_sparsity(self, mask: torch.Tensor) -> float:
        """Compute actual sparsity from mask."""
        return 1.0 - mask.float().mean().item()
    
    def expand_block_mask(
        self,
        block_mask: torch.Tensor,
        block_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Expand block mask to token-level mask."""
        B, H, M, _ = block_mask.shape
        
        # Repeat each block element
        token_mask = block_mask.repeat_interleave(block_size, dim=2)
        token_mask = token_mask.repeat_interleave(block_size, dim=3)
        
        # Trim to actual sequence length
        token_mask = token_mask[:, :, :seq_len, :seq_len]
        
        return token_mask


# =============================================================================
# Dense Attention (Oracle Upper Bound)
# =============================================================================

class DenseAttention(BaseAttentionMethod):
    """
    Dense (Full) Attention - Oracle upper bound.
    
    No sparsity applied. This represents the best possible performance
    and serves as the upper bound for all sparse methods.
    """
    
    def modify_attention_weights(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Return unmodified attention weights."""
        diagnostics = {
            'method': 'dense',
            'sparsity': 0.0,
            'kept_ratio': 1.0,
        }
        return attention_weights, diagnostics
    
    def get_block_mask(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Return all-ones mask (keep everything)."""
        B, H, N, _ = attention_weights.shape
        M = (N + block_size - 1) // block_size
        return torch.ones(B, H, M, M, dtype=torch.bool, device=attention_weights.device)


# =============================================================================
# H2O (Heavy-Hitter Oracle)
# =============================================================================

class H2OAttention(BaseAttentionMethod):
    """
    H2O: Heavy-Hitter Oracle - Magnitude-based selection.
    
    Paper: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models"
    Authors: Zhang et al., 2023
    arXiv: https://arxiv.org/abs/2306.14048
    
    Algorithm (faithful to paper):
    - Track CUMULATIVE attention scores across all tokens
    - Evict tokens with LOWEST cumulative attention
    - "Heavy hitters" (tokens that receive high attention) are preserved
    
    Key insight: Some tokens consistently receive more attention (heavy hitters),
    while others are rarely attended to and can be safely evicted.
    """
    
    def modify_attention_weights(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply H2O magnitude-based selection."""
        B, H, N, _ = attention_weights.shape
        
        # Calculate number of tokens to keep per query
        k_keep = max(1, int(N * (1 - self.config.sparsity)))
        
        # Compute importance scores (cumulative attention received)
        # Sum over query dimension to get how much each key is attended to
        importance_scores = attention_weights.sum(dim=2)  # [B, H, N]
        
        # Get top-k indices per head
        _, top_indices = torch.topk(importance_scores, k_keep, dim=-1)  # [B, H, k_keep]
        
        # Create mask
        mask = torch.zeros_like(attention_weights, dtype=torch.bool)
        
        # Scatter to create mask (all queries can attend to top-k keys)
        for b in range(B):
            for h in range(H):
                mask[b, h, :, top_indices[b, h]] = True
        
        # Apply causal masking if needed
        if self.config.causal:
            causal_mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=mask.device))
            mask = mask & causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply mask
        modified_weights = attention_weights.clone()
        modified_weights[~mask] = float('-inf')
        
        # Re-normalize
        modified_weights = F.softmax(modified_weights, dim=-1)
        modified_weights = torch.nan_to_num(modified_weights, nan=0.0)
        
        diagnostics = {
            'method': 'h2o',
            'sparsity': self.compute_effective_sparsity(mask),
            'k_keep': k_keep,
            'importance_scores_mean': importance_scores.mean().item(),
        }
        
        return modified_weights, diagnostics
    
    def get_block_mask(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Get block-level H2O mask."""
        B, H, N, _ = attention_weights.shape
        M = (N + block_size - 1) // block_size
        
        # Coarsen attention to block level
        # Use max pooling to get block importance
        block_weights = self._coarsen_attention(attention_weights, block_size)
        
        # Sum over query blocks to get key block importance
        block_importance = block_weights.sum(dim=2)  # [B, H, M]
        
        # Keep top-k blocks
        k_keep = max(1, int(M * (1 - self.config.sparsity)))
        _, top_indices = torch.topk(block_importance, k_keep, dim=-1)
        
        # Create block mask
        block_mask = torch.zeros(B, H, M, M, dtype=torch.bool, device=attention_weights.device)
        for b in range(B):
            for h in range(H):
                block_mask[b, h, :, top_indices[b, h]] = True
        
        return block_mask
    
    def _coarsen_attention(
        self,
        attention_weights: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Coarsen attention matrix to block level using max pooling."""
        B, H, N, _ = attention_weights.shape
        M = (N + block_size - 1) // block_size
        
        # Pad if necessary
        pad_size = M * block_size - N
        if pad_size > 0:
            attention_weights = F.pad(attention_weights, (0, pad_size, 0, pad_size), value=0)
        
        # Reshape to [B, H, M, block_size, M, block_size]
        attention_weights = attention_weights.view(B, H, M, block_size, M, block_size)
        
        # Max pool over tokens within each block
        block_weights = attention_weights.max(dim=3)[0].max(dim=-1)[0]  # [B, H, M, M]
        
        return block_weights


# =============================================================================
# CAB V3 (Pure FRC)
# =============================================================================

class CABV3Attention(BaseAttentionMethod):
    """
    CAB V3: Pure Forman-Ricci Curvature selection.
    
    Selects blocks with highest FRC scores (strong unique connections).
    Does NOT use magnitude information.
    """
    
    def __init__(self, config: MethodConfig):
        super().__init__(config)
        
        # Import CAB modules
        try:
            from cab_attention.kernels.coarsening import coarsen_qk_max_l2_pytorch
            from cab_attention.kernels.frc_kernel import compute_block_frc, generate_block_mask
            self.coarsen_fn = coarsen_qk_max_l2_pytorch
            self.frc_fn = compute_block_frc
            self.mask_fn = generate_block_mask
            self._cab_available = True
        except ImportError:
            self._cab_available = False
    
    def modify_attention_weights(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply CAB V3 FRC-based selection."""
        if not self._cab_available:
            raise ImportError("CAB attention modules not available")
        
        B, H, N, D = query.shape
        block_size = self.config.block_size
        
        # Coarsen Q, K
        q_coarse, k_coarse = self.coarsen_fn(query, key, block_size)
        
        # Compute FRC
        frc_scores, affinity, redundancy = self.frc_fn(
            q_coarse, k_coarse,
            temperature=1.0,
            lambda_redundancy=self.config.lambda_redundancy,
            formula=self.config.formula,
            normalization=self.config.normalization,
        )
        
        # Generate block mask (select HIGH FRC)
        block_mask = self.mask_fn(
            frc_scores,
            sparsity=self.config.sparsity,
            select_high=True,
            keep_diagonal=self.config.keep_diagonal,
            causal=self.config.causal,
        )
        
        # Expand to token level
        token_mask = self.expand_block_mask(block_mask, block_size, N)
        
        # Apply mask
        modified_weights = attention_weights.clone()
        modified_weights[~token_mask] = float('-inf')
        
        # Re-normalize
        modified_weights = F.softmax(modified_weights, dim=-1)
        modified_weights = torch.nan_to_num(modified_weights, nan=0.0)
        
        diagnostics = {
            'method': 'cab',
            'sparsity': self.compute_effective_sparsity(token_mask),
            'frc_mean': frc_scores.mean().item(),
            'frc_std': frc_scores.std().item(),
            'affinity_mean': affinity.mean().item(),
            'redundancy_mean': redundancy.mean().item(),
        }
        
        return modified_weights, diagnostics
    
    def get_block_mask(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Get CAB V3 block mask."""
        if not self._cab_available:
            raise ImportError("CAB attention modules not available")
        
        # Coarsen
        q_coarse, k_coarse = self.coarsen_fn(query, key, block_size)
        
        # Compute FRC
        frc_scores, _, _ = self.frc_fn(
            q_coarse, k_coarse,
            lambda_redundancy=self.config.lambda_redundancy,
            formula=self.config.formula,
            normalization=self.config.normalization,
        )
        
        # Generate mask
        return self.mask_fn(
            frc_scores,
            sparsity=self.config.sparsity,
            select_high=True,
            keep_diagonal=self.config.keep_diagonal,
            causal=self.config.causal,
        )


# =============================================================================
# CAB V4 (Hybrid: Magnitude + FRC) - OUR METHOD
# =============================================================================

class CABV4Attention(BaseAttentionMethod):
    """
    CAB V4: Hybrid selection combining magnitude and FRC.
    
    This is our main contribution:
    - Reserve X% of budget for top-magnitude blocks (like H2O)
    - Reserve (1-X)% for top-FRC blocks (topological)
    
    Default: 50/50 split (magnitude_ratio=0.5)
    
    This combines the best of both approaches:
    - H2O's strength on high-attention semantic tokens
    - FRC's strength on unique structural connections
    """
    
    def __init__(self, config: MethodConfig):
        super().__init__(config)
        
        try:
            from cab_attention.kernels.coarsening import coarsen_qk_max_l2_pytorch
            from cab_attention.kernels.frc_kernel import compute_block_frc, generate_block_mask
            self.coarsen_fn = coarsen_qk_max_l2_pytorch
            self.frc_fn = compute_block_frc
            self.mask_fn = generate_block_mask
            self._cab_available = True
        except ImportError:
            self._cab_available = False
    
    def modify_attention_weights(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply CAB V4 hybrid selection."""
        if not self._cab_available:
            raise ImportError("CAB attention modules not available")
        
        B, H, N, D = query.shape
        block_size = self.config.block_size
        M = (N + block_size - 1) // block_size
        
        # Coarsen Q, K
        q_coarse, k_coarse = self.coarsen_fn(query, key, block_size)
        
        # Compute FRC scores
        frc_scores, affinity, redundancy = self.frc_fn(
            q_coarse, k_coarse,
            temperature=1.0,
            lambda_redundancy=self.config.lambda_redundancy,
            formula=self.config.formula,
            normalization=self.config.normalization,
        )
        
        # Compute magnitude scores from attention weights
        # Coarsen attention to block level
        magnitude_scores = self._coarsen_attention(attention_weights, block_size)
        
        # Generate hybrid mask
        block_mask = self.mask_fn(
            frc_scores,
            sparsity=self.config.sparsity,
            select_high=True,
            keep_diagonal=self.config.keep_diagonal,
            causal=self.config.causal,
            magnitude_scores=magnitude_scores,
            magnitude_ratio=self.config.magnitude_ratio,
        )
        
        # Expand to token level
        token_mask = self.expand_block_mask(block_mask, block_size, N)
        
        # Apply mask
        modified_weights = attention_weights.clone()
        modified_weights[~token_mask] = float('-inf')
        
        # Re-normalize
        modified_weights = F.softmax(modified_weights, dim=-1)
        modified_weights = torch.nan_to_num(modified_weights, nan=0.0)
        
        diagnostics = {
            'method': 'cab',
            'sparsity': self.compute_effective_sparsity(token_mask),
            'magnitude_ratio': self.config.magnitude_ratio,
            'frc_mean': frc_scores.mean().item(),
            'frc_std': frc_scores.std().item(),
            'magnitude_mean': magnitude_scores.mean().item(),
            'affinity_mean': affinity.mean().item(),
            'redundancy_mean': redundancy.mean().item(),
        }
        
        return modified_weights, diagnostics
    
    def get_block_mask(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Get CAB V4 hybrid block mask."""
        if not self._cab_available:
            raise ImportError("CAB attention modules not available")
        
        # Coarsen
        q_coarse, k_coarse = self.coarsen_fn(query, key, block_size)
        
        # Compute FRC
        frc_scores, _, _ = self.frc_fn(
            q_coarse, k_coarse,
            lambda_redundancy=self.config.lambda_redundancy,
            formula=self.config.formula,
            normalization=self.config.normalization,
        )
        
        # Compute magnitude scores
        magnitude_scores = self._coarsen_attention(attention_weights, block_size)
        
        # Generate hybrid mask
        return self.mask_fn(
            frc_scores,
            sparsity=self.config.sparsity,
            select_high=True,
            keep_diagonal=self.config.keep_diagonal,
            causal=self.config.causal,
            magnitude_scores=magnitude_scores,
            magnitude_ratio=self.config.magnitude_ratio,
        )
    
    def _coarsen_attention(
        self,
        attention_weights: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Coarsen attention to block level using max pooling."""
        B, H, N, _ = attention_weights.shape
        M = (N + block_size - 1) // block_size
        
        # Pad if necessary
        pad_size = M * block_size - N
        if pad_size > 0:
            attention_weights = F.pad(attention_weights, (0, pad_size, 0, pad_size), value=0)
        
        # Reshape and max pool
        attention_weights = attention_weights.view(B, H, M, block_size, M, block_size)
        block_weights = attention_weights.max(dim=3)[0].max(dim=-1)[0]
        
        return block_weights


# =============================================================================
# StreamingLLM
# =============================================================================

class StreamingLLMAttention(BaseAttentionMethod):
    """
    StreamingLLM: Attention Sinks + Recent Tokens.
    
    Paper: "Efficient Streaming Language Models with Attention Sinks"
    Authors: Xiao et al., 2023
    arXiv: https://arxiv.org/abs/2309.17453
    
    Algorithm (faithful to paper):
    - Keep first K tokens ("attention sinks" - usually 4 tokens)
    - Keep last W tokens (sliding window for recent context)
    - Evict all tokens in between
    
    Key insight: Initial tokens act as "sinks" that absorb attention mass,
    and are critical for model stability even if semantically unimportant.
    """
    
    def modify_attention_weights(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply StreamingLLM sink + window selection."""
        B, H, N, _ = attention_weights.shape
        
        num_sinks = self.config.num_sink_tokens
        window_size = self.config.window_size
        
        # Create mask: keep sinks + window
        mask = torch.zeros(N, N, dtype=torch.bool, device=attention_weights.device)
        
        # Keep attention to sink tokens (first num_sinks)
        mask[:, :num_sinks] = True
        
        # Keep attention to recent window (last window_size)
        if window_size > 0:
            mask[:, -window_size:] = True
        
        # Apply causal masking
        if self.config.causal:
            causal_mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=mask.device))
            mask = mask & causal_mask
        
        # Expand to batch and head dims
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        
        # Apply mask
        modified_weights = attention_weights.clone()
        modified_weights[~mask] = float('-inf')
        
        # Re-normalize
        modified_weights = F.softmax(modified_weights, dim=-1)
        modified_weights = torch.nan_to_num(modified_weights, nan=0.0)
        
        diagnostics = {
            'method': 'streaming_llm',
            'sparsity': self.compute_effective_sparsity(mask),
            'num_sinks': num_sinks,
            'window_size': window_size,
        }
        
        return modified_weights, diagnostics
    
    def get_block_mask(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Get StreamingLLM block mask."""
        B, H, N, _ = attention_weights.shape
        M = (N + block_size - 1) // block_size
        
        num_sink_blocks = (self.config.num_sink_tokens + block_size - 1) // block_size
        window_blocks = (self.config.window_size + block_size - 1) // block_size
        
        # Create block mask
        block_mask = torch.zeros(M, M, dtype=torch.bool, device=attention_weights.device)
        
        # Keep sink blocks
        block_mask[:, :num_sink_blocks] = True
        
        # Keep window blocks
        if window_blocks > 0:
            block_mask[:, -window_blocks:] = True
        
        # Expand to batch and head dims
        block_mask = block_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        
        return block_mask


# =============================================================================
# Local + Strided Attention
# =============================================================================

class LocalStridedAttention(BaseAttentionMethod):
    """
    Local + Strided Attention Pattern.
    
    Paper: "Generating Long Sequences with Sparse Transformers"
    Authors: Child et al., 2019
    arXiv: https://arxiv.org/abs/1904.10509
    
    Algorithm (faithful to paper):
    - Local window: Each query attends to nearby keys within window W
    - Strided pattern: Each query also attends to every S-th token globally
    - Union of both patterns determines the final sparse mask
    
    Key insight: Combining local and strided patterns captures both
    local dependencies and long-range periodic patterns.
    """
    
    def modify_attention_weights(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply local + strided pattern."""
        B, H, N, _ = attention_weights.shape
        
        local_window = self.config.local_window
        stride = self.config.stride
        
        # Create mask
        mask = torch.zeros(N, N, dtype=torch.bool, device=attention_weights.device)
        
        # Local window: each query attends to nearby keys
        for i in range(N):
            start = max(0, i - local_window // 2)
            end = min(N, i + local_window // 2 + 1)
            mask[i, start:end] = True
        
        # Strided: attend to every stride-th token
        for i in range(N):
            strided_indices = torch.arange(0, N, stride, device=mask.device)
            mask[i, strided_indices] = True
        
        # Apply causal masking
        if self.config.causal:
            causal_mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=mask.device))
            mask = mask & causal_mask
        
        # Expand to batch and head dims
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        
        # Apply mask
        modified_weights = attention_weights.clone()
        modified_weights[~mask] = float('-inf')
        
        # Re-normalize
        modified_weights = F.softmax(modified_weights, dim=-1)
        modified_weights = torch.nan_to_num(modified_weights, nan=0.0)
        
        diagnostics = {
            'method': 'local_strided',
            'sparsity': self.compute_effective_sparsity(mask),
            'local_window': local_window,
            'stride': stride,
        }
        
        return modified_weights, diagnostics
    
    def get_block_mask(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Get local + strided block mask."""
        B, H, N, _ = attention_weights.shape
        M = (N + block_size - 1) // block_size
        
        local_blocks = (self.config.local_window + block_size - 1) // block_size
        stride_blocks = max(1, self.config.stride // block_size)
        
        # Create block mask
        block_mask = torch.zeros(M, M, dtype=torch.bool, device=attention_weights.device)
        
        # Local blocks
        for i in range(M):
            start = max(0, i - local_blocks // 2)
            end = min(M, i + local_blocks // 2 + 1)
            block_mask[i, start:end] = True
        
        # Strided blocks
        for i in range(M):
            strided_indices = torch.arange(0, M, stride_blocks, device=block_mask.device)
            block_mask[i, strided_indices] = True
        
        # Expand to batch and head dims
        block_mask = block_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
        
        return block_mask


# =============================================================================
# Random Selection (Baseline)
# =============================================================================

class RandomAttention(BaseAttentionMethod):
    """
    Random Selection Baseline.
    
    Randomly selects which tokens/blocks to keep.
    This provides a lower bound - any structured method should beat this.
    """
    
    def __init__(self, config: MethodConfig, seed: int = 42):
        super().__init__(config)
        self.seed = seed
    
    def modify_attention_weights(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        layer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Apply random selection."""
        B, H, N, _ = attention_weights.shape
        
        # Set seed for reproducibility
        generator = torch.Generator(device=attention_weights.device)
        generator.manual_seed(self.seed + layer_idx)
        
        # Calculate number to keep
        k_keep = max(1, int(N * (1 - self.config.sparsity)))
        
        # Random selection per query
        mask = torch.zeros(B, H, N, N, dtype=torch.bool, device=attention_weights.device)
        
        for b in range(B):
            for h in range(H):
                for q in range(N):
                    # Randomly select k_keep keys to attend to
                    perm = torch.randperm(N, generator=generator, device=attention_weights.device)
                    mask[b, h, q, perm[:k_keep]] = True
        
        # Apply causal masking
        if self.config.causal:
            causal_mask = torch.tril(torch.ones(N, N, dtype=torch.bool, device=mask.device))
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            mask = mask & causal_mask
        
        # Apply mask
        modified_weights = attention_weights.clone()
        modified_weights[~mask] = float('-inf')
        
        # Re-normalize
        modified_weights = F.softmax(modified_weights, dim=-1)
        modified_weights = torch.nan_to_num(modified_weights, nan=0.0)
        
        diagnostics = {
            'method': 'random',
            'sparsity': self.compute_effective_sparsity(mask),
            'seed': self.seed,
        }
        
        return modified_weights, diagnostics
    
    def get_block_mask(
        self,
        attention_weights: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        block_size: int = 64,
    ) -> torch.Tensor:
        """Get random block mask."""
        B, H, N, _ = attention_weights.shape
        M = (N + block_size - 1) // block_size
        
        generator = torch.Generator(device=attention_weights.device)
        generator.manual_seed(self.seed)
        
        # Random block mask
        k_keep = max(1, int(M * M * (1 - self.config.sparsity)))
        
        block_mask = torch.zeros(B, H, M, M, dtype=torch.bool, device=attention_weights.device)
        
        for b in range(B):
            for h in range(H):
                # Randomly select k_keep blocks
                flat_indices = torch.randperm(M * M, generator=generator, device=attention_weights.device)[:k_keep]
                for idx in flat_indices:
                    i, j = idx // M, idx % M
                    block_mask[b, h, i, j] = True
        
        return block_mask


# =============================================================================
# Method Registry
# =============================================================================

class MethodRegistry:
    """Registry of available sparse attention methods."""
    
    _methods = {
        MethodName.DENSE: DenseAttention,
        MethodName.H2O: H2OAttention,
        MethodName.CAB: CABV4Attention,  # CAB uses CABV4's hybrid implementation
        MethodName.STREAMING_LLM: StreamingLLMAttention,
        MethodName.LOCAL_STRIDED: LocalStridedAttention,
        MethodName.RANDOM: RandomAttention,
    }
    
    @classmethod
    def get_method_class(cls, name: MethodName) -> type:
        """Get method class by name."""
        if name not in cls._methods:
            raise ValueError(f"Unknown method: {name}")
        return cls._methods[name]
    
    @classmethod
    def list_methods(cls) -> List[str]:
        """List all available methods."""
        return [m.value for m in cls._methods.keys()]
    
    @classmethod
    def create_method(cls, config: MethodConfig) -> BaseAttentionMethod:
        """Create method instance from config."""
        method_class = cls.get_method_class(config.name)
        return method_class(config)


def get_method(
    name: str,
    config: Optional[MethodConfig] = None,
    **kwargs
) -> BaseAttentionMethod:
    """
    Get attention method by name.
    
    Args:
        name: Method name (e.g., "cab", "h2o")
        config: Optional custom configuration
        **kwargs: Override config parameters
    
    Returns:
        BaseAttentionMethod instance
    
    Example:
        >>> method = get_method("cab", sparsity=0.95)
        >>> modified_attn, diag = method.modify_attention_weights(attn, q, k)
    """
    from .config import METHOD_CONFIGS
    
    # Get or create config
    if config is None:
        if name not in METHOD_CONFIGS:
            raise ValueError(f"Unknown method: {name}. Available: {list(METHOD_CONFIGS.keys())}")
        config = METHOD_CONFIGS[name]
    
    # Apply kwargs overrides
    if kwargs:
        from dataclasses import replace
        config = replace(config, **kwargs)
    
    return MethodRegistry.create_method(config)


# =============================================================================
# Utility Functions for Fair Comparison
# =============================================================================

def compare_methods(
    attention_weights: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    methods: List[str] = None,
    sparsity: float = 0.9,
) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple methods on the same attention weights.
    
    Args:
        attention_weights: Raw attention weights [B, H, N, N]
        query: Query tensor [B, H, N, D]
        key: Key tensor [B, H, N, D]
        methods: List of method names to compare
        sparsity: Target sparsity level
    
    Returns:
        Dict mapping method name to results
    """
    if methods is None:
        methods = ["dense", "h2o", "cab", "streaming_llm", "random"]
    
    results = {}
    
    for method_name in methods:
        method = get_method(method_name, sparsity=sparsity)
        modified_weights, diagnostics = method.modify_attention_weights(
            attention_weights.clone(), query, key
        )
        
        results[method_name] = {
            'modified_weights': modified_weights,
            'diagnostics': diagnostics,
            'block_mask': method.get_block_mask(attention_weights, query, key),
        }
    
    return results


if __name__ == "__main__":
    # Test methods
    print("Testing attention methods...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create test data
    B, H, N, D = 1, 8, 256, 64
    q = torch.randn(B, H, N, D, device=device)
    k = torch.randn(B, H, N, D, device=device)
    
    # Compute raw attention
    attn_weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D), dim=-1)
    
    # Test each method
    for method_name in MethodRegistry.list_methods():
        print(f"\nTesting {method_name}...")
        try:
            method = get_method(method_name, sparsity=0.9)
            modified, diag = method.modify_attention_weights(attn_weights.clone(), q, k)
            print(f"  Sparsity: {diag['sparsity']:.2%}")
            print(f"  Output shape: {modified.shape}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\nAll tests complete!")

