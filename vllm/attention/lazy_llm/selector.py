# SPDX-License-Identifier: Apache-2.0
"""Token importance scoring and selection for LazyLLM.

Uses attention scores from the current layer to determine which tokens
are important for next-token prediction, then selects top-k percentile.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class TokenImportanceSelector:
    """Computes per-token importance from attention maps and selects top-k.

    The importance score for token t_i at layer l is defined as:
        s_i^l = (1/H) * sum_h A_{h,N,i}^l
    i.e., the mean attention weight that the *last* token (next-token predictor)
    assigns to token t_i, averaged across all heads.

    This follows the LazyLLM paper (Fu et al., 2024) Section 3.2.
    """

    @staticmethod
    def compute_importance_from_attention(
        attention_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-token importance from attention weights.

        Args:
            attention_weights: Attention probabilities of shape
                [num_heads, seq_len, seq_len] or [batch, num_heads, seq_len, seq_len].
                The last query position's attention over all keys is used.

        Returns:
            Importance scores of shape [seq_len] — higher means more important.
        """
        if attention_weights.dim() == 4:
            # [batch, heads, seq, seq] → take last query pos, mean over heads
            # Use batch=0 (single sequence during prefill)
            scores = attention_weights[0, :, -1, :]  # [heads, seq_len]
        elif attention_weights.dim() == 3:
            # [heads, seq, seq] → take last query pos
            scores = attention_weights[:, -1, :]  # [heads, seq_len]
        else:
            raise ValueError(
                f"Expected 3D or 4D attention_weights, got {attention_weights.dim()}D"
            )

        # Mean across heads → [seq_len]
        importance = scores.mean(dim=0)
        return importance

    @staticmethod
    def compute_importance_from_qk(
        query: torch.Tensor,
        key: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Compute importance by doing a lightweight Q*K^T for the last query only.

        This avoids materializing the full attention matrix.
        Only computes attention for the last query position.

        Args:
            query: [seq_len, num_heads, head_dim] or [num_tokens, num_heads * head_dim]
            key: [seq_len, num_kv_heads, head_dim] or [num_tokens, num_kv_heads * head_dim]
            scale: Attention scaling factor (1/sqrt(head_dim)).

        Returns:
            Importance scores of shape [seq_len].
        """
        if query.dim() == 2:
            # Flat format [num_tokens, num_heads * head_dim]
            # We need to know head_dim to reshape. Infer from key if possible.
            # For safety, fall back to treating the whole thing as one "head"
            q_last = query[-1:, :]  # [1, hidden]
            k_all = key  # [seq_len, hidden]
            # Simple dot product attention
            scores = torch.matmul(q_last, k_all.t()) * scale  # [1, seq_len]
            scores = F.softmax(scores, dim=-1)
            return scores.squeeze(0)

        # 3D format: [seq_len, num_heads, head_dim]
        q_last = query[-1:, :, :]  # [1, num_heads, head_dim]
        num_heads_q = query.shape[1]
        num_heads_k = key.shape[1]

        if num_heads_q != num_heads_k:
            # GQA: expand key heads to match query heads
            repeat_factor = num_heads_q // num_heads_k
            k_expanded = key.repeat_interleave(repeat_factor, dim=1)
        else:
            k_expanded = key

        # [1, num_heads, head_dim] x [seq_len, num_heads, head_dim]^T
        # → per-head dot product
        # Efficient: einsum('nhd,shd->nh', q_last, k_expanded) → [num_heads, seq_len]
        scores = torch.einsum(
            "qhd,shd->hs", q_last, k_expanded
        ) * scale  # [num_heads, seq_len]

        # Softmax per head, then mean across heads
        attn_weights = F.softmax(scores, dim=-1)  # [num_heads, seq_len]
        importance = attn_weights.mean(dim=0)  # [seq_len]
        return importance

    @staticmethod
    def select_tokens(
        importance: torch.Tensor,
        num_keep: int,
        protected_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Select the top-k most important token indices.

        Args:
            importance: Per-token importance scores [seq_len].
            num_keep: Number of tokens to keep.
            protected_indices: Indices of tokens that must always be kept
                (e.g., BOS, system prompt). Shape [num_protected].

        Returns:
            Sorted indices of tokens to keep, shape [num_keep].
        """
        seq_len = importance.shape[0]
        num_keep = min(num_keep, seq_len)

        if num_keep >= seq_len:
            return torch.arange(seq_len, device=importance.device)

        if protected_indices is not None and protected_indices.numel() > 0:
            # Set protected tokens to very high importance so they're always kept
            importance = importance.clone()
            importance[protected_indices] = importance.max() + 1.0

        # Select top-k by importance
        _, top_indices = torch.topk(importance, num_keep, sorted=False)

        # Sort indices to maintain original token order (important for causal attention)
        top_indices, _ = torch.sort(top_indices)
        return top_indices
