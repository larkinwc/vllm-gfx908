# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SnapKV: Efficient KV Cache Compression via Token Selection.

Implements the SnapKV algorithm (Li et al., NeurIPS 2024) which identifies
critical KV positions at prefill time using an observation window at the
end of the prompt. One-shot compression with zero decode overhead.

Algorithm:
    1. Extract observation window (last w tokens) queries
    2. Compute attention scores: obs_Q @ all_K^T
    3. Apply 1D average pooling along sequence dim to cluster nearby tokens
    4. Select top-b positions per head (highest pooled attention scores)
    5. Always retain sink tokens + recent window tokens
    6. Return indices of selected KV positions per head

Reference: https://arxiv.org/abs/2404.14469
"""

import torch
import torch.nn.functional as F


class SnapKVSelector:
    """Selects important KV cache positions using the SnapKV algorithm.

    The selector uses attention scores from an observation window at the
    end of the prompt to identify which KV positions are most critical.
    Nearby important positions are clustered via 1D average pooling to
    maintain spatial coherence.

    Args:
        observation_window_size: Number of tokens at prompt end to use
            as the observation window. Default 32.
        kernel_size: Pooling kernel size for clustering. Must be odd.
            Default 7.
        recent_window_size: Number of most recent tokens always retained.
            Default 32.
        sink_size: Number of initial tokens always retained (attention
            sinks). Default 4.
    """

    def __init__(
        self,
        observation_window_size: int = 32,
        kernel_size: int = 7,
        recent_window_size: int = 32,
        sink_size: int = 4,
    ):
        self.observation_window_size = observation_window_size
        self.kernel_size = kernel_size
        self.recent_window_size = recent_window_size
        self.sink_size = sink_size

    def compute_importance_scores(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        scale: float,
        seq_len: int,
    ) -> torch.Tensor:
        """Compute per-position importance scores using observation window.

        Uses the last `observation_window_size` query tokens to score all
        key positions. Applies 1D average pooling to cluster nearby tokens.

        Args:
            query: Query tensor, shape [num_tokens, num_heads, head_size]
                or [batch, num_heads, num_tokens, head_size].
            key: Key tensor, shape [num_tokens, num_kv_heads, head_size]
                or [batch, num_kv_heads, num_tokens, head_size].
            scale: Attention scaling factor (1/sqrt(head_size)).
            seq_len: Total sequence length.

        Returns:
            Importance scores per head per position.
            Shape: [num_kv_heads, seq_len] (pooled attention scores).
        """
        # Handle both vLLM format [tokens, heads, dim] and standard
        # [batch, heads, tokens, dim]
        if query.dim() == 3:
            # [tokens, heads, dim] -> [1, heads, tokens, dim]
            query = query.unsqueeze(0).transpose(1, 2)
            key = key.unsqueeze(0).transpose(1, 2)
        elif query.dim() == 4:
            pass  # Already [batch, heads, tokens, dim]
        else:
            raise ValueError(
                f"Expected query with 3 or 4 dims, got {query.dim()}"
            )

        num_heads_q = query.shape[1]
        num_heads_kv = key.shape[1]
        num_queries_per_kv = num_heads_q // num_heads_kv

        # Extract observation window: last w query tokens
        obs_window = min(self.observation_window_size, seq_len)
        obs_queries = query[:, :, -obs_window:, :]  # [B, Hq, w, D]

        # Compute attention scores: obs_Q @ K^T
        # For GQA: group query heads to match KV heads
        if num_queries_per_kv > 1:
            # Reshape to [B, Hkv, groups, w, D] then mean over groups
            obs_q_grouped = obs_queries.reshape(
                obs_queries.shape[0],
                num_heads_kv,
                num_queries_per_kv,
                obs_window,
                obs_queries.shape[-1],
            )
            # Average attention across query heads in each KV group
            obs_q_for_score = obs_q_grouped.mean(dim=2)  # [B, Hkv, w, D]
        else:
            obs_q_for_score = obs_queries  # [B, Hkv, w, D]

        # Attention scores: [B, Hkv, w, seq_len]
        attn_scores = torch.matmul(
            obs_q_for_score, key.transpose(-2, -1)
        ) * scale

        # Apply causal mask: observation window tokens can only attend
        # to positions up to their own position
        obs_start = seq_len - obs_window
        causal_mask = torch.arange(seq_len, device=attn_scores.device)
        obs_positions = torch.arange(
            obs_start, seq_len, device=attn_scores.device
        )
        # [w, seq_len]: True where obs token i can attend to position j
        valid_mask = causal_mask.unsqueeze(0) <= obs_positions.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(
            ~valid_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, Hkv, w, S]

        # Aggregate across observation window tokens: sum attention each
        # position received from all observation tokens
        # Shape: [B, Hkv, seq_len]
        position_importance = attn_weights.sum(dim=2)

        # Apply 1D average pooling to cluster nearby important positions
        # This is a key insight from SnapKV: important tokens often appear
        # in clusters, and pooling prevents selecting redundant nearby tokens
        if self.kernel_size > 1 and seq_len > self.kernel_size:
            padding = self.kernel_size // 2
            # Pool along sequence dimension: [B, Hkv, seq_len]
            pooled = F.avg_pool1d(
                position_importance,
                kernel_size=self.kernel_size,
                stride=1,
                padding=padding,
            )
        else:
            pooled = position_importance

        # Return [Hkv, seq_len] (squeeze batch dim for single-sequence)
        return pooled.squeeze(0)

    def select_tokens(
        self,
        importance_scores: torch.Tensor,
        budget_per_head: int | torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Select which token positions to retain per head.

        Always includes sink tokens and recent window tokens. Remaining
        budget is filled by top importance score positions.

        Args:
            importance_scores: Shape [num_kv_heads, seq_len].
            budget_per_head: Number of tokens to retain per head.
                Can be a scalar (same for all heads) or tensor
                [num_kv_heads] for per-head budgets.
            seq_len: Total sequence length.

        Returns:
            Boolean mask of shape [num_kv_heads, seq_len] where True
            means the position is retained.
        """
        num_kv_heads = importance_scores.shape[0]
        device = importance_scores.device

        # Create output mask
        mask = torch.zeros(
            num_kv_heads, seq_len, dtype=torch.bool, device=device
        )

        # Always retain sink tokens (first `sink_size` positions)
        actual_sinks = min(self.sink_size, seq_len)
        if actual_sinks > 0:
            mask[:, :actual_sinks] = True

        # Always retain recent window (last `recent_window_size` positions)
        actual_recent = min(self.recent_window_size, seq_len)
        if actual_recent > 0:
            mask[:, -actual_recent:] = True

        # Count already-selected positions per head
        already_selected = mask.sum(dim=1)  # [num_kv_heads]

        # Handle per-head vs uniform budget
        if isinstance(budget_per_head, int):
            budgets = torch.full(
                (num_kv_heads,), budget_per_head,
                dtype=torch.long, device=device,
            )
        else:
            budgets = budget_per_head.to(device=device, dtype=torch.long)

        # Remaining budget per head after mandatory selections
        remaining = (budgets - already_selected).clamp(min=0)

        # Determine the "selectable" region: between sinks and recent window
        selectable_start = actual_sinks
        selectable_end = seq_len - actual_recent

        if selectable_end > selectable_start:
            # Get scores only for selectable positions
            selectable_scores = importance_scores[
                :, selectable_start:selectable_end
            ]  # [Hkv, selectable_len]

            selectable_len = selectable_end - selectable_start

            # For each head, select top-k from selectable region
            for h in range(num_kv_heads):
                k = min(remaining[h].item(), selectable_len)
                if k > 0:
                    _, top_indices = torch.topk(
                        selectable_scores[h], k=int(k)
                    )
                    # Map back to original positions
                    mask[h, top_indices + selectable_start] = True

        return mask

    def select_tokens_batched(
        self,
        importance_scores: torch.Tensor,
        budget_per_head: int | torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """Vectorized token selection (no Python loop over heads).

        Same semantics as select_tokens but uses scatter ops for speed.

        Args:
            importance_scores: Shape [num_kv_heads, seq_len].
            budget_per_head: Number of tokens to retain per head.
            seq_len: Total sequence length.

        Returns:
            Boolean mask of shape [num_kv_heads, seq_len].
        """
        num_kv_heads = importance_scores.shape[0]
        device = importance_scores.device

        mask = torch.zeros(
            num_kv_heads, seq_len, dtype=torch.bool, device=device
        )

        actual_sinks = min(self.sink_size, seq_len)
        actual_recent = min(self.recent_window_size, seq_len)

        if actual_sinks > 0:
            mask[:, :actual_sinks] = True
        if actual_recent > 0:
            mask[:, -actual_recent:] = True

        already_selected = mask.sum(dim=1)

        if isinstance(budget_per_head, int):
            budgets = torch.full(
                (num_kv_heads,), budget_per_head,
                dtype=torch.long, device=device,
            )
        else:
            budgets = budget_per_head.to(device=device, dtype=torch.long)

        remaining = (budgets - already_selected).clamp(min=0)
        max_remaining = remaining.max().item()

        selectable_start = actual_sinks
        selectable_end = seq_len - actual_recent

        if selectable_end > selectable_start and max_remaining > 0:
            selectable_scores = importance_scores[
                :, selectable_start:selectable_end
            ]
            selectable_len = selectable_end - selectable_start
            k = min(int(max_remaining), selectable_len)

            if k > 0:
                _, top_indices = torch.topk(
                    selectable_scores, k=k, dim=1
                )  # [Hkv, k]

                # Mask out excess selections for heads with smaller budgets
                col_range = torch.arange(k, device=device).unsqueeze(0)
                valid = col_range < remaining.unsqueeze(1)
                top_indices_global = top_indices + selectable_start

                # Scatter True into mask
                top_indices_clamped = top_indices_global.clamp(
                    0, seq_len - 1
                )
                # Set invalid indices to a position already True (sinks)
                top_indices_clamped[~valid] = 0
                mask.scatter_(1, top_indices_clamped, valid)

        return mask

    def compress(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale: float,
        budget_per_head: int | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full SnapKV compression: score, select, and compact.

        Args:
            query: [num_tokens, num_heads, head_size]
            key: [num_tokens, num_kv_heads, head_size]
            value: [num_tokens, num_kv_heads, head_size]
            scale: Attention scale factor.
            budget_per_head: Tokens to retain per head.

        Returns:
            Tuple of (compressed_key, compressed_value, selection_mask):
                - compressed_key: [max_selected, num_kv_heads, head_size]
                - compressed_value: [max_selected, num_kv_heads, head_size]
                - selection_mask: [num_kv_heads, seq_len] boolean mask
        """
        seq_len = key.shape[0]

        # Step 1: Compute importance scores
        importance = self.compute_importance_scores(
            query, key, scale, seq_len
        )

        # Step 2: Select tokens
        mask = self.select_tokens_batched(
            importance, budget_per_head, seq_len
        )

        # Step 3: Compact KV using union of all heads' selections
        # For paged attention compatibility, we use the union mask
        union_mask = mask.any(dim=0)  # [seq_len]
        selected_indices = union_mask.nonzero(as_tuple=True)[0]

        compressed_key = key[selected_indices]    # [N_sel, Hkv, D]
        compressed_value = value[selected_indices]  # [N_sel, Hkv, D]

        return compressed_key, compressed_value, mask
