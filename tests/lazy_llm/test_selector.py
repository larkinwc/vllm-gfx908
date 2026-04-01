# SPDX-License-Identifier: Apache-2.0
"""Tests for TokenImportanceSelector."""

import torch
import pytest

from vllm.attention.lazy_llm.selector import TokenImportanceSelector


class TestComputeImportanceFromAttention:
    """Tests for attention-weight-based importance scoring."""

    def test_3d_attention_weights(self):
        num_heads, seq_len = 8, 100
        attn = torch.randn(num_heads, seq_len, seq_len).softmax(dim=-1)
        importance = TokenImportanceSelector.compute_importance_from_attention(attn)
        assert importance.shape == (seq_len,)
        assert (importance >= 0).all()

    def test_4d_attention_weights(self):
        batch, num_heads, seq_len = 1, 8, 100
        attn = torch.randn(batch, num_heads, seq_len, seq_len).softmax(dim=-1)
        importance = TokenImportanceSelector.compute_importance_from_attention(attn)
        assert importance.shape == (seq_len,)

    def test_uses_last_query_position(self):
        """Importance should reflect attention from the last query position."""
        num_heads, seq_len = 4, 10
        attn = torch.zeros(num_heads, seq_len, seq_len)
        # Make last query attend strongly to token 3
        attn[:, -1, 3] = 10.0
        attn = attn.softmax(dim=-1)
        importance = TokenImportanceSelector.compute_importance_from_attention(attn)
        assert importance[3] == importance.max()

    def test_invalid_dims(self):
        with pytest.raises(ValueError, match="Expected 3D or 4D"):
            TokenImportanceSelector.compute_importance_from_attention(
                torch.randn(100)
            )


class TestComputeImportanceFromQK:
    """Tests for Q*K^T based importance scoring."""

    def test_3d_format(self):
        seq_len, num_heads, head_dim = 100, 8, 64
        q = torch.randn(seq_len, num_heads, head_dim)
        k = torch.randn(seq_len, num_heads, head_dim)
        scale = head_dim ** -0.5
        importance = TokenImportanceSelector.compute_importance_from_qk(q, k, scale)
        assert importance.shape == (seq_len,)
        assert (importance >= 0).all()
        # Should sum to ~1 (softmax output averaged)
        assert abs(importance.sum().item() - 1.0) < 0.01

    def test_2d_format(self):
        seq_len, hidden = 100, 512
        q = torch.randn(seq_len, hidden)
        k = torch.randn(seq_len, hidden)
        scale = (hidden) ** -0.5
        importance = TokenImportanceSelector.compute_importance_from_qk(q, k, scale)
        assert importance.shape == (seq_len,)

    def test_gqa_format(self):
        """GQA: more query heads than key heads."""
        seq_len, num_q_heads, num_kv_heads, head_dim = 50, 32, 8, 64
        q = torch.randn(seq_len, num_q_heads, head_dim)
        k = torch.randn(seq_len, num_kv_heads, head_dim)
        scale = head_dim ** -0.5
        importance = TokenImportanceSelector.compute_importance_from_qk(q, k, scale)
        assert importance.shape == (seq_len,)

    def test_known_pattern(self):
        """Token with highest key alignment to last query should score highest."""
        seq_len, num_heads, head_dim = 20, 4, 32
        q = torch.zeros(seq_len, num_heads, head_dim)
        k = torch.zeros(seq_len, num_heads, head_dim)
        # Make last query and token 7 highly aligned
        q[-1, :, 0] = 10.0
        k[7, :, 0] = 10.0
        scale = head_dim ** -0.5
        importance = TokenImportanceSelector.compute_importance_from_qk(q, k, scale)
        assert importance[7] == importance.max()


class TestSelectTokens:
    """Tests for top-k token selection."""

    def test_basic_selection(self):
        importance = torch.tensor([0.1, 0.9, 0.3, 0.8, 0.2])
        kept = TokenImportanceSelector.select_tokens(importance, num_keep=3)
        assert kept.shape[0] == 3
        # Should keep indices 1, 3 (highest) and one of 2 or 0
        assert 1 in kept
        assert 3 in kept

    def test_sorted_output(self):
        importance = torch.randn(100)
        kept = TokenImportanceSelector.select_tokens(importance, num_keep=50)
        # Indices should be sorted
        assert (kept[1:] > kept[:-1]).all()

    def test_keep_all(self):
        importance = torch.randn(20)
        kept = TokenImportanceSelector.select_tokens(importance, num_keep=20)
        assert kept.shape[0] == 20
        assert (kept == torch.arange(20)).all()

    def test_keep_more_than_available(self):
        importance = torch.randn(10)
        kept = TokenImportanceSelector.select_tokens(importance, num_keep=100)
        assert kept.shape[0] == 10

    def test_protected_indices(self):
        importance = torch.tensor([0.01, 0.02, 0.03, 0.9, 0.8])
        protected = torch.tensor([0, 1])  # protect first 2
        kept = TokenImportanceSelector.select_tokens(
            importance, num_keep=3, protected_indices=protected,
        )
        assert 0 in kept
        assert 1 in kept
        assert kept.shape[0] == 3

    def test_empty_protected(self):
        importance = torch.randn(20)
        kept = TokenImportanceSelector.select_tokens(
            importance, num_keep=10, protected_indices=torch.tensor([]),
        )
        assert kept.shape[0] == 10
