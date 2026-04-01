# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for SnapKV token selection algorithm."""

import pytest
import torch

from vllm.attention.kv_compress.snapkv import SnapKVSelector


@pytest.fixture
def selector():
    return SnapKVSelector(
        observation_window_size=16,
        kernel_size=5,
        recent_window_size=8,
        sink_size=4,
    )


@pytest.fixture
def default_selector():
    return SnapKVSelector()


def _make_qkv(seq_len, num_heads_q, num_heads_kv, head_size, device="cpu"):
    """Create random Q, K, V tensors in vLLM format."""
    q = torch.randn(seq_len, num_heads_q, head_size, device=device)
    k = torch.randn(seq_len, num_heads_kv, head_size, device=device)
    v = torch.randn(seq_len, num_heads_kv, head_size, device=device)
    return q, k, v


class TestSnapKVImportanceScores:
    """Tests for importance score computation."""

    def test_basic_shape(self, selector):
        """Importance scores should have shape [num_kv_heads, seq_len]."""
        seq_len, num_heads_q, num_heads_kv, head_size = 128, 8, 8, 64
        q, k, _ = _make_qkv(seq_len, num_heads_q, num_heads_kv, head_size)
        scale = 1.0 / (head_size ** 0.5)

        scores = selector.compute_importance_scores(q, k, scale, seq_len)
        assert scores.shape == (num_heads_kv, seq_len)

    def test_gqa_shape(self, selector):
        """GQA: num_heads_q > num_heads_kv should still work."""
        seq_len, num_heads_q, num_heads_kv, head_size = 128, 32, 8, 64
        q, k, _ = _make_qkv(seq_len, num_heads_q, num_heads_kv, head_size)
        scale = 1.0 / (head_size ** 0.5)

        scores = selector.compute_importance_scores(q, k, scale, seq_len)
        assert scores.shape == (num_heads_kv, seq_len)

    def test_scores_non_negative(self, selector):
        """After softmax + pooling, scores should be non-negative."""
        seq_len, num_heads_q, num_heads_kv, head_size = 64, 8, 8, 64
        q, k, _ = _make_qkv(seq_len, num_heads_q, num_heads_kv, head_size)
        scale = 1.0 / (head_size ** 0.5)

        scores = selector.compute_importance_scores(q, k, scale, seq_len)
        assert (scores >= 0).all()

    def test_4d_input_format(self, selector):
        """Should handle [batch, heads, tokens, dim] format."""
        seq_len, num_heads_kv, head_size = 64, 8, 64
        q = torch.randn(1, num_heads_kv, seq_len, head_size)
        k = torch.randn(1, num_heads_kv, seq_len, head_size)
        scale = 1.0 / (head_size ** 0.5)

        scores = selector.compute_importance_scores(q, k, scale, seq_len)
        assert scores.shape == (num_heads_kv, seq_len)

    def test_short_sequence(self):
        """Sequences shorter than observation window should still work."""
        sel = SnapKVSelector(observation_window_size=32, kernel_size=3)
        seq_len = 16
        q, k, _ = _make_qkv(seq_len, 4, 4, 64)
        scale = 0.125
        scores = sel.compute_importance_scores(q, k, scale, seq_len)
        assert scores.shape == (4, seq_len)

    def test_concentrated_attention(self):
        """When a specific position has very high key norm, it should
        get high importance."""
        sel = SnapKVSelector(observation_window_size=8, kernel_size=1)
        seq_len, heads, dim = 64, 1, 32
        q = torch.randn(seq_len, heads, dim)
        k = torch.zeros(seq_len, heads, dim)
        # Make position 10 have very large key values
        k[10] = 10.0
        scale = 1.0 / (dim ** 0.5)

        scores = sel.compute_importance_scores(q, k, scale, seq_len)
        # Position 10 should have among the highest scores
        assert scores[0, 10] > scores[0].median()


class TestSnapKVTokenSelection:
    """Tests for token selection."""

    def test_mask_shape(self, selector):
        """Selection mask should be [num_kv_heads, seq_len]."""
        scores = torch.rand(8, 128)
        mask = selector.select_tokens(scores, budget_per_head=32, seq_len=128)
        assert mask.shape == (8, 128)
        assert mask.dtype == torch.bool

    def test_budget_respected(self, selector):
        """Number of selected tokens per head should not exceed budget."""
        scores = torch.rand(8, 256)
        budget = 64
        mask = selector.select_tokens(scores, budget, seq_len=256)
        per_head_count = mask.sum(dim=1)
        assert (per_head_count <= budget).all()

    def test_sinks_always_retained(self, selector):
        """First `sink_size` positions should always be retained."""
        scores = torch.zeros(4, 128)  # All zero scores
        mask = selector.select_tokens(scores, budget_per_head=20, seq_len=128)
        assert mask[:, :selector.sink_size].all()

    def test_recent_always_retained(self, selector):
        """Last `recent_window_size` positions should always be retained."""
        scores = torch.zeros(4, 128)
        mask = selector.select_tokens(scores, budget_per_head=20, seq_len=128)
        assert mask[:, -selector.recent_window_size:].all()

    def test_high_score_selected(self, selector):
        """Positions with highest scores should be selected."""
        num_heads, seq_len = 2, 128
        scores = torch.zeros(num_heads, seq_len)
        # Place high scores at specific positions (outside sinks/recent)
        high_positions = [20, 30, 50, 70, 90]
        for p in high_positions:
            scores[:, p] = 10.0

        mask = selector.select_tokens(
            scores, budget_per_head=20, seq_len=seq_len
        )
        for p in high_positions:
            assert mask[:, p].all(), f"Position {p} should be selected"

    def test_per_head_budget(self, selector):
        """Different budgets per head should work."""
        num_heads, seq_len = 4, 128
        scores = torch.rand(num_heads, seq_len)
        budgets = torch.tensor([20, 30, 40, 50])
        mask = selector.select_tokens(scores, budgets, seq_len=seq_len)

        for h in range(num_heads):
            assert mask[h].sum() <= budgets[h] + 1  # +1 for rounding

    def test_budget_exceeds_seq_len(self, selector):
        """If budget > seq_len, all positions should be selected."""
        scores = torch.rand(4, 32)
        mask = selector.select_tokens(scores, budget_per_head=100, seq_len=32)
        assert mask.all()

    def test_batched_matches_sequential(self, selector):
        """Batched selection should produce same results as sequential."""
        num_heads, seq_len = 4, 128
        scores = torch.rand(num_heads, seq_len)
        budget = 40

        mask_seq = selector.select_tokens(scores, budget, seq_len)
        mask_bat = selector.select_tokens_batched(scores, budget, seq_len)

        # Both should have same mandatory positions
        assert (mask_seq[:, :selector.sink_size]
                == mask_bat[:, :selector.sink_size]).all()
        assert (mask_seq[:, -selector.recent_window_size:]
                == mask_bat[:, -selector.recent_window_size:]).all()

        # Selection counts should be similar (exact match not guaranteed
        # due to different tie-breaking)
        count_seq = mask_seq.sum(dim=1)
        count_bat = mask_bat.sum(dim=1)
        assert torch.allclose(
            count_seq.float(), count_bat.float(), atol=2
        )


class TestSnapKVCompress:
    """Tests for the full compress pipeline."""

    def test_compress_basic(self, selector):
        """Basic compression should produce smaller KV tensors."""
        seq_len, num_heads_q, num_heads_kv, head_size = 256, 8, 8, 64
        q, k, v = _make_qkv(seq_len, num_heads_q, num_heads_kv, head_size)
        scale = 1.0 / (head_size ** 0.5)

        ck, cv, mask = selector.compress(q, k, v, scale, budget_per_head=64)
        assert ck.shape[0] < seq_len
        assert cv.shape[0] < seq_len
        assert ck.shape[1:] == k.shape[1:]
        assert cv.shape[1:] == v.shape[1:]
        assert mask.shape == (num_heads_kv, seq_len)

    def test_compress_preserves_values(self, selector):
        """Compressed KV values should be exact copies from original."""
        seq_len, num_heads_kv, head_size = 128, 4, 32
        q, k, v = _make_qkv(seq_len, 4, num_heads_kv, head_size)
        scale = 0.125

        ck, cv, mask = selector.compress(q, k, v, scale, budget_per_head=32)
        union_mask = mask.any(dim=0)
        selected = union_mask.nonzero(as_tuple=True)[0]

        assert torch.equal(ck, k[selected])
        assert torch.equal(cv, v[selected])

    def test_no_compression_when_budget_large(self, selector):
        """When budget >= seq_len, output should equal input."""
        seq_len = 32
        q, k, v = _make_qkv(seq_len, 4, 4, 64)
        scale = 0.125

        ck, cv, mask = selector.compress(q, k, v, scale, budget_per_head=64)
        assert ck.shape[0] == seq_len
        assert cv.shape[0] == seq_len


class TestSnapKVEdgeCases:
    """Edge case tests."""

    def test_single_token(self):
        sel = SnapKVSelector(
            observation_window_size=1,
            kernel_size=1,
            recent_window_size=1,
            sink_size=1,
        )
        q, k, v = _make_qkv(1, 4, 4, 64)
        scale = 0.125
        ck, cv, mask = sel.compress(q, k, v, scale, budget_per_head=1)
        assert ck.shape[0] == 1

    def test_two_tokens(self):
        sel = SnapKVSelector(
            observation_window_size=1,
            kernel_size=1,
            recent_window_size=1,
            sink_size=1,
        )
        q, k, v = _make_qkv(2, 4, 4, 64)
        scale = 0.125
        ck, cv, mask = sel.compress(q, k, v, scale, budget_per_head=2)
        assert ck.shape[0] == 2

    def test_deterministic(self, selector):
        """Same input should produce same output."""
        torch.manual_seed(42)
        q, k, v = _make_qkv(128, 8, 8, 64)
        scale = 0.125

        _, _, mask1 = selector.compress(q, k, v, scale, budget_per_head=32)
        _, _, mask2 = selector.compress(q, k, v, scale, budget_per_head=32)
        assert torch.equal(mask1, mask2)
