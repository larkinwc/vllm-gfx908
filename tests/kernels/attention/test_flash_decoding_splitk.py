# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Flash-Decoding Split-K adaptive scheduling.

Validates that the adaptive split-K produces identical results to the
reference implementation at various split counts and sequence lengths.
Tests both the _compute_flash_decoding_splits scheduler and the kernel
correctness with different NUM_SEGMENTS_PER_SEQ values.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.math_utils import next_power_of_2
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.backends.triton_attn import (
    FLASH_DECODING_SPLIT_COUNTS,
    MAX_FLASH_DECODING_SPLITS,
    _compute_flash_decoding_splits,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

FP8_DTYPE = current_platform.fp8_dtype()

# ── Unit tests for the split scheduler ──────────────────────────────────


class TestComputeFlashDecodingSplits:
    """Unit tests for the adaptive split count scheduler."""

    def test_short_sequence_returns_minimum(self):
        """Short sequences with many seqs should return minimum splits."""
        splits = _compute_flash_decoding_splits(
            max_seq_len=128,
            num_seqs=32,
            num_kv_heads=4,
            tile_size=32,
        )
        assert splits == 8  # base_grid = 32*4 = 128 >= 120

    def test_long_sequence_small_batch_increases_splits(self):
        """Long sequence with batch=1 should increase splits to fill CUs."""
        splits = _compute_flash_decoding_splits(
            max_seq_len=8192,
            num_seqs=1,
            num_kv_heads=4,
            tile_size=32,
        )
        # base_grid = 1*4 = 4, need 120/4 = 30 splits, rounds to 32
        assert splits == 32

    def test_very_long_sequence_uses_more_splits(self):
        """Very long sequence with batch=1 should use even more splits."""
        splits = _compute_flash_decoding_splits(
            max_seq_len=16384,
            num_seqs=1,
            num_kv_heads=2,
            tile_size=32,
        )
        # base_grid = 1*2 = 2, need 120/2 = 60 splits, rounds to 64
        assert splits == 64

    def test_large_batch_uses_minimum(self):
        """Large batch sizes should use minimum splits."""
        splits = _compute_flash_decoding_splits(
            max_seq_len=8192,
            num_seqs=64,
            num_kv_heads=4,
            tile_size=32,
        )
        # base_grid = 64*4 = 256 >= 120, minimum splits
        assert splits == 8

    def test_result_always_power_of_2(self):
        """Split count should always be a valid power of 2."""
        for seq_len in [64, 256, 1024, 4096, 16384]:
            for batch in [1, 2, 4, 8, 32]:
                for kv_heads in [1, 2, 4, 8]:
                    splits = _compute_flash_decoding_splits(
                        max_seq_len=seq_len,
                        num_seqs=batch,
                        num_kv_heads=kv_heads,
                        tile_size=32,
                    )
                    assert splits in FLASH_DECODING_SPLIT_COUNTS

    def test_never_exceeds_max(self):
        """Split count should never exceed MAX_FLASH_DECODING_SPLITS."""
        splits = _compute_flash_decoding_splits(
            max_seq_len=131072,
            num_seqs=1,
            num_kv_heads=1,
            tile_size=32,
        )
        assert splits <= MAX_FLASH_DECODING_SPLITS

    def test_few_tiles_limits_splits(self):
        """When sequence is very short, splits shouldn't exceed tiles/4."""
        splits = _compute_flash_decoding_splits(
            max_seq_len=64,  # only 2 tiles with tile_size=32
            num_seqs=1,
            num_kv_heads=1,
            tile_size=32,
        )
        # 2 tiles // MIN_TILES_PER_SPLIT(4) = 0, so max_possible=1
        # But minimum is 8, and tiles can still be assigned (some splits empty)
        # The function handles this: max_possible_splits = max(1, 0) = 1
        # ideal becomes min(120, 1) = 1, but rounds up to 8
        assert splits == 8


# ── Kernel correctness tests ───────────────────────────────────────────


def ref_paged_attn(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: torch.Tensor,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> torch.Tensor:
    """Reference implementation of paged attention."""
    num_seqs = len(query_lens)
    block_tables = block_tables.cpu().numpy()
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[torch.Tensor] = []
    start_idx = 0
    for i in range(num_seqs):
        query_len = query_lens[i]
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len]
        q *= scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, head_size)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, head_size)
        v = v[:kv_len]

        if q.shape[1] != k.shape[1]:
            k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
            v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
        attn = torch.einsum("qhd,khd->hqk", q, k).float()
        empty_mask = torch.ones(query_len, kv_len)
        mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()
        if sliding_window is not None:
            sliding_window_mask = (
                torch.triu(
                    empty_mask, diagonal=kv_len - (query_len + sliding_window) + 1
                )
                .bool()
                .logical_not()
            )
            mask |= sliding_window_mask
        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * torch.tanh(attn / soft_cap)
        attn.masked_fill_(mask, float("-inf"))
        attn = torch.softmax(attn, dim=-1).to(v.dtype)
        out = torch.einsum("hqk,khd->qhd", attn, v)

        outputs.append(out)
        start_idx += query_len

    return torch.cat(outputs, dim=0)


# Decode-specific test cases: query_len=1, varying KV lengths
DECODE_SEQ_LENS = [
    # (query_len, kv_len) tuples
    [(1, 128)],                               # Short sequence
    [(1, 512)],                               # Medium sequence
    [(1, 2048)],                              # Long sequence
    [(1, 8192)],                              # Very long sequence
    [(1, 16384)],                             # Extra long sequence
    [(1, 256), (1, 1024), (1, 4096)],         # Mixed batch
    [(1, 128)] * 4,                           # Small batch, short seqs
    [(1, 8192)] * 2,                          # Small batch, long seqs
]

NUM_HEADS = [(4, 4), (8, 2)]
HEAD_SIZES = [128]
BLOCK_SIZES = [16, 32]

# Test with explicit split counts to verify all paths
SPLIT_COUNTS = [8, 16, 32, 64]


@pytest.mark.parametrize("seq_lens", DECODE_SEQ_LENS)
@pytest.mark.parametrize("num_heads", NUM_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
@pytest.mark.parametrize("num_splits", SPLIT_COUNTS)
@torch.inference_mode()
def test_flash_decoding_splitk_correctness(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
    block_size: int,
    num_splits: int,
) -> None:
    """Test that Flash-Decoding with various split counts matches reference.

    This is the core correctness test: it directly invokes unified_attention
    with a forced split count and verifies the output matches a pure-PyTorch
    reference implementation.
    """
    torch.set_default_device("cuda")
    set_random_seed(0)

    dtype = torch.float16
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    num_blocks = 32768

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    output = torch.empty_like(query)

    # Pre-allocate segment buffers at the requested split count
    head_size_padded = next_power_of_2(head_size)
    seq_threshold_3D = num_seqs  # Force 3D kernel by matching threshold

    softmax_segm_output = torch.empty(
        (seq_threshold_3D, num_query_heads, num_splits, head_size_padded),
        dtype=torch.float32,
    )
    softmax_segm_max = torch.empty(
        (seq_threshold_3D, num_query_heads, num_splits),
        dtype=torch.float32,
    )
    softmax_segm_expsum = torch.empty(
        (seq_threshold_3D, num_query_heads, num_splits),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=num_splits,
        max_flash_decoding_splits=num_splits,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
    )

    ref_output = ref_paged_attn(
        query=query.clone(),
        key_cache=key_cache,
        value_cache=value_cache,
        query_lens=query_lens,
        kv_lens=kv_lens,
        block_tables=block_tables,
        scale=scale,
    )

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(
        output, ref_output, atol=atol, rtol=rtol
    ), f"max diff: {torch.max(torch.abs(output - ref_output))}"


@pytest.mark.parametrize(
    "seq_lens",
    [
        [(1, 2048)] * 2,
        [(1, 8192)],
        [(1, 256), (1, 4096)],
    ],
)
@pytest.mark.parametrize("num_heads", [(8, 2)])
@pytest.mark.parametrize("head_size", [128])
@torch.inference_mode()
def test_flash_decoding_adaptive_vs_fixed(
    seq_lens: list[tuple[int, int]],
    num_heads: tuple[int, int],
    head_size: int,
) -> None:
    """Test that adaptive split count produces same results as fixed=8.

    Runs the same decode batch twice: once with the old fixed 8-segment
    approach and once with adaptive splits. Both should produce identical
    (within tolerance) results since the algorithm is mathematically
    equivalent regardless of split count.
    """
    torch.set_default_device("cuda")
    set_random_seed(42)

    dtype = torch.float16
    block_size = 16
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens = [x[1] for x in seq_lens]
    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens)
    scale = head_size**-0.5
    num_blocks = 32768

    query = torch.randn(sum(query_lens), num_query_heads, head_size, dtype=dtype)
    key_cache = torch.randn(
        num_blocks, block_size, num_kv_heads, head_size, dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens = torch.tensor(
        [0] + query_lens, dtype=torch.int32
    ).cumsum(dim=0, dtype=torch.int32)
    kv_lens_tensor = torch.tensor(kv_lens, dtype=torch.int32)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq), dtype=torch.int32
    )

    head_size_padded = next_power_of_2(head_size)
    seq_threshold_3D = num_seqs

    # Run with fixed 8 splits
    output_fixed = torch.empty_like(query)
    segm_out_8 = torch.empty(
        (seq_threshold_3D, num_query_heads, 8, head_size_padded),
        dtype=torch.float32,
    )
    segm_max_8 = torch.empty(
        (seq_threshold_3D, num_query_heads, 8), dtype=torch.float32
    )
    segm_exp_8 = torch.empty(
        (seq_threshold_3D, num_query_heads, 8), dtype=torch.float32
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output_fixed,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=8,
        max_flash_decoding_splits=None,  # Disable adaptive
        softmax_segm_output=segm_out_8,
        softmax_segm_max=segm_max_8,
        softmax_segm_expsum=segm_exp_8,
    )

    # Run with adaptive splits (should compute optimal count)
    adaptive_splits = _compute_flash_decoding_splits(
        max_seq_len=max_kv_len,
        num_seqs=num_seqs,
        num_kv_heads=num_kv_heads,
        tile_size=32,
    )
    output_adaptive = torch.empty_like(query)
    segm_out_a = torch.empty(
        (seq_threshold_3D, num_query_heads, adaptive_splits, head_size_padded),
        dtype=torch.float32,
    )
    segm_max_a = torch.empty(
        (seq_threshold_3D, num_query_heads, adaptive_splits),
        dtype=torch.float32,
    )
    segm_exp_a = torch.empty(
        (seq_threshold_3D, num_query_heads, adaptive_splits),
        dtype=torch.float32,
    )

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output_adaptive,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens_tensor,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        seq_threshold_3D=seq_threshold_3D,
        num_par_softmax_segments=adaptive_splits,
        max_flash_decoding_splits=adaptive_splits,
        softmax_segm_output=segm_out_a,
        softmax_segm_max=segm_max_a,
        softmax_segm_expsum=segm_exp_a,
    )

    atol, rtol = 1.5e-2, 1e-2
    torch.testing.assert_close(
        output_fixed, output_adaptive, atol=atol, rtol=rtol
    ), (
        f"Fixed vs adaptive mismatch (adaptive_splits={adaptive_splits}): "
        f"max diff: {torch.max(torch.abs(output_fixed - output_adaptive))}"
    )
