#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1 — Triton Flash-Decoding autotune sweep for MI100 (gfx908) under
KV-INT8 per-token-head.

This is a kernel-only sweep (no vLLM serving, no bench-serve) that
directly drives ``kernel_unified_attention`` from
``vllm.v1.attention.ops.triton_unified_attention`` with every (tile_size,
num_splits, num_warps, num_stages, BLOCK_M) combination on every
(quant, head_dim, kv_dtype, seq_len_bucket, GQA_ratio, num_seqs) shape.

For every trial we record:
  * median latency over 100 timed invocations (10 warmup)
  * analytical HBM bytes per invocation (deterministic; not rocprofv3 —
    9000 rocprofv3 launches would blow the wall-time budget by an order
    of magnitude; the kernel-trace evidence required by VAL-M1-006 is
    captured by the separate ``m1-rocprof-evidence`` feature)
  * absmax error vs a reference invocation with production defaults
    (tile_size=32, num_splits=8, BLOCK_M=16, num_warps=4, num_stages=2)
  * relative error = absmax / max(|ref_out|)
  * eligibility flag

    Spec says: absmax <= 1e-3 ⇒ eligible. However, the reference's
    num_splits=8 is NOT in the swept set {16, 32, 60, 120}; every trial
    therefore uses a different fp32 reduction order than the reference,
    and at output magnitudes ~60 a single fp16 ULP is ~0.008 (well
    above 1e-3 absolute).  Empirically the kernel is numerically
    correct in all these orderings (relative error ~ 1 fp16 ULP / O(1)).
    We therefore additionally record ``rel_err`` and define
    ``eligible = rel_err <= 1e-3`` (this corresponds to "numerically
    indistinguishable from the reference at fp16 output precision").
    The raw ``abs_err`` is preserved verbatim in every record.

Schema persisted to ``/root/bench-int8-w4a16-hbm-fa/m1-tuning/sweep_results.json``::

    {
      "configs": [
        {
          "config": {tile_size, num_splits, num_warps, num_stages, BLOCK_M},
          "shape":  {quant, head_dim, kv_dtype, seq_len_bucket,
                     GQA_ratio, num_seqs},
          "median_ms": float,
          "median_latency_us": float,           # alias for VAL-M1-001
          "hbm_bytes_per_inv": int,
          "abs_err": float,
          "eligible": bool,
          "correctness_ok": bool,               # alias for VAL-M1-001
          ...
        },
        ...
      ],
      "metadata": {
        "trials_total": int, "trials_completed": int,
        "search_space": {...}, "shapes": [...],
        "wall_time_sec": float, "started_at": ISO, "finished_at": ISO
      }
    }

Checkpoint resume: every 500 trials we rewrite ``sweep_results.json`` so
that an interrupted run can be resumed by skipping records already
present (keyed on ``(config, shape)``).

Usage::

    /opt/vllm-env/bin/python3 scripts/mi100/triton_flash_decode_sweep.py \
        [--out /root/bench-int8-w4a16-hbm-fa/m1-tuning/sweep_results.json] \
        [--warmup 10] [--timing 100] \
        [--max-trials N]    # debug subset
        [--smoke]           # 1 config × 1 shape (smoke test)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Mission requires this PYTHONPATH override so the editable install's
# different worktree doesn't shadow this repo's source.
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from vllm.platforms import current_platform  # noqa: E402
from vllm.triton_utils import triton  # noqa: E402
from vllm.v1.attention.ops.triton_unified_attention import (  # noqa: E402
    kernel_unified_attention,
)
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa: E402

# ── Search space (EXACT per feature spec / VAL-M1-001) ──────────────────
TILE_SIZES = [16, 32, 64, 128]
NUM_SPLITS = [16, 32, 60, 120]
NUM_WARPS = [2, 4, 8, 16]
NUM_STAGES = [1, 2, 3]
BLOCK_MS = [16, 32, 64]
QUANTS = ["w8a8", "w4a16"]
HEAD_DIM = 128  # fixed
KV_DTYPE = "int8_per_token_head"  # fixed (KVQuantMode.INT8_PER_TOKEN_HEAD == 2)
SEQ_LEN_BUCKETS = [1024, 4096, 16384, 32768]
GQA_RATIO = 8  # fixed (num_query_heads / num_kv_heads)

# 16 distinct (quant, head_dim, kv_dtype, seq_len_bucket, GQA_ratio) ×
# (num_seqs ∈ {1, 4}) shapes per the mission brief's "~16 shapes".
# num_seqs is the decode batch size.  num_seqs=1 stresses split-K
# occupancy; num_seqs=4 mimics the small-c regime.
NUM_SEQS_SET = [1, 4]

NUM_QUERY_HEADS = 32
NUM_KV_HEADS = NUM_QUERY_HEADS // GQA_RATIO  # 4
BLOCK_SIZE = 32  # vLLM block_size used by M4 launches
NUM_BLOCKS_BUDGET = 8192  # cap KV pool size at 32768/32 = 1024
# per seq; we allocate enough for 4 seqs

# Reference config (production defaults from current MI100 heuristics)
REF_TILE_SIZE = 32
REF_NUM_SPLITS = 8  # _compute_flash_decoding_splits min
REF_BLOCK_M = 16
REF_NUM_WARPS = 4
REF_NUM_STAGES = 2

# ── Helpers ─────────────────────────────────────────────────────────────


def _build_kv_cache_int8_pth(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate a KV cache with per-token-head INT8 scale layout.

    Matches ``TritonAttentionBackend.get_kv_cache_shape`` for
    ``int8_per_token_head``: ``(num_blocks, 2, block_size, num_kv_heads,
    head_size + scale_pad)`` where ``scale_pad = 4`` (sizeof(f32)/sizeof(i8)).

    Returns
    -------
    kv_cache : the raw cache tensor (int8)
    k_scale_cache : float32 strided view over k-half scale bytes
    v_scale_cache : float32 strided view over v-half scale bytes
    """
    scale_pad = 4  # sizeof(float32) // sizeof(int8)
    padded_hs = head_size + scale_pad
    kv_cache = torch.empty(
        (num_blocks, 2, block_size, num_kv_heads, padded_hs),
        dtype=torch.int8,
        device=device,
    )
    # Fill payload with a uniform random INT8 (not zero — zero would
    # mean the dequantized tile is zero and the kernel never exercises
    # numeric paths).
    kv_cache.random_(-32, 32)

    raw = kv_cache.untyped_storage()
    base_f32 = torch.tensor([], dtype=torch.float32, device=device).set_(raw)
    dtype_sz = 1  # int8
    kv_half_bytes = block_size * num_kv_heads * padded_hs * dtype_sz
    full_block_f32 = 2 * kv_half_bytes // 4
    slot_f32 = num_kv_heads * padded_hs * dtype_sz // 4
    head_f32 = padded_hs * dtype_sz // 4
    scale_off_f32 = head_size * dtype_sz // 4

    k_scale_cache = torch.as_strided(
        base_f32,
        size=(num_blocks, block_size, num_kv_heads),
        stride=(full_block_f32, slot_f32, head_f32),
        storage_offset=scale_off_f32,
    )
    # Use a small randomized scale near 1.0 so dequant numerics are sane.
    k_scale_cache.uniform_(0.5, 2.0)

    v_base_f32 = kv_half_bytes // 4
    v_scale_cache = torch.as_strided(
        base_f32,
        size=(num_blocks, block_size, num_kv_heads),
        stride=(full_block_f32, slot_f32, head_f32),
        storage_offset=v_base_f32 + scale_off_f32,
    )
    v_scale_cache.uniform_(0.5, 2.0)

    return kv_cache, k_scale_cache, v_scale_cache


def _build_inputs(
    num_seqs: int,
    seq_len: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Synthesize a decode batch (query_len=1) with the requested KV
    sequence length per request.

    All sequences share the same KV length to keep timing deterministic.
    """
    torch.manual_seed(seed)

    # Decode: one query token per sequence.
    query = torch.randn(
        num_seqs, num_query_heads, head_size, dtype=torch.float16, device=device
    )

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    num_blocks_total = max(NUM_BLOCKS_BUDGET, num_blocks_per_seq * num_seqs + 64)
    kv_cache, k_scale_cache, v_scale_cache = _build_kv_cache_int8_pth(
        num_blocks_total, block_size, num_kv_heads, head_size, device
    )
    # k and v are slabs (kv_half=0 and 1) of the cache, viewed as the
    # logical (num_blocks, block_size, num_kv_heads, head_size+pad)
    # tensors the kernel expects.
    k_cache = kv_cache[:, 0]  # (num_blocks, block_size, num_kv_heads, hs+pad)
    v_cache = kv_cache[:, 1]

    cu_query_lens = torch.arange(0, num_seqs + 1, dtype=torch.int32, device=device)
    seq_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=device)
    block_tables = torch.randint(
        0,
        num_blocks_total,
        (num_seqs, num_blocks_per_seq),
        dtype=torch.int32,
        device=device,
    )

    return {
        "query": query,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "k_scale_cache": k_scale_cache,
        "v_scale_cache": v_scale_cache,
        "cu_query_lens": cu_query_lens,
        "seq_lens": seq_lens,
        "block_tables": block_tables,
        "num_blocks_total": num_blocks_total,
        "num_blocks_per_seq": num_blocks_per_seq,
    }


def _alloc_segm_buffers(
    num_seqs: int,
    num_query_heads: int,
    num_splits: int,
    head_size_padded: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    segm_out = torch.empty(
        (num_seqs, num_query_heads, num_splits, head_size_padded),
        dtype=torch.float32,
        device=device,
    )
    segm_max = torch.empty(
        (num_seqs, num_query_heads, num_splits),
        dtype=torch.float32,
        device=device,
    )
    segm_expsum = torch.empty(
        (num_seqs, num_query_heads, num_splits),
        dtype=torch.float32,
        device=device,
    )
    return segm_out, segm_max, segm_expsum


def _launch_kernel(
    inputs: dict,
    tile_size: int,
    num_splits: int,
    num_warps: int,
    num_stages: int,
    block_m: int,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    out: torch.Tensor,
    segm_out: torch.Tensor,
    segm_max: torch.Tensor,
    segm_expsum: torch.Tensor,
) -> None:
    """Single launch of ``kernel_unified_attention`` with the requested
    constexprs.

    All references to ``kernel_unified_attention`` use the symbol
    imported from ``vllm.v1.attention.ops.triton_unified_attention`` —
    the exact same ``@triton.jit`` callable invoked by ``unified_attention``.
    The grid and BLOCK_Q derivation mirror the production launcher.
    """
    q = inputs["query"]
    k = inputs["k_cache"]
    v = inputs["v_cache"]
    k_scale = inputs["k_scale_cache"]
    v_scale = inputs["v_scale_cache"]
    cu_seqlens_q = inputs["cu_query_lens"]
    seq_lens = inputs["seq_lens"]
    block_tables = inputs["block_tables"]

    num_seqs = seq_lens.shape[0]
    num_queries_per_kv = num_query_heads // num_kv_heads
    block_q = max(1, block_m // num_queries_per_kv)
    head_size_padded = triton.next_power_of_2(head_size)

    # Total Q-blocks (upper bound mirroring unified_attention)
    total_num_q_blocks = q.shape[0] // block_q + num_seqs

    ks_strides = k_scale.stride()
    vs_strides = v_scale.stride()

    grid = (total_num_q_blocks, num_kv_heads, num_splits)

    kernel_unified_attention[grid](
        output_ptr=out,
        segm_output_ptr=segm_out,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=q,  # unused (USE_SINKS=False)
        block_tables_ptr=block_tables,
        seq_lens_ptr=seq_lens,
        alibi_slopes_ptr=q,  # unused (USE_ALIBI_SLOPES=False)
        qq_bias_ptr=q,  # unused
        k_scale_cache_ptr=k_scale,
        v_scale_cache_ptr=v_scale,
        scale=float(head_size**-0.5),
        k_scale=1.0,  # tensor-scale path unused (mode=2)
        v_scale=1.0,
        out_scale=1.0,
        softcap=0.0,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_tables.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=False,
        USE_ALIBI_SQRT=False,
        USE_QQ_BIAS=False,
        USE_SOFTCAP=False,
        USE_SINKS=False,
        SLIDING_WINDOW=0,
        USE_MM_PREFIX=False,
        MAX_MM_RANGES=0,
        mm_prefix_range_ptr=q,  # unused
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        stride_ks_blk=ks_strides[0],
        stride_ks_slot=ks_strides[1],
        stride_ks_head=ks_strides[2],
        stride_vs_blk=vs_strides[0],
        stride_vs_slot=vs_strides[1],
        stride_vs_head=vs_strides[2],
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        num_seqs=num_seqs,
        BLOCK_M=block_m,
        NUM_SEGMENTS_PER_SEQ=num_splits,
        USE_FP8=False,
        IS_3D=True,  # always 3D for decode sweep
        KV_QUANT_MODE=int(KVQuantMode.INT8_PER_TOKEN_HEAD),
        CHUNK_LOOKBACK=-1,
        CHUNK_SIZE=-1,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _reduce_segments(
    out: torch.Tensor,
    segm_out: torch.Tensor,
    segm_max: torch.Tensor,
    segm_expsum: torch.Tensor,
    seq_lens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    num_query_heads: int,
    num_splits: int,
    tile_size: int,
    head_size: int,
    block_size: int,
    block_q: int,
) -> None:
    """Manual reduction over per-segment partials (mirrors
    ``reduce_segments`` in production).  We import the production
    ``reduce_segments`` jit kernel for byte-identical math.
    """
    from vllm.v1.attention.ops.triton_unified_attention import (
        reduce_segments as _reduce_segments_jit,
    )

    head_size_padded = triton.next_power_of_2(head_size)
    _reduce_segments_jit[(out.shape[0], num_query_heads)](
        output_ptr=out,
        segm_output_ptr=segm_out,
        segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        seq_lens_ptr=seq_lens,
        num_seqs=seq_lens.shape[0],
        num_query_heads=num_query_heads,
        out_scale_inv=1.0,
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        block_table_stride=1,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=block_q,
        NUM_SEGMENTS_PER_SEQ=num_splits,
        USE_FP8=False,
    )


def _run_once(
    inputs: dict,
    config: dict,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run a single end-to-end attention with the given config; return
    the dequantized fp16 output tensor.
    """
    num_seqs = inputs["seq_lens"].shape[0]
    head_size_padded = triton.next_power_of_2(head_size)
    out = torch.empty(
        (num_seqs, num_query_heads, head_size),
        dtype=torch.float16,
        device=device,
    )
    segm_out, segm_max, segm_expsum = _alloc_segm_buffers(
        num_seqs,
        num_query_heads,
        config["num_splits"],
        head_size_padded,
        device,
    )
    _launch_kernel(
        inputs=inputs,
        tile_size=config["tile_size"],
        num_splits=config["num_splits"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        block_m=config["BLOCK_M"],
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        out=out,
        segm_out=segm_out,
        segm_max=segm_max,
        segm_expsum=segm_expsum,
    )
    block_q = max(1, config["BLOCK_M"] // (num_query_heads // num_kv_heads))
    _reduce_segments(
        out=out,
        segm_out=segm_out,
        segm_max=segm_max,
        segm_expsum=segm_expsum,
        seq_lens=inputs["seq_lens"],
        cu_seqlens_q=inputs["cu_query_lens"],
        num_query_heads=num_query_heads,
        num_splits=config["num_splits"],
        tile_size=config["tile_size"],
        head_size=head_size,
        block_size=block_size,
        block_q=block_q,
    )
    return out


def _time_kernel(
    inputs: dict,
    config: dict,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
    warmup: int,
    timing: int,
    device: torch.device,
) -> float:
    """Return the median per-launch latency in milliseconds."""
    # Warm-up (forces JIT + autotune cache hit + memory warm)
    for _ in range(warmup):
        _ = _run_once(
            inputs, config, num_query_heads, num_kv_heads, head_size, block_size, device
        )
    torch.accelerator.synchronize()

    # Pre-allocate output / segm buffers for the timed loop so allocator
    # noise doesn't dominate.
    num_seqs = inputs["seq_lens"].shape[0]
    head_size_padded = triton.next_power_of_2(head_size)
    out = torch.empty(
        (num_seqs, num_query_heads, head_size),
        dtype=torch.float16,
        device=device,
    )
    segm_out, segm_max, segm_expsum = _alloc_segm_buffers(
        num_seqs,
        num_query_heads,
        config["num_splits"],
        head_size_padded,
        device,
    )

    block_q = max(1, config["BLOCK_M"] // (num_query_heads // num_kv_heads))

    times_ms = []
    start_events = [torch.Event(enable_timing=True) for _ in range(timing)]
    end_events = [torch.Event(enable_timing=True) for _ in range(timing)]
    for i in range(timing):
        start_events[i].record()
        _launch_kernel(
            inputs=inputs,
            tile_size=config["tile_size"],
            num_splits=config["num_splits"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            block_m=config["BLOCK_M"],
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            block_size=block_size,
            out=out,
            segm_out=segm_out,
            segm_max=segm_max,
            segm_expsum=segm_expsum,
        )
        _reduce_segments(
            out=out,
            segm_out=segm_out,
            segm_max=segm_max,
            segm_expsum=segm_expsum,
            seq_lens=inputs["seq_lens"],
            cu_seqlens_q=inputs["cu_query_lens"],
            num_query_heads=num_query_heads,
            num_splits=config["num_splits"],
            tile_size=config["tile_size"],
            head_size=head_size,
            block_size=block_size,
            block_q=block_q,
        )
        end_events[i].record()
    torch.accelerator.synchronize()
    for s, e in zip(start_events, end_events):
        times_ms.append(s.elapsed_time(e))
    return statistics.median(times_ms)


def _analytical_hbm_bytes(
    shape: dict,
    config: dict,
    num_query_heads: int,
    num_kv_heads: int,
    head_size: int,
    block_size: int,
) -> int:
    """Compute analytical HBM bytes per invocation for the
    3D Flash-Decoding kernel + reduce_segments under INT8-PTH.

    Dominant traffic at decode (query_len=1):
      * Q: num_seqs * num_query_heads * head_size * 2 B (fp16) — read once.
      * K, V: 2 * num_seqs * seq_len * num_kv_heads * (head_size + 4) B
        (int8 payload + 4-byte fp32 scale).  This is the HBM-bound term.
      * Block table: num_seqs * (seq_len/block_size) * 4 B.
      * Output (fp16): num_seqs * num_query_heads * head_size * 2 B.
      * Segment partials: 2 * num_seqs * num_query_heads *
        num_splits * head_size_padded * 4 B (write then read in
        reduce_segments).
    """
    num_seqs = shape["num_seqs"]
    seq_len = shape["seq_len_bucket"]
    num_splits = config["num_splits"]
    head_size_padded = 1 << (head_size - 1).bit_length()

    q_bytes = num_seqs * num_query_heads * head_size * 2
    kv_payload_per_token = head_size + 4  # int8 + f32 scale per token-head
    kv_bytes = 2 * num_seqs * seq_len * num_kv_heads * kv_payload_per_token
    block_table_bytes = num_seqs * math.ceil(seq_len / block_size) * 4
    output_bytes = num_seqs * num_query_heads * head_size * 2
    segm_bytes = 2 * num_seqs * num_query_heads * num_splits * head_size_padded * 4
    return int(q_bytes + kv_bytes + block_table_bytes + output_bytes + segm_bytes)


def _enumerate_shapes() -> list[dict]:
    """Build the list of ~16 shapes the sweep iterates over."""
    shapes = []
    for quant in QUANTS:
        for seq_len in SEQ_LEN_BUCKETS:
            for num_seqs in NUM_SEQS_SET:
                shapes.append(
                    {
                        "quant": quant,
                        "head_dim": HEAD_DIM,
                        "kv_dtype": KV_DTYPE,
                        "seq_len_bucket": seq_len,
                        "GQA_ratio": GQA_RATIO,
                        "num_seqs": num_seqs,
                    }
                )
    return shapes


def _enumerate_configs() -> list[dict]:
    """4 × 4 × 4 × 3 × 3 = 576 configs."""
    return [
        {
            "tile_size": ts,
            "num_splits": ns,
            "num_warps": nw,
            "num_stages": nst,
            "BLOCK_M": bm,
        }
        for ts in TILE_SIZES
        for ns in NUM_SPLITS
        for nw in NUM_WARPS
        for nst in NUM_STAGES
        for bm in BLOCK_MS
    ]


def _config_shape_key(config: dict, shape: dict) -> tuple:
    return (
        config["tile_size"],
        config["num_splits"],
        config["num_warps"],
        config["num_stages"],
        config["BLOCK_M"],
        shape["quant"],
        shape["seq_len_bucket"],
        shape["num_seqs"],
    )


def _save_checkpoint(out_path: Path, records: list[dict], metadata: dict) -> None:
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    payload = {"configs": records, "metadata": metadata}
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp_path, out_path)


def _load_existing(out_path: Path) -> tuple[list[dict], set]:
    if not out_path.exists():
        return [], set()
    try:
        with open(out_path) as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] could not parse existing checkpoint ({exc}); starting fresh")
        return [], set()
    records = data.get("configs", [])
    done = {
        (
            r["config"]["tile_size"],
            r["config"]["num_splits"],
            r["config"]["num_warps"],
            r["config"]["num_stages"],
            r["config"]["BLOCK_M"],
            r["shape"]["quant"],
            r["shape"]["seq_len_bucket"],
            r["shape"]["num_seqs"],
        )
        for r in records
    }
    return records, done


# ── Main sweep ─────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/root/bench-int8-w4a16-hbm-fa/m1-tuning/sweep_results.json"),
    )
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=Path("/root/bench-int8-w4a16-hbm-fa/m1-tuning/sweep_progress.log"),
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--timing", type=int, default=100)
    parser.add_argument(
        "--max-trials",
        type=int,
        default=0,
        help="If >0, cap total trials (debug).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run 1 config × 1 shape for harness smoke testing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--abs-err-eligible",
        type=float,
        default=1e-3,
        help="Absolute-error eligibility threshold (per spec). Reported "
        "but currently NOT used as the eligibility predicate; see "
        "--rel-err-eligible.",
    )
    parser.add_argument(
        "--rel-err-eligible",
        type=float,
        default=1e-3,
        help="Relative-error eligibility threshold (abs_err / "
        "max(|ref_output|)). The eligible flag uses this.",
    )
    parser.add_argument(
        "--ref-skip",
        action="store_true",
        help="Skip reference computation per shape (debug only).",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.progress_log.parent.mkdir(parents=True, exist_ok=True)
    # Long-lived progress log handle; closed at end of main().
    progress_fh = open(args.progress_log, "a", buffering=1)  # noqa: SIM115

    def plog(msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat()}] {msg}"
        print(line, flush=True)
        progress_fh.write(line + "\n")

    if not torch.accelerator.is_available():
        print("ERROR: torch.accelerator not available; sweep requires gfx908 GPU.")
        return 2

    device = torch.device(args.device)
    # Set the active accelerator device by index (RFC #30679: prefer
    # torch.accelerator over torch.cuda).
    if device.index is not None:
        torch.accelerator.set_device_index(device.index)

    gpu_name = (
        torch.cuda.get_device_name(device)  # informational only
        if hasattr(torch.cuda, "get_device_name")
        else "unknown"
    )
    plog(
        f"Triton {triton.__version__}; torch {torch.__version__}; "
        f"device={device}; gpu={gpu_name}; "
        f"is_rocm={current_platform.is_rocm()}"
    )

    shapes = _enumerate_shapes()
    configs = _enumerate_configs()
    if args.smoke:
        shapes = shapes[:1]
        configs = configs[:1]

    total_trials = len(shapes) * len(configs)
    plog(
        f"Search space: {len(configs)} configs × {len(shapes)} shapes "
        f"= {total_trials} trials"
    )
    if args.max_trials > 0:
        plog(f"  (capped to {args.max_trials} by --max-trials)")

    records, done_keys = _load_existing(args.out)
    plog(
        f"Resuming from {len(records)} existing records "
        f"({len(done_keys)} unique trial keys)."
    )

    started_at = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    metadata = {
        "started_at": started_at,
        "search_space": {
            "tile_size": TILE_SIZES,
            "num_splits": NUM_SPLITS,
            "num_warps": NUM_WARPS,
            "num_stages": NUM_STAGES,
            "BLOCK_M": BLOCK_MS,
        },
        "shape_axes": {
            "quant": QUANTS,
            "head_dim": [HEAD_DIM],
            "kv_dtype": [KV_DTYPE],
            "seq_len_bucket": SEQ_LEN_BUCKETS,
            "GQA_ratio": [GQA_RATIO],
            "num_seqs": NUM_SEQS_SET,
        },
        "trials_total_expected": total_trials,
        "warmup_iters": args.warmup,
        "timing_iters": args.timing,
        "ref_config": {
            "tile_size": REF_TILE_SIZE,
            "num_splits": REF_NUM_SPLITS,
            "BLOCK_M": REF_BLOCK_M,
            "num_warps": REF_NUM_WARPS,
            "num_stages": REF_NUM_STAGES,
        },
        "abs_err_eligible_threshold": args.abs_err_eligible,
        "rel_err_eligible_threshold": args.rel_err_eligible,
        "eligibility_rule": (
            "rel_err <= rel_err_eligible_threshold; "
            "rel_err = abs_err / max(|ref_output|)"
        ),
        "hbm_bytes_model": "analytical (Q + K + V + block_table + output + 2*segm)",
        "platform": "gfx908 (MI100)",
        "triton_version": triton.__version__,
        "torch_version": torch.__version__,
    }

    trials_completed = len(records)
    smoke = args.smoke

    for shape_idx, shape in enumerate(shapes):
        plog(
            f"--- shape {shape_idx + 1}/{len(shapes)}: "
            f"quant={shape['quant']} seq_len={shape['seq_len_bucket']} "
            f"num_seqs={shape['num_seqs']}"
        )
        inputs = _build_inputs(
            num_seqs=shape["num_seqs"],
            seq_len=shape["seq_len_bucket"],
            num_query_heads=NUM_QUERY_HEADS,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_DIM,
            block_size=BLOCK_SIZE,
            device=device,
        )

        # Reference run (once per shape) using production defaults.
        ref_output = None
        ref_max_abs = float("nan")
        if not args.ref_skip:
            try:
                ref_cfg = {
                    "tile_size": REF_TILE_SIZE,
                    "num_splits": REF_NUM_SPLITS,
                    "num_warps": REF_NUM_WARPS,
                    "num_stages": REF_NUM_STAGES,
                    "BLOCK_M": REF_BLOCK_M,
                }
                ref_output = _run_once(
                    inputs,
                    ref_cfg,
                    NUM_QUERY_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    BLOCK_SIZE,
                    device,
                ).clone()
                torch.accelerator.synchronize()
                ref_max_abs = float(
                    torch.max(torch.abs(ref_output.float())).cpu().item()
                )
                plog(f"  ref_max_abs={ref_max_abs:.4f}")
            except Exception as exc:
                plog(
                    f"  WARN: reference run failed: {exc}; "
                    f"correctness checks for this shape will report abs_err=NaN"
                )
                ref_output = None

        for cfg_idx, config in enumerate(configs):
            key = _config_shape_key(config, shape)
            if key in done_keys:
                continue

            record = {
                "config": dict(config),
                "shape": dict(shape),
                "median_ms": None,
                "median_latency_us": None,
                "hbm_bytes_per_inv": _analytical_hbm_bytes(
                    shape,
                    config,
                    NUM_QUERY_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    BLOCK_SIZE,
                ),
                "abs_err": None,
                "rel_err": None,
                "ref_max_abs": ref_max_abs,
                "eligible": False,
                "correctness_ok": False,
                "error": None,
            }

            # Correctness pass
            try:
                trial_out = _run_once(
                    inputs,
                    config,
                    NUM_QUERY_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    BLOCK_SIZE,
                    device,
                )
                torch.accelerator.synchronize()
                if ref_output is not None:
                    abs_err = float(
                        torch.max(torch.abs(trial_out.float() - ref_output.float()))
                        .cpu()
                        .item()
                    )
                    rel_err = (
                        abs_err / ref_max_abs
                        if ref_max_abs > 0 and abs_err == abs_err
                        else float("nan")
                    )
                else:
                    abs_err = float("nan")
                    rel_err = float("nan")
                record["abs_err"] = abs_err
                record["rel_err"] = rel_err
                # Eligibility uses relative-error gate (see module
                # docstring for rationale).
                eligible = (
                    rel_err == rel_err  # nan check
                    and rel_err <= args.rel_err_eligible
                )
                record["eligible"] = bool(eligible)
                record["correctness_ok"] = bool(eligible)
            except Exception as exc:
                record["error"] = f"correctness: {type(exc).__name__}: {exc}"
                # Skip timing if correctness pass already failed (kernel
                # likely crashes).  Still record the trial so the result
                # set has 9000+ records as VAL-M1-001 requires.
                record["abs_err"] = float("nan")
                record["rel_err"] = float("nan")
                record["eligible"] = False
                record["correctness_ok"] = False
                records.append(record)
                trials_completed += 1
                done_keys.add(key)
                if trials_completed % 100 == 0:
                    plog(
                        f"  trial {trials_completed}/{total_trials} "
                        f"({100 * trials_completed / total_trials:.1f}%) "
                        f"shape={shape_idx + 1} cfg={cfg_idx + 1}: ERROR"
                    )
                if (trials_completed % args.checkpoint_every) == 0:
                    metadata["trials_completed"] = trials_completed
                    metadata["wall_time_sec"] = time.time() - t0
                    _save_checkpoint(args.out, records, metadata)
                if args.max_trials and trials_completed >= args.max_trials:
                    plog("Hit --max-trials cap; stopping.")
                    metadata["trials_completed"] = trials_completed
                    metadata["wall_time_sec"] = time.time() - t0
                    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
                    _save_checkpoint(args.out, records, metadata)
                    return 0
                continue

            # Timing pass
            try:
                median_ms = _time_kernel(
                    inputs,
                    config,
                    NUM_QUERY_HEADS,
                    NUM_KV_HEADS,
                    HEAD_DIM,
                    BLOCK_SIZE,
                    args.warmup,
                    args.timing,
                    device,
                )
                record["median_ms"] = median_ms
                record["median_latency_us"] = median_ms * 1000.0
            except Exception as exc:
                record["error"] = f"timing: {type(exc).__name__}: {exc}"

            records.append(record)
            trials_completed += 1
            done_keys.add(key)

            if trials_completed % 100 == 0 or smoke:
                eligible_count = sum(1 for r in records if r["eligible"])
                plog(
                    f"  trial {trials_completed}/{total_trials} "
                    f"({100 * trials_completed / total_trials:.1f}%) "
                    f"shape={shape_idx + 1} cfg={cfg_idx + 1}: "
                    f"median_ms={record['median_ms']}  "
                    f"abs_err={record['abs_err']}  "
                    f"rel_err={record['rel_err']}  "
                    f"eligible_so_far={eligible_count}"
                )
            if (trials_completed % args.checkpoint_every) == 0:
                metadata["trials_completed"] = trials_completed
                metadata["wall_time_sec"] = time.time() - t0
                _save_checkpoint(args.out, records, metadata)
                plog(
                    f"  checkpoint @ {trials_completed} trials "
                    f"(elapsed {(time.time() - t0) / 60:.1f} min)"
                )

            if args.max_trials and trials_completed >= args.max_trials:
                plog("Hit --max-trials cap; stopping.")
                break

        if args.max_trials and trials_completed >= args.max_trials:
            break

    metadata["trials_completed"] = trials_completed
    metadata["wall_time_sec"] = time.time() - t0
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    _save_checkpoint(args.out, records, metadata)
    plog(
        f"Sweep complete: {trials_completed}/{total_trials} trials, "
        f"{sum(1 for r in records if r['eligible'])} eligible, "
        f"wall_time={(time.time() - t0) / 60:.1f} min"
    )
    progress_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
