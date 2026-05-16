# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M3 W4A16 numerical correctness vs an FP32 PyTorch reference (VAL-TRITON-006).

Sweeps the M1 hot-shape catalog × group sizes {32, 128} × 3 seeds and
asserts ``torch.allclose(out_fp16, ref_fp16, atol=1e-2, rtol=5e-2)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vllm.platforms import current_platform

HOT_SHAPES_W4A16 = [
    # (M, N, K, role) — same layer dims as W8A8 (Qwen3.5-9B).
    (1, 10240, 4096, "qkv_decode_tp1"),
    (1, 4096, 4096, "o_decode_tp1"),
    (1, 24576, 4096, "gate_up_decode_tp1"),
    (1, 4096, 12288, "down_decode_tp1"),
    (32, 10240, 4096, "qkv_small_prefill_tp1"),
    (128, 10240, 4096, "qkv_med_prefill_tp1"),
    (512, 4096, 4096, "o_prefill_tp1"),
    (512, 4096, 12288, "down_prefill_tp1"),
    # TP=4 sharded variants
    (1, 2560, 4096, "qkv_decode_tp4"),
    (512, 2560, 4096, "qkv_prefill_tp4"),
]

OUT_JSON = Path("/root/bench-int8-w4a16/triton/correctness_w4a16.json")


def _make_packed_b(K: int, N: int, seed: int) -> torch.Tensor:
    """Generate a [K, N//8] int32 GPTQ-packed weight tensor without
    materialising the (K, N) intermediate (which is ~4x larger and
    causes OOM on the largest hot shapes)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    # Each int32 packs 8 random nibbles (0..15). Build the packed tensor
    # directly: 8 independent uint4 streams stitched together.
    out = torch.zeros((K, N // 8), dtype=torch.int32, device="cuda")
    for j in range(8):
        nibble = torch.randint(
            0, 16, (K, N // 8), dtype=torch.int32, device="cuda", generator=g
        )
        out |= nibble << (j * 4)
        del nibble
    return out.contiguous()


def _pytorch_w4a16_reference(
    a: torch.Tensor, b_packed: torch.Tensor, scales: torch.Tensor,
    group_size: int, zp_bias: int = 8,
) -> torch.Tensor:
    """FP32 reference computed in K-chunks to keep peak memory bounded."""
    K, N_packed = b_packed.shape
    N = N_packed * 8
    M = a.shape[0]
    shifts = torch.arange(8, device=b_packed.device, dtype=torch.int32) * 4
    out_f32 = torch.zeros((M, N), dtype=torch.float32, device=a.device)
    # Keep peak small: an [N, 256] fp32 chunk is 10 MiB at N=10k.
    K_chunk = max(group_size, 128)
    for k0 in range(0, K, K_chunk):
        k1 = min(K, k0 + K_chunk)
        b_chunk = b_packed[k0:k1]  # [k1-k0, N//8] int32
        nibbles = ((b_chunk.unsqueeze(-1) >> shifts) & 0xF).reshape(
            k1 - k0, N
        ).to(torch.int32) - zp_bias
        # Per-row scales for this chunk: row k -> g_idx = k // group_size.
        g_idx = (torch.arange(k0, k1, device=a.device) // group_size)
        scales_chunk = scales[g_idx]                                 # [k1-k0, N]
        b_fp = nibbles.to(torch.float32) * scales_chunk.to(torch.float32)
        out_f32 += a[:, k0:k1].to(torch.float32) @ b_fp
        del b_chunk, nibbles, scales_chunk, b_fp
    return out_f32.to(a.dtype)


@pytest.fixture(scope="module")
def correctness_recorder():
    records: list[dict] = []
    yield records
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {
            "n_shapes": len(set((r["M"], r["N"], r["K"], r["g"])
                                for r in records)),
            "n_records": len(records),
            "tolerance": {"atol": 1e-2, "rtol": 5e-2},
            "records": records,
        },
        indent=2,
    ))


@pytest.mark.skipif(
    not current_platform.is_rocm() or not torch.cuda.is_available(),
    reason="ROCm GPU required",
)
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("M,N,K,role", HOT_SHAPES_W4A16)
def test_w4a16_correctness_vs_fp32(
    M, N, K, role, group_size, seed, correctness_recorder
):
    if K % group_size != 0:
        pytest.skip(f"K={K} not divisible by group_size={group_size}")

    # Aggressively clear allocator state between parametrizations to
    # combat fragmentation: PyTorch's caching allocator hangs on to
    # large blocks from prior test bodies (M=512 prefill tiles), then
    # cannot satisfy a fresh ~300 MiB temporary in the next test.
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    from vllm.model_executor.kernels.linear.scaled_mm.mi100_w4a16 import (
        mi100_w4a16_gemm,
    )

    torch.manual_seed(seed)
    a = (torch.randn((M, K), device="cuda", dtype=torch.float16) * 0.1)
    b_packed = _make_packed_b(K, N, seed=seed * 31 + group_size)
    scales = (0.01 * torch.rand(
        (K // group_size, N), device="cuda")
    ).to(torch.float16)

    out = mi100_w4a16_gemm(
        a, b_packed, scales, qzeros=None,
        group_size=group_size, zp_bias=8,
    )
    ref = _pytorch_w4a16_reference(
        a, b_packed, scales, group_size, zp_bias=8
    )

    abs_err = (out - ref).abs()
    max_abs = float(abs_err.max().item())
    max_rel = float((abs_err / (ref.abs() + 1e-3)).max().item())
    correctness_recorder.append({
        "M": M, "N": N, "K": K, "g": group_size, "seed": seed,
        "role": role,
        "max_abs_err": max_abs, "max_rel_err": max_rel,
    })
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=5e-2)
    del a, b_packed, scales, out, ref
    torch.cuda.empty_cache()
