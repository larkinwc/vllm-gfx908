# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M3 W8A8 numerical correctness vs an FP32 PyTorch reference (VAL-TRITON-005).

Sweeps the M1 hot-shape catalog at three random seeds and asserts
``torch.allclose(out_fp16, ref_fp16, atol=1e-2, rtol=5e-2)``. Failures
emit per-shape (max_abs_err, max_rel_err) so we can diagnose specific
shape failures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from vllm.platforms import current_platform

# Shapes derived from the Qwen3.5-9B layer dims & M1 hot-shape catalog.
HOT_SHAPES_W8A8 = [
    # (M, N, K, role)
    (1, 10240, 4096, "qkv_decode_tp1"),
    (1, 4096, 4096, "o_decode_tp1"),
    (1, 24576, 4096, "gate_up_decode_tp1"),
    (1, 4096, 12288, "down_decode_tp1"),
    (32, 10240, 4096, "qkv_small_prefill_tp1"),
    (32, 4096, 4096, "o_small_prefill_tp1"),
    (128, 10240, 4096, "qkv_med_prefill_tp1"),
    (512, 10240, 4096, "qkv_prefill_tp1"),
    (512, 4096, 4096, "o_prefill_tp1"),
    (512, 24576, 4096, "gate_up_prefill_tp1"),
    (512, 4096, 12288, "down_prefill_tp1"),
    # TP=4 sharded variants
    (1, 2560, 4096, "qkv_decode_tp4"),
    (32, 6144, 4096, "gate_up_small_prefill_tp4"),
    (512, 2560, 4096, "qkv_prefill_tp4"),
]

OUT_JSON = Path("/root/bench-int8-w4a16/triton/correctness_w8a8.json")


@pytest.fixture(scope="module")
def correctness_recorder():
    """Collect per-shape error stats and dump a JSON file at module teardown."""
    records: list[dict] = []
    yield records
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {
            "n_shapes": len(set((r["M"], r["N"], r["K"]) for r in records)),
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
@pytest.mark.parametrize("M,N,K,role", HOT_SHAPES_W8A8)
def test_w8a8_correctness_vs_fp32(
    M, N, K, role, seed, correctness_recorder
):
    """torch.allclose(out_fp16, ref_fp16, atol=1e-2, rtol=5e-2)."""
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
        mi100_int8_scaled_mm,
    )

    torch.manual_seed(seed)
    a = torch.randint(-32, 32, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-32, 32, (K, N), dtype=torch.int8, device="cuda")
    scale_a = (0.01 * torch.rand((M, 1), device="cuda")).to(torch.float32)
    scale_b = (0.01 * torch.rand((N, 1), device="cuda")).to(torch.float32)

    out = mi100_int8_scaled_mm(
        a, b, scale_a, scale_b, torch.float16, bias=None
    )
    # Chunk the FP32 reference computation along K to avoid OOM on the
    # largest shapes (down/gate_up @ M=512). The full [M,K] x [K,N] FP32
    # matmul peaks at ~2 GiB on a 32 GiB MI100; the kernel itself only
    # needs ~30 MiB so the OOM is purely from the reference, not the
    # device under test.
    K_chunk = 1024
    acc_f32 = torch.zeros((M, N), dtype=torch.float32, device="cuda")
    for k0 in range(0, K, K_chunk):
        k1 = min(K, k0 + K_chunk)
        a_chunk = a[:, k0:k1].to(torch.float32)
        b_chunk = b[k0:k1, :].to(torch.float32)
        acc_f32 += a_chunk @ b_chunk
        del a_chunk, b_chunk
    ref_f32 = scale_a * acc_f32 * scale_b.T
    ref = ref_f32.to(torch.float16)
    del acc_f32, ref_f32
    abs_err = (out - ref).abs()
    max_abs = float(abs_err.max().item())
    max_rel = float((abs_err / (ref.abs() + 1e-3)).max().item())
    correctness_recorder.append({
        "M": M, "N": N, "K": K, "role": role, "seed": seed,
        "max_abs_err": max_abs, "max_rel_err": max_rel,
    })
    torch.testing.assert_close(out, ref, atol=1e-2, rtol=5e-2)
    del a, b, scale_a, scale_b, out, ref
    torch.cuda.empty_cache()
