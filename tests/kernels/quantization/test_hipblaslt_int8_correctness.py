# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Numerical-correctness tests for the M2 hipBLASLt INT8 W8A8 path.

For every (M, N, K) listed in the committed
``vllm/model_executor/kernels/configs/gfx908/hipblaslt_tuned_shapes.json``
manifest, this test compares the hipBLASLt-routed output to a PyTorch fp32
reference (with applied per-token / per-channel scales) at
``atol=1e-2, rtol=5e-2`` on the fp16 dequantized output. Failure on any
shape blocks Milestone 2 (VAL-TENSILE-005).

Tests are skipped when not running on a gfx908 GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

pytest.importorskip(
    "vllm.model_executor.kernels.linear.scaled_mm.mi100_hipblaslt"
)

from vllm.model_executor.kernels.linear.scaled_mm.mi100_hipblaslt import (  # noqa: E402,E501
    mi100_hipblaslt_scaled_mm,
)


def _on_gfx908() -> bool:
    if not torch.cuda.is_available():
        return False
    arch = torch.cuda.get_device_properties(0).gcnArchName
    return arch.startswith("gfx908")


pytestmark = pytest.mark.skipif(
    not _on_gfx908(),
    reason="Requires gfx908 (MI100) GPU + ROCm hipBLASLt INT8 backend.",
)


_MANIFEST = Path(__file__).resolve().parents[3] / (
    "vllm/model_executor/kernels/configs/gfx908/"
    "hipblaslt_tuned_shapes.json"
)


def _tuned_shapes() -> list[tuple[int, int, int]]:
    if not _MANIFEST.exists():
        return []
    obj = json.loads(_MANIFEST.read_text())
    out = []
    for s in obj.get("shapes", []):
        try:
            out.append((int(s["M"]), int(s["N"]), int(s["K"])))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _reference_w8a8(a: torch.Tensor, b: torch.Tensor,
                    sa: torch.Tensor, sb: torch.Tensor,
                    out_dtype: torch.dtype) -> torch.Tensor:
    """fp32 reference: cast int8 inputs to fp32, GEMM in fp32, then apply
    per-token / per-channel scales. Avoids ``torch.matmul`` int32 path
    which is unimplemented on ROCm. With both operands in [-32, 31] and
    K up to 12288, fp32 retains full precision (max product ~12.5M < 2^24).
    """
    acc = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    sa_f = sa.to(torch.float32).reshape(-1, 1)
    sb_f = sb.to(torch.float32).reshape(1, -1)
    out_f = acc * sa_f * sb_f
    return out_f.to(out_dtype)


@pytest.mark.parametrize(
    "M,N,K", _tuned_shapes() or [(32, 64, 64)],
    ids=lambda v: str(v),
)
def test_hipblaslt_matches_reference(M: int, N: int, K: int) -> None:
    """``torch.allclose(out_fp16, ref_fp16, atol=1e-2, rtol=5e-2)`` must
    hold for every tuned shape."""
    torch.manual_seed(0)
    # Use a moderate value range to avoid INT32 overflow in the reference
    # (K up to 12288 with full int8 range can saturate). [-32, 31] keeps
    # max abs accumulator <= K * 32 * 32 = 12.5 M for K=12288 — comfortably
    # within INT32.
    a = torch.randint(-32, 31, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-32, 31, (K, N), dtype=torch.int8, device="cuda")
    # Per-token activation scales (M-axis), per-channel weight scales (N-axis).
    sa = (torch.rand(M, 1, device="cuda", dtype=torch.float32) * 0.05) + 0.001
    sb = (torch.rand(N, 1, device="cuda", dtype=torch.float32) * 0.05) + 0.001

    out = mi100_hipblaslt_scaled_mm(
        a, b, sa, sb, out_dtype=torch.float16, bias=None,
    )
    ref = _reference_w8a8(a, b, sa, sb, out_dtype=torch.float16)

    assert out.shape == (M, N), out.shape
    assert out.dtype == torch.float16

    diff = (out.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-6)
    max_abs_err = diff.max().item()
    max_rel_err = rel.max().item()

    # Print on failure for fast triage.
    msg = (f"M={M} N={N} K={K} max_abs_err={max_abs_err:.3e} "
           f"max_rel_err={max_rel_err:.3e}")
    assert torch.allclose(out, ref, atol=1e-2, rtol=5e-2), msg


def test_bias_is_applied() -> None:
    """Smoke-check that the optional bias argument is added correctly."""
    M, N, K = 32, 64, 64
    torch.manual_seed(1)
    a = torch.randint(-8, 7, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-8, 7, (K, N), dtype=torch.int8, device="cuda")
    sa = torch.full((M, 1), 0.01, dtype=torch.float32, device="cuda")
    sb = torch.full((N, 1), 0.02, dtype=torch.float32, device="cuda")
    bias = torch.full((N,), 0.5, dtype=torch.float16, device="cuda")

    out_no_bias = mi100_hipblaslt_scaled_mm(
        a, b, sa, sb, out_dtype=torch.float16, bias=None,
    )
    out_bias = mi100_hipblaslt_scaled_mm(
        a, b, sa, sb, out_dtype=torch.float16, bias=bias,
    )
    diff = (out_bias.float() - out_no_bias.float() - 0.5).abs().max().item()
    assert diff < 1e-2, f"bias not applied correctly, diff={diff}"
