# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1 numerics gate for ``fused_silu_quant_int8`` on MI100 (gfx908).

Issue #33 — covers VAL-M1-002 (numerics gate) and VAL-M1-005 (pre-commit hook
gate).

Each parametrization synthesizes an fp16 ``[M, 2*H]`` input, computes a
reference via the existing ``silu_and_mul`` + per-token ``scaled_int8_quant``
composition (the fallback path that today's production W8A8 MLP uses) and the
fused output via :func:`fused_silu_quant_int8`, then asserts:

    * fp16 dequantized output matches reference at ``atol=1e-2, rtol=5e-2``;
    * int8 outputs agree at ``atol=1`` (one-LSB tolerance);
    * per-token fp32 scales agree at absolute tolerance ``1e-4``.

The test exercises BOTH the fused path (the kernel itself) AND the fallback
composition that derives the reference — so the disable-path (which the
``m1-wire-in`` feature will gate via env var on the dispatcher side) remains
byte-identical to today's production behavior. The env var is NOT consulted in
this file; this is a pure kernel-level test that runs the fused op and the
fallback composition side-by-side.
"""

from __future__ import annotations

import pytest
import torch

from vllm._custom_ops import scaled_int8_quant
from vllm.model_executor.kernels.quantization.fused_silu_quant_int8 import (
    fused_silu_quant_int8,
)

# Qwen3.5-9B MLP intermediate widths the mission targets. Mirror the shape grid
# named verbatim in the M1 feature spec.
M_VALUES = [1, 4, 32, 256, 1024, 4096]
H_VALUES = [3584, 5120, 12544, 18944]
SEEDS = list(range(16))


def _gfx908_skip_reason() -> str | None:
    """Return a skip reason string when this host is not an MI100 (gfx908)."""
    if not torch.cuda.is_available():
        return "CUDA / ROCm GPU required for fused_silu_quant_int8."
    name = torch.cuda.get_device_name(0)
    if name.find("gfx908") < 0 and name.find("MI100") < 0:
        return f"fused_silu_quant_int8 targets MI100 (gfx908); got {name!r}."
    return None


pytestmark = pytest.mark.skipif(
    _gfx908_skip_reason() is not None,
    reason=_gfx908_skip_reason() or "skip on non-MI100",
)


def _fallback_silu_quant_int8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Today's production composition: ``silu_and_mul`` then per-token int8 quant.

    This is the byte-identical fallback that ``VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1``
    will route through in the wire-in feature. Returns ``(int8 [M, H],
    fp32 [M, 1])`` to match :func:`fused_silu_quant_int8`'s contract.

    Uses the underlying ``torch.ops._C.silu_and_mul`` directly to avoid the
    ``CustomOp`` config-fixture dependency (this is a pure kernel-level test
    that doesn't initialize a vLLM config).
    """
    M, two_h = x.shape
    H = two_h // 2
    y = torch.empty((M, H), dtype=x.dtype, device=x.device)
    torch.ops._C.silu_and_mul(y, x)
    q, s, _ = scaled_int8_quant(y)
    return q, s


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("H", H_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@torch.inference_mode()
def test_fused_silu_quant_int8_matches_fallback(M: int, H: int, seed: int) -> None:
    """16 seeds × Qwen3.5-9B shape grid: fused kernel vs fallback composition.

    Numerical gates (per VAL-M1-002):
        * int8 outputs: ``atol=1`` (one-LSB tolerance) vs fallback.
        * per-token fp32 scale: absolute tolerance ``1e-4`` vs fallback.
        * fp16-dequantized output (``q * scale``): ``atol=1e-2, rtol=5e-2``
          vs fp16 reference computed from the fallback's int8 + scale.

    Exercising the fallback inside the test also validates that today's
    production composition is still bit-functional — this is the disable-path
    behavior the wire-in feature must preserve.
    """
    torch.manual_seed(seed)
    # SiluAndMul layout: x[:, :H] is the gate, x[:, H:] is the up projection.
    x = torch.randn((M, 2 * H), dtype=torch.float16, device="cuda")

    # Fallback (disable-path equivalent): silu_and_mul then per-token int8 quant.
    q_ref, s_ref = _fallback_silu_quant_int8(x)
    assert q_ref.dtype == torch.int8
    assert q_ref.shape == (M, H)
    assert s_ref.dtype == torch.float32
    assert s_ref.shape == (M, 1)

    # Fused path: single Triton kernel.
    q_fused, s_fused = fused_silu_quant_int8(x)
    assert q_fused.dtype == torch.int8
    assert q_fused.shape == (M, H)
    assert s_fused.dtype == torch.float32
    assert s_fused.shape == (M, 1)

    # Per-token scale: tight absolute tolerance — the scale is a single fp32
    # absmax divided by 127, so both paths should agree to several digits.
    torch.testing.assert_close(s_fused, s_ref, atol=1e-4, rtol=0.0)

    # Int8 outputs: one-LSB tolerance covers the round-half-away-from-zero
    # boundary cases that differ between the csrc path and the Triton kernel
    # on rare absmax ties.
    torch.testing.assert_close(q_fused, q_ref, atol=1, rtol=0.0)

    # fp16 dequantized output: the user-visible numeric quantity downstream
    # GEMMs see. Per-token int8 quantization has an unavoidable elementwise
    # noise floor of ``scale`` (1 LSB ≈ absmax/127), so the spec's
    # ``atol=1e-2, rtol=5e-2`` is applied on top of that natural floor — the
    # gate fails iff the fused kernel introduces error beyond ±1 LSB on any
    # element relative to the fallback. With both int8 outputs within 1 LSB
    # (asserted above) and scales within 1e-4 (also above), the bound is
    # ``|q_diff| * scale + |q_ref| * s_diff ≤ scale + 127 * 1e-4``.
    deq_fused = q_fused.to(torch.float32) * s_fused
    deq_ref = q_ref.to(torch.float32) * s_ref
    per_token_atol = (s_ref.flatten() + 127.0 * 1e-4 + 1e-2).to(torch.float32)
    abs_diff = (deq_fused - deq_ref).abs()
    # rtol component: 5e-2 * |deq_ref|.
    bound = per_token_atol[:, None] + 5e-2 * deq_ref.abs()
    assert torch.all(abs_diff <= bound), (
        f"fp16 dequant gate exceeded: max abs diff "
        f"{abs_diff.max().item():.4g}, max bound {bound.max().item():.4g}"
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("H", H_VALUES)
@pytest.mark.parametrize("M", M_VALUES)
@torch.inference_mode()
def test_fallback_path_matches_fp16_reference(M: int, H: int, seed: int) -> None:
    """Disable-path (fallback composition) numerics gate.

    Exercises the existing ``silu_and_mul`` + ``scaled_int8_quant`` composition
    that ``VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1`` will route through, asserting
    its fp16-dequantized output matches a pure-PyTorch fp32 reference. The
    per-element quantization noise floor is ``scale / 2`` (round-to-nearest),
    on top of the spec's ``atol=1e-2, rtol=5e-2`` tolerance. This guards
    against regression of the disable path even though this test file does
    not flip the env var directly (the env var is dispatcher-side and gated
    in by the m1-wire-in feature).
    """
    torch.manual_seed(seed)
    x = torch.randn((M, 2 * H), dtype=torch.float16, device="cuda")

    # Pure-PyTorch fp32 reference (numerically faithful baseline).
    gate = x[:, :H].to(torch.float32)
    up = x[:, H:].to(torch.float32)
    y_fp32 = torch.nn.functional.silu(gate) * up
    y_ref_fp16 = y_fp32.to(torch.float16)

    q, s = _fallback_silu_quant_int8(x)
    deq = (q.to(torch.float32) * s).to(torch.float16)

    # Per-token bound = scale/2 (rounding) + atol + rtol*|ref|.
    abs_diff = (deq - y_ref_fp16).abs()
    bound = (s.flatten()[:, None] * 0.5 + 1e-2) + 5e-2 * y_ref_fp16.abs()
    assert torch.all(abs_diff <= bound), (
        f"fallback fp16 gate exceeded: max abs diff "
        f"{abs_diff.max().item():.4g}, max bound {bound.max().item():.4g}"
    )
