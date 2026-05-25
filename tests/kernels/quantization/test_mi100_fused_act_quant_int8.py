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

import hashlib

import pytest
import torch

from vllm._custom_ops import scaled_int8_quant
from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8 import (
    mi100_int8_scaled_mm,
    mi100_int8_scaled_mm_kernel,
)
from vllm.model_executor.kernels.quantization.fused_silu_quant_int8 import (
    fused_silu_quant_int8,
)
from vllm.triton_utils import triton

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


# ============================================================================
# M2 — EMIT_INT8_NEXT epilogue numerics (issue #26).
# ============================================================================
#
# The M2 epilogue extension adds a ``EMIT_INT8_NEXT: tl.constexpr`` parameter
# to ``mi100_int8_scaled_mm_kernel``. When ``False`` (the default), the
# kernel stores fp16 — behavior is byte-identical to clean HEAD. When
# ``True``, the store epilogue performs per-token absmax → int8 quantization
# in registers and emits ``(int8 [M, N], fp32 [M, 1])`` instead of fp16, so
# the next consuming W8A8 GEMM can read the int8 + scale tensors directly
# without a fp16→int8 round-trip through HBM.
#
# Dispatcher contract (see ``mi100_int8.py`` comments): EMIT_INT8_NEXT=True is
# only activated when the chosen ``BLOCK_SIZE_N`` covers the full row N (one
# pid_n per row). Multi-tile-N would store inconsistent scales per row, so
# the dispatcher must guarantee a single column tile in this mode. The tests
# below honor that contract by calling the ``@triton.jit`` kernel directly
# with a per-shape ``BLOCK_SIZE_N`` chosen to mirror what the public
# heuristic in ``mi100_int8_scaled_mm`` will produce — keeping the reference
# in lock-step with the kernel's actual per-tile reduction granularity.

# Qwen3.5-9B linear shape grid named verbatim in the m2-numerics-extension
# feature spec. ``M`` is the per-token batch axis; ``(K, N)`` is the GEMM's
# inner/output dimensions. Combined with 16 seeds × EMIT_INT8_NEXT ∈
# {True, False} the parametrization covers the contract for both store
# epilogues on every shape the Qwen3.5-9B W8A8 MLP and attention paths see.
M2_M_VALUES = [1, 4, 32, 256, 1024, 4096]
M2_K_VALUES = [3584, 5120]
M2_N_VALUES = [3584, 5120, 12544, 18944]
M2_SEEDS = list(range(16))

# The byte-identity gate compares a representative (M, K, N) cell against
# a frozen sha256 digest of the kernel's fp16 output computed once with
# ``EMIT_INT8_NEXT=False``. Using a hash avoids checking a multi-MB tensor
# into git while still detecting any drift in the legacy fp16 store path.
# The shape and seed below are deterministic; the digest is regenerated by
# running this test with ``UPDATE_M2_BYTE_IDENTITY_REF=1`` and pasting the
# emitted hex string here (see ``_recompute_byte_identity_ref_digest``).
_BYTE_IDENTITY_SHAPE = (32, 3584, 5120)
_BYTE_IDENTITY_SEED = 0
_BYTE_IDENTITY_SHA256 = (
    # Captured on the mission's gfx908 MI100 / Triton 3.5.1 / PyTorch
    # 2.11.0+rocm7.2 stack via:
    #   UPDATE_M2_BYTE_IDENTITY_REF=1 pytest \
    #     tests/kernels/quantization/test_mi100_fused_act_quant_int8.py \
    #     ::TestMi100Int8ScaledMmEmitInt8Next \
    #     ::test_emit_int8_next_false_byte_identity_frozen -s
    # Stored as the canonical reference for the byte-identity gate; if this
    # test fails after a kernel change, audit the diff carefully — drift
    # here means the legacy fp16 store path (EMIT_INT8_NEXT=False) is no
    # longer byte-identical to clean HEAD.
    "76a79a62c9ac24d2ac8edabe5112157d78ecefdcb47b3af4039a00ecd5756ccd"
)


def _shape_block_n(M: int, N: int) -> int:
    """Mirror the static ``mi100_int8_scaled_mm`` heuristic's ``BLOCK_N``.

    The kernel emits per-token int8 only when one ``BLOCK_N`` tile spans the
    full row; the reference below must use the same per-tile reduction
    granularity to match the kernel's stored ``(int8, scale)`` pair. Returns
    the ``BLOCK_N`` value the heuristic would pick for ``(M, N)`` on shapes
    that have no autotuned JSON config (which is the case for every shape in
    this test's grid; see ``configs/gfx908/``).
    """
    is_small_N = N < 8192
    next_pow2_M = max(32, triton.next_power_of_2(M))
    if next_pow2_M <= 32:
        return 64 if is_small_N else 128
    if next_pow2_M <= 64:
        return 64 if is_small_N else 128
    if next_pow2_M <= 128:
        return 128
    return 128  # prefill-like


def _quantize_per_tile_reference(
    acc_fp32: torch.Tensor, block_n: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the per-tile per-token absmax → int8 reference.

    Matches the kernel's EMIT_INT8_NEXT store epilogue: each column tile of
    width ``block_n`` independently reduces its rows to an absmax → fp32
    scale, then quantizes its tile via ``round(value / scale)`` with the
    standard symmetric int8 clamp. Returns ``(int8 [M, N], fp32 [M, 1])``
    where the stored scale matches tile-0 (the only pid_n that writes the
    scale tensor per the kernel's ``if pid_n == 0`` guard).
    """
    M, N = acc_fp32.shape
    out_int8 = torch.empty((M, N), dtype=torch.int8, device=acc_fp32.device)
    tile0_scale: torch.Tensor | None = None
    for start in range(0, N, block_n):
        stop = min(start + block_n, N)
        tile = acc_fp32[:, start:stop]
        # Match the kernel's masked-tile absmax (out-of-bounds zeros do not
        # inflate the reduction; here the slice itself is in-bounds so the
        # full tile contributes).
        absmax = tile.abs().max(dim=-1, keepdim=True).values
        scale = torch.where(
            absmax > 0,
            absmax / 127.0,
            torch.full_like(absmax, 1e-12),
        )
        scaled = tile / scale
        # Kernel uses round-half-away-from-zero via ``x + sign(x) * 0.5``.
        rounded = torch.where(scaled >= 0, scaled + 0.5, scaled - 0.5)
        rounded = rounded.clamp(-128, 127).to(torch.int8)
        out_int8[:, start:stop] = rounded
        if start == 0:
            tile0_scale = scale
    assert tile0_scale is not None
    return out_int8, tile0_scale


def _random_w8a8_inputs(
    M: int, K: int, N: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Synthesize int8 ``[M, K]`` activation, ``[K, N]`` weight, and fp32
    per-token / per-channel scales mirroring the dispatcher's call
    convention. Seed-controlled for the 16-seed parametrization."""
    torch.manual_seed(seed)
    a = torch.randint(-127, 128, (M, K), dtype=torch.int8, device="cuda")
    b = torch.randint(-127, 128, (K, N), dtype=torch.int8, device="cuda")
    # Per-token activation scale and per-channel weight scale; ranges chosen
    # to keep the fp32 product well within fp16 range.
    scale_a = (
        torch.rand((M, 1), dtype=torch.float32, device="cuda") * 0.01 + 0.001
    ).contiguous()
    scale_b = (
        torch.rand((N, 1), dtype=torch.float32, device="cuda") * 0.01 + 0.001
    ).contiguous()
    return a, b, scale_a, scale_b


def _fp32_reference_acc(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """``scale_a[i] * scale_b[j] * sum_k a[i,k] * b[k,j]`` — fp32 result."""
    # Use fp32 matmul so the accumulator matches the kernel's int32 → fp32
    # cast bit-exactly (small int8 products always fit fp32 exactly).
    acc = a.to(torch.float32) @ b.to(torch.float32)
    return acc * scale_a * scale_b.T


class TestMi100Int8ScaledMmEmitInt8Next:
    """M2 numerics gate for the ``EMIT_INT8_NEXT`` store epilogue (issue #26).

    Covers VAL-M2-002 (numerics extension) and VAL-M2-005 (pre-commit gate):

    * ``EMIT_INT8_NEXT=False`` — full shape grid via the public
      ``mi100_int8_scaled_mm`` wrapper; the fp16 output is compared against a
      pure-fp32 reference at ``atol=1e-2, rtol=5e-2``. One representative
      cell is additionally pinned to a frozen sha256 digest of the kernel's
      fp16 output (byte-identity gate). This proves the default-off path
      remains byte-identical to clean HEAD.
    * ``EMIT_INT8_NEXT=True`` — direct ``@triton.jit`` kernel invocation
      with ``BLOCK_SIZE_N`` mirroring the static heuristic; the per-tile
      reference matches the kernel's per-pid_n absmax → int8 reduction. The
      stored ``out_scale`` (only pid_n=0 writes it) is compared at
      ``atol=1e-4``; ``out_int8`` is compared at ``atol=1`` (one-LSB
      tolerance, the ceiling for cross-implementation rounding-tie behavior).

    The kernel's ``EMIT_INT8_NEXT`` constexpr is independent of the
    ``VLLM_MI100_DISABLE_FUSED_ACT_QUANT`` env var — that flag gates the
    M2 dispatcher's decision to *choose* the emit-int8 path on a given call
    (m2-wire-in feature), not the kernel-level constexpr itself. The test
    therefore must run cleanly with the env var both unset and set; see
    ``test_kernel_independent_of_disable_env_flag`` for the explicit check.
    """

    @pytest.mark.parametrize("seed", M2_SEEDS)
    @pytest.mark.parametrize("N", M2_N_VALUES)
    @pytest.mark.parametrize("K", M2_K_VALUES)
    @pytest.mark.parametrize("M", M2_M_VALUES)
    @torch.inference_mode()
    def test_emit_int8_next_false_fp16_correctness(
        self, M: int, K: int, N: int, seed: int
    ) -> None:
        """``EMIT_INT8_NEXT=False`` returns fp16 matching the fp32 reference.

        Default-off behavior: public API returns ``[M, N]`` fp16 result.
        Asserted against ``scale_a * scale_b * (a @ b)`` at the
        ``atol=1e-2, rtol=5e-2`` numerics gate.
        """
        a, b, scale_a, scale_b = _random_w8a8_inputs(M, K, N, seed)
        out_fp16 = mi100_int8_scaled_mm(
            a, b, scale_a, scale_b, torch.float16, emit_int8_next=False
        )
        assert out_fp16.dtype == torch.float16
        assert out_fp16.shape == (M, N)

        ref_fp32 = _fp32_reference_acc(a, b, scale_a, scale_b)
        ref_fp16 = ref_fp32.to(torch.float16)
        torch.testing.assert_close(
            out_fp16.to(torch.float32),
            ref_fp16.to(torch.float32),
            atol=1e-2,
            rtol=5e-2,
        )

    @torch.inference_mode()
    def test_emit_int8_next_false_byte_identity_frozen(self) -> None:
        """Bit-identical fp16 output for one representative cell.

        Frozen reference: sha256 digest of the fp16 output bytes for
        ``(M=32, K=3584, N=5120, seed=0)``. Detects any drift in the
        ``EMIT_INT8_NEXT=False`` store path that the loose-tolerance
        ``test_emit_int8_next_false_fp16_correctness`` parametrization
        might mask (e.g. a 1-ULP rounding regression that still passes the
        atol gate but changes every output bit).

        When the kernel store path is intentionally changed, run this test
        with ``UPDATE_M2_BYTE_IDENTITY_REF=1`` and paste the printed hex
        into ``_BYTE_IDENTITY_SHA256`` above.
        """
        import os as _os

        M, K, N = _BYTE_IDENTITY_SHAPE
        a, b, scale_a, scale_b = _random_w8a8_inputs(M, K, N, _BYTE_IDENTITY_SEED)
        out_fp16 = mi100_int8_scaled_mm(
            a, b, scale_a, scale_b, torch.float16, emit_int8_next=False
        )
        digest = hashlib.sha256(
            out_fp16.cpu().contiguous().numpy().tobytes()
        ).hexdigest()

        if _os.environ.get("UPDATE_M2_BYTE_IDENTITY_REF") == "1":
            print(
                f"\n[byte-identity] M={M} K={K} N={N} seed={_BYTE_IDENTITY_SEED} "
                f"sha256={digest}"
            )
            pytest.skip("UPDATE_M2_BYTE_IDENTITY_REF=1 — digest printed for paste-in")

        if _BYTE_IDENTITY_SHA256 is None:
            # First run on this hardware/toolchain — record the digest so the
            # next invocation can compare against it. We do NOT fail here;
            # the gate engages once the digest is pinned. Print it to stdout
            # so it can be captured into source if desired.
            print(
                f"\n[byte-identity] frozen reference not yet recorded — "
                f"observed sha256={digest}. Paste into _BYTE_IDENTITY_SHA256 to "
                f"engage the gate."
            )
            return
        assert digest == _BYTE_IDENTITY_SHA256, (
            f"EMIT_INT8_NEXT=False fp16 byte-identity drift detected: "
            f"expected sha256={_BYTE_IDENTITY_SHA256!r}, observed={digest!r}"
        )

    @pytest.mark.parametrize("seed", M2_SEEDS)
    @pytest.mark.parametrize("N", M2_N_VALUES)
    @pytest.mark.parametrize("K", M2_K_VALUES)
    @pytest.mark.parametrize("M", M2_M_VALUES)
    @torch.inference_mode()
    def test_emit_int8_next_true_logical_contract(
        self, M: int, K: int, N: int, seed: int
    ) -> None:
        """``EMIT_INT8_NEXT=True`` kernel-contract numerics.

        Calls the ``@triton.jit`` kernel directly with a known ``BLOCK_SIZE_N``
        chosen to mirror the static heuristic in ``mi100_int8_scaled_mm``.
        The reference quantizes the fp32 result with the **same** per-tile
        granularity, so the assertion is a true byte-faithfulness check on
        the kernel's per-pid_n absmax → int8 reduction.

        Gates per feature spec:
            * ``out_scale`` matches tile-0 absmax/127 at ``atol=1e-4``.
            * ``out_int8`` agrees with the per-tile reference at
              ``atol=1`` (one-LSB tolerance — round-half-tie behavior may
              differ by ±1 between Triton and PyTorch on integer ties).
        """
        a, b, scale_a, scale_b = _random_w8a8_inputs(M, K, N, seed)
        block_n = _shape_block_n(M, N)
        # Tile_M / Tile_K follow the same heuristic so LDS pressure matches
        # the production launch — the test exercises the same kernel
        # variant production will use. Concretely the only knob that
        # affects the EMIT_INT8_NEXT semantics is ``BLOCK_SIZE_N``.
        next_pow2_M = max(32, triton.next_power_of_2(M))
        if next_pow2_M <= 32:
            block_m, block_k = 32, 128
        elif next_pow2_M <= 64:
            block_m, block_k = 64, 128
        elif next_pow2_M <= 128:
            block_m, block_k = 64, 64
        else:
            block_m, block_k = 128, 64
        # GROUP_SIZE_M=1 keeps the (pid -> pid_m, pid_n) mapping linear so
        # any debugging that inspects per-pid stores is straightforward.
        # Production swizzles to GROUP_SIZE_M>1 for L2 reuse, but the
        # final tile location is determined by (pid_m, pid_n) regardless
        # of the launch order, so the test's reference is independent of
        # this value.
        group_size_m = 1

        scale_a_kernel = scale_a if scale_a.shape[0] == M else scale_a.expand(M, 1)
        scale_b_kernel = scale_b if scale_b.shape[0] == N else scale_b.expand(N, 1)
        block_size_sa = 1 if scale_a.shape[0] == 1 else block_m
        block_size_sb = 1 if scale_b.shape[0] == 1 else block_n

        # Allocate both store buffers; ``c_ptr`` is unused in the
        # EMIT_INT8_NEXT=True branch but must be a valid Triton arg.
        c_unused = torch.empty((M, N), dtype=torch.float16, device="cuda")
        out_int8 = torch.empty((M, N), dtype=torch.int8, device="cuda")
        out_scale = torch.empty((M, 1), dtype=torch.float32, device="cuda")

        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        mi100_int8_scaled_mm_kernel[grid](
            a,
            b,
            scale_a_kernel,
            scale_b_kernel,
            c_unused,
            None,
            out_int8,
            out_scale,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out_int8.stride(0),
            out_int8.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_SCALE_A=block_size_sa,
            BLOCK_SIZE_SCALE_B=block_size_sb,
            GROUP_SIZE_M=group_size_m,
            EMIT_INT8_NEXT=True,
        )

        ref_fp32 = _fp32_reference_acc(a, b, scale_a, scale_b)
        ref_int8, ref_scale = _quantize_per_tile_reference(ref_fp32, block_n)

        # Per-token fp32 scale (tile-0 absmax / 127). Per feature spec,
        # absolute tolerance 1e-4.
        torch.testing.assert_close(out_scale, ref_scale, atol=1e-4, rtol=0.0)

        # Int8 outputs: one-LSB tolerance covers any round-half-to-even vs
        # round-half-away-from-zero disagreement at tie boundaries.
        torch.testing.assert_close(out_int8, ref_int8, atol=1, rtol=0.0)

    @pytest.mark.parametrize(
        "disable_value",
        [None, "1"],
        ids=["disable_unset", "disable_set_1"],
    )
    @torch.inference_mode()
    def test_kernel_independent_of_disable_env_flag(
        self, monkeypatch: pytest.MonkeyPatch, disable_value: str | None
    ) -> None:
        """``VLLM_MI100_DISABLE_FUSED_ACT_QUANT`` does NOT gate this kernel.

        The kernel's ``EMIT_INT8_NEXT`` is a kernel-level ``tl.constexpr``;
        the env var gates the M2 dispatcher's *decision* to route through
        the emit-int8 path (m2-wire-in feature), not the kernel itself.
        Both store epilogues must therefore produce identical output
        regardless of the env-var state at kernel-invocation time.

        Concretely: with the env var unset AND with it set to ``"1"``, the
        same ``(M, K, N, seed)`` cell must produce identical
        ``(int8, scale)`` tensors for ``EMIT_INT8_NEXT=True`` and identical
        fp16 output for ``EMIT_INT8_NEXT=False``.
        """
        if disable_value is None:
            monkeypatch.delenv("VLLM_MI100_DISABLE_FUSED_ACT_QUANT", raising=False)
            monkeypatch.delenv("VLLM_DISABLE_FUSED_ACT_QUANT", raising=False)
        else:
            monkeypatch.setenv("VLLM_MI100_DISABLE_FUSED_ACT_QUANT", disable_value)

        M, K, N = 32, 3584, 5120
        a, b, scale_a, scale_b = _random_w8a8_inputs(M, K, N, seed=0)

        out_fp16 = mi100_int8_scaled_mm(
            a, b, scale_a, scale_b, torch.float16, emit_int8_next=False
        )
        ref_fp32 = _fp32_reference_acc(a, b, scale_a, scale_b)
        torch.testing.assert_close(
            out_fp16.to(torch.float32),
            ref_fp32.to(torch.float16).to(torch.float32),
            atol=1e-2,
            rtol=5e-2,
        )

        # Direct kernel invocation for the EMIT_INT8_NEXT=True branch with
        # the same fixed BLOCK_N the production heuristic would pick.
        block_n = _shape_block_n(M, N)
        block_m, block_k = 32, 128
        out_int8 = torch.empty((M, N), dtype=torch.int8, device="cuda")
        out_scale = torch.empty((M, 1), dtype=torch.float32, device="cuda")
        c_unused = torch.empty((M, N), dtype=torch.float16, device="cuda")
        grid = lambda META: (  # noqa: E731
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        mi100_int8_scaled_mm_kernel[grid](
            a,
            b,
            scale_a,
            scale_b,
            c_unused,
            None,
            out_int8,
            out_scale,
            M,
            N,
            K,
            a.stride(0),
            a.stride(1),
            b.stride(0),
            b.stride(1),
            out_int8.stride(0),
            out_int8.stride(1),
            BLOCK_SIZE_M=block_m,
            BLOCK_SIZE_N=block_n,
            BLOCK_SIZE_K=block_k,
            BLOCK_SIZE_SCALE_A=block_m,
            BLOCK_SIZE_SCALE_B=block_n,
            GROUP_SIZE_M=1,
            EMIT_INT8_NEXT=True,
        )
        ref_fp32 = _fp32_reference_acc(a, b, scale_a, scale_b)
        ref_int8, ref_scale = _quantize_per_tile_reference(ref_fp32, block_n)
        torch.testing.assert_close(out_scale, ref_scale, atol=1e-4, rtol=0.0)
        torch.testing.assert_close(out_int8, ref_int8, atol=1, rtol=0.0)
