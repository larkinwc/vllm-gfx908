# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dispatch + repack-hook tests for the MI100 Marlin-style W4A16 path.

These cover feature F-M1-dispatch: the env flag
``VLLM_MI100_W4A16_USE_MARLIN_REPACK`` (default-off), the offline repack
hook in ``process_weights_after_loading``, and the selection order in
``apply_weights``:

  1. ``VLLM_DISABLE_MI100_W4A16=1`` -> generic Triton path (wins over all).
  2. marlin flag on AND on MI100 AND g in {32,128} AND repacked tensors
     present -> ``mi100_w4a16_marlin_gemm``.
  3. otherwise -> ``triton_w4a16_gemm`` (legacy ``mi100_w4a16_gemm`` default).

The flag is toggled via env (monkeypatch); ``on_mi100`` is mocked where the
selection is being exercised. Module-level skip on non-ROCm.
"""

from __future__ import annotations

import importlib

import pytest
import torch

from vllm.platforms import current_platform

# ROCm/Triton specific; skip at import on non-ROCm.
if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("triton")

device = "cuda"

triton_w4a16_mod = importlib.import_module(
    "vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16"
)
TritonW4A16LinearKernel = triton_w4a16_mod.TritonW4A16LinearKernel
triton_w4a16_gemm = triton_w4a16_mod.triton_w4a16_gemm

mi100_marlin_mod = importlib.import_module(
    "vllm.model_executor.kernels.linear.scaled_mm.mi100_w4a16_marlin"
)
mi100_legacy_mod = importlib.import_module(
    "vllm.model_executor.kernels.linear.scaled_mm.mi100_w4a16"
)
rocm_platform_mod = importlib.import_module("vllm.platforms.rocm")

from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402,E501
    MPLinearLayerConfig,
)
from vllm.scalar_type import scalar_types  # noqa: E402

MARLIN_FLAG = "VLLM_MI100_W4A16_USE_MARLIN_REPACK"
DISABLE_FLAG = "VLLM_DISABLE_MI100_W4A16"


# --------------------------------------------------------------------------- #
#  Packing helpers
# --------------------------------------------------------------------------- #
def _pack_int4_along_n(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """[K, N] int4 nibbles -> [K, N//8] int32 (GPTQ N-packing)."""
    K, N = w_int4_kn.shape
    shifts = torch.arange(8, device=w_int4_kn.device, dtype=torch.int32) * 4
    return torch.sum(
        (w_int4_kn.view(K, N // 8, 8) & 0xF) << shifts,
        dim=2,
        dtype=torch.int32,
    ).contiguous()


def _pack_int4_along_k_to_ckpt(w_int4_kn: torch.Tensor) -> torch.Tensor:
    """[K, N] int4 -> [N, K//8] int32 (CT checkpoint layout, K-packed)."""
    K, N = w_int4_kn.shape
    out = torch.zeros((N, K // 8), dtype=torch.int32, device=w_int4_kn.device)
    for i in range(8):
        out |= (w_int4_kn[i::8, :].t() & 0xF) << (i * 4)
    return out.contiguous()


# --------------------------------------------------------------------------- #
#  Spies
# --------------------------------------------------------------------------- #
def _make_spy(M: int, N: int, dtype: torch.dtype):
    """A gemm replacement that records calls and returns a [M,N] tensor."""
    calls: list = []

    def _spy(*args, **kwargs):  # noqa: ANN001
        calls.append((args, kwargs))
        return torch.zeros((M, N), device=device, dtype=dtype)

    return _spy, calls


def _install_spies(monkeypatch, M, N, dtype):
    marlin_spy, marlin_calls = _make_spy(M, N, dtype)
    legacy_spy, legacy_calls = _make_spy(M, N, dtype)
    monkeypatch.setattr(mi100_marlin_mod, "mi100_w4a16_marlin_gemm", marlin_spy)
    monkeypatch.setattr(mi100_legacy_mod, "mi100_w4a16_gemm", legacy_spy)
    return marlin_calls, legacy_calls


# --------------------------------------------------------------------------- #
#  Layer builders
# --------------------------------------------------------------------------- #
def _make_legacy_layer(K, N, G, has_zp, seed=0):
    """Build a layer holding the legacy *kernel-layout* tensors directly.

    layer.w_q : [K, N//8] int32
    layer.w_s : [K//G, N] fp16
    layer.w_zp: [K//G, N//8] int32 (asym) or absent (sym)
    """
    torch.manual_seed(seed)
    num_groups = K // G
    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    w_q = _pack_int4_along_n(w_int4_kn)
    w_s = (0.05 * torch.rand((num_groups, N), device=device, dtype=torch.float32)).to(
        torch.float16
    )

    weight_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=weight_type,
        act_type=torch.float16,
        group_size=G,
        zero_points=has_zp,
        has_g_idx=False,
    )
    kernel = TritonW4A16LinearKernel(
        config,
        w_q_param_name="w_q",
        w_s_param_name="w_s",
        w_zp_param_name="w_zp" if has_zp else None,
        w_gidx_param_name=None,
    )

    class _Layer(torch.nn.Module):
        pass

    layer = _Layer()
    layer.w_q = w_q
    layer.w_s = w_s
    if has_zp:
        zeros_int4 = torch.randint(
            0, 16, (num_groups, N), device=device, dtype=torch.int32
        )
        layer.w_zp = _pack_int4_along_n(zeros_int4)
    return layer, kernel


def _tp_initialized() -> bool:
    """True iff the tensor-model-parallel group is currently live.

    A vLLM autouse fixture tears the distributed env down between tests, so
    this must be re-checked (not cached) on every checkpoint-layer build.
    """
    from vllm.distributed.parallel_state import (
        get_tensor_model_parallel_rank,
    )

    try:
        get_tensor_model_parallel_rank()
        return True
    except AssertionError:
        return False


def _ensure_distributed():
    if _tp_initialized():
        return
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:0",
            local_rank=0,
        )
        ensure_model_parallel_initialized(1, 1)


def _build_checkpoint_layer(K, N, G, has_zp, seed=0):
    """Build a layer with vLLM-wrapped CT *checkpoint*-layout params."""
    _ensure_distributed()
    from vllm.model_executor.parameter import (
        GroupQuantScaleParameter,
        PackedColumnParameter,
        PackedvLLMParameter,
    )

    torch.manual_seed(seed)
    w_int4_kn = torch.randint(0, 16, (K, N), device=device, dtype=torch.int32)
    w_ckpt_nk8 = _pack_int4_along_k_to_ckpt(w_int4_kn)  # [N, K//8]
    scales_ckpt_nkg = 0.05 * torch.rand((N, K // G), device=device, dtype=torch.float16)

    weight_type = scalar_types.uint4 if has_zp else scalar_types.uint4b8
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=weight_type,
        act_type=torch.float16,
        group_size=G,
        zero_points=has_zp,
        has_g_idx=False,
    )
    kernel = TritonW4A16LinearKernel(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name="weight_zero_point" if has_zp else None,
        w_gidx_param_name=None,
    )

    weight_loader = lambda *a, **k: None  # noqa: E731

    class _Layer(torch.nn.Module):
        pass

    layer = _Layer()
    layer.register_parameter(
        "weight_packed",
        PackedvLLMParameter(
            data=w_ckpt_nk8,
            weight_loader=weight_loader,
            input_dim=1,
            output_dim=0,
            packed_factor=8,
            packed_dim=1,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        GroupQuantScaleParameter(
            data=scales_ckpt_nkg,
            weight_loader=weight_loader,
            input_dim=1,
            output_dim=0,
        ),
    )
    if has_zp:
        zeros_int4_gn = torch.randint(
            0, 16, (K // G, N), device=device, dtype=torch.int32
        )
        zeros_ckpt_n8kg = _pack_int4_along_n(zeros_int4_gn).t().contiguous()
        layer.register_parameter(
            "weight_zero_point",
            PackedColumnParameter(
                data=zeros_ckpt_n8kg,
                weight_loader=weight_loader,
                output_dim=0,
                packed_factor=8,
                packed_dim=0,
            ),
        )
    return layer, kernel


_MARLIN_ATTRS = (
    triton_w4a16_mod._MARLIN_QWEIGHT_ATTR,
    triton_w4a16_mod._MARLIN_SCALE_ATTR,
    triton_w4a16_mod._MARLIN_ZERO_ATTR,
)


def _has_marlin_attrs(layer) -> bool:
    return all(getattr(layer, a, None) is not None for a in _MARLIN_ATTRS)


# --------------------------------------------------------------------------- #
#  Dispatch-selection tests
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("has_zp", [False, True], ids=["sym", "asym"])
def test_default_off_selects_legacy(monkeypatch, has_zp):
    """Flag unset -> legacy mi100_w4a16_gemm selected (not marlin)."""
    monkeypatch.delenv(MARLIN_FLAG, raising=False)
    monkeypatch.delenv(DISABLE_FLAG, raising=False)
    monkeypatch.setattr(rocm_platform_mod, "on_mi100", lambda: True)

    M, K, N, G = 16, 256, 256, 32
    layer, kernel = _make_legacy_layer(K, N, G, has_zp)
    marlin_calls, legacy_calls = _install_spies(monkeypatch, M, N, torch.float16)

    x = (0.1 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    kernel.apply_weights(layer, x)

    assert len(marlin_calls) == 0, "marlin must NOT be invoked when flag off"
    assert len(legacy_calls) == 1, "legacy mi100_w4a16_gemm must be selected"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("has_zp", [False, True], ids=["sym", "asym"])
def test_flag_on_selects_marlin(monkeypatch, has_zp, group_size):
    """Flag=1 + on_mi100 + g in {32,128} -> marlin selected."""
    monkeypatch.setenv(MARLIN_FLAG, "1")
    monkeypatch.delenv(DISABLE_FLAG, raising=False)
    monkeypatch.setattr(rocm_platform_mod, "on_mi100", lambda: True)

    M, K, N = 16, 256, 256
    layer, kernel = _make_legacy_layer(K, N, group_size, has_zp)
    # Populate marlin attrs via the (real) repack hook.
    kernel._maybe_marlin_repack(layer)
    assert _has_marlin_attrs(layer), "repack hook must stash marlin tensors"

    marlin_calls, legacy_calls = _install_spies(monkeypatch, M, N, torch.float16)
    x = (0.1 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    kernel.apply_weights(layer, x)

    assert len(marlin_calls) == 1, "marlin gemm must be selected"
    assert len(legacy_calls) == 0, "legacy gemm must NOT be invoked"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_disable_precedence_over_marlin(monkeypatch):
    """DISABLE=1 + marlin flag=1 -> generic path; marlin NOT invoked."""
    monkeypatch.setenv(MARLIN_FLAG, "1")
    monkeypatch.setenv(DISABLE_FLAG, "1")
    monkeypatch.setattr(rocm_platform_mod, "on_mi100", lambda: True)

    M, K, N, G = 16, 256, 256, 32
    layer, kernel = _make_legacy_layer(K, N, G, has_zp=True)
    # Even with marlin tensors stashed, disable must win.
    kernel._maybe_marlin_repack(layer)
    assert _has_marlin_attrs(layer)

    marlin_calls, legacy_calls = _install_spies(monkeypatch, M, N, torch.float16)
    x = (0.1 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    kernel.apply_weights(layer, x)

    assert len(marlin_calls) == 0, "DISABLE must prevent marlin invocation"
    assert len(legacy_calls) == 0, "DISABLE forces generic Triton (not legacy)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_group_gate(monkeypatch):
    """g in {32,128} -> marlin (flag on); g=64 -> fallback (no marlin)."""
    monkeypatch.setenv(MARLIN_FLAG, "1")
    monkeypatch.delenv(DISABLE_FLAG, raising=False)
    monkeypatch.setattr(rocm_platform_mod, "on_mi100", lambda: True)

    M, K, N = 16, 256, 256

    # g=32 -> marlin selected.
    layer32, kernel32 = _make_legacy_layer(K, N, 32, has_zp=True)
    kernel32._maybe_marlin_repack(layer32)
    assert _has_marlin_attrs(layer32)
    marlin_calls, _ = _install_spies(monkeypatch, M, N, torch.float16)
    x = (0.1 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    kernel32.apply_weights(layer32, x)
    assert len(marlin_calls) == 1, "g=32 must select marlin"

    # g=64 -> no repack attrs, fallback (marlin NOT invoked).
    layer64, kernel64 = _make_legacy_layer(K, N, 64, has_zp=True)
    kernel64._maybe_marlin_repack(layer64)
    assert not _has_marlin_attrs(layer64), "g=64 must not stash marlin tensors"
    marlin_calls2, _ = _install_spies(monkeypatch, M, N, torch.float16)
    kernel64.apply_weights(layer64, x)
    assert len(marlin_calls2) == 0, "g=64 must fall back (no marlin)"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_non_mi100_preserves_legacy(monkeypatch):
    """Non-MI100 -> marlin never invoked (legacy selection preserved)."""
    monkeypatch.setenv(MARLIN_FLAG, "1")
    monkeypatch.delenv(DISABLE_FLAG, raising=False)
    monkeypatch.setattr(rocm_platform_mod, "on_mi100", lambda: False)

    M, K, N, G = 16, 256, 256, 32
    layer, kernel = _make_legacy_layer(K, N, G, has_zp=True)
    # Repack hook is gated on on_mi100 -> no attrs stashed.
    kernel._maybe_marlin_repack(layer)
    assert not _has_marlin_attrs(layer), "non-MI100 must not stash marlin"

    marlin_calls, _ = _install_spies(monkeypatch, M, N, torch.float16)
    x = (0.1 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )
    kernel.apply_weights(layer, x)
    assert len(marlin_calls) == 0, "marlin must NOT be invoked on non-MI100"


# --------------------------------------------------------------------------- #
#  Repack-hook tests (process_weights_after_loading)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("has_zp", [False, True], ids=["sym", "asym"])
def test_repack_hook(monkeypatch, has_zp, group_size):
    """process_weights_after_loading: marlin layout only when flag on+qualify;
    legacy layout identical regardless of the flag."""
    monkeypatch.setattr(rocm_platform_mod, "on_mi100", lambda: True)
    K, N = 256, 256

    # Flag OFF: no marlin attrs; legacy layout produced as usual.
    monkeypatch.delenv(MARLIN_FLAG, raising=False)
    layer_off, kernel_off = _build_checkpoint_layer(K, N, group_size, has_zp)
    kernel_off.process_weights_after_loading(layer_off)
    assert not _has_marlin_attrs(layer_off), "flag off must not stash marlin"
    assert tuple(layer_off.weight_packed.shape) == (K, N // 8)
    assert tuple(layer_off.weight_scale.shape) == (K // group_size, N)

    # Flag ON: marlin attrs present with the K-packed [K//8, N] layout.
    monkeypatch.setenv(MARLIN_FLAG, "1")
    layer_on, kernel_on = _build_checkpoint_layer(K, N, group_size, has_zp)
    kernel_on.process_weights_after_loading(layer_on)
    assert _has_marlin_attrs(layer_on), "flag on must stash marlin tensors"
    qweight = getattr(layer_on, triton_w4a16_mod._MARLIN_QWEIGHT_ATTR)
    assert tuple(qweight.shape) == (K // 8, N)
    assert qweight.dtype == torch.int32

    # Legacy layout is byte-identical regardless of flag (marlin doesn't
    # mutate the legacy params; both built from the same seed).
    assert torch.equal(layer_off.weight_packed, layer_on.weight_packed)
    assert torch.equal(layer_off.weight_scale, layer_on.weight_scale)
    if has_zp:
        assert torch.equal(layer_off.weight_zero_point, layer_on.weight_zero_point)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("has_zp", [False, True], ids=["sym", "asym"])
def test_byte_identical_disable_path(monkeypatch, has_zp):
    """With the marlin flag OFF, apply_weights output is torch.equal to the
    pre-mission legacy path (triton_w4a16_gemm on the same tensors)."""
    monkeypatch.delenv(MARLIN_FLAG, raising=False)
    monkeypatch.delenv(DISABLE_FLAG, raising=False)
    monkeypatch.setattr(rocm_platform_mod, "on_mi100", lambda: True)

    K, N, G = 256, 256, 32
    layer, kernel = _build_checkpoint_layer(K, N, G, has_zp, seed=3)
    kernel.process_weights_after_loading(layer)
    assert not _has_marlin_attrs(layer)

    M = 16
    torch.manual_seed(123)
    x = (0.1 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.float16
    )

    out = kernel.apply_weights(layer, x)

    # Pre-mission reference: call the gemm directly on the legacy tensors.
    c = kernel.config
    zp_bias = c.weight_type.bias if c.weight_type.has_bias() else 0
    w_zp = getattr(layer, "weight_zero_point", None)
    ref = triton_w4a16_gemm(
        a=x.reshape(-1, K).contiguous(),
        b_q=layer.weight_packed,
        scales=layer.weight_scale,
        qzeros=w_zp,
        group_size=G,
        zp_bias=zp_bias,
    ).reshape(x.shape[:-1] + (N,))

    assert torch.equal(out, ref), "disable path must be byte-identical"
