# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_mod
from vllm.platforms import rocm
from vllm.platforms.rocm import RocmPlatform
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def test_gfx900_capability_contract(monkeypatch) -> None:
    monkeypatch.setattr(rocm, "_ON_GFX900", True)
    monkeypatch.setattr(rocm, "_ON_GFX9", False)
    monkeypatch.setattr(rocm, "_GCN_ARCH", "gfx900")
    assert rocm.on_gfx900()
    assert not RocmPlatform.use_custom_allreduce()
    assert not RocmPlatform.supports_fp8()
    assert not RocmPlatform.supports_mx()
    priorities = rocm._get_backend_priorities(False, False)
    assert [backend.name for backend in priorities] == [
        AttentionBackendEnum.TRITON_ATTN.name,
        AttentionBackendEnum.TURBOQUANT.name,
    ]

    def selected_backend_name(kv_cache_dtype: str) -> str:
        return next(
            backend.get_class().get_name()
            for backend in priorities
            if backend.get_class().supports_kv_cache_dtype(kv_cache_dtype)
        )

    assert selected_backend_name("auto") == "TRITON_ATTN"
    assert selected_backend_name("turboquant_k8v4") == "TURBOQUANT"


def test_gfx900_moe_gemv_is_default_on_and_has_kill_switch(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_GFX900_MOE_GEMV", raising=False)
    envs.disable_envs_cache()
    assert envs.VLLM_GFX900_MOE_GEMV
    monkeypatch.setenv("VLLM_GFX900_MOE_GEMV", "0")
    envs.disable_envs_cache()
    assert not envs.VLLM_GFX900_MOE_GEMV


def test_gfx900_moe_gemv_dispatch_follows_env(monkeypatch) -> None:
    """get_default_config's GEMV_MODE selection must track envs.VLLM_GFX900_MOE_GEMV
    (not a raw os.environ read), for the exact gate in fused_moe.py:
    `on_gfx900() and M <= 2 and envs.VLLM_GFX900_MOE_GEMV`."""
    monkeypatch.setattr(rocm, "_ON_GFX900", True)
    monkeypatch.setattr(
        fused_moe_mod, "should_moe_wna16_use_cuda", lambda *a, **kw: False
    )

    def config_for(M: int) -> dict:
        return fused_moe_mod.get_default_config(
            M=M,
            E=8,
            N=1024,
            K=1024,
            topk=2,
            dtype="int4_w4a16",
            block_shape=[128, 128],
        )

    monkeypatch.setenv("VLLM_GFX900_MOE_GEMV", "1")
    envs.disable_envs_cache()
    assert config_for(1).get("GEMV_MODE") is True

    monkeypatch.setenv("VLLM_GFX900_MOE_GEMV", "0")
    envs.disable_envs_cache()
    assert "GEMV_MODE" not in config_for(1)

    # M > 2 never takes the GEMV path, even with the gate enabled.
    monkeypatch.setenv("VLLM_GFX900_MOE_GEMV", "1")
    envs.disable_envs_cache()
    assert "GEMV_MODE" not in config_for(3)
