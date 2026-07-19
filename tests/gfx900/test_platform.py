# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

pytest.importorskip("torch")
pytest.importorskip("vllm")

import vllm.envs as envs
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
