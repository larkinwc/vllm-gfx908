# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the gfx908 autotune-config loader (VAL-TRITON-004)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from vllm.model_executor.kernels.configs.gfx908 import config_loader


@pytest.fixture
def tmp_config(monkeypatch, tmp_path):
    """Redirect the loader's CONFIG_DIR to a temp dir and reset the cache."""
    monkeypatch.setattr(config_loader, "CONFIG_DIR", tmp_path)
    config_loader.reset_cache()
    yield tmp_path
    config_loader.reset_cache()


def _write_cfg(dirpath: Path, kernel: str, M: int, N: int, K: int,
               group_size: int | None, block_m: int) -> Path:
    if group_size is not None:
        fname = f"{kernel}_M{M}_N{N}_K{K}_g{group_size}.json"
    else:
        fname = f"{kernel}_M{M}_N{N}_K{K}.json"
    payload = {
        "schema_version": 1,
        "kernel": kernel,
        "shape": {"M": M, "N": N, "K": K, "group_size": group_size},
        "config": {
            "BLOCK_M": block_m,
            "BLOCK_N": 64,
            "BLOCK_K": 32,
            "GROUP_SIZE_M": 8,
            "matrix_instr_nonkdim": 16,
            "kpack": 2,
            "waves_per_eu": 2,
            "num_warps": 4,
            "num_stages": 2,
        },
        "measured_tok_s": 1234.5,
    }
    p = dirpath / fname
    p.write_text(json.dumps(payload))
    return p


def test_load_returns_none_when_missing(tmp_config):
    config_loader.reset_cache()
    out = config_loader.load_config("mi100_int8", 1, 4096, 4096)
    assert out is None


def test_load_returns_config_dict(tmp_config):
    _write_cfg(tmp_config, "mi100_int8", 64, 4096, 4096, None, block_m=64)
    config_loader.reset_cache()
    out = config_loader.load_config("mi100_int8", 64, 4096, 4096)
    assert out is not None
    assert out["BLOCK_M"] == 64
    assert out["BLOCK_N"] == 64
    assert out["matrix_instr_nonkdim"] == 16
    assert out["kpack"] == 2


def test_load_groupsize_keyed(tmp_config):
    _write_cfg(tmp_config, "mi100_w4a16", 16, 4096, 4096, 128, block_m=16)
    _write_cfg(tmp_config, "mi100_w4a16", 16, 4096, 4096, 32, block_m=32)
    config_loader.reset_cache()
    a = config_loader.load_config("mi100_w4a16", 16, 4096, 4096, group_size=32)
    b = config_loader.load_config(
        "mi100_w4a16", 16, 4096, 4096, group_size=128
    )
    assert a is not None and b is not None
    assert a["BLOCK_M"] == 32
    assert b["BLOCK_M"] == 16


def test_disabled_via_env(tmp_config, monkeypatch):
    _write_cfg(tmp_config, "mi100_int8", 64, 4096, 4096, None, block_m=64)
    config_loader.reset_cache()
    monkeypatch.setenv("VLLM_MI100_DISABLE_AUTOTUNE_CONFIG", "1")
    assert config_loader.load_config("mi100_int8", 64, 4096, 4096) is None


def test_list_configs_round_trip(tmp_config):
    _write_cfg(tmp_config, "mi100_int8", 64, 4096, 4096, None, block_m=64)
    _write_cfg(tmp_config, "mi100_w4a16", 16, 4096, 4096, 32, block_m=16)
    config_loader.reset_cache()
    all_cfgs = config_loader.list_configs()
    assert len(all_cfgs) == 2
    only_w4a16 = config_loader.list_configs("mi100_w4a16")
    assert len(only_w4a16) == 1
    assert only_w4a16[0]["kernel"] == "mi100_w4a16"
