# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

from scripts.gfx900.cli import _semantic_matrix, _validate


def test_confirm_schema_requires_declared_three_launch_policy() -> None:
    matrix_path = (
        Path(__file__).resolve().parents[2]
        / "scripts/gfx900/matrices/reference.json"
    )
    matrix = json.loads(matrix_path.read_text())
    cell = matrix["cells"]["smoke-fp16-tp8"]
    cell["trial_policy"] = "confirm"
    assert "'confirm_launches' is a required property" in _validate(
        matrix, "matrix.schema.json"
    )

    cell["confirm_launches"] = 3
    cell["recorded_runs"] = 3
    assert not _validate(matrix, "matrix.schema.json")


def test_reference_matrix_declares_all_clean_source_confirm_cells() -> None:
    root = Path(__file__).resolve().parents[2]
    matrix = json.loads(
        (root / "scripts/gfx900/matrices/reference.json").read_text()
    )
    profile = json.loads(
        (root / "scripts/gfx900/profiles/c4130-2.json").read_text()
    )
    expected = {
        "confirm-capacity-auto-c32": {
            "phase": "profile",
            "port": 8010,
            "predecessor": "capacity-auto",
            "variant_keys": ["benchmark_argv"],
            "kv_cache_dtype": "auto",
        },
        "confirm-capacity-turboquant-c32": {
            "phase": "profile",
            "port": 8011,
            "predecessor": "capacity-turboquant-c32",
            "variant_keys": [],
            "kv_cache_dtype": "turboquant_k8v4",
        },
        "confirm-dense-eager-c1": {
            "phase": "profile",
            "port": 8012,
            "predecessor": "topology-fp16-tp8",
            "variant_keys": ["benchmark_argv", "server_argv"],
        },
        "confirm-graphs-full-decode-c1": {
            "phase": "confirm",
            "port": 8014,
            "predecessor": "confirm-dense-eager-c1",
            "variant_keys": ["server_argv"],
        },
    }

    assert not _validate(matrix, "matrix.schema.json")
    assert not _validate(profile, "profile.schema.json")
    assert not _semantic_matrix(profile, matrix)
    assert set(expected) <= matrix["cells"].keys()
    assert len({cell["port"] for cell in matrix["cells"].values()}) == len(
        matrix["cells"]
    )

    assert matrix["cells"]["smoke-fp16-tp8"]["timeout_seconds"] == 1200
    assert {
        cell_id
        for cell_id, cell in matrix["cells"].items()
        if cell["timeout_seconds"] == 1200
    } == {"smoke-fp16-tp8"}

    revision = profile["models"]["qwen35_fp16"]["revision"]
    for cell_id, details in expected.items():
        cell = matrix["cells"][cell_id]
        assert cell["trial_policy"] == "confirm"
        assert cell["confirm_launches"] == 3
        assert cell["warmup_runs"] == 1
        assert cell["recorded_runs"] == 3
        assert cell["model_alias"] == "qwen35_fp16"
        assert cell["device_group"] == "single_socket_8"
        assert cell["phase"] == details["phase"]
        assert cell["port"] == details["port"]
        assert cell["predecessor"] == details["predecessor"]
        assert cell["variant_keys"] == details["variant_keys"]
        assert cell["server_argv"][
            cell["server_argv"].index("--revision") + 1
        ] == revision

    for cell_id in (
        "confirm-capacity-auto-c32",
        "confirm-capacity-turboquant-c32",
    ):
        cell = matrix["cells"][cell_id]
        assert cell["benchmark_argv"][-2:] == ["--max-concurrency", "32"]
        assert (
            cell["server_argv"][cell["server_argv"].index("--kv-cache-dtype") + 1]
            == expected[cell_id]["kv_cache_dtype"]
        )
        assert "--enforce-eager" in cell["server_argv"]

    dense = matrix["cells"]["confirm-dense-eager-c1"]
    graph = matrix["cells"]["confirm-graphs-full-decode-c1"]
    assert dense["benchmark_argv"][3:10] == [
        "512",
        "--random-output-len",
        "512",
        "--num-prompts",
        "128",
        "--max-concurrency",
        "1",
    ]
    assert dense["benchmark_argv"] == graph["benchmark_argv"]
    assert "--enforce-eager" in dense["server_argv"]
    assert "--enforce-eager" not in graph["server_argv"]
    assert "FULL_DECODE_ONLY" in graph["server_argv"][
        graph["server_argv"].index("--compilation-config") + 1
    ]

    assert matrix["cells"]["capacity-auto"]["trial_policy"] == "screen"
    assert matrix["cells"]["capacity-turboquant-c32"]["trial_policy"] == "screen"


def _profile() -> dict:
    return {"models": {"model": {}}, "device_groups": {"group": {}}}


def _matrix() -> dict:
    return {
        "phases": ["smoke", "graphs"],
        "policies": {"smoke": {}},
        "workloads": {"short": {}},
        "cells": {
            "smoke": {
                "phase": "smoke",
                "model_alias": "model",
                "device_group": "group",
                "workload": "short",
                "promotion_policy": "smoke",
            },
            "graphs": {
                "phase": "graphs",
                "predecessor": "smoke",
                "model_alias": "model",
                "device_group": "group",
                "workload": "short",
                "promotion_policy": "smoke",
            },
        },
    }


def test_semantic_matrix_accepts_ordered_predecessor() -> None:
    assert not _semantic_matrix(_profile(), _matrix())


def test_semantic_matrix_rejects_unknown_predecessor() -> None:
    matrix = _matrix()
    matrix["cells"]["graphs"]["predecessor"] = "missing"
    assert "unknown predecessor missing" in _semantic_matrix(_profile(), matrix)[0]
