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
