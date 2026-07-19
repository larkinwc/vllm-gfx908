# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from scripts.gfx900.cli import _semantic_matrix


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
