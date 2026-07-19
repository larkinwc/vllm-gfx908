# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from scripts.gfx900 import manifest


def _profile() -> dict:
    return {
        "host_id": "test",
        "default_device_group": "group",
        "device_groups": {"group": {"id": "group", "visible_indices": [0]}},
        "expected_platform": {"reset_method": 2, "devices": [{"arch": "gfx900"}]},
        "numa_bindings": {"socket0": {}},
    }


def test_python_runtime_uses_probe_result(monkeypatch) -> None:
    monkeypatch.setattr(
        manifest,
        "run_command",
        lambda argv: {
            "returncode": 0,
            "stdout": '{"python":"3","vllm":"x","devices":[]}',
        },
    )
    assert manifest._python_runtime() == {
        "python": "3",
        "vllm": "x",
        "vllm_commit": None,
        "devices": [],
    }


def test_manifest_uses_live_selected_devices(monkeypatch) -> None:
    monkeypatch.setattr(
        manifest, "_git_source", lambda root: {"commit": "abc", "dirty": False}
    )
    monkeypatch.setattr(
        manifest,
        "_python_runtime",
        lambda: {
            "python": "3",
            "torch": "x",
            "hip": "x",
            "triton": "x",
            "vllm": "x",
            "devices": [
                {"visible_index": 0, "arch": "gfx900"},
                {"visible_index": 1, "arch": "gfx908"},
            ],
        },
    )
    monkeypatch.setattr(
        manifest,
        "_library_paths",
        lambda root, runtime: {
            name: {"path": "/x", "sha256": "a"}
            for name in ("rccl", "rocblas", "hipblaslt", "triton")
        },
    )
    monkeypatch.setattr(manifest, "_first_line", lambda command: "2")
    monkeypatch.setattr(
        manifest, "run_command", lambda command: {"returncode": 0, "stdout": "topology"}
    )
    value = manifest.build_manifest(_profile(), root=Path.cwd())
    assert value["devices"] == [{"visible_index": 0, "arch": "gfx900"}]
    assert not manifest.validate_manifest(value, _profile())


def test_manifest_comparison_uses_platform_digest() -> None:
    assert (
        manifest.compare_platforms(
            {"platform_sha256": "same"}, {"platform_sha256": "same"}
        )["verdict"]
        == "PASS"
    )
    assert (
        manifest.compare_platforms({"platform_sha256": "a"}, {"platform_sha256": "b"})[
            "verdict"
        ]
        == "INCOMPARABLE"
    )


def test_manifest_validation_requires_library_hashes() -> None:
    value = {
        "source": {},
        "kernel": {"reset_method": 2},
        "devices": [{"arch": "gfx900"}],
        "runtime": {"vllm": "x"},
        "libraries": {"rccl": {"sha256": None}},
        "platform_sha256": "x",
    }
    assert "unable to resolve required rccl library hash" in manifest.validate_manifest(
        value, _profile()
    )


def test_reference_manifest_requires_active_vllm_commit_match() -> None:
    value = {
        "source": {"commit": "abcdef0123456789", "dirty": False},
        "kernel": {"reset_method": 2},
        "devices": [{"arch": "gfx900"}],
        "runtime": {"vllm": "x", "vllm_commit": "0123456"},
        "libraries": {"rccl": {"sha256": "x"}},
        "platform_sha256": "x",
    }
    assert "active vLLM commit does not match Git HEAD" in manifest.validate_manifest(
        value, _profile(), require_clean=True
    )
