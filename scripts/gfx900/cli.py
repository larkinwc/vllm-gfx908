# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Command-line entry points for reproducible gfx900 experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import jsonschema

from scripts.gfx900.analysis import compare_cells
from scripts.gfx900.common import read_json, write_json
from scripts.gfx900.manifest import (
    build_manifest,
    compare_platforms,
    load_profile,
    validate_manifest,
)
from scripts.gfx900.runner import check_disk_headroom, run_cell

EXIT_OK = 0
EXIT_INVALID = 2
EXIT_PREFLIGHT = 3
EXIT_INCOMPARABLE = 4
EXIT_REGRESSION = 5
EXIT_FAILED = 6

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_DIR = Path(__file__).resolve().parent


def _schema(name: str) -> dict[str, Any]:
    return read_json(_SCHEMA_DIR / name)


def _validate(value: object, name: str) -> list[str]:
    return [
        error.message
        for error in jsonschema.Draft7Validator(_schema(name)).iter_errors(value)
    ]


def _semantic_matrix(profile: dict[str, Any], matrix: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    phase_index = {phase: index for index, phase in enumerate(matrix["phases"])}
    for cell_id, cell in matrix["cells"].items():
        if cell["phase"] not in phase_index:
            errors.append(f"{cell_id}: unknown phase {cell['phase']}")
        if cell["model_alias"] not in profile["models"]:
            errors.append(f"{cell_id}: unknown model alias")
        if cell["device_group"] not in profile["device_groups"]:
            errors.append(f"{cell_id}: unknown device group")
        if cell["workload"] not in matrix["workloads"]:
            errors.append(f"{cell_id}: unknown workload")
        if cell["promotion_policy"] not in matrix["policies"]:
            errors.append(f"{cell_id}: unknown promotion policy")
        predecessor = cell.get("predecessor")
        if predecessor:
            if predecessor not in matrix["cells"]:
                errors.append(f"{cell_id}: unknown predecessor {predecessor}")
            elif (
                phase_index[cell["phase"]]
                <= phase_index[matrix["cells"][predecessor]["phase"]]
            ):
                errors.append(f"{cell_id}: predecessor must be in an earlier phase")
    return errors


def command_manifest(args: argparse.Namespace) -> int:
    profile = load_profile(Path(args.profile))
    errors = _validate(profile, "profile.schema.json")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return EXIT_INVALID
    manifest = build_manifest(profile, root=_ROOT)
    errors = validate_manifest(manifest, profile, require_clean=args.reference)
    write_json(Path(args.output), manifest)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return EXIT_PREFLIGHT
    return EXIT_OK


def command_validate(args: argparse.Namespace) -> int:
    path = Path(args.path)
    value = read_json(path)
    if path.name.endswith("manifest.json"):
        errors = _validate(value, "manifest.schema.json")
    elif path.name == "cell.json":
        errors = _validate(value, "result.schema.json")
    else:
        errors = (
            _validate(value, "profile.schema.json")
            if "host_id" in value
            else _validate(value, "matrix.schema.json")
        )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return EXIT_INVALID
    return EXIT_OK


def command_run(args: argparse.Namespace) -> int:
    profile = load_profile(Path(args.profile))
    matrix = read_json(Path(args.matrix))
    errors = (
        _validate(profile, "profile.schema.json")
        + _validate(matrix, "matrix.schema.json")
        + _semantic_matrix(profile, matrix)
    )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return EXIT_INVALID
    output_dir = Path(args.output_dir)
    check_disk_headroom(
        output_dir.parent if output_dir.parent.exists() else Path.cwd(),
        profile["telemetry_limits"]["minimum_disk_gib"],
    )
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"missing manifest: {manifest_path}", file=sys.stderr)
        return EXIT_PREFLIGHT
    manifest = read_json(manifest_path)
    phases = args.phase or matrix["phases"]
    selected = [
        cell_id for cell_id, cell in matrix["cells"].items() if cell["phase"] in phases
    ]
    statuses: list[str] = []
    for cell_id in selected:
        artifact = run_cell(
            profile=profile,
            matrix=matrix,
            cell_id=cell_id,
            output_dir=output_dir,
            manifest=manifest,
        )
        statuses.append(artifact["verdict"]["status"])
    if any(status == "FAILED" for status in statuses):
        return EXIT_FAILED
    if any(status == "REGRESSION" for status in statuses):
        return EXIT_REGRESSION
    return EXIT_OK


def command_analyze(args: argparse.Namespace) -> int:
    baseline = read_json(Path(args.baseline) / "manifest.json")
    candidate = read_json(Path(args.candidate) / "manifest.json")
    result = compare_platforms(baseline, candidate)
    write_json(Path(args.output), result)
    return EXIT_OK if result["verdict"] == "PASS" else EXIT_INCOMPARABLE


def command_compare_cells(args: argparse.Namespace) -> int:
    baseline = read_json(Path(args.baseline))
    candidate = read_json(Path(args.candidate))
    errors = _validate(baseline, "result.schema.json") + _validate(
        candidate, "result.schema.json"
    )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return EXIT_INVALID
    baseline_platform = baseline["provenance"]["platform_sha256"]
    candidate_platform = candidate["provenance"]["platform_sha256"]
    identity = {
        "baseline_cell_id": baseline["identity"]["cell_id"],
        "candidate_cell_id": candidate["identity"]["cell_id"],
        "baseline_path": str(Path(args.baseline)),
        "candidate_path": str(Path(args.candidate)),
    }
    if baseline_platform != candidate_platform:
        write_json(
            Path(args.output),
            {
                "verdict": "INCOMPARABLE",
                "baseline_platform_sha256": baseline_platform,
                "candidate_platform_sha256": candidate_platform,
                **identity,
            },
        )
        return EXIT_INCOMPARABLE
    result = {**compare_cells(baseline, candidate), **identity}
    write_json(Path(args.output), result)
    return EXIT_REGRESSION if result["verdict"] == "REGRESSION" else EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="gfx900 reproducible benchmark harness"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("manifest")
    manifest.add_argument("--profile", required=True)
    manifest.add_argument("--output", required=True)
    manifest.add_argument("--reference", action="store_true")
    manifest.set_defaults(handler=command_manifest)
    run = subparsers.add_parser("run")
    run.add_argument("--profile", required=True)
    run.add_argument("--matrix", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--phase", action="append")
    run.add_argument("--resume", action="store_true")
    run.set_defaults(handler=command_run)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--baseline", required=True)
    analyze.add_argument("--candidate", required=True)
    analyze.add_argument("--output", required=True)
    analyze.set_defaults(handler=command_analyze)
    compare_cells_parser = subparsers.add_parser("compare-cells")
    compare_cells_parser.add_argument("--baseline", required=True)
    compare_cells_parser.add_argument("--candidate", required=True)
    compare_cells_parser.add_argument("--output", required=True)
    compare_cells_parser.set_defaults(handler=command_compare_cells)
    validate = subparsers.add_parser("validate")
    validate.add_argument("path")
    validate.set_defaults(handler=command_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
