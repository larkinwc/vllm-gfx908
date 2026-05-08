#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Validate every per-cell result JSON in `<root>/{synthetic,coding}/`
against `scripts/bench_schema.json`. Prints "<N> files validated, <K>
failures" and exits non-zero on any failure.

Usage:
    /opt/vllm-env/bin/python3 scripts/validate_results_schema.py /root/bench-int8-w4a16/baseline/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jsonschema

REPO = Path(
    "/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/"
    "emdash/fuzzy-hornets-see-szfl4"
)
SCHEMA_PATH = REPO / "scripts" / "bench_schema.json"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_results_schema.py <baseline-root>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    if not root.exists():
        print(f"root {root} does not exist", file=sys.stderr)
        return 2

    schema = json.loads(SCHEMA_PATH.read_text())
    validator = jsonschema.Draft7Validator(schema)

    files: list[Path] = []
    for sub in ("synthetic", "coding"):
        d = root / sub
        if d.is_dir():
            files.extend(sorted(d.glob("*.json")))

    if not files:
        print(f"no files found under {root}/synthetic|coding", file=sys.stderr)
        return 1

    failures = 0
    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {f}: parse error {e}")
            failures += 1
            continue
        errs = list(validator.iter_errors(data))
        if errs:
            print(f"FAIL {f}:")
            for e in errs[:5]:
                path = "/".join(str(p) for p in e.absolute_path) or "<root>"
                print(f"  - {path}: {e.message}")
            failures += 1
        else:
            print(f"OK   {f}")

    print(f"{len(files)} files validated, {failures} failures")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
