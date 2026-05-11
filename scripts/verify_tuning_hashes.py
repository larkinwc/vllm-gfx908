#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""VAL-CROSS-007 — verify the per-cell tuning-JSON SHA256 immutability.

For every tuning JSON shipped under
``vllm/model_executor/kernels/configs/gfx908/`` we compute its current
SHA256 and compare with a manifest at
``/root/bench-int8-w4a16/final/tuning_hashes.json``. If the manifest is
missing we (re)build it from the on-disk contents on first run.

The script additionally cross-checks the M2 hipBLASLt tuned-shapes
registry (``hipblaslt_tuned_shapes.json``) so the dispatcher's tuning
provenance is bound to a single SHA256.

Exit code is 0 if every recorded hash matches the on-disk content.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path("/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4")  # noqa: E501
TUNING_DIR = REPO / "vllm" / "model_executor" / "kernels" / "configs" / "gfx908"
MANIFEST = Path("/root/bench-int8-w4a16/final/tuning_hashes.json")
TENSILELITE_LIBPATH = Path("/root/bench-int8-w4a16/tensilelite/merged_library/library")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def collect() -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted(TUNING_DIR.glob("*.json")):
        out[str(f.relative_to(REPO))] = sha256(f)
    if TENSILELITE_LIBPATH.exists():
        for f in sorted(TENSILELITE_LIBPATH.glob("*.dat")):
            out[str(f)] = sha256(f)
        for f in sorted(TENSILELITE_LIBPATH.glob("*.yaml")):
            out[str(f)] = sha256(f)
    return out


def main() -> int:
    current = collect()

    if not MANIFEST.exists():
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(current, indent=2))
        print(f"[init] Wrote manifest with {len(current)} entries → {MANIFEST}")
        return 0

    pinned = json.loads(MANIFEST.read_text())
    fail = 0
    missing = 0
    new = 0
    for k, v in pinned.items():
        cur = current.get(k)
        if cur is None:
            print(f"  [MISSING] {k}")
            missing += 1
            continue
        if cur != v:
            print(f"  [FAIL] {k}\n          pinned={v}\n          on-disk={cur}")
            fail += 1
            continue
    for k in current:
        if k not in pinned:
            new += 1
            print(f"  [NEW]   {k}: {current[k]}")

    print()
    total = len(pinned)
    print(
        f"VAL-CROSS-007: scanned {total} pinned entries; "
        f"{fail} mismatch(es), {missing} missing, {new} new (un-pinned)."
    )
    return 0 if (fail == 0 and missing == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
