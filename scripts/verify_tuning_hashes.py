#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VAL-CROSS-007 / VAL-CROSS-002 — verify per-cell tuning-JSON SHA256 immutability.

For every tuning JSON shipped under
``vllm/model_executor/kernels/configs/gfx908/`` we compute its current
SHA256 and compare with a manifest at
``/root/bench-int8-w4a16/final/tuning_hashes.json``. If the manifest is
missing we (re)build it from the on-disk contents on first run.

The script additionally cross-checks the M2 hipBLASLt tuned-shapes
registry (``hipblaslt_tuned_shapes.json``) so the dispatcher's tuning
provenance is bound to a single SHA256.

When ``--hbm`` is passed, the manifest at
``/root/bench-int8-w4a16-hbm/m4-final/tuning_hashes_hbm.json`` is used
instead (HBM-mission carry-forward; VAL-CROSS-002).

When ``--hbm-fa`` is passed, the manifest at
``/root/bench-int8-w4a16-hbm-fa/m3-final/tuning_hashes_hbm_fa.json`` is
used (Flash-Decoding-tuning mission carry-forward; VAL-CROSS-002 of the
HBM-FA mission). The HBM-FA manifest carries every entry from the prior
HBM manifest forward unchanged AND adds a single new entry for the
``winners_lookup.json`` produced by M1 of this mission.

Exit code is 0 if every recorded hash matches the on-disk content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(
    "/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/cold-points-sit-rancb"
)  # noqa: E501
TUNING_DIR = REPO / "vllm" / "model_executor" / "kernels" / "configs" / "gfx908"
PRIOR_MANIFEST = Path("/root/bench-int8-w4a16/final/tuning_hashes.json")
HBM_MANIFEST = Path("/root/bench-int8-w4a16-hbm/m4-final/tuning_hashes_hbm.json")
HBM_FA_MANIFEST = Path(
    "/root/bench-int8-w4a16-hbm-fa/m3-final/tuning_hashes_hbm_fa.json"
)
HBM_FA_WINNERS_LOOKUP = Path(
    "/root/bench-int8-w4a16-hbm-fa/m1-tuning/winners_lookup.json"
)
TENSILELITE_LIBPATH = Path("/root/bench-int8-w4a16/tensilelite/merged_library/library")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def collect(include_hbm_fa: bool = False) -> dict[str, str]:
    out: dict[str, str] = {}
    for f in sorted(TUNING_DIR.glob("*.json")):
        out[str(f.relative_to(REPO))] = sha256(f)
    if TENSILELITE_LIBPATH.exists():
        for f in sorted(TENSILELITE_LIBPATH.glob("*.dat")):
            out[str(f)] = sha256(f)
        for f in sorted(TENSILELITE_LIBPATH.glob("*.yaml")):
            out[str(f)] = sha256(f)
    if include_hbm_fa and HBM_FA_WINNERS_LOOKUP.exists():
        out[str(HBM_FA_WINNERS_LOOKUP)] = sha256(HBM_FA_WINNERS_LOOKUP)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hbm",
        action="store_true",
        help="Verify against the HBM-mission manifest "
        "(/root/bench-int8-w4a16-hbm/m4-final/tuning_hashes_hbm.json) "
        "instead of the prior mission's manifest.",
    )
    parser.add_argument(
        "--hbm-fa",
        action="store_true",
        help="Verify against the HBM-FA-mission manifest "
        "(/root/bench-int8-w4a16-hbm-fa/m3-final/tuning_hashes_hbm_fa.json). "
        "This manifest carries every entry from the prior HBM manifest forward "
        "and adds a single new entry for winners_lookup.json.",
    )
    args = parser.parse_args()
    if args.hbm and args.hbm_fa:
        print("[error] --hbm and --hbm-fa are mutually exclusive")
        return 2
    if args.hbm_fa:
        MANIFEST = HBM_FA_MANIFEST
    elif args.hbm:
        MANIFEST = HBM_MANIFEST
    else:
        MANIFEST = PRIOR_MANIFEST

    current = collect(include_hbm_fa=args.hbm_fa)

    # When initializing the --hbm-fa manifest, carry every entry from the
    # prior HBM manifest forward verbatim and ADD the winners_lookup.json
    # hash from disk. This makes the HBM-FA manifest a strict superset of
    # the HBM manifest (68 prior + 1 new = 69 entries).
    if args.hbm_fa and not MANIFEST.exists():
        if not HBM_MANIFEST.exists():
            print(f"[error] HBM manifest missing at {HBM_MANIFEST}")
            return 2
        carry = json.loads(HBM_MANIFEST.read_text())
        if not HBM_FA_WINNERS_LOOKUP.exists():
            print(f"[error] winners_lookup.json missing at {HBM_FA_WINNERS_LOOKUP}")
            return 2
        new_entry_key = str(HBM_FA_WINNERS_LOOKUP)
        new_entry_hash = sha256(HBM_FA_WINNERS_LOOKUP)
        if new_entry_key in carry and carry[new_entry_key] != new_entry_hash:
            print(
                f"[error] HBM manifest already pins {new_entry_key} with a "
                f"different hash"
            )
            return 2
        carry[new_entry_key] = new_entry_hash
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps(carry, indent=2))
        print(
            f"[init] Wrote --hbm-fa manifest with {len(carry)} entries "
            f"({len(carry) - 1} carried + 1 new winners_lookup.json) → {MANIFEST}"
        )

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
    if args.hbm_fa:
        tag = "VAL-CROSS-002 (HBM-FA)"
    elif args.hbm:
        tag = "VAL-CROSS-002"
    else:
        tag = "VAL-CROSS-007"
    print(
        f"{tag}: scanned {total} pinned entries from {MANIFEST}; "
        f"{fail} mismatch(es), {missing} missing, {new} new (un-pinned)."
    )
    return 0 if (fail == 0 and missing == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
