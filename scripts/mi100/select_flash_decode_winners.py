#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M1 — Winner selection from the Triton Flash-Decoding sweep results.

Reads ``sweep_results.json`` produced by
``scripts/mi100/triton_flash_decode_sweep.py`` and emits a per-shape
winners lookup table consumed by the runtime wire-in feature
(``m1-wire-in-lookup``).

Group key: ``(quant, head_dim, seq_len_bucket, num_seqs)``. The mission
sanity check (feature description, step 6) requires that there be at
least one winner row for each ``(quant, seq_len_bucket)`` in
``{w8a8, w4a16} × {1024, 4096, 16384, 32768}`` (8 pairs); ``num_seqs``
is preserved in the key so the runtime lookup can pick the right config
for a given decode batch shape.

Eligibility (per the m1-autotune-sweep handoff clarification baked into
the feature description):

    eligible == True AND rel_err <= 1e-3

The sweep records both ``abs_err`` and ``rel_err`` per trial. The
reference uses ``num_splits=8`` which is NOT in the swept set
``{16, 32, 60, 120}``; trials with ``num_splits ∈ {16, 32}`` therefore
use a different fp32 reduction order than the reference, and at output
magnitudes ~60 the fp16 ULP is ~0.008 — above the 1e-3 *absolute*
threshold but ~1e-4 *relative* (numerically correct). Trials with
``num_splits ∈ {60, 120}`` have ``eligible == False`` due to a Triton
3.5.1 ``reduce_segments`` power-of-2 constraint; they fall out
naturally and winners only ever cite ``num_splits ∈ {16, 32}``.

Winner per group: minimum ``median_ms``. Ties (exact-float ``median_ms``
equality) are broken in this deterministic priority order, for
predictability under future re-tunes:

    1. smaller ``tile_size``
    2. smaller ``num_splits``
    3. smaller ``num_warps``
    4. smaller ``num_stages``     (extra deterministic key)
    5. smaller ``BLOCK_M``        (extra deterministic key)

Steps 4–5 are not in the feature spec but are needed to make the
selection truly deterministic across any conceivable tie pattern; the
feature spec's three keys are honored first.

Outputs (deterministic, byte-stable across reruns):

    /root/bench-int8-w4a16-hbm-fa/m1-tuning/winners_lookup.json
    /root/bench-int8-w4a16-hbm-fa/m1-tuning/winners_summary.md
    /root/bench-int8-w4a16-hbm-fa/m1-tuning/m1_winners.log (stdout)

Speedup is reported vs the production-default config recorded in the
sweep metadata (``ref_config``: tile_size=32, num_splits=8, BLOCK_M=16,
num_warps=4, num_stages=2). Because ``num_splits=8`` was never swept,
the default-config latency PER SHAPE is read from the sweep's
``default_config_latency`` shape table if present, and otherwise marked
``null`` (the M1 bench grid is the authoritative throughput comparison
vs M4 — this column is informational).

Usage::

    /opt/vllm-env/bin/python3 scripts/mi100/select_flash_decode_winners.py \
        --sweep /root/bench-int8-w4a16-hbm-fa/m1-tuning/sweep_results.json \
        --out   /root/bench-int8-w4a16-hbm-fa/m1-tuning/winners_lookup.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

# Constants pinned by the feature description.
REL_ERR_THRESHOLD: float = 1e-3
SELECTION_RULE: str = (
    "min median_ms among eligible (rel_err <= 1e-3); "
    "tiebreak smaller tile_size > num_splits > num_warps "
    "(secondary deterministic keys: num_stages, BLOCK_M)"
)

REQUIRED_QUANTS: tuple[str, ...] = ("w8a8", "w4a16")
REQUIRED_SEQ_LEN_BUCKETS: tuple[int, ...] = (1024, 4096, 16384, 32768)


def _sha256_of_file(path: Path) -> str:
    """Return the hex SHA-256 of the file at ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _group_key(shape: dict[str, Any]) -> tuple[str, int, int, int]:
    """Stable tuple key for grouping records."""
    return (
        shape["quant"],
        int(shape["head_dim"]),
        int(shape["seq_len_bucket"]),
        int(shape["num_seqs"]),
    )


def _tiebreak_key(config: dict[str, Any]) -> tuple[int, int, int, int, int]:
    """Deterministic ordering key applied AFTER median_ms.

    Smaller is preferred at every position. Order:

        tile_size, num_splits, num_warps, num_stages, BLOCK_M
    """
    return (
        int(config["tile_size"]),
        int(config["num_splits"]),
        int(config["num_warps"]),
        int(config["num_stages"]),
        int(config["BLOCK_M"]),
    )


def _winner_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    """Composite sort key: median_ms first, then tie-breakers."""
    return (float(record["median_ms"]), *_tiebreak_key(record["config"]))


def select_winners(
    sweep: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, int, int], float | None]]:
    """Group eligible records by ``_group_key`` and pick a winner per group.

    Returns ``(winners, default_latency_by_group)``. The default-latency
    map is best-effort (None when not available in the sweep).
    """
    configs: list[dict[str, Any]] = sweep["configs"]
    eligible: list[dict[str, Any]] = [
        rec
        for rec in configs
        if rec.get("eligible") is True
        and rec.get("rel_err") is not None
        and float(rec["rel_err"]) <= REL_ERR_THRESHOLD
    ]

    # Pre-compute the production-default latency per shape, if the sweep
    # happened to include it. We look for any eligible record whose
    # config matches the metadata-declared ref_config.
    ref_config = sweep.get("metadata", {}).get("ref_config") or {}
    default_latency_by_group: dict[tuple[str, int, int, int], float | None] = {}
    if ref_config:
        for rec in configs:
            if all(rec["config"].get(k) == ref_config[k] for k in ref_config):
                key = _group_key(rec["shape"])
                if key not in default_latency_by_group:
                    default_latency_by_group[key] = float(rec["median_ms"])

    groups: dict[tuple[str, int, int, int], list[dict[str, Any]]] = {}
    for rec in eligible:
        groups.setdefault(_group_key(rec["shape"]), []).append(rec)

    winners: list[dict[str, Any]] = []
    # Iterate group keys in a deterministic order: by quant, head_dim,
    # seq_len_bucket, num_seqs (all ascending; quant as string).
    for key in sorted(groups.keys()):
        candidates = groups[key]
        # `sorted` is a stable sort, but we feed a full composite key so
        # the order is fully determined by the key tuple.
        candidates.sort(key=_winner_sort_key)
        best = candidates[0]
        quant, head_dim, seq_len_bucket, num_seqs = key
        default_ms = default_latency_by_group.get(key)
        speedup = (
            (default_ms / float(best["median_ms"]))
            if (default_ms is not None and float(best["median_ms"]) > 0.0)
            else None
        )
        winners.append(
            {
                "key": {
                    "quant": quant,
                    "head_dim": head_dim,
                    "seq_len_bucket": seq_len_bucket,
                    "num_seqs": num_seqs,
                },
                "config": {
                    "tile_size": int(best["config"]["tile_size"]),
                    "num_splits": int(best["config"]["num_splits"]),
                    "num_warps": int(best["config"]["num_warps"]),
                    "num_stages": int(best["config"]["num_stages"]),
                    "BLOCK_M": int(best["config"]["BLOCK_M"]),
                },
                "median_ms": float(best["median_ms"]),
                "median_latency_us": float(best["median_ms"]) * 1e3,
                "hbm_bytes_per_inv": int(best["hbm_bytes_per_inv"]),
                "abs_err": float(best["abs_err"]),
                "rel_err": float(best["rel_err"]),
                "default_median_ms": default_ms,
                "speedup_vs_default": speedup,
                "eligible_candidates": len(candidates),
            }
        )

    return winners, default_latency_by_group


def sanity_check(winners: list[dict[str, Any]]) -> list[str]:
    """Verify every (quant, seq_len_bucket) in the required cross-product
    has at least one winner row.

    Returns a list of human-readable problem strings; empty list ⇒ OK.
    """
    have: set[tuple[str, int]] = {
        (w["key"]["quant"], w["key"]["seq_len_bucket"]) for w in winners
    }
    missing: list[str] = []
    for quant in REQUIRED_QUANTS:
        for slb in REQUIRED_SEQ_LEN_BUCKETS:
            if (quant, slb) not in have:
                missing.append(
                    f"  - quant={quant} seq_len_bucket={slb}: NO ELIGIBLE CONFIG"
                )
    return missing


def render_summary_markdown(
    winners: list[dict[str, Any]],
    sweep_meta: dict[str, Any],
    sweep_path: Path,
    sweep_sha: str,
) -> str:
    """Render the human-readable winners summary markdown."""
    lines: list[str] = []
    lines.append("# M1 Triton Flash-Decoding Winners Summary")
    lines.append("")
    lines.append(f"- Source sweep: `{sweep_path}`")
    lines.append(f"- Sweep SHA-256: `{sweep_sha}`")
    lines.append(
        f"- Sweep trials (total expected / completed): "
        f"{sweep_meta.get('trials_total_expected', '?')} / "
        f"{sweep_meta.get('trials_completed', '?')}"
    )
    lines.append(f"- Selection rule: {SELECTION_RULE}")
    lines.append(f"- Eligibility threshold: rel_err ≤ {REL_ERR_THRESHOLD:g}")
    lines.append(
        f"- Production-default reference config: "
        f"`{json.dumps(sweep_meta.get('ref_config', {}), sort_keys=True)}`"
    )
    lines.append("")

    # One section per quant; rows ordered by (seq_len_bucket, num_seqs).
    for quant in REQUIRED_QUANTS:
        lines.append(f"## {quant}")
        lines.append("")
        lines.append(
            "| seq_len | num_seqs | tile | splits | warps | stages | BLOCK_M | "
            "median_ms | default_ms | speedup×default | hbm_bytes |"
        )
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        rows = sorted(
            (w for w in winners if w["key"]["quant"] == quant),
            key=lambda w: (
                w["key"]["seq_len_bucket"],
                w["key"]["num_seqs"],
            ),
        )
        for w in rows:
            cfg = w["config"]
            default_ms = w.get("default_median_ms")
            speedup = w.get("speedup_vs_default")
            default_str = f"{default_ms:.4f}" if default_ms is not None else "n/a"
            speedup_str = f"{speedup:.3f}" if speedup is not None else "n/a"
            lines.append(
                f"| {w['key']['seq_len_bucket']} | {w['key']['num_seqs']} | "
                f"{cfg['tile_size']} | {cfg['num_splits']} | "
                f"{cfg['num_warps']} | {cfg['num_stages']} | "
                f"{cfg['BLOCK_M']} | {w['median_ms']:.4f} | "
                f"{default_str} | {speedup_str} | "
                f"{w['hbm_bytes_per_inv']} |"
            )
        lines.append("")

    lines.append("## Sanity checks")
    lines.append("")
    missing = sanity_check(winners)
    if missing:
        lines.append("**FAIL** — some required groups have no eligible config:")
        lines.extend(missing)
    else:
        lines.append("PASS — all 8 (quant × seq_len_bucket) pairs have ≥ 1 winner row.")
    lines.append("")

    return "\n".join(lines)


def _emit_winners_json(
    out_path: Path,
    winners: list[dict[str, Any]],
    sweep_path: Path,
    sweep_sha: str,
) -> None:
    """Write the byte-deterministic ``winners_lookup.json``."""
    # Top-level field ordering preserved verbatim per feature spec.
    doc: OrderedDict[str, Any] = OrderedDict()
    doc["selection_rule"] = SELECTION_RULE
    doc["eligibility_rule"] = f"rel_err <= {REL_ERR_THRESHOLD:g}"
    doc["generated_from"] = sweep_path.name
    doc["sweep_sha256"] = sweep_sha
    doc["group_key_fields"] = [
        "quant",
        "head_dim",
        "seq_len_bucket",
        "num_seqs",
    ]
    doc["tiebreak_fields"] = [
        "tile_size",
        "num_splits",
        "num_warps",
        "num_stages",
        "BLOCK_M",
    ]
    doc["winners"] = winners

    payload = json.dumps(doc, indent=2, sort_keys=False)
    if not payload.endswith("\n"):
        payload += "\n"
    out_path.write_text(payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select per-shape winning Triton flash-decoding configs "
            "from sweep_results.json."
        )
    )
    parser.add_argument(
        "--sweep",
        type=Path,
        default=Path("/root/bench-int8-w4a16-hbm-fa/m1-tuning/sweep_results.json"),
        help="Path to sweep_results.json produced by the m1-autotune-sweep feature.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/root/bench-int8-w4a16-hbm-fa/m1-tuning/winners_lookup.json"),
        help="Path to write winners_lookup.json.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
        help=(
            "Optional override for the winners_summary.md path "
            "(default: alongside --out)."
        ),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help=("Optional override for m1_winners.log (default: alongside --out)."),
    )
    args = parser.parse_args(argv)

    sweep_path: Path = args.sweep
    out_path: Path = args.out
    summary_path: Path = args.summary or (out_path.parent / "winners_summary.md")
    log_path: Path = args.log or (out_path.parent / "m1_winners.log")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Capture stdout to log file as well as printing it.
    log_buf: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        log_buf.append(line)

    emit(f"[select_flash_decode_winners] sweep   = {sweep_path}")
    emit(f"[select_flash_decode_winners] out     = {out_path}")
    emit(f"[select_flash_decode_winners] summary = {summary_path}")
    emit(f"[select_flash_decode_winners] log     = {log_path}")
    emit("")

    if not sweep_path.is_file():
        emit(f"ERROR: sweep file not found at {sweep_path}")
        log_path.write_text("\n".join(log_buf) + "\n")
        return 2

    sweep_sha = _sha256_of_file(sweep_path)
    emit(f"sweep_sha256 = {sweep_sha}")

    with sweep_path.open("r") as f:
        sweep = json.load(f)

    n_configs = len(sweep["configs"])
    n_eligible = sum(
        1
        for rec in sweep["configs"]
        if rec.get("eligible") is True
        and rec.get("rel_err") is not None
        and float(rec["rel_err"]) <= REL_ERR_THRESHOLD
    )
    emit(f"sweep records: total={n_configs}, eligible (rel_err≤1e-3)={n_eligible}")

    winners, _ = select_winners(sweep)
    emit(f"winners selected: {len(winners)}")
    emit("")

    # Sanity check: every (quant, seq_len_bucket) in the required
    # cross-product must have at least one winner row.
    missing = sanity_check(winners)
    if missing:
        emit("SANITY CHECK FAILED — missing winners for:")
        for line in missing:
            emit(line)
        emit("")
        emit(
            "ESCALATE: kernel is broken for the missing shape(s); "
            "m1-wire-in-lookup MUST NOT run."
        )
        # Persist log and exit non-zero so the orchestrator can route.
        log_path.write_text("\n".join(log_buf) + "\n")
        return 3

    emit("sanity check PASS — all 8 (quant × seq_len_bucket) pairs have a winner row.")

    # Print per-quant winner rows for human audit (also lands in
    # m1_winners.log).
    emit("")
    for quant in REQUIRED_QUANTS:
        emit(f"-- {quant} --")
        rows = sorted(
            (w for w in winners if w["key"]["quant"] == quant),
            key=lambda w: (
                w["key"]["seq_len_bucket"],
                w["key"]["num_seqs"],
            ),
        )
        for w in rows:
            cfg = w["config"]
            default_ms = w.get("default_median_ms")
            speedup = w.get("speedup_vs_default")
            speedup_str = f"{speedup:6.3f}×" if speedup is not None else "  n/a "
            default_str = (
                f"{default_ms:8.4f}ms" if default_ms is not None else "    n/a   "
            )
            emit(
                f"  seq_len={w['key']['seq_len_bucket']:>5} "
                f"num_seqs={w['key']['num_seqs']} "
                f"tile={cfg['tile_size']:>3} splits={cfg['num_splits']:>3} "
                f"warps={cfg['num_warps']:>2} stages={cfg['num_stages']} "
                f"BLOCK_M={cfg['BLOCK_M']:>3} "
                f"median={w['median_ms']:8.4f}ms "
                f"default={default_str} "
                f"speedup={speedup_str}"
            )
        emit("")

    _emit_winners_json(out_path, winners, sweep_path, sweep_sha)
    emit(f"wrote {out_path} ({out_path.stat().st_size} bytes)")

    summary_md = render_summary_markdown(
        winners, sweep.get("metadata", {}), sweep_path, sweep_sha
    )
    summary_path.write_text(summary_md)
    emit(f"wrote {summary_path} ({summary_path.stat().st_size} bytes)")

    # Persist log last so the file captures the final write lines.
    log_path.write_text("\n".join(log_buf) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
