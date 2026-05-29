#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M4-F1: synthesize raw AWQ-vs-GPTQ A/B artifacts into report fragments.

Ingests:
  - 36 raw bench JSONs (M2-F1):
      /root/bench-w4a16-ab/{a,b,c}/<cell>_<workload>.json
      cells: w4a16_tp{1,4}_c{1,2,4}; workloads: synthetic, coding
  - 9 quality JSONs (M1):
      /root/bench-w4a16-ab/quality/{a,b,c}_{perplexity,needle,coding}.json
  - gate summary (M1-F4):
      /root/bench-w4a16-ab/quality/gate_summary.json
  - path A drift check (M2-F4):
      /root/bench-w4a16-ab/a_drift_check.json
  - disable-path smoke (M2-F3):
      /root/bench-w4a16-ab/disable_path_smoke/{a,b,c}_smoke.json
  - rocprof hot-kernel summary (M3-F2):
      /root/bench-w4a16-ab/rocprof/hot_kernel_summary.csv

Emits:
  - Markdown table fragments mirroring architecture.md §reporting sections
    to /tmp/awq_ab_tables.md
  - Verdict + #45 repack target recommendation to /tmp/awq_ab_verdict.json

Verdict matrix (pre-declared in architecture.md, ±3% threshold on
geomean decode-synthetic delta across 6 cells; coding geomean is reported
alongside for context):
  - A wins by ≥+3% over both B and C       -> GPTQ-WINS
  - B wins by ≥+3% AND path B quality OK    -> AWQ-CALIBRATION-WINS
  - C wins by ≥+3% AND path C quality OK    -> AWQ-FULL-STACK-WINS
  - All three within ±3% on tput            -> NULL
  - B/C wins on quality within ±3% on tput  -> QUALITY-WIN
  - A path fails a quality gate             -> excluded from headline
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

# ----------------------------------------------------------------------
# Inputs / outputs
# ----------------------------------------------------------------------

BENCH_ROOT = Path("/root/bench-w4a16-ab")
PATHS = ("a", "b", "c")
CELLS = (
    "w4a16_tp1_c1",
    "w4a16_tp1_c2",
    "w4a16_tp1_c4",
    "w4a16_tp4_c1",
    "w4a16_tp4_c2",
    "w4a16_tp4_c4",
)
WORKLOADS = ("synthetic", "coding")

OUT_TABLES = Path("/tmp/awq_ab_tables.md")
OUT_VERDICT = Path("/tmp/awq_ab_verdict.json")

VERDICT_THRESHOLD_PCT = 3.0

PATH_LABEL = {
    "a": "A (GPTQ W4A16)",
    "b": "B (AWQ-compressed-tensors)",
    "c": "C (AWQ-gemm + Triton AWQ)",
}
PATH_MODEL = {
    "a": "/models/Qwen3.5-9B-w4a16",
    "b": "/models/Qwen3.5-9B-AWQ-INT4",
    "c": "/models/Qwen3.5-9B-AWQ-gemm",
}
PATH_GROUP_SIZE = {"a": 128, "b": 32, "c": 128}
PATH_KERNEL = {
    "a": "mi100_w4a16_gemm_kernel (tuned, in-tree)",
    "b": "mi100_w4a16_gemm_kernel (tuned, in-tree)",
    "c": "awq_dequantize_kernel + awq_gemm_kernel (Triton AWQ, untuned)",
}

# Verdict matrix → repack target recommendation for issue #45
VERDICT_TO_REPACK = {
    "GPTQ-WINS": "GPTQ",
    "AWQ-CALIBRATION-WINS": "AWQ-compressed-tensors",
    "AWQ-FULL-STACK-WINS": "AWQ-gemm",
    "NULL": "GPTQ",
    "QUALITY-WIN": "GPTQ",
}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def load_json(p: Path) -> Any:
    with p.open() as f:
        return json.load(f)


def geomean(values: list[float]) -> float:
    if not values:
        return float("nan")
    s = 0.0
    for v in values:
        if v <= 0:
            return float("nan")
        s += math.log(v)
    return math.exp(s / len(values))


def pct(x: float, digits: int = 2) -> str:
    if math.isnan(x):
        return "n/a"
    return f"{x:+.{digits}f}%"


def num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


# ----------------------------------------------------------------------
# Ingest
# ----------------------------------------------------------------------


def ingest_bench() -> dict[tuple[str, str, str], dict]:
    """Returns mapping (path, cell, workload) -> bench dict."""
    out: dict[tuple[str, str, str], dict] = {}
    for path in PATHS:
        for cell in CELLS:
            for wl in WORKLOADS:
                fp = BENCH_ROOT / path / f"{cell}_{wl}.json"
                if not fp.is_file():
                    raise SystemExit(f"missing bench JSON: {fp}")
                out[(path, cell, wl)] = load_json(fp)
    return out


def ingest_quality() -> dict[str, dict[str, dict]]:
    """Returns {path: {kind: data}}."""
    out: dict[str, dict[str, dict]] = {}
    for path in PATHS:
        out[path] = {}
        for kind in ("perplexity", "needle", "coding"):
            fp = BENCH_ROOT / "quality" / f"{path}_{kind}.json"
            if not fp.is_file():
                raise SystemExit(f"missing quality JSON: {fp}")
            out[path][kind] = load_json(fp)
    return out


def ingest_gate_summary() -> dict:
    return load_json(BENCH_ROOT / "quality" / "gate_summary.json")


def ingest_drift() -> dict:
    return load_json(BENCH_ROOT / "a_drift_check.json")


def ingest_disable_smoke() -> dict[str, dict]:
    return {
        p: load_json(BENCH_ROOT / "disable_path_smoke" / f"{p}_smoke.json")
        for p in PATHS
    }


def ingest_hot_kernel() -> tuple[list[str], list[dict]]:
    fp = BENCH_ROOT / "rocprof" / "hot_kernel_summary.csv"
    rows: list[dict] = []
    headers: list[str] = []
    with fp.open() as f:
        reader = csv.reader(f)
        for raw in reader:
            if not raw or raw[0].startswith("#"):
                continue
            if not headers:
                headers = raw
                continue
            row = dict(zip(headers, raw))
            rows.append(row)
    return headers, rows


# ----------------------------------------------------------------------
# Geomean delta computations (B vs A, C vs A)
# ----------------------------------------------------------------------


def per_cell_ratios(
    bench: dict, num_path: str, den_path: str, workload: str
) -> list[float]:
    ratios = []
    for cell in CELLS:
        n = bench[(num_path, cell, workload)]["output_throughput_toks_s"]
        d = bench[(den_path, cell, workload)]["output_throughput_toks_s"]
        ratios.append(n / d)
    return ratios


def geomean_delta_pct(
    bench: dict, num_path: str, den_path: str, workload: str
) -> float:
    r = per_cell_ratios(bench, num_path, den_path, workload)
    return (geomean(r) - 1.0) * 100.0


# ----------------------------------------------------------------------
# Verdict logic
# ----------------------------------------------------------------------


def pick_verdict(
    b_syn: float, c_syn: float, b_cod: float, c_cod: float, gates: dict
) -> tuple[str, str]:
    """Apply the pre-declared matrix on synthetic geomean deltas.

    Returns (verdict_label, rationale).
    """
    th = VERDICT_THRESHOLD_PCT
    b_ok = bool(gates["paths"]["b"].get("path_gate_passed"))
    c_ok = bool(gates["paths"]["c"].get("path_gate_passed"))

    # Headline workload = synthetic decode
    candidates: list[tuple[float, str, str]] = []
    if b_syn >= th and b_ok:
        candidates.append(
            (
                b_syn,
                "AWQ-CALIBRATION-WINS",
                f"B beats A by {pct(b_syn)} (≥+{th:.0f}%) on synthetic-decode "
                "geomean with quality preserved",
            )
        )
    if c_syn >= th and c_ok:
        candidates.append(
            (
                c_syn,
                "AWQ-FULL-STACK-WINS",
                f"C beats A by {pct(c_syn)} (≥+{th:.0f}%) on synthetic-decode "
                "geomean with quality preserved",
            )
        )
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        return candidates[0][1], candidates[0][2]

    # A wins by ≥+3% over BOTH B and C
    if -b_syn >= th and -c_syn >= th:
        return (
            "GPTQ-WINS",
            f"A beats both B ({pct(b_syn)}) and C ({pct(c_syn)}) by ≥{th:.0f}% on "
            "synthetic-decode geomean",
        )

    # All three within ±3% on tput?
    if abs(b_syn) <= th and abs(c_syn) <= th:
        # Check for QUALITY-WIN: any non-A path improves perplexity AND
        # gate-passes
        per_a = gates["paths"]["a"]["perplexity"]
        for cand, lbl in (("b", "AWQ-CALIBRATION-WINS"), ("c", "AWQ-FULL-STACK-WINS")):
            per = gates["paths"][cand]["perplexity"]
            ok = gates["paths"][cand].get("path_gate_passed")
            # ≥1% lower perplexity = improvement
            if ok and per_a > 0 and (per_a - per) / per_a * 100.0 >= 1.0:
                return (
                    "QUALITY-WIN",
                    f"Path {cand.upper()} improves perplexity by "
                    f"{(per_a - per) / per_a * 100.0:.2f}%"
                    f" while tput within ±{th:.0f}%",
                )
        return (
            "NULL",
            f"All three paths within ±{th:.0f}% on synthetic-decode "
            f"geomean (B={pct(b_syn)}, C={pct(c_syn)})",
        )

    # Mixed: A beats one but not both by ≥3%. Closest matrix cell is
    # GPTQ-WINS since A is still the leader; cite the mix.
    return (
        "GPTQ-WINS",
        f"A leads on synthetic-decode geomean (B={pct(b_syn)}, "
        f"C={pct(c_syn)}); A clearly beats at least one path by ≥{th:.0f}% "
        f"and no non-A path beats A by ≥{th:.0f}%",
    )


# ----------------------------------------------------------------------
# Markdown emit
# ----------------------------------------------------------------------


def emit_tables(
    bench: dict,
    quality: dict,
    gates: dict,
    drift: dict,
    disable_smoke: dict,
    hot_headers: list[str],
    hot_rows: list[dict],
    geo: dict,
    verdict_label: str,
    verdict_rationale: str,
    repack_target: str,
) -> str:
    lines: list[str] = []
    a = lines.append

    # 1. Headline verdict
    a("## 1. Headline verdict\n")
    a(f"**Verdict label:** `{verdict_label}`  ")
    a(f"**#45 repack target recommendation:** `{repack_target}`\n")
    a(verdict_rationale + ".\n")
    a("Synthetic-decode geomean deltas (6 cells, equal weights):\n")
    a(f"- B vs A: {pct(geo['b_vs_a_synthetic'])}")
    a(f"- C vs A: {pct(geo['c_vs_a_synthetic'])}\n")
    a("Coding-workload geomean deltas (6 cells, equal weights):\n")
    a(f"- B vs A: {pct(geo['b_vs_a_coding'])}")
    a(f"- C vs A: {pct(geo['c_vs_a_coding'])}\n")

    # 2. Hardware / software manifest — leave as a placeholder hook
    a("## 2. Hardware / software manifest\n")
    a("> Populated by report author; this script emits the bench-derived rows below.\n")

    # 3. Three-path quant config decomposition
    a("## 3. Three-path quant config decomposition\n")
    a("| Path | Model | Group size | Kernel dispatch |")
    a("| :--- | :--- | ---: | :--- |")
    for p in PATHS:
        a(
            f"| {PATH_LABEL[p]} | `{PATH_MODEL[p]}`"
            f" | {PATH_GROUP_SIZE[p]} | {PATH_KERNEL[p]} |"
        )
    a("")

    # 4. Setup + env identity proof
    a("## 4. Setup + bench env identity proof\n")
    # Sample first cell env for each path
    a(
        "Per-cell env captures live under "
        "`/root/bench-w4a16-ab/<path>/env_<cell>_<workload>.json`."
    )
    a("Sample (path A, `w4a16_tp1_c1_synthetic`):\n")
    env_sample = load_json(BENCH_ROOT / "a" / "env_w4a16_tp1_c1_synthetic.json")
    a("| Key | Value |")
    a("| :--- | :--- |")
    for k in (
        "KV_CACHE_DTYPE",
        "ENABLE_CHUNKED_PREFILL",
        "MAX_NUM_BATCHED_TOKENS",
        "NCCL_ALGO",
        "HIP_VISIBLE_DEVICES",
        "vllm_commit",
    ):
        a(f"| `{k}` | `{env_sample.get(k)}` |")
    a(
        "\nGPU GUID parity log: `/root/bench-w4a16-ab/gpu_guid_parity.log` "
        "(expected GUIDs {4106, 57403, 45163, 5017}).\n"
    )

    # 5. Quality gates
    a("## 5. Quality gates\n")
    a(
        f"FP16 reference perplexity: **{gates['fp16_reference_perplexity']}** "
        f"({gates['fp16_reference_source']}); threshold ≤ "
        f"{gates['thresholds']['perplexity_max']:.4f} "
        f"(+{gates['thresholds']['perplexity_tolerance_pct']:.0f}%).\n"
    )
    a(
        "| Path | perplexity | Δ vs FP16 | perplexity_ok"
        " | needle | needle_ok | coding | coding_ok | path_gate_passed |"
    )
    a("| :--- | ---: | ---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for p in PATHS:
        g = gates["paths"][p]
        a(
            f"| {PATH_LABEL[p]} | {num(g['perplexity'], 4)} | "
            f"{pct(g['perplexity_delta_pct_vs_fp16'])} | "
            f"{'✅' if g['perplexity_ok'] else '❌'} | "
            f"{g['needle_passed']}/{g['needle_total']} | "
            f"{'✅' if g['needle_ok'] else '❌'} | "
            f"{g['coding_passed']}/{g['coding_total']} | "
            f"{'✅' if g['coding_ok'] else '❌'} | "
            f"{'✅' if g['path_gate_passed'] else '❌'} |"
        )
    excluded = gates.get("excluded_from_headline", [])
    if excluded:
        a(
            f"\n_Paths excluded from headline (failed at least one gate):_ "
            f"`{', '.join(excluded)}`. Per M1-F4 the mission still publishes "
            "their bench numbers; the verdict matrix flags them.\n"
        )

    # 6. 36-cell throughput summary
    a("## 6. 36-cell throughput summary (output_throughput_toks_s)\n")
    for wl in WORKLOADS:
        a(f"### {wl.capitalize()} workload\n")
        a("| Cell | A | B | C |")
        a("| :--- | ---: | ---: | ---: |")
        for cell in CELLS:
            row = [cell]
            for p in PATHS:
                row.append(num(bench[(p, cell, wl)]["output_throughput_toks_s"]))
            a("| " + " | ".join(row) + " |")
        a("")

    # 7. 24-cell delta tables: B vs A, C vs A (per-cell %Δ tput + Δ p99 TPOT + Δ TTFT)
    a("## 7. Per-cell deltas: B vs A and C vs A\n")
    for cand in ("b", "c"):
        a(f"### {PATH_LABEL[cand]} vs {PATH_LABEL['a']}\n")
        for wl in WORKLOADS:
            a(f"#### {wl.capitalize()}\n")
            a("| Cell | Δ tput % | Δ p99 TPOT ms | Δ TTFT ms |")
            a("| :--- | ---: | ---: | ---: |")
            for cell in CELLS:
                rA = bench[("a", cell, wl)]
                rX = bench[(cand, cell, wl)]
                d_tput = (
                    rX["output_throughput_toks_s"] / rA["output_throughput_toks_s"]
                    - 1.0
                ) * 100.0
                d_p99 = rX["p99_tpot_ms"] - rA["p99_tpot_ms"]
                d_ttft = rX["mean_ttft_ms"] - rA["mean_ttft_ms"]
                a(f"| {cell} | {pct(d_tput)} | {d_p99:+.3f} | {d_ttft:+.2f} |")
            a("")

    # 8. rocprofv3 evidence
    a("## 8. rocprofv3 evidence: top hot kernels per path (TP=1 c1 synthetic)\n")
    a(
        "_Source: `/root/bench-w4a16-ab/rocprof/hot_kernel_summary.csv`. "
        "TP=4 rows omitted per M3-F1 SIGTERM-flush race (see "
        "`/root/bench-w4a16-ab/rocprof/tp4_known_issue.md`)._\n"
    )
    a(
        "| path | cell | kernel_name | invocations"
        " | hbm_bw_gbps | hbm_util_pct | mfma_util_pct |"
    )
    a("| :--- | :--- | :--- | ---: | ---: | ---: | ---: |")
    for r in hot_rows:
        # truncate Tensile kernel names to first 50 chars + ellipsis
        kn = r["kernel_name"]
        if len(kn) > 60:
            kn = kn[:60] + "…"
        # escape pipes in kernel_name to avoid breaking markdown table
        kn = kn.replace("|", "\\|")
        a(
            f"| {r['path']} | {r['cell']} | `{kn}` | {r['invocations']} | "
            f"{r['hbm_bw_gbps']} | {r['hbm_util_pct']} | {r['mfma_util_pct']} |"
        )
    a("")

    # 9. Disable-path smoke results
    a("## 9. Disable-path smoke results (`VLLM_MI100_DISABLE_FUSED_ACT_QUANT=1`)\n")
    a(
        "Re-run of `w4a16_tp1_c1_synthetic` per path with the fused act-quant "
        "epilogue disabled; tolerance ±5%.\n"
    )
    a("| Path | Baseline tput | Disable-path tput | Δ % | Passed |")
    a("| :--- | ---: | ---: | ---: | :---: |")
    for p in PATHS:
        s = disable_smoke[p]
        a(
            f"| {PATH_LABEL[p]} | {num(s['baseline_output_throughput_toks_s'])} | "
            f"{num(s['output_throughput_toks_s'])} | {pct(s['delta_pct'])} | "
            f"{'✅' if s['passed'] else '❌'} |"
        )
    a("")

    # 10. Path A drift check
    a("## 10. Path A drift check vs `BENCH_INT8_W4A16_HBM.md` §11.1 (±2%)\n")
    a(
        f"Overall: {'✅ PASSED' if drift['drift_check_passed'] else '❌ FAILED'}; "
        f"{drift['cells_within_gate']}/{drift['cells_total']} cells within gate; "
        f"max |Δ| = {drift['max_abs_delta_pct']:.4f}%.\n"
    )
    a("| Cell | Ref tput | Measured tput | Δ % | Within ±2% |")
    a("| :--- | ---: | ---: | ---: | :---: |")
    for r in drift["per_cell"]:
        a(
            f"| {r['cell']} | {num(r['ref_tput_toks_s'])} | "
            f"{num(r['measured_tput_toks_s'])} | {pct(r['delta_pct'], 4)} | "
            f"{'✅' if r['within_2pct'] else '❌'} |"
        )
    a("")

    # 11. Verdict matrix application
    a("## 11. Verdict matrix application\n")
    a(f"**Verdict:** `{verdict_label}`.\n")
    a(verdict_rationale + ".\n")
    a(
        "Pre-declared ±3% threshold applied. Geomean deltas "
        "(synthetic decode, 6 cells):\n"
    )
    a(f"- B vs A: **{pct(geo['b_vs_a_synthetic'])}**")
    a(f"- C vs A: **{pct(geo['c_vs_a_synthetic'])}**\n")
    a("Coding workload (6 cells) for context:\n")
    a(f"- B vs A: {pct(geo['b_vs_a_coding'])}")
    a(f"- C vs A: {pct(geo['c_vs_a_coding'])}\n")
    if gates.get("excluded_from_headline"):
        a(
            f"_Quality-gate caveat: paths {gates['excluded_from_headline']} failed at "
            "least one quality gate; their tput numbers are reported but flagged._\n"
        )

    # 12. Recommendation for issue #45
    a("## 12. Recommendation for issue #45\n")
    a(f"**Repack target:** `{repack_target}`.\n")
    a("Rationale: " + verdict_rationale + ".\n")

    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> int:
    bench = ingest_bench()
    quality = ingest_quality()
    gates = ingest_gate_summary()
    drift = ingest_drift()
    disable_smoke = ingest_disable_smoke()
    hot_headers, hot_rows = ingest_hot_kernel()

    geo = {
        "b_vs_a_synthetic": geomean_delta_pct(bench, "b", "a", "synthetic"),
        "c_vs_a_synthetic": geomean_delta_pct(bench, "c", "a", "synthetic"),
        "b_vs_a_coding": geomean_delta_pct(bench, "b", "a", "coding"),
        "c_vs_a_coding": geomean_delta_pct(bench, "c", "a", "coding"),
    }

    verdict_label, verdict_rationale = pick_verdict(
        geo["b_vs_a_synthetic"],
        geo["c_vs_a_synthetic"],
        geo["b_vs_a_coding"],
        geo["c_vs_a_coding"],
        gates,
    )
    repack_target = VERDICT_TO_REPACK[verdict_label]

    md = emit_tables(
        bench=bench,
        quality=quality,
        gates=gates,
        drift=drift,
        disable_smoke=disable_smoke,
        hot_headers=hot_headers,
        hot_rows=hot_rows,
        geo=geo,
        verdict_label=verdict_label,
        verdict_rationale=verdict_rationale,
        repack_target=repack_target,
    )
    OUT_TABLES.write_text(md)

    verdict_payload = {
        "verdict_label": verdict_label,
        "verdict_rationale": verdict_rationale,
        "verdict_threshold_pct": VERDICT_THRESHOLD_PCT,
        "geomean_deltas_pct": {
            "synthetic": {
                "b_vs_a": geo["b_vs_a_synthetic"],
                "c_vs_a": geo["c_vs_a_synthetic"],
            },
            "coding": {
                "b_vs_a": geo["b_vs_a_coding"],
                "c_vs_a": geo["c_vs_a_coding"],
            },
        },
        "issue_45_repack_target": repack_target,
        "quality_gate_excluded_paths": gates.get("excluded_from_headline", []),
        "inputs": {
            "bench_root": str(BENCH_ROOT),
            "n_bench_jsons": len(bench),
            "n_quality_jsons": sum(len(v) for v in quality.values()),
            "gate_summary": str(BENCH_ROOT / "quality" / "gate_summary.json"),
            "drift_check": str(BENCH_ROOT / "a_drift_check.json"),
            "disable_smoke": [
                str(BENCH_ROOT / "disable_path_smoke" / f"{p}_smoke.json")
                for p in PATHS
            ],
            "hot_kernel_csv": str(BENCH_ROOT / "rocprof" / "hot_kernel_summary.csv"),
        },
        "outputs": {
            "tables_md": str(OUT_TABLES),
            "verdict_json": str(OUT_VERDICT),
        },
    }
    OUT_VERDICT.write_text(json.dumps(verdict_payload, indent=2) + "\n")

    print(f"verdict: {verdict_label}")
    print(f"#45 repack target: {repack_target}")
    print("Synthetic geomean deltas (6 cells):")
    print(f"  B vs A: {pct(geo['b_vs_a_synthetic'])}")
    print(f"  C vs A: {pct(geo['c_vs_a_synthetic'])}")
    print("Coding geomean deltas (6 cells):")
    print(f"  B vs A: {pct(geo['b_vs_a_coding'])}")
    print(f"  C vs A: {pct(geo['c_vs_a_coding'])}")
    print(f"wrote {OUT_TABLES} ({OUT_TABLES.stat().st_size} bytes)")
    print(f"wrote {OUT_VERDICT} ({OUT_VERDICT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
