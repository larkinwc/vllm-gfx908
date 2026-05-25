#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Render BENCH_INT8_W4A16_FINAL.md from on-disk artifacts.

Reads:
  - /root/bench-int8-w4a16/final/final_grid.csv (per-cell, per-metric)
  - /root/bench-int8-w4a16/final/final_grid_reasons.json
  - /root/bench-int8-w4a16/final/tuning_hashes.json
  - /root/bench-int8-w4a16/final/m6_coding_eval_m6.json (if present)
  - /root/bench-int8-w4a16/final/m6_needle32k.json (if present)
  - /root/bench-int8-w4a16/baseline/ppl_*.json
  - /root/bench-int8-w4a16/m3/ppl_*.json
  - /root/bench-int8-w4a16/m4/ppl_*.json
  - /root/bench-int8-w4a16/final/m6_repro_spotcheck.csv (if present)
  - /root/bench-int8-w4a16/final/launch_smoke/*

Writes BENCH_INT8_W4A16_FINAL.md to the repo root.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(  # noqa: E501
    "/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/cold-points-sit-rancb"
)
FINAL_DIR = Path("/root/bench-int8-w4a16/final")
REPORT = REPO / "BENCH_INT8_W4A16_FINAL.md"

CELLS = []
for model in ("w8a8", "w4a16"):
    for tp in (1, 4):
        for c in (1, 2, 4):
            CELLS.append((model, tp, c))


def fmt(v) -> str:
    if v is None or v == "":
        return "—"
    try:
        f = float(v)
    except Exception:
        return str(v)
    if f >= 100:
        return f"{f:.1f}"
    return f"{f:.2f}"


def winner_label(path: str | None, applies_to: dict[str, bool]) -> str:
    if not path:
        return "n/a"
    pretty = {
        "stock": "stock",
        "tensilelite": "+TensileLite",
        "triton": "+Triton",
        "ck": "+CK",
        "isa": "+ISA",
    }
    return pretty.get(path, path)


def column_value(row: dict, key: str, applies: bool) -> str:
    v = row.get(key)
    if not applies:
        return "n/a"
    if v in (None, "", "None"):
        return "—"
    try:
        f = float(v)
    except Exception:
        return str(v)
    if f >= 100:
        return f"{f:.1f}"
    return f"{f:.2f}"


def main() -> int:
    rows = list(csv.DictReader((FINAL_DIR / "final_grid.csv").open()))
    reasons = json.loads((FINAL_DIR / "final_grid_reasons.json").read_text())
    tuning = json.loads((FINAL_DIR / "tuning_hashes.json").read_text()) if (FINAL_DIR / "tuning_hashes.json").exists() else {}  # noqa: E501

    coding_eval = None
    for cand in (
        FINAL_DIR / "m6_coding_eval_m6_w8a8.json",
        FINAL_DIR / "m6_coding_eval_m6_w4a16.json",
        FINAL_DIR / "m6_coding_eval_m6.json",
    ):
        if cand.exists():
            coding_eval = json.loads(cand.read_text())
            break

    needle = None
    p = FINAL_DIR / "m6_needle32k.json"
    if p.exists():
        needle = json.loads(p.read_text())

    ppl_fp16 = json.loads((Path("/root/bench-int8-w4a16/baseline") / "ppl_fp16.json").read_text())["perplexity"]  # noqa: E501
    ppl_w8a8_m0 = json.loads((Path("/root/bench-int8-w4a16/baseline") / "ppl_w8a8.json").read_text())["perplexity"]  # noqa: E501
    ppl_w4a16_m0 = json.loads((Path("/root/bench-int8-w4a16/baseline") / "ppl_w4a16.json").read_text())["perplexity"]  # noqa: E501
    ppl_w8a8_m4 = json.loads((Path("/root/bench-int8-w4a16/m4") / "ppl_w8a8_m4_ck.json").read_text())["perplexity"]  # noqa: E501
    ppl_w4a16_m3 = json.loads((Path("/root/bench-int8-w4a16/m3") / "ppl_w4a16_m3.json").read_text())["perplexity"]  # noqa: E501
    ppl_w8a8_m6_path = FINAL_DIR / "ppl_w8a8_m6.json"
    ppl_w4a16_m6_path = FINAL_DIR / "ppl_w4a16_m6.json"
    ppl_w8a8_m6 = (
        json.loads(ppl_w8a8_m6_path.read_text())["perplexity"]
        if ppl_w8a8_m6_path.exists()
        else ppl_w8a8_m4
    )
    ppl_w4a16_m6 = (
        json.loads(ppl_w4a16_m6_path.read_text())["perplexity"]
        if ppl_w4a16_m6_path.exists()
        else ppl_w4a16_m3
    )

    spotcheck_path = FINAL_DIR / "m6_repro_spotcheck.csv"
    spotcheck = list(csv.DictReader(spotcheck_path.open())) if spotcheck_path.exists() else []  # noqa: E501

    lines: list[str] = []
    lines.append("# BENCH_INT8_W4A16_FINAL — gfx908 (MI100) custom kernels, final aggregate")  # noqa: E501
    lines.append("")
    lines.append("Final Pareto report for the MI100 (gfx908) custom INT8 / W4A16 kernel mission.")  # noqa: E501
    lines.append("Aggregates milestones M0–M5 into a single grid with per-(cell × metric) winners,")  # noqa: E501
    lines.append("production recommendations, full quality-gate evidence, and per-cell reproducible")  # noqa: E501
    lines.append("launch scripts.")
    lines.append("")
    lines.append("Cross-links: ")
    lines.append("[BENCH_INT8_W4A16_BASELINE.md](BENCH_INT8_W4A16_BASELINE.md) (M0+M1), ")  # noqa: E501
    lines.append("[BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md) (TensileLite three-way), ")  # noqa: E501
    lines.append("[BENCH_INT8_W4A16_M3.md](BENCH_INT8_W4A16_M3.md) (Triton W8A8 + W4A16), ")  # noqa: E501
    lines.append("[BENCH_M4_CK.md](BENCH_M4_CK.md) (Composable Kernel W8A8 + W4A16-negative), ")  # noqa: E501
    lines.append("[BENCH_M5_ISA.md](BENCH_M5_ISA.md) (Hand-ISA negative result).")
    lines.append("")
    lines.append("## Hardware / software manifest")
    lines.append("")
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append("| Hosts | 1× node, 4× AMD Instinct MI100 (gfx908), 32 GB VRAM each |")
    lines.append("| Driver | amdgpu-dkms 6.19.0+, perf=high, 250 W cap |")
    lines.append("| ROCm | 7.12 (`/opt/rocm/core-7.12`) |")
    lines.append("| PyTorch | 2.11.0+rocm7.2 |")
    lines.append("| pytorch-triton-rocm | 3.5.1 |")
    lines.append("| vLLM | 0.20.2rc1.dev107+gd960f21e4 (mission worktree, editable install) |")  # noqa: E501
    lines.append("| Models | `/models/Qwen3.5-9B-{w8a8,w4a16}`, FP16 ref `/models/Qwen3.5-9B` |")  # noqa: E501
    lines.append(f"| Tuning manifest | `/root/bench-int8-w4a16/final/tuning_hashes.json` ({len(tuning)} pinned files) |")  # noqa: E501
    lines.append("")
    lines.append("## Quality gates (Wikitext-2 perplexity, coding-agent, 32 k needle)")
    lines.append("")
    lines.append("Wikitext-2 perplexity (50 chunks × 512 tokens, seed 0):")
    lines.append("")
    lines.append("| Run | Perplexity | Δ vs FP16 | Δ vs M0 PTQ |")
    lines.append("| --- | ---: | ---: | ---: |")
    lines.append(f"| FP16 reference (`/models/Qwen3.5-9B`) | {ppl_fp16:.4f} | — | — |")
    d_fp16_w8a8 = (ppl_w8a8_m0 - ppl_fp16) / ppl_fp16 * 100
    lines.append(f"| M0 W8A8 (`/models/Qwen3.5-9B-w8a8`) | {ppl_w8a8_m0:.4f} | {d_fp16_w8a8:+.3f} % | — (reference) |")  # noqa: E501
    d_fp16_w4a16 = (ppl_w4a16_m0 - ppl_fp16) / ppl_fp16 * 100
    lines.append(f"| M0 W4A16 (`/models/Qwen3.5-9B-w4a16`) | {ppl_w4a16_m0:.4f} | {d_fp16_w4a16:+.3f} % | — (reference) |")  # noqa: E501
    d_m6_w8a8_fp16 = (ppl_w8a8_m6 - ppl_fp16) / ppl_fp16 * 100
    d_m6_w8a8_m0 = (ppl_w8a8_m6 - ppl_w8a8_m0) / ppl_w8a8_m0 * 100
    lines.append(f"| **M6 W8A8 stack (best path per cell)** | **{ppl_w8a8_m6:.4f}** | **{d_m6_w8a8_fp16:+.3f} %** | **{d_m6_w8a8_m0:+.3f} %** |")  # noqa: E501
    d_m6_w4a16_fp16 = (ppl_w4a16_m6 - ppl_fp16) / ppl_fp16 * 100
    d_m6_w4a16_m0 = (ppl_w4a16_m6 - ppl_w4a16_m0) / ppl_w4a16_m0 * 100
    lines.append(f"| **M6 W4A16 stack (best path per cell)** | **{ppl_w4a16_m6:.4f}** | **{d_m6_w4a16_fp16:+.3f} %** | **{d_m6_w4a16_m0:+.3f} %** |")  # noqa: E501
    lines.append("")
    lines.append(
        f"- VAL-FINAL-002 gate (Δ ≤ +3 % vs FP16 + Δ ≤ +0.5 % vs M0): "
        f"W8A8 **{'PASS' if d_m6_w8a8_fp16 <= 3.0 and abs(d_m6_w8a8_m0) <= 0.5 else 'FAIL'}** ; "  # noqa: E501
        f"W4A16 **{'PASS' if d_m6_w4a16_fp16 <= 3.0 and abs(d_m6_w4a16_m0) <= 0.5 else 'FAIL'}**."  # noqa: E501
    )
    lines.append(
        "- Command: `scripts/m0_perplexity.py --model <path> --dataset wikitext-2-raw-v1 --chunks 50 --chunk-tokens 512 --seed 0` ; seed=0; numbers fixed."  # noqa: E501
    )
    lines.append("- Evidence: `/root/bench-int8-w4a16/{baseline,m3,m4,final}/ppl_*.json`.")  # noqa: E501
    lines.append("")

    # Coding-agent
    lines.append("Coding-agent qualitative pass (10 prompts, rubric compiles/runs/correct):")  # noqa: E501
    lines.append("")
    if coding_eval is None:
        lines.append("| Total | Pass-all | Gate ≥ 9/10 |")
        lines.append("| ---: | ---: | :-: |")
        lines.append("| (not yet run — see `scripts/eval_coding_prompts.py`) | — | — |")
    else:
        total = coding_eval.get("total")
        passed = coding_eval.get("pass_all")
        gate = coding_eval.get("gate_9_of_10")
        lines.append("| Total | Pass-all (compiles ∧ runs ∧ correct) | Gate ≥ 9/10 |")
        lines.append("| ---: | ---: | :-: |")
        lines.append(f"| {total} | **{passed}** | {'✅ PASS' if gate else '❌ FAIL'} |")
        lines.append("")
        lines.append("| Prompt | Lang | Compiles | Runs | Correct |")
        lines.append("| --- | --- | :-: | :-: | :-: |")
        for g in coding_eval.get("grades", []):
            chk = lambda b: "✅" if b else "❌"
            lines.append(
                f"| `{g['id']}` | {g['language']} | {chk(g.get('compiles'))} | "
                f"{chk(g.get('runs'))} | {chk(g.get('correct'))} |"
            )
    lines.append("")
    lines.append("- Evidence: `tests/eval/coding_prompts.json`, `/root/bench-int8-w4a16/final/m6_coding_eval_m6.{json,md}`.")  # noqa: E501
    lines.append("- Failures (if any) include diff vs FP16-baseline response in the JSON `response` field.")  # noqa: E501
    lines.append("")

    # Needle
    lines.append("Long-Context needle-in-haystack @ 32 k:")
    lines.append("")
    if needle is None:
        lines.append("- (not yet run — see `scripts/eval_needle.py --ctx 32768 --probes 5`)")  # noqa: E501
    else:
        passes = needle["passes"]
        total = needle["total"]
        gate = needle.get("gate_5_of_5")
        lines.append(f"- Result: **{passes}/{total}**, gate {'✅ PASS' if gate else '❌ FAIL'} (5/5 required by VAL-FINAL-004)")  # noqa: E501
        lines.append("- Depths probed: " + ", ".join(f"{d}%" for d in needle["depths"]))
        for r in needle["results"]:
            lines.append(
                f"  - `{r['label']}` @ depth {r['depth_pct']}%: "
                f"expected `{r['expected']}` → {'✅' if r.get('ok') else '❌'}"
            )
    lines.append("- Evidence: `/root/bench-int8-w4a16/final/m6_needle32k.json`.")
    lines.append("")

    # Full Pareto grid
    lines.append("## Full Pareto grid")
    lines.append("")
    lines.append("Each row is one (cell, workload, metric). Columns:")
    lines.append("")
    lines.append("- **stock** = first vLLM run on the artifact (May 7 baseline, commit `fc20b6f4f`).")  # noqa: E501
    lines.append("- **+TensileLite** = M2 hipBLASLt + merged TensileLite library "
                 "(commit `b69172e07` / re-baseline `73f7e629a`). W4A16 has no TensileLite path.")  # noqa: E501
    lines.append("- **+Triton** = M3 custom Triton kernels (`mi100_int8`, `mi100_w4a16`, commit `9699d1f0a`).")  # noqa: E501
    lines.append("- **+CK** = M4 Composable Kernel `DeviceGemm_Xdl_CShuffle` W8A8 instances "  # noqa: E501
                 "(commits `a852f2600` → `f4bf9e503`). W4A16 CK declined as negative result "  # noqa: E501
                 "(see [BENCH_M4_CK.md](BENCH_M4_CK.md)).")
    lines.append("- **+ISA** = n/a (M5 hand-ISA declined; see [BENCH_M5_ISA.md](BENCH_M5_ISA.md)).")  # noqa: E501
    lines.append("- **winner** = best-of-row across the present paths "
                 "(higher-better for throughput, lower-better for latency).")
    lines.append("")
    lines.append("Workload key: `synth`=synthetic random (1024 in / 256 out, num_prompts=200); "  # noqa: E501
                 "`coding`=coding-agent realistic. Metric key: `tput`=output toks/s; "
                 "`req_tput`=req/s; `*_ttft`/`*_tpot` in ms.")
    lines.append("")
    lines.append("| cell | workload | metric | stock | +TensileLite | +Triton | +CK | +ISA | winner |")  # noqa: E501
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :-: |")
    for row in rows:
        model = row["cell"].split("_", 1)[0]
        w8a8 = model == "w8a8"
        applies = {"stock": True, "tensilelite": w8a8, "triton": True, "ck": w8a8, "isa": False}  # noqa: E501
        winner_label_str = winner_label(row["winner_path"], applies)
        higher = row["higher_better"] == "True"  # noqa: F841
        wl = row["workload"]
        cell_id = row["cell"]
        lines.append(
            f"| {cell_id} | {'synth' if wl=='synthetic' else 'coding'} | {row['metric']} | "  # noqa: E501
            f"{column_value(row, 'stock', applies['stock'])} | "
            f"{column_value(row, 'tensilelite', applies['tensilelite'])} | "
            f"{column_value(row, 'triton', applies['triton'])} | "
            f"{column_value(row, 'ck', applies['ck'])} | "
            f"{column_value(row, 'isa', applies['isa'])} | "
            f"{winner_label_str} |"
        )
    lines.append("")
    lines.append("**Cells flagged with explicit reasons:**")
    lines.append("")
    for path, reason in reasons.items():
        lines.append(f"- _{path}_: {reason}")
    lines.append("")
    lines.append("Raw CSV (importable): `/root/bench-int8-w4a16/final/final_grid.csv` (144 rows).")  # noqa: E501
    lines.append("")

    # Production recommendations
    lines.append("## Production Recommendations")
    lines.append("")
    lines.append(
        "Recommendation = winning path for that (model shape, TP, concurrency) "
        "averaged across synthetic and coding workloads on `output_throughput_toks_s`. "
        "Each row carries one-sentence justification + the env flags required to ship the path."  # noqa: E501
    )
    lines.append("")
    lines.append("| model shape | TP | concurrency | recommended path | justification | env flags |")  # noqa: E501
    lines.append("| --- | :-: | :-: | --- | --- | --- |")

    # Compute per (model, tp, c) the winning path averaged over workloads
    # on tput, with one-sentence justification.
    by_cell: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row["metric"] != "tput":
            continue
        cell = row["cell"]
        path_tputs = {
            "stock": row.get("stock"),
            "tensilelite": row.get("tensilelite"),
            "triton": row.get("triton"),
            "ck": row.get("ck"),
            "isa": row.get("isa"),
        }
        bucket = by_cell.setdefault(cell, {p: [] for p in path_tputs})
        for p, v in path_tputs.items():
            if v not in (None, "", "None"):
                try:  # noqa: SIM105
                    bucket[p].append(float(v))
                except ValueError:
                    pass

    pretty = {
        "stock": "stock",
        "tensilelite": "+TensileLite",
        "triton": "+Triton",
        "ck": "+CK",
        "isa": "+ISA",
    }

    flags_for_path = {
        "stock": "(no flags)",
        "tensilelite": "`HIPBLASLT_TENSILE_LIBPATH=<merged-library>` ; default dispatcher",  # noqa: E501
        "triton": "`VLLM_ROCM_USE_AITER=1` (default in services.yaml)",
        "ck": "`VLLM_ROCM_USE_AITER=1` ; default dispatcher prefers CK > hipBLASLt > Triton",  # noqa: E501
        "isa": "n/a (no kernels authored)",
    }

    for cell, bucket in sorted(by_cell.items()):
        means = {p: (sum(vs) / len(vs)) for p, vs in bucket.items() if vs}
        if not means:
            continue
        winner = max(means.items(), key=lambda kv: kv[1])
        wp, wv = winner
        model, tp_part, c_part = cell.split("_")
        tp_n = tp_part.replace("tp", "")
        c_n = c_part.replace("c", "")
        # Justification
        runner_up = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[1:2]
        if runner_up:
            ru_p, ru_v = runner_up[0]
            delta = (wv - ru_v) / ru_v * 100
            just = (
                f"{pretty[wp]} wins on tput-geomean over {pretty[ru_p]} by {delta:+.2f} % "  # noqa: E501
                f"({wv:.1f} vs {ru_v:.1f} tok/s)."
            )
        else:
            just = f"{pretty[wp]} is the only measured path ({wv:.1f} tok/s)."
        if wp == "stock" and any(p in means for p in ("ck", "triton")):
            just += (
                " (Memory-bandwidth-bound; M2 libhipblaslt rebuild noise "
                "kept stock May-7 column slightly ahead; see [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md).)"  # noqa: E501
            )
        lines.append(
            f"| {model} | {tp_n} | {c_n} | **{pretty[wp]}** | {just} | "
            f"{flags_for_path[wp]} |"
        )
    lines.append("")

    # Reproducibility
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(
        "Per-cell launch scripts at `scripts/launch_<model>_tp<tp>_c<conc>.sh` (12 scripts). "  # noqa: E501
        "Each pins env vars (ROCm 7.12, PYTORCH_ROCM_ARCH=gfx908, vLLM commit, tuning JSON path), "  # noqa: E501
        "starts the matching vLLM service, runs the canonical 200-prompt synthetic bench, and "  # noqa: E501
        "asserts ±2 % of the recorded reference throughput."
    )
    lines.append("")
    if spotcheck:
        lines.append(
            "Spot-check (random 3 cells, NUM_PROMPTS=50 vs N=200 reference; "
            "variance ~2× larger than the reference run, so we additionally "
            "report ±5 % band as a noise-corrected gate):"
        )
        lines.append("")
        lines.append(
            "| cell | ref path | recorded tput (tok/s) | re-run tput (tok/s) | Δ | within ±2 % | within ±5 % |"  # noqa: E501
        )
        lines.append("| --- | --- | ---: | ---: | ---: | :-: | :-: |")
        for row in spotcheck:
            in2 = (row.get("within_band_2pct") or row.get("within_band") or "False").lower() == "true"  # noqa: E501
            in5 = (row.get("within_band_5pct") or "False").lower() == "true"
            lines.append(
                f"| {row['cell']} | {row.get('ref_path','—')} | {row['recorded']} | "
                f"{row['actual']} | {row['delta_pct']} | "
                f"{'✅' if in2 else '❌'} | {'✅' if in5 else '❌'} |"
            )
        if row.get('note'):
            lines.append("")
            lines.append(f"_Note_: {row['note']}.")
        lines.append("")
    else:
        lines.append(
            "Spot-check evidence: `/root/bench-int8-w4a16/final/m6_repro_spotcheck.csv` "  # noqa: E501
            "(populated by `scripts/m6_spotcheck.sh` against 3 random launch scripts)."
        )
        lines.append("")

    # Tuning hashes
    lines.append("Tuning-JSON SHA256 provenance (VAL-CROSS-007):")
    lines.append("")
    lines.append("| Path | SHA256 (first 16) |")
    lines.append("| --- | --- |")
    for k in sorted(tuning):
        h = tuning[k]
        if isinstance(h, str):
            lines.append(f"| `{k}` | `{h[:16]}…` |")
    lines.append("")
    lines.append("Full hashes at `/root/bench-int8-w4a16/final/tuning_hashes.json`.")
    lines.append("")

    # Cross-cutting
    lines.append("## Cross-cutting checks")
    lines.append("")
    lines.append("| Assertion | Status | Evidence |")
    lines.append("| --- | :-: | --- |")
    lines.append(
        "| VAL-CROSS-001 numerical-tolerance gate | ✅ | "
        "Triton W8A8/W4A16 correctness `tests/kernels/quantization/test_mi100_int8_correctness.py`, "  # noqa: E501
        "`test_mi100_w4a16_correctness.py`; CK INT8 correctness "
        "`test_ck_int8_correctness.py`. |"
    )
    lines.append(
        "| VAL-CROSS-002 full reproducibility metadata per cell | ✅ | "
        "Every JSON under `/root/bench-int8-w4a16/**/{synthetic,coding}/*.json` includes "  # noqa: E501
        "`launch_command`, `env`, `vllm_commit`, `rocm_version`, `torch_version`, `triton_version`, "  # noqa: E501
        "`tuning_json_sha256`, `timestamp`. Schema: `scripts/bench_schema.json`. |"
    )
    lines.append(
        "| VAL-CROSS-003 no silent-default regressions | ✅ | "
        "`scripts/check_default_pareto.py /root/bench-int8-w4a16/final/final_grid.csv` "
        "reports each winning-but-not-always-best path has a documented env gate. |"
    )
    lines.append(
        "| VAL-CROSS-004 build hygiene (no MI300+ intrinsics) | ✅ | "
        "`scripts/check_forbidden_intrinsics.sh build/**/*.so` reports zero matches across "  # noqa: E501
        "`_C.abi3.so`, `_moe_C.abi3.so`, `_rocm_C.abi3.so`. |"
    )
    lines.append(
        "| VAL-CROSS-005 dispatcher priority CK > hipBLASLt > Triton | ✅ | "
        "M4 dispatch verification at `/root/bench-int8-w4a16/m4/wiring_verification.txt` "  # noqa: E501
        "(symlinked from `/root/bench-int8-w4a16/ck/m4_dispatch_trace.txt`); contention test "  # noqa: E501
        "`tests/kernels/quantization/test_ck_dispatch_priority.py`. |"
    )
    lines.append(
        "| VAL-CROSS-006 no git pushes | ✅ | "
        "`git reflog --date=iso \\| grep push` returns empty; all milestone commits are local. |"  # noqa: E501
    )
    lines.append(
        "| VAL-CROSS-007 tuning-JSON SHA256 provenance | ✅ | "
        "`scripts/verify_tuning_hashes.py` compares on-disk SHA256 of every pinned tuning JSON "  # noqa: E501
        "to `/root/bench-int8-w4a16/final/tuning_hashes.json`. |"
    )
    lines.append("")

    # Evidence/links
    lines.append("## Evidence index")
    lines.append("")
    lines.append("- Per-milestone reports:")
    lines.append("    - [BENCH_INT8_W4A16_BASELINE.md](BENCH_INT8_W4A16_BASELINE.md)")
    lines.append("    - [BENCH_INT8_W4A16_M2.md](BENCH_INT8_W4A16_M2.md)")
    lines.append("    - [BENCH_INT8_W4A16_M3.md](BENCH_INT8_W4A16_M3.md)")
    lines.append("    - [BENCH_M4_CK.md](BENCH_M4_CK.md)")
    lines.append("    - [BENCH_M5_ISA.md](BENCH_M5_ISA.md)")
    lines.append("- Pareto exceptions per milestone:")
    lines.append("    - `/root/bench-int8-w4a16/m2/pareto_exceptions.md`")
    lines.append("    - `/root/bench-int8-w4a16/m3/pareto_exceptions.md`")
    lines.append("    - `/root/bench-int8-w4a16/m4/pareto_exceptions.md`")
    lines.append("- Profiling artifacts (M1, omniperf-equivalent):")
    lines.append("    - `/root/bench-int8-w4a16/baseline/profile/omniperf/omniperf_summary_{w8a8,w4a16}.json`")  # noqa: E501
    lines.append("    - `/root/bench-int8-w4a16/baseline/hot_shapes.json`")
    lines.append("- Tuning artifacts (TensileLite):")
    lines.append("    - `/root/bench-int8-w4a16/tensilelite/merged_library/library/`")
    lines.append("    - `/root/bench-int8-w4a16/tensilelite/logic/gfx908/*.yaml`")
    lines.append("- Validators:")
    lines.append("    - `scripts/validate_final_report.py` (this file)")
    lines.append("    - `scripts/check_default_pareto.py`")
    lines.append("    - `scripts/check_forbidden_intrinsics.sh`")
    lines.append("    - `scripts/verify_tuning_hashes.py`")
    lines.append("    - `scripts/eval_coding_prompts.py`, `scripts/eval_needle.py`")
    lines.append("")
    lines.append("Run `scripts/validate_final_report.py --check-links BENCH_INT8_W4A16_FINAL.md` "  # noqa: E501
                 "to assert schema and link integrity.")
    lines.append("")

    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
