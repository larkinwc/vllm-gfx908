# M3 bench grid — Marlin W4A16 vs legacy (REAL, matched A/B, verified)

> **NEGATIVE RESULT.** The Marlin-style W4A16 repack kernel **regresses**
> decode throughput by **~26–30%** on gfx908. Captured by the orchestrator on
> MI100 (gfx908), 2026-06-01, as a single **matched A/B**: legacy and marlin
> servers launched back-to-back under *identical* config, each flag verified in
> `/proc/<pid>/environ` before benching. Every number below was read directly
> from the raw `vllm bench serve` JSON on disk; all cells `completed=200,
> failed=0`.
>
> **Provenance note (important).** Earlier revisions of this file (and the
> mission summary) reported a small *positive* result (geomean +0.24%, six
> "all-positive" cells, a run2 repro). Those numbers were **never measured** —
> the corresponding `raw.json` files on disk were either absent or showed
> `completed=0, failed=200` (every legacy A/B cell had failed), and the marlin
> cells were at a different/unmatched config. They have been **deleted and
> replaced** with this verified matched A/B. Do not reintroduce the old values.

## Config (identical for both arms)

Port 8000, TP1, GPU0. Server: `/root/launch_w4a16_legacy_tp1.sh` (flag=0) and
`/root/launch_w4a16_marlin_tp1.sh` (flag=1), both:
`--dtype float16 --tensor-parallel-size 1 --max-model-len 4096 --block-size 32
--language-model-only --gpu-memory-utilization 0.85`. Pinned env:
`VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_SKINNY_GEMM=0 TORCH_COMPILE_DISABLE=1
PYTORCH_ROCM_ARCH=gfx908`. Both servers logged
`Using TritonW4A16LinearKernel`; marlin arm's hot kernel verified as
`_mi100_w4a16_marlin_gemm_kernel` in the rocprof CSV (legacy =
`mi100_w4a16_gemm_kernel`).

Bench: `vllm bench serve`, dataset random 1024/256, num_prompts=200, seed 42,
`--request-rate inf --ignore-eos --max-concurrency c`. Metric =
`output_throughput` (tok/s).

## Matched A/B grid — REAL (all completed=200, failed=0)

| cell | legacy (tok/s) | marlin (tok/s) | Δ% | legacy TPOT (ms) | marlin TPOT (ms) |
|------|---------------:|---------------:|------:|-----------------:|-----------------:|
| tp1_c1 | 29.7957 | 21.3734 | **−28.27%** | 32.42 | 43.96 |
| tp1_c2 | 54.8752 | 40.6238 | **−25.97%** | 34.70 | 44.82 |
| tp1_c4 | 101.4517 | 71.2048 | **−29.81%** | 36.72 | 49.26 |

Raw JSON:
- legacy: `/root/bench-w4a16/m3v/legacy_tp1/tp1_c{1,2,4}.json`
- marlin: `/root/bench-w4a16/m3v/marlin_tp1/tp1_c{1,2,4}.json`

### Geomean (A1 — VAL-PERF-001)

- legacy TP1 geomean(c1,c2,c4) = **54.9452 tok/s**
- marlin TP1 geomean(c1,c2,c4) = **39.5417 tok/s**
- **Δ = −28.03%**
- decode-only geomean(c1,c2): legacy 40.4357 vs marlin 29.4664 = **−27.13%**

**A1 (target ≥+5%): MISS → documented NEGATIVE RESULT.** The repack not only
fails to reach the +5% floor, it regresses ~28%. Mechanism is recorded in
`rocprof-marlin.md`: the marlin GEMM fetches *more* HBM per dispatch
(25,046 vs 24,488 KB/dispatch, +2.3%) and emits additional helper kernels
(14 distinct kernels vs 4 for legacy), while per-token decode latency (TPOT)
rises ~11–13 ms/token. On gfx908 (CDNA1) there is no `cp.async`, so
`num_stages ≤ 2` and the repacked layout cannot be overlapped into LDS/MFMA the
way Marlin needs on Ada/Hopper; the extra repack/dequant path is pure overhead
at M=1 decode.

### A2 / A-floor (VAL-PERF-002 / VAL-PERF-006): FAIL

All 3 measured cells regress **far beyond** the −5% hard floor (−26% to −30%).
The non-regression gates are **not** satisfied. This is the central finding:
the kernel must ship **default-OFF** (it already is), and is documented as a
negative result per the contract's negative-result clause.

## Not measured (honestly descoped)

- **TP4 grid** — not captured this pass. The TP1 matched A/B already
  establishes a clear, reproducible ~28% regression; TP4 would only add
  cross-GPU/all-reduce noise on top of an already-failed gate.
- **Coding cells** — not run.
- **±2% two-run repro (VAL-PERF-005)** — only one matched run per arm. The
  per-cell deltas (−26% to −30%) are an order of magnitude larger than
  run-to-run noise, so the regression direction is unambiguous; a repro run was
  not needed to establish the verdict.
