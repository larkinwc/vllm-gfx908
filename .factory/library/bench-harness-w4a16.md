# Bench / rocprof Harness — W4A16 mission

> Reusable bench + rocprof harness for the MI100/gfx908 **W4A16 Marlin-repack**
> mission. Written by `F-M0-script-adapt`. Consumed by `F-M0-baseline-lock`,
> `F-M3-bench-grid`, `F-M3-rocprof`.
>
> The prior W8A8/INT8 mission tooling was hard-coded to a stale worktree path
> and the W8A8 model. This adaptation makes the in-tree scripts target W4A16
> and resolve the repo dynamically. **Do not reinvent — reuse these.**

## What changed vs the prior (W8A8) harness

| Concern            | Prior (W8A8)                                   | Now (W4A16)                                        |
|--------------------|------------------------------------------------|----------------------------------------------------|
| REPO               | hard-coded `.../thin-hands-smell-2bxf5` etc.   | `git rev-parse --show-toplevel` (current worktree) |
| MODEL              | `/models/Qwen3.5-9B-w8a8`                       | `/models/Qwen3.5-9B-w4a16`                          |
| GEMM toggle        | `VLLM_MI100_DISABLE_FUSED_ACT_QUANT`           | `VLLM_MI100_W4A16_USE_MARLIN_REPACK` (off/0 = baseline, 1 = marlin) |
| KV/chunked env     | KV-INT8 + chunked prefill + merged TensileLite | none (mirrors `services.yaml` vllm-w4a16-* server) |
| HBM%               | FETCH_SIZE+WRITE_SIZE (kept)                    | FETCH_SIZE+WRITE_SIZE (unchanged; NEVER TCP_TCC_*) |

## Files

- `scripts/mi100/run_w4a16_baseline.sh` — **NEW** self-contained 12-cell W4A16
  grid harness. Resolves REPO via git, drives `TP{1,4} × c{1,2,4} ×
  {synthetic,coding}`, honors `VLLM_MI100_W4A16_USE_MARLIN_REPACK`, mirrors the
  `services.yaml` server launch, writes per-cell JSON via
  `scripts/postprocess_bench_result.py`.
- `scripts/mi100/run_grid.sh` — dispatches the W4A16 milestones
  (`m0-w4a16`, `m3-w4a16-baseline`, `m3-w4a16-marlin`) to the new harness,
  bypassing the prior-mission `/root` sed-copy machinery. W8A8 milestones are
  untouched.
- `scripts/mi100/rocprof_single_request.sh` — adapted to W4A16: git REPO,
  W4A16 model, arg2 = marlin state (`on|off`), env mirrors the W4A16 server.
- `scripts/mi100/aggregate_hbm.py` — unchanged; already derives per-token HBM
  bytes + HBM% from `FETCH_SIZE+WRITE_SIZE` only.
- `scripts/mi100/pmc_counters.txt` — `pmc: FETCH_SIZE WRITE_SIZE` (+ VALU/TCC_HIT
  diagnostics). No TCP_TCC_*.

## Cell definition (12 cells)

- `TP ∈ {1,4}` × `c ∈ {1,2,4}` × `{synthetic, coding}`.
- Cell id: `w4a16_tp{TP}_c{c}_{workload}`.
- synthetic = `--dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos`.
- coding = `--dataset-name custom --dataset-path <coding_agent.jsonl> --custom-output-len 256 --skip-chat-template`.
- `--num-prompts 200`, `--seed 42`, `--request-rate inf`, `--max-concurrency c`.

## Invocation

### Bench grid (baseline = marlin off)

```bash
cd <worktree>            # full-mirrors-dig-3ioww (resolved via git automatically)
# baseline lock:
VLLM_MI100_W4A16_USE_MARLIN_REPACK=0 scripts/mi100/run_grid.sh m0-w4a16
# full 12 cells -> /root/bench-w4a16/m0/{synthetic,coding}/<cell>.json
```

### Bench grid (marlin on, M3)

```bash
VLLM_MI100_W4A16_USE_MARLIN_REPACK=1 scripts/mi100/run_grid.sh m3-w4a16-marlin
VLLM_MI100_W4A16_USE_MARLIN_REPACK=0 scripts/mi100/run_grid.sh m3-w4a16-baseline
```

### Subset / dry-run

`run_grid.sh <milestone> <CELLS_FILTER-regex>` — the 2nd arg is a regex on the
cell id. Or drive the harness directly with env knobs:

```bash
# 1-cell dry-run, 4 prompts, custom output root:
CELLS_FILTER='w4a16_tp1_c1_synthetic' NUM_PROMPTS=4 \
  W4A16_BENCH_ROOT=/root/bench-w4a16/dryrun \
  VLLM_MI100_W4A16_USE_MARLIN_REPACK=0 \
  scripts/mi100/run_w4a16_baseline.sh 'w4a16_tp1_c1_synthetic'
```

Harness env knobs: `W4A16_BENCH_ROOT` (output root), `NUM_PROMPTS` (default 200),
`CELLS_FILTER` / arg `$1` (cell regex), `MODEL`, `PY`, `DATASET`,
`VLLM_MI100_W4A16_USE_MARLIN_REPACK`.

### rocprof single-request HBM capture

```bash
# arg1=cell_id  arg2=marlin_state(on|off)  arg3=out_dir
scripts/mi100/rocprof_single_request.sh w4a16_tp1_c1_coding off /root/bench-w4a16/rocprof/baseline
scripts/mi100/rocprof_single_request.sh w4a16_tp1_c1_coding on  /root/bench-w4a16/rocprof/marlin
# then derive HBM% (weight-bytes is the theoretical per-token W4A16 weight traffic):
scripts/mi100/aggregate_hbm.py --capture-dir /root/bench-w4a16/rocprof/marlin --weight-bytes <N>
```

`out_dir` gets `kernel_trace.csv`, `pmc.csv`, `capture_summary.json`.
`aggregate_hbm.py` writes `hbm_summary.json` (per-token bytes + HBM%).

### services.yaml equivalents

`services.yaml` exposes the same paths as named commands:
`bench-grid-synthetic` (calls `run_grid.sh m0-w4a16`), `rocprof-hbm`
(calls `rocprof_single_request.sh "$CELL_ID" "$MARLIN_STATE" "$OUT_DIR"`).
Servers: `vllm-w4a16-tp1`, `vllm-w4a16-tp4` (only ONE at a time, port 8000).

## Output layout

```text
$W4A16_BENCH_ROOT/
  harness_manifest.json          # env, versions, marlin flag, dataset sha, 12 cells
  synthetic/w4a16_tp{1,4}_c{1,2,4}.json
  coding/w4a16_tp{1,4}_c{1,2,4}.json
  env_<cell>_<workload>.json
  server_<svc>_<ts>.log / run_w4a16_<ts>.log
```

Validate with: `python3 scripts/validate_results_schema.py $W4A16_BENCH_ROOT/`.

## Critical rules (carried from mission AGENTS.md / bench-worker skill)

- Comparison reference is the **M0 W4A16 lock**, not FP16.
- rocprof HBM% from **FETCH_SIZE+WRITE_SIZE only** — NEVER `TCP_TCC_*` on
  gfx908 + rocprofv3 1.2.0.
- Save per-cell **raw** `vllm bench serve` JSON (no summary-only).
- Only ONE vLLM server at a time; tear down between cells / before switching TP
  (the harness does this via `kill_orphans`).

## Verification performed by F-M0-script-adapt

- `bash -n` on all three scripts: PASS.
- `run_grid.sh m0-w4a16` dispatches to the W4A16 harness; REPO resolved to the
  current worktree via git; MODEL=`/models/Qwen3.5-9B-w4a16`; marlin flag
  echoed through (=1).
- Harness exits cleanly (code 2) at the model-missing guard; rocprof marlin gate
  rejects bad state (code 2), accepts `on`/`off`, and reaches the rocprofv3
  kernel-trace pass.

## KNOWN BLOCKER (this sandbox)

The F-M0-script-adapt **editing** sandbox does **not** have the GPU host
resources mounted: `/models/Qwen3.5-9B-w4a16` is absent, `/opt/vllm-env` and
`rocprofv3`/`rocm-smi` are not installed, and `/root/bench-int8-w4a16` (prior
mission tree referenced by the legacy harness) is not present. A **live**
1-cell dry-run that actually launches the W4A16 server and writes per-cell JSON
could therefore NOT be executed here — it must be run on the MI100/gfx908 host
where the model + ROCm stack exist. The harness was validated up to the
model-presence guard; on the GPU host the documented 1-cell dry-run command
above will launch the server and emit `synthetic/w4a16_tp1_c1.json`.
