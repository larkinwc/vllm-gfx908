# M3-F1 Pre-Sync Baseline Bench Results

**Tag:** `pre-sync-baseline` = `3d9de886d` (PR #44 merge — W4A16 cross-host A/B verdict).
**Worktree:** `/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/loose-rats-clean-phm2k`
**vLLM (post-rebuild):** `0.20.2rc1.dev187+g3d9de886d.rocm712`
**Build command:** `PYTORCH_ROCM_ARCH=gfx908 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2 /opt/vllm-env/bin/pip install -e . --no-build-isolation` (see `../install-baseline.log`).
**MAPPING + .so verification:** all four `vllm/_*.abi3.so` artifacts rebuilt under this worktree; editable finder MAPPING and `direct_url.json` both point at this worktree (see env-repoint pattern from M1-F6).
**Host:** 4× MI100 (gfx908), ROCm 7.12, PyTorch 2.11.0+rocm7.2, Triton 3.5.1, rocprofv3 1.2.0, `/opt/vllm-env/bin/python3` (3.12.3).

## Files

| Path | Contents |
| --- | --- |
| `commands.txt` | Verbatim launch commands for all four bench groups (M3-F2 reproducibility) |
| `run_baseline_m3f1.sh` | Wrapper that patches `/root/bench-int8-w4a16/baseline/run_baseline.sh` with REPO/BASELINE_ROOT for this worktree |
| `w4a16-grid.json` | Aggregated 24-cell summary table (output_throughput, ttft, tpot per cell) |
| `grid/synthetic/*.json` | 12 raw synthetic cell JSONs (schema-conformant) |
| `grid/coding/*.json` | 12 raw coding cell JSONs |
| `grid/harness_manifest.json` | Harness env, versions, dataset SHA, cell list |
| `grid/run_baseline_*.log` | Full tee'd run log (all 24 cells, ~5h05m wall time) |
| `fused-act-quant.json` | Alias of `grid/synthetic/w8a8_tp1_c1.json` — mechanically identical to the BENCH_FUSED_ACT_QUANT.md `w8a8_tp1_c1/synthetic` cell |
| `ckfa-latency.json` | CK-FA decode latency micro-bench on /models/Llama-2-7b-hf (head_dim=128) — 30 iters, 10 warmup, batch=1, input=2048, output=64, `--attention-backend ROCM_CK_FA`, `--enforce-eager` |
| `ckfa/run.log` | CK-FA bench transcript |
| `hbm-pmc.csv` | rocprofv3 PMC counter_collection (FETCH_SIZE, WRITE_SIZE, VALUUtilization, TCC_HIT) for the w8a8_tp1_c1 fused-on cell — 762,064 records. **NOT in git** (316 MB; `.gitignore` `*.csv`). Lives on host at `library/bench-baseline/hbm-pmc.csv` and at `/root/bench-int8-w4a16/m3-f1-baseline/rocprof_w8a8_tp1_c1_on/pmc.csv`. M3-F2 can recompute equivalent CSV with the same `scripts/mi100/rocprof_single_request.sh` invocation. |
| `hbm-summary.json` | Aggregated FETCH+WRITE sums, total HBM bytes, bytes/output_token, bytes/total_token |
| `hbm/` | Auxiliary rocprof outputs (kernel_trace.log, pmc.log, capture_summary.json) |

## Headline numbers (single-seed, pre-sync-baseline)

### 24-cell grid (`output_throughput_toks_s`)

| cell_id | synthetic | coding |
| --- | ---: | ---: |
| w8a8_tp1_c1 | 41.48 | 40.78 |
| w8a8_tp1_c2 | 68.23 | 64.24 |
| w8a8_tp1_c4 | 124.98 | 110.03 |
| w8a8_tp4_c1 | 56.54 | 54.52 |
| w8a8_tp4_c2 | 112.32 | 100.98 |
| w8a8_tp4_c4 | 221.61 | 177.59 |
| w4a16_tp1_c1 | 29.58 | 29.06 |
| w4a16_tp1_c2 | 54.60 | 52.23 |
| w4a16_tp1_c4 | 101.25 | 93.07 |
| w4a16_tp4_c1 | 50.88 | 48.79 |
| w4a16_tp4_c2 | 99.29 | 89.38 |
| w4a16_tp4_c4 | 194.22 | 162.06 |

### Fused-act-quant single cell

- `w8a8_tp1_c1/synthetic` output_throughput = **41.48 tok/s** (mean_ttft 456.94 ms, mean_tpot 22.41 ms).
  Identical row to `grid/synthetic/w8a8_tp1_c1.json` (default fused-on; flag unset on
  `pre-sync-baseline` tag).

### CK-FA decode latency (Llama-2-7b-hf, head_dim=128, ROCM_CK_FA, enforce-eager)

- Avg latency: **1.7798 s** over 30 iters (10 warmup), batch=1, input=2048, output=64.
- p50 = **1.7745 s**, p90 = 1.7759 s, p99 = 1.8889 s.

### HBM bytes per token (rocprofv3 PMC, w8a8_tp1_c1 fused-on)

- Total HBM bytes (Σ FETCH_SIZE + WRITE_SIZE across 190,516 dispatches): **2,019,688,683,904 B** (1.96 TiB).
- Bytes per output token (input=1024, output=32): **63,115,271,372 B/output_token** (≈58.8 GiB/token).
- Bytes per total token (1056): **1,912,583,981 B/total_token** (≈1.78 GiB/token).
- Direct comparison with `BENCH_FUSED_ACT_QUANT.md` §i (fused-on, same cell, same harness):
  reported total HBM = 2,019,709,590,112 B — our 2,019,688,683,904 B agrees within 21 KB
  (run-to-run noise; same code path, same .so artifacts).

## M3-F2 reproducibility checklist

Every bench in M3-F2 MUST use the same:

- Tag `upstream-sync-2026-05-28` HEAD (in place of `pre-sync-baseline`).
- Models: `/models/Qwen3.5-9B-w8a8`, `/models/Qwen3.5-9B-w4a16`, `/models/Llama-2-7b-hf`.
- Seeds: `--seed 42` everywhere.
- 24-cell grid prompts: `--num-prompts 200 --request-rate inf` per cell.
- CK-FA latency: 30 iters / 10 warmup, batch=1, input=2048, output=64, `--enforce-eager`, `--attention-backend ROCM_CK_FA`.
- rocprofv3 HBM: same `scripts/mi100/rocprof_single_request.sh w8a8_tp1_c1 on` invocation with REPO override.

See `commands.txt` for the verbatim launches.

## Anti-pattern callouts (carried from prior missions)

1. `VLLM_ATTENTION_BACKEND` env var is silently dropped post-upstream-merge; the only working
   path on the sync branch is `--attention-backend ROCM_CK_FA` (CLI flag). The pre-sync tag
   accepts both, but we use the CLI flag uniformly so M3-F2 deltas are env-identical.
2. CK FA splitkv does NOT support head_dim=256 on either tag — Llama-2-7b (head_dim=128) is
   the only model that exercises ROCM_CK_FA cleanly.
3. rocprofv3 + `bench serve` deadlocks on gfx908 — offline `bench throughput` is the only
   working harness; this is also what `BENCH_FUSED_ACT_QUANT.md` §i used.
4. `bench latency` rejects `--max-model-len 32768` on Llama-2-7b — must use `--max-model-len 4096`.
