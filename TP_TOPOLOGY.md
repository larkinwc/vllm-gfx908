# TP Topology — Production Decision Table

Production decision table for the **MI100 (gfx908) Tensor-Parallel = 4
topology / NCCL algorithm selection** on the
[`mi100-fixes`](https://github.com/larkinwc/vllm-gfx908/tree/mi100-fixes)
branch. This file is the canonical reference for downstream consumers
(production rollout, follow-up missions); the per-cell measurement evidence
that drives the table lives in
[`BENCH_HBM_M3_TP.md`](BENCH_HBM_M3_TP.md) and
[`/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json`](/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json).

## Hardware / software manifest

- 4 × AMD MI100 (`gfx908`), XGMI full mesh; ROCm 7.12; rccl 2.27.7
- vLLM commit
  [`c3768adcb`](https://github.com/larkinwc/vllm-gfx908/commit/c3768adcb)
  (or any later commit on `mi100-fixes`)
- Cumulative HBM-mission stack: M1 KV-INT8 + M2 chunked-prefill + M3 NCCL_ALGO

## Constraint: `NCCL_ALGO=Tree` is NOT selectable on the M3 stack

rccl 2.27.7 does not support `NCCL_ALGO=Tree` for `AllGather` on the
`ncclInt8` datatype. KV-INT8 (`--kv-cache-dtype int8_per_token_head`) issues
an AllGather on int8 tensors during profile_run, and rccl aborts engine
init with `ncclInvalidUsage`. As a result, the per-cell winner search
reduces to `{Ring, default}`, which the M3 NCCL sweep measured to be within
~1 % of each other on every cell. See the "Tree-algo failure root cause"
block in
[`/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json`](/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json)
for the rccl error excerpt.

## Production decision table

| Cell ID | Model | Recommended NCCL_ALGO | Rationale + env-flag disable path |
|:--------|:------|:----------------------|:----------------------------------|
| `w8a8_tp4_c1` | Qwen3.5-9B-w8a8 | `default` (rccl heuristic) | M3 sweep: default 63.91 tok/s vs Ring 63.71 tok/s on synthetic — within 0.3 %, so defer to rccl's per-message-size heuristic. Disable path: leave `NCCL_ALGO=` empty (already the launch-script default) — `unset NCCL_ALGO` is byte-identical. |
| `w8a8_tp4_c2` | Qwen3.5-9B-w8a8 | `default` (rccl heuristic) | M3 sweep: default 122.59 tok/s vs Ring 122.36 tok/s on synthetic — within 0.2 %, so defer to rccl's per-message-size heuristic. Disable path: `unset NCCL_ALGO` (reverts to same behavior as the baked-in empty value). |
| `w8a8_tp4_c4` | Qwen3.5-9B-w8a8 | `Ring` | M3 sweep: Ring 230.43 tok/s vs default 230.23 tok/s on synthetic — within 0.1 %, but Ring marginally favored. Disable path: `unset NCCL_ALGO` reverts to rccl heuristic; expected throughput delta < 0.1 %. |
| `w4a16_tp4_c1` | Qwen3.5-9B-w4a16 | `Ring` | M3 sweep: Ring 55.35 tok/s vs default 55.21 tok/s on synthetic — within 0.3 %, but Ring marginally favored. Disable path: `unset NCCL_ALGO` reverts to rccl heuristic; expected throughput delta < 0.3 %. |
| `w4a16_tp4_c2` | Qwen3.5-9B-w4a16 | `default` (rccl heuristic) | M3 sweep: default 105.95 tok/s vs Ring 105.43 tok/s on synthetic — within 0.5 %, so defer to rccl's per-message-size heuristic. Disable path: leave `NCCL_ALGO=` empty (already the launch-script default) — `unset NCCL_ALGO` is byte-identical. |
| `w4a16_tp4_c4` | Qwen3.5-9B-w4a16 | `Ring` | M3 sweep: Ring 199.32 tok/s vs default 199.25 tok/s on synthetic — within 0.04 %, but Ring marginally favored. Disable path: `unset NCCL_ALGO` reverts to rccl heuristic; expected throughput delta < 0.1 %. |

## Operational notes

- **Per-cell winners are baked into the launch scripts.** The 6 TP=4 launch
  scripts at `scripts/launch_<model>_tp4_c<conc>.sh` carry a single
  `export NCCL_ALGO=<winner>` line in their "Pinned environment" block.
  Diff vs the pre-M3 launch scripts: 1 added line per script (6 total),
  plus a top-of-file documentation block describing the new env var.
- **Disable path for the whole M3 contribution:** `unset NCCL_ALGO` before
  invoking the launch script reverts to rccl's per-message-size heuristic.
  Because the per-cell sweep deltas were < 1 % on every cell, the production
  impact of disabling M3 is < 1 % on throughput. The M3 contribution is
  best framed as a no-regression alignment with the sweep result, not a
  performance lever in its own right; the headline +9 % to +19 % wins on
  the M3 grid come from M1+M2.
- **Tree-algo blocked.** `NCCL_ALGO=Tree` is rejected by rccl 2.27.7 under
  the M3 KV-INT8 stack. Production launches that try `NCCL_ALGO=Tree` will
  fail engine init with `ncclInvalidUsage`. Documented in
  [`BENCH_HBM_M3_TP.md`](BENCH_HBM_M3_TP.md) §NCCL sweep summary.
- **TP=1 cells:** `NCCL_ALGO` is a no-op at single-rank. No edits to
  TP=1 launch scripts were made (`scripts/launch_<model>_tp1_c<conc>.sh`).
- **Shard axis:** vLLM hard-codes the shard axis per `ParallelLinear`
  class; `ColumnParallelLinear` shards N, `RowParallelLinear` shards K.
  No runtime override exists. Full evidence in
  [`/root/bench-int8-w4a16-hbm/m3-tp/shard_axis_findings.md`](/root/bench-int8-w4a16-hbm/m3-tp/shard_axis_findings.md).

## Cross-references

- Per-cell measurement evidence + win-bar analysis:
  [`BENCH_HBM_M3_TP.md`](BENCH_HBM_M3_TP.md)
- NCCL algorithm sweep raw (18 entries: 6 cells × 3 algos, including the
  Tree-failure root cause):
  [`/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json`](/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json)
- NCCL sweep markdown summary:
  [`/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep_summary.md`](/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep_summary.md)
- Production baseline:
  [`BENCH_INT8_W4A16_FINAL.md`](BENCH_INT8_W4A16_FINAL.md),
  `/root/bench-int8-w4a16/final/final_grid.csv`
- HBM-mission M1 & M2 reports (prerequisite stack):
  [`BENCH_HBM_M1_KVINT8.md`](BENCH_HBM_M1_KVINT8.md),
  [`BENCH_HBM_M2_CHUNKED.md`](BENCH_HBM_M2_CHUNKED.md)
- M3 Pareto exceptions (TTFT trade-offs, etc.):
  [`/root/bench-int8-w4a16-hbm/m3-tp/pareto_exceptions.md`](/root/bench-int8-w4a16-hbm/m3-tp/pareto_exceptions.md)
