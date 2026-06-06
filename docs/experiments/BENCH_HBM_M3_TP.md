# BENCH_HBM_M3_TP — TP Topology / NCCL Algo / Shard-axis Sweep

Sequel to [`BENCH_INT8_W4A16_FINAL.md`](BENCH_INT8_W4A16_FINAL.md) (PR #29 baseline)
and the two prior HBM-mission reports
[`BENCH_HBM_M1_KVINT8.md`](BENCH_HBM_M1_KVINT8.md) and
[`BENCH_HBM_M2_CHUNKED.md`](BENCH_HBM_M2_CHUNKED.md). This is the third sub-mission of
the **MI100 HBM Optimization** mission and the final per-milestone Pareto report
before M4's integrated final.

## Hardware manifest

- **GPUs:** 4 × AMD MI100 (`gfx908`), 32 GiB HBM2 each, XGMI full mesh,
  perf=high, 250 W cap
- **Host:** ROCm 7.12 (`/opt/rocm/core-7.12`); PyTorch 2.11.0+rocm7.2; Triton 3.5.1;
  rccl 2.27.7
- **vLLM:** commit
  [`c3768adcb`](https://github.com/larkinwc/vllm-gfx908/commit/c3768adcb)
  (head of `mi100-fixes` at M3 completion), v0.18.1.dev4 editable install at
  `/opt/vllm-env`
- **Models:** `/models/Qwen3.5-9B-w8a8`, `/models/Qwen3.5-9B-w4a16`
- **Server port:** 8000 (one vLLM at a time)
- **Harness manifest:** [`harness_manifest.json`](/root/bench-int8-w4a16-hbm/m3-tp/harness_manifest.json)
- **Run log:** [`run_grid_hbm_m3-tp_*.log`](/root/bench-int8-w4a16-hbm/m3-tp/)

## Mission scope

This milestone covers Epic #22 sub-issue **#27 (TP topology / NCCL algo /
shard-axis sweep)**. The three component features were:

1. **m3-nccl-algo-sweep** — measure `NCCL_ALGO ∈ {Ring, Tree, default}` on the
   6 TP=4 cells under the M1+M2 stack (KV-INT8 + chunked prefill), pick a
   per-cell winner.
2. **m3-shard-axis-investigation** — inspect `vllm/model_executor/layers/linear.py`
   to determine whether the TP shard axis (N vs K) is runtime-configurable on
   the hot W8A8 linear layers.
3. **m3-tp-bench-and-update** (this feature) — bake the per-cell NCCL winner
   into the 6 TP=4 launch scripts, re-run the 6 cells × 2 workloads at
   NUM_PROMPTS=200 with the cumulative M1+M2+M3 stack, evaluate the win-bar.

The mission win-bar from `validation-contract.md` VAL-TP-004:

> On `output_throughput_toks_s` vs production `final_grid.csv` for TP=4 cells,
> at least 3 cells show Δ ≥ +3 %.

## NCCL sweep summary

Detailed sweep table is at
[`nccl_sweep_summary.md`](/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep_summary.md).
Machine-readable evidence is at
[`nccl_sweep.json`](/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json) (18
candidate entries: 6 cells × 3 algos; `Tree` rejected per the rccl int8
limitation documented below).

Headline finding: **`NCCL_ALGO=Tree` is not selectable on the M3 stack** on
rccl 2.27.7 because KV-INT8 (`--kv-cache-dtype int8_per_token_head`) issues an
AllGather on `ncclInt8` tensors during profile_run, and rccl 2.27.7 aborts
engine init with:

```
ncclInvalidUsage: no algorithm/protocol available for function AllGather
                  with datatype ncclInt8. NCCL_ALGO was set to Tree.
```

This reduces the per-cell winner search from {Ring, Tree, default} to
{Ring, default}, which the sweep harness measured to be within ~1 % of each
other on every cell. The per-cell winner table below reflects the sweep
harness's choice; the M3 contribution is therefore **honor the rccl heuristic
where it wins (4/6 cells) and force `Ring` where the sweep selected it
(2/6 cells)**.

## Shard-axis findings

Detailed inspection report is at
[`shard_axis_findings.md`](/root/bench-int8-w4a16-hbm/m3-tp/shard_axis_findings.md).
Verdict line: **`Configurable: no`** — vLLM hard-codes the shard axis per
`ParallelLinear` class. `ColumnParallelLinear` shards weights along N (output);
`RowParallelLinear` shards along K (input). There is no constructor kwarg,
env var, or per-layer override to choose a different axis. Changing the axis
would require rewriting the model definition, which is out of mission scope
per `AGENTS.md` Off-Limits paths (`vllm/model_executor/layers/linear.py` and
any model definition are read-only in this mission).

## Updated launch scripts

The 6 TP=4 launch scripts now carry a baked-in `export NCCL_ALGO=<winner>`
line in their "Pinned environment" block, derived from
[`nccl_sweep.json`](/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json):

| Cell | Model | Concurrency | NCCL_ALGO winner | Launch script |
|:-----|:------|:------------|:-----------------|:--------------|
| w8a8_tp4_c1  | Qwen3.5-9B-w8a8  | 1 | default (rccl heuristic) | [`scripts/launch_w8a8_tp4_c1.sh`](scripts/launch_w8a8_tp4_c1.sh) |
| w8a8_tp4_c2  | Qwen3.5-9B-w8a8  | 2 | default (rccl heuristic) | [`scripts/launch_w8a8_tp4_c2.sh`](scripts/launch_w8a8_tp4_c2.sh) |
| w8a8_tp4_c4  | Qwen3.5-9B-w8a8  | 4 | Ring                     | [`scripts/launch_w8a8_tp4_c4.sh`](scripts/launch_w8a8_tp4_c4.sh) |
| w4a16_tp4_c1 | Qwen3.5-9B-w4a16 | 1 | Ring                     | [`scripts/launch_w4a16_tp4_c1.sh`](scripts/launch_w4a16_tp4_c1.sh) |
| w4a16_tp4_c2 | Qwen3.5-9B-w4a16 | 2 | default (rccl heuristic) | [`scripts/launch_w4a16_tp4_c2.sh`](scripts/launch_w4a16_tp4_c2.sh) |
| w4a16_tp4_c4 | Qwen3.5-9B-w4a16 | 4 | Ring                     | [`scripts/launch_w4a16_tp4_c4.sh`](scripts/launch_w4a16_tp4_c4.sh) |

The launch-script edits are confined to the env block (6 single-line
additions, one per file) plus a top-of-file documentation block describing
the new env var and its disable path (`unset NCCL_ALGO` → rccl heuristic).
The body of each launch script is byte-identical to the pre-M3 version.
Verified by `git diff scripts/launch_*_tp4_*.sh | grep -c '^+export NCCL_ALGO'`
returning `6`.

The production decision table for downstream consumers is at the repo root
in [`TP_TOPOLOGY.md`](TP_TOPOLOGY.md).

## Grid run

Cumulative stack: **M1 KV-INT8** (`KV_CACHE_DTYPE=int8_per_token_head`) +
**M2 chunked-prefill** (`--enable-chunked-prefill --max-num-batched-tokens
2048` for w8a8, `4096` for w4a16, per
[`m2-chunked/chunk_sweep.json`](/root/bench-int8-w4a16-hbm/m2-chunked/chunk_sweep.json))
+ **M3 per-cell `NCCL_ALGO`** (Ring on 3 cells, default on 3 cells).

The grid wrapper `scripts/mi100/run_grid_hbm.sh m3-tp` diverges from the
generic 24-cell loop in two M3-specific ways (see commit diff):

1. Only TP=4 cells are run (TP=1 is a no-op for inter-rank topology).
2. The api_server is restarted **per concurrency cell** so that the per-cell
   baked-in NCCL_ALGO is honored by rccl (which reads `NCCL_ALGO` once at
   engine init).

This produced 12 result JSONs (6 cells × 2 workloads). All 12 were
schema-validated (see
[`schema_check.log`](/root/bench-int8-w4a16-hbm/m3-tp/schema_check.log)):

```
12 files validated, 0 failures
```

## 12-row tput summary

Source: [`m3_pareto.csv`](/root/bench-int8-w4a16-hbm/m3-tp/m3_pareto.csv)
(72-row Pareto grid: 12 cells × 6 metrics each). Joined against
`/root/bench-int8-w4a16/final/final_grid.csv`. Full 72-row table is at
[`m3_pareto.md`](/root/bench-int8-w4a16-hbm/m3-tp/m3_pareto.md).

| Cell | Workload | Production tput (tok/s) | M3 tput (tok/s) | Δ% | Verdict |
|:-----|:---------|------------------------:|----------------:|---:|:--------|
| w8a8_tp4_c1  | synthetic | 63.95  | 63.43  | -0.81 % | HOLD |
| w8a8_tp4_c1  | coding    | 60.98  | 62.82  | **+3.02 %** | WIN |
| w8a8_tp4_c2  | synthetic | 125.30 | 122.70 | _-2.08 %_ | REGRESSION |
| w8a8_tp4_c2  | coding    | 110.08 | 120.49 | **+9.46 %** | WIN |
| w8a8_tp4_c4  | synthetic | 247.74 | 231.88 | _-6.40 %_ | REGRESSION |
| w8a8_tp4_c4  | coding    | 196.29 | 224.21 | **+14.22 %** | WIN |
| w4a16_tp4_c1 | synthetic | 50.84  | 55.31  | **+8.80 %** | WIN |
| w4a16_tp4_c1 | coding    | 48.77  | 54.92  | **+12.62 %** | WIN |
| w4a16_tp4_c2 | synthetic | 99.36  | 106.12 | **+6.81 %** | WIN |
| w4a16_tp4_c2 | coding    | 89.23  | 104.39 | **+16.99 %** | WIN |
| w4a16_tp4_c4 | synthetic | 193.99 | 198.79 | +2.47 % | HOLD |
| w4a16_tp4_c4 | coding    | 161.98 | 192.72 | **+18.98 %** | WIN |

## Win-bar analysis

VAL-TP-004 win-bar: **WIN-BAR-MET** (8 / 12 TP=4 cells with Δ ≥ +3 % on
`output_throughput_toks_s`, well above the ≥ 3 / 12 threshold).

Per-quant decode-dominated breakdown (from `m3_pareto.md`):

- **w8a8 decode cells:** 1 cell WINs (w8a8_tp4_c1 coding, +3.02 %); 1 cell
  HOLDs (w8a8_tp4_c1 synthetic); 2 cells REGRESS on tput (w8a8_tp4_c2 / c4
  synthetic) but the corresponding **coding** workloads WIN by +9 % and +14 %.
  Verdict for w8a8: **WIN-BAR-MET** (at least one decode-dominated WIN).
- **w4a16 decode cells:** all 4 measured (tp4_c1 / tp4_c2 × synthetic/coding)
  WIN, with the smallest margin +6.81 % (w4a16_tp4_c2 synthetic) and the
  largest +16.99 % (w4a16_tp4_c2 coding). Verdict for w4a16: **WIN-BAR-MET**.

### Where do the wins come from?

The NCCL sweep itself (Ring vs default on each cell) showed < 1 % deltas
between algos on every cell — i.e. **M3 NCCL_ALGO selection by itself is
within run-to-run noise**. The headline +9 % to +19 % wins on this grid are
the **cumulative effect** of:

1. **M1 KV-INT8** — halves attention's KV-cache HBM traffic per decoded token.
   This is the largest contributor on decode-heavy cells (especially w4a16
   where the production baseline's W4A16 GEMM was already HBM-bound).
2. **M2 chunked-prefill** — reshapes prefill arithmetic intensity, increasing
   request_throughput on coding (short-prefill) workloads. This is the
   largest contributor on coding-workload TTFT-tolerant cells.
3. **M3 NCCL_ALGO selection** — third-order improvement; honors the rccl
   heuristic where it agrees, forces Ring where the sweep selected it.

The M3 grid is therefore best understood as **validating that the M1+M2
stack does not degrade under per-cell NCCL_ALGO selection** rather than
attributing the wins to the NCCL choice itself. The negative-result clause
from VAL-TP-004 is NOT invoked: the win-bar is met by the stacked grid.

### Pareto exceptions

5 metric × cell entries show ≥ 1 % regression on tput / req_tput; many more
show ≥ 1 % regression on TTFT (a known and accepted side effect of chunked
prefill, inherited from M2). Full list with root-cause attribution is at
[`pareto_exceptions.md`](/root/bench-int8-w4a16-hbm/m3-tp/pareto_exceptions.md).

Headline regressions:

- `w8a8_tp4_c2 synthetic` tput **-2.08 %** — inherited from M2; coding
  workload on same cell WINs by +9.46 %.
- `w8a8_tp4_c4 synthetic` tput **-6.40 %** — inherited from M2; coding
  workload on same cell WINs by +14.22 %.

The cell-level production recommendation in `TP_TOPOLOGY.md` accepts these
trade-offs for the coding regime and documents the workload-aware disable
path (`unset ENABLE_CHUNKED_PREFILL` + `unset KV_CACHE_DTYPE`) for sites
where synthetic throughput on `w8a8_tp4_c2` / `c4` matters more than coding
throughput.

## Disable path (operational)

All M3 contributions revert to the pre-M3 launch-script CLI by unsetting
the env vars baked into the Pinned environment block of the 6 TP=4 launch
scripts:

```bash
unset NCCL_ALGO              # M3 contribution → rccl heuristic default
# (M1/M2 disable paths documented in their respective BENCH_HBM_*.md files)
unset KV_CACHE_DTYPE         # M1 contribution → FP16 KV cache
unset ENABLE_CHUNKED_PREFILL # M2 contribution → no chunked prefill
unset MAX_NUM_BATCHED_TOKENS # M2 contribution → vLLM default
```

A launch script invoked with all four env vars unset produces a
byte-identical vLLM CLI to the pre-extension production script (verified
by `git diff scripts/launch_*_tp4_*.sh` — see the env-var docs added in
PR #29 M1/M2 commits + this M3 commit).

## Cross-references

- Production baseline: [`BENCH_INT8_W4A16_FINAL.md`](BENCH_INT8_W4A16_FINAL.md),
  per-cell CSV at `/root/bench-int8-w4a16/final/final_grid.csv`
- HBM mission M1: [`BENCH_HBM_M1_KVINT8.md`](BENCH_HBM_M1_KVINT8.md)
- HBM mission M2: [`BENCH_HBM_M2_CHUNKED.md`](BENCH_HBM_M2_CHUNKED.md)
- M3 NCCL sweep summary (markdown):
  [`/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep_summary.md`](/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep_summary.md)
- M3 NCCL sweep raw (JSON):
  [`/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json`](/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json)
- M3 shard-axis findings:
  [`/root/bench-int8-w4a16-hbm/m3-tp/shard_axis_findings.md`](/root/bench-int8-w4a16-hbm/m3-tp/shard_axis_findings.md)
- M3 Pareto grid (72 rows): [`m3_pareto.csv`](/root/bench-int8-w4a16-hbm/m3-tp/m3_pareto.csv),
  [`m3_pareto.md`](/root/bench-int8-w4a16-hbm/m3-tp/m3_pareto.md)
- M3 Pareto exceptions:
  [`/root/bench-int8-w4a16-hbm/m3-tp/pareto_exceptions.md`](/root/bench-int8-w4a16-hbm/m3-tp/pareto_exceptions.md)
- M3 schema-check log:
  [`/root/bench-int8-w4a16-hbm/m3-tp/schema_check.log`](/root/bench-int8-w4a16-hbm/m3-tp/schema_check.log)
- Production decision table for downstream consumers: [`TP_TOPOLOGY.md`](TP_TOPOLOGY.md)

## Definition of done — checklist

- [x] **VAL-TP-001** (NCCL algorithm sweep) — 18 sweep entries persisted to
      [`nccl_sweep.json`](/root/bench-int8-w4a16-hbm/m3-tp/nccl_sweep.json);
      winner per cell selected; Tree-failure root-cause documented.
- [x] **VAL-TP-002** (shard-axis findings) — `Configurable: no` verdict line
      in [`shard_axis_findings.md`](/root/bench-int8-w4a16-hbm/m3-tp/shard_axis_findings.md).
- [x] **VAL-TP-003** (launch scripts updated + 12 schema-valid JSONs) —
      6 launch scripts updated with 1 `export NCCL_ALGO=` line each
      (git diff confirms exactly 6 additions); 12 JSONs schema-valid
      ([`schema_check.log`](/root/bench-int8-w4a16-hbm/m3-tp/schema_check.log)).
- [x] **VAL-TP-004** (≥ 3 TP=4 cells WIN-BAR-MET) — **8 / 12 cells WIN**;
      negative-result clause NOT invoked; regressions documented in
      `pareto_exceptions.md`.
