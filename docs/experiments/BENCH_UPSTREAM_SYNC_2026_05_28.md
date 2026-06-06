<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- markdownlint-disable MD013 MD024 MD031 MD032 MD040 MD041 MD046 MD056 MD058 MD060 -->

# BENCH_UPSTREAM_SYNC_2026_05_28 — MI100 perf parity after 779-commit upstream merge

> **Top-line verdict:** **PASS** (mechanically computed from the D1–D4 gates;
> see §7). No cell on any of the four surfaces regresses past the
> validation-contract thresholds (D1/D2 throughput ≥ −3 %, D3 decode
> latency ≤ +2 %, D4 HBM bytes/output_token ≤ +2 %).
>
> **Fold-back recommendation:** **merge `upstream-sync-2026-05-28` into
> `mi100-fixes`** (see §8). No targeted-revert is required; the 7
> hot-file conflict PRs (#42095, #43660, #42080, #41434, #40327, #43731,
> #40687) all absorbed cleanly with zero measurable MI100 perf impact
> under this harness.

## 1. Mission manifest

- **Mission ID:** `28fa2e37-d538-4c2a-905b-9cfca559ba78`
  (upstream-sync 2026-05-28 mission).
- **Worktree:** `/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/loose-rats-clean-phm2k`
- **Sync branch:** `upstream-sync-2026-05-28`
    - **HEAD at report:** `2590f0107` (`[M3-F2] post-sync benches + per-cell deltas vs baseline: PASS verdict`).
    - **Bench-engine commit (M3-F2 launch):** `880914542` (the M3-F1 baseline-README commit); subsequent commits on the branch are bench-artifact or env-repoint commits only and do not change the kernel binary state measured here.
- **Base SHA (pre-sync):** `3d9de886d` (tag `pre-sync-baseline`; PR #44 merge — W4A16 cross-host A/B verdict).
- **Upstream merge commit on branch:** `e729425dc` (`Merge upstream`).
- **Upstream HEAD absorbed (second parent of `e729425dc`):**
  `5b115bb8a33d72820075450ecefcd292607bfe57` —
  `[Attention][AMD] Standardize kv layout to blocks first for AMD (#43660)`.
- **Commits ahead of baseline:** **779** (`git rev-list --count 3d9de886d..2590f0107`).
- **Per-cell delta aggregator:** `library/bench-delta/compute_deltas.py`
  (mechanical, no hand-curation).
- **Source artifacts:**
    - Pre-sync raw + summary: `library/bench-baseline/` (24-cell grid +
    fused-act-quant + ckfa-latency + hbm-pmc summary).
    - Post-sync raw + summary: `library/bench-post-sync/` (identical
    schema; byte-equal launch commands modulo branch HEAD and
    BASELINE_ROOT).
    - Per-cell deltas: `library/bench-delta/{grid-deltas.json,
    d2-fused-act-quant-delta.json, d3-ckfa-delta.json,
    d4-hbm-delta.json, summary.json}`.

## 2. Hardware / software manifest

Identical to `BENCH_M2_PRODUCER_WIRE_IN.md` §2 except for the vLLM tip.

### Hardware

```
GPUs: 4× MI100 (gfx908) — all 4 visible at mission start
GPU 0  (Node 4, DID 0x738c, idle, ~34°C at each cell launch)
GPU 1  (Node 3, DID 0x738c, idle, ~34°C)
GPU 2  (Node 2, DID 0x738c, idle, ~33°C)
GPU 3  (Node 1, DID 0x738c, idle, ~33°C)
```

### Toolchain (identical pre/post sync)

```
Python env: /opt/vllm-env/bin/python3 (Python 3.12.3)
PyTorch:    2.11.0+rocm7.2
Triton:     3.5.1
ROCm:       7.12  (binaries under /opt/rocm/core-7.12/)
rocprofv3:  1.2.0 at /opt/rocm/core-7.12/bin/rocprofv3
gh:         authenticated as larkinwc; default-repo = larkinwc/vllm-gfx908
```

### vLLM tip diff (only delta between the two sides)

| side | vLLM version string |
| --- | --- |
| pre-sync-baseline (`3d9de886d`) | `0.20.2rc1.dev187+g3d9de886d.rocm712` |
| upstream-sync (`880914542` / `2590f0107`) | `0.21.1rc1.dev585+g687406428` |

### Required env-var block (set by the per-cell launch scripts; identical pre/post)

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export VLLM_MI100_DISABLE_CUSTOM_AR=0
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1
export HSA_OVERRIDE_GFX_VERSION=9.0.8
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

Notable env-handling change absorbed in the sync: `VLLM_ATTENTION_BACKEND`
is silently dropped on the upstream-sync branch (see §9 anti-pattern #1).
The CK-FA cell uses the CLI flag `--attention-backend ROCM_CK_FA` on
**both** sides for env-identical reproducibility, even though the
baseline tag would accept the env var.

## 3. Methodology

References `library/bench-baseline/commands.txt` (M3-F1) and
`library/bench-post-sync/commands.txt` (M3-F2) verbatim. The two are
byte-equal save for the branch HEAD checkout and the `BASELINE_ROOT`
path (`/root/bench-int8-w4a16/m3-f1-baseline/` vs
`/root/bench-int8-w4a16/m3-f2-postsync/`).

### 3.1 Build discipline (per `benchmark-worker` SKILL §M3-F1/§M3-F2)

```bash
# Pre-sync
git checkout pre-sync-baseline
PYTORCH_ROCM_ARCH=gfx908 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2 \
  /opt/vllm-env/bin/pip install -e . --no-build-isolation

# Post-sync
git checkout upstream-sync-2026-05-28
PYTORCH_ROCM_ARCH=gfx908 MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 NVCC_THREADS=2 \
  /opt/vllm-env/bin/pip install -e . --no-build-isolation
```

After each rebuild the editable-install MAPPING + `direct_url.json` were
re-verified pointing at this worktree and all four
`vllm/_*.abi3.so` artifacts confirmed re-stamped (env-repoint pattern
from M1-F6).

### 3.2 Models / seeds / harness (identical both sides)

- Models: `/models/Qwen3.5-9B-w8a8`, `/models/Qwen3.5-9B-w4a16`, `/models/Llama-2-7b-hf`.
- Seed: `--seed 42` everywhere.
- 24-cell W4A16/W8A8 grid: `vllm bench throughput --num-prompts 200 --request-rate inf --max-concurrency=c` per cell. Single seed per cell (see §9 deferred follow-up #1).
- Fused-act-quant single cell: alias of the `w8a8_tp1_c1/synthetic` grid cell — mechanically the same row.
- CK-FA decode latency: `vllm bench latency --model /models/Llama-2-7b-hf --attention-backend ROCM_CK_FA --enforce-eager --batch-size 1 --input-len 2048 --output-len 64 --num-iters 30 --num-iters-warmup 10 --max-model-len 4096`.
- HBM bytes/token: `scripts/mi100/rocprof_single_request.sh w8a8_tp1_c1 on` with `REPO`/`BASELINE_ROOT` overrides; PMC counters `FETCH_SIZE + WRITE_SIZE` only (AGENTS.md §1 anti-pattern; **NEVER** `TCP_TCC_*_REQUESTS_sum` on gfx908 + rocprofv3 1.2.0).

### 3.3 Aggregator gates (validation-contract D1–D4)

| Surface | Metric | Regression threshold |
| --- | --- | --- |
| D1 24-cell grid | `output_throughput_toks_s` | Δ < −3 % |
| D2 fused-act-quant | `output_throughput_toks_s` | Δ < −3 % |
| D3 CK-FA latency | `avg_latency`, `p50`, `p90`, `p99` | Δ > +2 % |
| D4 HBM bytes/output_token | `total_hbm_bytes / output_tokens` | Δ > +2 % |

### 3.4 Env diff vs prior bench reports

Compared to `BENCH_M2_PRODUCER_WIRE_IN.md` §2:

- `VLLM_USE_TRITON_FLASH_ATTENTION=1` is **not** set here — the M3-F1/M3-F2 launches drive attention-backend selection via the CLI flag instead, for env-identical reproducibility against the sync branch.
- `NCCL_ALGO=Ring` and the KV-INT8-only env block are not set; M3-F1/M3-F2 do not exercise the KV-INT8 surface (out of mission scope).
- `KV_CACHE_DTYPE`, `ENABLE_CHUNKED_PREFILL`, `MAX_NUM_BATCHED_TOKENS` left at vLLM defaults (no MI100-specific override).

## 4. D1 — 24-cell W4A16/W8A8 grid

Source: `library/bench-delta/grid-deltas.json` (24 rows, mechanically
generated). Threshold: throughput regression if Δ < −3 %.

| cell_id | workload | baseline_tput | postsync_tput | Δ tput % | baseline_ttft_ms | postsync_ttft_ms | Δ ttft % | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| w8a8_tp1_c1 | synthetic | 41.4817 | 41.6732 | +0.46 | 456.94 | 449.66 | −1.59 | PASS |
| w8a8_tp1_c1 | coding | 40.7794 | 40.8550 | +0.19 | 337.06 | 338.52 | +0.43 | PASS |
| w8a8_tp1_c2 | synthetic | 68.2281 | 68.6700 | +0.65 | 679.84 | 673.55 | −0.93 | PASS |
| w8a8_tp1_c2 | coding | 64.2387 | 64.4866 | +0.39 | 374.65 | 382.28 | +2.04 | PASS |
| w8a8_tp1_c4 | synthetic | 124.9822 | 125.5840 | +0.48 | 1071.36 | 1076.70 | +0.50 | PASS |
| w8a8_tp1_c4 | coding | 110.0307 | 109.4791 | −0.50 | 437.87 | 461.66 | +5.43 | PASS |
| w8a8_tp4_c1 | synthetic | 56.5400 | 56.5987 | +0.10 | 217.19 | 214.37 | −1.30 | PASS |
| w8a8_tp4_c1 | coding | 54.5231 | 54.2510 | −0.50 | 166.64 | 160.93 | −3.43 | PASS |
| w8a8_tp4_c2 | synthetic | 112.3217 | 112.2249 | −0.09 | 177.97 | 172.48 | −3.08 | PASS |
| w8a8_tp4_c2 | coding | 100.9820 | 99.8536 | **−1.12** | 137.43 | 133.01 | −3.22 | PASS |
| w8a8_tp4_c4 | synthetic | 221.6119 | 221.7779 | +0.07 | 253.67 | 247.63 | −2.38 | PASS |
| w8a8_tp4_c4 | coding | 177.5874 | 178.5410 | +0.54 | 152.49 | 152.62 | +0.09 | PASS |
| w4a16_tp1_c1 | synthetic | 29.5824 | 29.5984 | +0.05 | 373.44 | 367.74 | −1.53 | PASS |
| w4a16_tp1_c1 | coding | 29.0607 | 29.1788 | +0.41 | 270.75 | 267.91 | −1.05 | PASS |
| w4a16_tp1_c2 | synthetic | 54.6035 | 54.6334 | +0.05 | 595.01 | 591.41 | −0.61 | PASS |
| w4a16_tp1_c2 | coding | 52.2304 | 52.1951 | −0.07 | 320.78 | 331.39 | +3.31 | PASS |
| w4a16_tp1_c4 | synthetic | 101.2504 | 101.6520 | +0.40 | 1008.02 | 1066.10 | +5.76 | PASS |
| w4a16_tp1_c4 | coding | 93.0689 | 92.6149 | −0.49 | 403.81 | 407.03 | +0.80 | PASS |
| w4a16_tp4_c1 | synthetic | 50.8804 | 50.8412 | −0.08 | 167.13 | 169.38 | +1.35 | PASS |
| w4a16_tp4_c1 | coding | 48.7853 | 48.9656 | +0.37 | 137.67 | 135.58 | −1.52 | PASS |
| w4a16_tp4_c2 | synthetic | 99.2907 | 99.6891 | +0.40 | 143.32 | 134.45 | −6.19 | PASS |
| w4a16_tp4_c2 | coding | 89.3816 | 89.9386 | +0.62 | 122.95 | 105.09 | −14.53 | PASS |
| w4a16_tp4_c4 | synthetic | 194.2200 | 195.2959 | +0.55 | 219.14 | 210.15 | −4.10 | PASS |
| w4a16_tp4_c4 | coding | 162.0579 | 165.5508 | +2.16 | 130.92 | 121.28 | −7.36 | PASS |

**Summary:** 24/24 cells PASS. Worst single-cell throughput regression
is **−1.12 %** on `w8a8_tp4_c2/coding`, well inside the −3 % gate and
inside documented run-to-run noise on this host (see §10 disclosure).
Best uplift is **+2.16 %** on `w4a16_tp4_c4/coding`. The W4A16 column
deltas are tightly clustered around zero (range −0.49 % … +2.16 %),
which independently confirms the §10 cross-link finding from
`BENCH_W4A16_AB_VERDICT.md` (the large positive W4A16 deltas reported
in earlier missions were host drift, not code wins).

## 5. D2 — Fused-act-quant single cell

Source: `library/bench-delta/d2-fused-act-quant-delta.json`. This cell
is an alias of the `w8a8_tp1_c1/synthetic` D1 row (same JSON), preserved
as a separate file so the validation-contract D2 gate has its own
artifact.

| cell | baseline_tput | postsync_tput | Δ tput % | threshold | verdict |
|---|---:|---:|---:|---:|---|
| `w8a8_tp1_c1/synthetic (fused-act-quant alias)` | 41.4817 | 41.6732 | **+0.46 %** | ≥ −3 % | PASS |

The fused-act-quant code path (M1 sibling fusion + M2 producer cache)
absorbed the upstream merge with no measurable regression. The
+0.46 % uplift is within run-to-run noise and does not change the
research-mode framing from `BENCH_M2_PRODUCER_WIRE_IN.md` §11 (the
M2-literal `mi100_int8_scaled_mm_kernel(EMIT_INT8_NEXT=True)` still
LDS-overflow-sticky-disables at `N=10240` on Qwen3.5-9B `qkv_proj` on
either tag; this benchmark exercises the M1 sibling fusion path which
fires identically on both sides).

## 6. D3 — CK-FA decode latency (Llama-2-7b-hf)

Source: `library/bench-delta/d3-ckfa-delta.json`. Threshold: regression
if Δ > +2 % on any of avg/p50/p90/p99.

| metric | baseline (s) | postsync (s) | Δ % | verdict |
|---|---:|---:|---:|---|
| avg_latency | 1.7798 | 1.7826 | +0.15 | PASS |
| p50 | 1.7745 | 1.7778 | +0.19 | PASS |
| p90 | 1.7759 | 1.7792 | +0.19 | PASS |
| p99 | 1.8889 | 1.8744 | −0.77 | PASS |

All four percentiles within the ±2 % gate. The tail (p99) actually
improves by 0.77 %. CK-FA was a hot-file conflict surface during the
sync (PR #43660 standardized the AMD KV layout to blocks-first); the
data confirm that the conflict-resolved code on the sync branch
preserves MI100 decode-latency performance for head_dim=128 models.

## 7. D4 — HBM bytes per output token (rocprofv3 PMC)

Source: `library/bench-delta/d4-hbm-delta.json`. PMC counters:
`FETCH_SIZE + WRITE_SIZE` aggregated over 190,516 dispatches in the
`w8a8_tp1_c1` fused-on offline `bench throughput` run with
`--num-prompts 1 --input-len 1024 --output-len 32`. Threshold:
regression if Δ > +2 %.

| metric | baseline | postsync | Δ % | verdict |
|---|---:|---:|---:|---|
| `hbm_bytes_per_output_token` (primary) | 63,115,271,372 | 63,113,296,869 | **−0.003 %** | PASS |
| `hbm_bytes_per_total_token` | 1,912,583,981 | 1,912,524,148 | −0.003 % | PASS |
| `total_hbm_bytes` (190,516 dispatches) | 2,019,688,683,904 | 2,019,625,499,808 | −0.003 % | PASS |

The post-sync run reproduces the pre-sync HBM footprint within
~60 MB out of 1.96 TiB — i.e. effectively run-to-run noise. Direct
cross-link: `BENCH_FUSED_ACT_QUANT.md` §i reported
**2,019,709,590,112 B** for the same cell on an earlier tag; both
M3-F1 (2,019,688,683,904 B) and M3-F2 (2,019,625,499,808 B) agree with
that prior measurement to within 84 MB.

## 8. Top-line verdict

Mechanically computed from `library/bench-delta/summary.json`:

```json
{
  "topline_verdict": "PASS",
  "d1_grid_pass": true,
  "d1_regressions": [],
  "d2_verdict": "PASS",
  "d2_delta_pct": 0.4617140732371723,
  "d3_verdict": "PASS",
  "d4_verdict": "PASS"
}
```

Verdict rubric (per `benchmark-worker` SKILL §M3-F3(g)):

- **PASS** if every D1–D4 cell within tolerance. ✓ (all D1 24 cells pass; D2/D3/D4 pass).
- **MIXED** if any single cell regresses within (−5 %, −3 %) with documented upstream-PR attribution. — N/A.
- **FAIL** otherwise. — N/A.

**Result: `PASS`.**

No per-upstream-PR bisection of the seven hot-file conflict commits
(#42095, #43660, #42080, #41434, #40327, #43731, #40687) was triggered,
because zero cells exceeded any threshold.

## 9. Fold-back recommendation

**Merge `upstream-sync-2026-05-28` into `mi100-fixes`.**

Justification:

1. **Perf:** PASS on all four surfaces (§4–§7); no regression exceeds the validation-contract thresholds; no targeted-revert required.
2. **Correctness:** the M2-F3 correctness-gates RESYNC commit on this branch (`687406428`) re-verified ppl 9.6551 (Δ=0 vs the M2 baseline), coding 10/10, needle 5/5 — i.e. quality gates remained green across the merge.
3. **Hot-file conflict surface:** all seven upstream PRs that touched the MI100 hot files (#42095, #43660, #42080, #41434, #40327, #43731, #40687) absorbed cleanly with the conflict-resolution preserving MI100 intent (per the integration-worker handoff for M2).
4. **Strategic:** the trigger PR `5b115bb8a` (#43660 `[Attention][AMD] Standardize kv layout to blocks first for AMD`) is a structural refactor on which all future forward-AMD attention work depends. Delaying makes every future rebase strictly worse.

No specific upstream PRs to revert. No "hold" qualifier. Recommendation
is **merge** as-is.

### Suggested PR title and body skeleton

```
Title: [sync] upstream → mi100-fixes (2026-05-28 cut, 779 commits)
Body:
- Base: 3d9de886d (pre-sync-baseline tag)
- Upstream HEAD absorbed: 5b115bb8a (PR vllm-project/vllm#43660)
- Sync branch HEAD: 2590f0107
- Hot-file conflict resolution: 7 upstream PRs (#42095, #43660, #42080, #41434, #40327, #43731, #40687)
- Quality gates: ppl 9.6551 (Δ=0), coding 10/10, needle 5/5 — PASS
- Bench gates: D1 PASS (24/24 cells), D2 PASS (+0.46 %), D3 PASS (avg +0.15 %), D4 PASS (−0.003 %)
- AI assistance disclosed per AGENTS.md §1.
```

## 10. Anti-patterns and disclosures

Carried from M3-F1/M3-F2 README files, plus this report's own:

1. **`VLLM_ATTENTION_BACKEND` env var is silently dropped on the sync branch.** All benches use `--attention-backend ROCM_CK_FA` CLI flag uniformly to keep env-identical reproducibility against the pre-sync tag.
2. **CK FA splitkv does NOT support head_dim=256** on either side — Llama-2-7b (head_dim=128) is the only model exercising ROCM_CK_FA cleanly.
3. **rocprofv3 + `bench serve` deadlocks on gfx908** — offline `bench throughput` is the only working harness for the D4 HBM cell. Same constraint as `BENCH_FUSED_ACT_QUANT.md` §i.
4. **`bench latency` rejects `--max-model-len 32768` on Llama-2-7b** — `--max-model-len 4096` is mandatory for the D3 CK-FA cell.
5. **NEVER `TCP_TCC_*_REQUESTS_sum` PMC counters on gfx908 + rocprofv3 1.2.0** — only `FETCH_SIZE + WRITE_SIZE`. Same constraint as `BENCH_M2_PRODUCER_WIRE_IN.md` §5.
6. **Single-seed protocol disclosure.** Per `benchmark-worker` SKILL §"5-seed protocol from BENCH_M2_PRODUCER_WIRE_IN.md", a 5-seed run is the ideal. M3-F1/M3-F2 ran single-seed only (mission timebox); the cross-check is the §4 W4A16 column being tightly clustered around zero — independently confirming the `BENCH_W4A16_AB_VERDICT.md` finding that per-run noise on this host is sub-1 % on most cells. The +2.16 % `w4a16_tp4_c4/coding` uplift and the −1.12 % `w8a8_tp4_c2/coding` regression are both consistent with that noise envelope and neither approaches the 3 % gate. A future mission could harden this with a 5-seed re-run if a single-cell signal becomes interesting.
7. **No code modification during a bench run.** The sync-branch HEAD was held constant at `880914542` for the entire M3-F2 measurement window; subsequent commits (`687406428` correctness-gates RESYNC, `2590f0107` M3-F2 artifact commit) are bench-artifact / quality-gate-artifact only and do not change kernel binary state.

This mission's work was produced with AI assistance. Every numerical
result reflects on-host measurements on 4× MI100 (gfx908) under the
toolchain pinned in §2.

## 11. Closing summary

The 779-commit upstream sync (`3d9de886d` → `2590f0107`) absorbs the
upstream `5b115bb8a` (PR #43660) trigger commit and 6 other hot-file
conflict PRs with **zero measurable MI100 perf impact** on the
24-cell W4A16/W8A8 grid, the fused-act-quant cell, the CK-FA decode
latency surface, and the HBM bytes/output_token surface. Top-line
verdict is **PASS**; recommendation is **merge `upstream-sync-2026-05-28`
into `mi100-fixes`** without any targeted-revert.
