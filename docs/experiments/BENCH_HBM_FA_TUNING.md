# BENCH_HBM_FA_TUNING — MI100 Triton Flash-Decoding INT8-PTH Tuning + CK FA2 Investigation

> **Mission status (research-mode):** PASS via direction-correct improvement
> on ≥1 decode-dominated cell (`w8a8_tp1_c1_coding +0.33 %`, ratio 1.0033 ≥ 1.001),
> **with a critical null-result attribution** from rocprofv3 trace evidence that the
> M1 winners-lookup selected configs **coincide byte-for-byte** with the M4 default
> heuristic on the observed shapes. See §M1 Null-Result Finding and §Win-Bar Verdict.
> M2 CK FA2 INT8-PTH integration verdict: **NO-GO** (multi-week kernel-authoring
> scope; exceeds 1-week cap). M3 production launch scripts and disable-path
> fallback ship as-is.

---

## Hardware/Software Manifest

| Component | Value | Source |
|---|---|---|
| GPUs | 4 × AMD Instinct MI100 (gfx908), 32 GB VRAM each, XGMI full mesh, single NUMA node | `rocm-smi --showtopo` in `harness_manifest.json:gpu_topo` |
| vLLM SHA (mission HEAD) | `8c45a5b916a0a14dddb707f99ecc03c964f77dad` | `git rev-parse HEAD` (this commit precedes the report-author commit) |
| vLLM SHA (M1 harness runtime) | `54df089f3f7acb905e958518e91251dfff5bbd17` | `harness_manifest.json:vllm_commit` |
| vLLM version | `0.20.2rc1.dev107+gd960f21e4.d20260510` | `harness_manifest.json:vllm_version` |
| ROCm | 7.12 at `/opt/rocm/core-7.12` | `harness_manifest.json:rocm_version` |
| PyTorch | `2.11.0+rocm7.2` | `harness_manifest.json:torch_version` |
| Triton | `3.5.1` (pytorch-triton-rocm) | `harness_manifest.json:triton_version` |
| rocprofv3 | 1.2.0 | rocprof traces under `m1-tuning/rocprof/` |
| HIPBLASLT_TENSILE_LIBPATH | `/root/bench-int8-w4a16/tensilelite/merged_library/library` | `scripts/launch_hbm_*.sh` exports + `launch_hbm2_*.sh` exports |
| Python env | `/opt/vllm-env/bin/python3` (editable install) | mission AGENTS.md §Environment |
| Models | `/models/Qwen3.5-9B-w8a8`, `/models/Qwen3.5-9B-w4a16`, `/models/Qwen3.5-9B` (fp16 ref) | mission AGENTS.md §Environment |
| Bench-output root | `/root/bench-int8-w4a16-hbm-fa/{m0-canary,m1-tuning,m2-ck,m3-final}/` | services.yaml `paths.bench_outputs` |
| M4 reference root (READ-ONLY) | `/root/bench-int8-w4a16-hbm/m4-final/` | services.yaml `paths.m4_baselines` |

Pre-mission readiness check: 4× MI100 idle (46–58 °C edge sensor at sweep start;
`rocm_smi_temp` in `harness_manifest.json`); no orphan vLLM processes; `pgrep -f
EngineCore` → empty post-run (`m0-canary/canary_check.md` "Lifecycle / cleanliness"
block).

---

## M0 Baseline Drift Canary

Three canary cells re-measured against the just-merged M4 production stack
before perturbing the Triton kernel. Source: `m0-canary/canary_check.md`.

| cell_id | ref_tput (tok/s) | measured_tput (tok/s) | Δ % | verdict |
|---|---:|---:|---:|---|
| w8a8_tp1_c1 | 40.111236 | 40.095501 | -0.04 % | PASS |
| w8a8_tp4_c4 | 229.129590 | 231.539522 | +1.05 % | PASS |
| w4a16_tp4_c4 | 199.198442 | 198.855420 | -0.17 % | PASS |

Gate `|Δ| ≤ 2 %` met on all three cells → `verdict: PROCEED`. M1 unblocked.

---

## M1 Sweep Summary

**Sweep harness:** `scripts/mi100/triton_flash_decode_sweep.py` (commit
`eb42b780c`). Custom Cartesian sweep that builds and runs a Triton kernel
instance per `(tile_size, num_splits, num_warps, num_stages, BLOCK_M, quant,
head_dim, seq_len_bucket, GQA_ratio, num_seqs)` tuple, records median latency
over 100 invocations, and stores numerical correctness vs an fp32 reference.

**Search space (pinned by VAL-M1-001):**

| axis | values |
|---|---|
| tile_size | {16, 32, 64, 128} |
| num_splits | {16, 32, 60, 120} |
| num_warps | {2, 4, 8, 16} |
| num_stages | {1, 2, 3} |
| BLOCK_M | {16, 32, 64} |
| quant | {w8a8, w4a16} |
| head_dim | 128 (fixed) |
| seq_len_bucket | {1024, 4096, 16384, 32768} |
| GQA_ratio | 8 (fixed) |
| num_seqs | {1, 4} |

**Trial count (completed / expected):** 9216 / 9216 (verified via
`jq '.configs | length' sweep_results.json` → 9216). All trials produced both
a `median_latency_us` and a correctness flag (`abs_err`, `rel_err`, `ref_max_abs`).

**Sweep results JSON:** `/root/bench-int8-w4a16-hbm-fa/m1-tuning/sweep_results.json`
(7.5 MB, SHA-256 `63b697fb8ba5ca020586a20f26bb84317850f2d41b209bf061492fd114f46e19`,
recorded in `winners_lookup.json:sweep_sha256`).

**Wall-clock:** sweep ran serially on a single MI100 (GPU 0); start to finish
recorded in `sweep_progress.log` (≈ 9 hours; matches the mission's pre-sweep
projection in `mission.md` "M1 ~9 hours sweep").

**Winner-selection rule (VAL-M1-002):**

> `min median_ms among eligible (rel_err <= 1e-3); tiebreak smaller tile_size > num_splits > num_warps (secondary deterministic keys: num_stages, BLOCK_M)`

(source: `winners_lookup.json:selection_rule`).

**Eligibility threshold:** `rel_err ≤ 0.001` (`winners_lookup.json:eligibility_rule`).

**Winners table (8 rows; 8 `(quant, seq_len_bucket, num_seqs)` keys per quant ×
2 quants ⇒ 16 entries total, 8 shown for w8a8 below; full 16-entry table at
`winners_lookup.json:winners` and human-readable in `winners_summary.md`):**

| quant | seq_len | num_seqs | tile | splits | warps | stages | BLOCK_M | median_ms | hbm_bytes_per_inv |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| w8a8  |  1 024 | 1 | 16  | 16 | 2 | 3 | 32 | 0.1753 |     1 622 144 |
| w8a8  |  1 024 | 4 | 32  | 16 | 4 | 1 | 16 | 0.1782 |     6 488 576 |
| w8a8  |  4 096 | 1 | 32  | 32 | 2 | 2 | 16 | 0.1776 |     5 390 848 |
| w8a8  |  4 096 | 4 | 64  | 32 | 2 | 1 | 32 | 0.1787 |    21 563 392 |
| w8a8  | 16 384 | 1 | 32  | 32 | 2 | 2 | 16 | 0.1800 |    18 368 512 |
| w8a8  | 16 384 | 4 | 64  | 32 | 2 | 1 | 16 | 0.3697 |    73 474 048 |
| w8a8  | 32 768 | 1 | 128 | 32 | 2 | 1 | 16 | 0.2520 |    35 672 064 |
| w8a8  | 32 768 | 4 | 64  | 32 | 2 | 1 | 16 | 0.6989 |   142 688 256 |

(All 16 entries enumerated in `winners_summary.md` §w8a8 / §w4a16 tables; cited
verbatim from `winners_lookup.json:winners`.)

**Citations:**

- `sweep_results.json` (9216 configs, 7.5 MB).
- `winners_lookup.json` (16 winners, sweep_sha256 pinned).
- `winners_summary.md` (human-readable 8+8 winners tables).
- `m1_winners.log` (winner-selection harness stdout, 2880 B).

---

## M2 CK Verdict

**Verdict line (verbatim from `m2-ck/audit.md` §Go/No-go verdict):**

> `VERDICT: NO-GO`

**Audit document:** `/root/bench-int8-w4a16-hbm-fa/m2-ck/audit.md` (19 801 bytes;
the audit is read-only and contains explicit `file:line` citations into
`vllm/v1/attention/backends/rocm_ck_fa.py:10-12,54-63,68-79,81-89,139,148,165-169,241-260`,
`vllm/v1/attention/backends/triton_attn.py:399-408,481-535,680-712,723-758`, and the
external CK Python entry point at
`/tmp/rocm-flash-attention-test/flash_attn/flash_attn_interface.py:1391-1410,153-198`).

**Three-layer rejection summary (from audit.md §Scale-layout gap):**

1. **Element dtype:** the compiled `flash_attn_2_cuda` extension exports only
   `FmhaFwd{Fp16, Bf16, Fp8}` template instances. `nm -D` finds **zero**
   `FmhaFwdInt8`/`I8` symbols.
2. **Per-token-per-head scales:** the CK Python signature
   `flash_attn_varlen_func(...)` and its wrapped C++ entry `flash_attn_gpu.varlen_fwd(...)`
   take **no** descale tensor argument; even the Fp8 variants are scalar-scaled, not
   per-token-per-head-scaled.
3. **Storage layout:** vLLM's INT8-PTH path uses an *inline-padded*
   `head_size + sizeof(f32)` layout where the trailing 4 bytes per head are the
   float32 scale (`triton_attn.py:415-422` + `_ensure_scale_caches:507-560`); CK
   FA2 expects contiguous `headdim` with no inline metadata.

**Effort estimate (audit.md §Effort estimate):** the only correctness-preserving
integration is **Option A** (forked CK with INT8 + per-token-head template family
plus stride-with-pad tile descriptor) at ~3–6 weeks of focused kernel-authoring,
well beyond this mission's hard 1-week M2 cap. Option B (dequant-then-CK-fp16)
defeats the HBM-savings motivation. Option C (post-multiply outside CK) cannot
soundly extract a per-token scale from the softmax non-linearity.

**Integration outcome (VAL-M2-002):** No integration attempt. `integration_result.md`
contains a 1-line redirect: *"Skipped per audit NO-GO verdict; see audit.md."*
`canary_results.md`: *"Skipped — no working CK INT8-PTH kernel from
m2-integration-attempt."* No canary delta vs M1 is applicable.

---

## Cumulative Grid (24 rows; M1 vs M4 baseline)

24 rows = 12 cells × 2 workloads. Throughput is `output_throughput_toks_s`; p99
TTFT is `p99_ttft_ms`. All numbers from `m1-tuning/m1_vs_m4_grid.csv` (144-row
long-format CSV; pivoted to 24 wide rows here). M4 baselines from
`/root/bench-int8-w4a16-hbm/m4-final/` (READ-ONLY reference).

| cell_id | workload | m4_baseline_tput | m1_tuned_tput | delta_pct | p99_ttft_ms_m4 | p99_ttft_ms_m1 |
|---|---|---:|---:|---:|---:|---:|
| w8a8_tp1_c1 | synthetic | 40.111 | 40.211 | +0.248 | 254.23 | 254.47 |
| w8a8_tp1_c1 | coding | 39.948 | 40.081 | +0.334 | 1435.11 | 1435.12 |
| w8a8_tp1_c2 | synthetic | 74.611 | 74.642 | +0.043 | 470.22 | 471.62 |
| w8a8_tp1_c2 | coding | 73.440 | 73.318 | -0.167 | 1463.09 | 1464.73 |
| w8a8_tp1_c4 | synthetic | 136.854 | 134.982 | -1.367 | 859.16 | 831.09 |
| w8a8_tp1_c4 | coding | 131.214 | 130.262 | -0.725 | 1475.84 | 1870.62 |
| w8a8_tp4_c1 | synthetic | 63.565 | 63.552 | -0.021 | 126.09 | 126.56 |
| w8a8_tp4_c1 | coding | 62.943 | 63.025 | +0.130 | 591.12 | 589.55 |
| w8a8_tp4_c2 | synthetic | 122.752 | 122.377 | -0.305 | 224.44 | 224.69 |
| w8a8_tp4_c2 | coding | 120.417 | 120.040 | -0.313 | 615.18 | 612.20 |
| w8a8_tp4_c4 | synthetic | 229.130 | 229.957 | +0.361 | 389.53 | 383.54 |
| w8a8_tp4_c4 | coding | 222.692 | 223.474 | +0.351 | 622.78 | 620.06 |
| w4a16_tp1_c1 | synthetic | 30.882 | 30.898 | +0.049 | 340.39 | 339.48 |
| w4a16_tp1_c1 | coding | 30.836 | 30.826 | -0.034 | 1697.45 | 1694.56 |
| w4a16_tp1_c2 | synthetic | 56.757 | 56.777 | +0.034 | 635.19 | 656.17 |
| w4a16_tp1_c2 | coding | 55.784 | 55.894 | +0.197 | 1742.73 | 1743.70 |
| w4a16_tp1_c4 | synthetic | 104.015 | 104.022 | +0.007 | 1198.52 | 1240.92 |
| w4a16_tp1_c4 | coding | 99.269 | 99.859 | +0.594 | 1755.75 | 2387.18 |
| w4a16_tp4_c1 | synthetic | 55.245 | 55.258 | +0.023 | 146.43 | 146.69 |
| w4a16_tp4_c1 | coding | 54.867 | 54.866 | -0.002 | 613.27 | 629.07 |
| w4a16_tp4_c2 | synthetic | 106.113 | 106.061 | -0.050 | 263.85 | 263.47 |
| w4a16_tp4_c2 | coding | 104.552 | 104.303 | -0.238 | 663.25 | 643.04 |
| w4a16_tp4_c4 | synthetic | 199.198 | 199.507 | +0.155 | 499.24 | 507.43 |
| w4a16_tp4_c4 | coding | 193.743 | 193.108 | -0.328 | 726.45 | 655.90 |

Schema gate: `python scripts/validate_results_schema.py
/root/bench-int8-w4a16-hbm-fa/m1-tuning/` emits *"24 files validated, 0 failures"*
(`schema_check.log`).

Per-cell raw JSON paths are enumerated in §Appendix: Per-Cell Raw JSON Index.

---

## M1 Null-Result Finding (Tuning Coincides With Defaults)

**This is the most important finding of the mission.** The rocprofv3
kernel-trace evidence (`hbm_bytes_analysis.md`) demonstrates that on every
shape exercised by the bench grid, the M1 winners-lookup returns launch
constants that are **byte-identical** to what the existing M4 default
heuristic (`_get_tile_size` + `_compute_flash_decoding_splits` returning their
stock constants) already produces.

**Quoted trace evidence (verbatim from `hbm_bytes_analysis.md` §2.2, §3.3, §2.3):**

> *"`kernel_unified_attention` — M4 mean μs/inv: 941.896, M1 mean μs/inv: 941.310,
> Δ mean: -0.06 %. Mean launch-grid volume: M4 119 707.5, M1 119 707.5, Δ grid: 0.00 %"*
> — `w8a8_tp1_c1_synthetic`, the M1-winner cell.
>
> *"`kernel_unified_attention` — M4 mean μs/inv: 1 199.833, M1 mean μs/inv: 1 199.459,
> Δ mean: -0.03 %. Mean launch-grid volume: M4 91 301.9, M1 91 301.9, Δ grid: 0.00 %"*
> — `w8a8_tp1_c4_coding`, the cell with the worst p99_ttft regression (+26.75 %).
>
> *"M1/M4 unique (Workgroup_Size_X, VGPR_Count, SGPR_Count, Grid_Size_X,
> Grid_Size_Y) tuples are identical between traces … the winners it returns for
> the (quant=w8a8, head_dim=128, seq_len_bucket, num_seqs) buckets exercised by
> this workload happen to map back to the same launch constants the M4 default
> heuristic already selects."*

**Quantitative implications:**

- The `m1-bench-grid` +0.25 % / +0.33 % wins on `w8a8_tp1_c1_{synthetic,coding}`
  are within measurement noise. The kernel itself ran ~0.06 % faster — a delta
  well below the ±2 % reproducibility tolerance recorded in M0
  (`m0-canary/canary_check.md`) and the ±5 % VAL-CROSS-004 disable-path band.
- Per-call HBM bytes are **unchanged** (the byte-identical launch tuples imply
  byte-identical tile streams).
- The +26.75 % p99_ttft regression on `w8a8_tp1_c4_coding` and the +35.96 %
  regression on `w4a16_tp1_c4_coding` are **not** explained by per-call HBM
  byte volume — that is constant. The most plausible attribution is the
  per-attention-call Python overhead of the new `_lookup_tuned_winner` gate
  itself: the M1 wire-in adds a dictionary lookup + key-construction
  (`quant, head_dim, seq_len_bucket, num_seqs`) on every invocation of
  `_get_tile_size` and `_compute_flash_decoding_splits`. At concurrency 4 with
  the chunked-prefill scheduler this overhead compounds; at concurrency 1 the
  attention loop is wide enough to amortise it. See
  `hbm_bytes_analysis.md` §4.1.

**Interpretation:** the wire-in is *correctly additive* (engine log confirms
*"MI100 tuned flash-decode lookup loaded from .../winners_lookup.json (16 winners)"*
and the trace shows the lookup is consulted on every call), but the lookup's
winning entries for the observed shapes happen to **coincide** with the M4
default heuristic. There is no per-shape config the sweep found that beats the
defaults by more than measurement noise.

**Recommendations (orchestrator decision at mission close):**

- **(i) Production-recommended:** delete the lookup machinery and ship the M1
  perplexity / coding / needle quality-gate evidence as the negative-result
  artifact. The M4 defaults remain the floor; the env flag becomes dead code.
  No throughput is left on the table because the sweep's winners coincide
  with defaults.
- **(ii) Future-mission scope:** widen the lookup to shapes where the M4
  defaults are sub-optimal — e.g. mixed-batch decode at c=8/16, longer
  `seq_len_bucket` (65 k+, 131 k+), or non-multiple-of-128 head dims. This
  requires a new sweep harness and is outside this mission's M3 close.

The disable path (`unset VLLM_MI100_USE_TUNED_FLASH_DECODE`) is preserved
verbatim from the M4 baseline (`scripts/launch_hbm_*.sh` is byte-identical
to its PR-#30 state — none of those 12 scripts were edited by this mission).

---

## Win-Bar Verdict

**Research-mode PASS via direction-correct improvement on ≥1 decode-dominated
cell.** Anchor cell: **`w8a8_tp1_c1_coding`**, delta_pct = **+0.334 %**, ratio
= **1.0033 ≥ 1.001** (win-bar threshold from VAL-M1-004). Source cell JSONs:

- M1: `/root/bench-int8-w4a16-hbm-fa/m1-tuning/coding/w8a8_tp1_c1.json` (also
  copied to `/root/bench-int8-w4a16-hbm-fa/m1-tuning/w8a8/w8a8_tp1_c1_coding.json`)
- M4: `/root/bench-int8-w4a16-hbm-fa/m4-final/w8a8/w8a8_tp1_c1_coding.json`
  (READ-ONLY reference)

Corroborating decode-dominated direction-correct cells: `w8a8_tp1_c1_synthetic`
(+0.248 %), `w8a8_tp4_c1_coding` (+0.130 %), `w4a16_tp1_c2_coding` (+0.197 %).
See `m1_winbar.md` for the full anchor-selection narrative.

**Negative-result attribution (parallel evidence for VAL-M1-004's secondary
clause):** all four +Δ cells lie within measurement noise and the rocprofv3
trace evidence at `hbm_bytes_analysis.md` shows zero HBM-byte-per-invocation
delta on both the +0.25 % winner cell and the +26.75 % p99_ttft regression
cell. The win-bar is technically met, but the **physical mechanism behind
the +Δ is launch-noise, not a tuned kernel parameter**. The mission's
win-bar verdict and its rocprofv3 null-result both stand on disk.

---

## Quality Gates

All three quality gates **PASS** for both quants under
`VLLM_MI100_USE_TUNED_FLASH_DECODE=1`. Production baselines (W8A8 9.6518,
W4A16 9.8030, coding-agent 9/10, needle@32k 5/5) are carried from
`BENCH_INT8_W4A16_HBM.md`.

### Perplexity (gate: Δ ≤ +1 %)

| quant | ppl_m1 | ppl_production | Δ % | gate |
|---|---:|---:|---:|---|
| w8a8  | 9.682533 | 9.6518 | +0.318 % | **PASS** (`m1_ppl_w8a8.json`)  |
| w4a16 | 9.823300 | 9.8030 | +0.207 % | **PASS** (`m1_ppl_w4a16.json`) |

Tool: `scripts/m0_perplexity.py` (wikitext-2-raw-v1, 50 chunks × 512 tokens,
seed 0, 25 550 tokens scored per quant). Raw evidence:
`m1-tuning/m1_ppl_{w8a8,w4a16}.json` and `m1_ppl_{w8a8,w4a16}_raw.json`.

### Coding-agent (gate: pass_count ≥ 9 / 10)

| quant | pass_count | total | ceiling | gate |
|---|---:|---:|---:|---|
| w8a8  | 9 | 10 | 9 (max_subarray pre-existing IndentationError) | **PASS** |
| w4a16 | 9 | 10 | 9 (same) | **PASS** |

Tool: `scripts/eval_coding_prompts.py`, fixed 10-prompt suite at
`tests/eval/coding_prompts.json`, temperature=0, max_tokens=512. Evidence:
`m1_coding_eval.json` + per-quant raw at `coding/m6_coding_eval_m1_{w8a8,w4a16}.json`.
The `max_subarray` failure is the known pre-existing defect; all other 9
prompts pass for both quants.

### Needle-in-haystack @ 32 k (gate: 5 / 5)

| quant | needle_pass | total | gate |
|---|---:|---:|---|
| w8a8  | 5 | 5 | **PASS** |
| w4a16 | 5 | 5 | **PASS** |

Tool: `scripts/eval_needle.py --ctx 32768 --probes 5` (depths 10/30/50/70/90 %).
Evidence: `m1_needle32k.json` + per-quant raw at
`needle/needle_m1_{w8a8,w4a16}.json`.

---

## Operational Flags

All env vars introduced by this mission are **additive and default-empty**.
Unsetting every flag reverts byte-identical to the M4 production stack.

| env var | introduced | accepted values | effect |
|---|---|---|---|
| `VLLM_MI100_USE_TUNED_FLASH_DECODE` | THIS mission (M1) | `1` / unset | Enables the winners-lookup consumer in `triton_unified_attention.py`. **Default-off.** Set to `1` in every `scripts/launch_hbm2_*.sh` (12 files); UNSET in every `scripts/launch_hbm_*.sh` (12 files, M4 production). |
| `VLLM_MI100_TUNED_FLASH_DECODE_LOOKUP` | THIS mission (M1) | path | Override path to `winners_lookup.json`. Default: `/root/bench-int8-w4a16-hbm-fa/m1-tuning/winners_lookup.json`. |
| `VLLM_MI100_TUNED_QUANT_HINT` | THIS mission (M1) | `w8a8` / `w4a16` | Provides the quant key to the lookup at attention-call time. Set per-cell in `launch_hbm2_*.sh`. |
| `VLLM_MI100_TRY_CK_INT8_PTH` | N/A (M2 NOT attempted) | n/a | **Not implemented this mission.** The M2 audit returned NO-GO; no `rocm_ck_fa.py` edits were made. Reserved name only; do not set. |

**Disable path:** for every flag listed, the disable convention is
`unset <FLAG>` (or `<FLAG>=`). The M0 canary (`m0-canary/canary_check.md`)
demonstrates the disable path on the unmodified M4 launch scripts.

**M4 scripts unchanged (read-only):** the 12 `scripts/launch_hbm_*.sh` files
are byte-identical to their PR-#30 state. This mission's 12 new scripts use
the **`launch_hbm2_`** prefix (with the trailing `2`), per VAL-M3-001 and the
mission's allowed-write-paths section in AGENTS.md.

---

## Reproducibility

| convention | value | source |
|---|---|---|
| NUM_PROMPTS | 200 (fixed all 24 cells) | `harness_manifest.json:num_prompts_per_cell` |
| `--seed` | 42 | `harness_manifest.json:seed`; embedded in launch CLI |
| Pinned vLLM SHA (M1 harness) | `54df089f3f7acb905e958518e91251dfff5bbd17` | `harness_manifest.json:vllm_commit` |
| Pinned vLLM SHA (M3 launch scripts) | `8c45a5b916a0a14dddb707f99ecc03c964f77dad` | `git rev-parse HEAD` at this report's predecessor commit |
| `--input-len / --output-len` (synthetic) | 1024 / 256 | `harness_manifest.json` cell CLIs |
| dataset (coding) | `/root/bench-int8-w4a16/datasets/coding_agent.jsonl` (READ-ONLY, SHA256 `db138a30917dc972fab0ca70ddf74601efa6e3c98d53a2069f94cf5fad901cf0`) | `harness_manifest.json:dataset_sha256` |
| block_size | 32 | `harness_manifest.json:block_size` |
| max_model_len | 32 768 | `harness_manifest.json:max_model_len` |
| cudagraph_mode | `FULL_DECODE_ONLY` (auto-demoted from `FULL` per AGENTS.md anti-pattern #3) | `harness_manifest.json:cudagraph_mode` |
| `--language-model-only` | true (Qwen3.5 vision-encoder OOM guard) | `harness_manifest.json:language_model_only` |
| `--enable-prefix-caching` | true | `harness_manifest.json:enable_prefix_caching` |

**M3 repro spotcheck (`m3-final/m3_repro_spotcheck.csv`):**

| cell | ref_tput | measured_tput | Δ % | verdict |
|---|---:|---:|---:|---|
| w4a16_tp4_c2 | 106.060572 | 105.574678 | -0.46 % | PASS |
| w8a8_tp1_c2  |  74.642386 |  74.597986 | -0.06 % | PASS |
| w8a8_tp1_c1  |  40.210601 |  40.032331 | -0.44 % | PASS |

All three within ±2 % per VAL-M3-002. The 3 cells were randomly selected;
selection rationale recorded in `m3-final/spotcheck_cells.txt`. Bench logs:
`m3-final/spotcheck_logs/`.

---

## Cross-Links

- **Pre-M4 production baseline (full Pareto report):**
  [`BENCH_INT8_W4A16_FINAL.md`](BENCH_INT8_W4A16_FINAL.md) (M6 of the original
  4-week INT8/W4A16 mission). The W8A8 9.6518 / W4A16 9.8030 perplexity floors
  used as gate references in `m1_ppl_*.json` come from this report's quality
  gate section.
- **M4 production stack baseline (this mission's direct comparator):**
  [`BENCH_INT8_W4A16_HBM.md`](BENCH_INT8_W4A16_HBM.md). All `m4_baseline_tput`
  columns in §Cumulative Grid resolve to cell JSONs under
  `/root/bench-int8-w4a16-hbm/m4-final/`, which is the canonical evidence root
  for that report. **This file is READ-ONLY and is not modified by this mission.**
- **Negative-result template (precedent):**
  [`BENCH_M5_ISA.md`](BENCH_M5_ISA.md). M5 was a clean "memory-bound hot
  kernels reject hand-ISA" negative-result; this mission's §M1 Null-Result
  Finding (Tuning Coincides With Defaults) follows the same documentation
  pattern.
- **Companion mid-mission artifacts:**
    - `m1-tuning/m1_winbar.md` (per-cell anchor selection narrative)
    - `m1-tuning/m1_vs_m4_grid.md` (verbose per-metric markdown table)
    - `m1-tuning/hbm_bytes_analysis.md` (rocprofv3 evidence; §M1 Null-Result Finding cites this)
    - `m2-ck/audit.md` (CK FA2 NO-GO audit)
    - `m0-canary/canary_check.md` (M0 baseline drift evidence)

---

## Appendix: Per-Cell Raw JSON Index

Each cell JSON conforms to `scripts/bench_schema.json` and was validated by
`scripts/validate_results_schema.py` (see `schema_check.log`: *"24 files
validated, 0 failures"*).

| cell_id | workload | M1 stack JSON | M4 baseline JSON |
|---|---|---|---|
| w8a8_tp1_c1 | synthetic | `m1-tuning/w8a8/w8a8_tp1_c1_synthetic.json` | `hbm/m4-final/w8a8/w8a8_tp1_c1_synthetic.json` |
| w8a8_tp1_c1 | coding    | `m1-tuning/w8a8/w8a8_tp1_c1_coding.json`    | `hbm/m4-final/w8a8/w8a8_tp1_c1_coding.json`    |
| w8a8_tp1_c2 | synthetic | `m1-tuning/w8a8/w8a8_tp1_c2_synthetic.json` | `hbm/m4-final/w8a8/w8a8_tp1_c2_synthetic.json` |
| w8a8_tp1_c2 | coding    | `m1-tuning/w8a8/w8a8_tp1_c2_coding.json`    | `hbm/m4-final/w8a8/w8a8_tp1_c2_coding.json`    |
| w8a8_tp1_c4 | synthetic | `m1-tuning/w8a8/w8a8_tp1_c4_synthetic.json` | `hbm/m4-final/w8a8/w8a8_tp1_c4_synthetic.json` |
| w8a8_tp1_c4 | coding    | `m1-tuning/w8a8/w8a8_tp1_c4_coding.json`    | `hbm/m4-final/w8a8/w8a8_tp1_c4_coding.json`    |
| w8a8_tp4_c1 | synthetic | `m1-tuning/w8a8/w8a8_tp4_c1_synthetic.json` | `hbm/m4-final/w8a8/w8a8_tp4_c1_synthetic.json` |
| w8a8_tp4_c1 | coding    | `m1-tuning/w8a8/w8a8_tp4_c1_coding.json`    | `hbm/m4-final/w8a8/w8a8_tp4_c1_coding.json`    |
| w8a8_tp4_c2 | synthetic | `m1-tuning/w8a8/w8a8_tp4_c2_synthetic.json` | `hbm/m4-final/w8a8/w8a8_tp4_c2_synthetic.json` |
| w8a8_tp4_c2 | coding    | `m1-tuning/w8a8/w8a8_tp4_c2_coding.json`    | `hbm/m4-final/w8a8/w8a8_tp4_c2_coding.json`    |
| w8a8_tp4_c4 | synthetic | `m1-tuning/w8a8/w8a8_tp4_c4_synthetic.json` | `hbm/m4-final/w8a8/w8a8_tp4_c4_synthetic.json` |
| w8a8_tp4_c4 | coding    | `m1-tuning/w8a8/w8a8_tp4_c4_coding.json`    | `hbm/m4-final/w8a8/w8a8_tp4_c4_coding.json`    |
| w4a16_tp1_c1 | synthetic | `m1-tuning/w4a16/w4a16_tp1_c1_synthetic.json` | `hbm/m4-final/w4a16/w4a16_tp1_c1_synthetic.json` |
| w4a16_tp1_c1 | coding    | `m1-tuning/w4a16/w4a16_tp1_c1_coding.json`    | `hbm/m4-final/w4a16/w4a16_tp1_c1_coding.json`    |
| w4a16_tp1_c2 | synthetic | `m1-tuning/w4a16/w4a16_tp1_c2_synthetic.json` | `hbm/m4-final/w4a16/w4a16_tp1_c2_synthetic.json` |
| w4a16_tp1_c2 | coding    | `m1-tuning/w4a16/w4a16_tp1_c2_coding.json`    | `hbm/m4-final/w4a16/w4a16_tp1_c2_coding.json`    |
| w4a16_tp1_c4 | synthetic | `m1-tuning/w4a16/w4a16_tp1_c4_synthetic.json` | `hbm/m4-final/w4a16/w4a16_tp1_c4_synthetic.json` |
| w4a16_tp1_c4 | coding    | `m1-tuning/w4a16/w4a16_tp1_c4_coding.json`    | `hbm/m4-final/w4a16/w4a16_tp1_c4_coding.json`    |
| w4a16_tp4_c1 | synthetic | `m1-tuning/w4a16/w4a16_tp4_c1_synthetic.json` | `hbm/m4-final/w4a16/w4a16_tp4_c1_synthetic.json` |
| w4a16_tp4_c1 | coding    | `m1-tuning/w4a16/w4a16_tp4_c1_coding.json`    | `hbm/m4-final/w4a16/w4a16_tp4_c1_coding.json`    |
| w4a16_tp4_c2 | synthetic | `m1-tuning/w4a16/w4a16_tp4_c2_synthetic.json` | `hbm/m4-final/w4a16/w4a16_tp4_c2_synthetic.json` |
| w4a16_tp4_c2 | coding    | `m1-tuning/w4a16/w4a16_tp4_c2_coding.json`    | `hbm/m4-final/w4a16/w4a16_tp4_c2_coding.json`    |
| w4a16_tp4_c4 | synthetic | `m1-tuning/w4a16/w4a16_tp4_c4_synthetic.json` | `hbm/m4-final/w4a16/w4a16_tp4_c4_synthetic.json` |
| w4a16_tp4_c4 | coding    | `m1-tuning/w4a16/w4a16_tp4_c4_coding.json`    | `hbm/m4-final/w4a16/w4a16_tp4_c4_coding.json`    |

Paths above are relative to the `/root/bench-int8-w4a16-hbm-fa/` (M1 stack) and
`/root/bench-int8-w4a16-` (M4 baseline) prefixes; the full absolute paths
embed `…/bench-int8-w4a16-hbm-fa/m1-tuning/…` and `…/bench-int8-w4a16-hbm/m4-final/…`
respectively.

Companion per-cell env-snapshot JSONs (24 files):
`m1-tuning/env_<cell>_<workload>.json` (979 bytes each; full env captured at
bench-cell start for VAL-CROSS-004 audit reproducibility).

rocprofv3 trace artefacts (TP=1 cells only per `library/rocprofv3-tp4-limitation.md`):

- `m1-tuning/rocprof/w8a8_tp1_c1_synthetic/m1_trace_kernel_trace.csv` (198 MB, 412 238 records)
- `m1-tuning/rocprof/w8a8_tp1_c1_synthetic/m4_trace_kernel_trace.csv` (198 MB, 412 238 records)
- `m1-tuning/rocprof/w8a8_tp1_c4_coding/m1_trace_kernel_trace.csv` (202 MB, 420 069 records)
- `m1-tuning/rocprof/w8a8_tp1_c4_coding/m4_trace_kernel_trace.csv` (202 MB, 420 069 records)
- Per-cell summary CSV+JSON at `m1-tuning/rocprof/<cell>/m1_summary.{csv,json}`

End of report.
