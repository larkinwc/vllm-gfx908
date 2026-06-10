# MI100 Benchmark Results

Benchmark results for vLLM on 4× AMD Instinct MI100 (gfx908) GPUs.

**Hardware:** 4× MI100 32GB HBM2, XGMI full mesh, AMD EPYC 7742 64C
**Software (current):** vLLM `0.20.2rc1.dev107+gd960f21e4` (worktree head `71fb9375c`), ROCm 7.12, PyTorch 2.11.0+rocm7.2, pytorch-triton-rocm 3.5.1, rccl 2.27.7
**Standard bench config:** `--block-size 32`, `--enable-prefix-caching`, `--language-model-only`, `--gpu-memory-utilization 0.93`, `--max-model-len 32768`, FULL_DECODE_ONLY graphs, 200 prompts/cell, seed 42

> The current production stack for the quantized Qwen3.5-9B models is the
> **M4 HBM final** (W8A8/W4A16 custom kernels + KV-INT8 + chunked prefill +
> per-cell NCCL_ALGO). [§Latest production figures](#latest-production-figures)
> below carries the headline 24-cell grids; older FP16-era results and
> superseded findings are retained for context in
> [§Historical results](#historical-results-fp16-era-and-superseded-findings).

---

## Table of Contents

1. [Latest production figures](#latest-production-figures)
   - [Quality gates (PPL / coding / needle@32k)](#quality-gates-ppl--coding--needle32k)
   - [W8A8 INT8 — 24-cell production grid](#w8a8-int8--24-cell-production-grid)
   - [W4A16 INT4 — 24-cell production grid](#w4a16-int4--24-cell-production-grid)
2. [Current optimization stack](#current-optimization-stack)
   - [Currently-shipping levers](#currently-shipping-levers)
   - [Tested / negative-result / blocked](#tested--negative-result--blocked)
3. [How to run](#how-to-run)
4. [Cross-references (mission artifacts)](#cross-references-mission-artifacts)
5. [Historical results (FP16-era and superseded findings)](#historical-results-fp16-era-and-superseded-findings)
6. [Upstream rebase notes](#upstream-rebase-notes)

---

## Latest production figures

All numbers in §Latest are from the **M4 HBM final** stack:
`Qwen3.5-9B-{w8a8,w4a16}` + KV-INT8 (`int8_per_token_head`) + chunked prefill
(`--max-num-batched-tokens=2048` for W8A8 / `4096` for W4A16) + per-cell
`NCCL_ALGO` winner, baked into `scripts/launch_hbm_<cell>.sh`. 24 cells = 12
(`{w8a8,w4a16}` × `{tp1,tp4}` × `c∈{1,2,4}`) × 2 workloads (`synthetic` =
random 1024/256, `coding` = realistic coding-agent JSONL). Full evidence:
[`BENCH_INT8_W4A16_HBM.md`](docs/experiments/BENCH_INT8_W4A16_HBM.md).

### Quality gates (PPL / coding / needle@32k)

Wikitext-2 perplexity is 50 chunks × 512 tokens, seed 0. Coding-agent is the
fixed 10-prompt suite at `tests/eval/coding_prompts.json` (temperature 0,
max_tokens 512). Needle@32k probes depths 10/30/50/70/90 %.

| Model variant | Path | PPL | Δ vs FP16 | Coding | Needle@32k |
|---|---|---:|---:|:-:|:-:|
| **FP16 reference** | `/models/Qwen3.5-9B` | **9.5254** | — | n/a | n/a |
| **W8A8 INT8** (base artifact) | `/models/Qwen3.5-9B-w8a8` (RedHatAI) | **9.6518** | **+1.33 %** | 9/10 | 5/5 |
| **W8A8 INT8 + M4 HBM stack** | `+KV-INT8 +chunked +NCCL` | **9.6825** | **+1.65 %** | 9/10 | 5/5 |
| **W4A16 INT4 g=32** (base artifact) | `/models/Qwen3.5-9B-w4a16` (apolo13x) | **9.8030** | **+2.91 %** | 9/10 | 5/5 |
| **W4A16 INT4 + M4 HBM stack** | `+KV-INT8 +chunked +NCCL` | **9.8233** | **+3.13 %** | 9/10 | 5/5 |

The single coding-rubric failure (`max_subarray` IndentationError) is
shared with the FP16 baseline and pre-dates quantisation. Gates VAL-FINAL-002
(Δ ≤ +3 % PPL vs FP16 + Δ ≤ +0.5 % vs M0 PTQ), VAL-FINAL-003 (≥ 9/10
coding), VAL-FINAL-004 (5/5 needle@32k) all PASS on both quants.

### W8A8 INT8 — 24-cell production grid

`output_throughput_toks_s` and `mean_ttft_ms` per cell × workload, M4 HBM
stack. "Δ vs prior" compares M4 against the PR #29 baseline winner from
`/root/bench-int8-w4a16/final/final_grid.csv` (best of stock/TensileLite/
Triton/CK paths for that cell). Source: §5 of `BENCH_INT8_W4A16_HBM.md`.

| Cell | Workload | Tput (tok/s) | Δ tput | mean TTFT (ms) | Δ TTFT | Notes |
|---|---|---:|---:|---:|---:|---|
| `w8a8_tp1_c1` | synthetic | **40.11** | +2.94 % | 250.17 | **−11.19 %** | KV-INT8 + chunked@2048 |
| `w8a8_tp1_c1` | coding    | **39.95** | +4.97 % | 249.53 | +16.25 %    | KV-INT8 + chunked@2048 |
| `w8a8_tp1_c2` | synthetic | **74.61** | +1.08 % | 362.43 | **−17.22 %** | KV-INT8 + chunked@2048 |
| `w8a8_tp1_c2` | coding    | **73.44** | +7.04 % | 278.30 | +8.90 %    | KV-INT8 + chunked@2048 |
| `w8a8_tp1_c4` | synthetic | **136.85**| +0.70 % | 646.16 | **−9.95 %**  | KV-INT8 + chunked@2048 |
| `w8a8_tp1_c4` | coding    | **131.21**| +9.02 % | 298.23 | +2.16 %    | KV-INT8 + chunked@2048 |
| `w8a8_tp4_c1` | synthetic | **63.57** | −0.60 % | 122.71 | **−25.68 %** | + NCCL default |
| `w8a8_tp4_c1` | coding    | **62.94** | +3.22 % | 130.28 | −0.86 %    | + NCCL default |
| `w8a8_tp4_c2` | synthetic |   122.75  | **−2.03 %** | 180.77 | +32.94 % | + NCCL default; HBM-bound regression |
| `w8a8_tp4_c2` | coding    | **120.42**| +9.39 % | 152.12 | +30.97 % | + NCCL default |
| `w8a8_tp4_c4` | synthetic |   229.13  | **−7.51 %** | 248.87 | +28.60 % | + NCCL=Ring; coding wins, synthetic regresses |
| `w8a8_tp4_c4` | coding    | **222.69**| +13.45 %| 155.70 | +27.98 % | + NCCL=Ring |

**Decode-dominated geomean (TP=1/4 × c=1/2 synthetic):** 69.5150 tok/s, ratio
1.0033× vs prior production (1.5× DoD bar NOT MET — documented as
VAL-FINAL-005 negative result, see §10 of `BENCH_INT8_W4A16_HBM.md`).
**Peak win:** `w8a8_tp4_c4_coding +13.45 %`. **Peak regression:**
`w8a8_tp4_c4_synthetic −7.51 %` (HBM-bound; chunked-prefill scheduling
overhead exceeds KV-INT8 savings at TP=4 c=4 synthetic).

### W4A16 INT4 — 24-cell production grid

| Cell | Workload | Tput (tok/s) | Δ tput | mean TTFT (ms) | Δ TTFT | Notes |
|---|---|---:|---:|---:|---:|---|
| `w4a16_tp1_c1` | synthetic | **30.88** | +4.46 % | 336.27 | −9.02 %     | KV-INT8 + chunked@4096 |
| `w4a16_tp1_c1` | coding    | **30.84** | +8.12 % | 340.40 | +27.30 %    | KV-INT8 + chunked@4096 |
| `w4a16_tp1_c2` | synthetic | **56.76** | +0.22 % | 496.36 | +76.31 %    | KV-INT8 + chunked@4096 |
| `w4a16_tp1_c2` | coding    | **55.78** | +5.45 % | 383.25 | +87.76 %    | KV-INT8 + chunked@4096 |
| `w4a16_tp1_c4` | synthetic | 104.01    | −0.27 % | 982.17 | +99.77 %    | KV-INT8 + chunked@4096 |
| `w4a16_tp1_c4` | coding    | **99.27** | +6.11 % | 414.06 | +45.99 %    | KV-INT8 + chunked@4096 |
| `w4a16_tp4_c1` | synthetic | **55.25** | +8.67 % | 142.36 | **−14.30 %** | + NCCL=Ring |
| `w4a16_tp4_c1` | coding    | **54.87** | +12.50 %| 146.89 | +6.74 %     | + NCCL=Ring |
| `w4a16_tp4_c2` | synthetic | **106.11**| +6.80 % | 208.02 | +47.14 %    | + NCCL default |
| `w4a16_tp4_c2` | coding    | **104.55**| +17.18 %| 175.41 | +44.70 %    | + NCCL default |
| `w4a16_tp4_c4` | synthetic | **199.20**| +2.68 % | 367.85 | +83.82 %    | + NCCL=Ring |
| `w4a16_tp4_c4` | coding    | **193.74**| +19.61 %| 182.20 | +38.13 %    | + NCCL=Ring (mission's largest single-cell win) |

**Decode-dominated geomean (TP=1/4 × c=1/2 synthetic):** 56.6173 tok/s, ratio
1.0499× vs prior production. **Peak win:** `w4a16_tp4_c4_coding +19.61 %`.
**Peak regression:** `w4a16_tp1_c4_synthetic −0.27 %` (within noise).
**TTFT note:** chunked prefill trades first-token latency for steady-state
throughput; mean_ttft regresses on most cells. Wins are concentrated on
coding workload and TP=4 high-concurrency cells.

---

## Current optimization stack

### Currently-shipping levers

These are baked into the production launch scripts (`scripts/launch_hbm_*.sh`,
`scripts/launch_*.sh` for FP16 paths). All ship enabled-by-default unless
noted; each has an env-var disable path.

| Lever | Where | Production effect | Evidence |
|---|---|---|---|
| **FULL_DECODE_ONLY HIP graphs** | global | +83 % tput vs eager (Llama-2-7B); enables decode pipeline | Auto-detected on gfx908; forces `--disable-custom-all-reduce` |
| **Prefix caching** | `--enable-prefix-caching` | 85–99 % TTFT reduction on cache hits | Stacks with graphs |
| **Block-size 32** | `--block-size 32` | −44 % TTFT, +9.6 % tput at c=4 (FP16); included in all M4 launches | FP16-era result, still adopted |
| **Triton MI100 tile tuning** | `triton_unified_attention.py`, `triton_prefill_attention.py`, `triton_attn.py` | Decode TILE 16→32, prefill BLOCK 128→64, NUM_PAR_SOFTMAX_SEGMENTS 16→8 | FP16 wins above; carries through to quant stacks |
| **Adaptive Flash-Decoding (Split-K)** | `triton_attn.py` | +35 % c=1 tput on FP16; default heuristic in M4 quant stack | Sweep confirmed defaults already optimal — see HBM-FA |
| **Skinny GEMM (gfx908)** | `VLLM_ROCM_USE_SKINNY_GEMM=1`, `__gfx908__` guard in `skinny_gemms.cu` | −27 % TPOT, +28 % c=1 tput on FP16 (off by default for quant stack) | `BENCH.md` §Historical |
| **Custom Triton W8A8 (`mi100_int8`)** | `vllm/model_executor/kernels/scaled_mm/mi100_int8.py` | INT8×INT8→INT32 MFMA, AOT-tuned configs at `vllm/model_executor/kernels/configs/gfx908/mi100_int8_*` | [`BENCH_INT8_W4A16_M3.md`](docs/experiments/BENCH_INT8_W4A16_M3.md) |
| **Custom Triton W4A16 (`mi100_w4a16`)** | `vllm/model_executor/kernels/linear/mi100_w4a16.py` | group_size=32 INT4 dequant + MFMA; AOT configs at `mi100_w4a16_*` | [`BENCH_INT8_W4A16_M3.md`](docs/experiments/BENCH_INT8_W4A16_M3.md) |
| **hipBLASLt TensileLite library** | `HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library` | Tuned per-shape rocBLAS solutions for W8A8 hot shapes | [`BENCH_INT8_W4A16_M2.md`](docs/experiments/BENCH_INT8_W4A16_M2.md) |
| **Composable Kernel INT8 GEMM** | `csrc/quantization/w8a8/int8/ck/instances/` (4 shapes) | `DeviceGemm_Xdl_CShuffle`, tile `MPerBlock=128 NPerBlock=128 KPerBlock=64 MPerXdl=16 NPerXdl=16` driving `v_mfma_i32_16x16x16i8`; dispatch priority CK > hipBLASLt > Triton | [`BENCH_M4_CK.md`](docs/experiments/BENCH_M4_CK.md) |
| **KV-INT8** | `--kv-cache-dtype int8_per_token_head` | W4A16 decode +8.19 % geomean, W8A8 +3.54 % geomean; halves attention KV HBM traffic | [`BENCH_HBM_M1_KVINT8.md`](docs/experiments/BENCH_HBM_M1_KVINT8.md) |
| **Chunked prefill** | `--enable-chunked-prefill --max-num-batched-tokens=2048 (w8a8) / 4096 (w4a16)` | 11/12 coding-workload cells clear `req_tput ≥ 1.03×` OR `mean_ttft ≤ 0.97×`; peak +17–20 % tput | [`BENCH_HBM_M2_CHUNKED.md`](docs/experiments/BENCH_HBM_M2_CHUNKED.md) |
| **Per-cell NCCL_ALGO** | `NCCL_ALGO=Ring` or empty (rccl heuristic) per TP=4 cell | < 1 % per-cell — no-regression alignment; baked into launch scripts | [`BENCH_HBM_M3_TP.md`](docs/experiments/BENCH_HBM_M3_TP.md) |
| **TunableOp GEMM cache** (FP16) | `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0` + cached CSVs | +13.4 % c=1 tput, −88 % TTFT on FP16 (still recommended for FP16 launches; superseded by hipBLASLt+CK on quant) | `BENCH.md` §Historical |
| **`--disable-custom-all-reduce` on gfx908** | auto-detected by `vllm/platforms/rocm.py` | Required for correct HIP graph capture on gfx908 (custom all-reduce IPC buffers go stale on replay) | `BENCH.md` §Historical |

### Tested / negative-result / blocked

Levers that were investigated and rejected, or are blocked by missing
hardware/software support. All disclosed for completeness so the live
config above is unambiguous.

| Lever | Status | Reason / impact | Evidence |
|---|---|---|---|
| **Triton Flash-Decoding tuning sweep** | **Null result** (M1 HBM-FA) | 9216-config sweep winners coincide **byte-identically** with M4 default heuristic; rocprofv3 shows zero HBM-bytes/inv delta. Flag `VLLM_MI100_USE_TUNED_FLASH_DECODE` ships **default-off** | [`BENCH_HBM_FA_TUNING.md`](docs/experiments/BENCH_HBM_FA_TUNING.md) |
| **CK FA2 INT8-PTH attention** | **NO-GO** (audit) | No `FmhaFwdI8` template in compiled `flash_attn_2_cuda`; no per-token-per-head scale plumbing; storage-layout mismatch with vLLM's inline-padded `head_size + sizeof(f32)` INT8-PTH layout. 3–6 weeks kernel-authoring scope | [`BENCH_HBM_FA_TUNING.md`](docs/experiments/BENCH_HBM_FA_TUNING.md) §M2 |
| **Hand-ISA path (gfx908)** | **Declined** (M5 negative result) | Hot W8A8/W4A16 GEMMs HBM-bound (~21 % HBM / <1 % VALU for W8A8, ~32 % HBM for W4A16); hand-ISA compute levers cannot lift HBM ceiling | [`BENCH_M5_ISA.md`](docs/experiments/BENCH_M5_ISA.md) |
| **W4A16 Composable Kernel** | **Declined** | ROCm 7.12 ships no gfx908-validated packed-INT4 + groupwise-scale-and-zero device template matching vLLM's layout. Op bound for API stability, `ck_w4a16_gemm_supports()` returns `False`; transparent Triton fallback | [`BENCH_M4_CK.md`](docs/experiments/BENCH_M4_CK.md) |
| **AMD AITER (`VLLM_ROCM_USE_AITER=1`)** | **Declined** (negative result) | Build wall (#74) fixed — vendored CK + 3 ISA-fallback patches build 43/50 modules — but AITER is **−6 % to −26 %** vs native/Triton (fp16 batch 1–64; w8a8 norm-only). The int8 `module_gemm_a8w8` and MoE `module_moe_ck2stages` CK instances never finish compiling on CDNA1 → engine **hangs at startup**. Ships **default-off**; build fix kept in `library/aiter-gfx908/` for future toolchains | [`AITER_GFX908_EMPIRICAL.md`](docs/experiments/AITER_GFX908_EMPIRICAL.md) |
| **`NCCL_ALGO=Tree` on KV-INT8** | **Blocked** (rccl 2.27.7) | KV-INT8 issues AllGather on `ncclInt8`; rccl 2.27.7 has no Tree algo for that dtype combo. Engine init aborts with `ncclInvalidUsage` | [`TP_TOPOLOGY.md`](docs/experiments/TP_TOPOLOGY.md) |
| **Chunked prefill < 2048 tokens** | **Infeasible** | vLLM's `attention block_size ≤ max_num_batched_tokens` guard rejects sizes below Qwen3.5's Mamba-aligned attention block size | [`BENCH_HBM_M2_CHUNKED.md`](docs/experiments/BENCH_HBM_M2_CHUNKED.md) §M2 |
| **`ROCM_ATTN` prefill-decode split** | Rejected | −5 % tput regression across all concurrency vs Triton unified | FP16-era result |
| **`--max-num-seqs 8` scheduler tuning** | Rejected | Neutral on coding; −33 % tput on bursty synthetic c=4 | FP16-era result |
| **TurboQuant KV compression** | Rejected | 6–11 % overhead synthetic, 42–49 % coding on Qwen3.5-9B (only 8/32 layers full-attention) | FP16-era result |
| **MTP speculative decoding** | Partial | Incompatible with graph mode; 25–45 % slower in eager. Not recommended | FP16-era result |
| **DFlash speculative decoding** | Rejected | Functionally correct on gfx908 (non-causal attn + CUDA graphs validated, ~4.8 acceptance length), but **net-negative**: 0.37–0.50× baseline tput across synthetic + coding c=1/2/4. MI100's compute-bound FP16 decode can't absorb the wider batched verify. | [`BENCH_DFLASH_GFX908.md`](docs/experiments/BENCH_DFLASH_GFX908.md) |
| **W8A8 GPTQ (off-the-shelf symmetric)** | Superseded | Original symmetric per-channel GPTQ failed on GatedDeltaNet `in_proj_qkv` (gibberish output despite PPL 10.02). **Resolved** by RedHatAI's `Qwen3.5-9B-w8a8` calibration artifact (PPL 9.6518, coherent generation) | `BENCH.md` §Historical; [`BENCH_INT8_W4A16_FINAL.md`](docs/experiments/BENCH_INT8_W4A16_FINAL.md) |
| **AITER unified attention** | Blocked | `aiter` package not available for gfx908 | FP16-era finding |
| **FP8 native** | Blocked | MI100 lacks FP8 hardware. Software dequant path emulated only | FP16-era finding |

---

## How to run

```bash
# === Quantized (recommended for Qwen3.5-9B) ===
# M4 HBM final stack — per-cell launch scripts in scripts/launch_hbm_<cell>.sh
scripts/launch_hbm_w8a8_tp4_c4.sh        # 4-GPU TP, c=4 W8A8 + KV-INT8 + chunked + NCCL=Ring
scripts/launch_hbm_w4a16_tp4_c4.sh       # 4-GPU TP, c=4 W4A16 + KV-INT8 + chunked + NCCL=Ring
# (All 12 cells: w{8a8,4a16}_tp{1,4}_c{1,2,4})

# Research-only: Flash-Decoding tuned-lookup (default-off; M4 defaults already optimal)
scripts/launch_hbm2_w8a8_tp4_c4.sh       # sets VLLM_MI100_USE_TUNED_FLASH_DECODE=1

# === FP16 ===
/root/launch-vllm-optimized.sh           # FP16 production config

# === Benchmarks ===
.venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --model /models/Qwen3.5-9B-w8a8 --num-prompts 200 --seed 42 \
  --dataset-name random --save-result

.venv/bin/python /root/benchmark-scripts/coding_agent_bench.py \
  --concurrency 4 --requests 20 --model /models/Qwen3.5-9B-w8a8

# === Quality gates ===
.venv/bin/python scripts/m0_perplexity.py --model /models/Qwen3.5-9B-w8a8 \
  --dataset wikitext-2-raw-v1 --chunks 50 --chunk-tokens 512 --seed 0
.venv/bin/python scripts/eval_coding_prompts.py
.venv/bin/python scripts/eval_needle.py --ctx 32768 --probes 5

# === Disable / escape-hatch env vars ===
unset VLLM_MI100_USE_TUNED_FLASH_DECODE  # use default heuristic (recommended)
VLLM_DISABLE_CK=1                        # force hipBLASLt + Triton path
VLLM_DISABLE_HIPBLASLT=1                 # force Triton-only quant path
VLLM_DISABLE_MI100_W4A16=1               # disable custom W4A16 kernel
unset KV_CACHE_DTYPE                     # disable KV-INT8 (revert to FP16 KV)
unset ENABLE_CHUNKED_PREFILL             # disable chunked prefill
unset NCCL_ALGO                          # use rccl per-message-size heuristic
```

---

## Cross-references (mission artifacts)

These files contain frozen evidence (per-cell raw JSON paths, `file:line`
citations, tuning-hash manifests, rocprofv3 traces). **Do not edit them
retroactively** — they are anchored to specific vLLM SHAs.

- **W8A8 / W4A16 custom kernels (PR #29 mission):**
    - [`BENCH_INT8_W4A16_FINAL.md`](docs/experiments/BENCH_INT8_W4A16_FINAL.md) — final Pareto aggregate (M0–M5)
    - [`BENCH_INT8_W4A16_BASELINE.md`](docs/experiments/BENCH_INT8_W4A16_BASELINE.md) — M0 stock + M1 baseline grid
    - [`BENCH_INT8_W4A16_M2.md`](docs/experiments/BENCH_INT8_W4A16_M2.md) — hipBLASLt + TensileLite three-way
    - [`BENCH_INT8_W4A16_M3.md`](docs/experiments/BENCH_INT8_W4A16_M3.md) — custom Triton W8A8 + W4A16 kernels
    - [`BENCH_M4_CK.md`](docs/experiments/BENCH_M4_CK.md) — Composable Kernel W8A8 (+ W4A16 declined)
    - [`BENCH_M5_ISA.md`](docs/experiments/BENCH_M5_ISA.md) — hand-ISA path declined (negative result)
- **HBM Optimization Config-Sweep mission (Epic #22):**
    - [`BENCH_INT8_W4A16_HBM.md`](docs/experiments/BENCH_INT8_W4A16_HBM.md) — M4 final aggregate (24-cell grid above is from §5 here)
    - [`BENCH_HBM_M1_KVINT8.md`](docs/experiments/BENCH_HBM_M1_KVINT8.md) — KV-INT8 sub-mission
    - [`BENCH_HBM_M2_CHUNKED.md`](docs/experiments/BENCH_HBM_M2_CHUNKED.md) — chunked-prefill + chunk-size sweep
    - [`BENCH_HBM_M3_TP.md`](docs/experiments/BENCH_HBM_M3_TP.md) — TP topology / NCCL_ALGO sweep
- **HBM Flash-Decoding tuning + CK FA2 investigation:**
    - [`BENCH_HBM_FA_TUNING.md`](docs/experiments/BENCH_HBM_FA_TUNING.md) — 9216-config sweep null result + CK FA2 NO-GO audit
- **Speculative decoding on gfx908:**
    - [`BENCH_DFLASH_GFX908.md`](docs/experiments/BENCH_DFLASH_GFX908.md) — DFlash end-to-end enablement + A/B (correct but net-negative on MI100)

---

## Historical results (FP16-era and superseded findings)

> The numbers in this section were measured on older vLLM builds
> (`0.18.1.dev4` and earlier) against `/models/Qwen3.5-9B` FP16, before
> the W8A8/W4A16 custom-kernel mission. They are retained because the FP16
> launch path (`/root/launch-vllm-optimized.sh`) still produces these
> figures, and several of the levers documented here (Triton tile tuning,
> Adaptive Flash-Decoding, block-size 32, custom all-reduce + CUDA graph
> fix) are still part of the current production stack. **Throughput values
> here are not directly comparable to the §Latest grids above** because the
> binary, model, and bench harness all changed.

### Qwen3.5-9B FP16 — synthetic benchmarks (`vllm bench serve`, 100 prompts, random)

| Configuration | c=1 tok/s | c=2 tok/s | c=4 tok/s | TPOT c=1 | TPOT c=2 | TTFT c=1 |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (enforce-eager) | 228 | 412 | 822 | 50.2 ms | 49.6 ms | 880 ms |
| FULL_DECODE_ONLY + prefix cache | 248 | 478 | 884 | 13.6 ms | 15.8 ms | 715 ms |
| + custom all-reduce | 250 | 486 | 910 | 11.3 ms | 12.1 ms | 119 ms |
| + Triton MI100 tuning | 250 | 485 | 909 | 10.5 ms | 11.9 ms | 120 ms |
| + block-size 32 | 250 | 486 | 914 | 10.3 ms | 11.6 ms | 96 ms |
| **+ skinny GEMM (gfx908)** | **251** | **488** | **916** | **8.8 ms** | **11.1 ms** | **93 ms** |
| TQ capture_only + graphs | 239 | 447 | — | 30.6 ms | 52.1 ms | 4107 ms |
| TQ hybrid + graphs | 233 | 448 | — | 31.2 ms | 33.0 ms | 854 ms |
| ROCM_ATTN (prefill-decode split) | 248 | 480 | 891 | 13.0 ms | 14.7 ms | 128 ms |

### Qwen3.5-9B FP16 — coding agent (realistic prompts, 256 tok max output)

| Configuration | c=1 tok/s | c=2 tok/s | c=4 tok/s | TPOT c=1 | TTFT c=1 |
|---|---:|---:|---:|---:|---:|
| Baseline (enforce-eager) | 22.0 | 42.8 | 82.7 | 50.2 ms | — |
| FULL_DECODE_ONLY + prefix cache | 55.1 | 89.8 | 319 | 11.0 ms | — |
| + custom all-reduce | 87.9 | 168.7 | 260 | 8.9 ms | — |
| + Triton MI100 tuning | 87.6 | 167.8 | 308 | 9.0 ms | 130 ms |
| + block-size 32 | 88.1 | 172.2 | 338 | 9.1 ms | 73 ms |
| **+ skinny GEMM (gfx908)** | **112.6** | **209.4** | **386.0** | **6.6 ms** | **74 ms** |
| **+ flash decoding split-K** | **152.1** | — | **386.7** | **6.5 ms** | **75 ms** |
| TQ hybrid + graphs | 31.0 | — | — | 24.2 ms | — |
| ROCM_ATTN (prefill-decode split) | 82.8 | 159.5 | 295 | 9.6 ms | 133 ms |

Notable FP16-era wins:

- **Skinny GEMM (gfx908)** adds `__gfx908__` to the compile guard in `skinny_gemms.cu`, enabling `wvSplitK` and `LLMM1` kernels for small-M GEMM shapes. Single largest per-optimisation FP16 win after CUDA graphs.
- **Adaptive Flash-Decoding (Split-K)** dynamically scales `NUM_PAR_SOFTMAX_SEGMENTS ∈ {8, 16, 32, 64}` to fully saturate MI100's 120 CUs during decode. **+35 % c=1 tput** (113→152 tok/s) on coding agent. The M4 default heuristic is still this kernel; the HBM-FA tuning sweep confirmed no better configs exist on the workload shapes (see [`BENCH_HBM_FA_TUNING.md`](docs/experiments/BENCH_HBM_FA_TUNING.md)).
- **Block-size 32**: −44 % TTFT, +9.6 % c=4 tput. Carried into all M4 launches.
- **CUDA graphs on gfx908** were enabled by auto-disabling custom all-reduce (IPC buffers go stale on HIP graph replay):

| Mode (Llama-2-7B TP=4, 10×100 tok) | Throughput | Correct | Improvement |
|---:|---:|---|---:|
| enforce-eager | 383 tok/s | yes | baseline |
| torch.compile only | 368 tok/s | yes | −4 % |
| FULL_DECODE_ONLY + disable_custom_ar | **700 tok/s** | **yes** | **+83 %** |
| PIECEWISE + disable_custom_ar | **859 tok/s** | **yes** | **+124 %** |

### Llama-2-7B reference numbers

Wikitext-2 perplexity (50 chunks × 512 tokens, TP=4):

| Variant | PPL | Model size | Δ PPL | Generation |
|---|---:|---:|---:|---|
| FP16 | 7.59 | 12.6 GB | — | OK |
| W8A8 INT8 (GPTQ, MI100 Triton kernel) | 8.01 | 6.5 GB | +5.5 % | OK |

Serving throughput (TP=4, enforce-eager, 50 prompts, 128 in / 128 out):

| Variant | Output tok/s | TPOT median | TTFT median | Notes |
|---|---:|---:|---:|---|
| FP16 | 225.6 | 26.5 ms | 77.3 ms | |
| W8A8 INT8 | 219.7 | 32.3 ms | 98.4 ms | −2.6 % tput |

These numbers were eager-mode; the W8A8 path was subsequently extended with
hipBLASLt and Composable Kernel for the Qwen3.5-9B-w8a8 production artifact
(see §Latest).

### TunableOp GEMM tuning (FP16, still recommended for FP16 launches)

| Metric | No TunableOp | TunableOp | Δ |
|---|---:|---:|---:|
| c=1 throughput (tok/s) | 63.1 | 71.6 | **+13.4 %** |
| c=1 TTFT (ms) | 684 | 81 | **−88.2 %** |
| c=1 TPOT (ms) | 11.2 | 11.7 | +4.4 % |

Set `PYTORCH_TUNABLEOP_ENABLED=1 PYTORCH_TUNABLEOP_TUNING=0
PYTORCH_TUNABLEOP_FILENAME=…` to replay cached per-GPU CSVs. The quant
stack supersedes this via hipBLASLt-tuned shapes + CK instances; TunableOp
is still useful for FP16 launches.

### Sustained load (FP16, 200 requests c=4)

| Config | Requests | Concurrency | Success Rate | GPU Temp | VRAM |
|---|---:|---:|---:|---:|---:|
| FULL_DECODE_ONLY + prefix cache | 200 | 4 | 100 % | 45–54 °C | 93 % stable |

### Superseded: W8A8 GPTQ "BROKEN on Qwen3.5-9B" finding

The original W8A8 INT8 GPTQ artifact for Qwen3.5-9B failed end-to-end despite
acceptable PPL (10.02 vs 9.78 FP16) — generation degenerated into gibberish
due to GPTQ reconstruction errors of 50–124 on GatedDeltaNet `in_proj_qkv`
layers. **This is no longer the current state.** RedHatAI's
`/models/Qwen3.5-9B-w8a8` uses a calibration that resolves the issue
(PPL 9.6518, 9/10 coding, 5/5 needle@32k — see §Latest quality gates).
Off-the-shelf symmetric per-channel GPTQ is still incompatible with
GatedDeltaNet; the lesson stands that **perplexity alone is insufficient to
validate quantisation** — always pair PPL with coding-rubric + needle@32k.

---

## Upstream rebase notes

The fork is currently on vLLM `0.20.2rc1.dev107+gd960f21e4` (worktree head
`71fb9375c`, editable install at `/opt/vllm-env`). MI100-specific code
footprint has grown well beyond the original ~200-line / 5-file estimate:

- `vllm/platforms/rocm.py` — `_ON_MI100`, `on_mi100()`, `use_custom_allreduce()` update
- `vllm/model_executor/kernels/linear/scaled_mm/mi100.py` — FP8 emulation kernel
- `vllm/model_executor/kernels/scaled_mm/mi100_int8.py` (+ `_dispatcher.py`) — Triton W8A8 + CK/hipBLASLt dispatcher
- `vllm/model_executor/kernels/linear/mi100_w4a16.py` — custom W4A16 Triton kernel
- `vllm/model_executor/kernels/configs/gfx908/` — AOT-tuned configs (`mi100_int8_*`, `mi100_w4a16_*`, `hipblaslt_tuned_shapes.json`)
- `csrc/quantization/w8a8/int8/ck/` — Composable Kernel INT8 GEMM instances + Python bindings (`torch.ops._rocm_C.ck_int8_gemm`)
- `csrc/rocm/attention.cu` and `vllm/v1/attention/backends/triton_attn.py` — gfx908 MFMA guards + KV-INT8 inline-padded `head_size + sizeof(f32)` layout
- `scripts/launch_*.sh` — 24 production launch scripts (12 base + 12 hbm variants); 12 `launch_hbm2_*` research scripts for the tuned-flash-decode lookup

**Tuning manifests** (SHA256-pinned):

- `/root/bench-int8-w4a16/final/tuning_hashes.json` — 68 entries (TensileLite library + per-shape Triton/W4A16 JSON configs)
- `/root/bench-int8-w4a16-hbm/m4-final/tuning_hashes_hbm.json` — carry-forward, zero new tuning files (HBM mission is config-only)

**Cross-cutting gates** (must pass before rebase):

- `scripts/check_forbidden_intrinsics.sh build/**/*.so` — zero MI300+ intrinsics (`v_smfmac_*`, `v_mfma_*scale*`)
- `scripts/verify_tuning_hashes.py` — all pinned files unchanged
- `scripts/validate_results_schema.py` — every bench JSON conforms to `scripts/bench_schema.json`
- `scripts/check_default_pareto.py` — no silent default-config regressions

**Rebase strategy:**

1. Create a clean patch series from MI100-specific commits
2. Rebase onto latest upstream main or release tag
3. Resolve conflicts (most likely in `rocm.py`, `triton_attn.py`, `kernels/linear/`, `kernels/scaled_mm/`)
4. Re-verify dispatcher priority CK > hipBLASLt > Triton (`tests/kernels/quantization/test_ck_dispatch_priority.py`)
5. Re-run M4 HBM grid (24 cells × 200 prompts × 2 workloads) + 3 quality gates (PPL / coding / needle)
6. Confirm forbidden-intrinsics + tuning-hash gates still PASS

---

*Last updated: 2026-05-19 | Aggregates: MI100 Throughput Optimization (FP16),
TurboQuant Backend, Triton MI100 Tile Tuning, Block-Size & Backend Sweep,
Skinny GEMM gfx908, GEMM TunableOp Autotuning, MI100 Custom INT8/W4A16
Kernels (M0–M5), MI100 HBM Optimization Config-Sweep (M1–M4), MI100 Triton
Flash-Decoding INT8-PTH Tuning + CK FA2 Investigation.*
