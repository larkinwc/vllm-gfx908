# Research: FP16-path optimizations + weight-only quant that fits the FP16 path (gfx908/MI100)

Companion to `BENCH_FP16_VS_QUANT_2026_06.md`. Two questions:

1. What can still speed up the **FP16 decode path** on gfx908?
2. What quant format keeps the **fast FP16 compute path** while shrinking weights so we can fit **larger models**?

Sources: this repo's kernels/`rocm.py`/BENCH docs, the community
`btbtyler09/mi100-llm-testing` MI100 work (same project our MoE configs came
from), vLLM quantization docs, and the 2026-06 Marlin-repack negative result.

---

## 0. The key reframe (why this matters)

**"W4A16" and "W8A16" are not a different compute path from FP16 — they ARE the
FP16 path with compressed weights.** Our `triton_w4a16` kernel does *fused
int4→fp16 dequant + fp16 MFMA GEMM* in one pass
(`vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py:1`).
Activations stay FP16; only the weights are stored quantized and expanded to
FP16 right before the matmul. So the question "what quant fits the FP16 path"
has a clean answer: **any weight-only (`WxA16`) format** — GPTQ/AWQ INT4,
INT8-weight, or FP8-weight-only. The activation-quant paths (W8A8 INT8) are the
ones that leave the FP16 path.

The 2026-06 bench already proved the cost side: at low concurrency the decode
GEMM is **compute/issue/launch-bound, not HBM-bound** (Marlin doc §6: hot kernel
63.7% of GPU time but only **15.3% HBM util** at M=1). That's why 4-bit weights
*regress* tok/s — the dequant + helper-kernel overhead isn't hidden behind
enough arithmetic. This governs both halves of the answer below.

---

## Part 1 — FP16-path optimizations still on the table

Ranked by expected payoff on gfx908. "Shipping" = already in our production FP16
config (`/root/launch-vllm-optimized.sh`).

### 1.1 torch.compile + PIECEWISE graphs — **highest-upside, contested**

- **Status:** our `rocm.py` *disables* torch.compile and forces
  `FULL_DECODE_ONLY` on MI100 (`vllm/platforms/rocm.py:745-799`), citing a
  −9.7% c=8 regression on REAP-172B-AWQ and Inductor fusions being unavailable
  on ROCm.
- **Contradicting evidence:** the community MI100 image makes
  `VLLM_MI100_TORCH_COMPILE=1` + `--compilation-config '{"mode":3,
  "cudagraph_mode":"FULL_AND_PIECEWISE"}'` its **headline v0.19 decode win**
  ("main decode-throughput win in v0.19 vs v0.16"), and reports 1365 tok/s
  aggregate @ c=128 / 10.87 ms TPOT @ c=1 on Qwen3.6-35B.
- **Why the conflict:** their win is at **high concurrency / MoE models**; our
  regression note is **dense 172B-AWQ at c=8**. These don't actually contradict
  — the payoff is workload-dependent.
- **Action:** A/B `VLLM_MI100_TORCH_COMPILE=1` + `FULL_AND_PIECEWISE` on
  Qwen3.5-9B FP16 across c=1/2/4 vs our `FULL_DECODE_ONLY` baseline. Lowest-cost
  experiment with the biggest potential upside; gated behind an existing flag.

### 1.2 AITER Triton RoPE/attention — **untested in our stack, shipping in community's**

- **Status:** `VLLM_ROCM_USE_AITER` is read by our `rocm.py` but the FP16 launch
  script does **not** set it. Community runs `VLLM_ROCM_USE_AITER=1` by default
  on gfx908 (Triton RoPE + attention; CK/FP8/UA auto-disabled).
- **Caveat (their hard-won bug):** **AITER Unified Attention corrupts state
  after ~200 sustained requests on gfx908** — must keep `--attention-backend
  TRITON_ATTN`. Matches our own `BENCH.md` "AITER unified attention: Blocked".
- **Action:** test `VLLM_ROCM_USE_AITER=1` *with* `TRITON_ATTN` forced, RoPE
  only. Low risk; measure decode delta.

### 1.3 Attention backend explicit pin (`TRITON_ATTN`)

- Community explicitly pins `--attention-backend TRITON_ATTN` as the stable
  gfx908 choice. We rely on auto-selection. Worth pinning for reproducibility +
  to avoid any UA/CK auto-dispatch surprise. Near-zero risk.

### 1.4 Already shipping (no further action, documented for completeness)

- Skinny GEMM (`__gfx908__` guard) — biggest FP16 win after CUDA graphs.
- Adaptive Flash-Decoding split-K — sweep already confirmed optimal
  (`BENCH_HBM_FA_TUNING.md`).
- TunableOp rocBLAS replay — +13.4% c=1.
- Triton tile tuning, block-size 32, custom all-reduce + FULL_DECODE_ONLY graph.

### 1.5 Low-yield / proven-negative (do not re-attempt)

- **Marlin-style repack:** verified **−28%** on gfx908 — no `cp.async`, decode is
  issue-bound (`BENCH_W4A16_MARLIN_REPACK.md`). Settled.
- **Hand-ISA GEMM:** declined, HBM/issue-bound ceiling (`BENCH_M5_ISA.md`).
- **FP8 native / CK:** no hardware on CDNA1.

---

## Part 2 — Quant formats that fit the FP16 path (to run larger models)

The goal here is **capacity, not decode speed** — the 2026-06 bench shows quant
costs decode tok/s, but it buys the VRAM to run models that otherwise don't fit
4×32 GB. All options below keep FP16 activations (the fast path).

### 2.1 Format comparison (weight-only, FP16 activations)

| Format | Weights | vLLM kernel on gfx908 | ~Bytes/param | Qwen3.5-9B size | Fits bigger? | Quality (our data) |
|---|---|---|---|---:|---|---|
| **FP16** (ref) | fp16 | native MFMA | 2.0 | 19 GB | baseline | PPL 9.53 |
| **W8A16 / INT8 weight-only** | int8 + scale | `triton_w4a16` family / Triton WNA16 | ~1.0 | ~10–11 GB | **yes, ~2×** | best quant fidelity (8-bit) |
| **W4A16 GPTQ g=128** | int4 + scale | `mi100_w4a16_gemm_kernel` (tuned) ✅ | ~0.55 | 11 GB | **yes, ~3.5×** | PPL 9.80 (+2.9%) |
| **W4A16 AWQ→CT g=32** | int4 + scale | same kernel | ~0.6 | 11 GB | yes | PPL +4.08% (worse than GPTQ) |
| **AWQ-gemm (autoawq)** | int4 | Triton AWQ (untuned) ❌ | ~0.55 | 12 GB | yes | −36% tput, +4.5% PPL |
| W8A8 INT8 | int8 | leaves FP16 path (INT8 acts) | ~1.0 | 14 GB | yes | PPL +1.65% but slower decode |

### 2.2 Recommendation: **GPTQ for 4-bit, INT8 weight-only (W8A16) for 8-bit**

Our own three-way A/B already settled the 4-bit question
(`BENCH_W4A16_AWQ_VS_GPTQ_AB.md`): **GPTQ-W4A16 wins** — it leads AWQ-compressed
by ~1.6–1.8% throughput *and* on quality (PPL +3.11% vs +4.08%), and beats raw
autoawq-gemm by ~36% (the Triton AWQ kernel is untuned on gfx908). The community
MI100 repo independently lands on the same answer: **"GPTQ works well in 4-bit
and 8-bit"** as their primary recommendation, with curated GPTQ providers
(jart25, QuantTrio, cpatonn, btbtyler09).

**The biggest unexploited lever for "larger models that keep quality" is
8-bit weight-only (W8A16 / INT8-weight GPTQ)**, which we have **not** benched:

- ~2× capacity vs FP16 (10–11 GB) at near-FP16 quality (8-bit >> 4-bit fidelity).
- Community reports **Qwen3.6-35B-A3B GPTQ-8bit** as a top performer on 4×MI100
  (1365 tok/s aggregate). Our largest tested is 9B.
- Stays on the FP16 compute path (int8→fp16 dequant + fp16 MFMA).

### 2.3 What this unlocks (4×MI100 = 128 GB, ~119 GB usable @ 0.93)

| Model | FP16 | GPTQ-8bit (~1B/param) | GPTQ-4bit (~0.55B/param) |
|---|---|---|---|
| 9B | 19 GB ✅ | 10 GB ✅ | 6 GB ✅ |
| 30–35B (MoE A3B) | ~70 GB ✅ (tight) | ~35 GB ✅ | ~20 GB ✅ |
| 70B dense | 140 GB ❌ | ~70 GB ✅ | ~40 GB ✅ |
| 122B-A10B MoE | 244 GB ❌ | ~120 GB borderline | **~65 GB ✅** (community runs this) |

The community already runs **122B-A10B GPTQ-4bit** and **35B GPTQ-8bit** on the
same 4×MI100 hardware — these are the concrete "larger model" targets quant
unlocks for us, and both keep the FP16 activation path.

### 2.4 MoE tuning is required for these larger models

The community's `Model_Reports` note that the big wins come with **per-model
fused-MoE configs** (e.g. `int8_w8a16` E=256/N=128 tune for 35B-8bit;
`int4_w4a16` E=256/N=256 for 122B-4bit). We already merged gfx908 fused-MoE
configs (git log: "Add gfx908 fused-MoE tuning configs"). Extending those tunes
to the target model shapes is the enabling work, not new kernels.

---

## 3. Concrete next steps (ordered)

1. **Bench INT8 weight-only (W8A16) on Qwen3.5-9B** — the missing data point;
   expected ~2× capacity at near-FP16 quality, same FP16 path. Quantize with
   llm-compressor or grab a GPTQ-8bit checkpoint; run the 4-cell decode subset +
   PPL/coding/needle gates.
2. **A/B `VLLM_MI100_TORCH_COMPILE=1` + `FULL_AND_PIECEWISE`** on FP16
   Qwen3.5-9B (§1.1) — cheapest potential FP16 decode win, flag already exists.
3. **A/B `VLLM_ROCM_USE_AITER=1` + forced `TRITON_ATTN`** on FP16 (§1.2) — RoPE
   speedup, keep UA off.
4. **Scale up:** pull a **GPTQ-8bit 30–35B** (or 4-bit 70B/122B) checkpoint and
   validate it loads + serves on 4×MI100, reusing/extending our fused-MoE
   configs. This is the actual "fit larger models" deliverable.
5. **Capacity benchmark, not just tok/s:** measure max context / max concurrent
   sequences at fixed VRAM for FP16 vs W8A16 vs W4A16 — the axis where quant
   wins (noted as missing in `BENCH_FP16_VS_QUANT_2026_06.md`).

---

## 4. One-line answers

- **Fastest FP16 wins left:** torch.compile+PIECEWISE A/B, then AITER-RoPE — both
  behind existing flags, both contested/untested in *our* stack (community
  reports gains; verify on our workload).
- **Best quant that keeps the FP16 path:** **GPTQ weight-only** — 4-bit for max
  capacity (already validated as our best 4-bit), and **8-bit (W8A16), which we
  have not yet benched**, for near-FP16 quality at ~2× capacity. Both dequant to
  FP16 and use the same MFMA GEMM; AWQ and Marlin are proven worse on gfx908.

---

*Generated 2026-06-02. Grounded in repo kernels + BENCH docs + the
`btbtyler09/mi100-llm-testing` community MI100 reports. AI assistance was used.*
