<!-- markdownlint-disable MD013 -->
# Feasibility: a hand-built M=1 int4 MoE kernel for gfx908 (issue #58 methodology, 2026-06)

> **RESOLVED → WIN.** This note is the diagnostic half (#58 steps 1–2) that
> issued a "conditional GO." The kernel was then built and **wins 1.51× on the
> isolated GEMM** — see `BENCH_MOE_HANDKERNEL_GEMV_2026_06.md` for the result.
> Note: the winning design is **MFMA + split-K** (keep `tl.dot`, fix occupancy),
> NOT the GEMV the early draft below assumed. The diagnosis here (occupancy) was
> right; the predicted *kernel choice* (GEMV) was wrong and lost 0.63× before
> the split-K MFMA version won.

After `BENCH_MOE_RETUNE_FROM_SCRATCH_2026_06.md` showed **tile-tuning** has no
headroom, the next question was: would the issue-#58 evidence-driven loop (find
hot kernel → roofline → reproduce → hand-build the right primitive) find a
*structural* win that tuning can't? This note applies #58 steps 1–2 (the cheap
diagnostic half) and makes a go/no-go call **before** writing a kernel.

## Important context vs #57/#58

#57's win was **gfx900 (Vega10), which has NO MFMA**: `tl.dot` at M=1 lowered to
a padded FP32 GEMM, so a hand GEMV (no `tl.dot`) was 2–3× faster. **Our gfx908
(MI100/CDNA1) HAS MFMA**, so that exact win does not transfer. The question is
whether a *different* structural problem exists on gfx908.

## Step 1 — find the hot kernel (rocprofv3, M=1, real W4A16 MoE shape)

Profiled `fused_experts` for the real Qwen3-Coder-Next shape (E=512, top-10,
H=2048, N=128/shard at TP=4, gs=32). Per-iteration GPU kernel breakdown
(excluding `torch.randn` harness noise):

| kernel | us/iter | share | role |
|---|---:|---:|---|
| `fused_moe_kernel_gptq_awq` | ~57 | ~63% | **int4 expert GEMM** |
| `reduce_kernel` (at::native) | ~12 | ~13% | routing softmax/sum |
| `moe_align_block_size` | ~10 | ~11% | expert sort/align |
| `act_and_mul` | ~5 | ~6% | silu |
| `count_and_sort_expert_tokens` | ~4.6 | ~5% | routing |
| `topkGating` | ~1.4 | ~2% | router |
| **total real MoE** | **~90** | 100% | per layer, per token |

The int4 GEMM dominates (~63%). The rest is routing/glue.

## Step 2 — roofline

| M | measured (GEMM path) | HBM floor @1.2TB/s | distance |
|---|---:|---:|---:|
| 1 | 455 us* | 3.7 us | **124× above** |
| 4 | 452 us* | 14.7 us | 31× above |
| 16 | 467 us* | 59 us | 8× above |

\*full `benchmark_config` incl. all phases; the GEMM alone is ~57 us/layer.
Either way, **M=1 decode is ~15× above roofline on the GEMM and ~124× on the
fused path** — enormous nominal headroom. (Contrast: dense FP16 9B decode was
already near roofline; this MoE kernel is not.)

## Step 1b — WHERE the GEMM time goes (the decisive diagnostic)

Two experiments pinned the cause:

1. **Not wasted work on inactive experts.** Kernel time is **flat ~59–68 us
   across E=16/64/128/512** at fixed top-10, M=1. The kernel correctly touches
   only the ~10 active experts; adding 500 idle experts costs nothing. So the
   cost is the *active* GEMM work being slow, not redundant expert processing.

2. **Under-occupancy.** Launch geometry of the expensive GEMM (gate/up,
   N=256): `grid=20480, block=256 → 80 workgroups, VGPR=108, LDS=0`. **MI100 has
   120 CUs**, so 80 workgroups leave ~⅓ of CUs idle, and VGPR=108 caps occupancy
   near 1 wave/CU. The M=1 GEMV is **latency/occupancy-bound, not
   bandwidth-bound** — it never gets enough parallel waves to hide HBM latency,
   which is exactly why it sits 15× above roofline.

This is a genuine *structural* gap that tile-tuning cannot fix (tuning only
reshuffles tiles within the same kernel that already can't fill the machine at
M=1). A hand-built kernel that (a) maps the 10 active experts' GEMVs across all
120 CUs (split-K / persistent-CU scheme), and (b) lowers VGPR pressure to raise
occupancy, could plausibly approach roofline.

## Step 9 (done upfront) — end-to-end ceiling

Honest accounting against the measured decode budget:
- 48 MoE layers × ~90 us = **~4,320 us** of the **18,464 us** TPOT = **~23%** of
  decode is the MoE fused path.
- The int4 GEMM specifically: 48 × ~57 us = ~2,740 us = **~15% of decode**.
- A hand kernel that takes the GEMM 57 → ~10 us (realistic, not the 4 us ideal)
  saves 48 × ~47 us ≈ 2,260 us → **~12% lower TPOT → ~14% higher decode tok/s.**

So the prize is real and meaningful (~14% tok/s), but bounded — the GEMM is only
~15% of decode, so even a perfect kernel can't 2× the model.

## Go / no-go

**Conditional GO, scoped as research** — this is the one lever in the whole
series with real structural headroom (unlike tile-tuning, which was ±2%). But
the difficulty and risks are substantial and must be stated:

Risks / hard parts:
1. **MFMA changes the game vs #57.** The win here is occupancy + split-K, not
   "avoid tl.dot." A naive GEMV may *lose* to the existing MFMA-capable kernel;
   the hand kernel must out-occupy it, which is subtle on CDNA1.
2. **Correctness surface is large.** Must match the exact AWQ/gptq int4 packing
   + group-scale + reverse-order unpack, top-k routing, and silu fusion, then
   pass rel-err ≤1e-3 vs the in-tree kernel (#58 step 4).
3. **Integration gate.** Must dispatch only on gfx908 + M≤~8 + this expert shape,
   falling back everywhere else (#58 step 8), and the microbench win must survive
   end-to-end (routing/align kernels stay; only the GEMM is replaced).
4. **The tuner instability** (uncatchable HIP aborts under sustained Triton
   JIT, documented in the retune doc) will also bite kernel iteration; the
   subprocess-isolated harness (`moe_tune_isolated.py`) is reusable for the
   eventual config sweep.

Recommended next steps if pursued (in #58 order):
1. Standalone Triton int4 MoE-GEMV microbench at the real shape; establish
   correctness vs `fused_moe_kernel_gptq_awq` first.
2. Structural iteration: split-K across CUs for the M=1 active-expert GEMVs;
   target ≥110/120 CUs occupied, VGPR < 96.
3. Only if microbench beats the in-tree kernel by a wide margin (it must clear
   the MFMA baseline, not just FP16), wire in behind `on_gfx908()` + M/shape gate
   and re-run the decode A/B.

## Reproduction
- rocprof breakdown: `rocprofv3 --kernel-trace` on `/tmp/moe_prof.py` (real
  shape), parsed for per-kernel duration + launch geometry.
- E-scaling + occupancy probes: inline scripts in the session log (CUDA-graph
  timed, `HIP_VISIBLE_DEVICES=2`).

Build: vLLM `0.20.2rc1.dev107+gd960f21e4`, torch `2.11.0+rocm7.2`, Triton
`3.5.1`, ROCm 7.12, 4× MI100 gfx908. Model `/models/Qwen3-Coder-Next-AWQ-4bit`.

---

*Generated 2026-06-03. AI assistance was used.*
