<!-- markdownlint-disable MD013 -->
# Capstone: torch.compile + PIECEWISE on a QUANTIZED MoE (W4A16, gfx908/MI100, 2026-06)

Fourth and final in the compile series — the full-stack config that stacks all
three levers:

| doc | model | build | Δ geomean |
|---|---|---|---|
| `BENCH_FP16_TORCH_COMPILE_AB_2026_06.md` | dense 9B | 0.19.2 | +3.64% |
| `BENCH_FP16_COMPILE_PORT_0202_2026_06.md` | dense 9B | 0.20.2 (fixed) | +1.81% |
| `BENCH_FP16_COMPILE_MOE_2026_06.md` | fp16 35B-A3B MoE | 0.20.2 (fixed) | +10.14% |
| **this** | **W4A16 512e MoE** | **0.20.2 (fixed)** | **+9.85%** |

Question: does compile's big MoE win survive when the model is **also
quantized** (W4A16)? Quant adds a fused dequant→fp16 step per GEMM; if that
broke compile's fusion advantage, the win would shrink.

## Result: yes — quant MoE keeps the full ~10% compile win, with much better TTFT

**Qwen3-Coder-Next-AWQ-4bit** (`qwen3_next`, **512 experts / 10 active**, W4A16
compressed-tensors group_size=32, 45 GB, TP=4), fixed 0.20.2 build, decode
subset, same HBM-identical harness:

| cell | baseline (FULL_DECODE_ONLY) | compile (mode3+FULL_AND_PIECEWISE) | Δ tput | Δ TTFT |
|---|---:|---:|---:|---:|
| tp4_c1 | 51.19 | 56.84 | **+11.03%** | **−28.2%** |
| tp4_c2 | 83.32 | 91.27 | **+9.54%** | **−48.8%** |
| tp4_c4 | 144.88 | 157.89 | **+8.98%** | **−32.2%** |
| **decode geomean** | **85.18** | **93.57** | **+9.85%** | — |

**+8.98% to +11.03% on every cell — statistically the same win as fp16 MoE
(+10.14%)**, and like fp16 MoE there's **no low-concurrency regression** (even
c=1 is +11%). The headline difference vs fp16 MoE: **TTFT collapses by 28–49%**
under compile here, a much bigger prefill latency improvement than fp16 MoE saw
(−21% / +2% / −12%).

## Why quant doesn't hurt the compile win (and helps TTFT)

- **Compile fuses around the dequant, not through it.** The W4A16 path is
  dequant→fp16→MFMA; Inductor still fuses the surrounding routing/activation/
  elementwise ops and captures the whole thing into the piecewise graph. The
  per-GEMM dequant is a kernel compile leaves intact, so the launch-overhead
  reduction (the actual source of the MoE win) is unaffected.
- **Quant makes the model more launch-bound, so compile helps more on TTFT.**
  W4A16 shrinks the GEMM compute (4-bit weights) but adds dequant kernels and
  keeps 512-expert routing — the decode/prefill is even more dominated by many
  small kernel launches than fp16. cudagraph capture removes exactly that
  overhead, which is why prefill TTFT drops so hard (−49% at c=2).
- This is the same mechanism as the fp16 MoE doc, amplified by quant's kernel
  count.

## The capstone validates the whole stack on gfx908

This single config exercises everything we built/fixed:

1. **Quant capacity** — W4A16 fits a 512-expert MoE (45 GB) on 4×MI100 with KV
   cache for **1.21 M tokens** (74× concurrency @ 16k ctx); the fp16 35B only
   reached 896k / 54.7×. Quant buys ~35% more KV headroom here.
2. **The Dynamo fix** — `try_stash_fused_silu_quant_int8`'s
   `torch.compiler.is_compiling()` guard. The W4A16 MoE routes its MLP through
   the shared `qwen2_moe.py` forward (`qwen3_next.py:56` imports `Qwen2MoeMLP`),
   so this is the third distinct model (dense 9B, fp16 35B, now W4A16 512e) on
   which the one-line fix lets compile complete with no Dynamo crash.
3. **compile + PIECEWISE** — +9.85% on top of quant.

Compile cost: 72.8 s (vs 59 s fp16 35B, 39 s dense 9B); cudagraph capture 124 s;
model load 11.74 GiB/GPU. All one-time startup.

## Caveat (perf is *understated* here)

vLLM logged `Using default MoE config ... E=512,N=128,...int4_w4a16.json not
found` — there is **no gfx908-tuned fused-MoE config for this W4A16 expert
shape**, so the absolute throughput is sub-optimal. This affects **both arms
equally**, so the **+9.85% compile delta is valid**; only the absolute tok/s
would rise with a tuned config. Generating an `E=512,N=128` int4_w4a16 tuning
config for MI100 is the obvious follow-up to lift the baseline (ties into the
merged gfx908 fused-MoE config work).

## Recommendation (series conclusion)

1. **Enable compile for all MoE serving on gfx908 — fp16 or quantized — by
   default.** +9–10% throughput, dramatically better TTFT, no concurrency
   regression. This is the strongest and most robust FP16-path win in the whole
   series, and it now holds across dense-vs-MoE and fp16-vs-W4A16.
2. **Dense models stay concurrency-gated** (compile for TP=4, off for TP=1).
3. **Ship the W4A16 MoE config as a reference deployment:** it maximizes both
   capacity (1.21M-token KV) and speed (compile +9.85%) on 4×MI100.
4. **Next perf lever:** tune the `E=512,N=128 int4_w4a16` fused-MoE config for
   gfx908 to raise the absolute baseline (orthogonal to, and stacks with, the
   compile win measured here).

## Reproduction

```bash
/root/fp16-bench/run_qmoe_compile_ab.sh   # TP=4, c{1,2,4}, baseline vs compile
# results: /root/fp16-bench/qmoe_compile_ab/{baseline,compile}_tp4_c{1,2,4}/raw.json
```

Model `/models/Qwen3-Coder-Next-AWQ-4bit` (`qwen3_next`, 512e/10a, W4A16
compressed-tensors gs=32). Build: vLLM `0.20.2rc1.dev107+gd960f21e4` (fixed),
torch `2.11.0+rocm7.2`, Triton `3.5.1`, ROCm 7.12, 4× MI100 gfx908. Harness:
200 prompts, seed 42, random 1024/256, `--dtype float16`, max-model-len 16384,
gpu-mem-util 0.92.

---

*Generated 2026-06-02. AI assistance was used.*
