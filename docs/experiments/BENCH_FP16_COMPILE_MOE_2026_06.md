<!-- markdownlint-disable MD013 -->
# torch.compile + PIECEWISE on MoE (Qwen3.5-35B-A3B, gfx908/MI100, 2026-06)

Third in the compile series:

- `BENCH_FP16_TORCH_COMPILE_AB_2026_06.md` — dense 9B on 0.19.2: +3.64%
- `BENCH_FP16_COMPILE_PORT_0202_2026_06.md` — Dynamo fix + dense 9B on 0.20.2: +1.81%
- **this** — MoE 35B on the fixed 0.20.2 build

Hypothesis going in: the community's headline compile wins were on **35B-A3B
MoE**, where Inductor has far more to fuse (256 experts, routing, grouped GEMMs)
than on dense 9B. If true, compile should win more here.

## Result: confirmed — compile wins ~2× more on MoE than dense

**Qwen3.5-35B-A3B** (`qwen3_5_moe`, 256 experts / 8 active, 67 GB bf16→fp16,
TP=4), fixed 0.20.2 build, decode subset, same HBM-identical harness:

| cell | baseline (FULL_DECODE_ONLY) | compile (mode3+FULL_AND_PIECEWISE) | Δ tput | Δ TTFT |
|---|---:|---:|---:|---:|
| tp4_c1 | 61.99 | 68.60 | **+10.66%** | −20.9% |
| tp4_c2 | 108.70 | 120.61 | **+10.96%** | +2.0% |
| tp4_c4 | 188.35 | 204.96 | **+8.82%** | −12.2% |
| **decode geomean** | **108.27** | **119.25** | **+10.14%** | — |

Every cell wins **+8.8% to +11.0%**, and TTFT improves on 2 of 3. Compare to
the **dense 9B** on the same 0.20.2 build: +4.47% / +7.16% at tp4_c1/c2. **MoE
roughly doubles the compile payoff**, and unlike dense there's **no low-c
regression** — even c=1 is +10.66% here.

## Why MoE benefits more

- **Far more fusible work per layer:** expert routing (`grouped_topk`, softmax,
  top-k), the gate/up/down expert GEMMs, and the silu activation give Inductor
  many small ops to fuse + capture into the piecewise graph, where dense 9B is
  dominated by a few big GEMMs already near roofline.
- **Launch-overhead amortization:** MoE decode issues many small kernels per
  token (per-expert), so the cudagraph + compile launch-overhead reduction pays
  off even at c=1 — exactly the regime where dense 9B regressed (−3.2% at
  tp1_c1) because it lacked enough small kernels to amortize.
- This matches the community's finding (their biggest compile wins were on
  Qwen3.6-35B-A3B), and our own merged gfx908 fused-MoE configs are part of why
  the MoE path is well-formed enough to compile cleanly.

## The Dynamo fix held on the MoE path

The same `torch.compiler.is_compiling()` guard in `fused_silu_quant_int8.py`
(from `BENCH_FP16_COMPILE_PORT_0202_2026_06.md`) was exercised here: the 35B MoE
routes its MLP through the shared `qwen2_moe.py` forward that calls
`try_stash_fused_silu_quant_int8`. Compile completed in **59 s** (vs 39 s dense
9B) with **no Dynamo crash**, confirming the fix generalizes beyond the dense
model it was diagnosed on. Model load: 16.3 GiB/GPU across TP=4; KV cache
896,691 tokens (54.7× max concurrency @ 16k ctx).

## Recommendation (updated)

1. **Enable compile for MoE serving on 0.20.2 — unconditionally.** +8.8–11%
   across all concurrencies with better TTFT, no c=1 regression. This is the
   strongest FP16-path win found in the whole series.
2. **Dense models stay concurrency-gated** (compile for TP=4 multi-stream, off
   for TP=1) per the port doc.
3. **This compounds with the capacity story:** MoE models like 35B-A3B are
   exactly what quant (W8A16/W4A16) lets us fit on 4×MI100 — and now we know
   compile gives them an *additional* ~10% on top. The two levers stack:
   quant unlocks the model, compile speeds it up.
4. **Merge the Dynamo fix** — it's now validated on both dense and MoE, gated
   behind `VLLM_MI100_TORCH_COMPILE=1`, zero effect when off.

## Caveat

The 35B here is **unquantized fp16** (fits TP=4 at 16k ctx). The natural next
step is **GPTQ-8bit/4bit MoE** (e.g. a quantized 35B-A3B or larger), which would
combine: (a) quant capacity to fit bigger MoE / longer context, (b) the +10%
compile win, (c) the gfx908 fused-MoE tuned configs. That's the full-stack
config worth standing up next.

## Reproduction

```bash
/root/fp16-bench/run_moe_compile_ab.sh   # TP=4, c{1,2,4}, baseline vs compile
# results: /root/fp16-bench/moe_compile_ab/{baseline,compile}_tp4_c{1,2,4}/raw.json
```

Model `/models/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled`
(`qwen3_5_moe`, 256e/8a). Build: vLLM `0.20.2rc1.dev107+gd960f21e4` (fixed),
torch `2.11.0+rocm7.2`, Triton `3.5.1`, ROCm 7.12, 4× MI100 gfx908. Harness:
200 prompts, seed 42, random 1024/256, `--dtype float16`, max-model-len 16384,
gpu-mem-util 0.92.

---

*Generated 2026-06-02. AI assistance was used.*
