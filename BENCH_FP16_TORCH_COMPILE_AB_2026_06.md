<!-- markdownlint-disable MD013 -->
# FP16 torch.compile + PIECEWISE graphs A/B (gfx908/MI100, 2026-06)

Follow-up to `RESEARCH_FP16_OPTS_AND_QUANT_FIT.md` §1.1. Tests whether the
community MI100 config (`VLLM_MI100_TORCH_COMPILE=1` +
`--compilation-config '{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'`)
beats our production `FULL_DECODE_ONLY` default on FP16 Qwen3.5-9B decode.

## Headline

**torch.compile + FULL_AND_PIECEWISE is a real but modest decode win: +3.64%
geomean, scaling with concurrency/TP (+0.06% at tp1_c1 → +6.92% at tp4_c2).**
The win is concentrated exactly where the community reported it (higher
concurrency, TP=4), and is ~neutral at single-stream c=1.

| cell | baseline (FULL_DECODE_ONLY) | compile (mode3+FULL_AND_PIECEWISE) | Δ tput | Δ TTFT | Δ TPOT |
|---|---:|---:|---:|---:|---:|
| tp1_c1 | 44.63 | 44.65 | **+0.06%** | −3.8% | +0.7% |
| tp1_c2 | 81.84 | 84.15 | **+2.82%** | −2.4% | −2.3% |
| tp4_c1 | 73.39 | 76.97 | **+4.88%** | −4.8% | −5.3% |
| tp4_c2 | 130.28 | 139.30 | **+6.92%** | −12.9% | −6.1% |
| **decode geomean** | **76.87** | **79.67** | **+3.64%** | — | — |

TTFT and TPOT also improve on every cell except tp1_c1 TPOT (noise). The win
grows with batch — consistent with Inductor fusions + piecewise prefill graphs
paying off once there's enough work to amortize them.

## Critical constraints discovered (these gate deployability)

### 1. PIECEWISE cudagraphs REQUIRE torch.compile — they are not separable
vLLM rejects piecewise-without-compile:
> `Cudagraph mode FULL_AND_PIECEWISE is not compatible with compilation mode 0. Overriding to NONE.`

So you cannot get piecewise graphs while keeping `mode=NONE`. The only way to
the community config is full `mode=3` (VLLM_COMPILE/Inductor) + piecewise
together. There is no "cheap" piecewise-only path.

### 2. torch.compile CRASHES on our 0.20.2 build — works only on 0.19.2
On the **0.20.2** build that matches the current BENCH docs
(`fuzzy-hornets`, `0.20.2rc1.dev107+gd960f21e4`), `mode=3` aborts engine init:
> `torch._dynamo.exc.Unsupported: torch.* op returned non-Tensor` on
> `torch.cuda.is_current_stream_capturing()` during Dynamo fullgraph capture.

On the **0.19.2** build (`smooth-sheep`, `0.19.2rc1.dev88+g16c8cbd0b`) the same
config compiles cleanly (≈40 s warmup) and serves healthy. **This A/B was
therefore run entirely on 0.19.2, both arms on the same binary**, so the +3.64%
is an honest config-vs-config delta. It does **not** transfer to our current
0.20.2 production build as-is — that build needs the Dynamo crash fixed first
(or a forward-port of the community's `+mi100` compile patch).

This matches `rocm.py`'s own caution ("Inductor fusions not available on ROCm /
torch.compile adds overhead") — but the caution is now quantified: on a build
where compile *works*, it's a +3.64% win, not a loss. The community runs 0.19.2
specifically because that's where their compile patch lands cleanly.

### 3. Cost: ~40 s (TP1) compile warmup per server start + higher peak memory
`max_model_len` had to stay at 32768 and `--gpu-memory-utilization` at 0.90
(vs 0.93 baseline) for headroom — consistent with the `rocm.py` note that
Inductor increases peak memory. Compile cache (`/root/.cache/vllm/torch_compile_cache`)
amortizes warmup across restarts.

## Verdict

- **Is it a win?** Yes — **+3.64% decode geomean**, up to **+6.92%** at tp4_c2,
  with TTFT/TPOT improvements too. Real, not noise.
- **Can we ship it today?** **Not on 0.20.2** — torch.compile crashes there.
  It's deployable only on 0.19.2 today.
- **Is it worth pursuing?** Marginal-but-positive. Unlike the quant decode
  regressions, this is the FP16 path getting genuinely faster. The blocker is
  purely the 0.20.2 Dynamo incompatibility, not the optimization itself.

## Recommendation

1. **Short term:** if running FP16 production on 0.19.2, enable
   `VLLM_MI100_TORCH_COMPILE=1` + `FULL_AND_PIECEWISE` for TP=4 / multi-stream
   serving (the +5–7% cells). Leave it off for pure single-stream c=1 (neutral,
   not worth the 40 s warmup).
2. **To unlock on 0.20.2+:** fix the `is_current_stream_capturing()` Dynamo
   graph break — either mark that op as a Dynamo-allowed graph break / wrap it
   so it isn't traced into the FX output, or forward-port the community
   `+mi100` compile patch. That's the gating engineering task; the perf payoff
   (+3.6% geomean) is now measured and justifies it.
3. **Bigger upside is likely on MoE / larger models:** the community's headline
   compile wins were on Qwen3.6-35B-A3B (MoE), where Inductor has more to fuse
   than on dense 9B. Re-test the A/B once a larger GPTQ-8bit MoE model is
   loadable (ties into the W8A16 capacity work).

## Reproduction

```bash
# Both arms on the 0.19.2 build (smooth-sheep worktree)
/root/fp16-bench/run_compile_ab.sh
# Results: /root/fp16-bench/compile_ab/{baseline,compile}_tp{1,4}_c{1,2}/raw.json
```

Build manifest: vLLM `0.19.2rc1.dev88+g16c8cbd0b`, torch `2.11.0+rocm7.2`,
Triton `3.5.1`, ROCm 7.12, 4× MI100 gfx908. Bench harness identical to
`BENCH_FP16_VS_QUANT_2026_06.md` (200 prompts, seed 42, random 1024/256).

> Note: geomean baseline here (76.87) ≈ the 0.20.2 FP16 baseline (76.29 in
> `BENCH_FP16_VS_QUANT_2026_06.md`) — the two builds are within ~0.8% on
> FULL_DECODE_ONLY, so the +3.64% compile delta is meaningful relative to either.

---

*Generated 2026-06-02. AI assistance was used.*
