# gfx900 MoE decode: grouped small-M int4 expert GEMV

A custom kernel that redeems int4 MoE **decode** on gfx900 (Vega10), the direct
analog of the dense int4 decode GEMV (#57) for the fused-MoE expert matmul.

Validated on **Qwen3.5-35B-A3B-GPTQ-Int4** (256 experts / 8 active, 40 layers,
hidden 2048, expert-intermediate 512, GPTQ int4 group_size 128 symmetric; attn +
shared_expert + lm_head kept FP16), TP4 on one socket, FULL_DECODE_ONLY CUDA graphs.

## TL;DR

| config | c=1 | c=8 | c=32 |
|---|---:|---:|---:|
| baseline (`tl.dot`) | 33.5 | 49.7 | 111.0 |
| untuned GEMV | 41.4 | 53.6 | 113.7 |
| **tuned GEMV (this)** | **47.1** | 52.2 | 113.7 |
| Δ single-stream | **+40.6%** | (no regression) | (unaffected) |

Single-stream decode **33.5 → 47.1 tok/s (+40.6%)**, tpot 29.9 → 21.2 ms, output
coherent. No regression at higher concurrency; batched decode (M>2) and prefill keep
the stock `tl.dot`. Progression: 33.5 (padded `tl.dot`) → 41.4 (untuned GEMV, first
cut) → 47.1 (tuned `BLOCK_SIZE_N=8`).

## Why: the expert GEMM is ~46% of the graph decode step

Profiling discipline mattered here. The **eager** decode profile is misleading — it
shows comm (RCCL) at 82.5% and the expert GEMM at only 8.8%. But that eager "comm"
is mostly **straggler-wait** from MoE's imbalanced per-layer expert routing (1620 µs
per collective vs the dense model's 155 µs), which CUDA graphs collapse by replaying
in lockstep. The graph-mode step is **29.9 ms**; the expert-GEMM GPU time (13.9
ms/step, pure compute, unchanged by graphs) is therefore **~46% of it** — the
dominant cost once graphs remove the wait.

The expert GEMM runs `fused_moe_kernel_gptq_awq`, which uses `tl.dot(a, b)`. gfx900
has no MFMA, so a 1-token decode is padded into a **16-row** `tl.dot` (the config
picks `BLOCK_SIZE_M=16`): ~16× of the issued work is padding waste. Isolated
microbench on the expert shapes (M=1): a real GEMV is **7.1× (gate_up, K2048→N1024)**
and **2.7× (down, K512→N2048)** faster than the padded path.

## The kernel

`vllm/model_executor/layers/fused_moe/fused_moe.py`:

- Added `GEMV_MODE: tl.constexpr` to `fused_moe_kernel_gptq_awq`. When set, the K-loop
  accumulates with an explicit reduction instead of a padded matmul:
  ```python
  accumulator += tl.sum(a[:, :, None] * b[None, :, :], axis=1)   # vs tl.dot(a, b)
  ```
  (`tl.dot` can't take M=1 on the no-MFMA path anyway — it pads to 16.)
- `get_default_config` (the `int4_w4a16` branch): for `on_gfx900() and M <= 2`, force
  `BLOCK_SIZE_M=1` + `GEMV_MODE=True`. `moe_align_block_size` then pads to 1 (no
  waste). Everything else — batched decode, prefill, non-gfx900 — is untouched
  (`GEMV_MODE` defaults False → `tl.dot`).

**Gate = M ≤ 2.** The initial `M ≤ 20` gate regressed c=8 by −21% (a per-token
`BLOCK_SIZE_M=1` GEMV loses the batching a 16-row `tl.dot` gets for 8 real rows). At
`M ≤ 2` only latency-critical single-stream decode uses the GEMV; batched decode keeps
`tl.dot`.

**Tuned config: `BLOCK_SIZE_N=8, BLOCK_SIZE_K=128, num_warps=4`** (swept via
`gemv_sweep.sh`). The sweep found a clear monotonic trend — *smaller* `BLOCK_SIZE_N`
is better (N=64→35, N=32→43, N=16→44, N=8→48 tok/s in the quick harness): a tiny M=1
GEMV wants many small programs to fill all 56 CUs. `BLOCK_SIZE_K=128` (= the group
size) beats 64; `BLOCK_SIZE_K=256` fails to compile. Tuning lifted the untuned GEMV
(`N=32,K=64`) from the +23.6% first cut to the number in the table above.

## Scope / correctness

- Applies only to gfx900 + int4 (`int4_w4a16`) MoE decode at M ≤ 2, and only when
  the kill-switch `VLLM_GFX900_MOE_GEMV != 0` (default on; set `=0` to force the
  stock `tl.dot`). All other paths — batched decode M>2, prefill, non-gfx900 — are
  bit-identical to upstream (`GEMV_MODE` defaults False).
- **Verified against the baseline `tl.dot` path.** Decode A/B (greedy, 128 tokens,
  same prompt, `VLLM_GFX900_MOE_GEMV=1` vs `=0`): **128/128 exact token match**,
  chosen-token logprob delta **mean 0.0019 / max 0.069 nats**. The reduction is the
  same dot product, differing only in fp accumulation order, and never flips the
  argmax. Note a prefill-based perplexity would *not* exercise this kernel — it only
  runs at decode M≤2 — so the A/B is decode-path, not prefill-ppl.

## Repro

- `moe35_prof.py bench` — TP4 graph decode c=1/8/32 (this table).
- `moe35_prof.py prof` + `parse_any.py` — eager kernel breakdown.
- `mb_expert_gemv.py` — single-die expert-shape GEMV vs padded microbench.
- `gemv_sweep.sh` — BLOCK_SIZE_N/K/num_warps sweep (env GEMV_BN/BK/NW/NS).
- `corr_gen.py` + `corr_cmp.py` — decode-path correctness A/B vs baseline.
