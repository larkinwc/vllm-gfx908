# BENCH — FP16 Qwen3.5-9B (gfx900 / Vega10 / Radeon Pro V340)

This document is the gfx900 counterpart to the MI100 (`gfx908`) quant-kernel
benchmark docs. Because gfx900 (Vega10) has **no MFMA matrix cores**, the
INT8 (w8a8) and W4A16 quant kernels used in the MI100 grid do not run on this
arch; this grid therefore benchmarks the **FP16** model. It follows the same
methodology: `vllm bench serve`, server + `--max-concurrency`, reporting output
tok/s plus TTFT/TPOT p50/p99.

## Hardware / software manifest

| | |
|---|---|
| Host | `tyangpu1`, 8× AMD Radeon Pro V340 (= 16× gfx900, 8 GB VRAM each), dual Xeon E5-2640 v3 |
| OS | Ubuntu 24.04 |
| ROCm | 7.2.4 |
| PyTorch | 2.12.0+rocm7.2 |
| Triton | 3.7.0 (custom `triton-gfx900` fork: dedicated GFX900 ISA family) |
| RCCL | custom build `--amdgpu_targets gfx900` |
| vLLM | 0.1.dev1+g07032c7ba (this fork, `gfx900-support`) |
| Model | `Qwen/Qwen3.5-9B` (hybrid Gated DeltaNet + full attention, BF16→FP16) |
| Attention backend | `TRITON_ATTN` (forced on gfx900; no MFMA paged-attn) |
| All-reduce | `PYNCCL` over custom gfx900 RCCL (XGMI custom-AR disabled — Vega10 has no XGMI) |

## Run configuration (uniform across all cells)

```
dtype=float16            # gfx900 has no native BF16
enforce_eager=True       # no CUDA/HIP graphs
language_model_only=True # skip vision encoder (text-only)
max_model_len=4096
gpu_memory_utilization=0.90
seed=42, request_rate=inf
num_prompts = 8 * concurrency
percentile_metrics = ttft,tpot,itl,e2el @ p50,p99
```

- **synthetic**: `--dataset-name random --random-input-len 1024 --random-output-len 256 --ignore-eos`
- **coding**: `--dataset-name sharegpt` (ShareGPT V3, conversational; variable output length, EOS honored)
- **TP=8** uses GPUs 0-7 (a single CPU socket / NUMA node — keeps the tensor-parallel
  all-reduce off the ~0.26 GB/s cross-socket QPI link).
- **TP=4** uses GPUs 0-3.

> First-run note: Triton autotunes/JIT-compiles ~290 GDN/FLA kernels on the
> first server start of each TP shape (the TP=8 set was already warm from a
> prior run; the TP=4 server cold-compiled its shard shapes, taking ~10 min to
> become ready). Kernels are disk-cached, so subsequent starts are fast.

## Grid results (8 cells)

Output tok/s is per-stream summed across in-flight requests. TTFT/TPOT are
per-request p50/p99.

| TP | c | Workload | Out tok/s | Total tok/s | Req/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) |
|---:|--:|----------|----------:|------------:|------:|--------------:|--------------:|--------------:|--------------:|
| 8 | 1 | synthetic | 7.94 | 39.70 | 0.031 | 823.0 | 1260.8 | 120.41 | 141.44 |
| 8 | 1 | coding | 6.00 | 8.51 | 0.018 | 231.6 | 633.0 | 175.34 | 184.26 |
| 8 | 4 | synthetic | 19.54 | 97.72 | 0.076 | 3394.2 | 5133.5 | 194.07 | 208.69 |
| 8 | 4 | coding | 20.50 | 45.10 | 0.093 | 791.3 | 1778.4 | 190.87 | 214.18 |
| 4 | 1 | synthetic | 4.58 | 22.90 | 0.018 | 1953.0 | 3295.8 | 198.68 | 292.90 |
| 4 | 1 | coding | 3.68 | 5.21 | 0.011 | 497.4 | 1392.2 | 273.59 | 275.35 |
| 4 | 4 | synthetic | 12.23 | 61.13 | 0.048 | 5761.5 | 8850.8 | 309.45 | 343.46 |
| 4 | 4 | coding | 13.58 | 29.96 | 0.062 | 1445.3 | 3115.0 | 281.92 | 322.00 |

## Observations

- **Concurrency scales throughput**: c=1→c=4 gives ~2.5–3.4× output tok/s
  (TP8 synthetic 7.94→19.54 = 2.46×; TP8 coding 6.00→20.50 = 3.42×), at the
  cost of higher TTFT (queueing) — the expected batching trade-off.
- **TP=8 vs TP=4**: doubling GPUs improves single-stream output tok/s ~1.6–1.7×
  (synthetic c1 4.58→7.94; coding c1 3.68→6.00) and roughly halves TPOT
  (synthetic c1 198.7→120.4 ms). Sub-linear because TP adds an all-reduce per
  layer over PCIe (no XGMI on Vega10) and the GDN recurrence is partly serial.
- **TPOT is the headline cost of no MFMA**: ~120 ms/token best case (TP8 c1).
  FP16 GEMMs run on the VALU path instead of matrix cores, so decode is compute-
  bound. This is the fundamental gfx900 ceiling; the MI100 (gfx908) numbers in
  the sibling docs are far faster precisely because of MFMA + (there) quant.
- **coding (ShareGPT) vs synthetic**: lower total tok/s because outputs are
  shorter / EOS-terminated and prompt lengths vary, but per-request TTFT is
  lower at c=1 (shorter prompts than the fixed 1024-token synthetic input).

## Reproduce

Harness: `~/gpubench/bench_gfx900/` on `tyangpu1`. Per-cell raw JSON +
`vllm bench serve` logs are saved alongside. Re-run:

```
/tmp/gfx900_bench.sh   # starts a TP=8 then TP=4 server, runs the 4 cells each
```

---

# Pipeline-parallel grid (TP2 × PP{2,4})

Motivation: tensor parallelism issues an all-reduce on *every* layer, so it is
very sensitive to interconnect bandwidth. **Pipeline parallelism** instead only
passes the layer activations point-to-point between adjacent stages, so it
tolerates a slow link far better. On this box the cross-socket QPI link is only
~0.26 GB/s (a hard Xeon E5 v3 limit), so all cells here are kept **within
socket0 (GPUs 0-7)** to make a clean apples-to-apples comparison against the TP
grid above. The two layouts use the same GPU counts as the TP cells:

- **TP2×PP2 = 4 GPUs** (GPUs 0-3) — compare to **TP4**
- **TP2×PP4 = 8 GPUs** (GPUs 0-7) — compare to **TP8**

Same run config as the TP grid (FP16, enforce_eager, language_model_only,
max_model_len=4096, gpu_util=0.90, num_prompts=8×c, synthetic=random 1024/256
`--ignore-eos`, coding=ShareGPT).

> **Required fix for PP on gfx900:** the V1 *async-scheduling* path has the last
> PP rank broadcast its sampled token IDs back to rank 0 via a raw
> `torch.distributed.broadcast` (`gpu_model_runner.py:_pp_broadcast_prev_sampled_token_ids`).
> On our custom gfx900 RCCL this broadcast throws `unhandled cuda error`, killing
> the engine (`EngineDeadError`) — *only* on the server's async path; the offline
> `LLM.generate` path worked fine. Running the server with **`--no-async-scheduling`**
> skips that broadcast and PP runs cleanly. (Offline `LLM(...)` does not need the
> flag.)

## Results (8 cells)

| Layout | GPUs | c | Workload | Out tok/s | Total tok/s | Req/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) |
|--------|-----:|--:|----------|----------:|------------:|------:|--------------:|--------------:|--------------:|--------------:|
| TP2xPP4 | 8 | 1 | synthetic | 6.95 | 34.73 | 0.027 | 2180.9 | 4943.8 | 133.70 | 141.11 |
| TP2xPP4 | 8 | 1 | coding | 7.42 | 10.52 | 0.022 | 267.9 | 1568.5 | 133.74 | 134.12 |
| TP2xPP4 | 8 | 4 | synthetic | 22.87 | 114.37 | 0.089 | 4638.6 | 7971.0 | 157.96 | 167.02 |
| TP2xPP4 | 8 | 4 | coding | 25.69 | 54.94 | 0.111 | 1170.0 | 2141.8 | 140.61 | 154.73 |
| TP2xPP2 | 4 | 1 | synthetic | 6.81 | 34.04 | 0.027 | 2289.1 | 2981.2 | 137.93 | 141.67 |
| TP2xPP2 | 4 | 1 | coding | 6.99 | 9.91 | 0.021 | 281.2 | 1622.8 | 142.72 | 143.46 |
| TP2xPP2 | 4 | 4 | synthetic | 18.14 | 90.68 | 0.071 | 6533.2 | 9623.7 | 186.10 | 273.54 |
| TP2xPP2 | 4 | 4 | coding | 23.69 | 51.72 | 0.106 | 1238.5 | 2566.0 | 156.15 | 198.46 |

## TP vs PP head-to-head (same GPU count, same socket)

Output tok/s:

| GPUs | c | Workload | TP (pure) | TP2×PP | Winner |
|-----:|--:|----------|----------:|-------:|--------|
| 4 | 1 | synthetic | 4.58 | 6.81 | **PP +49%** |
| 4 | 1 | coding | 3.68 | 6.99 | **PP +90%** |
| 4 | 4 | synthetic | 12.23 | 18.14 | **PP +48%** |
| 4 | 4 | coding | 13.58 | 23.69 | **PP +74%** |
| 8 | 1 | synthetic | 7.94 | 6.95 | TP +14% |
| 8 | 1 | coding | 6.00 | 7.42 | **PP +24%** |
| 8 | 4 | synthetic | 19.54 | 22.87 | **PP +17%** |
| 8 | 4 | coding | 20.50 | 25.69 | **PP +25%** |

## Observations

- **At 4 GPUs, TP2×PP2 beats pure TP4 across the board (+48% to +90%).** Pure
  TP4 splits each GEMM 4 ways, so each gfx900 does a small, inefficient slice of
  a matrix-core-less FP16 GEMM and then pays a 4-way all-reduce per layer over
  PCIe. TP2×PP2 keeps GEMMs at a 2-way split (better VALU utilization) and
  replaces the wide all-reduce with cheap stage-to-stage activation hand-offs.
- **At 8 GPUs, TP2×PP4 wins on 3 of 4 cells**, losing only single-stream
  synthetic (where PP's pipeline can't fill at c=1, so its 4 serial stages add
  latency without a throughput payoff). With any concurrency (c=4) the pipeline
  fills and PP pulls ahead (+17–25%).
- **TPOT is markedly steadier under PP**: TP2×PP4 holds ~134–158 ms across all
  cells, whereas pure TP8 ranged 120–194 ms and pure TP4 198–309 ms. Less
  per-layer collective traffic means less variance.
- **TTFT is higher for PP at c≥1 synthetic** (long 1024-token prefill must walk
  all pipeline stages before the first token), the expected PP latency tax. For
  the shorter-prompt coding workload TTFT stays low.
- **Takeaway for this box**: with no XGMI and weak PCIe/QPI, the lowest-TP /
  higher-PP layout that still fits the model is the throughput sweet spot.
  TP2×PP4 (8 GPUs, one socket) is the best concurrent-serving config measured,
  and notably **PP is the layout that makes spanning both sockets viable** since
  it only needs point-to-point sends across the slow QPI link rather than a
  bandwidth-hungry all-reduce.

## Reproduce (PP grid)

```
/tmp/gfx900_bench_pp.sh   # TP2xPP4 then TP2xPP2 server (note --no-async-scheduling), 4 cells each
```
