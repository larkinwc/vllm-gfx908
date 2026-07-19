
# The biggest decode win: enable LLMM1 skinny GEMV for gfx900 (issue #59)

> **Historical kernel evidence:** retain this LLMM1 result and the `wvSplitK`
> failure as shape-specific gfx900 evidence. It is not authorization to widen the
> dispatch gate: a future kernel change requires the post-systems profiler to show
> at least 5% non-RCCL GPU time, a ≥10% microbenchmark win, and a confirmed ≥3%
> end-to-end win through `scripts/gfx900`.

The decode profile (#56, BENCH_DECODE_PROFILE.md) showed M=1 GEMMs were 43% of decode
wall time, running at only 20% of the bandwidth roofline because rocBLAS pads a 1-row
GEMV to a 64x64 macrotile (`Cijk_..MT64x64x4`). Root cause found:

**gfx900 is excluded from `on_gfx9()`** (which only lists gfx908/gfx90a/gfx942/gfx950).
vLLM already ships hand-written skinny GEMV ops (`LLMM1`, `wvSplitK`) gated on
`on_gfx9() or on_gfx1x()`, so on gfx900 they were never used and every M=1 decode
matmul fell through to the padded rocBLAS path.

## Microbench (lm_head shape, M=1, k=4096, single gfx900)
| vocab shard | rocBLAS | LLMM1 | speedup | rel err |
|---|---:|---:|---:|---:|
| 62080 (TP4) | 5.33 ms | **1.46 ms** | **3.66x** | 5.5e-4 |
| 124160 (TP2) | 10.14 ms | 2.92 ms | 3.47x | 5.4e-4 |
| 248320 (TP1) | 20.21 ms | 5.83 ms | 3.46x | 5.0e-4 |

`LLMM1` is correct on wave64 and ~3.5x faster. **`wvSplitK` (the n>1 branch) hits a
device-side assertion and crashes the GPU on gfx900** (recovered cleanly thanks to
reset_method=2). So we enable only the `LLMM1` branch: n==1, m%4==0, k<=8192, bias=None.

## The fix
A narrow gfx900 block in `rocm_unquantized_gemm_impl` (vllm/model_executor/layers/utils.py)
that routes M=1 unquantized FP16 GEMMs through `LLMM1`. This covers **lm_head AND the
M=1 FP16 projections** (qkv / o / gate_up where k<=8192, m%4==0). down_proj (k=12288)
and any m not %4 stay on rocBLAS.

## End-to-end (Qwen3.5-9B, TP4, enforce_eager, bs=1 decode)
| | tok/s | ms/step |
|---|---:|---:|
| FP16, before | 6.65 | ~150 |
| **FP16, LLMM1** | **14.85** | **67.3** |

**2.2x decode speedup**, coherent output. Re-profile confirms the mechanism: GEMM time
**65.9 -> 6.5 ms/step (10x)** (5.46 ms LLMM1 + 1.0 ms remaining rocBLAS for down_proj /
gate_up). The padded-macrotile waste is gone.

## Compounding with the other gfx900 wins
lm_head stays FP16 even under AWQ, so LLMM1 also speeds up the int4 + TurboQuant stack:

| Config | decode tok/s |
|---|---:|
| AWQ int4 + TQ KV (before LLMM1) | 9.97 |
| **AWQ int4 + TQ KV + LLMM1** | **14.12** |

All three optimizations active and coherent (TP2, half the GPUs).

Repro: `bench_scripts/skinny_bench.py` (microbench), `bench_scripts/lmhead_e2e.py` (e2e).

## General lesson
Hardware that is "GFX9 ISA but no MFMA" (gfx900/Vega10) often falls outside vendor
fast-path gates written for MFMA-capable CDNA. Several of those fast paths
(`LLMM1` here) are pure ALU GEMVs that work fine on wave64 — they just need the device
let into the gate. Always check the existing skinny/GEMV ops before writing a new
kernel; here the win was a 1-condition gate, not new code. (Verify each op: `wvSplitK`
from the same family crashes — never enable a whole gate blind.)
