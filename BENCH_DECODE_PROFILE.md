# Decode-step profile — gfx900 TP4, FP16 Qwen3.5-9B (issue #56)

torch profiler over a clean 127-step decode-only window (TP4, one socket, eager,
bs=1). Per-rank GPU kernel time bucketed by class. This **corrects** the earlier
assumption (carried in older PERF_GFX900 notes) that decode was "RCCL all-reduce +
GDN kernel" bound.

## Result — where the 152.5 ms/step actually goes

| Component | ms/step | % of wall | % of GPU-busy |
|---|---:|---:|---:|
| **GEMM (M=1, rocBLAS `Cijk_..MT64x64x4`, padded)** | **65.9** | **43%** | **88%** |
| **GPU-idle / launch-dispatch gap** | **77.9** | **51%** | — |
| RCCL all-reduce (`ncclDevKernel` + `vllm::all_reduce`) | 3.1 | 2% | 4% |
| elementwise / copy | 1.9 | 1.3% | 2.6% |
| GDN linear-attn (fused_recurrent + conv + norm) | 0.4 | 0.3% | 0.5% |
| RMSNorm | 0.2 | 0.1% | 0.3% |
| Full attention (`unified_attention`) | 0.2 | 0.1% | 0.2% |
| SiLU/act | 0.1 | 0.1% | 0.2% |

GPU-busy/step = 74.6 ms; wall/step = 152.5 ms → **51% of every decode step the GPU
is idle**, waiting on host-side per-kernel dispatch (eager mode).

## Two real levers (both different from the old assumption)

### 1. Padded M=1 GEMMs — 43% of wall, running at 20% of the bandwidth roofline
The decode matmuls go through rocBLAS `Cijk_Alik_Bljk_HHS_BH_MT64x64x4` — a
**64×64 macrotile** kernel. A 1-row GEMV is padded to 64 rows, so ~63/64 of the
issued work is waste. Bandwidth floor for the per-GPU weight read (4.83 GB / 365
GB/s) is **13.2 ms/step**, but we spend **65.9 ms** → only **20% of roofline**, the
gap being padding overhead, not bandwidth.

This is exactly the failure mode our int4 GEMV (#57) fixed for the MLP by replacing
`tl.dot` with a real M≤8 vector GEMV. The profile says the same fix applied to the
**remaining FP16 GEMMs** is the single highest-value kernel work:
- **lm_head** (#59): vocab 248320×4096, FP16, every token — biggest single GEMM.
- **attention / linear-attn projections** (#61): q/k/v/o + GDN in/out projections,
  left in FP16 even under AWQ.

A general **FP16 M≤8 GEMV** (not just int4) would cut into the 43% directly.

### 2. The 51% GPU-idle dispatch gap
Over half the step is GPU idle between launches. This is host dispatch overhead in
eager mode. Tension to resolve: HIP CUDA graphs gave **zero** decode speedup in our
earlier A/B (PERF_GFX900 § graphs) despite this large gap — likely because the
GDN/FLA Triton kernels can't be captured (mode=NONE forced) so the captured region
doesn't cover the hot path, or the gap is inter-kernel host latency the partial
capture didn't remove. Worth a focused re-test: which kernels actually sit inside
the idle gap, and whether fewer/fused launches (combo kernels, reduced op count)
shrink it. Lower-risk than it sounds — it's a measurement question first.

## Demoted (do NOT build these)
- **Full-attention decode kernel (#60)**: 0.2 ms/step (0.1%). Not worth it. Close.
- **RCCL split / comm tuning (part of #56)**: 3.1 ms/step (2%). Not the bottleneck.
- **GDN decode kernel**: 0.4 ms/step. Already fine (uses fused_recurrent, 0 tl.dot).

## Re-ranked candidate list (by measured payoff)
1. **FP16 / int GEMV for lm_head (#59)** — biggest single padded GEMM, certain win.
2. **GEMV for attn + GDN projections (#61)** — the rest of the 43%.
3. **Investigate the 51% dispatch gap** — measurement first; potential large win but
   uncertain (graphs already failed once).
4. ~~Full-attention kernel (#60)~~ — close, negligible.
5. ~~RCCL tuning~~ — negligible.

## Repro
`bench_scripts/prof_decode.py` (writes torch traces to ~/decode_prof),
`bench_scripts/parse_trace.py` (buckets GPU kernel time by class).
