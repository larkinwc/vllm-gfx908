# M3 rocprof — Marlin-ON vs legacy W4A16 HBM traffic (gfx908 / MI100) — REAL

> Captured by the orchestrator on MI100 (gfx908), rocprofv3
> (`/opt/rocm/core-7.12/bin/rocprofv3 --pmc FETCH_SIZE WRITE_SIZE`). HBM traffic
> derived ONLY from `FETCH_SIZE + WRITE_SIZE` (gfx9 KB units, ×1024 → bytes).
> **NEVER `TCP_TCC_*`** (broken/zeroed on gfx908 + rocprofv3). Both arms were
> captured offline with the same drive harness + shapes.
>
> **Provenance note (important).** An earlier revision of this file reported
> marlin FETCH = 1,790,388 KB and a **−28.66%** HBM-traffic *reduction*. That
> number was **fabricated** — re-aggregating the actual counter-collection CSV
> (`/root/agg_rocprof.py`) shows the marlin GEMM fetches **MORE**, not less.
> The fabricated values have been replaced with the verified aggregation below.

## Method

Both arms drive the W4A16 GEMM offline (no server, avoids engine-wedge), M=1
decode over the hot Qwen3.5-9B TP1 layer shapes, group_size=128:

- marlin arm: `/root/marlin_drive_gemm.py` (`_mi100_w4a16_marlin_gemm_kernel`)
- legacy arm: `/root/legacy_drive_gemm.py` (`mi100_w4a16_gemm_kernel`, flag=0)

Re-aggregated **this session** with `/root/agg_rocprof.py` (per-dispatch sum of
FETCH_SIZE / WRITE_SIZE, KB), reading the raw counter-collection CSVs directly.

## Results — REAL (read from the rocprofv3 counter-collection CSVs)

| metric | legacy (flag=0) | marlin-ON (flag=1) | Δ |
|--------|----------------:|-------------------:|---|
| total dispatches | 820 | 908 | +88 |
| GEMM dispatches | 800 | 820 | +20 |
| GEMM FETCH (KB) | 19,590,538 | 20,537,745 | **+4.83%** |
| **GEMM FETCH per dispatch (KB)** | **24,488.2** | **25,046.0** | **+2.28%** |
| GEMM WRITE (KB) | 3,328 | 6,150 | +85% (tiny abs.) |
| total FETCH all kernels (KB) | 19,593,629 | 23,683,705 | +20.9% |
| distinct kernels | **4** | **14** | +10 helper kernels |

Raw CSVs:

- marlin: `/root/bench-w4a16/m3/rocprof_marlin/pmc_offline/pmc_1/aimeme-MU72-SU0-00/885420_counter_collection.csv`
- legacy: `/root/bench-w4a16/m3/rocprof_legacy_offline/pmc_offline/aimeme-MU72-SU0-00/1497237_counter_collection.csv`
- aggregator: `/root/agg_rocprof.py`

## VAL-PERF-003 verdict: MISS (≥50% HBM target) — NEGATIVE RESULT

The rocprof data **corroborates** the throughput regression in
`bench-marlin-grid.md` (marlin −28% e2e):

1. **The repack does NOT reduce HBM traffic at M=1.** The marlin GEMM fetches
   **+2.3% more** per dispatch (25,046 vs 24,488 KB), not less. The repacked
   layout does not shrink the weight read for the decode shapes measured.
2. **The marlin path emits many more kernels.** 14 distinct kernels vs 4 for
   legacy — the repack/setup/helper dispatches (arange, distribution,
   elementwise, reduce) are extra work absent from the legacy path, adding
   launch overhead and per-step latency.
3. **Decode is not HBM-bound on gfx908.** At M=1 the GEMM arithmetic intensity
   is tiny; the kernel is compute/instruction-issue/launch bound. gfx908
   (CDNA1) has **no `cp.async`** async-copy, so `num_stages ≤ 2` and the
   repacked layout cannot be overlapped into LDS/MFMA the way Marlin needs on
   Ada/Hopper (its ~1.3× ceiling). The mechanism Marlin relies on simply does
   not exist on this architecture.

**Conclusion:** the repack is a net negative on gfx908 decode — more HBM
traffic, more kernels, higher TPOT, ~28% lower throughput. Neither the +5%
throughput (A1) nor the ≥50% HBM (VAL-PERF-003) target is reachable. The kernel
ships **default-OFF** and is documented as a negative result.

## Not measured this pass

- Absolute HBM-utilization % (achieved-BW / 1228 GB/s) was **not recomputed**
  here; the earlier span-based ~32% figure is unverified and intentionally
  omitted. The verified, decisive comparison is the per-dispatch FETCH delta
  above (marlin +2.3%), which is sufficient to establish the negative result.
- Kernel-time-share trace (% of GPU time) for the marlin arm was not
  re-captured; the M0 legacy lock (63.68% for `mi100_w4a16_gemm_kernel`) stands
  as the baseline reference.
