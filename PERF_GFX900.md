# Aggregate compute / bandwidth + power tuning — gfx900 (V340 / Vega10)

Companion to `BENCH_GFX900.md`. That doc measures end-to-end LLM serving; this
one measures the **raw hardware ceiling** (FP16 GEMM TFLOPS + HBM2 bandwidth),
how it **aggregates across the 16 GPUs**, and how it responds to **power-cap
tuning** — to answer "what do I actually get in aggregate, and what's the
efficient operating point?"

Microbenchmarks: `/tmp/gpu_peak.py` (matmul TFLOPS + `copy_` bandwidth, one
process per GPU run concurrently), `/tmp/power_bench.py` + `/tmp/power_scan.sh`
(sustained load while sweeping the power cap). Harness output in
`~/gpubench/peak_results/` and `~/gpubench/power_scan/`.

## Theoretical per-GPU ceiling (gfx900, 56 CU @ 1500 MHz)

| Metric | Formula | Peak |
|--------|---------|-----:|
| FP32 | 56 CU × 64 lanes × 2 (FMA) × 1.5 GHz | **10.8 TFLOPS** |
| FP16 (packed) | 2× FP32 via `v_pk_fma_f16` | **21.5 TFLOPS** |
| HBM2 | 2048-bit × 945 MHz × 2 | **~483 GB/s** |
| Power cap | (writable via sysfs / rocm-smi) | 110 W |

## Measured per-GPU (n=8192 FP16 matmul, single cold GPU)

| Metric | Measured | % of peak |
|--------|---------:|----------:|
| FP16 TFLOPS (cold burst) | ~5.0 | 23% |
| FP16 TFLOPS (sustained) | **~4.5** | 21% |
| FP32 TFLOPS | ~4.2 | 39% |
| HBM bandwidth | **~365 GB/s** | 76% |

**Why FP16 is only ~21% of peak:** Vega10 reaches its 21.5 TFLOPS number only
with *packed* FP16 (`v_pk_fma_f16`, two FP16 FMAs per lane per clock). The
rocBLAS kernels selected for this `hgemm` path on gfx900 do **not** emit packed
math — they run essentially at the FP32 rate (note FP16 ≈ FP32 ≈ 4–5 TFLOPS).
This is the same root cause as the slow decode TPOT in `BENCH_GFX900.md`: no
matrix cores (MFMA) and no packed-FP16 GEMM, so FP16 lives on the regular VALU
at scalar-FP32 throughput. HBM, by contrast, hits a healthy 76% of peak.

## Aggregate (concurrent, one process per GPU)

Each GPU works in its own VRAM, so there is **no interconnect traffic** in this
test — it measures pure parallel compute/bandwidth headroom (the opposite end
from the TP all-reduce-bound serving numbers).

| Scope | GPUs | FP16 TFLOPS | HBM GB/s | Scaling |
|-------|-----:|------------:|---------:|--------:|
| Single | 1 | 4.5 | 365 | 1.0× |
| Socket0 | 8 | **38.3** | **2 917** | 7.8× (97%) |
| Full box | 16 | **76.4** | **5 783** | 16.0× (linear) |

Scaling is essentially linear because the workloads are independent — confirming
the box has ~**76 TFLOPS FP16** and ~**5.8 TB/s** of aggregate memory bandwidth
to give, *if* a workload can be partitioned to avoid the slow PCIe/QPI fabric
(i.e. data/replica parallel, or pipeline parallel — see `BENCH_GFX900.md`, where
TP2×PP layouts beat pure TP precisely by avoiding per-layer all-reduce).

> Caveat: in real TP serving you will **not** see 76 TFLOPS — that number is the
> embarrassingly-parallel ceiling. TP shares one model and pays all-reduce every
> layer over PCIe, so effective utilisation is far lower; PP recovers a chunk of
> it by trading all-reduce for cheap point-to-point sends.

## Power-cap scan (single GPU, sustained FP16, n=8192)

| Cap (W) | TFLOPS | Mean draw (W) | TFLOPS/W |
|--------:|-------:|--------------:|---------:|
| 110 | 4.48 | 85.0 | 0.0527 |
| 100 | 4.45 | 85.5 | 0.0521 |
| 90 | 4.44 | 81.5 | 0.0545 |
| 80 | 4.44 | 80.0 | **0.0555** |
| 70 | 4.44 | 80.8 | 0.0549 |
| 60 | 4.25 | 77.2 | 0.0551 |
| 50 | 3.10 | 60.5 | 0.0513 |

### What this tells us

- **The cards are compute-bound, not power-bound, at stock.** A flat-out FP16
  GEMM draws only **~85 W** against the 110 W cap. The chip simply can't burn
  110 W on VALU FP16 (no matrix cores lighting up), so the top 25 W of the cap
  is never used.
- **You can cap to 80 W with zero throughput loss** — TFLOPS is identical
  (4.44) from 110 W down to 70 W. Peak efficiency is at **80 W: 0.0555 TFLOPS/W,
  ~5% better than stock** while saving 30 W of cap headroom (and the real ~5 W
  of draw).
- **The knee is ~70 W.** At 60 W throughput starts to dip (4.25), and 50 W
  forces a real clock cut (3.10 TFLOPS, −31%). So **70–80 W is the sweet spot.**
- **Aggregate implication:** capping all 16 GPUs at 80 W trims the box's power
  budget by 16 × 30 W = **480 W of cap** (and the cards already only pull ~85 W
  each, so you're mostly removing thermal/PSU headroom risk) for **no perf
  loss**, and a small efficiency gain.

## ⚠️ The real limiter is COOLING, not power

The V340 is a **passively-cooled** datacenter card (no onboard fan spinning —
`Fan RPM 0`); it relies entirely on chassis airflow. During back-to-back testing
one GPU heat-soaked to **junction 89–93 °C**, at which point it **throttled sclk
to the 300 MHz idle floor → sustained dropped from 4.5 to ~1.0 TFLOPS (−78%)**.
Idle GPUs sat at 24–33 °C; the hot one took minutes to recover (still 93 °C
after a minute idle).

**Consequences for inference:**

1. **Thermal throttling, not the power cap, will gate sustained multi-GPU
   serving.** A long TP/PP run that keeps all GPUs busy will heat-soak the whole
   tray; expect sustained throughput **below** the cold-burst numbers in
   `BENCH_GFX900.md` unless chassis airflow is strong.
2. **Lower power caps directly reduce heat.** Since 80 W costs no throughput but
   cuts ~5–8 W of draw and (more importantly) reduces the heat the cooling has
   to remove, **capping at 70–80 W is the single best knob to keep clocks up
   under sustained load** — i.e. a lower cap can yield *higher* sustained
   throughput once thermals dominate.
3. **Watch junction temp, not edge temp.** Throttle triggers off junction
   (~95–100 °C). Monitor `rocm-smi -t` (junction sensor) during long runs.

## Recommendations for inference on this box

1. **Cap all GPUs at 80 W** (`sudo rocm-smi --setpoweroverdrive 80`): no
   throughput loss, best TFLOPS/W, less heat → better sustained clocks.
2. **Prioritise airflow.** These passive cards need aggressive chassis fans;
   without it, sustained serving will thermally throttle regardless of caps.
3. **Prefer layouts that avoid the fabric** (DP/PP over wide TP) to actually
   approach the 76 TFLOPS / 5.8 TB/s aggregate — see the PP wins in
   `BENCH_GFX900.md`.
4. **Don't expect FP16 matrix-core throughput.** The honest per-GPU number is
   ~4.5 TFLOPS FP16 sustained; plan capacity around ~36 TFLOPS/socket, not the
   21.5×8 theoretical.

## Reproduce

```
/tmp/run_peak.sh fp16          # single / socket0 / all-16 aggregate
/tmp/power_scan.sh 1 2         # power-cap efficiency scan on GPU1 (card2)
```
