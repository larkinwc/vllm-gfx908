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

---

# Realistic scaling under LLM inference (TP / PP) — not the linear ceiling

The 76 TFLOPS / 5.8 TB/s aggregate above is the **embarrassingly-parallel**
ceiling: 16 independent jobs, each in its own VRAM, zero communication. Real
single-model LLM serving never reaches it. Below we back out the **effective**
realized compute and bandwidth from the actual serving runs in
`BENCH_GFX900.md` (Qwen3.5-9B, 9.65 B params, 19.3 GB FP16).

LLM inference has two regimes that scale very differently:

- **Prefill** (processing the prompt) is **compute-bound** — a big GEMM over all
  prompt tokens. From TTFT we recover effective **TFLOPS**.
- **Decode** (generating tokens) is **bandwidth-bound** — every step must stream
  all weights from HBM to emit (batch) tokens. From output tok/s we recover
  effective **GB/s**.

Formulas: prefill FLOPs ≈ `2 · P · prompt_tokens · batch` over the TTFT window;
decode bytes ≈ `2 · P` (FP16 weights) streamed per forward step, `steps/s =
out_tok_s / batch`.

## Prefill: effective TFLOPS vs the parallel ceiling

| Layout | GPUs | c | TTFT p50 | Eff TFLOPS | % of agg ceiling |
|--------|-----:|--:|---------:|-----------:|-----------------:|
| TP8 | 8 | 1 | 0.82 s | 24.0 | **67%** |
| TP8 | 8 | 4 | 3.39 s | 23.3 | 65% |
| TP4 | 4 | 4 | 5.76 s | 13.7 | 76% |
| TP4 | 4 | 1 | 1.95 s | 10.1 | 56% |
| TP2×PP2 | 4 | 4 | 6.53 s | 12.1 | 67% |
| TP2×PP4 | 8 | 4 | 4.64 s | 17.1 | 47% |
| TP2×PP4 | 8 | 1 | 2.18 s | 9.1 | 25% |

**Prefill realizes 50–76% of the parallel-compute ceiling.** It is the regime
that actually uses the GPUs' arithmetic, so wide **TP wins here** (TP8 hits the
best single-stream 24 TFLOPS effective). PP is weaker for single-stream prefill
because the prompt must walk all pipeline stages before the first token (the
25% outlier), but it recovers with concurrency as the pipeline fills.

## Decode: effective GB/s vs the bandwidth ceiling

| Layout | GPUs | c | out tok/s | Eff GB/s | % of agg HBM | % of bw-roofline |
|--------|-----:|--:|----------:|---------:|-------------:|-----------------:|
| TP2×PP2 | 4 | 1 | 6.81 | 131 | 9.0% | **9.0%** |
| TP4 | 4 | 1 | 4.58 | 88 | 6.1% | 6.1% |
| TP8 | 8 | 1 | 7.94 | 153 | 5.2% | 5.2% |
| TP2×PP4 | 8 | 1 | 6.95 | 134 | 4.6% | 4.6% |
| TP8 | 8 | 4 | 19.54 | 94/step | 3.2% | 3.2% |

A 19.3 GB model over ~2.9 TB/s of aggregate HBM gives a **bandwidth roofline of
~151 tok/s** (8 GPU, batch 1) if decode were purely memory-bound. We measure
**~5–9% of that.** Decode on this box is **NOT actually bandwidth-bound** — it
is **latency/overhead-bound**:

- No matrix cores + no packed-FP16, so each tiny per-token GEMM is slow on the
  VALU and **enforce_eager** (no HIP graphs — required for stability here) adds
  full kernel-launch overhead on every one of the ~hundreds of ops per layer.
- TP adds an all-reduce *per layer* over PCIe (no XGMI); at batch 1 this is pure
  serial latency the HBM never gets to hide.
- The hybrid Gated-DeltaNet recurrence is partly sequential.

So the HBM is **>90% idle during decode** — the bottleneck is per-token kernel
launch + collective latency, not memory bandwidth.

## Bottom line: what you realistically get

| Quantity | Parallel ceiling | **Realistic (this model, serving)** |
|----------|-----------------:|------------------------------------:|
| Prefill compute (8 GPU) | 36 TFLOPS | **~24 TFLOPS (≈67%)** |
| Prefill compute (4 GPU) | 18 TFLOPS | **~13 TFLOPS (≈70%)** |
| Decode bandwidth (8 GPU) | 2.9 TB/s | **~0.15 TB/s used (≈5%)** |
| Decode speed, best single-stream | — | **~8 tok/s (TP8)** |
| Decode speed, best aggregate | — | **~26 tok/s (TP2×PP4, c4 coding)** |

**Takeaways**

1. **Prefill scales well (~70% efficiency)** and is TP-friendly — that is the
   part of the box that behaves like the aggregate spec.
2. **Decode is the bottleneck and does *not* scale with bandwidth** — it is
   gated by per-token kernel-launch + all-reduce latency, so it sits at ~5–9% of
   the HBM roofline. Throwing more GPUs at one stream barely helps (TP8 only
   ~1.7× TP4); **batching/concurrency is the only real decode lever** (c1→c4
   gives 2.5–3.4×).
3. **PP > TP for decode at equal GPUs** because it removes the per-layer
   all-reduce that dominates decode latency — exactly the TP2×PP wins in
   `BENCH_GFX900.md`.
4. **Realistic planning number:** count on **~24 TFLOPS/socket of usable prefill
   compute** and **decode throughput in the tens of tok/s aggregate**, *not* the
   76 TFLOPS / 5.8 TB/s parallel figure. The two regimes want opposite layouts
   (TP for prefill, PP/low-TP for decode), so the best serving config is a
   compromise — here TP2×PP4 within one socket.

## Reproduce

```
python3 bench_scripts/realistic.py   # back-of-envelope from BENCH_GFX900 numbers
```

---

# HIP CUDA-graph investigation (the `enforce_eager` question)

`BENCH_GFX900.md` shows decode is **latency-bound**, not bandwidth-bound: it
realizes only ~5–9% of the HBM roofline, with a ~120 ms/token floor. The leading
suspect was per-token **kernel-launch overhead** under `enforce_eager` (no
graphs), so we tried to enable HIP/CUDA graphs to collapse the hundreds of
per-layer kernel launches into a single graph replay. **Result: graphs are not
usable on this gfx900 stack.** Details below so nobody burns time (or a GPU)
re-treading this.

### Layer 1 — `enforce_eager=False` defaults to torch.compile, which crashes

With graphs left on (default), vLLM picks `mode=VLLM_COMPILE` (Inductor) +
`cudagraph_mode=FULL_AND_PIECEWISE`. Inductor immediately fails on ROCm:

```
RuntimeError: torch.* op returned non-Tensor bool
  target: torch.cuda.is_current_stream_capturing
```

This is the same reason the gfx908/MI100 path force-sets `mode=NONE`: Inductor
fusions aren't available/working on ROCm here. **torch.compile is a dead end on
gfx900.**

### Layer 2 — pure HIP graphs (no Inductor) hard-hang the GPU

The correct config to isolate graphs from compile is
`mode=NONE` + `cudagraph_mode=FULL_DECODE_ONLY`. The model loads, but during the
graph-capture warmup at **TP=8** the hardware hung and the driver's recovery
**failed**:

```
amdgpu 0000:25:00.0: KCQ enable failed
resume of IP block <gfx_v9_0> failed -110
GPU reset end with ret = -22          ← reset FAILED, not recovered
VRAM is lost due to GPU reset!
[powerplay] No response from smu / fw load failed
[powerplay] firmware(0x1) doesn't match SMU9_DRIVER_IF_VERSION(0xe)
```

`rocm-smi` then enumerated only **15/16 GPUs** — the hung GPU's SMU (power
microcontroller) was wedged and **only a full host reboot brought it back**.
The stale SMU firmware (`0x1` vs expected `0xe`) means the in-driver GPU-reset
path cannot recover a Vega10 hang on this box, so a graph-capture hang escalates
to a dead GPU until reboot.

### Layer 3 — even the safest settings never finish (infinite autotune)

After rebooting (all 16 GPUs healthy again), we retried with the most
conservative settings possible: **TP=4, single socket, `cudagraph_capture_sizes=[1]`,
`max_num_batched_tokens=2048`, 600 s hard timeout.** This time **no GPU hang**
(dmesg clean), but init **never completed** — the worker sat in
`triton/backends/amd/compiler.py make_amdgcn` → `autotuner do_bench`,
re-JIT-compiling kernels for the capture shapes without ever converging
(`~/.triton/cache` `.hsaco` count frozen at 426 for >8 min). It hit the 600 s
timeout still in warmup. The graph-capture code path re-triggers Triton
autotuning for the GDN/FLA kernels in a way that does not terminate in
reasonable time on gfx900.

### Conclusion

**`enforce_eager=True` is correct and required on gfx900 for this stack.** HIP
CUDA graphs are not viable here:

1. torch.compile (Inductor) is broken on ROCm → `mode=NONE` mandatory;
2. full graph capture can hard-hang the GPU, and the SMU-firmware mismatch makes
   the driver reset path fail → **a hang = a dead GPU until reboot** (risky);
3. even the minimal-footprint capture never escapes Triton autotune.

So the ~120 ms/token decode floor is **not** simply removable launch overhead —
on this hardware the eager path is the only stable one, and the per-token cost
is intrinsic (no MFMA, no packed-FP16, per-layer PCIe all-reduce). The
realistic levers remain the ones already validated in `BENCH_GFX900.md`:
**pipeline parallelism over wide TP** (removes per-layer all-reduce) and
**batching/concurrency** (amortizes fixed per-step latency). A future ROCm with
working ROCm-Inductor and updated Vega10 SMU firmware would be the prerequisite
to revisit graphs.

### Repro / guardrails

If anyone retries graphs on gfx900: use a **hard `timeout`**, watch
`dmesg | grep amdgpu` for `KCQ enable failed` / `GPU reset`, and be ready to
**reboot** — do not run it unattended across all 16 GPUs.
