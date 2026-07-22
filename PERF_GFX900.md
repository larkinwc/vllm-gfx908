# Aggregate compute / bandwidth + power tuning — gfx900 (V340 / Vega10)

Companion to `BENCH_GFX900.md`. That doc measures end-to-end LLM serving; this
one measures the **raw hardware ceiling** (FP16 GEMM TFLOPS + HBM2 bandwidth),
how it **aggregates across the 16 GPUs**, and how it responds to **power-cap
tuning** — to answer "what do I actually get in aggregate, and what's the
efficient operating point?"

> **Historical evidence notice:** this report predates the reproducible gfx900
> manifest/matrix harness. Its power, topology, eager/graph, and TP/PP conclusions
> are tied to the recorded host and code era. New comparisons must use
> `scripts/gfx900`, retain raw trials, and reject different `platform_sha256`
> values; the harness records thermal telemetry but does not change power caps.

## 2026-07 c4130-2 reproducibility campaign addendum

The historical measurements below share platform digest
`61835f7caf7bf4057f4314e0d5f669c935e5d1ae5cbb83120745d5339e76bf36`.
The clean, commit-validated campaign source is
`689cbbba3ae5400bd583e1435177464e48f1f94a`. Its one-pass dense-graph screen
was evidence only until its independently launched confirmation trial
(`confirm-dense-eager-c1` / `confirm-graphs-full-decode-c1`) completed: 3
independent launches per arm, both cells PASS, zero failed requests, and a
mechanized `compare-cells` verdict of `IMPROVEMENT` (see the table row
below). CUDAGraph capture changes no numerics for this eager-vs-graph
comparison — it has no perplexity/coding/needle quality-gate requirement in
this campaign's promotion framework, so the throughput/latency capacity
confirmation is the complete gate for this result. The TurboQuant capacity
screen's independent-launch confirmation trial has since completed (see the
capacity row below): the throughput/latency capacity gates are confirmed and
passing. TurboQuant's quality gates (perplexity, coding suite,
needle-in-haystack) have since been reconfirmed on this clean substrate too,
at source commit `0054f988bcca5d50bcfaddf2ff7545246df6e8ba` (2026-07-21, see
the table row and detail below) — all three **PASS**. Combining the capacity
and quality results into a single promoted default/profile is still a
separate write-up decision, not made by this note.

| Screen | Measured result | Decision |
|---|---|---|
| RCCL TP4/TP8, 1 KiB–8 MiB, default/Ring/Tree/LL/Simple | Default RCCL 2.27.7 was the only no-regression choice | No static RCCL override. |
| FP16 topology, 4,096/256, c=8 | TP8: 26.476 output tok/s; TP4: 20.895; TP4×PP2: 26.304. **Confirmed** (3 independent launches per config, all PASS, 0 failed requests): TP8 26.5186 (Δ+0.161%); TP4 21.0840 (Δ+0.903%); TP4×PP2 26.3448 (Δ+0.154%), each `compare-cells` verdict `PASS` against its own screen. | Keep separate topology observations; no universal replacement for TP8 eager. Per-config reproducibility confirmed 2026-07-21/22 — not a cross-config ranking; see addendum below. |
| `FULL_DECODE_ONLY`, 4,096/256, c=8 | 26.551 versus 26.476 output tok/s eager (+0.28%) | Declined for the prefill-heavy reference workload. |
| `FULL_DECODE_ONLY`, 512/512, c=1 | Historical: 58.18 versus 12.73 output tok/s eager (+357.1%); clean single-launch screen: 57.651 versus 12.707. **Confirmed** (3 independent launches per arm, both cells PASS, 0 failed requests): `confirm-dense-eager-c1` (eager) 12.724873664265084 versus `confirm-graphs-full-decode-c1` (`FULL_DECODE_ONLY`) 56.336206381615746 output tok/s (+342.725%); p99 TPOT 79.0647771127532 versus 16.297473464836184 ms (-79.387%); p99 TTFT 772.2754819784313 versus 781.2450528336922 ms (+1.161%, within the 2% gate). `compare-cells` verdict: `IMPROVEMENT`, zero regressions. | Throughput/latency capacity gates confirmed and passing. CUDAGraph capture changes no numerics here, so there is no separate quality-gate requirement (unlike TurboQuant's KV-cache-dtype quantization) — promote `FULL_DECODE_ONLY` for this low-concurrency, decode-dominant c=1 workload. |
| `FULL_DECODE_ONLY`, 512/512, c=32 | 190.51 versus 188.33 output tok/s eager (+1.16%); p99 TPOT 163.09 versus 165.01 ms | Do not enable for throughput batching alone. Repeated scheduler rows show an unpadded size-32 `FULL` graph, so this is not a fallback artifact. |
| AWQ TP4, `FULL_DECODE_ONLY`, 512/512, c=1 | Historical: graph 39.637 versus eager 10.629 output tok/s. On clean source, eager started only with text-only multimodal limits and `--max-num-batched-tokens 512` (10.677 output tok/s); tested VLLM_COMPILE paths hung during warmup. | Do not enable AWQ graphs on gfx900. |
| TP8 DecodeBenchConnector burst, 4,096/256, 0.0827 RPS, burstiness 0.25 | Graph 18.119 versus eager 18.064 output tok/s (+0.31%); p99 TPOT 126.93 versus 129.48 ms; p99 TTFT 2,431 versus 1,621 ms | Do not enable graph mode for burst/open-loop traffic: p99 TTFT fails the +2% gate. |
| TurboQuant versus auto, 8,192/256, c=32 | Historical: 22.092 versus 18.450 output tok/s (+19.7%). Clean single-launch screen: 20.521 versus 18.583 (+10.4%). **Confirmed** (3 independent launches per arm, all PASS, 0 failed requests): 22.505 versus 18.600 output tok/s (+20.997%); p99 TPOT 1377.553 versus 1672.270 ms (-17.624%); p99 TTFT 319853.228 versus 379302.669 ms (-15.673%). `compare-cells` verdict: `IMPROVEMENT`, zero regressions. | Throughput/latency capacity gates confirmed and passing. Quality gates (perplexity, coding, needle-in-haystack) reconfirmed on the clean substrate 2026-07-21: cache-read PPL Δ -0.10504% (gate ≤+1% PASS), coding suite 8/10 both arms with identical prompt-id-level pass/fail sets (no new failure), needle@32k 5/5 both arms (all depths). All three quality gates PASS. Combining capacity+quality into a promoted default is still a separate write-up decision.
| Prefix-repetition, TurboQuant, c=8 | Cache-on 36.029 output tok/s, 82.8% hit; cache-off 12.740 | Promote prefix caching only for repeated-prefix workloads. |
| MTP K=1 / CPU ngram K=4 | 22.376 / 24.007 output tok/s versus 26.476 eager | `DECLINED_NO_END_TO_END_WIN`. |

The matched 32,768-context needle control passed all five depths for both auto
and TurboQuant. It used a 34,816-token single-sequence server with
`--max-num-batched-tokens 4096` and `--gpu-memory-utilization 0.75`; an initial
0.85 TurboQuant attempt failed with `torch.OutOfMemoryError` during the prefill
attention activation. The matched coding suite was 8/10 in both modes with the
same two environmental/extractor failures, so it found no new TurboQuant
failure.

**Clean-substrate quality reconfirmation (2026-07-21, source commit
`0054f988bcca5d50bcfaddf2ff7545246df6e8ba`, `platform_sha256` reconfirmed
matching `61835f7caf7bf4057f4314e0d5f669c935e5d1ae5cbb83120745d5339e76bf36`):**
all three checks were rerun end-to-end on this clean substrate rather than
relied on from the earlier `3973e0ec9` commit. Cache-read PPL: `auto`
7.73235049556447, `turboquant_k8v4` 7.724228731405752 (Δ **-0.10504%**, gate
≤+1% **PASS**), 50 chunks × 512 tokens, 12,800 scored tokens each arm —
matches the historical Δ (-0.105%) to within noise. Coding suite: 8/10 both
arms, and the prompt-id-level pass/fail sets are identical between arms (no
new failure, no new pass) — `node_check_js` fails with the same
missing-Node-interpreter error on both; `max_subarray` fails on both too,
though by a different mechanism per arm (`auto`'s emitted code block has a
bad indent, `turboquant_k8v4`'s response truncates mid-reasoning before
emitting a code block at all) — the id-level binary outcome is nonetheless
identical. Needle@32k: both arms passed 5/5, verified per-probe (all five
depths 10/30/50/70/90% individually correct, not just the aggregate count),
using the same bounded `--max-model-len 34816 --max-num-seqs 1
--max-num-batched-tokens 4096 --gpu-memory-utilization 0.75` launch. All
three quality gates **PASS** on the clean substrate. Raw artifacts:
`/home/larkinwc/gfx900-runs/quality-reconfirm-20260721-134221/` on `c4130-2`.

The pinned WikiText-2 cache-read control used 50 fixed 512-token chunks:
256-token prefills followed by 256 teacher-forced corpus decode tokens, with
raw target logits recorded before masking forced outputs. It scored 12,800
tokens and measured auto PPL 7.7323505 versus `turboquant_k8v4` PPL 7.7242287
(-0.105%). Because this delta is below per-chunk variation, it is
quality-neutral within measurement noise rather than evidence of a quality
improvement; it is below the +1% promotion gate. The prefill-only
`echo=true` prompt-logprob test remains a loader/numerics smoke because it
intentionally skips prefix-cache reads.

The burst screen exposed a hybrid-cache compatibility defect in
`DecodeBenchConnector`: Qwen3.5 Mamba cache entries are lists of state tensors,
not a single tensor. The connector now maps each configured KV-cache group to
its layers and fills every physical tensor in a list-backed entry. The initial
isolated graph launch failed with the old `AttributeError`; only the subsequent
fixed graph/eager cells above are valid evidence. Both completed 16/16 requests
with the connector active.

**RCCL/communication-tuning microbenchmark sweep (2026-07-21, source commit
`fad847da6dc43b66d4c0ec4042c98154057b9000`, extends
`benchmarks/kernels/benchmark_device_communicators.py`):** a full
communication-tuning sweep completed on this substrate and is independently
verified against raw JSON for both TP4 and TP8 device groups. This is a
genuine, complete negative result, not a gap: RCCL's own default auto-tuning
is already the best-performing option tested, so no static
`NCCL_ALGO`/`NCCL_PROTO`/channel-count override, and no `vllm/envs.py` or
host-profile change, is warranted from this sweep. It does not touch
topology, capacity, or the capacity/graph/quality results elsewhere in this
addendum — it is scoped to communication tuning only.

TP4 (visible indices 0-3): baseline cell-geomean over the 8/64/256/512/768
KiB message-size cells = 0.16579 ms; RCCL confirmed as
`2.27.7-release/rocm-rel-7.2`. `NCCL_ALGO=Ring` (no protocol override):
**+0.30%** — does not clear the +5% promotion gate. `NCCL_ALGO=Tree`:
**-29.5%** (worse), with per-size regressions up to +124.9% at 8192 KiB. The
four explicit `(algo,proto)` combinations `Ring+Simple`, `Ring+LL`,
`Tree+Simple`, `Tree+LL` were all worse than baseline, with `Ring+LL` and
`Tree+LL` regressing over +100% at large message sizes. The best of those,
`Ring+Simple`, was carried into a `NCCL_MIN_NCHANNELS` sweep ({2, 8, 16, 32}
with `NCCL_MAX_NCHANNELS=32`); none cleared the gate either (best: minch=32
at -10.85% vs. baseline). **Verdict: `DECLINED_STATIC_POLICY_SUFFICIENT`.**

TP8 (visible indices 0-7) reproduced the same shape of result: baseline
cell-geomean = 0.30791 ms, same RCCL version confirmed, same pattern of
explicit overrides underperforming default. **Verdict:
`DECLINED_STATIC_POLICY_SUFFICIENT`, `winning_combo: null`.**

**Operational hazard — distinct from the performance-decline findings
above, not a benchmark result:** during the TP8 sweep, `NCCL_ALGO=Ring
NCCL_PROTO=LL` genuinely deadlocked. All 8 rank processes pegged at ~99% CPU
with 0% GPU CU occupancy (confirmed via `rocm-smi`: VRAM allocated, no
compute-unit activity) for several minutes with zero progress; the run had
to be killed manually. `Tree+LL` at the same TP8 scale completed normally
immediately afterward, so this hang is specific to the `Ring`+`LL`
combination at 8-way scale on this host's PCIe topology, not a general
LL-protocol defect. **Do not set `NCCL_ALGO=Ring NCCL_PROTO=LL` together on
an 8-die gfx900 group on this platform.**

Raw evidence: `/home/larkinwc/gfx900-runs/rccl-sweep-20260721-183309/` on
`c4130-2` (`analysis/{tp4,tp8}_analysis.json` — complete per-size regression
tables; `json/` — raw benchmark output; `logs/` — including
`NCCL_DEBUG=VERSION` and `NCCL_DEBUG=INFO,TUNING` captures).

**Topology independent-launch confirmation (2026-07-21/22, source commit
`bfc80f4db504a4878a8dffac711b643756922e17`, adds `confirm-topology-fp16-tp8`
/ `confirm-topology-fp16-tp4` / `confirm-topology-fp4-pp2` to
`scripts/gfx900/matrices/reference.json`):** closes the topology item of the
Recommendation-gate checklist in `GFX900_RECOMMENDED.md`, scoped strictly to
**per-config reproducibility** — does each config's independent-launch
confirm reproduce its own 2026-07-13 single-launch screen — not a
cross-config performance ranking of TP8 vs. TP4 vs. TP4×PP2. Each new cell
(`trial_policy: confirm`, `confirm_launches: 3`) reused the existing
`promotion` policy unchanged (±3%/2% throughput/latency gate); no new gate
was introduced.

| Config | Screen (2026-07-13) | Confirm launches (2026-07-21/22) | Confirm aggregate | Δ output_throughput | Δ p99 TPOT | Δ p99 TTFT | `compare-cells` |
|---|---|---|---|---:|---:|---:|---|
| TP8 | 26.476 tok/s | 26.5317, 26.5161, 26.5186 | 26.5186 tok/s | +0.161% | -0.054% | -0.115% | `PASS` |
| TP4 | 20.895 tok/s | 21.0618, 21.0840, 21.0955 | 21.0840 tok/s | +0.903% | -0.944% | -1.175% | `PASS` |
| TP4×PP2 | 26.304 tok/s | 26.3800, 26.3448, 26.3361 | 26.3448 tok/s | +0.154% | -0.141% | -0.417% | `PASS` |

All 9 launches PASS with 0 failed requests; no config crosses the ±3%/2% gate
in either direction, so every verdict is `PASS` rather than `IMPROVEMENT` —
exactly what a reproducibility check should show, not a performance
intervention. **The TP8/TP4/TP4×PP2 absolute values above are not
comparable to each other under this gate** — each row only compares a
config against its own earlier screen. TP8's ~26.5 tok/s versus TP4's ~21.1
tok/s reflects the same workload split across twice as many GPUs, a
parallelism effect already documented elsewhere in this file, not new
topology evidence.

Live host topology was independently re-queried this session (not assumed
from prose): `numactl --hardware`, `lscpu`, `dmidecode -t processor`,
`rocm-smi --showtopo`, and `/sys/devices/system/node` / per-GPU
`/sys/.../numa_node` all confirm the accepted c4130-2 group is single-socket
(only CPU1 populated per `dmidecode -t processor`), a single NUMA node (all
36 CPU threads under `node0`), and all 8 gfx900 dies uniformly PCIe-attached
(topology weight 40, 2 hops, `Numa Node: 0` for every GPU per
`rocm-smi --showtopo`) — see `GFX900_SETUP.md`'s "Historical c4130-2
reference record" section for the full re-verification note. This is
consistent with, not a contradiction of, the "do not infer unselected
inventory" guidance in `GFX900_SETUP.md`: the dual-socket/16-die/QPI
hardware description in this document's own historical "Multi-GPU topology
review" section above predates the accepted c4130-2 group and matches the
same dual Xeon E5-2640 v3 / 16-die inventory named in `BENCH_GFX900.md`'s
historical hardware table (host `tyangpu1`) — not the accepted c4130-2
group's live topology confirmed here.

Raw artifacts:
`/home/larkinwc/gfx900-runs/durable-confirm-d600eaaa-f331-4ec1-998f-c1e1e9b2e8cf/cells/confirm-topology-{fp16-tp8,fp16-tp4,fp4-pp2}/cell.json`
and `compare-cells` outputs at
`/home/larkinwc/gfx900-runs/topology-gate-compare-20260722/{tp8,tp4,fp4pp2}.json`
on `c4130-2`. Screen baselines:
`/home/larkinwc/gfx900-runs/topology/cells/topology-fp16-tp8/cell.json`,
`/home/larkinwc/gfx900-runs/topology-tp4/cells/topology-fp16-tp4/cell.json`,
`/home/larkinwc/gfx900-runs/topology-pp2/cells/topology-fp4-pp2/cell.json`
(all share `manifest_sha256 9ffd3ab1d71629984918598f06901bfcbb457925a044c78048ee30942f6bc81d`,
`platform_sha256 61835f7caf7bf4057f4314e0d5f669c935e5d1ae5cbb83120745d5339e76bf36`).

### Promoted-workload profile

A bounded server-side PyTorch profiler captured the TurboQuant
prefix-repetition stack (16 prompts, c=8) at
`/home/larkinwc/gfx900-runs/profile-turboquant-prefix/`. The rank-0 summary
reports aggregate GPU time of 54.05% in RCCL generic all-reduce, 36.96% in
`aten::mm`, and 23.58% in the leading `Cijk_*MT64x64x16*` hipBLASLt GEMM
kernel. This authorizes further kernel investigation only with a concrete
shape-bounded proposal and a full microbenchmark/end-to-end gate; no source
kernel change is made from this aggregate observation alone. The older
rocprofv3 wrapper attempt aborted during rocprofiler configuration, so the
built-in server profiler is the retained capture.

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

---

# HIP CUDA-graph investigation (the `enforce_eager` question) — RESOLVED

**Question:** decode on gfx900 has a ~150 ms/token floor and realizes only
~5–9% of the HBM roofline (see `BENCH_GFX900.md`). Is that floor just
per-token **kernel-launch overhead** from running eager (no CUDA graphs)? If so,
capturing HIP/CUDA graphs should collapse the hundreds of per-layer launches
into one replay and remove the floor.

**Answer: No. Graphs work on gfx900, but they give ZERO decode speedup.** The
floor is real per-token compute/comm cost, not launch overhead. Measured A/B
below. `enforce_eager=True` remains the right default — not because graphs are
broken, but because they don't help and cost a long one-time autotune.

## The A/B result (TP4, Qwen3.5-9B, FP16, bs=1 single-stream decode, 128 tok)

| Mode | Config | tok/s | ms/token |
|------|--------|-------|----------|
| Eager | `enforce_eager=True` | 6.69 | 149.4 |
| Graph | `mode=NONE` + `FULL_DECODE_ONLY`, capture_sizes=[1] | 6.55–6.64 | 150.7–152.6 |

Three independent graph runs all landed at 150–153 ms/tok — within noise of
eager (often *slightly* slower). Graph capture was confirmed active in the logs:
`Capturing CUDA graphs (decode, FULL): 100%|████| 1/1` →
`Graph capturing finished in 1 secs`. So the captured graph genuinely replaced
the eager decode launches and produced identical, correct output — and the
per-token time did not move.

**Conclusion:** the ~150 ms/token decode cost is intrinsic to this hardware:
no MFMA (matmuls run at FP32 rate), no packed-FP16 from rocBLAS, per-layer PCIe
all-reduce under TP, plus the GDN linear-attention Triton kernels themselves.
Removing launch overhead changes nothing because launch overhead was never the
bottleneck. The real levers stay **PP-over-TP** (removes per-layer all-reduce)
and **batching/concurrency** (amortizes the fixed per-step cost) — both
validated in `BENCH_GFX900.md`.

## What it took to get graphs working (three real blockers)

### Blocker A — torch.compile/Inductor is the wrong target (don't bother)

With graphs left fully on, vLLM defaults to `mode=VLLM_COMPILE` (Inductor) +
`FULL_AND_PIECEWISE`. Inductor crashes on ROCm:
`torch.* op returned non-Tensor bool, target: is_current_stream_capturing`. This
is also why the gfx908/MI100 path force-sets `mode=NONE`. **Fixing Inductor is
pointless here** — the MI100 path proves disabling it is correct, and our A/B
shows pure HIP graphs (no Inductor) already give no speedup. The correct config
is `mode=NONE` (compile off) + `cudagraph_mode=FULL_DECODE_ONLY` (HIP graphs on).

### Blocker B — graph-capture hang + the REAL fix (`reset_method`, not firmware)

An early TP8 capture hung **two** GPUs at once (PCI 22:00.0 and 25:00.0, both
socket0). The driver's auto reset chose **BACO** (`reset_method=-1` → BACO),
which "succeeded" but **lost VRAM** and left one GPU's SMU wedged
(`No response from smu`, the bogus `firmware 0x1 vs 0xe` line is a *symptom* of
the half-dead GPU, **not** stale firmware — a clean boot shows the SMU loading
fine with no mismatch). `rocm-smi` then saw only 15/16 GPUs; **only a reboot
recovered it.**

The fix is **not** flashing firmware. It is forcing a reliable reset method:

```
# /etc/modprobe.d/amdgpu-reset.conf   (reversible: delete file + reboot)
options amdgpu reset_method=2          # 2 = mode1 (PSP-assisted whole-chip)
```

`sudo update-initramfs -u && reboot`, then verify
`cat /sys/module/amdgpu/parameters/reset_method` → `2`. **mode1 (value 2)** is
the robust path for Vega10. Do **not** use mode2 (value 3) — that's an
SMU/Arcturus(gfx908)-only reset and is not implemented for Vega10. With mode1 in
place, subsequent capture experiments never lost a GPU again (driver can
actually recover instead of corrupting VRAM via BACO).

### Blocker C — one-time Triton autotune during capture warmup is very slow

`mode=NONE + FULL_DECODE_ONLY` triggers `_warmup_prefill_kernels`
(`qwen_gdn_linear_attn.py:1152`) → `recompute_w_u_fwd` (`wy_fast.py:156`) →
`chunk_gated_delta_rule_fwd`, which `@triton.autotune`s the GDN/FLA kernels.
Each config is a full `make_amdgcn`/`make_hsaco` compile, and gfx900 compiles
slowly, so the GDN warmup grinds for ~10 min on a **cold** Triton cache
(`INIT_DONE in 647s`). It is slow, not infinite. The autotune result **persists
across runs**, so a **warm** start is `INIT_DONE in 63s` and graph capture
itself is only **1 second**. (The FLA configs are gfx900-valid — num_warps 2/4/8;
the one `num_warps=32` config in `l2norm.py:23` is invalid on wave64 but isn't
on this hot path.)

## Bottom line / guidance

- Keep **`enforce_eager=True`** for serving on gfx900 — graphs add a long cold
  autotune and buy nothing.
- If you must experiment with graphs: set `amdgpu.reset_method=2` first (mode1),
  use `mode=NONE` + `FULL_DECODE_ONLY` + small `cudagraph_capture_sizes`, run
  with a hard `timeout`, watch `dmesg | grep -E "GPU reset|amdgpu.*fail"`, and
  pre-warm the Triton cache so init is ~1 min instead of ~11 min.
- The decode floor is a **hardware** property (no MFMA/packed-FP16 + PCIe TP
  all-reduce), not a software/launch artifact. Optimize via PP + batching.

---

# Decode performance: where the gains are (and aren't) on gfx900

Starting point: single-stream (bs=1) decode of Qwen3.5-9B TP4 FP16 is ~150
ms/token (6.8 tok/s). HIP graphs gave 0 speedup (see above), proving the floor
is **fixed per-STEP overhead** (RCCL all-reduce + GDN linear-attn Triton
kernels), not kernel-launch overhead and not HBM bandwidth.

Roofline check (per GPU, TP4): each GPU reads 1/4 of weights = 4.83 GB at the
measured ~365 GB/s HBM = **~13 ms/step floor**. We measure ~150 ms/step → we are
**~10x above the bandwidth roofline**. So at bs=1 we use only ~9% of HBM
bandwidth. The headroom is real; the question is how to convert idle bandwidth
into throughput.

## Lever 1 — batching/concurrency (the big, free win) — issue #53

A decode *step* costs ~150 ms regardless of how many sequences are in the batch
(it reads the weights once, then does B sequences' worth of cheap per-token
work). So running B concurrent sequences yields ~B tokens per step → throughput
scales nearly linearly until something saturates.

`bench_scripts/batch_sweep.py`, TP4 FP16, gen=96:

| Batch | agg tok/s | scaling | ms/step |
|-------|-----------|---------|---------|
| 1 | 6.80 | 1.0x | 147 |
| 2 | 12.82 | 1.9x | 156 |
| 4 | 25.36 | 3.7x | 158 |
| 8 | 49.37 | 7.3x | 162 |
| 16 | 83.91 | 12.3x | 191 |
| 32 | 225–231 | 33x | 138–142 |
| 48 | 289 | 42x | 166 |
| 64 | 293 | 43x | 219 |
| 96 | ~498 | ~70x | 193 |
| 128 | 321 (regress) | — | 398 |

**Decode goes from 6.8 tok/s (bs=1) to ~300–500 tok/s aggregate (50–70x) purely
via concurrency.** Peak sits around B=64–96; B=128 regresses (KV-cache pressure /
scheduler recompute). The curve is approximate (wall time includes prefill of B
prompts, which grows with batch), but the regime is unambiguous: **decode is
latency/overhead-bound, and concurrency is the primary throughput lever.** Even
at B=96 per-step time (~190 ms) is still ~10x above the bandwidth roofline, so we
never actually become bandwidth-bound — the per-step overhead (RCCL + GDN
kernels) is the true ceiling.

**Takeaway:** serve with high concurrency. For latency-sensitive bs=1 use, the
floor is hardware-intrinsic; for throughput, batch hard.

## Lever 2 — INT4 weight quantization — issue #54 — TESTED, DOES NOT HELP

Hypothesis: shrink weights with INT4 so the model fits on fewer GPUs (cutting the
all-reduce floor) and reduce bytes/step. Tested `QuantTrio/Qwen3.5-9B-AWQ` (AWQ
4-bit, group_size 128; only MLP weights quantized — attn/linear_attn stay FP16;
12.4 GB → fits TP2).

**It loads and runs correctly on gfx900.** vLLM auto-selects
`TritonW4A16LinearKernel` for `AWQMarlinLinearMethod` (Marlin MoE is disabled on
ROCm; the dense Triton W4A16 dequant path is used — the right path for gfx900,
which has no MFMA/Marlin). Correct output, TP2.

`bench_scripts/awq_sweep.py`, AWQ INT4 TP2, gen=96:

| Batch | agg tok/s | ms/step |
|-------|-----------|---------|
| 1 | 4.98 | 200.6 |
| 8 | 27.28 | 293 |
| 32 | 54.88 | 583 |
| 64 | 56.12 | 1140 |
| 96 | 52.50 | 1829 |

**AWQ INT4 is SLOWER than FP16 on both latency and throughput:**

- bs=1: 200 ms/tok (AWQ TP2) vs 147 ms/tok (FP16 TP4) — **36% slower**.
- peak throughput: ~56 tok/s (plateaus at B=32) vs ~300–500 tok/s (FP16) — **~6–9x worse**.

**Why (the key lesson):**

1. gfx900 has **no native INT4/dequant hardware**, so `TritonW4A16` must
   dequantize INT4→FP16 in software every forward, then do the FP16 matmul. That
   dequant is **pure added compute**, not hidden behind anything.
2. Quantization only helps when you are **bandwidth-bound** (trade compute for
   fewer bytes). gfx900 decode is **overhead/latency-bound**, not bandwidth-bound
   (Lever 1 proved we never hit the HBM roofline). So fewer bytes buys nothing
   while the dequant adds cost.
3. At scale the Triton dequant kernel becomes the bottleneck: per-step time
   explodes (583→1829 ms) and throughput plateaus at B=32, where FP16 kept
   scaling to B=96.

**Takeaway:** INT4 quantization is the **wrong** lever for gfx900 decode, *because*
decode is not bandwidth-bound here. Quant would only pay off on a bandwidth-bound
GPU with native INT4/INT8 units (MI300, etc.). On Vega10 it adds software-dequant
overhead with no benefit. (Quant may still be worth it purely to *fit a larger
model* in limited VRAM — but not for speed.)

## Where the remaining decode gains actually are

Since both batching headroom and the all-reduce floor dominate, the productive
directions are:

- **#55 PP-over-TP under concurrency** — pipeline parallel swaps per-layer
  all-reduce for point-to-point sends; should cut the per-step overhead that
  caps the batch-sweep ceiling.
- **#56 profile the 150 ms step** — split RCCL all-reduce vs GDN/FLA Triton
  kernel time; tells us whether to attack comms (PP, fewer ranks) or kernels
  (tune the GDN linear-attention Triton ops for wave64/gfx900).

Both target the real bottleneck (fixed per-step overhead), unlike quantization
which targets bandwidth we aren't actually limited by.

---

# Redeeming INT4 on gfx900: a hand-written decode GEMV — issue #57

`#54` found stock AWQ INT4 is *slower* than FP16 on gfx900 because the W4A16
path uses `tl.dot` even at M=1, and gfx900 has no MFMA → `tl.dot` becomes a
padded FP32 GEMM (one real token padded to BLOCK_M=16/32). The fix is not better
dequant (that was already register-level) — it's the **wrong primitive**. Decode
is M=1, so the right primitive is a **GEMV**: purely HBM-bandwidth-bound, no
matrix units needed, and int4 weights move 4× fewer bytes than FP16.

## The kernel (`vllm/.../mixed_precision/gfx900_w4a16_gemv.py`)

M=1..8 int4 GEMV: register-level unpack (GPTQ-sequential packing, shift =
`(j%8)*4`, matching the generic `triton_w4a16` kernel), group scales/zeros loaded
**once per group** (not per K-row), and **split-K** to parallelize long-K /
small-N shapes across gfx900's 56 CUs. Per-shape tuned configs
(`_GFX900_GEMV_CONFIGS`). `tl.sum` reduction over K instead of `tl.dot`.

Dispatch: `triton_w4a16_gemm` routes `on_gfx900() and group_size in {32,64,128}
and M<=8` to this GEMV; prefill (M>8) still uses the generic `tl.dot` path.

## Microbench (real Qwen3.5-9B MLP shapes, M=1, correctness rel err ≤ 0.001)

| shape | FP16 GEMV | int4 GEMV | vs FP16 | vs stock tl.dot (the path it replaces) |
|-------|-----------|-----------|---------|------------------------------------------|
| gate/up K=4096 N=24576 | 0.726 ms | 0.257 ms | **2.82×** | 2.86 ms → 0.282 ms = **10.1×** |
| down K=12288 N=4096    | 0.576 ms | 0.143 ms | **4.02×** | 2.21 ms → 0.141 ms = **15.6×** |

The 10–15× vs stock is because the gfx900 `tl.dot` int4 path is pathologically
bad (~2–3 ms for one token). ~54% of the 365 GB/s HBM roofline; correct across
M=1/4/8 vs the existing kernel.

## End-to-end (Qwen3.5-9B-AWQ, TP2, enforce_eager, bs=1 decode)

| Config | GPUs | bs=1 ms/tok | bs=1 tok/s |
|--------|------|-------------|------------|
| FP16 (baseline) | TP4 | 147 | 6.80 |
| AWQ INT4, stock `tl.dot` | TP2 | 200 | 4.98 |
| **AWQ INT4, gfx900 GEMV** | **TP2** | **95.6** | **10.46** |

Output verified correct. The kernel flips INT4 from the *worst* decode option to
the **best**: **2.1× faster than stock AWQ**, **1.54× faster than FP16 at bs=1**,
and on **half the GPUs** (TP2 vs TP4) — freeing 2 GPUs per replica.

**When to use it:** latency-sensitive / low-to-moderate concurrency decode, or
when VRAM/GPU count is the constraint. At very high batch, FP16 still wins on
aggregate throughput (decode there is step-overhead-bound, not weight-bandwidth-
bound, so fewer weight bytes stops helping — consistent with the batch-sweep
finding). The GEMV gate is M<=8, so high-batch prefill/decode automatically falls
back to the generic path.

## Why this matters (the general lesson)

Quantization only helps where you're **bandwidth-bound**. gfx900 bs=1 decode
*is* weight-bandwidth-bound (each token re-reads all weights), so int4's 4×-fewer
bytes is a real win — but only once you stop forcing the work through MFMA-less
`tl.dot`. A GEMV that reads int4 and accumulates in vector ALU is the right shape
for this hardware. Same principle would apply to any matmul-light, M=1 path on a
no-MFMA GPU.

## TurboQuant KV-cache quantization on gfx900 (issue #62)

TurboQuant (arXiv 2504.19874; rotation + Lloyd-Max scalar codebook) compresses the
KV cache to roughly double usable context / concurrency per VRAM. vLLM ships a full
in-tree implementation (`vllm/model_executor/layers/quantization/turboquant/`,
`vllm/v1/attention/{backends/turboquant_attn,ops/triton_turboquant_*}`), selected
via `--kv-cache-dtype turboquant_*`.

### Why it works on gfx900 (where the llama.cpp HIP port does not)

The reference llama.cpp-turboquant-hip port **explicitly blocks wave64 hardware
(Vega/GCN/CDNA)**: its codebook lookup uses `__shfl_sync(width=32)`, which returns
garbage on 64-wide wavefronts. vLLM's TurboQuant kernels are **pure Triton** — no
`__shfl_sync`, no hand-rolled warp shuffle, no `tl.dot`. Triton handles the
wavefront width internally, so the wave64 blocker simply does not apply. We only had
to add `TURBOQUANT` to the gfx900 attention-backend list in `rocm.py` (opt-in; the
default stays TRITON_ATTN unless `--kv-cache-dtype turboquant_*` is passed). No
kernel changes were needed. Prefill uses the `F.scaled_dot_product_attention`
fallback since gfx900 has no flash-attn.

### Preset choice: Qwen3.5 is "quirky-K"

Per the ollama#15051 sweep, the Qwen3 family is extremely sensitive to *key*-side
quantization (tq2k = +150% PPL on qwen3:8b — output destroyed), while tolerant of
value quantization. So on Qwen3.5-9B use **`turboquant_k8v4`** (FP8 keys + 4-bit
values) — it leaves keys effectively unquantized and only compresses values. V-heavy
presets are mandatory here; aggressive K presets (`turboquant_3bit_nc`) are unsafe.

### Measured — Qwen3.5-9B, TP=4 gfx900, fp16 weights, max_model_len=16384

| Metric | FP16 KV (`auto`) | `turboquant_k8v4` | Δ |
|---|---|---|---|
| GPU KV cache size | 194,267 tokens | **454,382 tokens** | **2.34×** |
| Max concurrency @16k ctx | 11.86× | **27.73×** | **2.34×** |
| Decode tok/s, B=1 (~8k prefill) | 4.75 | 5.10 | +7% |
| Decode tok/s, B=8 | 12.20 | 13.78 | +13% |

Output stays coherent ("The capital of France is Paris"; correct photosynthesis
explanation). **2.34× more context/concurrency per VRAM, plus a small decode speedup**
(fewer KV bytes read per step). Note Qwen3.5 is hybrid — only 8/32 layers are full
attention with a KV cache (rest are GDN linear-attn), so the cache footprint is
already small; the 2.34× multiplier applies to that KV-cached portion.

Cold start pays the usual GDN/FLA Triton autotune (~647s cold, ~60s warm) plus the
TurboQuant centroid init. `enforce_eager=True` as always on gfx900.

Repro: `bench_scripts/tq_measure.py {auto|turboquant_k8v4}` (TP4, 16k ctx; prints
GPU KV cache size + decode tok/s at B=1/8).

### Composing with the int4 weight GEMV (#57)

TurboQuant (KV-cache quant) and the gfx900 int4 weight GEMV (#57) are orthogonal —
they live in different parts of the stack (attention-backend KV path vs the
`triton_w4a16_gemm` weight-matmul path) and share no code. They stack cleanly.

Verified on `QuantTrio/Qwen3.5-9B-AWQ` (int4 MLP weights, exercises the gfx900 GEMV)
loaded with `--kv-cache-dtype turboquant_k8v4`, TP2, gfx900:

- Both paths active simultaneously: log shows `quantization=awq_marlin` +
  `_w4a16_gemv_splitk_kernel` JIT (our GEMV) **and** `Overriding with TURBOQUANT`.
- Coherent output ("...is Paris").
- Short-ctx decode B=1: **9.97 tok/s (100.3 ms/tok)** — statistically identical to
  standalone AWQ int4 (10.46 tok/s / 95.6 ms; ~5% delta is JIT-warmup noise). So
  **TurboQuant adds its ~2.34x KV-capacity benefit at near-zero decode cost on top
  of the int4 weight win.**

Caveat — VRAM ceiling: these V340 dies are ~8 GiB each. AWQ(TP2) + 16k-token KV
exhausts memory at batch>1 (`torch.OutOfMemoryError`, 16 MiB free). For long context

- batching with the AWQ model, use TP4 (more aggregate VRAM) or a shorter
max_model_len. The compression itself is what lets you push context further per GiB;
it doesn't remove the hard per-die cap.

Repro: `bench_scripts/combo_dec.py turboquant_k8v4` (AWQ TP2, short-ctx decode).

### Quality: teacher-forced WikiText-2 cache-read perplexity

The older 24-window prefill-logprob numbers are a loader/numerics smoke only:
the API skips prefix-cache reads. The valid c4130-2 TP8 Qwen3.5-9B control
prefills 256 tokens and teacher-forces the next 256 corpus tokens from each of
50 fixed WikiText-2 chunks, recording raw target logits before masking forced
outputs. Across 12,800 scored tokens, auto PPL was 7.7323505 and
`turboquant_k8v4` PPL was 7.7242287 (-0.105%). The delta is quality-neutral
within measurement noise rather than a quality improvement, and falls below
the +1% gate. It supports the capacity-oriented profile decision; it does not
make TurboQuant a global default. Repro:
`.venv/bin/python -m scripts.gfx900.cache_read_ppl {auto|turboquant_k8v4}`.

# UPDATE: HIP CUDA graphs now give 3.6x decode (revisit after LLMM1) — issue #63

The "HIP CUDA-graph investigation — RESOLVED" section above concluded graphs give
**zero** decode speedup and that enforce_eager should stay the default. **That
conclusion is now obsolete.** It was correct for its data (GPU-busy 74.6 ms/step, so
the host dispatch gap was a small slice), but the LLMM1 win (#59) cut GPU-busy to
27.9 ms/step, making the GPU-idle dispatch gap **59% of the step**. Graphs remove
exactly that gap.

## Re-test (Qwen3.5-9B TP4, FP16, bs=1 decode, best of 3 x 160 tok)

| Mode | ms/tok | tok/s |
|---|---:|---:|
| Eager (LLMM1) | 67.0 | 14.93 |
| **Graph + LLMM1** | **18.4** | **54.41** |

3.6x, reproduced twice (18.4 / 18.5 ms), coherent identical output. Config:
`mode=NONE` + `cudagraph_mode=FULL_DECODE_ONLY` + `capture_sizes=[1]` (Inductor/
compile stays OFF — only HIP graph capture; Inductor still crashes on ROCm).

## Full decode progression (Qwen3.5-9B, bs=1)

| Config | tok/s | vs original |
|---|---:|---:|
| FP16 TP4 eager (original) | 6.65 | 1.0x |
| + LLMM1 GEMV (#59) | 14.85 | 2.2x |
| + CUDA graphs (#63) | **54.4** | **8.2x** |

Full stack (AWQ int4 + TurboQuant KV + LLMM1 + graphs), TP2 on half the GPUs:
14.12 -> **37.71 tok/s**, coherent.

## Gotcha: hybrid (Mamba/GDN) graph capture needs max_num_seqs <= Mamba cache blocks

Qwen3.5 is hybrid; each decode sequence needs one Mamba/GDN state cache block. Graph
capture fails if `max_num_seqs` exceeds the available blocks:
`max_num_seqs (256) exceeds available Mamba cache blocks (N) ... CUDA graph capture
cannot proceed`. Fix: set `--max-num-seqs <= N` (N depends on
gpu_memory_utilization and model; was 42 for AWQ TP2 on 8 GiB dies — use 32).

## New guidance

**Use CUDA graphs for gfx900 decode** (do NOT pass `enforce_eager`). Set
`compilation_config(mode=NONE, cudagraph_mode=FULL_DECODE_ONLY, capture_sizes=[1])`
and `max_num_seqs` within the Mamba-block budget. Cold start still pays the
GDN/FLA Triton autotune + a 1-2s capture; both cache.

Lesson: a perf conclusion is only valid for the bottleneck profile under which it was
measured. After a structural change (here LLMM1 halving GPU-busy), revisit prior
negatives — the cheap A/B flipped from 0% to 3.6x.

Repro: `bench_scripts/graph_ab.py {eager|graph}`, `bench_scripts/graph_combo.py graph turboquant_k8v4`.

# PP-over-TP revisited after LLMM1 + CUDA graphs (issue #55) — the advantage inverted

BENCH_GFX900.md (eager, pre-LLMM1) found pipeline parallel beat tensor parallel at
equal GPU count by +48-90%, for two reasons: (1) PP swaps the per-layer all-reduce
for cheap point-to-point sends, and (2) pure TP makes each gfx900 do a small,
inefficient slice of a matrix-core-less FP16 GEMM. **Both reasons were undermined by
later work**, so we re-measured.

## What changed since the original PP benchmark

- **#56 profile**: RCCL all-reduce is only ~2% of the decode step, not the dominant
  cost the PP motivation assumed.
- **#59 LLMM1**: decode GEMMs are now real GEMVs, not padded macrotile matmuls. A
  GEMV is not "inefficiently sliced" by a 4-way TP split -> reason (2) is gone.
- **#63 CUDA graphs**: the host dispatch gap that PP's pipelining used to hide is
  now removed for everyone.

## Head-to-head, 4 GPUs, one socket, LLMM1 + CUDA graphs, Qwen3.5-9B

Offline `LLM.generate`, decode out tok/s (aggregate), max_num_seqs=32.

| concurrency | TP4 | TP2xPP2 | winner |
|---|---:|---:|---|
| c=1 (single-stream) | **52.3** | 32.5 | **TP4 +61%** |
| c=8 | 48.9 | **54.8** | PP +12% |
| c=32 (peak throughput) | **220.0** | 176.4 | **TP4 +25%** |

Contrast with the old eager numbers (PP won every cell, +48-90%). With LLMM1 +
graphs, **TP4 now wins single-stream and peak throughput**; PP only edges ahead in a
narrow mid-concurrency band (c~8) where its pipeline fills but TP hasn't yet reached
its batched sweet spot.

### Why TP flipped to winning

- TP4 c=1 went from ~4.6 tok/s (old eager) to **52.3** (LLMM1 + graphs) = ~11x. The
  things that made TP slow (padded GEMM slices, per-step dispatch) are exactly what
  LLMM1 + graphs fixed, so TP no longer needs PP to hide them.
- PP's structural latency tax (c=1 must walk all pipeline stages) is now a net
  negative because there's no longer a big per-layer overhead for it to amortize.

## 8-GPU PP not usable here (offline)

TP2xPP4 (8 GPUs) crashes during init on the offline `LLM.generate` path in this build
— `hipErrorLaunchFailure` across all PP ranks, in **both** eager and graph modes
(GPUs recover via reset_method=2). The old TP2xPP4 numbers came from server mode with
`--no-async-scheduling`; the offline PP4 path regressed since. Not pursued further —
the 4-GPU result already answers the question, and TP4 is the better config anyway.

## Recommendation (updated)

**Use TP (TP4 within one socket), not PP, for gfx900 decode now that LLMM1 + CUDA
graphs are in.** PP was the right call in the eager/padded-GEMM era; it is no longer.
The remaining decode levers are batching/concurrency (TP4 scales to ~220 tok/s at
c=32) and the int4 + TurboQuant stack.

Repro: `bench_scripts/pp_bench.py {tp4|tp2pp2} {graph|eager} <num_gpus>`.

# Multi-GPU topology review: TP4 vs TP8 vs TP16 vs cross-socket PP (issue #55 follow-up)

This box is 8x AMD Radeon Pro V340 = **16 gfx900 dies**, split across two CPU sockets:
GPUs 0-7 on socket 0, GPUs 8-15 on socket 1. Intra-socket P2P is ~8 GB/s; **the
cross-socket link is the Xeon E5 v3 QPI path at ~0.26 GB/s** (a hardware ceiling, not
tunable). Every layout decision is dominated by that 30x cross-socket bandwidth cliff.

All runs below use the recommended decode stack: LLMM1 skinny GEMV (#59) + FULL_DECODE_ONLY
CUDA graphs (#63), Qwen3.5-9B FP16, `max_num_seqs=32`, offline `LLM.generate`, out tok/s.

## Results

| Layout | GPUs | Socket(s) | c=1 | c=8 | c=32 | Outcome |
|---|---:|---|---:|---:|---:|---|
| **TP4** | 4 | one (0-3) | **52.3** | 48.9 | **220.0** | **best** |
| TP8 | 8 | one (0-7) | 14.7 | 12.7 | 61.8 | ~3.5x worse than TP4 |
| TP16 | 16 | both (0-15) | stall | - | - | catastrophic, killed |
| TP8xPP2 | 16 | both | (inconclusive) | - | - | needs clean revisit |

## TP8 (one socket) is much WORSE than TP4 — more TP hurts here

TP8 c=1 = 14.7 tok/s, *worse than TP4 running eager* (14.9). Doubling TP from 4 to 8:

- doubles the per-layer all-reduce participant count (8-way collective over PCIe), and
- halves each die's GEMV slice. With LLMM1 the decode matmul is already a skinny GEMV;
  splitting an M=1 GEMV 8 ways makes each slice too small to amortize launch overhead,
  so the all-reduce cost dominates and throughput collapses.
- Cold init also scaled with GPU count: TP8 took ~662s (vs ~67s for TP4) for GDN/FLA
  Triton autotune + graph capture.
**Takeaway: TP4 is the sweet spot. Going past TP4 on this box is a net loss for decode.**

## TP16 (cross-socket TP) is catastrophic

Init completed but the first decode step stalled 20+ min and never returned. py-spy
showed every worker blocked in `all_gather_into_tensor` inside
`logits_processor._gather_logits` -- the lm_head logits all-gather (vocab=248320)
across all 16 TP ranks, which **must traverse the 0.26 GB/s QPI link**. Pure TP issues
a collective on every layer plus this giant logits gather; routing all of that over QPI
makes a single decode step effectively never finish. Killed; GPUs recovered via
reset_method=2. **Cross-socket pure TP is unusable -- do not span sockets with TP.**

## TP8xPP2 (cross-socket PP) -- inconclusive, flagged for clean revisit

The theoretically-correct way to use all 16 GPUs: TP8 *within* each socket (intra-socket
all-reduce only) with the PP stage boundary crossing QPI as cheap point-to-point sends.
It initialized and started decoding (8 active requests reached the decode loop), then
hit a collective `TimeoutError` mid-decode. **But the host was simultaneously overloaded
(load avg ~30-50) and SSH was timing out, so this is not a clean measurement.** Worse,
the cross-socket teardown left two socket-1 dies (GPUs 14/15) with a non-responsive SMU
(`amdgpu: Failed to send message ... ret 0xffffffff` looping) and ~57 kernel ttm workers
stuck in uninterruptible D-state; those two dies then fell off the PCIe bus and a warm
reboot did not recover them (needs a cold power cycle). **Verdict: TP8xPP2 is worth a
proper, isolated retry on a quiet box, NOT a recorded negative result.** See "Revisit"
below.

## Conclusion / recommendation

- **Use TP4 on one socket for decode.** It wins single-stream (52.3 tok/s) and peak
  throughput (220 tok/s at c=32), and avoids every cross-socket pitfall.
- More TP than 4 hurts (TP8 ~3.5x worse); cross-socket TP (TP16) is unusable.
- The QPI link (0.26 GB/s) is the hard wall for any 16-GPU layout. The only layout that
  could plausibly use both sockets is cross-socket PP (point-to-point, not all-reduce),
  which remains an open question pending a clean retry.
- For multi-tenant serving, run **independent TP4 (or TP8) engines per socket** rather
  than one model spanning sockets.

## Revisit later: TP8xPP2 on a quiet box

Retry `bench_scripts/run_topo.sh tp8pp2 eager 16` (and `graph 16`) only when (a) all 16
GPUs are healthy after a cold power cycle, (b) the host is idle, and (c) consider raising
the distributed collective timeout (`--distributed-timeout` / `TORCH_NCCL_*` env) so the
QPI activation hand-off doesn't trip the default timeout. The question to answer: can PP
keep the per-layer all-reduce inside each socket and only pay QPI at the single stage
boundary, and is the resulting tok/s competitive with two independent TP4 engines?

Scripts: `bench_scripts/topo_bench.py {tp4|tp8|tp16|tp8pp2} {graph|eager}` via
`bench_scripts/run_topo.sh <layout> <mode> <num_gpus>`.

---

## 2026-07-13 reusable-platform evidence — c4130-2

This entry supersedes none of the historical topology conclusions above; it is
the first schema-valid reusable-platform reference. The clean source commit is
`3973e0ec9cd10b95f4663096237c025806efdfdb`. The accepted manifest is
`/home/larkinwc/gfx900-runs/reference-3/manifest.json` with
`platform_sha256`
`61835f7caf7bf4057f4314e0d5f669c935e5d1ae5cbb83120745d5339e76bf36` and
`manifest_sha256`
`9ffd3ab1d71629984918598f06901bfcbb457925a044c78048ee30942f6bc81d`.

The selected eight gfx900 dies are 56 CU / 8,573,157,376 bytes each, all on
NUMA node 0 with PCIe links, and `amdgpu.reset_method=2` was verified. The
screened TP8 FP16 eager cell (4,096 input / 256 output / c=8) produced 26.476
output tok/s with p99 TPOT 279.810 ms, p99 TTFT 39,304.785 ms, and no failed
requests. The 8,192-token TurboQuant capacity screen also had no failed
requests (18.078 output tok/s), but it lacks the auto-KV control, capacity
frontier, and quality evidence required for promotion. Native MTP K=1 was
declined: compared with that eager reference it reduced output throughput by
15.485%, increased p99 TPOT by 50.879%, and increased p99 TTFT by 5.969%.

The artifacts are
`/home/larkinwc/gfx900-runs/topology/cells/topology-fp16-tp8/cell.json`,
`/home/larkinwc/gfx900-runs/capacity/cells/capacity-turboquant/cell.json`,
and
`/home/larkinwc/gfx900-runs/speculation/cells/speculation-mtp-k1/cell.json`.
Do not set RCCL, graph, TurboQuant, or MTP defaults from these screens.
