# Recommended inference setup — gfx900 (V340 / Vega10)

The historical sections cover **Qwen3.5-9B** (dense hybrid) and
**Qwen3-Coder-Next-AWQ-4bit** (sparse MoE). The clean-source campaign below
uses the selected c4130-2 eight-die group; its results remain
workload-specific.

> **Current verification status (2026-07-20):** the clean, commit-validated
> substrate has platform digest
> `61835f7caf7bf4057f4314e0d5f669c935e5d1ae5cbb83120745d5339e76bf36`
> at source commit `689cbbba3ae5400bd583e1435177464e48f1f94a`. The TurboQuant
> vs. auto KV-cache capacity comparison (TP8 Qwen3.5-9B FP16, 8,192/256, c=32)
> now has the required independent-launch confirmation: three independent
> launches per arm, all PASS, zero failed requests, mechanized `compare-cells`
> verdict `IMPROVEMENT`. This confirms the throughput/latency capacity gates
> only. TurboQuant's quality gates (perplexity, coding suite,
> needle-in-haystack) were reconfirmed on this clean substrate on
> 2026-07-21 (see the dedicated note below) — all three **PASS**. Combining
> the capacity and quality results into one promoted default/profile is
> still a separate follow-up decision, not made by this note. The dense c=1
> graph/eager confirm
> trial (`confirm-dense-eager-c1` / `confirm-graphs-full-decode-c1`) has now
> completed its independent-launch confirmation: three independent launches
> per arm, both cells PASS, zero failed requests. Confirmed
> `confirm-dense-eager-c1` (`--enforce-eager` baseline) output_throughput
> 12.724873664265084 tok/s, p99 TPOT 79.0647771127532 ms, p99 TTFT
> 772.2754819784313 ms; confirmed `confirm-graphs-full-decode-c1`
> (`FULL_DECODE_ONLY`, capture sizes [1,2,4,8,16,32,64,96]) output_throughput
> 56.336206381615746 tok/s, p99 TPOT 16.297473464836184 ms, p99 TTFT
> 781.2450528336922 ms. Mechanized `compare-cells` verdict: `IMPROVEMENT`,
> zero regressions, comparable `platform_sha256` — output_throughput
> +342.725%, p99 TPOT -79.387%, p99 TTFT +1.161% (within the 2%
> latency-regression gate). Unlike the TurboQuant KV-cache-dtype comparison,
> CUDAGraph capture for this dense eager-vs-graph decode comparison changes
> no numerics (identical FP16 weights and math, replayed rather than
> dispatched); it has no perplexity/coding/needle quality-gate requirement in
> this campaign's promotion framework, so the throughput/latency capacity
> confirmation above is the complete gate for this result. The prior AWQ
> graph result is historical only:
> on this substrate, AWQ requires `--max-num-batched-tokens 512` for eager
> startup, while every tested VLLM_COMPILE configuration hangs during warmup
> **TurboQuant quality-gate reconfirmation (2026-07-21):** the three quality
> checks (cache-read perplexity, coding suite, needle-in-haystack) have now
> been rerun on this clean substrate at source commit
> `0054f988bcca5d50bcfaddf2ff7545246df6e8ba` (four commits ahead of
> `689cbbba3`, not diverged; `platform_sha256` reconfirmed matching
> `61835f7caf7bf4057f4314e0d5f669c935e5d1ae5cbb83120745d5339e76bf36`). Cache-read
> PPL: `auto` 7.73235049556447, `turboquant_k8v4` 7.724228731405752 (Δ
> **-0.10504%**, gate ≤+1% **PASS**), 50 chunks × 512 tokens (256 prefill +
> 256 teacher-forced decode), 12,800 scored tokens for each arm — matches the
> earlier-commit historical numbers (7.7323505 / 7.7242287, Δ -0.105%) to
> within noise. Coding suite: 8/10 for both `auto` and `turboquant_k8v4`,
> identical prompt-id-level pass/fail sets (no new failure, no new pass vs.
> the matched `auto` run) — the same two known-environmental failures on both
> (`node_check_js`: no Node interpreter on this host; `max_subarray`:
> extractor-format failure on a reasoning-heavy prompt — the exact failure
> mechanism differs slightly between arms, `auto` emits a code block with a
> bad indent while `turboquant_k8v4`'s response truncates mid-reasoning
> before emitting one, but the binary id-level pass/fail outcome is
> identical). Needle@32k: both `auto` and `turboquant_k8v4` passed 5/5 (all
> depths 10/30/50/70/90%, per-probe verified, not just the aggregate count),
> launched with `--max-model-len 34816 --max-num-seqs 1
> --max-num-batched-tokens 4096 --gpu-memory-utilization 0.75`. All three
> quality gates now **PASS** on the clean substrate. Combined with the
> already-confirmed capacity/latency gates above, TurboQuant's quality
> reconfirmation is complete; deciding whether/how to write up a "recommended
> profile" combining both is a separate follow-up, not made by this note. Raw
> artifacts: `/home/larkinwc/gfx900-runs/quality-reconfirm-20260721-134221/`
> on `c4130-2` (`cache-read-ppl-{auto,turboquant}.json`,
> `coding/m6_coding_eval_*.json`, `needle-32768-{auto,turboquant}.json`,
> `gate_summary.json`, `campaign.status`/`campaign.log` for the full run
> trace).
>
> **RCCL communication-tuning sweep (2026-07-21, source commit
> `fad847da6dc43b66d4c0ec4042c98154057b9000`, extending
> `benchmarks/kernels/benchmark_device_communicators.py`):** the RCCL/NCCL
> communication-tuning microbenchmark sweep from the roadmap has completed on
> this substrate for both the TP4 and TP8 device groups, independently
> verified against raw JSON for each. TP4 (visible indices 0-3, baseline
> cell-geomean over the 8/64/256/512/768 KiB cells = 0.16579 ms) and TP8
> (visible indices 0-7, baseline cell-geomean = 0.30791 ms) both confirmed
> RCCL `2.27.7-release/rocm-rel-7.2` and both reached verdict
> `DECLINED_STATIC_POLICY_SUFFICIENT`: every tested `NCCL_ALGO`/`NCCL_PROTO`
> combination and `NCCL_MIN_NCHANNELS` sweep underperformed RCCL's own
> default auto-tuning (best case TP4 `Ring` with no protocol override at
> +0.30%, short of the +5% promotion gate; worst case TP4 `Tree` at -29.5%,
> with per-size regressions up to +124.9% at 8192 KiB). TP8 reproduced the
> same shape of result, `winning_combo: null`. This is a genuine, complete
> negative result, not a gap in coverage — no `vllm/envs.py` or host-profile
> RCCL variable change is warranted from this sweep, and it does not touch
> topology, capacity, or the capacity/graph/quality results above. **Separate
> operational hazard, not a performance finding:** `NCCL_ALGO=Ring
> NCCL_PROTO=LL` together on the 8-die TP8 group genuinely deadlocked — all 8
> rank processes pegged at ~99% CPU with 0% GPU CU occupancy (confirmed via
> `rocm-smi`: VRAM allocated, no compute-unit activity) for several minutes
> with zero progress, and had to be killed manually; `Tree+LL` at the same
> TP8 scale completed normally immediately afterward, so the hang is specific
> to `Ring`+`LL` at 8-way scale on this host's PCIe topology. **Do not set
> `NCCL_ALGO=Ring NCCL_PROTO=LL` together on an 8-die gfx900 group on this
> platform.** Full per-size regression tables and detail:
> `PERF_GFX900.md` § 2026-07 c4130-2 reproducibility campaign addendum. Raw
> artifacts: `/home/larkinwc/gfx900-runs/rccl-sweep-20260721-183309/` on
> `c4130-2` (`analysis/{tp4,tp8}_analysis.json`, `json/`, `logs/` including
> `NCCL_DEBUG=VERSION` and `NCCL_DEBUG=INFO,TUNING` captures).
>
> **Recommendation gate:** do not apply historical V340 count, per-die VRAM,
> topology, RCCL variables, graph buckets, quantization defaults, or
> performance figures to another host. First run the versioned `scripts.gfx900`
> manifest and matrix workflow; publish a recipe only after its matching
> `platform_sha256` passes the topology, graph, KV/prefix, speculation, and
> quality gates.

## Historical c4130-2 reference record (2026-07-13)

The historical manifest is
`/home/larkinwc/gfx900-runs/reference-3/manifest.json`
(`platform_sha256`
`61835f7caf7bf4057f4314e0d5f669c935e5d1ae5cbb83120745d5339e76bf36`,
`manifest_sha256`
`9ffd3ab1d71629984918598f06901bfcbb457925a044c78048ee30942f6bc81d`).
It covers eight 56-CU, 8,573,157,376-byte gfx900 dies on `c4130-2`, all on
NUMA node 0 and connected by PCIe. The source was clean commit
`3973e0ec9cd10b95f4663096237c025806efdfdb`.

- `topology-fp16-tp8` — Qwen3.5-9B FP16, TP8, eager, 4,096 input / 256
  output, c=8: 26.476 output tok/s, p99 TPOT 279.810 ms, p99 TTFT
  39,304.785 ms, and 0 failures. This is a historical eager reference for this
  exact workload. Artifact:
  `/home/larkinwc/gfx900-runs/topology/cells/topology-fp16-tp8/cell.json`.
- `capacity-turboquant` — TP8 eager, `turboquant_k8v4`, 8,192 input / 256
  output, c=8: 18.078 output tok/s, p99 TPOT 402.193 ms, p99 TTFT
  71,276.167 ms, and 0 failures. This was the initial capacity screen; its
  cache-read quality gate did not replace the pending capacity confirmation.
  Artifact:
  `/home/larkinwc/gfx900-runs/capacity/cells/capacity-turboquant/cell.json`.
- `speculation-mtp-k1` — TP8 eager, native MTP K=1, 4,096 input / 256 output,
  c=8: 22.376 output tok/s, p99 TPOT 422.173 ms, p99 TTFT 41,651.013 ms, 0
  failures, and acceptance length 1.182. **Declined:** versus the eager TP8
  reference, output throughput is -15.485%, p99 TPOT is +50.879%, and p99 TTFT
  is +5.969%; all exceed the promotion policy. Artifact:
  `/home/larkinwc/gfx900-runs/speculation/cells/speculation-mtp-k1/cell.json`.

No RCCL tuning variable, speculative-decoding configuration, CUDAGraph
profile, or `turboquant_k8v4` configuration is a current gfx900 default from
this campaign. The cache-read perplexity control remains quality-neutral
within measurement noise, but quality alone does not satisfy the pending
capacity confirmation.

For the historical eager reference only, use the exact matrix-controlled options:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V1=1 \
vllm serve Qwen/Qwen3.5-9B \
  --tensor-parallel-size 8 --dtype float16 \
  --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --language-model-only --max-model-len 4352 --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 --enforce-eager
```

This is a reproducible benchmark control, not a general production prescription:
retain explicit workload, capacity, and quality gates before changing it.

### Historical workload results and guarded recommendations

The rows below record the earlier source commit
`3973e0ec9cd10b95f4663096237c025806efdfdb` and model revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`. They are not promotion evidence
for the clean `689cbbba3` campaign. The 2026-07-19 dense c=1 screen measured
57.651 versus 12.707 output tok/s (graph versus eager); that confirm trial
(`confirm-dense-eager-c1` / `confirm-graphs-full-decode-c1`) has since
completed (see the table row below) with 3 independent launches per arm,
both cells PASS, zero failed requests, and a mechanized `compare-cells`
verdict of `IMPROVEMENT`. The matched TurboQuant c=32 screen measured
20.521 versus 18.583 output tok/s (TurboQuant versus auto) as a
single-launch pilot; the 3-launch independent-confirm trial for this
comparison has since completed (see the capacity row below) and passed the
throughput/latency capacity gates, though TurboQuant's quality gates remain
historical-only and unreconfirmed on this substrate.

| Workload / decision | Result | Recommendation |
|---|---|---|
| TP8 FP16 eager, 4,096/256, c=8 | 26.476 output tok/s | Retain as the general c4130-2 control. TP4 and TP4×PP2 did not meet a universal topology-promotion gate. |
| `FULL_DECODE_ONLY`, 4,096/256, c=8 | 26.551 output tok/s, +0.28% versus eager | Declined for this prefill-heavy reference workload; retain `--enforce-eager` for that control. |
| `FULL_DECODE_ONLY`, 512/512, c=1 | Historical result: 58.18 versus 12.73 output tok/s eager (+357.1%); clean single-launch screen: 57.651 versus 12.707. **Confirmed** (3 independent launches per arm, both cells PASS, 0 failed requests): `confirm-dense-eager-c1` (eager baseline) 12.724873664265084 versus `confirm-graphs-full-decode-c1` (`FULL_DECODE_ONLY`) 56.336206381615746 output tok/s (+342.725%); p99 TPOT 79.0647771127532 versus 16.297473464836184 ms (-79.387%); p99 TTFT 772.2754819784313 versus 781.2450528336922 ms (+1.161%, within the 2% latency-regression gate). Mechanized `compare-cells` verdict: `IMPROVEMENT`, zero regressions, comparable `platform_sha256`. | Throughput/latency capacity gates confirmed and passing. CUDAGraph capture changes no numerics for this eager-vs-graph comparison, so there is no separate quality-gate requirement (unlike TurboQuant's KV-cache-dtype quantization) — promote `FULL_DECODE_ONLY` for this low-concurrency, decode-dominant c=1 workload. |
| `FULL_DECODE_ONLY`, 512/512, c=32 | 190.51 versus 188.33 output tok/s eager (+1.16%); full 32-token graph steps, no steady-state padding | Do not enable solely for high-throughput batching; graph launch savings are amortized at the full batch. |
| AWQ TP4, `FULL_DECODE_ONLY`, 512/512, c=1 | Historical result: graph 39.637 versus eager 10.629 output tok/s. Clean source required `--max-num-batched-tokens 512` even for eager; VLLM_COMPILE warmup hung with and without CUDAGraphs. | Do not enable AWQ graphs on gfx900. The only successful clean screen is eager, text-only AWQ at a 512-token batch cap (10.677 output tok/s). |
| TP8 decode-isolated burst, 4,096/256, 0.0827 RPS, burstiness 0.25 | Graph: 18.119 versus eager 18.064 output tok/s (+0.31%); p99 TPOT 126.93 versus 129.48 ms; p99 TTFT 2,431 versus 1,621 ms | Do not promote graph mode for burst/open-loop serving: it misses the p99 TTFT gate despite zero failures and a small TPOT reduction. |
| Repeated-prefix workload, TurboQuant, c=8 | 4,096 shared-prefix / 256 suffix / 128 output; prefix cache on: 36.029 output tok/s, 82.8% hit rate; off: 12.740 output tok/s | Enable `--enable-prefix-caching` only when the application has demonstrably repeated prefixes. The observed +182.8% output goodput and -77.1% mean TTFT exceed the workload-specific gate; it is not a generic latency claim. |
| Long input, c=32 capacity screen | Historical result: TurboQuant 22.092 versus auto 18.450 output tok/s (+19.7%). Clean single-launch screen: TurboQuant 20.521 versus auto 18.583 (+10.4%). **Confirmed** (3 independent launches per arm, all PASS, 0 failed requests): TurboQuant 22.505 versus auto 18.600 output tok/s (+20.997%); p99 TPOT 1377.553 versus 1672.270 ms (-17.624%); p99 TTFT 319853.228 versus 379302.669 ms (-15.673%). Mechanized `compare-cells` verdict: `IMPROVEMENT`, zero regressions, comparable `platform_sha256`. | Throughput/latency capacity gates confirmed and passing — clears the +10% capacity gate with no latency regression. TurboQuant's quality gates (perplexity, coding, needle-in-haystack) remain historical-only, not yet reconfirmed on this clean substrate: this is a deliberate open decision, not an oversight. Do not promote `turboquant_k8v4` as a default until quality reconfirmation completes. |
| Native MTP K=1 and CPU ngram K=4 | MTP: 22.376 output tok/s; ngram: 24.007 output tok/s, versus eager 26.476 | `DECLINED_NO_END_TO_END_WIN`: retain non-speculative serving. |

For the graph-promoted c=1 workload class, replace `--enforce-eager` with:

```bash
--compilation-config \
  '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16,32]}'
```

This remains deliberately a host/model-profile choice, not a source default.
The scheduler logged repeated steady-state c=32 decode steps in `FULL` mode at
size 32 with zero padding, so the near-neutral c=32 result is a performance
outcome rather than an observed graph fallback. A decode-isolated burst at 80%
of the confirmed c=8 request rate completed without failures but regressed p99
TTFT, so use this profile only for the demonstrated low-concurrency,
decode-dominant workload.

TurboQuant's coding control was 8/10 for both auto and
`turboquant_k8v4`; the same two prompts failed in both modes (one extractor
format failure and one missing-Node environment failure), so it added no new
failure. Both auto and TurboQuant passed the five-depth 32,768-context needle
test (5/5) when launched with `--max-model-len 34816`,
`--max-num-seqs 1`, `--max-num-batched-tokens 4096`, and
`--gpu-memory-utilization 0.75`. The initial 0.85 long-context TurboQuant
attempt OOMed during prefill; the bounded launch is therefore required for that
quality probe, not a throughput setting.

The initial WikiText-2 Parquet evaluation remains a loader/pre-fill numerical
smoke only: vLLM intentionally skips prefix-cache reads for `echo=true` prompt
logprobs. The valid cache-read gate uses a teacher-forced decode harness: each
of 50 fixed 512-token chunks prefills 256 tokens, forces the next 256 corpus
tokens, and records each raw target logit before masking the sampled output.
Across 12,800 scored tokens, auto PPL was 7.7323505 and
`turboquant_k8v4` PPL was 7.7242287 (-0.105%). This difference is
quality-neutral within measurement noise, not a claimed quality improvement.

Artifacts: prefix cache on/off
`/home/larkinwc/gfx900-runs/prefix-turboquant{,-off}/`; matched c=32 controls
`/home/larkinwc/gfx900-runs/capacity-{auto-one-wave,one-wave}/`; quality probes
`/home/larkinwc/gfx900-runs/quality-{auto,turboquant}-34k-gmu75/`; prefill-only
perplexity smoke controls `/home/larkinwc/gfx900-runs/quality-{auto,turboquant}-ppl/`;
valid cache-read controls
`/home/larkinwc/gfx900-runs/quality-{auto,turboquant}-cache-read-ppl/`; and the
promoted-workload torch-profiler capture
`/home/larkinwc/gfx900-runs/profile-turboquant-prefix/`.
The matched decode-heavy graph/eager controls and graph scheduler logs are at
`/home/larkinwc/gfx900-runs/graphs-c32-decode/{graph,eager}/`; the AWQ c=1
control is `/home/larkinwc/gfx900-runs/graphs-awq-decode/`; and the
DecodeBenchConnector burst controls are
`/home/larkinwc/gfx900-runs/graphs-burst-decode/`.

---

## Historical TL;DR — retained evidence, not the current default

**For maximum capability per GPU (the recommended default):**

```bash
# int4 weights (custom gfx900 GEMV) + TurboQuant KV cache, on one socket.
HIP_VISIBLE_DEVICES=0,1,2,3 VLLM_USE_V1=1 \
vllm serve QuantTrio/Qwen3.5-9B-AWQ \
  --tensor-parallel-size 4 \
  --dtype float16 \
  --kv-cache-dtype turboquant_k8v4 \
  --gpu-memory-utilization 0.90 \
  --language-model-only \
  --max-model-len 16384 \
  --max-num-seqs 32 \
  --compilation-config '{"mode":0,"cudagraph_mode":[2,0],"cudagraph_capture_sizes":[1]}'
# NOTE: use CUDA graphs (do NOT pass --enforce-eager) — 3.6x decode on gfx900 (#63).
# --max-num-seqs must be <= the model's Mamba/GDN cache-block budget or capture fails.
```

This stacks the gfx900 wins we built and validated (LLMM1 M=1 GEMV is on by default via `VLLM_ROCM_USE_SKINNY_GEMM=1`):

| Layer | Mechanism | Win | Issue |
|---|---|---|---|
| **Weights** | int4 AWQ via hand-written gfx900 GEMV (sidesteps MFMA-less `tl.dot`) | **~1.5× decode vs FP16 on half the GPUs** | #57 |
| **All M=1 FP16 GEMMs** | LLMM1 skinny GEMV (lm_head + qkv/o/gate_up projections) | **2.2× FP16 decode (150→67 ms/step)** | #59 |
| **Per-step dispatch** | HIP CUDA graphs (FULL_DECODE_ONLY) once GPU-busy is small | **3.6× decode on top of LLMM1** | #63 |
| **KV cache** | TurboQuant `turboquant_k8v4` (FP8 keys + 4-bit values) | **2.34× context/concurrency per VRAM, ~0 decode cost** | #62 |

Both verified active together (coherent output; decode 9.97 tok/s ≈ standalone AWQ
10.46). They are orthogonal and compose cleanly.

---

## Why these specific choices (each is measured, not assumed)

### 1. `--dtype float16` (never bf16)

gfx900 has no usable bf16 path for this model. FP16 is mandatory.

### 2. CUDA graphs ON (do NOT use `--enforce-eager`) — updated, see #63

HIP CUDA graphs give a **3.6× decode speedup** on gfx900 (eager 67 → graph 18.4
ms/tok) once the LLMM1 GEMV (#59) shrinks GPU-busy time so the host dispatch gap
(59% of the step) dominates. Use `mode=NONE` (Inductor/compile stays off — it
crashes on ROCm) + `cudagraph_mode=FULL_DECODE_ONLY` + `capture_sizes=[1]`.
**Constraint:** Qwen3.5 is hybrid (Mamba/GDN); `--max-num-seqs` must be ≤ the
available Mamba cache blocks or capture fails. (Earlier docs said eager was best —
that was true before LLMM1; the bottleneck moved. See `PERF_GFX900.md` § #63.)

### 3. `--kv-cache-dtype turboquant_k8v4` (FP8 keys — NOT a K-quant preset)

Qwen3 family is **"quirky-K"**: key-side quantization is catastrophic
(`tq2k` = +150% PPL on qwen3:8b — output destroyed). `k8v4` keeps keys at FP8
(effectively unquantized) and only compresses values. **Do not** use
`turboquant_3bit_nc` / other aggressive-K presets on Qwen3.5.

**Quality is measured, not assumed.** The older 24-window prefill-logprob
numbers are a loader/numerics smoke only because that API skips prefix-cache
reads. The valid c4130-2 TP8 Qwen3.5-9B gate teacher-forced 256 decode tokens
after each 256-token prefill across 50 fixed WikiText-2 chunks (12,800 scored
tokens): auto PPL = 7.7323505 and `turboquant_k8v4` PPL = 7.7242287
(-0.105%). The result is quality-neutral within measurement noise, not a
quality improvement, and is below the +1% gate. It qualifies the preset as a
capacity-oriented profile option, not a safe global production default.

### 4. int4 weights via AWQ (`QuantTrio/Qwen3.5-9B-AWQ`)

Stock INT4 on gfx900 is *slower* than FP16 because the Triton path dequants then
runs `tl.dot` through a padded FP32 GEMM (no MFMA). Our hand-written M≤8 GEMV
(#57) sidesteps `tl.dot` entirely → 2.8–4.0× the MLP matmuls vs FP16, 10–15× vs
the stock int4 path. It auto-dispatches when `on_gfx900()` and `group_size ∈
{32,64,128}` and `M≤8`; high-batch falls back to FP16 automatically.

### 5. Layout: TP within one socket

GPUs 0–7 = socket 0, 8–15 = socket 1. **Cross-socket P2P is 0.26 GB/s** (Xeon
E5 v3 QPI HW limit) vs ~8 GB/s intra-socket. **Never span sockets in one TP/PP
group.** Use `HIP_VISIBLE_DEVICES=0,1,2,3` (or 0–7) for socket 0.

---

## Choosing a layout for your workload

Decode does **not** scale with bandwidth here (it sits at ~5–9% of the HBM
roofline, gated by per-token all-reduce + GDN kernel latency). So:

| Goal | Recommended layout | Why |
|---|---|---|
| **Single-stream latency** | TP4 (or TP2) one socket | TP8 only ~1.7× TP4 — diminishing |
| **Aggregate throughput** | **batch/concurrency** is the only real lever — c1→c4 gives 2.5–3.4× | decode per-step time is ~flat with batch up to ~64–96 |
| **Decode-optimized serving** | **TP2×PP4** within one socket | PP removes the per-layer all-reduce that dominates decode latency; best aggregate we measured (~26 tok/s, c4 coding) |
| **Max context per VRAM** | add `turboquant_k8v4` | 2.34× KV capacity |

**Realistic planning numbers** (not the parallel spec): ~24 TFLOPS/socket usable
prefill; decode in the **tens of tok/s aggregate**. Prefill scales well (~70%
efficiency, TP-friendly); decode is the bottleneck.

---

## VRAM ceiling — important

Each die is **~8 GiB**. AWQ(TP2) + 16k-token KV **OOMs at batch>1**. The
compression lets you push context/concurrency further per GiB, but does not remove
the per-die cap. For long context **and** batching with the AWQ model, use **TP4**
(more aggregate VRAM) or a shorter `--max-model-len`.

---

## Model facts that drive all of the above (Qwen3.5-9B)

- 9.65B params (19.3 GB FP16), 32 layers, hidden 4096, vocab 248320, 16 attn / 4 KV heads.
- **Hybrid attention**: `full_attention_interval=4` → only **8/32 layers** are full
  attention with a KV cache; the rest are **GDN linear-attn** (no KV cache). So the
  KV footprint is already small — TurboQuant's 2.34× applies to that 1/4 of layers.
- GDN **decode** uses `fused_recurrent` (0 `tl.dot`) — already gfx900-appropriate.
  The `tl.dot`-heavy FLA ops are **prefill-only**.
- Dense hybrid, **not** MoE. Use `--language-model-only`.

---

## Operational notes

- **Cold start is slow** (~647s) due to GDN/FLA Triton autotune + TurboQuant
  centroid init; warm is ~60s. Autotune results cache.
- `amdgpu.reset_method=2` (mode1) is set so a GPU hang auto-recovers instead of
  needing a reboot. (`/etc/modprobe.d/amdgpu-reset.conf`; reversible.)
- Pure-Triton TurboQuant kernels run on wave64 — the wave32 `__shfl_sync` blocker
  that stops the llama.cpp HIP port does **not** apply to us.

---

## Second model: Qwen3-Coder-Next (sparse MoE, 80B total / 3B active)

`bullpoint/Qwen3-Coder-Next-AWQ-4bit` runs well on gfx900. It is a **hybrid
sparse-MoE**: 48 layers of `3×(GatedDeltaNet→MoE) + 1×(GatedAttn→MoE)`, 512
experts (10 active + 1 shared) at expert-intermediate 512, AWQ 4-bit group_size 32
(compressed-tensors / pack-quantized). ~45 GB of weights. Gated-attn here is
head_dim 256, 16 Q / 2 KV heads (different from Qwen3.5-9B).

**Recommended run (TP4×PP2 on one socket, graphs, optional TurboQuant):**

```bash
# 8 dies, one socket. TP stays at the efficient 4; shallow PP2 holds the ~45 GB.
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 VLLM_USE_V1=1 \
vllm serve bullpoint/Qwen3-Coder-Next-AWQ-4bit \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --no-async-scheduling \
  --dtype float16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --trust-remote-code \
  --kv-cache-dtype turboquant_k8v4 \
  --compilation-config '{"mode":0,"cudagraph_mode":[2,0],"cudagraph_capture_sizes":[1,8,16,32]}'
# PP needs --no-async-scheduling in server mode.
# Drop --kv-cache-dtype to use FP16 KV (slightly faster single-stream, half the KV capacity).
```

### TP4×PP2 vs TP8 — pick by workload

The ~45 GB of weights do not fit a single-socket TP4 (8 GiB/die × 4), so the two
viable 8-die layouts are **TP4×PP2** (shallow PP to span 8 dies, TP stays at the
efficient width 4) and **TP8**. Both stay inside socket 0 (never cross the
0.26 GB/s QPI link). Clean A/B (graphs, identical settings):

| Layout | c=1 | c=8 |
|---|---|---|
| **TP8** | **33.2** | 42.3 |
| **TP4×PP2** | 23.2 | **44.9** |

- **Single-stream / latency:** use **TP8** (+43% at c=1). It is one pipeline stage,
  so it avoids the PP bubble — at bs=1 the decode profile shows TP4×PP2 spends
  ~80 ms/step waiting for the other PP stage, which costs more than TP8's heavier
  8-way all-reduce. (TP8 needs gpu_mem ~0.92 for KV headroom; 0.95 OOMs.)
- **Throughput / concurrent serving:** use **TP4×PP2** — concurrency fills the PP
  bubble and TP4's cheaper 4-way all-reduce then wins (c≥8), scaling to 114 tok/s
  at c=32.

(This inverts the dense-model result where TP4 beat TP8: for the MoE the choice is
TP8 vs TP4×*PP2*, so it is the 8-way-all-reduce penalty vs the PP-bubble penalty,
and at bs=1 the bubble is the bigger cost. See issue #65.)

### Measured decode progression (bs=1, TP4×PP2)

| config | tok/s | note |
|---|---|---|
| eager | 8.5 | baseline |
| + CUDA graphs (#63) | 19.3 | 2.3× |
| + small-m GEMV fix (#65) | **26.8** | **3.2× total** |

The small-m GEMV fix (`shared_expert_gate` is a `Linear[2048→1]`, m=1, which
missed LLMM1's `m%4==0` gate and fell to a wasteful 64×64 rocBLAS macrotile)
routes that scalar projection to a fused multiply-reduce. +39% MoE decode; dense
Qwen3.5-9B is unaffected (the gate is narrow). The MoE expert path itself
(`fused_moe_kernel_gptq_awq`, Triton, ~177 µs/call on wave64) is healthy and not
the bottleneck.

### Throughput scales strongly with concurrency

| c | FP16 KV tok/s | turboquant_k8v4 tok/s |
|---|---|---|
| 1 | 26.7 | 23.8 |
| 8 | 48.3 | 51.7 |
| 16 | 79.6 | 88.7 |
| 32 | **113.7** | **114.0** |

**4.3× aggregate from c=1→32.** Concurrency is the real throughput lever (decode
per-step time is ~flat with batch), exactly as for the dense model.

### TurboQuant on this MoE: throughput-neutral, 2.5× KV capacity

TurboQuant is the same lossless quirky-K preset (FP8 keys + 4-bit values) and is
**throughput-neutral** here (table above). Its payoff is **KV-cache capacity**:

| KV dtype | KV cache tokens | Max concurrency @ 32K ctx |
|---|---|---|
| auto (FP16) | 62,100 | 1.90× |
| **turboquant_k8v4** | **156,483** | **4.78×** |

**2.52× more KV capacity** → 4.78× vs 1.90× concurrent 32K-context streams on the
same VRAM. Use it whenever long context or high concurrency matters; otherwise
FP16 KV is marginally faster at single-stream. (No code change was needed — the TQ
backend already uses the gfx900-safe SDPA prefill fallback, since
`is_flash_attn_varlen_func_available()` is False on this box.)

### Cold start

~759 s first run (MoE + GDN + TQ Triton autotune), ~156 s warm (autotune caches).

### Repro

- `bench_scripts/coder_tput.py {auto|turboquant_k8v4}` — concurrency sweep + KV size.
- `bench_scripts/coder_smoke.py`, `coder_prof.py` — smoke + decode profile.

## Repro / benchmarks

- `bench_scripts/tq_measure.py {auto|turboquant_k8v4}` — KV size + decode tok/s.
- `bench_scripts/combo_dec.py turboquant_k8v4` — int4 weights + TQ KV together.
- `bench_scripts/batch_sweep.py` — concurrency scaling.
- Deep dives: `PERF_GFX900.md`, `BENCH_GFX900.md`, `GFX900_SETUP.md`.
