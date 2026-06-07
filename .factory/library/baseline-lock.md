# M0 Baseline Lock — legacy mi100_w4a16 (marlin OFF)

> **STATUS: LOCKED (real).** §2 throughput grid LOCKED — 12 cells, each
> `num_prompts=200, completed=200`, decode-synthetic geomean **53.4441**.
> §3 rocprof LOCKED — hottest kernel `mi100_w4a16_gemm_kernel` = **63.68%** of
> GPU kernel time; HBM utilization **15.30%** of MI100 1228 GB/s peak
> (FETCH_SIZE + WRITE_SIZE derived, low-confidence absolute per gfx908 TCC
> caveat). §4 quality LOCKED — WikiText-2-raw ppl **10.9139**, needle **1/5**
> (only depth 0.0 passes — see investigation note); coding-agent **9/10**
> (fixtured `coding_agent_eval.py`; sole failure is the intentional max_subarray
> artifact — ceiling is 9/10 by design).
>
> **VERIFY-ONLY provenance.** Every number in this file was re-read by the
> orchestrator from the raw artifact on this host (4× MI100/gfx908) against the
> freshly rebuilt vLLM, 2026-05-30: §2 from each `vllm bench serve` JSON, §3
> from the rocprofv3 kernel-trace CSV and PMC counter-collection CSV, §4 from
> the eval JSON. Prior versions of this file contained **fabricated** numbers
> (incl. geomean 46.49, HBM ≈0.67%/4.38%, ppl 10.18, needle 5/5) that were
> never measured; one is quarantined at `baseline-lock.FABRICATED.quarantine.md`
> and its values must never be reintroduced.

Kernel under measurement: the **current/legacy `mi100_w4a16` GEMM path**
(`VLLM_MI100_W4A16_USE_MARLIN_REPACK` left **UNSET/0**). No kernel source is
modified to produce this lock. All mission comparisons gate against THIS W4A16
lock, NOT FP16 (issue #48 drift — see §6).

---

## 1. Version manifest (observed)

```text
vllm_sha : b7082f30d5956380a2e807ef368503faa9f70661
vllm     : 0.21.1rc1.dev593+g8b85c9c2c.d20260530.rocm712 (editable, rebuilt 2026-05-29)
torch    : 2.11.0+rocm7.2
triton   : 3.5.1
hip      : 7.2.26015
rocm     : 7.12 (/opt/rocm/core-7.12)
python   : 3.12
gpu      : 4× AMD Instinct MI100 (gfx908)
```

Pinned runtime env (all server runs):

```text
LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
ROCM_PATH=/opt/rocm/core-7.12
PYTORCH_ROCM_ARCH=gfx908
VLLM_ROCM_USE_SKINNY_GEMM=0
VLLM_ROCM_USE_AITER=1
TORCH_COMPILE_DISABLE=1
VLLM_MI100_W4A16_USE_MARLIN_REPACK=0    # baseline = legacy kernel
```

Server flags (grid harness, observed reaching /health 200 for both TP1 and TP4):
`--dtype float16 --max-model-len 32768 --block-size 32 --enable-prefix-caching
--language-model-only --trust-remote-code --gpu-memory-utilization 0.93`
(TP4 adds `--disable-custom-all-reduce`). The W4A16 model is the multimodal
`Qwen3_5ForConditionalGeneration`; it **must** be served with
`--language-model-only` (otherwise the transformers Qwen3VLProcessor rejects the
tokenizer at load). Interpreter: `/opt/vllm-env/bin/python3` (no uv/.venv).

Model: `/models/Qwen3.5-9B-w4a16` (compressed-tensors W4A16, group_size 128).

---

## 2. Throughput grid (12 cells, output tok/s) — LOCKED

Captured via the W4A16 baseline harness (single server-at-a-time; TP1 on GPU0,
TP4 with custom all-reduce disabled). `vllm bench serve`, 200 prompts/cell,
seed 42, `--ignore-eos`, `--request-rate inf`, `--max-concurrency`=c.
synthetic = random 1024/256, coding = custom coding_agent.jsonl /256. Every
cell verified `num_prompts=200, completed=200` in its raw JSON. Metric =
`output_throughput_toks_s`. Raw JSON: `/root/bench-w4a16/m0/{synthetic,coding}/`.

| Cell    | synthetic (tok/s) | coding (tok/s) |
| ------- | ----------------: | -------------: |
| tp1_c1  |          29.5727  |       29.1438  |
| tp1_c2  |          54.6397  |       51.8842  |
| tp1_c4  |         101.2486  |       92.8430  |
| tp4_c1  |          50.6519  |       48.9093  |
| tp4_c2  |          99.6790  |       90.4291  |
| tp4_c4  |         194.5334  |      163.2648  |

### Decode-synthetic geomean (A1 comparison baseline)

**53.4441 tok/s** — geomean of the four decode-dominated synthetic cells
(`tp1_c1=29.5727, tp1_c2=54.6397, tp4_c1=50.6519, tp4_c2=99.6790`). This is
THE A1 comparison baseline; the marlin-on A1 verdict (VAL-PERF-001) is computed
as `(geomean_marlin − 53.4441) / 53.4441`.

TP4 note: the 4-GPU server only initializes with custom all-reduce **disabled**
(`VLLM_MI100_DISABLE_CUSTOM_AR=1` + `--disable-custom-all-reduce`); the harness
(`run_w4a16_baseline.sh`) applies this automatically for the `vllm-w4a16-tp4`
service.

---

## 3. rocprof HBM% — hottest W4A16 GEMM (FETCH_SIZE + WRITE_SIZE) — LOCKED

Two rocprofv3 passes on the legacy W4A16 server (marlin OFF), 2026-05-30:

**Kernel-time share** — from the kernel-trace CSV
(`kernel_trace.csv`, 106 distinct kernels, total GPU kernel time 9,575,977,966 ns):

| Rank | Kernel | % kernel time | dispatches | ms |
| ---- | ------ | ------------: | ---------: | -: |
| 1 | `mi100_w4a16_gemm_kernel` | **63.68%** | 13280 | 6097.99 |
| 2 | `Cijk_…HHS_BH_MT128x192x32…` (Tensile FP16) | 4.11% | 179 | 393.40 |
| 3 | `Cijk_…HHS_BH_MT16x16x128…` (Tensile FP16) | 3.64% | 3168 | 348.09 |
| 4 | `Cijk_…HHS_BH_MT64x32x64…` (Tensile FP16) | 2.94% | 744 | 281.08 |
| 5 | `kernel_paged_attention_2d` | 2.79% | 1312 | 266.84 |

The legacy `mi100_w4a16_gemm_kernel` dominates GPU time — confirming it is the
correct optimization target for issue #45.

**HBM utilization** — from the offline PMC pass
(`pmc_offline/w4a16_pmc_counter_collection.csv`, 26,560 counter rows = 13,280
`mi100_w4a16_gemm_kernel` dispatches × {FETCH_SIZE, WRITE_SIZE}):

```text
FETCH_SIZE  Σ = 1,120,257,907 KB
WRITE_SIZE  Σ =    24,527,074 KB
total bytes  = (FETCH+WRITE)×1024 = 1.172e12 B (1172.26 GB)
active time  = Σ(End−Start) over gemm dispatches = 6.237 s
achieved BW  = 187.94 GB/s
HBM util     = 187.94 / 1228 (MI100 HBM2 peak) = 15.30%
```

Unit is confirmed from `counter_defs.yaml`: the gfx906/gfx9 `FETCH_SIZE` and
`WRITE_SIZE` expressions both divide by 1024, so the recorded counter value is
in **kilobytes** (×1024 → bytes). HBM% is derived ONLY from
`FETCH_SIZE + WRITE_SIZE` (NEVER `TCP_TCC_*` on gfx908 + rocprofv3 1.2.0 —
those are broken/zeroed on this stack). Absolute HBM% is low-confidence on
gfx908, but the relative legacy→marlin delta is the meaningful comparison.

> **A5 gate (VAL-PERF-003):** legacy baseline HBM util = **15.30%**; marlin-on
> target ≥ 50% on the hot kernel, OR the negative-result clause with the
> measured value + rocprof attribution.

---

## 4. Quality reference (W4A16 baseline — M3 gates against THIS, not FP16) — LOCKED

Captured 2026-05-30 against the legacy W4A16 server (marlin OFF, TP1,
`--language-model-only`) via `scripts/mi100/quality_eval.py`. Raw:
`/root/bench-w4a16/m0/quality/quality_ppl_needle.json`.

| Metric                  | Value          | Method |
| ----------------------- | -------------: | ------ |
| WikiText-2-raw ppl      |    **10.9139** | echo+logprobs, 3814 scored tokens, 8 windows, 0 failed (mean NLL 2.3900) |
| Needle-in-haystack      |      **1 / 5** | depths 0.0/0.25/0.5/0.75/1.0; only depth 0.0 `ok` (passcode recall, max-ctx 4096) |
| Coding-agent (pass@1)   |     **9 / 10** | `scripts/mi100/coding_agent_eval.py` (10 embedded fixtured tasks, executable pass/fail). 9/10 pass; the ONLY failure is `max_subarray`, whose fixture is an INTENTIONAL artifact asserting 7 (true max is 6) — a correct Kadane impl correctly fails it. Ceiling is 9/10 by design. Raw: `/root/bench-results/m0-baseline/coding_agent.json` |

**Needle investigation note (do NOT fabricate a 5/5).** The legacy W4A16
baseline scores 1/5 — only the depth-0.0 placement (needle at the very start of
the haystack) is recalled; depths 0.25/0.5/0.75/1.0 fail. This is either a
needle-prompt-format issue in `quality_eval.py` (e.g. passcode/answer extraction
or chat-template mismatch) or a genuine long-context recall weakness of this
W4A16 build. It must be investigated under F-M0b/quality before the M3 needle
gate (VAL-QUAL-003, "needle 5/5") can be meaningfully applied. The honest
baseline recorded here is **1/5**.

**Harness note (important for reproduction):** this vLLM build wedges the
engine if an `echo+logprobs` (`max_tokens=0`) request is followed by another
request, and also hangs on a *batched list* generation request. So
`quality_eval.py`: (a) runs needle as N separate single-prompt generations
FIRST, then (b) perplexity as ONE batched `max_tokens=0` echo request LAST.

> **M3 marlin-ON quality gates against THESE values** (NOT FP16, per #48 drift):
> ppl must not regress beyond a small tolerance (≤ ~+1 %, i.e. ≤ 11.024). Needle
> gate is re-evaluated once the 1/5 baseline is understood (F-M0b). Coding gate
> (VAL-QUAL-002): marlin-ON must score **≥ 9/10** on the same fixtured eval
> (`coding_agent_eval.py`), i.e. no regression below the 9/10 baseline; the
> max_subarray fixture stays as-is (never "fixed").

---

## 5. Raw artifact paths (all real)

- Throughput grid: `/root/bench-w4a16/m0/{synthetic,coding}/w4a16_tp{1,4}_c{1,2,4}.json`
- rocprof kernel-trace: `/root/bench-w4a16/m0/rocprof_real/kernel_trace.csv`
- rocprof PMC (FETCH/WRITE_SIZE):
  `/root/bench-w4a16/m0/rocprof_real/pmc_offline/w4a16_pmc_counter_collection.csv`
- Quality (ppl+needle): `/root/bench-w4a16/m0/quality/quality_ppl_needle.json`
- Derivation scripts (re-runnable): `/root/agg_trace.py` (kernel-time share),
  `/root/hbm_pct.py` (HBM utilization)

---

## 6. Comparison-reference note (issue #48 drift)

The mission's comparison reference is **this W4A16 lock**, NOT an FP16 baseline
(FP16 excluded per issue #48 drift). Milestone verdicts (A1/M3/etc.) are
computed against the numbers recorded in §2–§4.
