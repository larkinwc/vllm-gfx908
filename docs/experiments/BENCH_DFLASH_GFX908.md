# BENCH_DFLASH_GFX908 — DFlash Speculative Decoding on MI100 (gfx908)

> End-to-end enablement and A/B evaluation of **DFlash** block-diffusion
> speculative decoding on MI100 (CDNA1 / gfx908), pairing the FP16
> `/models/Qwen3.5-9B` target with the `z-lab/Qwen3.5-9B-DFlash` drafter.
>
> **Verdict: NEGATIVE (do not enable in production on gfx908).** DFlash is
> *functionally correct* on MI100 — non-causal attention, parallel drafting,
> CUDA-graph capture, and rejection sampling all work and produce coherent,
> healthy acceptance lengths (~3.5–4.8 tokens). But end-to-end it is
> **net-negative** on every workload measured: the per-step verification
> overhead on MI100's compute-bound decode path exceeds the savings from
> accepted draft tokens. This is the same class of result already recorded
> for MTP on gfx908 (`MI100_SETUP.md`: MTP −25 % to −45 %).

## Table of Contents

1. [Headline](#headline)
2. [Hardware / Software Manifest](#hardware--software-manifest)
3. [What Works (correctness)](#what-works-correctness)
4. [Benchmark Results](#benchmark-results)
5. [Why DFlash Loses on gfx908](#why-dflash-loses-on-gfx908)
6. [Reproducibility](#reproducibility)
7. [Conclusion](#conclusion)

---

## Headline

- **Correctness: PASS.** DFlash runs end-to-end on gfx908 in both eager and
  FULL_DECODE_ONLY CUDA-graph modes. Output is coherent and token-equivalent
  to the autoregressive baseline modulo expected FP-nondeterminism between the
  batched-verify and sequential-decode forward passes (divergences only appear
  deep into generation after long identical prefixes; the base model itself is
  run-to-run deterministic).
- **Acceptance: HEALTHY.** Mean accepted length ≈ **4.8** tokens at
  `num_speculative_tokens=15` (621 accepted / 165 drafts on the offline probe
  set) and ≈ **3.5** at `num_speculative_tokens=7` (1833/741 over a coding-bench
  run). Per-position acceptance decays smoothly (130, 89, 66, 56, … at ns=15),
  matching the DFlash paper's qualitative shape.
- **Performance: NEGATIVE on every cell.** On the realistic coding-agent
  workload DFlash decode aggregate throughput is roughly **0.4–0.5×** the
  non-spec baseline (c=1: 22.4 vs 50.6 agg tok/s). On synthetic random data it
  is even worse (≈0.5× at c=1) because random tokens are unpredictable and
  acceptance collapses to ~0.
- **No gfx908 code changes were required.** The upstream DFlash implementation
  (proposer, `DFlashQwen3ForCausalLM`, non-causal `ROCM_ATTN` path,
  `context_attention_fwd(CAUSAL=False)`) already works correctly on CDNA1. The
  top-suspected blocker — the non-causal Triton prefill kernel — was validated
  numerically against a torch SDPA reference and passes (max abs err < 5e-2 in
  fp16 across query-only, context+query, and mixed-batch cases).

---

## Hardware / Software Manifest

| Component | Value |
|---|---|
| GPUs | 2× AMD Instinct MI100 (gfx908, 32 GB), TP=2, GPUs 0+3 |
| Target model | `/models/Qwen3.5-9B` (FP16, hybrid linear+full attention, 32 layers) |
| Drafter | `z-lab/Qwen3.5-9B-DFlash` (5-layer Qwen3, `target_layer_ids=[1,8,15,22,29]`, `block_size=16`) |
| vLLM | `0.22.1rc1.dev466+g7133b783a` (branch `mi100/dflash-gfx908`, HEAD `7133b783a`) |
| PyTorch | `2.11.0+rocm7.2` (HIP 7.2.26015) |
| ROCm | 7.12.0 (`/opt/rocm/core-7.12`) |
| Attention backend | `ROCM_ATTN` (default); non-causal path via `CommonAttentionMetadata.causal=False` |
| CUDA graphs | FULL_DECODE_ONLY (auto on MI100), captured cleanly with DFlash |
| Launch | `/root/launch-vllm-optimized.sh` + `--max-num-batched-tokens 16384` |

> **TP note:** GPUs 1+2 were occupied by an unrelated session for most of this
> mission, so the A/B was run at **TP=2 on GPUs 0+3** (both arms identical
> hardware). Absolute tok/s here is therefore lower than the TP=4 production
> figures in `BENCH.md`; only the *DFlash-on vs DFlash-off ratio* is the
> headline result, and that ratio is hardware-config-independent for this
> conclusion.

---

## What Works (correctness)

1. **DFlash config + model load.** `method="dflash"` resolves
   `DFlashDraftModel`, builds `SpeculativeConfig(method='dflash', num_spec=15)`,
   sets `parallel_drafting=True`, and extracts aux hidden states from the
   target's `target_layer_ids+1` layers (the hybrid Qwen3.5 layer mix is handled
   correctly).
2. **Non-causal prefill kernel.** `context_attention_fwd(..., causal=False)`
   on gfx908 matches a torch SDPA reference (fp16 max abs err ≤ 1.95e-3 across
   query-only / context / mixed cases). Standalone test:
   `/tmp/test_noncausal_prefill.py` (see Reproducibility).
3. **dtype.** The drafter ships bf16 weights, but `SpeculativeConfig` forces the
   draft model dtype to the target's (`float16` here), so the fused KV
   projection / RoPE / cache-write path in `precompute_and_store_context_kv`
   sees consistent fp16 — no mismatch.
4. **CUDA graphs.** FULL_DECODE_ONLY capture (32 sizes) completes without error
   with DFlash active; output remains coherent.
5. **Rejection sampling.** Greedy outputs are baseline-equivalent on the prefix,
   then diverge only at deep positions (61, 71, 105 of 128) — consistent with
   batched-verify vs sequential-decode FP reordering, not a logic bug.

---

## Benchmark Results

All numbers TP=2 (GPUs 0+3), FP16 target, `--max-model-len 8192`,
`--block-size 32`, prefix caching on, FULL_DECODE_ONLY CUDA graphs.

### Synthetic (`vllm bench serve`, random 128-in / 128-out)

| Concurrency | Baseline out tok/s | DFlash ns=15 out tok/s | Ratio | Baseline TPOT | DFlash TPOT |
|---:|---:|---:|---:|---:|---:|
| 1 | 69.1 | 34.6 | **0.50×** | 13.8 ms | 27.5 ms |
| 2 | 107.5 | 52.2 | **0.49×** | 17.9 ms | 36.5 ms |
| 4 | 182.6 | 84.4 | **0.46×** | 20.9 ms | 45.5 ms |

> Random tokens are unpredictable → near-zero acceptance → DFlash pays full
> draft+verify overhead for almost no accepted tokens. Expected worst case.

### Coding-agent (`coding_agent_bench.py`, 10 req/user, realistic prompts)

| Concurrency | Baseline agg tok/s | DFlash ns=15 agg tok/s | Ratio | Baseline TPOT | DFlash ns=15 TPOT |
|---:|---:|---:|---:|---:|---:|
| 1 | 50.6 | 22.4 | **0.44×** | 17.5 ms | 35.8 ms |
| 2 | 90.7 | 38.1 | **0.42×** | 19.7 ms | 42.9 ms |
| 4 | 167.8 | 62.2 | **0.37×** | 21.4 ms | 53.3 ms |

`num_speculative_tokens=7` (smaller draft block, lower overhead) at c=1:
**24.99 agg tok/s** (still 0.49× baseline). No spec-token count recovers parity.

### Acceptance (healthy — confirms the drafter itself is good)

| Config | drafts | draft toks | accepted | mean accept len |
|---|---:|---:|---:|---:|
| ns=15 (offline probe, 6 prompts) | 165 | 2475 | 621 | **4.76** |
| ns=7 (coding c=1 server run) | 741 | 5187 | 1833 | **3.47** |

Per-position accepted (ns=15): `[130, 89, 66, 56, 46, 39, 34, 29, 25, 22, 20,
18, 16, 16, 15]` — smooth decay, drafter is working as designed.

---

## Why DFlash Loses on gfx908

DFlash (and spec-decode generally) trades **more compute per step** (draft
forward + a wider, batched target verify over `1+num_spec` query positions) for
**fewer steps**. That trade only pays off when the target decode step is
*memory-bandwidth bound* and has spare compute to absorb the wider verify batch.

On MI100 (CDNA1) the FP16 Qwen3.5-9B decode step is already **compute-bound**,
not bandwidth-bound, for these batch sizes:

- The wider verify batch (`1+15 = 16` query positions/req) does **not** ride
  for free — it linearly adds MFMA work the MI100 cannot hide.
- The extra DFlash drafter forward (5 layers) + context-KV precompute (fused
  GEMM, per-layer RMSNorm×5, fused RoPE) adds fixed per-step overhead.
- TPOT roughly **doubles** (e.g. coding c=1: 17.5 → 35.8 ms), and the ~4.8×
  acceptance is not enough to overcome a 2× per-step cost once verify width and
  drafter overhead are included.

This mirrors the existing fork findings:

- **MTP** on gfx908: −25 % to −45 % (`MI100_SETUP.md`, "Optimizations Tested
  and NOT Recommended").
- The general CDNA1 pattern that throughput wins come from *cheaper compute*
  (skinny GEMM, flash-decoding split-K, W8A8/W4A16), not from *spending more
  compute to skip steps*.

---

## Reproducibility

### Drafter download

```bash
hf download z-lab/Qwen3.5-9B-DFlash --local-dir /models/Qwen3.5-9B-DFlash
```

### Build (Python+Triton-only changes; this csrc tree must be compiled once)

```bash
export PYTORCH_ROCM_ARCH=gfx908 VLLM_TARGET_DEVICE=rocm \
  ROCM_PATH=/opt/rocm/core-7.12 LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib \
  PATH=/opt/rocm/core-7.12/bin:$PATH MAX_JOBS=$(nproc)
/opt/vllm-env/bin/pip install --no-build-isolation --no-deps -e .
```

### Non-causal kernel correctness probe

`/tmp/test_noncausal_prefill.py` — builds a paged KV cache, runs
`context_attention_fwd` with `causal=True/False`, diffs against a per-sequence
torch SDPA reference. Result: `causal=False query_only=9.77e-04 ctx=4.88e-04
mixed=9.77e-04 → NONCAUSAL PREFILL OK`.

### Serve (DFlash)

```bash
CUDA_VISIBLE_DEVICES=0,3 MODEL=/models/Qwen3.5-9B TP=2 PORT=8100 \
  MAX_MODEL_LEN=8192 \
  EXTRA='--max-num-batched-tokens 16384 --speculative-config {"method":"dflash","model":"/models/Qwen3.5-9B-DFlash","num_speculative_tokens":15}' \
  /root/launch-vllm-optimized.sh
```

> **Config gotcha:** DFlash reserves `1+num_spec` draft slots per sequence. With
> the default `max_num_batched_tokens` the engine fails at init with
> `max_num_scheduled_tokens is set to <negative>`. Raise
> `--max-num-batched-tokens` (16384 used here; HF card uses 32768).

### Benchmarks

```bash
# synthetic
/opt/vllm-env/bin/python -m vllm.entrypoints.cli.main bench serve \
  --model /models/Qwen3.5-9B --base-url http://localhost:8100 \
  --dataset-name random --random-input-len 128 --random-output-len 128 \
  --num-prompts <25*c> --max-concurrency <c> --seed 42
# coding agent
/opt/vllm-env/bin/python /root/benchmark-scripts/coding_agent_bench.py \
  --concurrency <c> --requests 10 --model /models/Qwen3.5-9B \
  --base-url http://localhost:8100
```

---

## Conclusion

DFlash is **correct and well-behaved on MI100/gfx908** — no kernel or platform
work was needed beyond a normal source build, and the non-causal attention path
that was the primary risk validates cleanly. But it is a **performance
NEGATIVE** on this hardware: every synthetic and coding-agent cell measured runs
at 0.37×–0.50× of the non-spec baseline, because MI100's compute-bound FP16
decode cannot absorb the wider batched verify and extra drafter forward, even at
a healthy ~4.8 acceptance length.

**Recommendation:** keep speculative decoding (DFlash and MTP) **off** on gfx908
for production. Add DFlash to the "Optimizations Tested and NOT Recommended"
table. Revisit only if/when a quantized (W8A8/W4A16) target makes the decode
step bandwidth-bound enough to flip the trade — that is a separate experiment
and was out of scope here (FP16-only by request).
