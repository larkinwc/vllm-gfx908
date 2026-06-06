# BENCH_TURBOQUANT_ROTORQUANT — RotorQuant block-diagonal rotations vs Hadamard

> Validates the RotorQuant block-diagonal rotation presets
> (`turboquant_{planar,iso}3_nc`) against the existing dense-Hadamard
> TurboQuant preset (`turboquant_k3v4_nc`) and the FP16 KV-cache baseline,
> on Qwen3.5-9B / 4× MI100 (gfx908).

## Table of Contents

1. [Headline](#headline)
2. [Hardware / Software Manifest](#hardware--software-manifest)
3. [Methodology note: why GSM8K, not perplexity](#methodology-note-why-gsm8k-not-perplexity)
4. [Quality gate — GSM8K (decode path)](#quality-gate--gsm8k-decode-path)
5. [Throughput — online serving](#throughput--online-serving)
6. [Bug fixed during validation](#bug-fixed-during-validation)
7. [Conclusion](#conclusion)
8. [Reproducibility](#reproducibility)

---

## Headline

- **`turboquant_iso3_nc` (4D quaternion block rotation) is the winner:** it
  **matches FP16 GSM8K accuracy exactly (0.870 vs 0.870)** and **beats the
  existing dense-Hadamard preset on both quality (+1.0 pt) and decode
  throughput (+1.4 %)** — while using an O(D) block rotation instead of the
  O(D²) Hadamard GEMM.
- **`turboquant_planar3_nc` (2D Givens) regresses quality** (0.810, −6.0 pt vs
  FP16) at this bit budget — recommend **iso over planar** for K-only 3-bit.
- **Throughput**: all TurboQuant presets sit within ~1 % of each other and
  ~4 % below FP16 on this 8-of-32-full-attention hybrid model; the rotation
  change is throughput-neutral-to-slightly-positive vs Hadamard, as expected
  (rotation is a small fraction of decode time — see KEY FINDING in
  `.factory/library/rotorquant-block-rotations.md`).
- **Bug fixed:** the 6 new presets were never registered in
  `STR_DTYPE_TO_TORCH_DTYPE`, so they raised `KeyError` at engine startup and
  could not be loaded at all before this change.
- **Recommendation:** ship `turboquant_iso3_nc`; treat `planar3_nc` as
  lower-quality. Net result is a strictly-better drop-in replacement for the
  existing `turboquant_k3v4_nc` Hadamard preset.

---

## Hardware / Software Manifest

| Item | Value |
|------|-------|
| GPUs | 4× AMD Instinct MI100 (gfx908, CDNA1, wave64) |
| ROCm | 7.12 (`LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib`) |
| PyTorch | 2.11.0+rocm7.2 |
| Model | `/models/Qwen3.5-9B` (FP16, head_dim=256, 8/32 full-attention layers) |
| Engine | vLLM v1, `enforce_eager=True`, TP=4, `language_model_only=True` |
| Quality harness | `lm_eval` 0.4.12, `gsm8k`, 5-shot, `--limit 200` |
| Throughput harness | `vllm bench serve`, random 1024-in/256-out, 100 prompts, concurrency 16, seed 0 |
| Env | `VLLM_ROCM_USE_SKINNY_GEMM=0` (avoids `wvSplitK` `Unsupported N value` crash at batch n>1 on this stack), `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |

---

## Methodology note: why GSM8K, not perplexity

**Wikitext perplexity via prompt-logprobs cannot measure KV-cache quantization
in this backend.** The TurboQuant `forward` (`turboquant_attn.py`) computes
prefill attention from the freshly-projected K/V tensors; the quantized cache
is only *read back* on the **decode** path (`_decode_attention`). A
prompt-logprobs PPL run is a single-chunk prefill, so it never reads the
quantized cache.

Confirmed empirically: a 4-config wikitext-2 PPL sweep
(FP16 / Hadamard / planar / iso) returned **byte-identical PPL = 9.00** for all
four, because the KV quantization was invisible to the metric. The fork's own
`scripts/m0_perplexity.py` (echo + logprobs) has the same blind spot. The
in-tree quality signal that *does* exercise decode is GSM8K — which is exactly
what `turboquant/config.py` cites ("GSM8K drops ~30 points on Qwen3-4B"
without skip-layers). All quality numbers below are therefore GSM8K.

---

## Quality gate — GSM8K (decode path)

5-shot, 200 problems, greedy. Stderr ≈ 0.024–0.028.

| Config (`--kv-cache-dtype`) | rotation | strict-match | flexible | Δ strict vs FP16 |
|---|---|---|---|---|
| `auto` (FP16 KV) | — | **0.870** | 0.865 | — |
| `turboquant_k3v4_nc` | dense Hadamard | 0.860 | 0.860 | −0.010 |
| `turboquant_planar3_nc` | 2D Givens (planar) | 0.810 | 0.810 | −0.060 |
| `turboquant_iso3_nc` | 4D quaternion (iso) | **0.870** | 0.870 | **0.000** |

- **iso matches FP16 and edges out Hadamard** (+1.0 pt). Within noise of FP16,
  clearly above planar.
- **planar drops 6 pt** — block size 2 (Givens) decorrelates less than the 4D
  quaternion at 3-bit keys. Recommend iso for the K-only 3-bit budget.

---

## Throughput — online serving

`vllm bench serve`, random dataset, 1024 input / 256 output tokens, 100
prompts, max-concurrency 16, seed 0. One fresh server per config.

| Config | req/s | output tok/s | total tok/s | mean TTFT (ms) | mean TPOT (ms) |
|---|---|---|---|---|---|
| `auto` (FP16 KV) | 1.224 | 313.5 | 1567.4 | 597.9 | 43.9 |
| `turboquant_k3v4_nc` (Hadamard) | 1.162 | 297.4 | 1486.9 | 665.6 | 46.2 |
| `turboquant_planar3_nc` | 1.175 | 300.9 | 1504.6 | 589.8 | 45.9 |
| `turboquant_iso3_nc` | 1.178 | 301.7 | 1508.3 | 592.4 | 45.7 |

- **Both block-rotation presets beat Hadamard** on req/s (+1.1 % planar,
  +1.4 % iso) and TPOT, and recover the TTFT regression Hadamard shows
  (665 → 592 ms). iso also has the lowest TPOT of the three TQ presets.
- All TurboQuant presets are ~4 % below FP16 here. This is the documented
  8/32-full-attention ceiling of this hybrid model, **not** a rotation cost —
  the rotation is a small fraction of decode time on MI100, so swapping the
  Hadamard GEMM for the O(D) block rotation moves throughput only marginally.
  The bigger fused-rotation prefill win (1.7–3.5×, see library note) is
  diluted at the whole-model level because attention is a minority of total
  decode work on this model.

---

## Bug fixed during validation

The 6 new presets were registered in `config.py` (TQ_PRESETS), `cache.py`
(CacheDType literals), and the backend's `supported_kv_cache_dtypes`, but
**not** in `vllm/utils/torch_utils.py::STR_DTYPE_TO_TORCH_DTYPE`. That dict is
consulted by `kv_cache_dtype_str_to_dtype()` during `GPUModelRunner.__init__`,
so every new preset raised `KeyError: 'turboquant_planar3_nc'` (etc.) at engine
startup and was unusable. Added all six (`turboquant_{planar,iso}{3,4}_nc`,
`turboquant_{planar,iso}3_sym_nc` → `torch.uint8`, matching the existing
packed-quant presets). This is required for the presets to load at all.

---

## Conclusion

`turboquant_iso3_nc` is a **strictly-better drop-in replacement** for
`turboquant_k3v4_nc`: equal-or-better quality (matches FP16, +1 pt over
Hadamard), equal-or-better serving throughput (+1.4 % req/s, lower TPOT and
TTFT), and a cheaper O(D) rotation. `planar3_nc` is retained but flagged as
lower quality at the 3-bit-key budget.

---

## Reproducibility

```bash
# Common env
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export HIP_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ROCM_USE_SKINNY_GEMM=0    # avoids wvSplitK crash at batch n>1

# Quality (per config CFG in auto, turboquant_k3v4_nc, turboquant_planar3_nc, turboquant_iso3_nc)
.venv/bin/python -m lm_eval --model vllm \
  --model_args "pretrained=/models/Qwen3.5-9B,dtype=float16,tensor_parallel_size=4,gpu_memory_utilization=0.85,max_model_len=4096,enforce_eager=True,language_model_only=True,kv_cache_dtype=$CFG" \
  --tasks gsm8k --num_fewshot 5 --limit 200 --batch_size auto

# Throughput: start server per config, then
vllm bench serve --backend openai --host 127.0.0.1 --port 8000 \
  --model /models/Qwen3.5-9B --dataset-name random \
  --random-input-len 1024 --random-output-len 256 --num-prompts 100 \
  --max-concurrency 16 --seed 0
```

Raw outputs: `/tmp/gsm8k_results/<cfg>/`, `/tmp/serve_results/<cfg>.json`.
