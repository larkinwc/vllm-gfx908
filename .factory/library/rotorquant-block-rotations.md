# RotorQuant block-diagonal rotations for the gfx908 TurboQuant backend

**Date:** 2026-06-03
**Hardware:** 4× AMD MI100 (gfx908, CDNA, wave64), ROCm 7.12, PyTorch 2.11+rocm7.2
**Scope:** Tier 2 (quality + speed), planar + iso rotations, target Qwen3.5-9B.

## Motivation

The fork already ships a working native TurboQuant KV-cache backend
(`vllm/v1/attention/backends/turboquant_attn.py` + Triton store/decode kernels +
Lloyd-Max centroids, 4 Hadamard presets). It runs on gfx908 because it performs
the decorrelating rotation as an **external rocBLAS GEMM** (`y = x_hat @ PiT`),
not the `__shfl_sync(width=32)` codebook trick that forces the llama.cpp / ollama
HIP ports to hard-block wave64 hardware. So porting RotorQuant here is *not*
blocked by the CDNA wavefront issue.

RotorQuant (Pope 2026) and IsoQuant/PlanarQuant (ParaMind2025) replace the dense
`D×D` Hadamard rotation with a **block-diagonal** orthonormal rotation:

- `planar`: `D/2` independent 2×2 Givens rotations.
- `iso`: `D/4` independent 4×4 quaternion rotations.

Cost drops from `O(N·D²)` (the GEMM) to `O(N·D)`, with `O(D)` params instead of
`D²`. Their published claim: equal/better PPL, ~5× faster prefill, faster decode.

## What was implemented

- `vllm/model_executor/layers/quantization/turboquant/rotations.py` — builds the
  dense `[D,D]` forward matrix `PiT` (for GEMM/continuation paths and cache
  symmetry) and compact per-block params (cos/sin or quaternions) for the fused
  kernels. Deterministic from `(kind, D)` only (must match across spawned
  workers). `Pi = PiT.T` is the exact inverse (verified to 1e-7).
- `vllm/v1/attention/ops/triton_block_rotate.py` — fused Triton block-rotate
  kernel (pure elementwise strided loads, **no warp shuffles → wave64-safe**).
  Bit-identical to the dense GEMM (`fusedVsGemmMaxDiff = 0.00`).
- New presets in `config.py` / `cache.py` / backend `supported_kv_cache_dtypes`:
  - K-only: `turboquant_{planar,iso}{3,4}_nc`
  - symmetric K+V: `turboquant_{planar,iso}3_sym_nc`
- Symmetric V: value plane rotated at store; decode accumulates in rotated space
  and inverse-rotates the final `[B,Hq,D]` output once (valid because attention
  is a linear softmax-weighted sum and the rotation is token-independent:
  `Σ pₜ (vₜ @ PiT) = (Σ pₜ vₜ) @ PiT`). Continuation-prefill V dequant also
  inverse-rotates.
- 77 new unit tests (rotation orthonormality/inverse/determinism, block_rotate
  == GEMM, fused kernel == GEMM, RotorQuant store/decode roundtrip, fused-store
  == GEMM-store). Full suite: **257 passed**.

## KEY FINDING: fused rotation wins at prefill, loses at decode on MI100

Microbench, `D=256`, fused block-rotate vs rocBLAS GEMM (`x @ PiT`):

| workload          | rows   | GEMM     | planar fused | speedup |
|-------------------|--------|----------|--------------|---------|
| decode B=1        | 32     | 14.5 µs  | 43.9 µs      | 0.3×    |
| decode B=8        | 256    | 14.7 µs  | 43.6 µs      | 0.3×    |
| decode B=64       | 2048   | 20.3 µs  | 43.3 µs      | 0.5×    |
| prefill 2k tok    | 16384  | 73.2 µs  | 43.2 µs      | **1.7×**|
| prefill 8k tok    | 65536  | 452.8 µs | 131.0 µs     | **3.5×**|

The fused Triton launch has a fixed ~40 µs floor on ROCm/gfx908; the small `D×D`
GEMM dispatches via rocBLAS in ~15 µs. So **RotorQuant's "faster decode" claim
does not hold on MI100** — decode query rotation (rows = B×Hq) is launch-bound.

**Resolution:** adaptive selection by row count in both launchers
(`ROTATE_FUSED_MIN_ROWS = 8192`): fused for prefill/store (the big win), GEMM for
the small decode-query rotation. Both paths are numerically identical, so this is
a pure-perf switch with no quality impact.

## Quality (synthetic, random Gaussian K/V — the hardest case)

`valCos` = cosine(decoded attention output, true value), single-token, D=256:

| preset                      | rotation | valCos |
|-----------------------------|----------|--------|
| turboquant_k3v4_nc (base)   | hadamard | 0.9929 |
| turboquant_planar3_nc       | planar   | 0.9915 |
| turboquant_iso3_nc          | iso      | 0.9912 |
| turboquant_planar4_nc       | planar   | 0.9938 |

Block rotations **match the Hadamard baseline** — confirms RotorQuant's core
claim that block-diagonal rotation decorrelates as well as the dense WHT. (The
PPL *edge* they report needs real low-rank attention vectors, not Gaussians; this
test only proves no-regression + correctness.)

Multi-token (T=64) attention vs exact fp16 softmax:

| preset                      | attnCos | note |
|-----------------------------|---------|------|
| turboquant_planar3_nc       | 0.9832  | K-only, V@4-bit |
| turboquant_iso3_nc          | 0.9759  | K-only, V@4-bit |
| turboquant_planar3_sym_nc   | 0.9522  | K+V@3-bit rotated |
| turboquant_iso3_sym_nc      | 0.9484  | K+V@3-bit rotated |

K-only beats symmetric K+V on random data (symmetric also 3-bit-quantizes V).
Matches the entire ollama-thread consensus that K-only / V-only beats full K+V.

## How to use

```bash
--kv-cache-dtype turboquant_planar3_nc   # 3-bit MSE keys (planar rot) + 4-bit V
--kv-cache-dtype turboquant_iso4_nc      # 4-bit MSE keys (iso rot) + 4-bit V
--kv-cache-dtype turboquant_planar3_sym_nc  # symmetric 3-bit K+V
```

## End-to-end validation (2026-06-04, Qwen3.5-9B, 4× MI100)

Full writeup: `BENCH_TURBOQUANT_ROTORQUANT.md`. Summary:

**Quality — GSM8K 5-shot, 200 problems (decode path):**

| preset | strict-match | vs FP16 |
|--------|-------------|---------|
| FP16 KV (`auto`)         | 0.870 | — |
| `turboquant_k3v4_nc` (Hadamard) | 0.860 | −0.010 |
| `turboquant_iso3_nc`     | **0.870** | **0.000** |
| `turboquant_planar3_nc`  | 0.810 | −0.060 |

**iso matches FP16 exactly and beats Hadamard by +1 pt.** planar drops 6 pt at
3-bit keys → recommend iso over planar for the K-only 3-bit budget.

**Throughput — `vllm bench serve`, 1024-in/256-out, 100 prompts, conc 16:**

| preset | req/s | TPOT (ms) |
|--------|-------|-----------|
| FP16 KV                 | 1.224 | 43.9 |
| `turboquant_k3v4_nc`    | 1.162 | 46.2 |
| `turboquant_planar3_nc` | 1.175 | 45.9 |
| `turboquant_iso3_nc`    | **1.178** | **45.7** |

Both block-rotation presets beat Hadamard on req/s (+1.4 % iso) and TPOT, and
recover Hadamard's TTFT regression. All TQ presets ~4 % below FP16 — the 8/32
full-attention ceiling of this hybrid model, not rotation cost.

### CRITICAL METHODOLOGY: perplexity is blind to KV quantization here

A wikitext-2 PPL sweep returned **byte-identical 9.00** for FP16 / Hadamard /
planar / iso. Reason: the TQ `forward` computes *prefill* attention from the
freshly-projected K/V; the quantized cache is only read back on the *decode*
path. Prompt-logprobs PPL is pure prefill → never reads the cache. The fork's
own `scripts/m0_perplexity.py` has the same blind spot. **Use GSM8K (or any
generation/decode task) for KV-cache quality, never prompt-logprobs PPL.**

### Bug found + fixed during validation

The 6 new presets were registered in `config.py`, `cache.py`, and the backend's
`supported_kv_cache_dtypes` but **NOT** in
`vllm/utils/torch_utils.py::STR_DTYPE_TO_TORCH_DTYPE`. That dict is read by
`kv_cache_dtype_str_to_dtype()` in `GPUModelRunner.__init__`, so every new
preset raised `KeyError` at startup and was completely unloadable. Fixed by
adding all six → `torch.uint8`. **Lesson: a new KV-cache-dtype preset must be
registered in FOUR places** — `TQ_PRESETS` (config.py), `CacheDType`
(cache.py), `supported_kv_cache_dtypes` (backend), and `STR_DTYPE_TO_TORCH_DTYPE`
(torch_utils.py).

### Env gotchas for running full Qwen3.5-9B on this stack

- `VLLM_ROCM_USE_SKINNY_GEMM=0` — the `wvSplitK` skinny GEMM crashes with
  `RuntimeError: Unsupported N value: 3072,4096,5` at batch n>1 (fine at n=1,
  which is why single-seq PPL didn't hit it; batched GSM8K/serving did).
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — the full-vocab
  `log_softmax` in `compute_logprobs` spikes ~1.9 GB; needed headroom at
  gpu_util=0.85 / max_len 2048.
- Run from `/opt/vllm-env` with the compiled `_C/_rocm_C/_moe_C` `.so` files
  symlinked from the main checkout (`/home/aimeme/Desktop/vllm-gfx908/vllm/`)
  — the worktree has no built extension; pure-Python changes are ABI-compatible
  with that env's torch 2.11+rocm7.2.

## Caveats / open items

- **Qwen3.5-9B has only 8/32 full-attention layers** (24 are linear/GDN). The
  earlier shelving verdict (`docs/turboquant-production-recommendation.md`) was
  driven by that structural cap, *not* by rotation cost. RotorQuant lowers the
  prefill rotation overhead but cannot change the 25%-of-layers ceiling, so the
  end-to-end serving win on this specific model is bounded. Dense models
  (Llama-3.1-8B, Qwen2.5-7B) are where RotorQuant's published wins live.

## Gotcha (recorded for future edits)

The redacted-token display (`key_fp8=****`) in this environment will write
**literal asterisks to disk** if a redacted line is included in an Edit `new_str`.
Two store/decode call sites got corrupted this way and had to be rewritten as
`key_fp8=self.tq_config.key_fp8`. Always `py_compile` after edits near redactions.
