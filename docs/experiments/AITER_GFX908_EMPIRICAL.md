# AITER on MI100 (gfx908) — Empirical Build + Perf Results

Hands-on follow-up to the theoretical [`docs/AITER_FORK_ANALYSIS.md`](../AITER_FORK_ANALYSIS.md).
That analysis (Mar 2026) *predicted* a 5–12 % TPOT win and recommended **against**
a full AITER fork. This document **confirms that verdict with measurements**:
AITER now *builds cleanly* on gfx908 (issue #75 / unblocking #74), but it is
**slower than the native/Triton path in every case measured**, and the paths
AITER is actually designed to accelerate (int8/fp8 w8a8 GEMM, MoE) **do not
build on CDNA1 at all** — they hang the engine at startup.

**Date:** 2026-06-07
**Hardware:** 1× AMD Instinct MI100 (gfx908 / CDNA1), ROCm 7.12, PyTorch
2.11.0+rocm7.2
**AITER:** `0.1.1.dev1420+g428e8e761` (CK pin `b0c13f31`)
**Reproduction artifacts:** [`library/aiter-gfx908/`](../../library/aiter-gfx908/)
(build script + ISA-fallback patches + per-module status)

---

## TL;DR

| Question | Answer |
|---|---|
| Can AITER build on gfx908? | **Yes**, after vendoring CK + 3 ISA-fallback patches. 43/50 attempted modules build. |
| Is AITER faster on gfx908? | **No.** −6 % to −26 % vs native/Triton across fp16 (batch 1–64) and w8a8. |
| Do the quantized / MoE GEMM paths work? | **No.** `module_gemm_a8w8` and `module_moe_ck2stages` do not finish compiling on CDNA1 → engine **hangs at startup**. |
| Recommendation | Keep `VLLM_ROCM_USE_AITER=0` (default). Ship the build fix + docs only; AITER stays opt-in for compatibility/debugging. |

---

## 1. The two build blockers (issue #74) and how they were fixed

1. **Missing Composable Kernel headers.** AITER's bundled `aiter_meta/3rdparty/`
   ships only `ck_helper`, not the full CK tree, so JIT builds fail with
   `fatal error: 'rmsnorm2d_fwd.hpp' file not found`.
   → **Fix:** vendor CK at the pinned revision `b0c13f31` into
   `aiter_meta/3rdparty/composable_kernel`.

2. **CDNA2+-only ISA.** Several kernels use instructions/builtins that do not
   exist on gfx908:
   - `v_pk_mul_f32` (packed FP32 multiply) — gfx90a+
   - `v_cvt_pk_fp8_f32` / `v_cvt_pk_bf8_f32` / `__builtin_amdgcn_cvt_f32_fp8`
     (FP8 conversion) — gfx942+ (gfx908 has no native FP8)
   - `row_newbroadcast` / `row_share` DPP (cross-lane) — gfx90a+
   → **Fix:** scalar fallbacks under `#if defined(__gfx908__)` in
   `vec_convert.h`, `rmsnorm_quant_kernels.cu`, `activation_kernels.cu`, plus an
   allowlist patch adding `gfx908` to AITER's `core.py`.

`library/aiter-gfx908/build-aiter-gfx908.sh` applies both fixes idempotently and
AOT-builds every buildable module so there is **no runtime JIT at serve time**.

## 2. Build surface — 43/50 attempted modules

`43 ok, 4 failed, 3 timeout, of 50 attempted` (FP8 a8w8 GEMM and CK MHA/FMHA
excluded by default). Genuine failures, all real CDNA1 ISA gaps confirmed from
compiler output:

| Module(s) | Result | Root cause (verbatim from hipcc) |
|---|---|---|
| `module_custom_all_reduce`, `module_quick_all_reduce`, `module_fused_qk_norm_rope_cache_quant_shuffle` | FAIL | `'__builtin_amdgcn_cvt_f32_fp8' needs target feature fp8-conversion-insts` |
| `module_moe_asm` | FAIL | `Invalid dpp_ctrl value: row_newbroadcast/row_share is not supported before GFX90A/GFX10` |
| `module_moe_ck2stages`, `module_aiter_operator`, `module_mla_reduce` | TIMEOUT | pathological CK compile, does not finish at 1800 s |

`module_rmsnorm_quant` with the gfx908 fallback was checked for **numerical
correctness**: scale error ~1.5e-8, int8 codes within ±1 of the reference path.

## 3. Performance A/B — AITER is slower in every measured case

`enforce_eager`, greedy (`temperature=0`), single MI100. Each cell is the steady
state after a warmup pass.

| Model / config | `USE_AITER=0` | `USE_AITER=1` | Δ |
|---|---:|---:|---:|
| Qwen3-0.6B fp16, batch=1 decode | 345.0 tok/s | 291.5 tok/s | **−15 %** |
| Qwen3-0.6B fp16, batch=32 | 1454.1 tok/s | 1079.1 tok/s | **−26 %** |
| Qwen3-0.6B fp16, batch=64 | 2826.1 tok/s | 2361.3 tok/s | **−16 %** |
| Llama-2-7B w8a8 int8, batch=1 (AITER norm only, `LINEAR=0`) | 34.0 tok/s | 32.0 tok/s | **−6 %** |

AITER's rmsnorm/activation/rope kernels are tuned for CDNA2/CDNA3; the native
vLLM/Triton kernels win on CDNA1. fp16 numbers are stable across repeats
(±1 tok/s).

## 4. The high-value paths don't build → engine hangs at startup

The cases AITER is *designed* to accelerate fail at **build/startup**, not just
on perf:

- **w8a8 int8 GEMM (`VLLM_ROCM_USE_AITER_LINEAR=1`).** Loading a w8a8 model with
  full AITER triggers a runtime JIT build of `module_gemm_a8w8`. Its
  `a8w8_rowwise_*_intrawave_v3_*` Composable Kernel template instances each take
  20+ minutes to compile on the gfx908 toolchain; with `-j 51` the build was
  still stuck on the first 4 such files after ~27 minutes (1633 s) and a full
  build never completed in hours. **Result: the vLLM engine hangs at init**
  (42 compiler processes spawned, waiting on a build baton). Must serve with
  `VLLM_ROCM_USE_AITER_LINEAR=0`.

- **MoE (`VLLM_ROCM_USE_AITER_MOE=1`).** `aiter.fused_moe` needs
  `module_moe_ck2stages` (same pathological CK compile / timeout) or
  `module_moe_asm` (fails — `row_share` DPP is gfx90a+). Must serve with
  `VLLM_ROCM_USE_AITER_MOE=0`.

So the only AITER surface that both builds *and* runs on gfx908 is
norm/activation/rope — and that is the surface measured as slower above.

Note: AITER's MHA/FMHA flash-attention kernels also do not build on gfx908. The
real attention win on MI100 comes from the **separate** CK Flash Attention path
(`scripts/build_ck_flash_attn_gfx908.sh`), which is unrelated to AITER.

## 5. Recommendation

- **Keep `VLLM_ROCM_USE_AITER=0` (the default) for production on MI100.**
- The committed build fix (`library/aiter-gfx908/`) + the `on_mi3xx → on_gfx9`
  dispatch gate make AITER *available and clean* on gfx908 so it can be enabled
  for compatibility/debugging or re-benchmarked on future toolchains — it is
  **not** a speedup today.
- If a future ROCm toolchain compiles the `a8w8_rowwise` CK instances in
  reasonable time, re-run the w8a8/MoE A/B before enabling those paths.

## How to reproduce

```bash
# 1. Build the gfx908 AITER surface (no runtime JIT afterwards):
library/aiter-gfx908/build-aiter-gfx908.sh           # writes build-status.json

# 2. Serve smoke (AITER norm path, prebuilt .so, no JIT):
VLLM_ROCM_USE_AITER=1 HSA_OVERRIDE_GFX_VERSION=9.0.8 \
PYTORCH_ROCM_ARCH=gfx908 GPU_ARCHS=gfx908 \
  <vllm offline-generate or serve a small fp16 model>

# 3. A/B: flip VLLM_ROCM_USE_AITER between 0 and 1, same model/prompts.
#    For w8a8 models add VLLM_ROCM_USE_AITER_LINEAR=0 VLLM_ROCM_USE_AITER_MOE=0
#    or the engine will hang building module_gemm_a8w8.
```

---

*See also: [`AITER_FORK_ANALYSIS.md`](../AITER_FORK_ANALYSIS.md) (the prior
theoretical analysis this confirms), [`MI100_SETUP.md`](../../MI100_SETUP.md)
§"AMD AITER on MI100" (serve config), [`library/aiter-gfx908/`](../../library/aiter-gfx908/)
(build script + patches).*
