# AMD AITER for MI100 (gfx908 / CDNA1)

Build artifacts that make the buildable AMD AITER surface compile and run
**gfx908-correctly**, resolving the MI100 build wall described in issue #74 and
delivering the clean AITER build requested in issue #75.

Upstream AITER targets CDNA2/CDNA3 (MI2xx/MI3xx). On MI100 two things break:

1. **Missing Composable Kernel headers.** AITER's vendored tree ships without
   the CK submodule, so JIT builds fail with `rmsnorm2d_fwd.hpp: file not found`.
2. **CDNA2+-only ISA.** Several kernels use instructions/builtins absent on
   gfx908:
   - `v_pk_mul_f32` — packed FP32 multiply, gfx90a+
   - `v_cvt_pk_fp8_f32` / `v_cvt_pk_bf8_f32` / `__builtin_amdgcn_cvt_f32_fp8`
     — FP8 conversion, gfx942+ (no native FP8 on gfx908)
   - `row_newbroadcast` / `row_share` DPP — cross-lane, gfx90a+

## Contents

| File | Purpose |
|---|---|
| `build-aiter-gfx908.sh` | Idempotent driver: vendors CK, applies patches, AOT-builds all buildable modules, writes `build-status.json`. |
| `patches/vec_convert.h.gfx908.patch` | Scalar FP8/BF8 conversion fallback under `#if defined(__gfx908__)`. |
| `patches/rmsnorm_quant_kernels.cu.gfx908.patch` | gfx908 fallback for packed-FP32 / FP8 paths in rmsnorm+quant. |
| `patches/activation_kernels.cu.gfx908.patch` | gfx908 fallback for `v_pk_mul_f32` in activation kernels. |
| `patches/core.py.gfx908-allowlist.patch` | Adds `gfx908` to AITER's `validate_and_update_archs()` allowlist. |
| `build-status.json` | Per-module result of the last run on this MI100. |

All patches apply with `patch -p1` from the relevant source root and are
idempotent (the script checks before applying).

## Usage

```bash
# From the repo root. Builds every buildable module so there is NO runtime JIT
# during serving. Re-running skips modules whose .so already exists.
library/aiter-gfx908/build-aiter-gfx908.sh
```

Environment the script sets / expects:

```text
PYTORCH_ROCM_ARCH=gfx908
HSA_OVERRIDE_GFX_VERSION=9.0.8
GPU_ARCHS=gfx908
ROCM_PATH=/opt/rocm/core-7.12
```

Knobs (env vars):

| Var | Default | Effect |
|---|---|---|
| `PER_MODULE_TIMEOUT` | `600` | Per-module wall-clock cap (seconds); module is marked `TIMEOUT` if exceeded. |
| `INCLUDE_FP8_GEMM` | `0` | Set `1` to also attempt the FP8 a8w8 CK GEMM family (no native FP8 on gfx908; expected to fail/be slow). |
| `INCLUDE_MHA` | `0` | Set `1` to also attempt the CK MHA/FMHA family (huge kernels, time out on CDNA1; MI100 serves attention via CK-FA/Triton). |
| `AITER_BUILD_STATUS` | `library/aiter-gfx908/build-status.json` | Where to write the result table. |

## Results on this MI100

`43 ok, 4 failed, 3 timeout, of 50 attempted` (FP8 a8w8 GEMM and MHA/FMHA
excluded by default). See `build-status.json` for the full table and
`MI100_SETUP.md` → "AMD AITER on MI100 (gfx908)" for the per-module breakdown.

Genuine failures (all real CDNA1 ISA gaps, not build-config issues):

- `module_custom_all_reduce`, `module_quick_all_reduce`,
  `module_fused_qk_norm_rope_cache_quant_shuffle` — need `fp8-conversion-insts`.
- `module_moe_asm` — needs `row_newbroadcast/row_share` DPP (gfx90a+).
  Use `module_moe_ck2stages` (CK MoE) instead.

Timeouts (pathological CK compile on CDNA1, raise `PER_MODULE_TIMEOUT` to build):
`module_moe_ck2stages`, `module_aiter_operator`, `module_mla_reduce`.

## Correctness & performance

- `module_rmsnorm_quant` with the gfx908 fallback is numerically correct vs the
  reference path (scale error ~1.5e-8, int8 codes within ±1).
- Serve smoke (`VLLM_ROCM_USE_AITER=1`, Qwen3-0.6B): engine starts cleanly and
  loads prebuilt `module_rmsnorm.so` / `module_rmsnorm_quant.so` with **no
  runtime JIT** — the issue #74 acceptance criterion.
- **AITER is slower than the native/Triton path on gfx908 in every case
  measured** (Qwen3-0.6B fp16: −15% at batch 1, −26% at batch 32, −16% at batch
  64; Llama-2-7B w8a8 int8 norm-only: −6%). The kernels are tuned for CDNA2/3.
- **The quantized/MoE GEMM paths cannot be used at all on gfx908**: with
  `VLLM_ROCM_USE_AITER_LINEAR=1` the engine JIT-builds `module_gemm_a8w8`, whose
  `a8w8_rowwise_*_intrawave_*` CK instances take 20+ min each and never finish a
  full build → **startup hangs**. AITER MoE needs `module_moe_ck2stages`
  (same timeout) or `module_moe_asm` (gfx90a+ DPP, fails). Serve quantized/MoE
  models with `VLLM_ROCM_USE_AITER_LINEAR=0` and `VLLM_ROCM_USE_AITER_MOE=0`.
- Net: keep `VLLM_ROCM_USE_AITER=0` for production on MI100. This work delivers a
  *clean, correct, honestly-characterized* build (issue #75), **not** a speedup.

## Notes

- The patches and CK vendoring are applied to the **installed** AITER tree
  under `site-packages/aiter_meta` and `site-packages/aiter`; original files are
  backed up next to them as `*.gfx908orig`. The reproducible source of truth is
  this directory (script + patches + pin).
- CK pin: `b0c13f312443332c7c13a8cd26b3662582c8d3d4` (matches the AITER commit
  `428e8e76` shipped in this environment).
