# Progress — gfx900 (Vega10) upstream-library paths research

**Branch:** `gfx900/upstream-libs-research`
**Date:** 2026-07-26
**Status:** Research complete + issues filed. No code written yet — this branch
carries the research record only.
**Tracking epic:** [#84](https://github.com/larkinwc/vllm-gfx908/issues/84)

---

## TL;DR

Question we chased: *are the modern ROCm inference libraries really "unsupported"
on gfx900, or is that just un-maintained allowlists we could patch and optimize?*

Answer: **mostly soft gates.** Two of the three real paths (rocBLAS/Tensile and
Triton) are patchable with high confidence and have working precedent on
gfx906/other distros; Composable Kernel and AITER are genuine dead ends for
Vega10. Verified against current LLVM/ROCm source, not docs.

The single most important byproduct: **upstream Triton silently mis-models
gfx900** (resolves to `ISAFamily::Unknown` → warp size 32 on wave64 hardware).
That turns the Triton work from a perf nice-to-have into a **correctness
prerequisite** for trusting any gfx900 Triton kernel output.

Caveat that governs everything: *soft gate = compiles & runs correctly, NOT
gfx908 speed.* gfx900 has no MFMA and no `V_DOT2`; its wins come from being
HBM-bandwidth-bound (weight-only quant), per methodology #58.

---

## Soft-gate vs hard-barrier audit (verified against current source, 2026-07)

| Library | Verdict | What actually blocks gfx900 | Fix | Issue |
|---|---|---|---|---|
| **rocBLAS / Tensile** | **SOFT** (high) | Nothing in source. Prebuilt PyTorch wheels drop gfx900 from the target list → `Illegal seek for GPU arch : gfx900`. | Build with `-DAMDGPU_TARGETS=gfx900`. The `vega10` GEMM logic YAMLs are still in-tree and already tuned. | [#85](https://github.com/larkinwc/vllm-gfx908/issues/85) |
| **Triton (JIT)** | **SOFT to fix, correctness landmine** | gfx900 (`9.0.0`) → `ISAFamily::Unknown` → warp size **32** on wave64 HW (silent wrong reductions). Also emits `v_permlanex16` (RDNA op) → `Cannot select`. | VEGA10 → `ISAFamily` enum + `getISAFamily()` (→ wave64) + `isCDNA()`; **omit** `supportsVDot()`. | [#86](https://github.com/larkinwc/vllm-gfx908/issues/86) |
| **Composable Kernel** | DL path SOFT to build, **dead end**; DPP/XDL **HARD** | `CK_UNSUPPORTED_GPU_TARGETS` allowlist; DL path is plain FMA (compiles) but CK's value is MFMA/WMMA gfx900 lacks. DPP hardcodes wave32+DPP8 (RDNA). | Opening the gate buys only untuned generic FMA — use Triton instead. | [#87](https://github.com/larkinwc/vllm-gfx908/issues/87) |
| **AITER** | **HARD** | Hand-written CDNA3 asm (MFMA + FP8), runtime-gated to MI3XX. | Non-portable, no vector fallback. Skip. | [#87](https://github.com/larkinwc/vllm-gfx908/issues/87) |

### Load-bearing evidence (same-version contradictions)
- **rocBLAS:** Arch Linux `rocblas 7.2.4` and Gentoo (`amdgpu_targets_gfx900`
  USE flag) ship *working gfx900 kernels built from unmodified upstream source*
  — at the exact ROCm 7.2 line whose PyTorch wheel crashes. Source has it;
  packagers trim it to shrink build size.
  - default targets (gfx900 present): https://github.com/ROCm/rocBLAS/blob/develop/CMakeLists.txt
  - `vega10` logic in tree: https://github.com/ROCm/rocBLAS/tree/develop/library/src/blas3/Tensile/Logic/asm_full
  - Arch pkg files: https://archlinux.org/packages/extra/x86_64/rocblas/files/
  - the crash: https://github.com/pytorch/pytorch/issues/179865
- **Triton:** a gfx900 fork already runs vLLM —
  [`Said-Akbar/triton-gcn5`](https://github.com/Said-Akbar/triton-gcn5) (~5-line
  GCN5 enablement). `permlanex16` + `isCDNA` fix shown in
  [`nlzy/triton-gfx906`](https://github.com/nlzy/triton-gfx906) commit `7976d68f`.
  gfx900 needs the same, minus `supportsVDot` (no V_DOT2).
- **CK:** `ck.hpp` already routes gfx900 to `v_mac_f32` and withholds
  MFMA/V_DOT2; `gemm_dl`→`inner_product` is plain FMA. But the whole perf story
  is MFMA/WMMA → dead end as an optimization target.

---

## Key technical finding: packed-FP16 ("Rapid Packed Math") — RESOLVED

A hypothesis came up that gfx900 packed-FP16 might be an undocumented capability
worth exploiting. Verified against LLVM tablegen + the Vega ISA reference:

- **It's documented, not a discovery.** gfx900 has `V_PK_FMA/ADD/MUL/MAX/MIN_F16`
  (Vega "Rapid Packed Math", 2017), gated in LLVM by `HasVOP3PInsts`, which
  gfx900 satisfies.
- **But it's FP16-accumulate** → useless for GEMV/GEMM reductions. The packed op
  that helps matmul is `V_DOT2_F32_F16` (2×FP16 mul + FP32 acc), which is
  **gfx906+; gfx900 has no dot instructions at all.** So #58's "no packed-FP16"
  is correct in spirit.
- **Toolchain:** LLVM *will* emit `v_pk_fma_f16` for gfx900 from `<2 x half>` /
  `__half2` — but only via hand-written HIP. **Upstream Triton won't** (doesn't
  target gfx900).
- **Realistic exploit (small):** 2× FP16 on pointwise stages where FP16-accumulate
  is safe (SiLU/GELU, RMSNorm elementwise, RoPE, residual adds) via HIP
  `__half2`. On an HBM-bound card the value is halving FP16↔FP32 *intermediate
  traffic* in fused kernels, not the ALU 2×. Microbench-worthy, not a rewrite.
  **Non-starter** for reductions.

Sources: LLVM `AMDGPU.td` (`FeatureGFX9`⊇`FeatureVOP3PInsts`; `9_0_0`=gfx900 no
Dot vs `9_0_6`=gfx906 adds Dot1/2/7/10), `VOP3PInstructions.td`, GPUOpen "Rapid
Packed Math in Vega".

---

## Issues filed (all on `larkinwc/vllm-gfx908`, NOT upstream)

| # | Title | Labels | Notes |
|---|---|---|---|
| [#84](https://github.com/larkinwc/vllm-gfx908/issues/84) | [EPIC] Modify/optimize upstream ROCm libraries for Vega10 | `rocm,epic` | Carries the full audit + evidence; doubles as the research doc |
| [#85](https://github.com/larkinwc/vllm-gfx908/issues/85) | rocBLAS/Tensile gfx900 rebuild PoC | `rocm,infra` | Highest-certainty new path; has cheap first step (copy distro TensileLibrary) |
| [#86](https://github.com/larkinwc/vllm-gfx908/issues/86) | Fix Triton arch classification (wave64 correctness + isCDNA/permlanex16) | `rocm,kernel,infra` | **Correctness**, not perf — gates trusting #57's kernels on any Triton upgrade |
| [#87](https://github.com/larkinwc/vllm-gfx908/issues/87) | Document CK + AITER as dead-ends | `rocm,documentation` | Negative result to prevent re-attempts |

Deliberately **not** filed as new work: "port Triton" and "write a GEMV" — the
`gfx900-support` branch already has both (int4 GEMV #57, working custom Triton,
`_GFX900_GEMV_CONFIGS`). The gaps were the library/build paths above.

---

## Recommended next steps (priority order)

1. **#86 first (correctness).** Confirm what warp size our *current* working
   gfx900 Triton actually gets. If it's 32, existing reduction/shuffle kernels
   may be subtly wrong — validate #57 outputs before anything else. Consider a
   `P1` label.
2. **#85 (unblock dense GEMM).** Cheap path: drop a distro gfx900
   `TensileLibrary` into the venv and confirm the "Illegal seek" crash clears +
   GEMM is correct. Then the proper `-DAMDGPU_TARGETS=gfx900` rebuild + roofline
   baseline.
3. **#87 (document dead-ends).** Write `BENCH_GFX900_CK_AITER_NEGATIVE.md` on
   `gfx900-support` so CK/AITER aren't re-attempted.
4. Optional/marginal: a HIP `__half2` pointwise microbench for the packed-FP16
   traffic-halving idea — only if profiling shows pointwise stages matter.

## Open questions
- Does our current gfx900 Triton already patch wave64, or has it been silently
  32 this whole time? (Answered by #86 step 1 — highest urgency.)
- Is stock rocBLAS gfx900 GEMM already near the ~365 GB/s vector-FMA roofline at
  our shapes (→ just use it) or far above (→ custom-kernel headroom)? (#85 step 5.)

## Provenance
Findings produced via a deep-research fan-out (106 agents, 3-vote adversarial
verification) plus three targeted source-verification agents against LLVM/ROCm/
Triton/CK source. 20/25 verified research claims confirmed; 5 refuted (notably:
Stream-K applicability to gfx900 unproven; "MoE/full-precision inherently bad on
Vega" refuted; "vLLM ROCm targets CDNA3 exclusively" refuted).
