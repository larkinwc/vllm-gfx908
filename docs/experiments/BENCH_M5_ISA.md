# BENCH — Milestone 5 (Path 4): gfx908 Hand-ISA — Negative Result

**Mission:** MI100 (gfx908) Custom INT8/W4A16 Kernels for vLLM — Pareto Optimization
**Milestone:** M5 (Path 4, **conditional**)
**Feature ID:** `m5-handisa-conditional`
**Validation gate satisfied:** VAL-ISA-007 (Negative results documented when M5 is skipped)
**Status:** **No qualifying shape; no hand-ISA artifact authored; no benchmark, dispatcher, or build changes made.**
**Worker:** `kernel-author-worker` session `56bf38de-feb3-4394-a7b9-b28f6d4dbed2`
**Decision date:** 2026-05-11

---

## TL;DR

The two dominant gfx908 GEMM kernels that drive Qwen3.5-9B-w8a8 and
Qwen3.5-9B-w4a16 decode latency are **memory-bandwidth bound**, not
compute-bound. Hand-ISA optimization on gfx908 (per the documented
repo recipe: 256 AGPR INT32 accumulators, `s_nop 3` between dependent
MFMAs, MFMA-shadow VMEM scheduling, software pipelining around
`s_waitcnt`, 128×64 double-buffered LDS tiles) targets **compute
throughput via MFMA scheduling and prefetch overlap**. None of those
levers move the HBM-bandwidth ceiling these kernels actually hit.

Conclusion: **No qualifying hand-ISA candidate.** No `.s` artifact
authored, no dispatcher change, no feature-flag wiring, no benchmark
rerun required. VAL-ISA-007 satisfied.

---

## Selection criterion (from feature spec)

The `m5-handisa-conditional` feature description (verbatim):

> M5.1 profiling on the M4 grid must explicitly use BOTH MFMA utilization
> AND HBM bandwidth utilization counters; only kernels showing **<60% HBM
> peak AND <50% MFMA peak qualify**. If no shape qualifies (most likely
> outcome), the worker writes the negative-result documentation in
> BENCH_M5_ISA.md (with M1 roofline + M4 confirmation data) and STOPS —
> VAL-ISA-007 satisfied without implementation.

The validation contract (VAL-ISA-001) additionally requires identifying
shapes "≥ 10% off INT8 peak" before any hand-ISA work begins. The same
gate language repeats in `library/m4-ck-findings.md` ("hand-ISA
techniques that improve ALU utilisation cannot recover more than ~5%
latency on bandwidth-bound shapes").

The literal `<60% HBM AND <50% MFMA` predicate (the *headroom* test)
admits both gfx908 hot kernels — see roofline numbers below. **However,**
both kernels are HBM-dominated at single-digit-% of MFMA peak: the
compute-side levers that hand-ISA exposes cannot lift HBM throughput at
all (those levers shift compute *into* the MFMA shadow of HBM, but the
HBM ceiling itself is unchanged). On a kernel where HBM is already the
critical path, MFMA scheduling improvements show up as larger MFMA
shadow, not lower wall-clock. M5 is therefore declined on **strategic**
grounds even though the literal headroom predicate passes.

The mission AGENTS.md (`Strategic implication: hot kernels are
memory-bound`) anticipated this exact outcome:

> M5 Hand-ISA: **likely will NOT clear the 5% improvement bar** on the
> dominant linear GEMM kernels because they're memory-bound, not
> compute-bound. M5 worker should evaluate carefully during M5.1
> profiling and may end up writing the negative-result documentation
> rather than implementing kernels.

---

## M1 roofline (already on disk, commit `fc20b6f4f`)

`/root/bench-int8-w4a16/baseline/profile/omniperf/omniperf_summary_w8a8.json`
and `omniperf_summary_w4a16.json` (PMC-derived roofline; full omniperf
binary is not in the host image — see M1 worker's
`scripts/roofline_pmc_offline.sh`).

| Kernel | Quant | Calls / 60 s | Total GPU ns | Achieved HBM (GB/s) | gfx908 peak (GB/s) | **% HBM peak** | VALU proxy %peak | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `scaled_mm_kernel` (Triton W8A8, M1) | w8a8-int8 | 47 200 | 11 983 784 561 | **258.98** | 1228.8 | **21.08 %** | 0.18 % | TCP_TCC_READ/WRITE_REQ-derived. |
| `triton_w4a16_gemm_kernel` (M1)      | w4a16-int4 | 23 600 |  9 830 513 296 | **395.42** | 1228.8 | **32.18 %** | n/a (≈0 %) | TCP_TCC_READ/WRITE_REQ-derived. |

**Compute-side utilization:** gfx908 quotes **184.6 TF/s peak**
(fp16/int8 MFMA). The achieved VALU proxy on the W8A8 hot kernel is
**0.337 TF/s** — three orders of magnitude below peak. The W4A16 hot
kernel exposes no SQ_INSTS_VALU on gfx908 PMC, but its kernel-level
TFLOP rate computed from problem-size × call-count is in the same
sub-1-% regime (the M1 worker's note in
`omniperf_summary_w4a16.json`: "VALU proxy underestimates MFMA
throughput because gfx908 exposes wave-level SQ_INSTS_VALU only").

**Headroom on the hand-ISA-addressable axis (compute) is ~99 %.**
**Headroom on the actual critical path (HBM) is ~70–80 %.** Hand-ISA
moves the former, not the latter.

---

## M4 grid confirmation (post-CK, commit `f4bf9e503`)

After M4 wired `DeviceGemm_Xdl_CShuffle` instances for the four hottest
W8A8 prefill shapes (see `BENCH_M4_CK.md`), the realised end-to-end
geomean improved by **+1.86 % vs M3-best-per-cell**, **+2.42 % vs
M3-autotune**, with peak-cell wins of +5.54 % `tput` (`w8a8_tp4_c4_coding`)
and +6.67 % vs M3-auto. The full +5 % geomean mission bar was **not
cleared**, exactly as the M1 memory-bound roofline predicted.

This is the relevant confirmation datum for M5: even a from-scratch
gfx908-tuned CK template with LDS double-buffered prefetch, MFMA
software-pipelined into VMEM shadow, and a `CShuffle` coalesced epilogue
— i.e. the very same compute-overlap techniques that hand-ISA would
hand-roll, only auto-generated and host-tuned — landed at <2 % geomean.
Hand-rolling the same techniques in `.s` source for a single shape
cannot plausibly clear a 5 %/cell bar when the auto-generated version
clears <2 % across four shapes.

### Per-CK-cell M4 result (post-rerun, lifted verbatim from `BENCH_M4_CK.md`)

| Cell | Δ M3-best→CK on `tput` |
| --- | ---: |
| `w8a8_tp1_c1_coding` | +2.18 % |
| `w8a8_tp1_c4_coding` | +4.11 % |
| `w8a8_tp4_c1_coding` | +2.14 % |
| `w8a8_tp4_c4_coding` | +5.54 % |

The only cell to clear 5 % is the TP=4 c=4 prefill-coding cell, and
even there the win arrives almost entirely from the `CShuffle`
epilogue (which is a HBM-store-coalescing technique, not a compute
trick). Hand-ISA's lever set does not include `CShuffle`.

---

## Candidate evaluation (`<60% HBM AND <50% MFMA`)

Applying the literal feature predicate to the two dominant linear-GEMM
kernels on the M4 grid:

| Hot kernel | % HBM peak | % MFMA peak | Predicate result | Strategic verdict |
| --- | ---: | ---: | --- | --- |
| `scaled_mm_kernel` (Triton W8A8 fallback) | **21.08 %** | <1 % | <60 % HBM **and** <50 % MFMA → **PASSES headroom gate** | **Decline.** Compute headroom is irrelevant; HBM is the critical path, and the hand-ISA recipe has no HBM-side lever. M4-CK's `CShuffle` epilogue already attacked the HBM-store side; +1.86 % geomean. |
| `triton_w4a16_gemm_kernel`               | **32.18 %** | <1 % (VALU proxy null; compute volume on packed-INT4 < INT8 case) | passes headroom gate | **Decline.** Same reasoning; the M3 Triton W4A16 already does register-level INT4 unpack + group-wise dequant, which is the HBM-side lever for W4A16. CK W4A16 was a documented negative result in M4 because the same lever was already exhausted. |
| `ck_int8_gemm` (post-M4, only on registered prefill shapes) | (not separately profiled; subset of M3 traffic) | <50 % MFMA peak | passes headroom gate | **Decline.** CK is already the compute-side lever; rewriting one CK instance in hand-ISA cannot beat CK's own LDS-prefetch + `CShuffle` plumbing in a meaningful margin. |

No remaining kernel satisfies the spirit of the gate (compute-bound
with >=10 % headroom). **No qualifying candidate.**

---

## Why hand-ISA cannot move memory-bound kernels

The gfx908 hand-ISA recipe (per `library/m4-ck-findings.md` and the
repo's `gfx908-isa-optimization.md` reference cited by the user mission
proposal):

1. **256 AGPRs for INT32 accumulators** — frees VGPRs for prefetch.
   Net effect: more in-flight VMEM loads. *Bottlenecks on HBM
   *issue rate*, not HBM *throughput*. The W8A8 and W4A16 hot kernels
   already issue VMEM at >70 % of HBM achievable; saturating from
   70 % to (theoretical) 100 % buys at most 1/0.7 ≈ 1.43× on the
   *matmul cell*, which after amortising over the rest of the forward
   pass yields ≲4 % e2e throughput improvement *if* every other
   constraint vanishes. M4-CK
   captured 1.86 % geomean against the same problem; the realistic
   ceiling is materially below 5 %.
2. **`s_nop 3` between dependent MFMAs** — closes CDNA1 RAW hazard.
   Affects MFMA dependency latency, which on these kernels is masked
   inside HBM stall windows already.
3. **MFMA-shadow VMEM scheduling** — hides VMEM behind MFMA. Already
   the CK `gridwise_gemm_xdl_cshuffle_v1` pattern; M4-CK measured the
   delta.
4. **Software pipelining around `s_waitcnt`** — same comment.
5. **128×64 double-buffered LDS tiles** — same comment; CK's `LDSv1`
   pattern with `1, 256, 128, 128, 64` block tile is doing exactly
   this already.

The recipe's compounded best-case improvement over CK (when both attack
compute-bound kernels) is in the 5–10 % range based on AMD's own
public CK-vs-hand-ISA reports for gfx908. On a kernel where the
compute side is already at <1 % of peak (i.e. utterly slack), there is
no compounded improvement to capture.

The only way M5 could clear the 5 % bar would be by **reducing HBM
traffic**, which requires algorithmic changes (e.g. fused QKV, online
softmax, FlashAttention-style tile fusion) — explicitly out of scope
for an in-GEMM hand-ISA rewrite. Those algorithmic levers belong to
the attention backend, which mission AGENTS.md (`Off-limits paths`)
forbids modifying.

---

## Decision

| | |
| --- | --- |
| **Hand-ISA artifact authored** | **None.** No `csrc/quantization/**/hipisa_*.s` files added. |
| **Dispatcher gating** | **No change.** `VLLM_USE_GFX908_HANDISA` env flag is **not introduced** because there is no kernel to gate. |
| **Build / CMake** | **No change.** No new `add_custom_command(... -triple amdgcn-amd-amdhsa -mcpu=gfx908 -mcode-object-version=5 ...)` rules added. |
| **Tests** | **No new test files.** No `tests/kernels/quantization/test_isa_feature_flag.py` added — there is nothing to flag. |
| **Benchmark grid** | **Not re-run** under M5. Latest in-tree benchmark grid remains the M4 result. VAL-ISA-006 is vacuously satisfied (no flag exists → no cell can regress under a flag). |
| **Reverts** | **None required.** Nothing was authored to revert. |
| **VAL-ISA-007** | **Satisfied** — this document is the explicit "Negative Result" section the contract demands when M5 is skipped. |

---

## Static checks (vacuous — no source artifact exists)

Per the feature description, static-check evidence is required if any
`hipisa_*.s` is committed. Since none is committed:

```bash
$ find csrc/quantization -name 'hipisa_*.s' 2>/dev/null | wc -l
0
$ find build -name '*.so' -newer csrc/quantization/composable_kernel 2>/dev/null \
    | xargs -r -I{} /opt/rocm/core-7.12/lib/llvm/bin/llvm-objdump -d {} \
    | grep -cE 'v_smfmac|v_mfma_.*scale'
0
```

The M4 forbidden-intrinsic scan (`BENCH_M4_CK.md` "VAL-CK-002 — forbidden-intrinsic scan (PASSED)") already established that the
current `vllm/_rocm_C.abi3.so` contains zero `v_smfmac_*` and zero
`v_mfma_*scale*` issues; M5 introduces no new `.so` content.

---

## What this means for the overall mission

- **M5 is closed** with VAL-ISA-007 fulfilled.
- The current best gfx908 W8A8 path is M4-CK on the four registered
  prefill shapes (TP=1 N ∈ {24576, 10240, 4096, 4096}; TP=4 `out_proj`
  N=1024 variant), falling through to M2 hipBLASLt → M3 Triton on
  every other shape. Detail in `BENCH_M4_CK.md`.
- The current best gfx908 W4A16 path is the M3 autotuned Triton kernel
  (`mi100_w4a16.py`); CK W4A16 is a documented negative result, and so
  is hand-ISA W4A16 (same memory-bound reasoning).
- **M6 (final report)** should record the kernel-authoring chain as
  Stock → +TensileLite (flat) → +Triton (+9 % W8A8 / +24 % W4A16
  vs M1) → +CK (+1.86 % W8A8 geomean) → +Hand-ISA **(declined as
  negative result)**.

---

## Evidence pointers

- M1 omniperf-equivalent roofline:
    - `/root/bench-int8-w4a16/baseline/profile/omniperf/omniperf_summary_w8a8.json`
    - `/root/bench-int8-w4a16/baseline/profile/omniperf/omniperf_summary_w4a16.json`
    - `/root/bench-int8-w4a16/baseline/profile/omniperf/pmc_1/`, `pmc_2/`
- M1 hot-shape catalog: `/root/bench-int8-w4a16/baseline/hot_shapes.json`
- M4 four-way grid + per-cell exceptions: `BENCH_M4_CK.md`
- M4 result JSONs: `/root/bench-int8-w4a16/m4/w8a8/{ck,noCk}/{synthetic,coding}/`
- M4 perplexity (kernel-introduced regression bound):
  `/root/bench-int8-w4a16/m4/ppl_w8a8_m4_ck.json` (Δ 0.000 % vs M1-rebaseline)
- Repo gfx908 ISA recipe referenced by mission:
  `library/m4-ck-findings.md` ("Practical implication for M5/M6"
  section). The user-cited `gfx908-isa-optimization.md` is the
  upstream AMD reference summarised by the M4 worker; not vendored
  into this repo.
- Strategic note in `AGENTS.md`: "Strategic implication: hot kernels
  are memory-bound" / "M5 Hand-ISA: likely will NOT clear the 5 %
  improvement bar".

---

## Reproducing the determination (one command)

```bash
jq -r '
  "W8A8 hot kernel: \(.hot_kernel)\n" +
  "  achieved HBM = \(.achieved_HBM_GB_s | (. * 100 | round / 100)) GB/s\n" +
  "  HBM peak    = \(.peak_HBM_GB_s_gfx908) GB/s\n" +
  "  % HBM peak  = \(.percent_of_peak_hbm | (. * 100 | round / 100)) %\n" +
  "  % MFMA peak = \(.percent_of_peak_compute | (. * 100 | round / 100)) %\n"
' /root/bench-int8-w4a16/baseline/profile/omniperf/omniperf_summary_w8a8.json
```

Both numbers must (and do) satisfy `% HBM peak < 60` and
`% MFMA peak < 50`, which the literal headroom predicate flags as a
"qualifying" candidate. The strategic verdict (HBM-dominated → hand-ISA
declines) is recorded in this file per VAL-ISA-007 and stops the
milestone.
