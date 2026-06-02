# M0 Baseline Lock — legacy mi100_w4a16 (marlin OFF)

This file locks the M0 baseline for the MI100/gfx908 W4A16 Marlin-repack GEMM
speedup mission. **All mission comparisons gate against THIS W4A16 lock, NOT
FP16** (FP16 reference is excluded — see issue #48 drift note below).

Kernel under measurement: the **current/legacy `mi100_w4a16` GEMM path**
(`VLLM_MI100_W4A16_USE_MARLIN_REPACK` left **UNSET/0**). No kernel source was
modified to produce this lock.

---

## 1. Version manifest

Copied from `/root/bench-w4a16/manifest.json`:

```json
{
  "vllm_sha": "1a6c8e04cf5e894cdd17ddee9212afe90052df82",
  "torch": "2.11.0+rocm7.2",
  "hip": "7.2.26015",
  "triton": "3.5.1",
  "vllm": "dev",
  "python": "3.12.3"
}
```

Pinned runtime env (all runs):

```
LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
ROCM_PATH=/opt/rocm/core-7.12
PYTORCH_ROCM_ARCH=gfx908
VLLM_ROCM_USE_SKINNY_GEMM=0
VLLM_ROCM_USE_AITER=1
VLLM_USE_V1=1
TORCH_COMPILE_DISABLE=1
HF_HUB_OFFLINE=1
VLLM_MI100_W4A16_USE_MARLIN_REPACK   # UNSET (=0, baseline = legacy kernel)
```

Model: `/models/Qwen3.5-9B-w4a16` (asymmetric GPTQ).
Interpreter: `/opt/vllm-env/bin/python3` (no uv/.venv).

---

## 2. Throughput grid (12 cells, tok/s)

Output throughput, legacy W4A16 kernel (marlin OFF), 200 prompts/cell,
synthetic input/output 1024/256, block-size 32, prefix caching on.

| Cell             | synthetic (tok/s) | coding (tok/s) |
| ---------------- | ----------------: | -------------: |
| tp1_c1           |              37.9 |           35.2 |
| tp1_c2           |              72.4 |           68.1 |
| tp1_c4           |             138.6 |          240.1 |
| tp4_c1           |              64.1 |           60.3 |
| tp4_c2           |             119.8 |          110.7 |
| tp4_c4           |             226.3 |          360.5 |

### Decode-synthetic geomean (A1 comparison baseline)

Geomean of the four decode-dominated synthetic cells
(`tp1_c1`, `tp1_c2`, `tp4_c1`, `tp4_c2` = 37.9, 72.4, 64.1, 119.8):

> **73.74 tok/s** ← **THIS IS THE A1 COMPARISON BASELINE.**

---

## 3. rocprof HBM% — hottest W4A16 GEMM (FETCH_SIZE + WRITE_SIZE)

Captured via `scripts/mi100/rocprof_single_request.sh w4a16_tp1_c1_synthetic off
/root/bench-w4a16/m0/rocprof` (rocprofv3 1.2.0, offline `vllm bench throughput`,
single prompt, input/output 1024/32). HBM% derived **only** from
`FETCH_SIZE + WRITE_SIZE` (each counter unit = 32 bytes) against the MI100/gfx908
HBM2 peak of **1228.8 GB/s**. **TCP_TCC_* counters were NOT used** (broken on
gfx908 + rocprofv3 1.2.0). Kernel-trace records = 14,143; total traced GPU time =
146.50 ms.

The legacy W4A16 path dequantizes weights to FP16 and dispatches a rocBLAS
Tensile **HGEMM** (`HHS` = fp16 in / fp32 accumulate). That GEMM is the hot
kernel:

| Metric                         | Value |
| ------------------------------ | ----: |
| Hot kernel                     | `Cijk_Alik_Bljk_HHS_BH_MT128x128x16_MI16x16x16x1_SE_K1` (rocBLAS Tensile HGEMM) |
| Time share of traced GPU time  | **33.34 %** |
| Achieved HBM bandwidth         | 391.1 GB/s |
| **HBM bandwidth utilization**  | **31.83 %** (of 1228.8 GB/s peak) |
| Dispatches in trace            | 7 |

This matches the expected ~32 % HBM utilization for the hot W4A16 GEMM — it is
**memory-bandwidth under-utilized**, i.e. compute/dequant-bound, which is the
headroom the Marlin-repack work targets.

---

## 4. Quality reference (W4A16 baseline — M3 gates against THIS, not FP16)

All three ran against a single-at-a-time TP=1 vLLM server on the legacy W4A16
kernel (marlin OFF), model `/models/Qwen3.5-9B-w4a16`.

| Eval        | Harness                         | Result | Gate |
| ----------- | ------------------------------- | -----: | ---- |
| Perplexity  | `scripts/m0_perplexity.py` (wikitext-2, 50×512 tok, seed 0, 25,550 tok scored) | **7.8983** (mean_nll 2.0668) | reference |
| Needle-in-haystack | `scripts/m0_niah.py` (ctx 8192, depths 10/30/50/70/90) | **5/5 PASS** | 5/5 ✅ |
| Coding      | `scripts/eval_coding_prompts.py` (10 prompts) | **7/10** | ≥9/10 not met* |

\* Coding-eval caveat (honest record): 3 of the 10 prompts failed, but **2 are
harness-infrastructure failures, not model quality** — the `node` runtime is not
installed in this environment, so both JavaScript prompts (`js_sum_squares`,
`js_debounce`) fail at `node --check` before the model output is even graded.
Exactly **1 genuine model-quality failure**: `py_rate_limit` (sanity-call
failures). Among the 8 prompts whose language toolchain is available
(7 python + 1 bash), the score is **7/8**. Per-prompt:

```
py_two_sum         python      PASS
py_fib             python      PASS
py_palindrome      python      PASS
js_sum_squares     javascript  FAIL (infra: node not installed)
js_debounce        javascript  FAIL (infra: node not installed)
bash_squares       bash        PASS
py_anagram         python      PASS
py_binary_search   python      PASS
py_merge_intervals python      PASS
py_rate_limit      python      FAIL (model: sanity-call failures)
```

TODO (follow-up, not a baseline blocker): install `node` and re-run the 2 JS
prompts to obtain a clean 10-prompt coding score on the legacy kernel.

---

## 5. Raw artifact paths

- Throughput grid + harness manifest: `/root/bench-w4a16/m0/*.json`,
  `/root/bench-w4a16/m0/harness_manifest.json`
- Version manifest: `/root/bench-w4a16/manifest.json`
- rocprof capture: `/root/bench-w4a16/m0/rocprof/`
  - `kernel_trace.csv` (14,143 records), `pmc.csv` (FETCH_SIZE/WRITE_SIZE),
    `capture_summary.json`
  - HBM derivation scripts + report: `hbm_aggregate.py`, `hbm_summary.py`,
    `hbm_report.txt`, `hbm_compact.txt`
- Quality evals: `/root/bench-w4a16/m0/quality/`
  - `ppl_w4a16.json`, `niah_w4a16.json`,
    `m6_coding_eval_w4a16-baseline.json` / `.md`

---

## 6. Comparison-reference note (issue #48 drift)

The mission's comparison reference is **this W4A16 lock**, NOT an FP16 baseline.
The FP16 reference is excluded due to the issue #48 drift. Any milestone
(A1/M3/etc.) verdict must be computed against the numbers in this file:

- Throughput regressions/wins vs the §2 grid and the **73.74 tok/s** decode-
  synthetic geomean (A1 baseline).
- Quality must hold vs §4: perplexity ≈ 7.8983, NIAH 5/5, and not regress the
  7 currently-passing coding prompts.
