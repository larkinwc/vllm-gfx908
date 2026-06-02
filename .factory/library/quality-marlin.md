# M3 quality — Marlin-ON W4A16 vs legacy (REAL, verified, apples-to-apples)

> Captured 2026-05-30/31 by the orchestrator on MI100 (gfx908). BOTH arms were
> run through the SAME `scripts/mi100/quality_eval.py` invocation against a
> verified server on port 8000, identical flags
> (`--dtype float16 --max-model-len 4096 --block-size 32 --language-model-only
> --gpu-memory-utilization 0.85`), only the marlin flag differing:
>
> - **marlin-ON**: `/proc/<pid>/environ` confirmed
>   `VLLM_MI100_W4A16_USE_MARLIN_REPACK=1`; server log confirmed
>   `Using TritonW4A16LinearKernel`; group_size=128 so the dispatch gate
>   (`flag=1 AND on_mi100() AND g in {32,128}`) is satisfied and the
>   `_maybe_marlin_repack` hook fires.
> - **legacy**: `/proc/<pid>/environ` confirmed
>   `VLLM_MI100_W4A16_USE_MARLIN_REPACK=0`.
>
> Every number below was read directly from the raw result JSON (paths in "Raw
> artifacts"); no value is transcribed from memory.

## Method note (why legacy is the reference here, not the M0 lock number)

The M0 baseline-lock recorded ppl **10.9139** at **8 windows / 3814 tokens**.
The M3 quality_eval run used the documented default invocation
(`--ppl-tokens 8192 --max-len 2048`), which on this run produced **16 windows /
7506 tokens** and a different absolute ppl. Because absolute ppl depends on the
window/token count, the **only valid lossless check is marlin-vs-legacy at the
IDENTICAL config**, which is what this file reports. (The M0 lock value is still
the recorded historical baseline; it was captured with a different effective
window count and is not directly comparable token-for-token.)

## Results (identical config, marlin vs legacy)

| metric | legacy (flag=0) | marlin-ON (flag=1) | gate | verdict |
|--------|----------------:|-------------------:|------|:-------:|
| WikiText-2-raw perplexity | 11.788052776285937 | 11.787984062106219 | ≤ +1% | **PASS** (Δ −0.00058%) |
| mean NLL | 2.467086541985034 | 2.46708071283061 | — | Δ ~6e-6 |
| scored tokens / windows | 7506 / 16 | 7506 / 16 | — | identical |
| Coding-agent (pass@1) | 9/10 (M0 ref) | 9/10 | ≥ 9/10 | **PASS** |
| Needle-in-haystack | 1/5 | 1/5 | no regression | **PASS** |

Detail (all from the raw JSON):

- **Perplexity** marlin 11.787984062106219 vs legacy 11.788052776285937 —
  **near-identical but NOT bit-identical** (Δ = −0.00058%, marlin marginally
  lower; mean_nll 2.46708071 vs 2.46708654, Δ ~6e-6; both 7506 scored tokens,
  16 windows, 0 failed). The sub-0.001% delta is far inside the ≤+1% tolerance
  and reflects floating-point reduction-order differences in the two kernels,
  not a quality change. **PASS.**
- **Coding** marlin-ON = 9/10 via `coding_agent_eval.py --max-tokens 1024`
  (`/v1/chat/completions`). All 9 real tasks pass (`method: test`); the sole
  failure is the INTENTIONAL `max_subarray` fixture (`ok:false`) — identical to
  the M0 baseline; ceiling is 9/10 by design. Not "fixed". The M0 reference
  (also 9/10) is `/root/bench-results/m0-baseline/coding_agent.json`.
- **Needle** marlin-ON = 1/5; per-depth pattern `{0.0:✓, 0.25:✗, 0.5:✗,
  0.75:✗, 1.0:✗}` is identical to the legacy arm AND to the M0 baseline. The
  1/5 is a property of this W4A16 build / needle-prompt format (flagged in
  `baseline-lock.md` §4), NOT a marlin regression.

## Why quality is preserved exactly

The Marlin repack + GEMM is numerically equivalent to the legacy W4A16 path
(M1 correctness tests pass at atol=1e-2, rtol=5e-2 vs an FP32 dequant
reference). The repack only changes the int4 memory layout and pre-fuses
scales/zeros; it does not change the dequant math. The empirically near-identical
ppl (marlin 11.787984 vs legacy 11.788053 at the same config, Δ −0.00058%) is
the direct confirmation — the tiny delta is FP reduction-order only.

## VAL-QUAL verdicts

- **VAL-QUAL-001 (ppl ≤ +1%): PASS** (Δ −0.00058%, marlin 11.787984 vs legacy
  11.788053 at identical config).
- **VAL-QUAL-002 (coding ≥ 9/10): PASS** (9/10, max_subarray fixture as-is).
- **VAL-QUAL-003 (needle): PASS** (1/5, depth pattern identical to legacy + M0;
  no regression).

## Raw artifacts (orchestrator-measured)

- marlin ppl+needle: `/root/bench-w4a16/m3/quality/quality_ppl_needle_marlin.json`
  (ppl 11.787984062106219, needle 1/5, 7506 tok / 16 windows)
- legacy ppl+needle: `/root/bench-w4a16/m3/quality/quality_ppl_needle_legacy.json`
  (ppl 11.788052776285937, needle 1/5, 7506 tok / 16 windows)
- marlin coding: `/root/bench-w4a16/m3/quality/coding_agent_marlin.json`
  (score 9/10, sole fail=max_subarray)
- M0 reference: `/root/bench-w4a16/m0/quality/quality_ppl_needle.json`
  (ppl 10.9139 at 8 windows — different config, see method note),
  `/root/bench-results/m0-baseline/coding_agent.json` (9/10)
- harnesses: `scripts/mi100/quality_eval.py`, `scripts/mi100/coding_agent_eval.py`
- server logs: `/root/marlin_tp1_server.log`, `/root/legacy_tp1_server.log`
