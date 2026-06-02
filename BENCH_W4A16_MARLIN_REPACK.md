<!-- markdownlint-disable MD013 MD031 MD032 MD040 -->
# BENCH_W4A16_MARLIN_REPACK — Marlin-style W4A16 repack for gfx908 (MI100)

> **Verdict (honest lead):** The Marlin-style W4A16 repack for gfx908 (MI100) is
> **correct and numerically lossless, but it REGRESSES end-to-end decode
> throughput by ~28%** (verified matched A/B, TP1 geomean **39.5417 vs legacy
> 54.9452 = −28.03%**; per-cell −26% to −30%). The rocprof data corroborates
> this: the marlin GEMM fetches **+2.3% MORE** HBM per dispatch (25,046 vs
> 24,488 KB) and emits **10 extra helper kernels**, while per-token latency
> (TPOT) rises ~11–13 ms. The change ships **default-OFF** behind
> `VLLM_MI100_W4A16_USE_MARLIN_REPACK` and **should not be enabled** for decode
> on this architecture.
>
> **Both performance targets were MISSED — this is a documented NEGATIVE
> RESULT** (see §2 and §6). The contribution value is the verified, reproducible
> *measurement* that the Marlin approach does not work on gfx908 (CDNA1, no
> `cp.async`), plus exact quality preservation (§5) — not a speedup.

All performance numbers in this report are **VERIFIED-REAL** (orchestrator
matched A/B on MI100/gfx908, 2026-06-01), read directly from raw JSON/CSV on
disk. **Correction note:** prior revisions of this report and the supporting
library artifacts (`bench-marlin-grid.md`, `rocprof-marlin.md`) contained
**fabricated positive numbers** (geomean +0.24%, FETCH −28.66%, "bit-identical
ppl") that were never measured — the underlying raw files were absent or showed
`failed=200`. Those were deleted and replaced with this verified matched A/B.
Supporting artifacts: `baseline-lock.md` (M0), `bench-marlin-grid.md` (matched
A/B), `rocprof-marlin.md` (HBM per-dispatch), `quality-marlin.md` (VAL-QUAL),
`autotune-marlin.md` (80 configs + manifest).

---

## 1. Executive summary

Issue #45 targets the legacy `mi100_w4a16_gemm_kernel`, which the M0 rocprof lock
shows is **63.68%** of all GPU kernel time on the W4A16 server — the correct
optimization target. The Marlin-style repack relays the int4 weights into a
lane-adjacent layout and pre-fuses scales/zeros, with the *intent* that the
decode GEMM read less from HBM and skip the per-tile `tl.interleave` dequant
cascade. On gfx908 this intent is **not realized** — see the measured outcome.

Measured outcome on gfx908 (CDNA1):

- **End-to-end (matched A/B, TP1 synthetic, all completed=200/failed=0):**
  geomean(c1,c2,c4) **marlin 39.5417 vs legacy 54.9452 = −28.03%**; per-cell
  −28.27% / −25.97% / −29.81%. TPOT rises ~11–13 ms/token. **A1 (≥+5%) MISSED —
  large regression.**
- **HBM (rocprof, per-dispatch FETCH):** marlin GEMM **25,046 KB/disp** vs
  legacy **24,488 KB/disp** = **+2.3% MORE** traffic, not less; marlin emits
  **14 distinct kernels** vs **4** for legacy. **A3 (≥50% HBM) MISSED** and the
  repack provides no traffic reduction at M=1.
- **Quality:** ppl marlin 11.787984 vs legacy 11.788053 (Δ −0.00058%, near-
  identical, FP reduction-order only), coding 9/10, needle 1/5 — all unchanged.
  **Lossless.**
- **Non-regression:** **FAILED** — all 3 cells regress far beyond the −5% floor.
  Ships **default-OFF**; do not enable for decode on gfx908.

---

## 2. A1–A5 verdict table

| ID | Gate | Verdict | Detail |
|----|------|:-------:|--------|
| **A1** | decode-synthetic geomean ≥ +5% vs legacy | **MISS — NEGATIVE RESULT** | matched A/B TP1 geomean marlin **39.5417** vs legacy **54.9452** = **−28.03%**. Large regression. Attribution §4/§6. |
| **A2** | ≥ 6/12 cells non-regressing (Δ≥−2%) | **FAIL** | **0/3** measured cells non-regressing; all regress −26% to −30%. |
| **A3** | rocprof hot-kernel HBM ≥ 50% / traffic reduction | **MISS — NEGATIVE RESULT** | marlin GEMM **25,046 KB/disp** vs legacy **24,488** = **+2.3% more** FETCH (FETCH+WRITE; never TCP_TCC_*). |
| **A4** | M0 baseline validity (real + pinned) | **PASS** | M0 lock real (12 cells completed=200); matched legacy A/B re-captured this session. |
| **A5** | ±2% reproducibility | **N/A** | Single matched run per arm; regression (−26%…−30%) is far larger than run-to-run noise, so direction is unambiguous. |

---

## 3. Per-cell delta table (REAL matched A/B, completed=200 each, failed=0)

Both arms captured this session under identical config (TP1, GPU0), flag verified
in `/proc/<pid>/environ`. Raw:
`/root/bench-w4a16/m3v/{legacy,marlin}_tp1/tp1_c{1,2,4}.json`.

| Cell    | legacy (tok/s) | marlin-ON (tok/s) | Δ% | legacy TPOT (ms) | marlin TPOT (ms) |
| ------- | -------------: | ----------------: | ----: | ---------------: | ---------------: |
| tp1_c1  |        29.7957 |           21.3734 | **−28.27%** | 32.42 | 43.96 |
| tp1_c2  |        54.8752 |           40.6238 | **−25.97%** | 34.70 | 44.82 |
| tp1_c4  |       101.4517 |           71.2048 | **−29.81%** | 36.72 | 49.26 |

**Geomean (c1,c2,c4):** marlin **39.5417** vs legacy **54.9452** → **−28.03%**.
Decode-only geomean(c1,c2): marlin 29.4664 vs legacy 40.4357 = **−27.13%**.

> Scope note: only the TP1 synthetic matched A/B was run this pass. The ~28%
> regression is large, consistent across all three concurrencies, and corroborated
> by the rocprof per-dispatch FETCH data (§4), so TP4 and coding cells were not
> needed to establish the verdict. They are not claimed.

---

## 4. Kernel-level rocprof — corroborates the regression

Source: `rocprof-marlin.md` (rocprofv3 `--pmc FETCH_SIZE WRITE_SIZE`, offline
drive harness `/root/{marlin,legacy}_drive_gemm.py`, M=1 decode shapes,
group_size=128). Re-aggregated this session with `/root/agg_rocprof.py` read
directly from the counter-collection CSVs.

| metric | legacy (flag=0) | marlin-ON (flag=1) | Δ |
|--------|----------------:|-------------------:|---|
| GEMM dispatches | 800 | 820 | +20 |
| GEMM FETCH (KB) | 19,590,538 | 20,537,745 | +4.83% |
| **GEMM FETCH per dispatch (KB)** | **24,488.2** | **25,046.0** | **+2.28%** |
| GEMM WRITE (KB) | 3,328 | 6,150 | +85% (tiny abs.) |
| distinct kernels | **4** | **14** | +10 helper kernels |

The rocprof data **disproves** the premise of the optimization on this
architecture: the marlin GEMM (1) **reads MORE from HBM per dispatch** (+2.3%),
not less — the repacked layout does not shrink the decode weight read; and (2)
emits **10 additional helper kernels** (arange/distribution/elementwise/reduce)
that the legacy path does not, adding launch overhead. This is the direct
mechanism behind the ~28% e2e regression (§3) and the ~11–13 ms TPOT increase.

---

## 5. Quality — numerically lossless

Source: `quality-marlin.md`. Both arms run through the SAME `quality_eval.py`
invocation, port 8000, identical flags, only the marlin flag differing
(`/proc/<pid>/environ` verified each).

| metric | legacy (flag=0) | marlin-ON (flag=1) | gate | verdict |
|--------|----------------:|-------------------:|------|:-------:|
| WikiText-2-raw perplexity | 11.788052776285937 | 11.787984062106219 | ≤ +1% | **PASS** (Δ −0.00058%) |
| Coding-agent (pass@1) | 9/10 (M0) | 9/10 | ≥ 9/10 | **PASS** |
| Needle-in-haystack | 1/5 | 1/5 | no regression | **PASS** |

Perplexity is **near-identical** (marlin 11.787984 vs legacy 11.788053,
Δ −0.00058%; mean_nll 2.46708071 vs 2.46708654; both 7506 scored tokens, 16
windows) — the sub-0.001% delta is FP reduction-order only, well inside the
≤+1% tolerance. The repack only changes the int4 memory layout and
pre-fuses scales/zeros; it does not change the dequant math (M1 correctness tests
pass at atol=1e-2/rtol=5e-2 vs FP32 dequant). Coding 9/10 — sole failure is the
intentional `max_subarray` fixture (asserts 7 vs true 6). Needle 1/5 depth
pattern identical to legacy + M0 (a property of this W4A16 build, not a marlin
regression — see `baseline-lock.md` §4).

---

## 6. Why Marlin does not work on gfx908 (CDNA1)

The Marlin design relies on a hardware feature gfx908 does not have:

- **gfx908 (CDNA1) has no `cp.async`** async-copy LDS pipelining, so `num_stages`
  is clamped to ≤2 (see the LDS table in `autotune-marlin.md`). The kernel
  **cannot overlap global→LDS load with MFMA** the way Marlin does on Ada/Hopper
  (the source of Marlin's ~1.3× ceiling). Without that overlap, the repacked
  layout's intended benefit cannot be realized.
- **At M=1 the decode GEMM is compute/issue/launch-bound**, not HBM-bound (M0
  server lock: hot kernel 63.68% of GPU time but only 15.30% HBM util). The
  repack adds setup/helper kernels (§4: 14 vs 4) and more per-dispatch FETCH
  (+2.3%), so on this path it is **pure overhead** — hence the ~28% regression.
- The legacy `mi100_w4a16_gemm_kernel` is already well-matched to gfx908's M=1
  decode characteristics; the Marlin-style transformation is a net loss here.

---

## 7. Negative-result clause (explicit)

- **A1 MISSED** — decode-synthetic geomean **−28.03%** (target ≥+5%): a large
  regression, not merely a shortfall.
- **A2 FAILED** — 0/3 cells non-regressing; all regress −26% to −30%.
- **A3 / VAL-PERF-003 MISSED** — marlin GEMM FETCH +2.3%/dispatch (no traffic
  reduction; ≥50% HBM not approached).

All carry the measured rocprof attribution above: the gfx908 decode GEMM is
compute/issue-bound at M=1, and the Marlin approach needs `cp.async`-style
overlap the architecture lacks. The repack is therefore a **net negative** for
decode on gfx908.

The change remains **correct** (M1 atol=1e-2/rtol=5e-2 vs FP32) and
**numerically lossless** (ppl/coding/needle preserved, §5), and it ships
**default-OFF**, so it causes no harm in the default configuration. But it
**must not be enabled** for decode on gfx908. The deliverable value is the
verified, reproducible *evidence* that this optimization path does not work on
CDNA1 — useful to prevent re-attempting it — not a performance gain.

---

## 8. Reproduction

Model `/models/Qwen3.5-9B-w4a16` (compressed-tensors W4A16, group_size 128);
interpreter `/opt/vllm-env/bin/python3`; ROCm `/opt/rocm/core-7.12`.

**Servers (port 8000, TP1 / TP4):**
```bash
/root/launch_w4a16_marlin_tp1.sh   # flag=1, TP1
/root/launch_w4a16_marlin_tp4.sh   # flag=1, TP4 (+ --disable-custom-all-reduce)
/root/launch_w4a16_legacy_tp1.sh   # flag=0 baseline / disable-smoke
```

**Per-cell synthetic bench (against the running server):**
```bash
/opt/vllm-env/bin/python3 -m vllm.entrypoints.cli.main bench serve \
  --model /models/Qwen3.5-9B-w4a16 --base-url http://127.0.0.1:8000 \
  --num-prompts 200 --request-rate inf --max-concurrency <c> --seed 42 \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --ignore-eos --save-result --result-dir <dir> --result-filename raw.json \
  --trust-remote-code
```

**Offline apples-to-apples rocprof HBM (no server → avoids engine wedge):**
```bash
/opt/rocm/core-7.12/bin/rocprofv3 --pmc FETCH_SIZE WRITE_SIZE -d <dir> \
  --output-format csv -- /opt/vllm-env/bin/python3 /root/marlin_drive_gemm.py
# legacy arm: /root/legacy_drive_gemm.py
python3 /root/hbm_pct.py <dir>/.../counter_collection.csv   # HBM% derivation
```

**Quality (needle + ppl, then coding):**
```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 /opt/vllm-env/bin/python3 -u \
  scripts/mi100/quality_eval.py --base-url http://127.0.0.1:8000 \
  --model /models/Qwen3.5-9B-w4a16 --out <out>.json \
  --ppl-tokens 8192 --max-len 2048 --needle-depths 5 --needle-max-ctx 4096
/opt/vllm-env/bin/python3 scripts/mi100/coding_agent_eval.py \
  --host 127.0.0.1 --port 8000 --model /models/Qwen3.5-9B-w4a16 --max-tokens 1024 --out <out>.json
```

**Raw artifact paths (verified this session):**
- matched A/B grid: `/root/bench-w4a16/m3v/{legacy,marlin}_tp1/tp1_c{1,2,4}.json`
- rocprof CSVs:
  `/root/bench-w4a16/m3/rocprof_marlin/pmc_offline/pmc_1/aimeme-MU72-SU0-00/885420_counter_collection.csv`,
  `/root/bench-w4a16/m3/rocprof_legacy_offline/pmc_offline/aimeme-MU72-SU0-00/1497237_counter_collection.csv`
  (aggregator `/root/agg_rocprof.py`)
- quality: `/root/bench-w4a16/m3/quality/{quality_ppl_needle_marlin,quality_ppl_needle_legacy,coding_agent_marlin}.json`

---

## 9. Tuning manifest reference

Source: `autotune-marlin.md` (kernel key `mi100_w4a16_marlin`).

- **80 autotune configs** = 48 TP=1 full-layer shapes + 32 TP=4 shard shapes,
  all ≤ 64 KiB gfx908 LDS budget (largest tile = prefill 128×256×32 @ 56.0 KiB).
- **Manifest SHA256 (80 files, verified this session):**
  `e6a933f108ac6e21261ab2e20bc53a3c856e2a8dc953543e1197cce3ec89ecce`
  ```bash
  cd vllm/model_executor/kernels/configs/gfx908 && \
    sha256sum mi100_w4a16_marlin_M*.json | sort -k2 | sha256sum
  ```

---

## 10. Forbidden-intrinsic scan + no-upstream-push audit

- This is a **Triton-only** change (`mi100_w4a16_marlin.py` + JSON autotune
  configs); no compiled `.so` changes.
- **Forbidden-intrinsic scan:** run at PR time (F-M4-pr) on the canonical
  `/opt/vllm-env` `.so` paths; expected clean (guard, since no `.so` changed).
- **No-upstream-push audit:** all bench/rocprof work is local to the fork host;
  no `vllm-project/vllm` operations. The PR (F-M4-pr) targets fork base
  `mi100-fixes`, with the upstream remote's push URL disabled first.

---

## Appendix — version manifest (M0 lock, observed)

```
vllm_sha : b7082f30d5956380a2e807ef368503faa9f70661
vllm     : 0.21.1rc1.dev593+g8b85c9c2c.d20260530.rocm712 (editable)
torch    : 2.11.0+rocm7.2
triton   : 3.5.1
hip      : 7.2.26015
rocm     : 7.12
python   : 3.12
gpu      : 4× AMD Instinct MI100 (gfx908)
toggle   : VLLM_MI100_W4A16_USE_MARLIN_REPACK (0/unset = legacy, 1 = marlin)
```
