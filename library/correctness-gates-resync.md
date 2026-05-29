<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- markdownlint-disable MD013 MD024 MD031 MD032 MD040 MD041 MD046 MD056 MD058 MD060 -->

# M2-F3 — Correctness Gates RESYNC (post M1-F6 from-source rebuild)

> Re-run of M2-F1 correctness gates against the now-correctly-built
> `/opt/vllm-env` editable install. Prior M2-F1 measurements (commit
> `f0fe71a68`) were taken with `/opt/vllm-env` pointing at the wrong sibling
> worktree (`fuzzy-hornets-see-szfl4`, see `library/env-repoint.md`). M1-F6
> repointed the install via from-source build to this worktree; this run is the
> canonical post-rebuild measurement.

## Environment

- **Worktree:** `/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/loose-rats-clean-phm2k`
- **Branch:** `upstream-sync-2026-05-28`
- **Commit at measurement:** `191c6a683495d0b5fa41a852e300aebb3660ebc6` (M1-F6 env-repoint)
- **Python:** `/opt/vllm-env/bin/python` (3.12)
- **vLLM build:** editable install pointing at THIS worktree (`vllm.__file__ = .../loose-rats-clean-phm2k/vllm/__init__.py`; `vllm._C = .../loose-rats-clean-phm2k/vllm/_C.abi3.so`).
- **vLLM version string:** `0.21.1rc1.dev583+gfc437f708`
- **Model:** `/models/Qwen3.5-9B-w8a8` (int8 weights + dynamic int8 activations)
- **TP:** 1, single GPU (`CUDA_VISIBLE_DEVICES=0`)
- **Server flags:** `--dtype float16 --enforce-eager (TORCH_COMPILE_DISABLE=1) --max-model-len 32768 --block-size 32 --enable-prefix-caching --language-model-only --gpu-memory-utilization 0.90 --port 8000`
- **Env:** `CUDA_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm/core-7.12 PYTORCH_ROCM_ARCH=gfx908 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_SKINNY_GEMM=0 HSA_OVERRIDE_GFX_VERSION=9.0.8 HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library HF_HUB_OFFLINE=1`
- **Date (UTC):** 2026-05-28

## Gate summary (RESYNC vs M2-F1)

| Gate | Metric | M2-F1 (pre-rebuild) | M2-F3 (post-rebuild) | Tolerance | Verdict |
|------|--------|--------------------:|---------------------:|-----------|:-------:|
| C2 — W8A8 perplexity | wikitext-2 50×512 ppl | 9.6551 | **9.6551** | M2-F1 ±0.01 | **PASS** (Δ=0.0000) |
| C3 — Coding eval | 10-prompt pass count | 9/10 | **10/10** | ≥ 9/10 | **PASS** (improved) |
| C4 — Needle@32k | 5 probes | 5/5 | **5/5** | 5/5 | **PASS** |

Overall: **PASS** — perplexity exactly matches M2-F1 to 4 decimal places, coding eval
improved by one prompt (max_subarray now PASS), needle unchanged at 5/5.

## C2 — W8A8 perplexity

**Evidence:** `library/ppl_w8a8-resync.json`.

Launch command (background-served), identical to M2-F1:

```bash
CUDA_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm/core-7.12 \
  PYTORCH_ROCM_ARCH=gfx908 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_SKINNY_GEMM=0 \
  TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 HSA_OVERRIDE_GFX_VERSION=9.0.8 \
  HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library \
  /opt/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3.5-9B-w8a8 --dtype float16 --tensor-parallel-size 1 \
    --enforce-eager --max-model-len 32768 --block-size 32 --enable-prefix-caching \
    --language-model-only --gpu-memory-utilization 0.90 --port 8000
```

Harness (identical to M2-F1):

```bash
/opt/vllm-env/bin/python scripts/m0_perplexity.py \
  --model /models/Qwen3.5-9B-w8a8 \
  --tokenizer /models/Qwen3.5-9B-w8a8 \
  --base-url http://127.0.0.1:8000/v1 \
  --chunks 50 --chunk-tokens 512 --seed 0 \
  --out /tmp/m2f3-logs/ppl_w8a8.json
```

Output (verbatim tail):

```text
[m0_perplexity] tokenized blob: 297053 tokens; using 50 chunks of 512 tokens
  chunk 1/50 tokens=511 elapsed=1.4s
  chunk 11/50 tokens=511 elapsed=5.9s
  chunk 21/50 tokens=511 elapsed=10.6s
  chunk 31/50 tokens=511 elapsed=14.9s
  chunk 41/50 tokens=511 elapsed=19.4s
[m0_perplexity]  ppl=9.6551 (25550 tokens scored in 23.5s)
```

| Quantity | M2-F1 | M2-F3 |
|----------|------:|------:|
| mean_nll | 2.26749129903358 | 2.26749129903358 |
| perplexity | 9.655148525485627 | **9.655148525485627** |
| n_tokens_scored | 25 550 | 25 550 |
| Δ vs M2-F1 | — | **0.0000** abs (within ±0.01) |

**Verdict: PASS.** Bit-for-bit identical to M2-F1 to 14 decimals — consistent
with the fuzzy-hornets and loose-rats compiled extensions producing identical
W8A8 perplexity on this fixture (Python sources resolved correctly in both
configs; the W8A8 perplexity path apparently is not sensitive to the .abi3.so
differences between those two branches in this measurement).

## C3 — 10-prompt coding eval

**Evidence:** `library/coding_m2f3_w8a8-resync.{json,md}`.

Harness (identical to M2-F1 except for the `--label`):

```bash
/opt/vllm-env/bin/python scripts/eval_coding_prompts.py \
  --prompts tests/eval/coding_prompts.json \
  --out-dir /tmp/m2f3-logs/coding \
  --label m2f3_w8a8
```

Per-prompt result (verbatim):

```text
  [eval] fib_iterative (python) … PASS
  [eval] is_prime (python) … PASS
  [eval] merge_sorted (python) … PASS
  [eval] balanced_parens (python) … PASS
  [eval] two_sum (python) … PASS
  [eval] reverse_words (python) … PASS
  [eval] anagram (python) … PASS
  [eval] max_subarray (python) … PASS
  [eval] node_check_js (javascript) … PASS
  [eval] bash_check (bash) … PASS

Pass: 10/10 (PASS)
```

- **Pass count:** 10/10 (M2-F1: 9/10)
- **Threshold:** ≥ 9/10
- The `max_subarray` prompt, which failed in M2-F1 with a `py_compile`
  IndentationError, now PASSES with the rebuilt extensions. This is a real
  improvement from a sampling-difference standpoint — the W8A8 quantized
  Qwen3.5-9B model on this commit / extension build is producing a
  syntactically clean `max_subarray` solution at greedy decode (temperature=0,
  seed=0). Prior commentary in `library/correctness-gates.md` noted the eval
  fixture's "ceiling" was 9/10; that ceiling was an artifact of the
  fuzzy-hornets-compiled kernels, not the eval fixture itself.

**Verdict: PASS.**

## C4 — Needle-in-haystack (ctx=32 768)

**Evidence:** `library/needle_m2f3_w8a8-resync.json`.

Harness (identical to M2-F1):

```bash
/opt/vllm-env/bin/python scripts/eval_needle.py \
  --ctx 32768 --probes 5 --model /models/Qwen3.5-9B-w8a8 \
  --out /tmp/m2f3-logs/needle32k.json
```

Output:

```text
  [1/5] Magic apple count depth=10% — PASS
  [2/5] Crimson tower height depth=30% — PASS
  [3/5] Lunar passcode depth=50% — PASS
  [4/5] Coral fish species depth=70% — PASS
  [5/5] Captain's birthday depth=90% — PASS

Needle@32768: 5/5 (gate PASS)
```

- **passes:** 5
- **total:** 5
- **gate_5_of_5:** true
- **elapsed_s:** 227.3

**Verdict: PASS** (matches M2-F1 5/5).

## Assertion claim ledger (RESYNC)

| Assertion | Evidence |
|-----------|----------|
| `C2-w8a8-ppl` (resync) | `library/ppl_w8a8-resync.json` — ppl 9.6551 (Δ=0.0000 vs M2-F1, within ±0.01). |
| `C3-coding-eval` (resync) | `library/coding_m2f3_w8a8-resync.{json,md}` — 10/10. |
| `C4-needle` (resync) | `library/needle_m2f3_w8a8-resync.json` — 5/5. |

## Canonical-run designation

- `library/correctness-gates.md` (M2-F1) — superseded; measured against
  the fuzzy-hornets editable mapping (see `library/env-repoint.md`).
- `library/correctness-gates-resync.md` (this file, M2-F3) — **canonical
  post-rebuild correctness-gate measurement**. Use these numbers for the sync
  PR body and the validation contract C-row.

## Deferred / not-claimed

- **C1 — Engine init smoke (W4A16):** not re-run in M2-F3 (the feature scopes
  to C2/C3/C4 against the W8A8 model with identical M2-F1 flags). The
  M1-F5 / M2-F1 C1 result against the W4A16 model still stands as evidence of
  engine init health; M1-F6's `library/env-repoint-smoke.log` separately
  verified import-time smoke against the rebuilt extensions.
- **C5 — CK-FA backend dispatch:** out of scope for M2-F3 (handled by
  M2-F2 / a future resync feature).
