<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- markdownlint-disable MD013 MD024 MD031 MD032 MD040 MD041 MD046 MD056 MD058 MD060 -->

# M2-F1 — Correctness Gates (upstream-sync-2026-05-28)

> Surface C of the upstream-sync mission validation contract on the post-sync tip.
> Single-GPU MI100 (gfx908), CUDA_VISIBLE_DEVICES=0. Engine and harness commands
> captured verbatim below for reproducibility (per AGENTS.md §"Bench reproducibility").

## Environment

- **Worktree:** `/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/loose-rats-clean-phm2k`
- **Branch:** `upstream-sync-2026-05-28`
- **Commit:** `f0fe71a683e62365444524a871f7c813f6b1a4ae` (M1-F5 post-sync smoke + pre-commit finalize)
- **Python:** `/opt/vllm-env/bin/python` (3.12)
- **vLLM build:** editable install at worktree path
- **TP:** 1 (single GPU per feature description — only one GPU exercised in this sync mission)
- **Server flags:** `--dtype float16 --enforce-eager (TORCH_COMPILE_DISABLE=1) --max-model-len 32768 --block-size 32 --enable-prefix-caching --language-model-only --gpu-memory-utilization 0.90 --port 8000`
- **Env:** `CUDA_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm/core-7.12 PYTORCH_ROCM_ARCH=gfx908 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_SKINNY_GEMM=0 HSA_OVERRIDE_GFX_VERSION=9.0.8 HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library HF_HUB_OFFLINE=1`
- **Date (UTC):** 2026-05-28

## Gate summary

| Gate | Assertion | Metric | Measured | Threshold | Verdict |
|------|-----------|--------|---------:|-----------|:-------:|
| C1 — Engine init smoke | `C1-engine-init` | Qwen3.5-9B W4A16 load + `/v1/completions "Hello"` non-empty | 32-token completion returned (HTTP 200, model id served) | server healthy + non-empty completion | **PASS** |
| C2 — W8A8 perplexity | `C2-w8a8-ppl` | wikitext-2-raw-v1, 50×512 chunks, seed=0 | **9.6551** (mean_nll=2.2675) | ≤ 9.7583 (+1 % over 9.6518) | **PASS** (+0.05 % vs reference) |
| C3 — Coding eval | `C3-coding-eval` | 10-prompt coding-agent eval, temperature=0, seed=0 | **9/10** | ≥ 9/10 | **PASS** |
| C4 — Needle-in-haystack | `C4-needle` | 5 needles @ ctx 32 768 | **5/5** | 5/5 | **PASS** |

Overall: **PASS** on all four C-gates for this feature (C5 CK-FA dispatch is handled by M2-F2, out of M2-F1 scope).

## C1 — Engine init smoke (W4A16)

**Model:** `/models/Qwen3.5-9B-w4a16` (gptq-style pack-quantized 4-bit, group_size=128).
**Evidence file:** `library/smoke-init.log` (full stdout including `/health`, `/v1/models`, and the completion body).

Launch command (background-served):

```bash
CUDA_VISIBLE_DEVICES=0 ROCM_PATH=/opt/rocm/core-7.12 \
  PYTORCH_ROCM_ARCH=gfx908 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_SKINNY_GEMM=0 \
  TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1 HSA_OVERRIDE_GFX_VERSION=9.0.8 \
  HIPBLASLT_TENSILE_LIBPATH=/root/bench-int8-w4a16/tensilelite/merged_library/library \
  /opt/vllm-env/bin/python -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen3.5-9B-w4a16 --dtype float16 --tensor-parallel-size 1 \
    --max-model-len 32768 --block-size 32 --enable-prefix-caching \
    --language-model-only --gpu-memory-utilization 0.90 --port 8000
```

Server became healthy in ~80 s (one health-check poll at 10 s intervals). Smoke request:

```bash
curl -sf http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/Qwen3.5-9B-w4a16","prompt":"Hello","max_tokens":32,"temperature":0,"seed":0}'
```

Response (excerpt): non-empty 32-token completion (`finish_reason="length"`,
`completion_tokens=32`, `system_fingerprint=vllm-0.20.2rc1.dev107+gd960f21e4.d20260510-c9dcedf3`).
See `library/smoke-init.log` for the full body.

**Verdict: PASS** — server starts, `/health` returns 200, `/v1/completions` returns
a non-empty completion.

## C2 — W8A8 perplexity (post-sync re-measurement)

**Model:** `/models/Qwen3.5-9B-w8a8` (int8 weights + dynamic int8 activations,
the same model used in `BENCH_M2_PRODUCER_WIRE_IN.md` §3 quality gates).
**Evidence file:** `library/ppl_w8a8.json`.

Server: same launch invocation as C1, with `--model /models/Qwen3.5-9B-w8a8`
substituted. Server became healthy in ~60 s.

Harness:

```bash
/opt/vllm-env/bin/python scripts/m0_perplexity.py \
  --model /models/Qwen3.5-9B-w8a8 \
  --tokenizer /models/Qwen3.5-9B-w8a8 \
  --base-url http://127.0.0.1:8000/v1 \
  --chunks 50 --chunk-tokens 512 --seed 0 \
  --out /tmp/m2f1-logs/ppl_w8a8.json
```

Output (verbatim tail):

```text
[m0_perplexity] tokenized blob: 297053 tokens; using 50 chunks of 512 tokens
  chunk 1/50 tokens=511 elapsed=4.0s
  chunk 11/50 tokens=511 elapsed=8.5s
  chunk 21/50 tokens=511 elapsed=13.0s
  chunk 31/50 tokens=511 elapsed=17.7s
  chunk 41/50 tokens=511 elapsed=22.5s
[m0_perplexity]  ppl=9.6551 (25550 tokens scored in 26.7s)
```

| Quantity | Value |
|----------|------:|
| mean_nll | 2.2675 |
| perplexity | **9.6551** |
| n_tokens_scored | 25 550 |
| elapsed_s | 26.7 |
| Reference (prior M6, BENCH_M2_PRODUCER_WIRE_IN.md §3) | 9.6518 |
| Δ vs reference | +0.0033 abs (+0.034 %) |
| Gate threshold | ≤ 9.7583 (+1.0 %) |

Notes on the C2 gate's ±0.01 sub-clause: the validation-contract gate also asks
for "within ±0.01 of pre-sync re-measurement on `mi100-fixes` HEAD". This worker
did not re-measure perplexity on `mi100-fixes` HEAD as part of M2-F1 (the
feature description scopes M2-F1 to post-sync measurement). The pre-sync
reference value of 9.6518 documented in `BENCH_M2_PRODUCER_WIRE_IN.md` §3 was
measured on the post-PR-#41 tip (`bec0dcef0`); the post-sync delta vs that
reference is +0.0033 (within ±0.01), so this gate clause is satisfied against
the documented reference. A direct re-measurement on `mi100-fixes` HEAD is
deferred to M3-F1 (per the mission's pre-sync baseline plan).

**Verdict: PASS** (+0.034 % vs reference, well under the +1.0 % gate;
absolute delta 0.0033 within ±0.01).

## C3 — 10-prompt coding eval

**Evidence files:** `library/coding_m2f1_w8a8.json`, `library/coding_m2f1_w8a8.md`.

Harness:

```bash
/opt/vllm-env/bin/python scripts/eval_coding_prompts.py \
  --prompts tests/eval/coding_prompts.json \
  --out-dir /tmp/m2f1-logs/coding \
  --label m2f1_w8a8
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
  [eval] max_subarray (python) … FAIL (py_compile: Sorry: IndentationError: unexpected indent (snippet.py, line 2))
  [eval] node_check_js (javascript) … PASS
  [eval] bash_check (bash) … PASS

Pass: 9/10 (PASS)
```

- **Pass count:** 9/10
- **Threshold:** ≥ 9/10
- The single failure (`max_subarray`) is the pre-existing IndentationError in
  the eval fixture documented in `BENCH_M2_PRODUCER_WIRE_IN.md` §3 (coding
  ceiling is 9/10 on this fixture; do NOT fix the eval fixture per the
  project's anti-pattern #6). This matches both the M2-producer mission's
  9/10 result and the M6 baseline.

**Verdict: PASS** (matches M2 baseline ceiling).

## C4 — Needle-in-haystack (ctx=32 768)

**Evidence file:** `library/needle_m2f1_w8a8.json`.

Harness:

```bash
/opt/vllm-env/bin/python scripts/eval_needle.py \
  --ctx 32768 --probes 5 --model /models/Qwen3.5-9B-w8a8 \
  --out /tmp/m2f1-logs/m2f1_needle32k.json
```

Output:

```text
  [1/5] Magic apple count        depth=10% — PASS
  [2/5] Crimson tower height     depth=30% — PASS
  [3/5] Lunar passcode           depth=50% — PASS
  [4/5] Coral fish species       depth=70% — PASS
  [5/5] Captain's birthday       depth=90% — PASS

Needle@32768: 5/5 (gate PASS)
```

- **passes:** 5
- **total:** 5
- **gate_5_of_5:** true

**Verdict: PASS** (matches prior 5/5 baseline).

## Assertion claim ledger

| Assertion | Evidence |
|-----------|----------|
| `C1-engine-init` | `library/smoke-init.log` — `/health` 200, `/v1/models` lists the model, `/v1/completions` returned a 32-token non-empty completion for prompt "Hello". |
| `C2-w8a8-ppl` | `library/ppl_w8a8.json` — perplexity 9.6551 ≤ 9.7583, within ±0.01 of reference 9.6518. |
| `C3-coding-eval` | `library/coding_m2f1_w8a8.{json,md}` — 9/10 pass count, gate ≥ 9/10. |
| `C4-needle` | `library/needle_m2f1_w8a8.json` — `passes=5/total=5/gate_5_of_5=true`. |

## Deferred / not-claimed

- **C5 — CK-FA backend dispatch** (`C5-ckfa-dispatch`): out of scope for M2-F1.
  Handled by M2-F2 per the feature partitioning.
- **Pre-sync `mi100-fixes`-HEAD perplexity re-measurement** for the ±0.01
  sub-clause of C2: deferred to M3-F1 (the baseline-bench feature). The current
  M2-F1 result is reported against the documented pre-sync reference (9.6518)
  in `BENCH_M2_PRODUCER_WIRE_IN.md` §3 and falls within the ±0.01 absolute band.
