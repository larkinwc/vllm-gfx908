# M3-F2 Quality Gates — W8A8 (M1-promoted placeholder-view code)

**Feature**: M3-F2 (VAL-M3-002)
**Run date**: 2026-05-25
**Model**: `/models/Qwen3.5-9B-w8a8`
**Server**: TP=1, single GPU, `--gpu-memory-utilization 0.90`, `--max-model-len 32768`, `--enable-prefix-caching`, `--block-size 32`, `--dtype float16`. Standard pinned env (ROCm 7.12 + AITER + skinny-gemm off + TensileLite merged library + tuning JSONs).
**Codepath**: `mi100/redundant-silu-elimination` @ commit `7239a3cf8`. M1-F5 promoted placeholder-view path is unconditional whenever `VLLM_MI100_DISABLE_FUSED_ACT_QUANT` is unset (default).
**Run dir**: `/root/bench-int8-w4a16-redundant-silu/m3-quality/`

## Gate Summary

<!-- markdownlint-disable MD060 -->
| Gate                    | Threshold              | This run                    | Verdict |
|-------------------------|------------------------|-----------------------------|---------|
| W8A8 perplexity (wikitext-2-raw-v1, 50×512 chunks, seed=0) | ≤ +1 % over **9.6518** (i.e. ≤ 9.7483) | **9.6936** (Δ = +0.43 %) | **PASS** |
| Coding-agent eval (10 prompts, temperature=0, seed=0)      | ≥ 9 / 10               | **8 / 10**                  | **FAIL** |
| Needle-in-haystack @ ctx 32 768 (5 needles)                | 5 / 5                  | **5 / 5**                   | **PASS** |
<!-- markdownlint-enable MD060 -->

## Overall Verdict — VAL-M3-002 FAIL

2 of 3 gates pass. **Coding gate fails reproducibly** (8/10 on two consecutive independent runs against the same server). This blocks promotion of the M1-F5 placeholder-view path until either:

1. the regression is root-caused and fixed in M3 code, or
2. the orchestrator explicitly waives the coding gate with documented justification.

## Evidence Files

- `ppl_w8a8.json` — perplexity raw output (mean_nll, n_tokens_scored, elapsed_s)
- `m6_coding_eval_m3_w8a8.json` / `.md` — first coding run (per-prompt grades + extracted code + response bodies)
- `m6_coding_eval_m3_w8a8_retry.json` / `.md` — second coding run (confirms determinism of the failure)
- `m6_needle32k.json` — needle-in-haystack 5-probe raw output
- `server_tp1.log` — engine startup log for the TP=1 single-GPU server used for these evals
- `start_w8a8_tp1.sh` — exact launch script (CUDA_VISIBLE_DEVICES=0, gpu-memory-utilization=0.90)

## Per-Gate Details

### 1. W8A8 Perplexity — PASS

```text
$ scripts/m0_perplexity.py --model /models/Qwen3.5-9B-w8a8 \
    --tokenizer /models/Qwen3.5-9B-w8a8 --base-url http://127.0.0.1:8000/v1 \
    --chunks 50 --chunk-tokens 512 --seed 0
[m0_perplexity] tokenized blob: 297053 tokens; using 50 chunks of 512 tokens
[m0_perplexity]  ppl=9.6936 (25550 tokens scored in 22.8s)
```

| Quantity              | Value     |
|-----------------------|-----------|
| Reference (prior M6)  | 9.6518    |
| This run              | 9.6936    |
| Δ (absolute)          | +0.0418   |
| Δ (relative)          | +0.433 %  |
| Gate threshold        | ≤ +1.0 %  |
| Verdict               | **PASS**  |

The +0.43 % drift is well inside the +1 % gate and is consistent with the prior M6 measurement (9.6518) plus minor sampling jitter from the M1-promoted placeholder-view path (the placeholder is a real-shaped fp16 view of `gate_up[:,:H]`, not a numerical bypass).

### 2. Coding-Agent Eval — FAIL (reproducible 8/10)

Two independent passes over `tests/eval/coding_prompts.json` against the same warm server, each invoked via `scripts/eval_coding_prompts.py` with `temperature=0, seed=0`:

<!-- markdownlint-disable MD060 -->
| Run            | Pass count | Failing prompts                            |
|----------------|------------|--------------------------------------------|
| run-1 (m3_w8a8)        | 8 / 10  | `anagram`, `max_subarray`         |
| run-2 (m3_w8a8_retry)  | 8 / 10  | `anagram`, `max_subarray`         |
<!-- markdownlint-enable MD060 -->

Prior M6 reference (`/root/bench-int8-w4a16/final/m6_coding_eval_m6_w8a8.json`): **9 / 10** (only `max_subarray` failed). The new failure is `anagram`.

#### `anagram` (regression vs M6 baseline)

The model returns the function below. It strips whitespace and lowercases but **does not check that the two strings share the same multiset of characters** (and does not sort), so any pair of equal-length lowercased strings without spaces is judged an anagram only when they are literally equal:

```python
def is_anagram(a: str, b: str) -> bool:
    def normalize(s: str) -> str:
        return "".join(s.lower().split())
    return normalize(a) == normalize(b)
```

Failing sanity cases (expected `True`, got `False`):

- `("listen", "silent")` — different character orders
- `("Hello", "olleh")` — different character orders

The prior M6 run on the same prompt produced a correct anagram implementation (using sorted-string or character-count comparison). This is a real semantic regression in the coding behavior of the M1-promoted W8A8 stack, **not** an evaluation-harness artifact.

#### `max_subarray` (pre-existing, unchanged)

Same Kadane's-algorithm scaffolding emitted with leading indent that breaks `py_compile`. This identical IndentationError was already present in the M6 baseline (`compiles=False, runs=False, correct=False` in `/root/bench-int8-w4a16/final/m6_coding_eval_m6_w8a8.json`). Pre-existing failure, not introduced by M3.

### 3. Needle-in-Haystack @ 32 k — PASS

```text
$ scripts/eval_needle.py --ctx 32768 --probes 5 --model /models/Qwen3.5-9B-w8a8
[1/5] Magic apple count        depth=10% — PASS
[2/5] Crimson tower height     depth=30% — PASS
[3/5] Lunar passcode           depth=50% — PASS
[4/5] Coral fish species       depth=70% — PASS
[5/5] Captain's birthday       depth=90% — PASS
Needle@32768: 5/5 (gate PASS)
```

## Notes / Pre-Existing Issues

- `scripts/launch_w8a8_tp4_c1.sh --serve-only` at `--gpu-memory-utilization 0.93` OOMs as soon as `/v1/completions` is exercised with `echo=true, logprobs=0` (970 MiB allocation on each of 4 GPUs fails after a partial fill from the 93 % KV-cache reservation). This was first hit at the start of M3-F2 before falling back to a TP=1 single-GPU server at 0.90 util. Cause is the prompt-logprobs path materialising a `[B,T,V]` softmax tensor on top of the already-large KV cache. **Suggestion (non-blocking):** add a `LAUNCH_GPU_MEM_UTIL` override hook to the production launch script(s) so future ppl runs can dial back utilisation without editing the script. Tracked here for the mission-ops-worker who will assemble the final PR.

---

## M1 Quality Gates — W4A16 AWQ-vs-GPTQ A/B (3-path comparison)

**Feature**: M1-F4 (gate_summary.json synthesis)
**Run date**: 2026-05-27
**Paths**: A=GPTQ (`/models/Qwen3.5-9B-w4a16`), B=AWQ-calibrated (`/models/Qwen3.5-9B-AWQ-INT4`), C=AWQ-gemm (`/models/Qwen3.5-9B-AWQ-gemm`)
**Server**: TP=1, M4 env stack (KV_CACHE_DTYPE=int8_per_token_head, ENABLE_CHUNKED_PREFILL=1, MAX_NUM_BATCHED_TOKENS=4096)
**FP16 reference perplexity**: 9.527 (from BENCH_INT8_W4A16_HBM.md baseline)
**Threshold**: ≤ 9.527 × 1.03 = 9.81281

<!-- markdownlint-disable MD060 -->
| Path | Perplexity | Δ vs FP16 | Perplexity Gate | Needle (5/5) | Coding (≥8/10) | Path Gate |
|---|---|---|---|---|---|---|
| A (GPTQ) | 9.823 | +3.11% | **FAIL** | PASS | PASS (9/10) | FAIL |
| B (AWQ-calibrated) | 9.916 | +4.08% | **FAIL** | PASS | PASS (9/10) | FAIL |
| C (AWQ-gemm) | 9.956 | +4.51% | **FAIL** | PASS | PASS (10/10) | FAIL |
<!-- markdownlint-enable MD060 -->

## Key Observations

- All three paths fail the +3% perplexity gate. **Mission explicitly non-blocking** (`mission_blocking=false` in gate_summary.json).
- Path A (GPTQ) is the same model used to derive the FP16 baseline in BENCH_INT8_W4A16_HBM.md, yet fails the gate marginally (+3.11%). This suggests the +3% tolerance may be too tight for this eval config (50×512 chunks, seed=0, TP=1, KV-INT8), or that the FP16 reference was captured under slightly different conditions.
- Needle and coding gates pass for all paths (5/5 needle; 9/10 and 10/10 coding).
- Excluded paths are flagged in gate_summary.json under `excluded_from_headline: ["a","b","c"]` for M4 verdict matrix use.
