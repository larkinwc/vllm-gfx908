<!-- SPDX-License-Identifier: Apache-2.0 -->

# Post-merge fixes — 2026-06-05 upstream sync

Two issues were surfaced while building + benching the
`upstream/main -> emdash/honest-candies-tap-vhke2` merge (commit `6c53c45d7`,
~270 upstream commits). Neither was caused by the merge-conflict resolution.

## 1. CK-FA backend: stale `flash_attn` editable pointer

**Symptom.** `vllm bench latency --attention-backend ROCM_CK_FA` crashed with
`ModuleNotFoundError: No module named 'flash_attn'` in
`vllm/v1/attention/backends/rocm_ck_fa.py::_resolve_flash_attn`.

**Root cause.** `flash_attn` 2.8.4 *is* installed into `/opt/vllm-env` as an
editable package, but its setuptools finder MAPPING pointed at
`/tmp/rocm-flash-attention-test/...`, a build dir that gets wiped on reboot.
The real CK flash-attention tree (with the built
`flash_attn_2_cuda.cpython-312-*.so`) persists at
`/home/aimeme/Desktop/rocm-flash-attention`.

**Fix.** Repoint the editable finder MAPPING at the persistent tree. Use the
idempotent helper:

```bash
bash library/fix-flash-attn-editable-pointer.sh
```

It rewrites the `MAPPING` line in
`/opt/vllm-env/lib/python3.12/site-packages/__editable___flash_attn_2_8_4_finder.py`
and verifies `import flash_attn` + `flash_attn_2_cuda`.

**Verification.** After the fix the CK-FA decode-latency bench runs and the log
shows `Using ROCM_CK_FA backend (selected via --attention-backend)` (see
`library/bench-post-merge-2026-06-05/ckfa-latency.json`).

> Not a merge regression — purely an environment pointer to a wiped `/tmp`
> directory.

## 2. Qwen3.5 multimodal serve crash: tokenizer-class override

**Symptom.** Serving `/models/Qwen3.5-9B-w8a8` crashed during multimodal
processor construction:

```
TypeError: Received a CachedPreTrainedTokenizerFast for argument tokenizer,
but a ('Qwen2Tokenizer', 'Qwen2TokenizerFast') was expected.
```

**Root cause.** `qwen3_5` is a fork-added entry in
`_MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS` (in `vllm/tokenizers/registry.py`).
transformers 4.57.3 does not recognise the `qwen3_5` model type, so
`AutoTokenizer.from_pretrained` raises `ValueError`; the override exists to
bypass that. The override loaded the **generic** `PreTrainedTokenizerFast`
(via the v5 `TokenizersBackend` fallback) and wrapped it through
`get_cached_tokenizer`, producing a `CachedPreTrainedTokenizerFast`.

The multimodal `Qwen3VLProcessor` (which comes from **transformers**, unlike
the *vendored* Step3 processors) enforces a strict
`isinstance(tokenizer, (Qwen2Tokenizer, Qwen2TokenizerFast))` check and rejects
the generic class.

Upstream does not hit this because:
- it does not carry `qwen3_5` in the override set, and
- its own override entries (`step3_vl`, `step3p7`) use vLLM-vendored
  processors that do not do the strict transformers isinstance check.

**Fix.** Load the *concrete* fast tokenizer class the processor expects.
`get_cached_tokenizer` dynamically subclasses `tokenizer.__class__`, so loading
`Qwen2TokenizerFast` yields a `CachedQwen2TokenizerFast` that still passes the
isinstance check and keeps the vLLM-required cached attributes.

`vllm/tokenizers/registry.py` now carries a concrete-class map alongside the
override set:

```python
_MODEL_TYPE_TO_TOKENIZER_CLASS_OVERRIDE: dict[str, str] = {
    "qwen3_5": "transformers:Qwen2TokenizerFast",
}
```

When an overridden model type is in this map, the override loads that concrete
class instead of the generic fallback. Entries without a mapping
(`step3_vl`, `step3p7`) keep the generic-fallback behaviour unchanged.

**Verification.** `vllm serve /models/Qwen3.5-9B-w8a8` now starts cleanly and
generates coherently ("The capital of France is Paris."). The text-only path
and the v5 `TokenizersBackend` fallback are preserved.

## transformers version context

The environment runs `transformers 4.57.3`. vLLM's own requirement
(`transformers >= 4.56.0, != 5.0.*..!= 5.5.0`) permits v5.6+; the upper bound is
imposed by `llmcompressor 0.9.0.2` (`transformers<=4.57.3`), an env-only
quantization tool that is **not** a vLLM dependency.

vLLM 0.22 deprecates the transformers-v4 codepath (removed in v0.24.0). A
transformers-v5 upgrade is a separate, larger initiative that also requires
replacing/upgrading `llmcompressor` (already in conflict with the ROCm torch
build). The Qwen3.5 fix above is version-agnostic and works on both v4 and v5.
