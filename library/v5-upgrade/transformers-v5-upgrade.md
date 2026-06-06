<!-- SPDX-License-Identifier: Apache-2.0 -->

# transformers v4 → v5 upgrade + llm-compressor decouple (gfx908 / MI100)

Branch: `mi100/transformers-v5-upgrade` (off `27aa70230`).

## Summary

The serving env `/opt/vllm-env` was pinned to `transformers 4.57.3` only because
the offline quantization tool `llm-compressor` (not a vLLM runtime dependency)
was installed into it and caps `transformers<=4.57.x`. Decoupling llm-compressor
into its own venv frees the serving env to move to transformers v5, which vLLM
0.22 supports (and which vLLM 0.24 will *require*, since v4 is deprecated).

| | before | after |
|---|---|---|
| `transformers` (serving env) | 4.57.3 | **5.10.2** |
| `tokenizers` | 0.21.x | 0.22.2 |
| `huggingface-hub` | 0.36.2 | 1.18.0 |
| `compressed-tensors` | 0.17.0 | 0.17.0 (unchanged) |
| `llm-compressor` (serving env) | 0.9.0.2 (broken import) | **removed** |
| `llm-compressor` | — | 0.11.0 in `/opt/llmcompressor-env` |

## Phase 1 — llm-compressor decouple

See `library/v5-upgrade/llmcompressor-env-split.md`. llm-compressor was
already broken in the serving env (`0.9.0.2` imported `has_offloaded_params`
removed in compressed-tensors 0.17). Moved to `/opt/llmcompressor-env`
(uv venv, python 3.12, llm-compressor 0.11.0, transformers 4.57.6,
compressed-tensors 0.16.0). Updated `examples/deployment/quantize_w8a8_mi100.py`
prereqs + ImportError message to point at the dedicated venv.

## Phase 2 — transformers v5 install

```bash
uv pip install --python /opt/vllm-env/bin/python "transformers>=5.6"
# -> transformers 5.10.2, tokenizers 0.22.2, huggingface-hub 1.18.0
```

vLLM's `requirements/common.txt` already allows v5.6+ (`!= 5.0.*..!= 5.5.0`),
so no requirements change is needed. `uv pip check` shows only the pre-existing
unrelated `eagle-llm` pins (torch==2.0.1 / transformers==4.46.2, never
satisfiable in this ROCm env).

## Phase 3 — validation (4× MI100, gfx908)

Serve env vars (per `library/correctness-gates.md`, plus the two notes below):
`VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_RMSNORM=0 VLLM_ROCM_USE_SKINNY_GEMM=0
VLLM_MI100_DISABLE_CUSTOM_AR=0 TORCH_COMPILE_DISABLE=1 HF_HUB_OFFLINE=1
HSA_OVERRIDE_GFX_VERSION=9.0.8 VLLM_WORKER_MULTIPROC_METHOD=spawn
PYTORCH_ROCM_ARCH=gfx908 ROCM_PATH=/opt/rocm/core-7.12`

### Serve smoke — all PASS on transformers 5.10.2
| model | result |
|---|---|
| Llama-2-7b-hf | ✅ "...a city of contrasts. The city is home to the Eiffel Tower" |
| Qwen3.5-9B-w8a8 | ✅ "Paris." — tokenizer override works on v5 |
| Qwen3.5-9B-w4a16 | ✅ "Paris." |

The Qwen3.5 tokenizer-class override (PR #71) is **version-agnostic** and
confirmed working on v5: the concrete `Qwen2TokenizerFast` is loaded, cached,
and passes `Qwen3VLProcessor`'s strict isinstance check.

### Benchmarks (Llama-2-7b, gfx908, single GPU)
| bench | pre-v5 (4.57.3) | v5 (5.10.2) | note |
|---|---|---|---|
| throughput (in1024/out256, 100 prompts) | 1404.8 tok/s | **1538.3 tok/s** | +9.5%, no regression |
| CK-FA decode latency (default dims, eager) | 2.5287 s | 3.835 s | ⚠ regression, see below |

## Known issues / follow-ups

### A. aiter `module_rmsnorm` CK JIT build failure (env, not v5)

With `VLLM_ROCM_USE_AITER=1` (default rmsnorm), aiter tries to JIT-compile
`module_rmsnorm` / `module_rmsnorm_quant`, which need a Composable Kernel source
tree. aiter_meta in this env ships only `3rdparty/ck_helper`, not the full
`composable_kernel`, so the build fails:

- without `CK_DIR`: `fatal error: 'rmsnorm2d_fwd.hpp' file not found`
- with `CK_DIR=/home/aimeme/Desktop/rocm-flash-attention/csrc/composable_kernel`:
  `module_rmsnorm` blob-gen succeeds but `module_rmsnorm_quant` fails to compile
  (CK version mismatch between the flash-attention CK tree and aiter's
  `rmsnorm_quant_kernels.cu`).

**Workaround (in use):** `VLLM_ROCM_USE_AITER_RMSNORM=0`. This keeps aiter for
GEMM and uses vLLM's native/`vllm_c` rmsnorm. Measured to have **no latency
cost** (CK-FA latency identical with aiter+RMSNORM=0 vs aiter fully off:
3.835 s both).

This is independent of transformers version (the aiter JIT gate does not depend
on transformers). It was latent before — the pre-v5 serves happened not to
trigger the rmsnorm JIT path. Proper fix = ship a matching CK tree in aiter_meta
or point `CK_DIR` at a CK revision compatible with aiter's `rmsnorm_quant`
kernels.

### B. CK-FA decode-latency regression (2.53 s → 3.84 s)

Same model/dims/eager config; throughput went *up* while CK-FA decode latency
went *down*. Isolation showed it is **not** caused by the rmsnorm workaround
(identical with aiter fully off). Likely a v5/attention-path or env interaction.
Deferred to a dedicated perf pass — not a correctness blocker; serving and
generation are correct on all models.
