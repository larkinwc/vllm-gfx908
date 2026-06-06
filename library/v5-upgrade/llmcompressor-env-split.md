<!-- SPDX-License-Identifier: Apache-2.0 -->

# llm-compressor environment split (gfx908 / MI100)

## Why

`llm-compressor` is the **offline** model-quantization tool used by
`examples/deployment/quantize_w8a8_mi100.py` (SmoothQuant + GPTQ → W8A8 INT8).
It is **not** a vLLM runtime/serving dependency — vLLM only needs
`compressed-tensors` (the on-disk format reader) at serve time.

`llm-compressor` hard-pins dependencies that conflict with the vLLM serving env:

| Package | llm-compressor 0.11.0 needs | vLLM serving env has |
|---|---|---|
| `transformers` | `>=4.56.1, <=4.57.6` | `5.x` (after this upgrade) |
| `compressed-tensors` | `==0.16.0` | `==0.17.0` (vLLM requirement) |
| `torch` | `<=2.11.0` | `2.11.0+rocm7.2` |

Because of the `transformers` and `compressed-tensors` pins, llm-compressor
**cannot** coexist with the transformers-v5 serving env. It was also already
broken in the serving env before this change (`llmcompressor 0.9.0.2` imported
`has_offloaded_params` from `compressed_tensors`, a symbol removed in 0.17).

## Layout

- **Serving env** `/opt/vllm-env` — vLLM, transformers v5, compressed-tensors
  0.17. No llm-compressor.
- **Quantization env** `/opt/llmcompressor-env` — llm-compressor 0.11.0,
  transformers 4.57.6, compressed-tensors 0.16.0. Offline use only.

## Create / recreate the quantization env

```bash
uv venv --python 3.12 /opt/llmcompressor-env
uv pip install --python /opt/llmcompressor-env/bin/python llmcompressor
# verify
/opt/llmcompressor-env/bin/python -c "from llmcompressor import oneshot; print('ok')"
```

## Run a quantization job

```bash
/opt/llmcompressor-env/bin/python examples/deployment/quantize_w8a8_mi100.py \
    --model Qwen/Qwen3.5-9B \
    --output /models/Qwen3.5-9B-w8a8
```

The resulting model directory is then served by the separate `/opt/vllm-env`.

> Note: `/opt/llmcompressor-env` installs the upstream (CUDA/CPU) torch wheel,
> which is fine for CPU-side calibration/quantization. It does not need the
> ROCm torch build and must stay isolated from the serving env.
