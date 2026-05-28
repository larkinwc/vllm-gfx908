# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M3-F1 rocprofv3 inference target.

Drives vLLM LLM().generate() synchronously and exits cleanly via natural
Python finalization — avoiding the SIGTERM-based EngineCore teardown that
truncates rocprofv3 CSV output on TP=4 (see kernel_trace.log
"rocprofv3 caught signal 15" path).

Args:
    --model PATH        model directory
    --tp INT            tensor parallel size
    --num-prompts INT   number of prompts (default 20)
    --input-len INT     random input length (default 1024)
    --output-len INT    output length (default 32)
    --quantization STR  optional quantization flag (e.g. awq)
    --seed INT          default 42

Mirrors the same engine config as `vllm bench throughput` so kernels are
representative.
"""

import argparse
import random
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Import vLLM after argparse to keep --help fast.
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    random.seed(args.seed)

    # Build random integer prompts of the requested input_len. We use token
    # ids directly via prompt_token_ids to bypass tokenization differences.
    # ModelConfig accepts `language_model_only` as an InitVar; pass via
    # multimodal_config dict so it gets routed correctly.
    llm_kwargs = dict(
        model=args.model,
        dtype="float16",
        tensor_parallel_size=args.tp,
        max_model_len=32768,
        block_size=32,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.93,
        trust_remote_code=True,
        seed=args.seed,
        kv_cache_dtype="int8_per_token_head",
        enable_chunked_prefill=True,
        max_num_batched_tokens=4096,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    if args.tp >= 2:
        llm_kwargs["disable_custom_all_reduce"] = True

    print(
        f"[awq_ab_rocprof_inference] LLM init "
        f"model={args.model} tp={args.tp} quant={args.quantization}",
        flush=True,
    )
    llm = LLM(**llm_kwargs)

    # Use a conservative vocab range valid for Qwen3.5 family (~152k)
    vocab_max = 100000
    prompts_token_ids = []
    for _ in range(args.num_prompts):
        ids = [random.randint(10, vocab_max) for _ in range(args.input_len)]
        prompts_token_ids.append(ids)

    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.output_len,
        ignore_eos=True,
        seed=args.seed,
    )

    print(
        f"[awq_ab_rocprof_inference] generate "
        f"n={args.num_prompts} in={args.input_len} out={args.output_len}",
        flush=True,
    )
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in prompts_token_ids]
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling,
        use_tqdm=False,
    )
    total_out = sum(len(o.outputs[0].token_ids) for o in outputs)
    print(
        f"[awq_ab_rocprof_inference] done "
        f"n_out_seqs={len(outputs)} total_output_tokens={total_out}",
        flush=True,
    )
    # Explicit cleanup: drop LLM reference so engine workers can shutdown
    # via Python finalization rather than the wrapper's SIGTERM path.
    del llm
    return 0


if __name__ == "__main__":
    sys.exit(main())
