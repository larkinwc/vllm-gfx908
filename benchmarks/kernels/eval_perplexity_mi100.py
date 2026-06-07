# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Evaluate perplexity of Qwen3.5-9B variants on wikitext-2.

Computes per-token cross-entropy loss (perplexity = exp(loss))
using vLLM for fast batched inference.

Usage:
    /opt/vllm-env/bin/python3 /tmp/eval_perplexity.py --model /models/Qwen3.5-9B
"""

import argparse
import json
import math
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def compute_perplexity_vllm(
    model_path,
    max_samples=100,
    max_len=2048,
    tp_size=4,
    dtype="float16",
    kv_cache_dtype="auto",
):
    """Compute perplexity using vLLM's offline LLM interface."""
    from vllm import LLM, SamplingParams

    print(f"\n{'=' * 60}")
    print(f"Model: {model_path}")
    print(f"{'=' * 60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Load wikitext-2 test set
    print("Loading wikitext-2-raw-v1 test set...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Concatenate all text into one string (standard perplexity eval)
    full_text = "\n\n".join([t for t in ds["text"] if t.strip()])

    # Tokenize
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False)
    input_ids = encodings.input_ids[0]
    total_tokens = len(input_ids)
    print(f"Total tokens in wikitext-2 test: {total_tokens}")

    # Create sliding window chunks with stride = max_len // 2
    stride = max_len // 2
    chunks = []
    for begin in range(0, total_tokens - 1, stride):
        end = min(begin + max_len, total_tokens)
        chunk_ids = input_ids[begin:end].tolist()
        if len(chunk_ids) < 10:
            continue
        chunks.append((begin, chunk_ids))
        if len(chunks) >= max_samples:
            break

    print(
        f"Created {len(chunks)} evaluation chunks (max_len={max_len}, stride={stride})"
    )

    # Initialize vLLM
    print("Loading model with vLLM...")
    t0 = time.time()
    llm = LLM(
        model=model_path,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        trust_remote_code=True,
        tensor_parallel_size=tp_size,
        max_model_len=max_len + 16,
        enforce_eager=True,
        language_model_only=True,
        gpu_memory_utilization=0.90,
        max_num_seqs=1,
        max_num_batched_tokens=max_len + 16,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Use prompt_logprobs to get per-token log-probabilities
    sampling_params = SamplingParams(
        max_tokens=1,
        temperature=0,
        prompt_logprobs=0,  # return logprobs for prompt tokens
    )

    # Prepare prompts (as token ID lists)
    prompts = [{"prompt_token_ids": chunk_ids} for _, chunk_ids in chunks]

    print(f"Running inference on {len(prompts)} chunks...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    infer_time = time.time() - t0
    print(f"Inference completed in {infer_time:.1f}s")

    # Compute average negative log-likelihood
    total_nll = 0.0
    total_count = 0

    for i, output in enumerate(outputs):
        if output.prompt_logprobs is None:
            continue
        # prompt_logprobs[0] is None (no logprob for first token)
        for j, token_logprobs in enumerate(output.prompt_logprobs):
            if token_logprobs is None:
                continue
            # Get the logprob for the actual token
            token_id = chunks[i][1][j]
            if token_id in token_logprobs:
                logprob = token_logprobs[token_id].logprob
                total_nll -= logprob
                total_count += 1

    if total_count == 0:
        print("ERROR: No logprobs collected!")
        return float("inf")

    avg_nll = total_nll / total_count
    perplexity = math.exp(avg_nll)

    print(f"\nResults for {model_path}:")
    print(f"  Tokens evaluated: {total_count}")
    print(f"  Avg NLL: {avg_nll:.4f}")
    print(f"  Perplexity: {perplexity:.2f}")
    print(f"  Load time: {load_time:.1f}s")
    print(f"  Inference time: {infer_time:.1f}s")

    # Cleanup
    del llm
    torch.accelerator.empty_cache()
    import gc

    gc.collect()

    return {
        "model": model_path,
        "perplexity": perplexity,
        "avg_nll": avg_nll,
        "tokens_evaluated": total_count,
        "load_time_s": load_time,
        "inference_time_s": infer_time,
    }


def main():
    parser = argparse.ArgumentParser(description="Perplexity evaluation")
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=[
            "/models/Qwen3.5-9B",
            "/models/Qwen3.5-9B-W8A8",
            "/models/Qwen3.5-9B-AWQ-INT4",
        ],
        help="Model paths to evaluate",
    )
    parser.add_argument(
        "--max-samples", type=int, default=50, help="Max number of text chunks"
    )
    parser.add_argument(
        "--max-len", type=int, default=2048, help="Max sequence length per chunk"
    )
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument(
        "--kv-cache-dtype",
        type=str,
        default="auto",
        help="KV cache dtype, e.g. auto, turboquant_k3v4_nc, turboquant_planar3_nc",
    )
    parser.add_argument("--output", type=str, default=None, help="JSON output file")
    args = parser.parse_args()

    results = []
    for model_path in args.model:
        if not os.path.exists(model_path):
            print(f"SKIP: {model_path} does not exist")
            continue
        try:
            result = compute_perplexity_vllm(
                model_path,
                max_samples=args.max_samples,
                max_len=args.max_len,
                tp_size=args.tp_size,
                dtype=args.dtype,
                kv_cache_dtype=args.kv_cache_dtype,
            )
            result["kv_cache_dtype"] = args.kv_cache_dtype
            results.append(result)
        except Exception as e:
            print(f"ERROR evaluating {model_path}: {e}")
            import traceback

            traceback.print_exc()
            results.append({"model": model_path, "error": str(e)})

    # Print comparison table
    print(f"\n{'=' * 70}")
    print("PERPLEXITY COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':<40} {'PPL':>10} {'NLL':>10} {'Size':>8}")
    print(f"{'-' * 70}")
    for r in results:
        model_name = os.path.basename(r["model"])
        if "error" in r:
            print(f"{model_name:<40} {'ERROR':>10}")
        else:
            # Get model size
            model_dir = r["model"]
            size_gb = (
                sum(
                    os.path.getsize(os.path.join(model_dir, f))
                    for f in os.listdir(model_dir)
                    if f.endswith((".safetensors", ".bin"))
                )
                / 1024**3
            )
            print(
                f"{model_name:<40} {r['perplexity']:>10.2f} "
                f"{r['avg_nll']:>10.4f} {size_gb:>7.1f}G"
            )
    print(f"{'=' * 70}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
