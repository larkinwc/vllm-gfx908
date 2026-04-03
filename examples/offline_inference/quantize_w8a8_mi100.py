# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SmoothQuant W8A8 quantization recipe for MI100 (gfx908).

This script quantizes a model to W8A8 INT8 using SmoothQuant + GPTQ
from llm-compressor, producing a model that leverages MI100's INT8 MFMA
instructions (185 TOPS vs 46 TFLOPS FP16).

Prerequisites:
    pip install llmcompressor transformers datasets

Usage:
    # Quantize Qwen3.5-9B for MI100
    python examples/offline_inference/quantize_w8a8_mi100.py \
        --model Qwen/Qwen3.5-9B \
        --output ./Qwen3.5-9B-W8A8

    # With custom calibration samples
    python examples/offline_inference/quantize_w8a8_mi100.py \
        --model Qwen/Qwen3.5-9B \
        --output ./Qwen3.5-9B-W8A8 \
        --num-samples 1024 \
        --max-seq-len 4096

    # Run the quantized model in vLLM on MI100
    python -m vllm.entrypoints.openai.api_server \
        --model ./Qwen3.5-9B-W8A8 \
        --dtype float16 \
        --tensor-parallel-size 4

Expected benefits on MI100:
    - 50% memory reduction (18GB -> 9GB for 9B model)
    - 30-50% prefill throughput gain (compute-bound, INT8 MFMA)
    - 10-20% decode throughput gain (memory-bandwidth-bound, smaller weights)
    - Enables 70B INT8 models on 4x32GB MI100
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="SmoothQuant W8A8 quantization for MI100"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="HuggingFace model ID or local path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output directory for quantized model"
    )
    parser.add_argument(
        "--num-samples", type=int, default=512, help="Number of calibration samples"
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length for calibration",
    )
    parser.add_argument(
        "--smoothing-strength",
        type=float,
        default=0.8,
        help="SmoothQuant migration strength (0.0-1.0)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceH4/ultrachat_200k",
        help="Calibration dataset",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        help="HuggingFace repo ID to push quantized model",
    )
    args = parser.parse_args()

    try:
        from llmcompressor import oneshot
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
    except ImportError:
        print("ERROR: llmcompressor not installed. Run:")
        print("  pip install llmcompressor")
        sys.exit(1)

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading calibration data: {args.dataset} ({args.num_samples} samples)")
    ds = load_dataset(args.dataset, split="train_sft")
    ds = ds.shuffle(seed=42).select(range(args.num_samples))

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(example["messages"], tokenize=False)
        }

    ds = ds.map(preprocess)

    def tokenize(sample):
        return tokenizer(
            sample["text"],
            padding=False,
            max_length=args.max_seq_len,
            truncation=True,
            add_special_tokens=False,
        )

    ds = ds.map(tokenize, remove_columns=ds.column_names)

    print(
        f"Applying SmoothQuant W8A8 quantization (strength={args.smoothing_strength})"
    )
    recipe = [
        SmoothQuantModifier(smoothing_strength=args.smoothing_strength),
        GPTQModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=["lm_head"],
        ),
    ]

    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_seq_len,
        num_calibration_samples=args.num_samples,
    )

    print(f"Saving quantized model to: {args.output}")
    model.save_pretrained(args.output, save_compressed=True)
    tokenizer.save_pretrained(args.output)

    if args.push_to_hub:
        print(f"Pushing to HuggingFace Hub: {args.push_to_hub}")
        model.push_to_hub(args.push_to_hub)
        tokenizer.push_to_hub(args.push_to_hub)

    print("\nDone! Run the quantized model with:")
    print("  python -m vllm.entrypoints.openai.api_server \\")
    print(f"    --model {args.output} \\")
    print("    --dtype float16 \\")
    print("    --tensor-parallel-size 4")

    param_count = sum(p.numel() for p in model.parameters())
    fp16_size_gb = param_count * 2 / (1024**3)
    int8_size_gb = param_count * 1 / (1024**3)
    print("\nModel stats:")
    print(f"  Parameters: {param_count / 1e9:.1f}B")
    print(f"  FP16 size:  {fp16_size_gb:.1f} GB")
    print(f"  INT8 size:  {int8_size_gb:.1f} GB")
    print(f"  Savings:    {(1 - int8_size_gb / fp16_size_gb) * 100:.0f}%")


if __name__ == "__main__":
    main()
