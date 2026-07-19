#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Teacher-forced decode perplexity for testing KV-cache numerical quality.

Unlike OpenAI prompt-logprob scoring, this drives a real decode step for every
reference token. A custom logits processor records the unmodified probability
of the reference token, then forces that token as the next output. Subsequent
steps must attend through the server's stored KV cache, so auto and compressed
KV modes can be compared on identical token sequences.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from scripts.m0_perplexity import build_chunks, load_wikitext_token_ids
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
)
from vllm.v1.sample.logits_processor.builtin import process_dict_updates


@dataclass
class _ForcedRequest:
    target_tokens: list[int]
    score_path: str
    index: int = 0


class TeacherForcedLogitsProcessor(LogitsProcessor):
    """Record raw target logprobs before forcing deterministic decode tokens."""

    def __init__(
        self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool
    ) -> None:
        self.requests: dict[int, _ForcedRequest] = {}

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        extra_args = params.extra_args or {}
        targets = extra_args.get("teacher_forced_targets")
        score_path = extra_args.get("teacher_forced_score_path")
        if not isinstance(targets, list) or not all(
            isinstance(token, int) for token in targets
        ):
            raise ValueError("teacher_forced_targets must be a list of token ids")
        if not isinstance(score_path, str):
            raise ValueError("teacher_forced_score_path must be a path string")

    def is_argmax_invariant(self) -> bool:
        return False

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        def request_state(params: SamplingParams) -> _ForcedRequest | None:
            self.validate_params(params)
            extra_args = params.extra_args or {}
            return _ForcedRequest(
                target_tokens=list(extra_args["teacher_forced_targets"]),
                score_path=extra_args["teacher_forced_score_path"],
            )

        process_dict_updates(
            self.requests,
            batch_update,
            lambda params, _, __: request_state(params),
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        for row, request in self.requests.items():
            if request.index >= len(request.target_tokens):
                continue
            target_token = request.target_tokens[request.index]
            raw_logits = logits[row]
            target_logprob = (
                raw_logits[target_token] - torch.logsumexp(raw_logits, dim=0)
            ).item()
            if get_tensor_model_parallel_rank() == 0:
                with Path(request.score_path).open("a") as score_file:
                    score_file.write(f"{target_logprob:.17g}\n")
            raw_logits.fill_(float("-inf"))
            raw_logits[target_token] = 0.0
            request.index += 1
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--wikitext-parquet", type=Path, required=True)
    parser.add_argument(
        "--kv-cache-dtype",
        choices=["auto", "turboquant_k8v4"],
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunks", type=int, default=50)
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--prefix-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=4352)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.prefix_tokens < args.chunk_tokens:
        raise ValueError("prefix-tokens must be positive and shorter than chunk-tokens")

    token_ids, dataset = load_wikitext_token_ids(
        args.tokenizer, args.wikitext_parquet
    )
    chunks = build_chunks(token_ids, args.chunks, args.chunk_tokens, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    score_path = args.output.with_suffix(".logprobs.txt")
    score_path.unlink(missing_ok=True)

    llm = LLM(
        model=args.model,
        revision=args.revision,
        dtype="float16",
        tensor_parallel_size=args.tensor_parallel_size,
        kv_cache_dtype=args.kv_cache_dtype,
        language_model_only=True,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        logits_processors=[TeacherForcedLogitsProcessor],
    )

    per_chunk: list[dict[str, Any]] = []
    expected_tokens = 0
    for chunk_index, chunk in enumerate(chunks):
        prompt = chunk[: args.prefix_tokens]
        continuation = chunk[args.prefix_tokens :]
        params = SamplingParams(
            temperature=0.0,
            max_tokens=len(continuation),
            ignore_eos=True,
            detokenize=False,
            extra_args={
                "teacher_forced_targets": continuation,
                "teacher_forced_score_path": str(score_path),
            },
        )
        outputs = llm.generate(
            [TokensPrompt(prompt_token_ids=prompt)], params, use_tqdm=False
        )
        generated = outputs[0].outputs[0].token_ids
        if generated != continuation:
            raise RuntimeError(
                f"chunk {chunk_index}: forced output did not match continuation"
            )
        expected_tokens += len(continuation)
        per_chunk.append({"chunk": chunk_index, "tokens": len(continuation)})
        print(f"[cache-read-ppl] chunk {chunk_index + 1}/{len(chunks)}", flush=True)

    logprobs = [
        float(line)
        for line in score_path.read_text().splitlines()
        if line.strip()
    ]
    if len(logprobs) != expected_tokens:
        raise RuntimeError(
            f"recorded {len(logprobs)} logprobs; expected {expected_tokens}"
        )
    if any(not math.isfinite(logprob) for logprob in logprobs):
        raise RuntimeError("non-finite forced-token logprob")

    mean_nll = -sum(logprobs) / len(logprobs)
    result = {
        "model": args.model,
        "revision": args.revision,
        "kv_cache_dtype": args.kv_cache_dtype,
        "chunks": args.chunks,
        "chunk_tokens": args.chunk_tokens,
        "prefix_tokens": args.prefix_tokens,
        "seed": args.seed,
        "n_tokens_scored": len(logprobs),
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "dataset": dataset,
        "logprob_artifact": str(score_path),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"[cache-read-ppl] {args.kv_cache_dtype} "
        f"ppl={result['perplexity']:.6f} tokens={len(logprobs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
