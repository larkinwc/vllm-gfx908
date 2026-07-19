#!/usr/bin/env python3
"""
M0 Perplexity (VAL-M0-006) — server-mode.

Computes WikiText-2 perplexity over `--chunks` fixed chunks of `--chunk-tokens`
tokens (deterministic offset using `--seed`) through a running vLLM
OpenAI-compatible `/v1/completions` endpoint with `echo=true` and prompt
logprobs. vLLM intentionally skips prefix-cache reads for prompt-logprob
requests, so this measures prefill numerical stability only; it is not a
quantized-KV read-quality gate.

This avoids spinning up three separate vLLM engines from the same script —
the caller is expected to have started the server pointing at the model
under test.

Usage:
    # Start server pointing at the target model first, then:
    python scripts/m0_perplexity.py \
        --model /models/Qwen3.5-9B-w8a8 \
        --tokenizer /models/Qwen3.5-9B-w8a8 \
        --base-url http://127.0.0.1:8000/v1 \
        --chunks 50 --chunk-tokens 512 --seed 0 \
        --out /root/bench-int8-w4a16/baseline/ppl_w8a8.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import requests  # noqa: E402


def load_wikitext_token_ids(
    tokenizer_path: str, wikitext_parquet: Path | None
) -> tuple[list[int], dict[str, str]]:
    # Bypass AutoTokenizer (which fails on model_type=qwen3_5 in transformers v4)
    # and load the fast tokenizer directly from tokenizer.json.
    from transformers import PreTrainedTokenizerFast

    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(Path(tokenizer_path) / "tokenizer.json"),
    )
    if wikitext_parquet is None:
        from datasets import load_dataset

        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        texts = [text for text in ds["text"] if text and text.strip()]
        source = {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "source": "datasets",
        }
    else:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "--wikitext-parquet requires the pyarrow dependency"
            ) from exc
        if not wikitext_parquet.is_file():
            raise FileNotFoundError(wikitext_parquet)
        texts = [
            text
            for text in pq.read_table(wikitext_parquet, columns=["text"])
            .column("text")
            .to_pylist()
            if text and text.strip()
        ]
        with wikitext_parquet.open("rb") as parquet_file:
            parquet_sha256 = hashlib.file_digest(
                parquet_file, "sha256"
            ).hexdigest()
        source = {
            "dataset": "Salesforce/wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "test",
            "source": str(wikitext_parquet),
            "sha256": parquet_sha256,
        }
    blob = "\n\n".join(texts)
    return tok(blob, add_special_tokens=False)["input_ids"], source


def build_chunks(token_ids: list[int], num_chunks: int, chunk_tokens: int,
                 seed: int) -> list[list[int]]:
    start = (seed * 17) % max(1, chunk_tokens)
    chunks = []
    pos = start
    while len(chunks) < num_chunks and pos + chunk_tokens <= len(token_ids):
        chunks.append(token_ids[pos:pos + chunk_tokens])
        pos += chunk_tokens
    if len(chunks) < num_chunks:
        raise RuntimeError(
            f"Wikitext-2 has only {len(token_ids)} tokens; need >= "
            f"{num_chunks * chunk_tokens + start}"
        )
    return chunks


def score_chunk(
    base_url: str, model: str, ids: list[int], timeout: int = 600
) -> tuple[float, int]:
    """Return NLL and count for prompt tokens in one prefill request."""
    url = base_url.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": ids,
        "max_tokens": 1,
        "echo": True,
        "logprobs": 0,
        "temperature": 0.0,
        "seed": 0,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    choice = r.json()["choices"][0]
    lp_obj = choice.get("logprobs") or {}
    token_logprobs = lp_obj.get("token_logprobs") or []
    # The first position has no logprob. The final logprob belongs to the
    # generated token, so only retain prompt-position logprobs.
    prompt_lp = token_logprobs[: len(ids)]
    sum_nll = 0.0
    n = 0
    for lp in prompt_lp:
        if lp is None:
            continue
        if math.isnan(lp) or math.isinf(lp):
            raise RuntimeError("NaN/Inf in token_logprobs")
        sum_nll += -lp
        n += 1
    return sum_nll, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True,
                    help="Model id served by vLLM (must match /v1/models)")
    ap.add_argument("--tokenizer", required=True,
                    help="Path to tokenizer (usually same as --model)")
    ap.add_argument("--chunks", type=int, default=50)
    ap.add_argument("--chunk-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--wikitext-parquet",
        type=Path,
        help="Pinned WikiText-2 raw test Parquet shard, bypassing datasets loading.",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    ids, dataset = load_wikitext_token_ids(
        args.tokenizer, args.wikitext_parquet
    )
    chunks = build_chunks(ids, args.chunks, args.chunk_tokens, args.seed)
    print(f"[m0_perplexity] tokenized blob: {len(ids)} tokens; "
          f"using {len(chunks)} chunks of {args.chunk_tokens} tokens")

    t0 = time.time()
    sum_nll = 0.0
    total_n = 0
    per_chunk = []
    for ci, ids_chunk in enumerate(chunks):
        s, n = score_chunk(args.base_url, args.model, ids_chunk)
        sum_nll += s
        total_n += n
        per_chunk.append({
            "chunk": ci, "tokens": n,
            "mean_nll": s / n if n else float("nan"),
        })
        if ci % 10 == 0:
            elapsed = time.time() - t0
            print(f"  chunk {ci+1}/{len(chunks)} tokens={n} elapsed={elapsed:.1f}s")

    elapsed = round(time.time() - t0, 1)
    mean_nll = sum_nll / max(1, total_n)
    ppl = math.exp(mean_nll)

    summary = {
        "label": args.label,
        "model": args.model,
        "chunks": args.chunks,
        "chunk_tokens": args.chunk_tokens,
        "seed": args.seed,
        "n_tokens_scored": total_n,
        "mean_nll": mean_nll,
        "perplexity": ppl,
        "dataset": dataset,
        "elapsed_s": elapsed,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[m0_perplexity] {args.label} ppl={ppl:.4f} "
          f"({total_n} tokens scored in {elapsed}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
