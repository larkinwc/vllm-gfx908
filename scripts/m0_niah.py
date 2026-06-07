#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
M0 Needle-in-a-Haystack at 8k context (VAL-M0-005).

Builds 5 haystack prompts at the requested context length with the needle
inserted at depths 10/30/50/70/90% of the haystack. Asks vLLM via OpenAI
Chat Completions to retrieve the needle; passes if the retrieved magic
string appears verbatim in the response. Required: 5/5.

Usage:
    python scripts/m0_niah.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model /models/Qwen3.5-9B-w8a8 \
        --ctx 8192 \
        --depths 10,30,50,70,90 \
        --out /root/bench-int8-w4a16/baseline/niah_w8a8.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# A repetitive but content-free filler for haystack body.  We use periodic
# narrative so the model has to retrieve a specific value, not just guess.
HAYSTACK_FILLER = (
    "The Pacific Ocean is the largest and deepest of the world's oceans. "
    "It extends from the Arctic Ocean in the north to the Southern Ocean in "
    "the south and is bounded by the continents of Asia and Australia in the "
    "west and the Americas in the east. The equator subdivides it into the "
    "North Pacific Ocean and South Pacific Ocean. Marine life in the Pacific "
    "is diverse, ranging from microscopic plankton to massive blue whales. "
)

# Five distinct needles (each holds a unique magic string).
NEEDLES = [
    {
        "key": "Magic apple count",
        "value": "47823",
        "phrase": "The magic apple count is 47823.",
    },
    {
        "key": "Crimson tower height",
        "value": "9216 meters",
        "phrase": "The crimson tower stands exactly 9216 meters tall.",
    },
    {
        "key": "Lunar passcode",
        "value": "MOON-7741-XQ",
        "phrase": "The lunar passcode is MOON-7741-XQ.",
    },
    {
        "key": "Coral fish species",
        "value": "1582",
        "phrase": ("Exactly 1582 coral fish species were catalogued in the registry."),
    },
    {
        "key": "Captain's birthday",
        "value": "March 22, 1873",
        "phrase": "Captain Aldrin was born on March 22, 1873.",
    },
]


def make_haystack_chars(target_chars: int) -> str:
    n = (target_chars // len(HAYSTACK_FILLER)) + 1
    return (HAYSTACK_FILLER * n)[:target_chars]


def insert_at_depth(haystack: str, needle_phrase: str, depth_pct: int) -> str:
    pos = int(len(haystack) * (depth_pct / 100.0))
    # snap to nearest space to keep readability
    while pos < len(haystack) and haystack[pos] != " " and pos > 0:
        pos -= 1
    return haystack[:pos] + " " + needle_phrase + " " + haystack[pos:]


def chat_complete(
    base_url: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 768,
    timeout: int = 300,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument(
        "--ctx",
        type=int,
        default=8192,
        help="Approximate target context tokens. We size the "
        "haystack body so total prompt fits inside.",
    )
    ap.add_argument("--depths", default="10,30,50,70,90")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    assert len(depths) == len(NEEDLES), "Need exactly 5 depths for 5 needles"

    # Conservative chars-per-token of ~3.5; reserve headroom for system/user/answer
    target_chars = max(2000, int(args.ctx * 3.0))
    haystack = make_haystack_chars(target_chars)

    system = (
        "You are a fact-retrieval assistant. The user gives you a long passage "
        "and asks for a specific fact stated somewhere inside it. Locate the "
        "fact and copy its exact value verbatim from the passage. Output your "
        "answer in this exact format and nothing else: 'ANSWER: <value>'. "
        "Do not write reasoning, analysis, or commentary."
    )

    results = []
    passes = 0
    for needle, depth in zip(NEEDLES, depths):
        t0 = time.time()
        full = insert_at_depth(haystack, needle["phrase"], depth)
        question = (
            f"Passage:\n\n{full}\n\nQuestion: What is the {needle['key']} mentioned "
            f"in the passage? Answer with the exact value as written."
        )
        try:
            ans = chat_complete(
                args.base_url, args.model, system, question, max_tokens=768
            )
        except Exception as e:
            results.append(
                {
                    "needle": needle["key"],
                    "depth_pct": depth,
                    "ok": False,
                    "error": f"{type(e).__name__}: {e}",
                    "elapsed_s": round(time.time() - t0, 2),
                }
            )
            continue
        ok = needle["value"] in ans
        if ok:
            passes += 1
        results.append(
            {
                "needle": needle["key"],
                "depth_pct": depth,
                "expected": needle["value"],
                "answer": ans[:1000],
                "answer_len": len(ans),
                "ok": ok,
                "elapsed_s": round(time.time() - t0, 2),
            }
        )

    summary = {
        "label": args.label,
        "model": args.model,
        "ctx": args.ctx,
        "depths": depths,
        "passes": passes,
        "total": len(NEEDLES),
        "gate_5_of_5": passes == 5,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[m0_niah] {args.label} -> {passes}/{len(NEEDLES)} "
        f"gate={'PASS' if summary['gate_5_of_5'] else 'FAIL'}"
    )
    return 0 if summary["gate_5_of_5"] else 1


if __name__ == "__main__":
    sys.exit(main())
