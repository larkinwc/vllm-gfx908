#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""VAL-FINAL-004 — needle-in-haystack at 32k context.

Plants 5 needles at fixed depths (10/30/50/70/90 %) inside a 32k-token
haystack built from a deterministic filler sentence, asks the model to
retrieve each needle verbatim, and grades pass = needle string is a
substring of the response (case-insensitive).

Pass criteria: 5/5 needles retrieved.

This is the M0 NIAH prompt-shape (see scripts/m0_niah.py) extended to
ctx=32768. The system-message guardrail keeps the model from emitting
"Thinking Process:" preamble that otherwise eats the token budget at
large contexts.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_OUT = Path("/root/bench-int8-w4a16/final/m6_needle32k.json")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"

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
        "phrase": "Exactly 1582 coral fish species were catalogued in the registry.",
    },
    {
        "key": "Captain's birthday",
        "value": "March 22, 1873",
        "phrase": "Captain Aldrin was born on March 22, 1873.",
    },
]

DEPTHS = [10, 30, 50, 70, 90]

FILLER = (
    "The Pacific Ocean is the largest and deepest of the world's oceans. "
    "It extends from the Arctic Ocean in the north to the Southern Ocean "
    "in the south and is bounded by the continents of Asia and Australia "
    "in the west and the Americas in the east. The equator subdivides it "
    "into the North Pacific Ocean and South Pacific Ocean. Marine life in "
    "the Pacific is diverse, ranging from microscopic plankton to massive "
    "blue whales. "
)

SYSTEM = (
    "You are a fact-retrieval assistant. The user gives you a long passage "
    "and asks for a specific fact stated somewhere inside it. Locate the "
    "fact and copy its exact value verbatim from the passage. Output your "
    "answer in this exact format and nothing else: 'ANSWER: <value>'. "
    "Do not write reasoning, analysis, or commentary."
)


def make_haystack_chars(target_chars: int) -> str:
    n = (target_chars // len(FILLER)) + 1
    return (FILLER * n)[:target_chars]


def insert_at_depth(haystack: str, needle_phrase: str, depth_pct: int) -> str:
    pos = int(len(haystack) * (depth_pct / 100.0))
    while pos < len(haystack) and haystack[pos] != " " and pos > 0:
        pos -= 1
    return haystack[:pos] + " " + needle_phrase + " " + haystack[pos:]


def chat(
    base_url: str,
    system: str,
    user: str,
    model: str,
    max_tokens: int = 768,
    timeout: float = 900.0,
) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
    }).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        blob = json.loads(resp.read())
    return blob["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--probes", type=int, default=5)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--model", type=str, default=None)
    ap.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = args.model
    if model is None:
        models_url = f"{args.base_url.rstrip('/')}/v1/models"
        with urllib.request.urlopen(models_url, timeout=5.0) as resp:
            blob = json.loads(resp.read())
            model = blob["data"][0]["id"]

    needle_count = min(args.probes, len(NEEDLES))
    # ~3.0 chars/token on average — gives a comfortable margin for instructions+answer.
    target_chars = max(2000, int(args.ctx * 3.0))
    haystack = make_haystack_chars(target_chars)

    results = []
    passes = 0
    t0 = time.time()
    for i in range(needle_count):
        needle = NEEDLES[i]
        depth = DEPTHS[i]
        full = insert_at_depth(haystack, needle["phrase"], depth)
        question = (
            f"Passage:\n\n{full}\n\nQuestion: What is the {needle['key']} mentioned "
            f"in the passage? Answer with the exact value as written."
        )
        try:
            response = chat(args.base_url, SYSTEM, question, model)
        except Exception as exc:
            results.append({
                "label": needle["key"],
                "depth_pct": depth,
                "expected": needle["value"],
                "ok": False,
                "error": str(exc),
                "answer": "",
            })
            print(f"  [{i+1}/{needle_count}] {needle['key']} depth={depth}% — FAIL ({exc})")  # noqa: E501
            continue
        ok = needle["value"].lower() in response.lower()
        results.append({
            "label": needle["key"],
            "depth_pct": depth,
            "expected": needle["value"],
            "answer": response,
            "ok": ok,
        })
        passes += int(ok)
        print(f"  [{i+1}/{needle_count}] {needle['key']} depth={depth}% — {'PASS' if ok else 'FAIL'}")  # noqa: E501
    elapsed = time.time() - t0

    args.out.write_text(json.dumps({
        "model": model,
        "ctx": args.ctx,
        "depths": DEPTHS[:needle_count],
        "passes": passes,
        "total": needle_count,
        "gate_5_of_5": passes == needle_count,
        "elapsed_s": elapsed,
        "results": results,
    }, indent=2))

    print(f"\nNeedle@{args.ctx}: {passes}/{needle_count} (gate {'PASS' if passes == needle_count else 'FAIL'})")  # noqa: E501
    print(f"Wrote {args.out}")
    return 0 if passes == needle_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
