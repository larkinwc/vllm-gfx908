#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
M0 Determinism / no-NaN/Inf check (VAL-M0-007).

For a fixed prompt, asks the running vLLM server twice at temperature=0,
seed=0, returning top-5 logprobs. Asserts:
- top-1 token identical across both runs
- top-5 logprobs vector identical (same tokens + same values)
- no NaN / Inf in any logprob

Usage:
    python scripts/m0_determinism.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model /models/Qwen3.5-9B-w8a8 \
        --out /root/bench-int8-w4a16/baseline/determinism_w8a8.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests

PROMPT = "def fibonacci(n):"


def completion(base_url: str, model: str, prompt: str) -> dict:
    url = base_url.rstrip("/") + "/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.0,
        "max_tokens": 1,
        "logprobs": 5,
        "seed": 0,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    j = r.json()
    choice = j["choices"][0]
    lp = choice.get("logprobs") or {}
    tokens = lp.get("tokens", [])
    top_logprobs = lp.get("top_logprobs", [])  # list per generated token
    return {
        "text": choice.get("text", ""),
        "top1": tokens[0] if tokens else None,
        "top_logprobs_0": top_logprobs[0] if top_logprobs else {},
    }


def has_nan_or_inf(d: dict) -> bool:
    for v in d.values():
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(x) or math.isinf(x):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    runs = []
    for i in range(2):
        t0 = time.time()
        r = completion(args.base_url, args.model, PROMPT)
        r["elapsed_s"] = round(time.time() - t0, 2)
        runs.append(r)

    top1_match = runs[0]["top1"] == runs[1]["top1"]
    lp0 = runs[0]["top_logprobs_0"]
    lp1 = runs[1]["top_logprobs_0"]
    lp_keys_match = sorted(lp0.keys()) == sorted(lp1.keys())
    lp_values_match = (
        all(abs(float(lp0[k]) - float(lp1[k])) < 1e-6 for k in lp0 if k in lp1)
        if lp_keys_match
        else False
    )

    nan_inf = has_nan_or_inf(lp0) or has_nan_or_inf(lp1)

    passes = top1_match and lp_keys_match and lp_values_match and not nan_inf

    summary = {
        "label": args.label,
        "model": args.model,
        "prompt": PROMPT,
        "top1_match": top1_match,
        "logprob_keys_match": lp_keys_match,
        "logprob_values_match": lp_values_match,
        "nan_or_inf": nan_inf,
        "passes": passes,
        "runs": runs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[m0_determinism] {args.label} top1_match={top1_match} "
        f"logprobs_match={lp_keys_match and lp_values_match} "
        f"nan_inf={nan_inf} gate={'PASS' if passes else 'FAIL'}"
    )
    return 0 if passes else 1


if __name__ == "__main__":
    sys.exit(main())
