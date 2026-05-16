#!/usr/bin/env python3
"""
M0 Coherence Check (VAL-M0-004).

Sends 10 fixed coding prompts at temperature=0, max_tokens=512 to a running
vLLM server (default http://127.0.0.1:8000/v1) and asserts:
- non-empty output
- no repeated 20-token n-gram covering > 50% of the response
- syntax-valid via py_compile / node --check for >= 8/10 prompts.

Writes a JSON report to --out.

Usage:
    python scripts/m0_coherence_check.py \
        --base-url http://127.0.0.1:8000/v1 \
        --model /models/Qwen3.5-9B-w8a8 \
        --prompts /root/bench-int8-w4a16/baseline/prompts_coding.jsonl \
        --out /root/bench-int8-w4a16/baseline/coherence_w8a8_tp1.json
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

# ---- repetition heuristic ----

# Crude tokenizer: split on whitespace; sufficient for n-gram repetition probe.
_TOKEN_SPLIT = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_SPLIT.split(text.strip()) if t]


def max_ngram_coverage(tokens: list[str], n: int = 20) -> float:
    """Return the fraction of tokens covered by the most-frequent n-gram.

    For short outputs (< 2*n tokens), repetition cannot dominate, so return 0.
    """
    if len(tokens) < 2 * n:
        return 0.0
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        ng = tuple(tokens[i : i + n])
        counts[ng] = counts.get(ng, 0) + 1
    if not counts:
        return 0.0
    top_count = max(counts.values())
    if top_count <= 1:
        return 0.0
    # coverage = (count * n) / total_tokens, capped at 1.0
    coverage = min(1.0, (top_count * n) / len(tokens))
    return coverage


# ---- syntax checks ----


_PY_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_JS_BLOCK = re.compile(r"```(?:javascript|js)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str, lang: str) -> str:
    """Extract code from markdown fences. Falls back to whole text if no fence.

    Returns dedented code so a fenced block extracted from a nested markdown
    context (e.g., bullet-list reasoning) compiles standalone.
    """
    import textwrap

    pat = _PY_BLOCK if lang == "python" else _JS_BLOCK
    m = pat.search(text)
    if m:
        return textwrap.dedent(m.group(1))
    return text


def syntax_check(code: str, lang: str) -> tuple[bool, str]:
    """Returns (valid, message). Uses py_compile / node --check."""
    suffix = ".py" if lang == "python" else ".js"
    with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
        f.write(code)
        path = f.name
    try:
        if lang == "python":
            r = subprocess.run(
                [sys.executable, "-m", "py_compile", path],
                capture_output=True, text=True, timeout=10,
            )
        else:
            node = shutil.which("node")
            if node is None:
                # If node isn't installed, fall back to a permissive check:
                # require balanced braces and the keyword "function" or "=>".
                ok = (
                    code.count("{") == code.count("}")
                    and ("function" in code or "=>" in code)
                )
                return ok, "node-not-installed; permissive check"
            r = subprocess.run(
                [node, "--check", path],
                capture_output=True, text=True, timeout=10,
            )
        ok = r.returncode == 0
        msg = r.stderr.strip() if not ok else "ok"
        return ok, msg[:400]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def chat_complete(base_url: str, model: str, prompt: str, max_tokens: int = 512,
                  timeout: int = 120) -> str:
    """Send a single chat completion request; returns the text content."""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", required=True, help="Model id served by vLLM (path)")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--label", default="", help="Label for the run (e.g. w8a8_tp1)")
    args = ap.parse_args()

    with open(args.prompts) as _pf:
        prompts = [json.loads(line) for line in _pf if line.strip()]
    results: list[dict] = []
    syntax_pass = 0
    repetition_fail = 0
    nonempty_fail = 0

    for p in prompts:
        t0 = time.time()
        try:
            text = chat_complete(args.base_url, args.model, p["prompt"],
                                 args.max_tokens)
        except Exception as e:
            results.append({
                "id": p["id"], "lang": p["lang"], "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 2),
            })
            nonempty_fail += 1
            continue
        elapsed = round(time.time() - t0, 2)

        toks = tokenize(text)
        coverage = max_ngram_coverage(toks, n=20)
        repeated = coverage > 0.5
        if repeated:
            repetition_fail += 1

        nonempty = len(text.strip()) > 0
        if not nonempty:
            nonempty_fail += 1

        code = extract_code(text, p["lang"])
        ok, msg = syntax_check(code, p["lang"])
        if ok:
            syntax_pass += 1

        results.append({
            "id": p["id"], "lang": p["lang"],
            "len_chars": len(text),
            "len_tokens_approx": len(toks),
            "ngram20_coverage": round(coverage, 3),
            "repeated_ngram": repeated,
            "nonempty": nonempty,
            "syntax_ok": ok,
            "syntax_msg": msg,
            "elapsed_s": elapsed,
            "preview": text[:300],
        })

    summary = {
        "label": args.label,
        "model": args.model,
        "base_url": args.base_url,
        "n_prompts": len(prompts),
        "syntax_pass": syntax_pass,
        "repetition_fail": repetition_fail,
        "nonempty_fail": nonempty_fail,
        "passes_gate": (
            nonempty_fail == 0
            and repetition_fail == 0
            and syntax_pass >= 8
        ),
        "results": results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[m0_coherence] {args.label} -> syntax={syntax_pass}/{len(prompts)} "
          f"repetition_fail={repetition_fail} nonempty_fail={nonempty_fail} "
          f"gate={'PASS' if summary['passes_gate'] else 'FAIL'}")
    return 0 if summary["passes_gate"] else 1


if __name__ == "__main__":
    sys.exit(main())
