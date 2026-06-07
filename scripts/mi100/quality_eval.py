#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W4A16 quality reference: WikiText-2 perplexity + needle-in-haystack.

RUN ON THE gfx908 HOST BY THE ORCHESTRATOR, against an already-running vLLM
OpenAI server (legacy W4A16, marlin OFF). This script does NOT start a server.

Perplexity uses the completions endpoint with ``echo=true, max_tokens=0,
logprobs=1`` to read teacher-forced token logprobs over WikiText-2-raw test
text. Needle-in-haystack inserts a unique passcode at N evenly-spaced depths in
a long filler context and checks recall.

Interpreter: /opt/vllm-env/bin/python3 (no uv/.venv). Stdlib + optional
`datasets` (for loading wikitext from the local HF cache).

Example:
  /opt/vllm-env/bin/python3 scripts/mi100/quality_eval.py \
    --base-url http://127.0.0.1:8000 --model /models/Qwen3.5-9B-w4a16 \
    --out /root/bench-w4a16/m0/quality/quality_ppl_needle.json \
    --ppl-tokens 8192 --max-len 2048 --needle-depths 5 --needle-max-ctx 4096 \
    --out-pretty
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request


def _post(url: str, payload: dict, timeout: float = 90.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_wikitext(path: str | None) -> str:
    """Return concatenated non-empty WikiText-2-raw test lines."""
    if path:
        with open(path, encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return "\n".join(lines)
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    lines = [t.strip() for t in ds["text"] if t and t.strip()]
    return "\n".join(lines)


def measure_perplexity(
    base_url: str, model: str, text: str, ppl_tokens: int, max_len_chars: int
) -> dict:
    """Teacher-forced perplexity via a SINGLE batched echo+logprobs request.

    ``max_len_chars`` windows the raw text by characters (an approximation of
    token windows). All windows are sent as ONE completions request with
    ``prompt=[w0, w1, ...]`` — this is required because the vLLM build wedges
    on the *second* consecutive ``max_tokens=0`` echo request, but handles a
    batched list in a single engine pass fine. The first token_logprob in each
    choice is null and is skipped.
    """
    url = base_url.rstrip("/") + "/v1/completions"
    n = len(text)
    # ~4 chars/token rough estimate to size the window count to the budget.
    approx_tok_per_window = max(1, max_len_chars // 4)
    n_windows = max(1, math.ceil(ppl_tokens / approx_tok_per_window))
    windows = []
    pos = 0
    while pos < n and len(windows) < n_windows:
        w = text[pos : pos + max_len_chars]
        pos += max_len_chars
        if w.strip():
            windows.append(w)
    if not windows:
        return {
            "perplexity": None,
            "mean_nll": None,
            "scored_tokens": 0,
            "windows": 0,
            "n_failed_windows": 0,
            "error": "no text",
        }
    try:
        t0 = time.time()
        r = _post(
            url,
            {
                "model": model,
                "prompt": windows,
                "max_tokens": 0,
                "echo": True,
                "logprobs": 1,
                "temperature": 0,
            },
            timeout=600.0,
        )
        total_nll = 0.0
        scored = 0
        for ch in r["choices"]:
            vals = [x for x in ch["logprobs"]["token_logprobs"] if x is not None]
            total_nll += -sum(vals)
            scored += len(vals)
        print(
            f"[ppl] {len(windows)} windows, scored {scored} tok, "
            f"{time.time() - t0:.1f}s",
            flush=True,
        )
    except Exception as e:  # noqa: BLE001
        return {
            "perplexity": None,
            "mean_nll": None,
            "scored_tokens": 0,
            "windows": len(windows),
            "n_failed_windows": len(windows),
            "error": f"batched echo request failed: {e}",
        }
    if scored == 0:
        return {
            "perplexity": None,
            "mean_nll": None,
            "scored_tokens": 0,
            "windows": len(windows),
            "n_failed_windows": len(windows),
            "error": "no tokens scored (echo+logprobs unsupported?)",
        }
    mean_nll = total_nll / scored
    return {
        "perplexity": math.exp(mean_nll),
        "mean_nll": mean_nll,
        "scored_tokens": scored,
        "windows": len(windows),
        "n_failed_windows": 0,
    }


def measure_needle(base_url: str, model: str, depths: int, max_ctx_chars: int) -> dict:
    """Insert a unique passcode at N evenly-spaced depths; check recall."""
    url = base_url.rstrip("/") + "/v1/completions"
    secret = "BLUE-WHALE-42"
    needle = f" The secret passcode is {secret}. "
    filler = (
        "The garden was quiet in the early morning light. "
        "Birds moved between the old oak branches. "
    )
    base = (filler * ((max_ctx_chars // len(filler)) + 1))[:max_ctx_chars]
    results = []
    passed = 0
    # One separate generation request per depth. (Generation requests are safe
    # to issue consecutively — unlike echo+logprobs — but a batched *list*
    # generation request wedges this vLLM build, so we do NOT batch here.)
    for i in range(depths):
        frac = 0.0 if depths == 1 else i / (depths - 1)
        cut = int(len(base) * frac)
        hay = base[:cut] + needle + base[cut:]
        prompt = hay + "\n\nQuestion: What is the secret passcode?\nAnswer:"
        ok = False
        try:
            r = _post(
                url,
                {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 24,
                    "temperature": 0,
                },
                timeout=120.0,
            )
            ok = secret.lower() in r["choices"][0].get("text", "").lower()
        except Exception as e:  # noqa: BLE001
            print(f"[needle] depth {frac:.2f} failed: {e}", flush=True)
        results.append({"depth": round(frac, 4), "ok": ok})
        passed += int(ok)
        print(f"[needle] depth {frac:.2f} ok={ok}", flush=True)
    return {"passed": passed, "total": depths, "depths": results}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--wikitext",
        default=None,
        help="optional path; default loads HF wikitext-2-raw test",
    )
    ap.add_argument("--ppl-tokens", type=int, default=8192)
    ap.add_argument(
        "--max-len",
        type=int,
        default=2048,
        help="per-window length (chars approx tokens)",
    )
    ap.add_argument("--needle-depths", type=int, default=5)
    ap.add_argument("--needle-max-ctx", type=int, default=4096)
    ap.add_argument("--out-pretty", action="store_true")
    args = ap.parse_args()

    # Needle (normal generation) runs FIRST; perplexity (echo+logprobs) runs
    # LAST because this vLLM build can wedge the engine on a request that
    # *follows* an echo+logprobs request. Each metric issues a single batched
    # request (see measure_* docstrings).
    needle = measure_needle(
        args.base_url, args.model, args.needle_depths, args.needle_max_ctx
    )
    print(f"[quality] needle={needle['passed']}/{needle['total']}")
    text = _load_wikitext(args.wikitext)
    print(f"[quality] wikitext chars={len(text)}")
    ppl = measure_perplexity(
        args.base_url, args.model, text, args.ppl_tokens, args.max_len
    )
    print(
        f"[quality] perplexity={ppl.get('perplexity')} "
        f"scored={ppl.get('scored_tokens')}"
    )

    out = {
        **ppl,
        "needle": needle,
        "model": args.model,
        "base_url": args.base_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1 if args.out_pretty else None)
    print(f"[quality] wrote {args.out}")


if __name__ == "__main__":
    main()
