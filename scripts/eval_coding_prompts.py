#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""VAL-FINAL-003 — coding-agent qualitative pass on 10 prompts.

Queries the running vLLM OpenAI-compatible server at :8000 for each
prompt in ``tests/eval/coding_prompts.json``, extracts the first fenced
code block, then grades the rubric (compiles / runs / correct).

Pass criteria: ≥ 9/10 prompts must pass *all three* rubric columns.

Outputs:
  - <output>.md : human-readable grade table
  - <output>.json : per-prompt JSON (prompt, response, rubric, errors)

The script never starts a server; it expects one of the M6 vLLM services
(see services.yaml) to already be up. Failure mode if no server: a clear
"connection refused" message and non-zero exit.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path("/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4")  # noqa: E501
DEFAULT_PROMPTS = REPO / "tests" / "eval" / "coding_prompts.json"
DEFAULT_OUT_DIR = Path("/root/bench-int8-w4a16/final")
SERVER = "http://127.0.0.1:8000"


def chat(prompt: str, model: str, max_tokens: int = 2048, timeout: float = 600.0) -> str:  # noqa: E501
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "seed": 0,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        blob = json.loads(resp.read())
    return blob["choices"][0]["message"]["content"]


FENCE_RE = re.compile(r"```(?:python|javascript|js|bash|sh)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str, lang: str) -> str | None:
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Heuristic fallback: extract from the first def/function/script-shebang
    # to the end. Many small models omit the fence on terse prompts.
    if lang == "python":
        for kw in ("def ", "import ", "class "):
            i = text.find(kw)
            if i >= 0:
                snippet = text[i:].strip()
                # Trim trailing prose (lines starting with English-y prose
                # are unlikely after the function — keep as-is for py_compile).
                return snippet
    if lang == "javascript":
        i = text.find("function ")
        if i >= 0:
            return text[i:].strip()
    if lang == "bash":
        i = text.find("#!")
        if i >= 0:
            return text[i:].strip()
        i = text.find("set -")
        if i >= 0:
            return text[i:].strip()
    return None


def grade(prompt: dict, response: str) -> dict:
    lang = prompt["language"]
    code = extract_code(response, lang)
    out = {
        "id": prompt["id"],
        "language": lang,
        "code_extracted": code is not None,
        "code_len": len(code) if code else 0,
        "compiles": False,
        "runs": False,
        "correct": False,
        "error": "",
        "response_len": len(response),
    }
    if code is None:
        out["error"] = "no fenced code block in response"
        return out

    with tempfile.TemporaryDirectory() as td:
        if lang == "python":
            fpath = Path(td) / "snippet.py"
            fpath.write_text(code)
            r = subprocess.run(["/usr/bin/env", "python3", "-m", "py_compile", str(fpath)], capture_output=True, text=True)  # noqa: E501
            if r.returncode != 0:
                out["error"] = f"py_compile: {r.stderr.strip()[:240]}"
                return out
            out["compiles"] = True

            sanity = prompt.get("sanity")
            if not sanity:
                out["runs"] = True
                out["correct"] = True
                return out

            fn_name = sanity["fn"]
            calls = sanity["calls"]
            harness_lines = [
                f"# auto-harness for {prompt['id']}",
                "import json, sys",
                code,
                f"calls = json.loads({json.dumps(json.dumps(calls))})",
                "results = []",
                "def coerce(v):",
                "    if isinstance(v, tuple): return list(v)",
                "    return v",
                "for spec in calls:",
                "    args = spec[0] if isinstance(spec[0], list) else [spec[0]]",
                "    expected = spec[1]",
                "    try:",
                "        actual = " + fn_name + "(*args)",
                "    except Exception as e:",
                "        results.append({'ok': False, 'err': str(e), 'expected': expected})",  # noqa: E501
                "        continue",
                "    results.append({'ok': coerce(actual) == coerce(expected), 'expected': expected, 'actual': coerce(actual)})",  # noqa: E501
                "print(json.dumps(results))",
            ]
            harness = "\n".join(harness_lines)
            hpath = Path(td) / "harness.py"
            hpath.write_text(harness)
            r = subprocess.run(["/usr/bin/env", "python3", str(hpath)], capture_output=True, text=True, timeout=20)  # noqa: E501
            if r.returncode != 0:
                out["error"] = f"runtime: {r.stderr.strip()[:240]}"
                return out
            out["runs"] = True
            try:
                results = json.loads(r.stdout.strip().splitlines()[-1])
            except Exception:
                out["error"] = f"parse harness stdout: {r.stdout[:240]}"
                return out
            out["correct"] = all(rr.get("ok") for rr in results)
            if not out["correct"]:
                out["error"] = f"sanity-call failures: {[r for r in results if not r.get('ok')][:3]}"  # noqa: E501
            out["sanity_results"] = results
            return out

        if lang == "javascript":
            fpath = Path(td) / "snippet.js"
            fpath.write_text(code)
            r = subprocess.run(["/usr/bin/env", "node", "--check", str(fpath)], capture_output=True, text=True)  # noqa: E501
            if r.returncode != 0:
                out["error"] = f"node --check: {r.stderr.strip()[:240]}"
                return out
            out["compiles"] = True
            # crude runtime check: declare function + invoke a trivial case.
            harness = code + "\nconsole.log(JSON.stringify({a: typeof sumOfSquares === 'function' ? sumOfSquares([1,2,3,'x']) : null}));"  # noqa: E501
            hpath = Path(td) / "harness.js"
            hpath.write_text(harness)
            r = subprocess.run(["/usr/bin/env", "node", str(hpath)], capture_output=True, text=True, timeout=20)  # noqa: E501
            if r.returncode != 0:
                out["error"] = f"runtime: {r.stderr.strip()[:240]}"
                return out
            out["runs"] = True
            try:
                parsed = json.loads(r.stdout.strip().splitlines()[-1])
                out["correct"] = parsed.get("a") == 14  # 1 + 4 + 9
            except Exception:
                out["correct"] = False
            return out

        if lang == "bash":
            fpath = Path(td) / "snippet.sh"
            fpath.write_text(code)
            r = subprocess.run(["bash", "-n", str(fpath)], capture_output=True, text=True)  # noqa: E501
            if r.returncode != 0:
                out["error"] = f"bash -n: {r.stderr.strip()[:240]}"
                return out
            out["compiles"] = True
            r = subprocess.run(["bash", str(fpath)], capture_output=True, text=True, timeout=10)  # noqa: E501
            if r.returncode != 0:
                out["error"] = f"runtime: {r.stderr.strip()[:240]}"
                return out
            out["runs"] = True
            lines = r.stdout.strip().splitlines()
            try:
                got = [int(s.strip()) for s in lines]
                out["correct"] = got == [1, 4, 9, 16, 25]
            except Exception:
                out["correct"] = False
            return out

    out["error"] = f"unsupported language: {lang}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--model", type=str, required=False, default=None,
                    help="Model id served by vLLM; if omitted, picked from /v1/models.")
    ap.add_argument("--label", type=str, default="m6")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.prompts.read_text())

    # Pick model id from /v1/models if not specified.
    model = args.model
    if model is None:
        try:
            with urllib.request.urlopen(f"{SERVER}/v1/models", timeout=5.0) as resp:
                blob = json.loads(resp.read())
                model = blob["data"][0]["id"]
        except (urllib.error.URLError, KeyError, IndexError) as exc:
            print(f"ERROR: cannot reach vLLM at {SERVER} or /v1/models malformed: {exc}", file=sys.stderr)  # noqa: E501
            return 2

    grades = []
    pass_all = 0
    t0 = time.time()
    for p in spec["prompts"]:
        print(f"  [eval] {p['id']} ({p['language']}) … ", end="", flush=True)
        try:
            response = chat(p["prompt"], model)
        except Exception as exc:
            print(f"FAIL ({exc})")
            grades.append({"id": p["id"], "language": p["language"], "error": str(exc), "compiles": False, "runs": False, "correct": False})  # noqa: E501
            continue
        g = grade(p, response)
        g["response"] = response
        grades.append(g)
        if g["compiles"] and g["runs"] and g["correct"]:
            pass_all += 1
            print("PASS")
        else:
            print(f"FAIL ({g.get('error', '?')})")
    elapsed = time.time() - t0

    out_json = args.out_dir / f"m6_coding_eval_{args.label}.json"
    out_md = args.out_dir / f"m6_coding_eval_{args.label}.md"

    out_json.write_text(json.dumps({
        "model": model,
        "label": args.label,
        "elapsed_s": elapsed,
        "pass_all": pass_all,
        "total": len(grades),
        "gate_9_of_10": pass_all >= 9,
        "grades": grades,
    }, indent=2))

    lines = [
        f"# M6 coding-agent evaluation — {args.label}",
        "",
        f"- Model: `{model}`",
        f"- Prompts: {args.prompts}",
        f"- Pass (compiles AND runs AND correct): **{pass_all}/{len(grades)}**",
        f"- Gate ≥ 9/10: **{'PASS' if pass_all >= 9 else 'FAIL'}**",
        "",
        "| Prompt ID | Lang | Compiles | Runs | Correct | Notes |",
        "| --- | --- | :-: | :-: | :-: | --- |",
    ]
    for g in grades:
        chk = lambda b: "✅" if b else "❌"
        notes = (g.get("error") or "").replace("\n", " ")[:120]
        lines.append(
            f"| `{g['id']}` | {g['language']} | {chk(g.get('compiles'))} | "
            f"{chk(g.get('runs'))} | {chk(g.get('correct'))} | {notes} |"
        )
    lines.append("")
    out_md.write_text("\n".join(lines))

    print()
    print(f"Pass: {pass_all}/{len(grades)} ({'PASS' if pass_all >= 9 else 'FAIL'})")
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    return 0 if pass_all >= 9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
