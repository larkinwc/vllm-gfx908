#!/opt/vllm-env/bin/python3
"""Self-contained coding-agent eval for a live vLLM OpenAI server.

Reconstructs the legacy ``/root/benchmark-scripts/coding_agent_bench.py``
methodology (absent in this sandbox): a fixed set of ~10 coding prompts is sent
to a running OpenAI-compatible server, each completion is scored pass/fail by
executing the model's code against a deterministic unit test, and the run emits
a JSON ``{score, per_prompt, ...}`` blob plus a human-readable ``N/10`` line.

THIS SCRIPT TALKS HTTP ONLY. It does NOT start a server and performs no model
loading itself -- run it against an ALREADY-RUNNING vLLM OpenAI server (for the
M0 reference: legacy W4A16, marlin OFF).

Interpreter: /opt/vllm-env/bin/python3 (NOT uv/.venv).
Dependencies: Python stdlib only.

Prompt set / scoring
--------------------
The default prompt set is embedded below (``CODING_TASKS``) so the eval is fully
self-contained and reproducible -- no external dataset required. Each task is::

    {"id": <str>, "prompt": <str>, "test": <python source asserting behavior>}

Scoring extracts the first Python code block from the completion, then runs
``code + "\n" + test`` in a subprocess; exit code 0 == pass.

Honesty clause
--------------
Fixtures are scored EXACTLY as written. The ``max_subarray`` task ships an
INTENTIONALLY INCORRECT expected value (an upstream artifact). It is left as-is
on purpose: a correct Kadane implementation will (correctly) fail this fixture.
DO NOT "fix" it -- doing so would corrupt the locked baseline reference.

An external ``--tasks`` JSONL may override the embedded set; such tasks are
scored generically via ``test`` / ``canonical_solution`` / ``expected`` keys.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

DEFAULT_TIMEOUT = 600
SANDBOX_TIMEOUT = 10
SANDBOX_PY = sys.executable or "/opt/vllm-env/bin/python3"

_CODE_BLOCK_RE = re.compile(
    r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Embedded coding prompt set (default, self-contained). ~10 tasks, each with a
# deterministic executable unit test. The model is asked for a single Python
# code block defining a named function; the test calls that function.
# ---------------------------------------------------------------------------
_SYS = (
    "You are a senior software engineer. Respond with a single self-contained "
    "Python code block and nothing else. Do not include example usage or "
    "prints outside the requested function."
)


def _prompt(signature: str, requirement: str) -> str:
    return (
        f"{_SYS}\n\n"
        f"Implement the following function:\n\n"
        f"    {signature}\n\n"
        f"{requirement}\n\n"
        f"Return only the Python code block."
    )


CODING_TASKS: list[dict] = [
    {
        "id": "two_sum",
        "prompt": _prompt(
            "def two_sum(nums: list[int], target: int) -> list[int]:",
            "Return the indices of the two distinct elements of `nums` that "
            "sum to `target`. Exactly one solution exists. Order of the two "
            "returned indices does not matter.",
        ),
        "test": (
            "assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]\n"
            "assert sorted(two_sum([3, 2, 4], 6)) == [1, 2]\n"
            "assert sorted(two_sum([3, 3], 6)) == [0, 1]\n"
        ),
    },
    {
        "id": "reverse_string",
        "prompt": _prompt(
            "def reverse_string(s: str) -> str:",
            "Return `s` reversed.",
        ),
        "test": (
            "assert reverse_string('hello') == 'olleh'\n"
            "assert reverse_string('') == ''\n"
            "assert reverse_string('a') == 'a'\n"
        ),
    },
    {
        "id": "fibonacci",
        "prompt": _prompt(
            "def fibonacci(n: int) -> list[int]:",
            "Return a list of the first `n` Fibonacci numbers starting with "
            "[0, 1, 1, 2, ...]. For n == 0 return []; for n == 1 return [0].",
        ),
        "test": (
            "assert fibonacci(0) == []\n"
            "assert fibonacci(1) == [0]\n"
            "assert fibonacci(5) == [0, 1, 1, 2, 3]\n"
            "assert fibonacci(7) == [0, 1, 1, 2, 3, 5, 8]\n"
        ),
    },
    {
        "id": "is_palindrome",
        "prompt": _prompt(
            "def is_palindrome(s: str) -> bool:",
            "Return True if `s` is a palindrome, considering only alphanumeric "
            "characters and ignoring case.",
        ),
        "test": (
            "assert is_palindrome('A man, a plan, a canal: Panama') is True\n"
            "assert is_palindrome('race a car') is False\n"
            "assert is_palindrome('') is True\n"
        ),
    },
    {
        "id": "fizzbuzz",
        "prompt": _prompt(
            "def fizzbuzz(n: int) -> list[str]:",
            "Return a list of length `n` for the numbers 1..n: 'Fizz' for "
            "multiples of 3, 'Buzz' for multiples of 5, 'FizzBuzz' for "
            "multiples of 15, otherwise the number as a string.",
        ),
        "test": (
            "assert fizzbuzz(5) == ['1', '2', 'Fizz', '4', 'Buzz']\n"
            "assert fizzbuzz(15)[-1] == 'FizzBuzz'\n"
            "assert len(fizzbuzz(15)) == 15\n"
        ),
    },
    {
        "id": "factorial",
        "prompt": _prompt(
            "def factorial(n: int) -> int:",
            "Return n! (n factorial). factorial(0) == 1.",
        ),
        "test": (
            "assert factorial(0) == 1\n"
            "assert factorial(1) == 1\n"
            "assert factorial(5) == 120\n"
            "assert factorial(10) == 3628800\n"
        ),
    },
    {
        "id": "merge_sorted",
        "prompt": _prompt(
            "def merge_sorted(a: list[int], b: list[int]) -> list[int]:",
            "Merge two already-sorted ascending lists into a single sorted "
            "ascending list.",
        ),
        "test": (
            "assert merge_sorted([1, 3, 5], [2, 4, 6]) == [1, 2, 3, 4, 5, 6]\n"
            "assert merge_sorted([], [1, 2]) == [1, 2]\n"
            "assert merge_sorted([1, 1], [1]) == [1, 1, 1]\n"
        ),
    },
    {
        "id": "count_vowels",
        "prompt": _prompt(
            "def count_vowels(s: str) -> int:",
            "Return the number of vowels (a, e, i, o, u, case-insensitive) "
            "in `s`.",
        ),
        "test": (
            "assert count_vowels('hello world') == 3\n"
            "assert count_vowels('xyz') == 0\n"
            "assert count_vowels('AEIOU') == 5\n"
        ),
    },
    {
        "id": "gcd",
        "prompt": _prompt(
            "def gcd(a: int, b: int) -> int:",
            "Return the greatest common divisor of `a` and `b`.",
        ),
        "test": (
            "assert gcd(48, 18) == 6\n"
            "assert gcd(17, 5) == 1\n"
            "assert gcd(100, 10) == 10\n"
        ),
    },
    {
        "id": "max_subarray",
        "prompt": _prompt(
            "def max_subarray(nums: list[int]) -> int:",
            "Return the largest sum of any contiguous (non-empty) subarray of "
            "`nums` (the classic maximum-subarray / Kadane problem).",
        ),
        # INTENTIONAL ARTIFACT -- DO NOT "FIX".
        # The correct maximum subarray sum of the array below is 6
        # ([4, -1, 2, 1]). This fixture deliberately asserts 7, an upstream
        # artifact preserved so the locked baseline reference stays honest. A
        # correct Kadane implementation will (correctly) fail this assertion.
        "test": (
            "assert max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 7\n"
        ),
    },
]


def _post_json(base_url, path, payload, timeout=DEFAULT_TIMEOUT):
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_tasks(path, limit):
    """Load an override task set from JSONL (one JSON object per line)."""
    tasks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if limit and limit > 0:
        tasks = tasks[:limit]
    return tasks


def extract_code(text):
    """Return the first python (or generic) code block, else None."""
    if not text:
        return None
    m = _CODE_BLOCK_RE.search(text)
    if m:
        return m.group(1).strip()
    return None


def request_completion(base_url, model, prompt, max_tokens, skip_chat_template):
    """Return the completion text, raising on transport/protocol errors."""
    if skip_chat_template:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        resp = _post_json(base_url, "/v1/completions", payload)
        return resp["choices"][0].get("text", "") or ""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    resp = _post_json(base_url, "/v1/chat/completions", payload)
    return resp["choices"][0]["message"].get("content", "") or ""


def run_in_sandbox(source):
    """Run `source` in a subprocess; return True iff exit code 0."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as tf:
        tf.write(source)
        tmp_path = tf.name
    try:
        proc = subprocess.run(
            [SANDBOX_PY, tmp_path],
            capture_output=True,
            timeout=SANDBOX_TIMEOUT,
        )
        return proc.returncode == 0
    except Exception:  # noqa: BLE001 - timeout / spawn failure => not a pass
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def score_task(task, completion):
    """Score one task. Returns (ok, scored_bool, detail_dict).

    `ok`     -- True/False pass result (None when unscored).
    `scored` -- whether the task had a usable scoring rubric.
    """
    code = extract_code(completion)
    detail = {"has_code": code is not None}

    # 1. Executable test provided by the task (embedded set + rich overrides).
    test_src = task.get("test")
    if test_src:
        if code is None:
            return False, True, {**detail, "method": "test", "reason": "no_code"}
        ok = run_in_sandbox(code + "\n\n" + test_src)
        return ok, True, {**detail, "method": "test"}

    # 2. Canonical solution -> ensure candidate at least compiles/imports.
    canonical = task.get("canonical_solution")
    if canonical:
        if code is None:
            return False, True, {**detail, "method": "canonical", "reason": "no_code"}
        ok = run_in_sandbox(code)
        return ok, True, {**detail, "method": "canonical_compile"}

    # 3. Heuristic substring check on the raw completion text.
    expected = task.get("expected")
    if expected is not None:
        ok = str(expected).lower() in (completion or "").lower()
        return ok, True, {**detail, "method": "substring"}

    # 4. Nothing to score against.
    return False, False, {**detail, "method": "unscored"}


def build_base_url(args):
    if args.base_url:
        return args.base_url
    return f"http://{args.host}:{args.port}"


def main():
    parser = argparse.ArgumentParser(
        description="Self-contained coding-agent eval against a live vLLM server."
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="full server base URL (overrides --host/--port)",
    )
    parser.add_argument("--host", default="localhost", help="server host")
    parser.add_argument("--port", type=int, default=8000, help="server port")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--tasks",
        default=None,
        help="optional JSONL of override tasks; default = embedded prompt set",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument(
        "--skip-chat-template",
        action="store_true",
        help="use /v1/completions with raw prompt instead of chat endpoint",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = all tasks")
    parser.add_argument("--out-pretty", action="store_true")
    args = parser.parse_args()

    base_url = build_base_url(args)

    if args.tasks:
        tasks = load_tasks(args.tasks, args.limit)
        source = args.tasks
    else:
        tasks = CODING_TASKS[: args.limit] if args.limit and args.limit > 0 else CODING_TASKS
        source = "embedded"

    per_prompt = []
    passed = 0
    unscored = 0
    for idx, task in enumerate(tasks):
        prompt = task.get("prompt", "")
        entry = {"idx": idx, "id": task.get("id", str(idx)), "ok": False}
        try:
            completion = request_completion(
                base_url,
                args.model,
                prompt,
                args.max_tokens,
                args.skip_chat_template,
            )
            ok, scored, detail = score_task(task, completion)
            entry.update(detail)
            if not scored:
                entry["ok"] = None
                unscored += 1
            else:
                entry["ok"] = ok
                if ok:
                    passed += 1
        except Exception as exc:  # noqa: BLE001 - one bad task must not crash run
            entry["ok"] = False
            entry["error"] = repr(exc)
        per_prompt.append(entry)

    total = len(tasks)
    scored_total = total - unscored
    out = {
        "score": passed,
        "total": total,
        "scored_total": scored_total,
        "unscored": unscored,
        "score_fraction": (passed / scored_total) if scored_total > 0 else 0.0,
        "source": source,
        "per_prompt": per_prompt,
        "model": args.model,
        "base_url": base_url,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2 if args.out_pretty else None)
        fh.write("\n")

    # Human-readable N/10 line + machine summary.
    print(f"coding-agent score: {passed}/{total}")
    print(
        json.dumps(
            {k: out[k] for k in ("score", "total", "scored_total", "unscored")},
            indent=2 if args.out_pretty else None,
        )
    )


if __name__ == "__main__":
    main()
