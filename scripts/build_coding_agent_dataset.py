#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Build /root/bench-int8-w4a16/datasets/coding_agent.jsonl.

Source: 10 hand-crafted coding-agent prompts from
/root/benchmark-scripts/coding_agent_bench.py (CODING_PROMPTS), each
formatted as a chat-style user turn (system + context + request) and
wrapped to a target token-length window.

Properties guaranteed:
  * 200 lines (vllm bench serve --num-prompts 200 with no oversampling).
  * Mixed input lengths: 256-8k tokens (per task description).
  * Mixed output lengths: 64-1024 tokens (per task description).
  * Deterministic: fixed seed; idempotent.
  * Schema: {"prompt": "<full prompt>", "output_tokens": <int>}.

We approximate token counts via a 3.5 chars/token heuristic for the
Qwen3.5 tokenizer (close enough for the +/-20 % tolerance the
contract allows). The resulting JSONL is consumed by
`vllm bench serve --dataset-name custom`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Mirror /root/benchmark-scripts/coding_agent_bench.py CODING_PROMPTS.
# Kept inline (~10 entries) to remove an external runtime dependency.
CODING_PROMPTS: list[dict] = [
    {
        "system": (
            "You are a senior software engineer. Write clean, efficient, and"
            " well-documented code. Follow best practices and include"
            " appropriate error handling."
        ),
        "context": (
            "def calculate_average(numbers):\n"
            "    total = 0\n"
            "    for n in numbers:\n"
            "        total += n\n"
            "    return total / len(numbers)"
        ),
        "request": (
            "Add type hints and error handling for empty lists."
            " Also add a docstring explaining the function."
        ),
    },
    {
        "system": (
            "You are a Python expert specializing in data processing."
            " Provide optimized solutions with clear explanations."
        ),
        "context": (
            "class DataProcessor:\n"
            "    def __init__(self):\n"
            "        self.data = []\n\n"
            "    def add_data(self, item):\n"
            "        self.data.append(item)\n\n"
            "    def get_average(self):\n"
            "        return sum(self.data) / len(self.data)"
        ),
        "request": (
            "Add methods for filtering, sorting, and statistical analysis"
            " (mean, median, std). Include comprehensive docstrings."
        ),
    },
    {
        "system": (
            "You are a backend engineer with expertise in API design."
            " Write RESTful, scalable code."
        ),
        "context": (
            "# Flask endpoint for user management\n"
            "@app.route('/users', methods=['GET'])\n"
            "def get_users():\n"
            "    users = User.query.all()\n"
            "    return jsonify([u.to_dict() for u in users])"
        ),
        "request": (
            "Add pagination, filtering, and proper error handling."
            " Include request validation and response schemas."
        ),
    },
    {
        "system": (
            "You are a debugging specialist. Analyze code issues and"
            " provide clear fixes with explanations."
        ),
        "context": (
            "async def fetch_data(urls):\n"
            "    results = []\n"
            "    for url in urls:\n"
            "        response = await fetch(url)\n"
            "        results.append(response)\n"
            "    return results"
        ),
        "request": (
            "This code is slow when fetching many URLs. Optimize it for"
            " concurrent fetching and add timeout handling."
        ),
    },
    {
        "system": (
            "You are a JavaScript/TypeScript expert."
            " Write modern, type-safe frontend code."
        ),
        "context": (
            "interface User {\n"
            "  name: string;\n"
            "  email: string;\n"
            "}\n\n"
            "function validateUser(user: User): boolean {\n"
            "  return user.name.length > 0 && user.email.includes('@');\n"
            "}"
        ),
        "request": (
            "Add comprehensive validation for email format, password"
            " strength, and phone number. Include error messages."
        ),
    },
    {
        "system": (
            "You are a security engineer. Focus on identifying"
            " vulnerabilities and providing secure implementations."
        ),
        "context": (
            "def login(username, password):\n"
            '    query = f"SELECT * FROM users WHERE'
            " username='{username}' AND password='{password}'\"\n"
            "    result = db.execute(query)\n"
            "    return result.fetchone()"
        ),
        "request": (
            "Identify the security vulnerability and rewrite this to be"
            " secure. Add proper authentication practices."
        ),
    },
    {
        "system": (
            "You are a database specialist."
            " Optimize queries and design efficient schemas."
        ),
        "context": (
            "-- Query to get user orders\n"
            "SELECT u.name, o.id, o.total, p.name as product_name\n"
            "FROM users u\n"
            "JOIN orders o ON u.id = o.user_id\n"
            "JOIN products p ON o.product_id = p.id\n"
            "WHERE o.created_at > '2024-01-01'"
        ),
        "request": (
            "Analyze query performance and suggest optimizations."
            " Add indexes and rewrite for better efficiency."
        ),
    },
    {
        "system": (
            "You are an ML engineer. Focus on efficient model"
            " implementations and data pipelines."
        ),
        "context": (
            "import numpy as np\n\n"
            "def train_model(X, y, epochs=100):\n"
            "    weights = np.random.randn(X.shape[1])\n"
            "    for _ in range(epochs):\n"
            "        predictions = X @ weights\n"
            "        errors = predictions - y\n"
            "        weights -= 0.01 * X.T @ errors\n"
            "    return weights"
        ),
        "request": (
            "Add batch processing, learning rate scheduling, and early"
            " stopping. Include logging and validation."
        ),
    },
    {
        "system": (
            "You are a DevOps engineer."
            " Write automation scripts and infrastructure code."
        ),
        "context": (
            "# Docker build script\n"
            "docker build -t myapp:latest .\n"
            "docker run -d -p 8080:80 myapp:latest"
        ),
        "request": (
            "Create a complete CI/CD pipeline script with health checks,"
            " rollback, and monitoring integration."
        ),
    },
    {
        "system": (
            "You are a code reviewer."
            " Provide constructive feedback and suggest improvements."
        ),
        "context": (
            "def process_file(filename):\n"
            "    with open(filename) as f:\n"
            "        data = f.read()\n"
            "    lines = data.split('\\n')\n"
            "    results = []\n"
            "    for line in lines:\n"
            "        if line.startswith('#'):\n"
            "            continue\n"
            "        parts = line.split(',')\n"
            "        results.append(parts)\n"
            "    return results"
        ),
        "request": (
            "Review this code for issues: error handling, performance,"
            " readability. Provide improved version."
        ),
    },
]

CHARS_PER_TOKEN = 3.5  # Qwen3.5 tokenizer ratio for English+code


def _approx_tokens(s: str) -> int:
    return max(1, int(len(s) / CHARS_PER_TOKEN))


def _pad_context(base_context: str, target_tokens: int) -> str:
    """Pad/truncate ``base_context`` toward ``target_tokens``."""
    cur_tok = _approx_tokens(base_context)
    if cur_tok >= target_tokens:
        # Truncate to roughly target.
        target_chars = int(target_tokens * CHARS_PER_TOKEN)
        return base_context[:target_chars]
    # Pad with a deterministic filler that mimics agent context (file
    # paths, log lines, code chunks).  Repeating the original context
    # is enough to span 256-8k tokens without becoming pathological.
    pad_unit = (
        "\n\n# additional related code (for context)\n"
        + base_context
        + "\n\n# additional related logs\n"
        + "\n".join(
            f"[INFO] step {i}: validated input dict and output schema" for i in range(5)
        )
        + "\n"
    )
    out = base_context
    while _approx_tokens(out) < target_tokens:
        out = out + pad_unit
    # Final trim back toward target.
    target_chars = int(target_tokens * CHARS_PER_TOKEN)
    return out[:target_chars]


def build(num_prompts: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items: list[dict] = []
    # Bucket schedule: input 256-8k, output 64-1024.
    # Mix:
    #   60 % short turns  : input 256-1024,  output 64-256
    #   25 % medium turns : input 1024-3072, output 256-512
    #   15 % long turns   : input 3072-8000, output 512-1024
    n_short = int(num_prompts * 0.60)
    n_med = int(num_prompts * 0.25)
    n_long = num_prompts - n_short - n_med
    plan: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for _ in range(n_short):
        plan.append(((256, 1024), (64, 256)))
    for _ in range(n_med):
        plan.append(((1024, 3072), (256, 512)))
    for _ in range(n_long):
        plan.append(((3072, 8000), (512, 1024)))
    rng.shuffle(plan)

    for i, ((in_lo, in_hi), (out_lo, out_hi)) in enumerate(plan):
        base = CODING_PROMPTS[i % len(CODING_PROMPTS)]
        target_in = rng.randint(in_lo, in_hi)
        target_out = rng.randint(out_lo, out_hi)
        # Build a chat-style prompt; vLLM custom dataset supplies it
        # raw, server treats as plain prompt (no chat-template by
        # default; --skip-chat-template not used).  We still include
        # the system framing inline so the prompt content matches a
        # realistic coding-agent payload.
        ctx = _pad_context(base["context"], target_in - 80)
        prompt = (
            f"### System\n{base['system']}\n\n"
            f"### Code Context\n```\n{ctx}\n```\n\n"
            f"### Request\n{base['request']}\n\n"
            f"### Response\n"
        )
        items.append({"prompt": prompt, "output_tokens": target_out})
    return items


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        default=Path("/root/bench-int8-w4a16/datasets/coding_agent.jsonl"),
    )
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    items = build(args.num_prompts, args.seed)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    in_lens = sorted(_approx_tokens(it["prompt"]) for it in items)
    out_lens = sorted(it["output_tokens"] for it in items)
    print(f"wrote {len(items)} prompts to {args.out}")
    print(
        f"  input tokens (approx): min={in_lens[0]}, "
        f"p50={in_lens[len(in_lens) // 2]}, max={in_lens[-1]}"
    )
    print(
        f"  output tokens         : min={out_lens[0]}, "
        f"p50={out_lens[len(out_lens) // 2]}, max={out_lens[-1]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
