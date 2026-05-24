#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""M2 dispatcher smoke harness for ``EMIT_INT8_NEXT`` (VAL-M2-003).

Imports :func:`mi100_int8_dispatch.choose_emit_int8_next` and calls it
twice with two synthetic shapes: one that satisfies every activation
rule clause (so the dispatcher returns ``True``) and one that
deliberately trips the N-cap (so the dispatcher returns ``False``).
Each decision is printed as a grep-friendly line containing the
literal substrings ``emit_int8_next=True`` and ``emit_int8_next=False``
so VAL-M2-003 can verify the log via simple ``grep``.

Output is the on-disk log at
``/root/bench-int8-w4a16-fused/m2-linear/dispatch_smoke.log``.
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import sys


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=pathlib.Path(
            "/root/bench-int8-w4a16-fused/m2-linear/dispatch_smoke.log"
        ),
        help="Output log path (default: %(default)s).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _make_parser().parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Surface the dispatcher's debug log as well so the dispatch_smoke
    # log shows both the wrapper-line we print AND the internal
    # dispatcher reasoning.
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(name)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    # Import here so any import-time failure surfaces in the log alongside
    # the dispatcher call lines.
    from vllm.model_executor.kernels.linear.scaled_mm.mi100_int8_dispatch import (
        EMIT_INT8_NEXT_MAX_N,
        choose_emit_int8_next,
    )

    cases: list[tuple[str, dict]] = [
        (
            "fusable Qwen3.5-9B gate_up_proj shape "
            "(M=32,N=5120,K=3584, env-on, gfx908)",
            dict(
                M=32,
                N=5120,
                K=3584,
                layer_name="model.layers.0.mlp.gate_up_proj",
                on_gfx908=True,
                env_disabled=False,
            ),
        ),
        (
            "N-cap trip (M=32,N>32768,K=3584) — should route to emit_int8_next=False",
            dict(
                M=32,
                N=EMIT_INT8_NEXT_MAX_N + 1024,
                K=3584,
                layer_name="model.layers.0.mlp.gate_up_proj",
                on_gfx908=True,
                env_disabled=False,
            ),
        ),
    ]

    lines: list[str] = []
    lines.append("# M2 EMIT_INT8_NEXT dispatcher smoke (VAL-M2-003)")
    lines.append(f"# EMIT_INT8_NEXT_MAX_N = {EMIT_INT8_NEXT_MAX_N}")
    for label, kwargs in cases:
        decision = choose_emit_int8_next(**kwargs)
        line = (
            f"emit_int8_next={decision} layer={kwargs['layer_name']!r} "
            f"shape=(M={kwargs['M']},N={kwargs['N']},K={kwargs['K']}) "
            f"-- {label}"
        )
        print(line)
        lines.append(line)

    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nWrote dispatch smoke log to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
