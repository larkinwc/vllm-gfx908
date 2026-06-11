#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# torch.compile launch — dense FP16 Qwen3.5-9B, TP=4 c=4.
# Concurrency-gated compile win; see scripts/_compile_cell_common.sh header.
# No recorded compile-arm reference for this cell (the 0.20.2 dense A/B in
# BENCH_FP16_COMPILE_PORT_0202_2026_06.md measured c1/c2 only); the ±5 % gate is
# skipped. The compile win grows with concurrency, so c=4 is expected >= c=2.
set -euo pipefail
cell_id=compile_fp16_tp4_c4
model_path=/models/Qwen3.5-9B
tp=4
conc=4
ref_tput=
max_model_len=32768
lm_only=1
source "$(dirname "$(readlink -f "$0")")/_compile_cell_common.sh"
