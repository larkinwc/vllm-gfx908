#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# torch.compile launch — dense FP16 Qwen3.5-9B, TP=4 c=1.
# Concurrency-gated compile win; see scripts/_compile_cell_common.sh header.
# Compile-arm reference: BENCH_FP16_COMPILE_PORT_0202_2026_06.md (0.20.2 fixed),
#   compile (mode3+FULL_AND_PIECEWISE) tp4_c1 = 75.81 tok/s (+4.47% vs baseline).
set -euo pipefail
cell_id=compile_fp16_tp4_c1
model_path=/models/Qwen3.5-9B
tp=4
conc=1
ref_tput=75.81
max_model_len=32768
lm_only=1
source "$(dirname "$(readlink -f "$0")")/_compile_cell_common.sh"
