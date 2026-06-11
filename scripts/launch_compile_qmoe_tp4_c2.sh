#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# torch.compile launch — W4A16 Qwen3-Coder-Next (qwen3_next, 512e/10a), TP=4 c=2.
# Quant-MoE capstone config; see scripts/_compile_cell_common.sh header.
# Compile-arm reference: BENCH_FP16_COMPILE_QUANT_MOE_2026_06.md (0.20.2 fixed),
#   compile (mode3+FULL_AND_PIECEWISE) tp4_c2 = 91.27 tok/s (+9.54%, TTFT -48.8%).
set -euo pipefail
cell_id=compile_qmoe_tp4_c2
model_path=/models/Qwen3-Coder-Next-AWQ-4bit
tp=4
conc=2
ref_tput=91.27
max_model_len=16384
lm_only=1
source "$(dirname "$(readlink -f "$0")")/_compile_cell_common.sh"
