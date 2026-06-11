#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# torch.compile launch — FP16 Qwen3.5-35B-A3B MoE (256e/8a), TP=4 c=1.
# Strongest gfx908 compile win; see scripts/_compile_cell_common.sh header.
# Compile-arm reference: BENCH_FP16_COMPILE_MOE_2026_06.md (0.20.2 fixed),
#   compile (mode3+FULL_AND_PIECEWISE) tp4_c1 = 68.60 tok/s (+10.66% vs baseline).
set -euo pipefail
cell_id=compile_moe_tp4_c1
model_path=/models/Qwen3.5-35B-A3B-Claude-4.6-Opus-Reasoning-Distilled
tp=4
conc=1
ref_tput=68.60
max_model_len=16384
lm_only=1
source "$(dirname "$(readlink -f "$0")")/_compile_cell_common.sh"
