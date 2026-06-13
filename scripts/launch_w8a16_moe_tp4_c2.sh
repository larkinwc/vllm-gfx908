#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# W8A16 (GPTQ-8bit) Qwen3.5-35B-A3B MoE (qwen3_5_moe, 256e/8a), TP=4 c=2.
# Capacity reference deployment + compile MoE winner; see the c1 sibling and
# scripts/_compile_cell_common.sh header + BENCH_W8A16_35B_MOE_2026_06.md.
set -euo pipefail
cell_id=w8a16_moe_tp4_c2
model_path=/models/Qwen3.5-35B-A3B-GPTQ-8bit
tp=4
conc=2
ref_tput=
max_model_len=16384
lm_only=1
source "$(dirname "$(readlink -f "$0")")/_compile_cell_common.sh"
