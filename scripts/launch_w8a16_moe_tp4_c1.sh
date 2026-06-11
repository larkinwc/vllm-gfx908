#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# W8A16 (GPTQ-8bit) Qwen3.5-35B-A3B MoE (qwen3_5_moe, 256e/8a), TP=4 c=1.
#
# Capacity reference deployment: 8-bit weight-only keeps the FP16 MFMA compute
# path (int8->fp16 dequant) while halving weight bytes (~39 GB vs ~67 GB fp16),
# leaving far more VRAM for KV cache / context on 4x MI100. torch.compile
# (mode=3 + FULL_AND_PIECEWISE) is the winning config for MoE on gfx908 so it is
# baked in here. Measured same-binary A/B on this W8A16 model (v0.22.1, TP=2
# GPUs 1,2): compile c1 +14.2% (55.12->62.97), c2 +40.4% (74.47->104.54),
# c4 +48.1% (115.27->170.70) tok/s — win grows with concurrency, c1 positive.
# See scripts/_compile_cell_common.sh header + BENCH_W8A16_35B_MOE_2026_06.md.
#
# ref_tput (TP=4 ±5% gate) unmeasured — A/B above was the TP=2 thermal subset.
set -euo pipefail
cell_id=w8a16_moe_tp4_c1
model_path=/models/Qwen3.5-35B-A3B-GPTQ-8bit
tp=4
conc=1
ref_tput=
max_model_len=16384
lm_only=1
source "$(dirname "$(readlink -f "$0")")/_compile_cell_common.sh"
