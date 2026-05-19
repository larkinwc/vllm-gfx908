#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# VAL-CROSS-004 — Disable-path smoke harness.
#
# Modes:
#   (default)  Original HBM-mission disable-path: for each new env flag
#              introduced by the HBM-mission (KV_CACHE_DTYPE,
#              ENABLE_CHUNKED_PREFILL+MAX_NUM_BATCHED_TOKENS, NCCL_ALGO),
#              run one canary cell with the flag UNSET and verify the
#              cell reproduces the corresponding production
#              output_throughput_toks_s within ±5 %.
#
#   --hbm-fa   HBM-FA-mission disable-path (this mission, VAL-CROSS-004):
#              for two canary cells (w8a8_tp1_c1, w4a16_tp4_c4) `unset
#              VLLM_MI100_USE_TUNED_FLASH_DECODE` and invoke the EXISTING
#              `scripts/launch_hbm_<cell>.sh` (M4 fallback path). Assert
#              the measured throughput reproduces the M4 baseline
#              `ref_tput` (baked into the launch script) within ±5 %.
#              Writes the log + CSV under
#              /root/bench-int8-w4a16-hbm-fa/m3-final/.
#
# Per-flag canary assignment (per mission spec):
#   KV_CACHE_DTYPE          -> w8a8_tp1_c1 (KV-INT8 + chunked unset)
#   ENABLE_CHUNKED_PREFILL  -> w8a8_tp1_c1 (covered by the same launch)
#   MAX_NUM_BATCHED_TOKENS  -> w8a8_tp1_c1 (covered by the same launch)
#   NCCL_ALGO               -> w8a8_tp4_c4 (only baked NCCL_ALGO on w8a8)
#
# Implementation strategy:
#   For each canary cell we invoke the *production* launch script
#   `scripts/launch_<cell>.sh --check` (which references the
#   `final_grid.csv` production tput and gates at ±2 %), but FIRST we
#   `unset` the flag(s) under test so the resulting CLI is byte-identical
#   to the pre-extension production launch. If the launch's own ±2 %
#   gate passes, our ±5 % smoke gate trivially passes too; we record
#   the measured Δ as the smoke result. If the script's ±2 % gate fails
#   but the actual delta is within ±5 %, we still pass the smoke gate
#   (the ±5 % gate is intentionally looser than the script's ±2 % gate).
#
# Output:
#   /root/bench-int8-w4a16-hbm/m4-final/disable_path_smoke.log
#   /root/bench-int8-w4a16-hbm/m4-final/disable_path_smoke.csv
#
# Exit code: 0 if every smoke gate (≤±5 %) passes; non-zero otherwise.
set -euo pipefail

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4

# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------
MODE=${1:-hbm}
case "$MODE" in
  --hbm-fa|hbm-fa)
    MODE=hbm-fa
    ;;
  ""|--hbm|hbm)
    MODE=hbm
    ;;
  *)
    echo "usage: $0 [--hbm | --hbm-fa]" >&2
    exit 2
    ;;
esac

if [[ "$MODE" == "hbm-fa" ]]; then
  OUT_DIR=/root/bench-int8-w4a16-hbm-fa/m3-final
else
  OUT_DIR=/root/bench-int8-w4a16-hbm/m4-final
fi
mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/disable_path_smoke.log"
CSV="$OUT_DIR/disable_path_smoke.csv"
: > "$LOG"
echo "flag_under_test,canary_cell,production_ref_tput,measured_tput,delta_pct,verdict" > "$CSV"

run_canary() {
  local flag="$1" cell="$2" launch_script="$3"
  echo "=== Disable-path smoke for flag=$flag canary=$cell ===" | tee -a "$LOG"
  # Unset every HBM-mission flag (so production CLI is reproduced) then
  # invoke the production launch script.
  unset KV_CACHE_DTYPE
  unset ENABLE_CHUNKED_PREFILL
  unset MAX_NUM_BATCHED_TOKENS
  unset NCCL_ALGO
  unset CUDAGRAPH_MODE
  echo "  flags unset; invoking $launch_script --check" | tee -a "$LOG"
  set +e
  bash "$REPO/$launch_script" --check >> "$LOG" 2>&1
  rc=$?
  set -e
  # The launch script's `set -euo pipefail` aborts BEFORE the final echo
  # when its ±2% gate fails (the python helper exit's non-zero, and `set -e`
  # propagates that), so we cannot rely on `[launch_*] delta=…` being
  # present in the log. Instead, locate the most recent raw.json produced by
  # the launch script and compute ref / measured / delta directly from disk.
  out_root="/root/bench-int8-w4a16/final/launch_smoke/${cell}"
  raw_json=$(ls -1t "$out_root"/bench_*/raw.json 2>/dev/null | head -n 1)
  if [[ -z "$raw_json" ]]; then
    echo "  [WARN] no raw.json found under $out_root" | tee -a "$LOG"
    echo "$flag,$cell,?,?,NA,MISSING" >> "$CSV"
    return 1
  fi
  measured=$(/opt/vllm-env/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_throughput"])' "$raw_json")
  # The reference is the launch script's baked-in `ref_tput` (production
  # `final_grid.csv` value for that cell). Pull it from the launch script
  # itself (`ref_tput=<value>` line in the launch script body).
  ref=$(grep -E "^ref_tput=" "$REPO/$launch_script" | head -n 1 | cut -d= -f2)
  delta_pct=$(/opt/vllm-env/bin/python3 -c "import sys; r=float(sys.argv[1]); a=float(sys.argv[2]); print(f'{(a-r)/r*100:+.4f}')" "$ref" "$measured")
  within_5=$(/opt/vllm-env/bin/python3 -c "import sys; print('PASS' if abs(float(sys.argv[1])) <= 5.0 else 'FAIL')" "$delta_pct")
  echo "  ref=$ref measured=$measured delta=${delta_pct}% verdict=$within_5" | tee -a "$LOG"
  echo "$flag,$cell,$ref,$measured,$delta_pct,$within_5" >> "$CSV"
  if [[ "$within_5" != "PASS" ]]; then
    return 1
  fi
  return 0
}

overall_rc=0

# ---------------------------------------------------------------------------
# HBM-FA-mission canary: VLLM_MI100_USE_TUNED_FLASH_DECODE unset on the
# EXISTING launch_hbm_<cell>.sh (M4 fallback path). The M4 launch scripts
# never set VLLM_MI100_USE_TUNED_FLASH_DECODE, so explicitly unsetting it
# here exercises the env-gate-off code path in
# vllm/v1/attention/ops/triton_unified_attention.py. The ±5 % gate is
# evaluated vs the launch script's baked-in ref_tput (the M4 baseline).
# ---------------------------------------------------------------------------
run_canary_hbm_fa() {
  local cell="$1" launch_script="$2"
  echo "=== HBM-FA disable-path smoke for cell=$cell ===" | tee -a "$LOG"
  unset VLLM_MI100_USE_TUNED_FLASH_DECODE
  unset VLLM_MI100_TUNED_FLASH_DECODE_LOOKUP
  echo "  VLLM_MI100_USE_TUNED_FLASH_DECODE unset; invoking $launch_script --check" | tee -a "$LOG"
  set +e
  bash "$REPO/$launch_script" --check >> "$LOG" 2>&1
  rc=$?
  set -e
  # Locate the most recent raw.json produced by the launch script.
  out_root="/root/bench-int8-w4a16-hbm/m4-final/launch_smoke/${cell}"
  raw_json=$(ls -1t "$out_root"/bench_*/raw.json 2>/dev/null | head -n 1)
  if [[ -z "$raw_json" ]]; then
    echo "  [WARN] no raw.json found under $out_root" | tee -a "$LOG"
    echo "VLLM_MI100_USE_TUNED_FLASH_DECODE,$cell,?,?,NA,MISSING" >> "$CSV"
    return 1
  fi
  measured=$(/opt/vllm-env/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_throughput"])' "$raw_json")
  ref=$(grep -E "^ref_tput=" "$REPO/$launch_script" | head -n 1 | cut -d= -f2)
  delta_pct=$(/opt/vllm-env/bin/python3 -c "import sys; r=float(sys.argv[1]); a=float(sys.argv[2]); print(f'{(a-r)/r*100:+.4f}')" "$ref" "$measured")
  within_5=$(/opt/vllm-env/bin/python3 -c "import sys; print('PASS' if abs(float(sys.argv[1])) <= 5.0 else 'FAIL')" "$delta_pct")
  echo "  ref=$ref measured=$measured delta=${delta_pct}% verdict=$within_5" | tee -a "$LOG"
  echo "VLLM_MI100_USE_TUNED_FLASH_DECODE,$cell,$ref,$measured,$delta_pct,$within_5" >> "$CSV"
  if [[ "$within_5" != "PASS" ]]; then
    return 1
  fi
  return 0
}

if [[ "$MODE" == "hbm-fa" ]]; then
  # Two canary cells per the HBM-FA mission spec (m3-cross-checks step 3).
  run_canary_hbm_fa "w8a8_tp1_c1" "scripts/launch_hbm_w8a8_tp1_c1.sh" || overall_rc=1
  run_canary_hbm_fa "w4a16_tp4_c4" "scripts/launch_hbm_w4a16_tp4_c4.sh" || overall_rc=1
else
  # 1. KV_CACHE_DTYPE on w8a8_tp1_c1 (also exercises ENABLE_CHUNKED_PREFILL +
  #    MAX_NUM_BATCHED_TOKENS unset, which the spec explicitly groups as
  #    a single canary cell since all three are TP=1-decode-side flags).
  run_canary "KV_CACHE_DTYPE+chunked" "w8a8_tp1_c1" "scripts/launch_w8a8_tp1_c1.sh" || overall_rc=1

  # 2. NCCL_ALGO on w8a8_tp4_c4 (the only w8a8 TP=4 cell with NCCL_ALGO baked).
  run_canary "NCCL_ALGO" "w8a8_tp4_c4" "scripts/launch_w8a8_tp4_c4.sh" || overall_rc=1
fi

echo "" | tee -a "$LOG"
echo "=== Disable-path smoke summary ===" | tee -a "$LOG"
column -s, -t < "$CSV" | tee -a "$LOG"

if [[ "$overall_rc" -eq 0 ]]; then
  echo "" | tee -a "$LOG"
  echo "VAL-CROSS-004: PASS (every disable-path within ±5 %)" | tee -a "$LOG"
else
  echo "" | tee -a "$LOG"
  echo "VAL-CROSS-004: FAIL (at least one disable-path > ±5 %)" | tee -a "$LOG"
fi
exit "$overall_rc"
