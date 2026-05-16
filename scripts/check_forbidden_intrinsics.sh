#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# VAL-CROSS-004 / VAL-CK-002 / VAL-ISA-003 — build-hygiene scan for
# MI300+-only intrinsics (forbidden on gfx908):
#   - v_smfmac_*
#   - v_mfma_*scale*
#
# Usage:
#   scripts/check_forbidden_intrinsics.sh <so-or-glob> [<so-or-glob> ...]
#
# Exits 0 if zero matches across all inputs, non-zero otherwise.
# When called with no args, scans the repo's standard vLLM build outputs.
set -euo pipefail

OBJDUMP=${OBJDUMP:-/opt/rocm/core-7.12/lib/llvm/bin/llvm-objdump}
[ -x "$OBJDUMP" ] || OBJDUMP=llvm-objdump

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4

if [ "$#" -eq 0 ]; then
  set -- \
    "$REPO/vllm/_rocm_C.abi3.so" \
    "$REPO/vllm/_C.abi3.so" \
    "$REPO/vllm/_moe_C.abi3.so"
fi

PATTERN='v_smfmac|v_mfma_.*scale'
FAIL=0
TOTAL=0

for so in "$@"; do
  if [ ! -f "$so" ]; then
    printf "  [skip] %s (missing)\n" "$so"
    continue
  fi
  TOTAL=$((TOTAL + 1))
  matches=$("$OBJDUMP" -d "$so" 2>/dev/null | grep -E -c "$PATTERN" || true)
  matches=${matches:-0}
  if [ "$matches" -gt 0 ]; then
    printf "  [FAIL] %s : %s forbidden intrinsic match(es)\n" "$so" "$matches"
    FAIL=$((FAIL + 1))
  else
    printf "  [ok]   %s : zero forbidden intrinsics\n" "$so"
  fi
done

printf "\nVAL-CROSS-004: scanned %d .so file(s); %d failed.\n" "$TOTAL" "$FAIL"
if [ "$FAIL" -ne 0 ]; then
  exit 1
fi
exit 0
