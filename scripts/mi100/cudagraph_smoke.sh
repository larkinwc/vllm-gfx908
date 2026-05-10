#!/usr/bin/env bash
# =============================================================================
# scripts/mi100/cudagraph_smoke.sh
#
# Verifies that FULL_DECODE_ONLY cudagraph capture succeeds under the M3
# Triton kernels for both quant schemes on TP=1 and TP=4 (VAL-TRITON-008).
#
# Strategy (matches services.yaml::vllm-w{8a8,4a16}-tp{1,4}):
#   1) `pkill` orphans for safety.
#   2) Start vLLM with the appropriate quant model + `--compilation-config
#      '{"cudagraph_mode":"FULL_DECODE_ONLY"}'`.
#   3) Wait for /health up to 360 s (graph capture can be slow first run).
#   4) grep the server log for `cudagraph_mode=FULL_DECODE_ONLY` and
#      `captured ... graphs` to prove the capture succeeded.
#   5) Run a 100-step decode probe via /v1/completions and confirm no
#      RuntimeError / HIP error appears in the log.
#   6) Stop the server.
#
# Usage:
#   scripts/mi100/cudagraph_smoke.sh w8a8 1   # TP=1 W8A8
#   scripts/mi100/cudagraph_smoke.sh w8a8 4   # TP=4 W8A8
#   scripts/mi100/cudagraph_smoke.sh w4a16 1  # TP=1 W4A16
#   scripts/mi100/cudagraph_smoke.sh w4a16 4  # TP=4 W4A16
#
# Outputs:
#   /root/bench-int8-w4a16/triton/cudagraph_<quant>_tp<tp>.log
# =============================================================================
set -uo pipefail

QUANT=${1:-w8a8}
TP=${2:-1}

# Resolve repo root from this script's location (scripts/mi100/cudagraph_smoke.sh)
# so the script works in any worktree (no hard-coded absolute path).
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)
# Sanity: prefer git's view if available.
if command -v git >/dev/null 2>&1; then
  GIT_TOPLEVEL=$(git -C "$REPO" rev-parse --show-toplevel 2>/dev/null || true)
  if [[ -n "$GIT_TOPLEVEL" ]]; then
    REPO=$GIT_TOPLEVEL
  fi
fi
OUT_DIR=/root/bench-int8-w4a16/triton
mkdir -p "$OUT_DIR"
LOG=$OUT_DIR/cudagraph_${QUANT}_tp${TP}.log

case "$QUANT" in
  w8a8)  MODEL=/models/Qwen3.5-9B-w8a8 ;;
  w4a16) MODEL=/models/Qwen3.5-9B-w4a16 ;;
  *) echo "unknown quant $QUANT" >&2; exit 2 ;;
esac

# Kill orphans (manifest stop semantics).
pkill -9 -f 'vllm.entrypoints' 2>/dev/null
pkill -9 -f 'VLLM::' 2>/dev/null
sleep 2

ENV_ARGS=(
  LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
  ROCM_PATH=/opt/rocm/core-7.12
  PATH=/opt/rocm/core-7.12/bin:$PATH
  PYTORCH_ROCM_ARCH=gfx908
  PYTHONPATH=$REPO
  VLLM_ROCM_USE_AITER=1
  VLLM_ROCM_USE_SKINNY_GEMM=0
  TORCH_COMPILE_DISABLE=1
)
if [[ "$TP" == "4" ]]; then
  ENV_ARGS+=(VLLM_MI100_DISABLE_CUSTOM_AR=1)
fi
if [[ "$TP" == "1" ]]; then
  ENV_ARGS+=(CUDA_VISIBLE_DEVICES=0)
fi
if [[ "$QUANT" == "w4a16" && "$TP" == "4" ]]; then
  ENV_ARGS+=(
    TRITON_CACHE_DIR=/root/bench-int8-w4a16/baseline/triton_cache_w4a16
    NCCL_TIMEOUT_HOURS=2
    TORCH_NCCL_BLOCKING_WAIT=1
    TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
  )
fi

VLLM_ARGS=(
  --model "$MODEL"
  --dtype float16
  --tensor-parallel-size "$TP"
  --max-model-len 8192
  --block-size 32
  --enable-prefix-caching
  --language-model-only
  --gpu-memory-utilization 0.93
  --port 8000
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
)
if [[ "$TP" == "4" ]]; then
  VLLM_ARGS+=(--disable-custom-all-reduce)
fi

env "${ENV_ARGS[@]}" \
  /opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
  "${VLLM_ARGS[@]}" \
  > "$LOG" 2>&1 &
PID=$!

# Wait up to 360s for /health.
DEADLINE=$(( $(date +%s) + 360 ))
HEALTHY=0
while [[ $(date +%s) -lt $DEADLINE ]]; do
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "FATAL: server died early (see $LOG)" >&2
    break
  fi
  if curl -sf http://localhost:8000/health > /dev/null; then
    HEALTHY=1
    break
  fi
  sleep 5
done

if [[ "$HEALTHY" != "1" ]]; then
  echo "FATAL: server did not become healthy in 360s" >&2
  kill -9 "$PID" 2>/dev/null
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null
  pkill -9 -f 'VLLM::' 2>/dev/null
  exit 3
fi

# Inspect log for graph capture markers.
echo "[cudagraph_smoke] checking log markers..."
GOT_MODE=0
GOT_CAPTURED=0
grep -q "cudagraph_mode.*FULL_DECODE_ONLY" "$LOG" && GOT_MODE=1
# vLLM logs either "captured N graphs" (older) or
# "Graph capturing finished in N secs" + "Capturing CUDA graphs ... N/N"
# (current). Accept any of these.
if grep -qE "captured [0-9]+ graphs|[Cc]aptured [0-9]+ cudagraphs|Graph capturing finished|Capturing CUDA graphs.*[0-9]+/[0-9]+" "$LOG"; then
  GOT_CAPTURED=1
fi
if [[ "$GOT_MODE" != "1" || "$GOT_CAPTURED" != "1" ]]; then
  echo "FAIL: missing capture markers in $LOG (mode=$GOT_MODE captured=$GOT_CAPTURED)" >&2
fi

# 100-step decode probe.
PROBE_OUT=$(mktemp)
curl -sf http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"def fibonacci(n):\\n\", \"max_tokens\": 100, \"temperature\": 0}" \
  -o "$PROBE_OUT"
PROBE_OK=$?

if [[ "$PROBE_OK" != "0" ]]; then
  echo "FAIL: decode probe HTTP error" >&2
fi

if grep -qE "(RuntimeError|HIP error|hipError)" "$LOG"; then
  echo "FAIL: HIP/runtime error in server log" >&2
  PROBE_OK=99
fi

# Stop server.
kill -9 "$PID" 2>/dev/null
pkill -9 -f 'vllm.entrypoints' 2>/dev/null
pkill -9 -f 'VLLM::' 2>/dev/null
sleep 2

# Result.
if [[ "$GOT_MODE" == "1" && "$GOT_CAPTURED" == "1" && "$PROBE_OK" == "0" ]]; then
  echo "PASS: cudagraph smoke for $QUANT TP=$TP"
  exit 0
else
  echo "FAIL: cudagraph smoke for $QUANT TP=$TP (see $LOG)"
  exit 4
fi
