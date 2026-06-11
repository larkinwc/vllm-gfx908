#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Shared helper for the gfx908 (MI100) torch.compile launch scripts
# (scripts/launch_compile_*.sh). NOT a standalone launcher — it is sourced by
# the per-cell scripts after they set the cell-specific variables below.
#
# torch.compile + FULL_AND_PIECEWISE is a concurrency-gated decode win on
# gfx908 (measured 2026-06):
#   - dense 9B FP16  : +4.5% / +7.2% at TP=4 c1/c2 (regresses TP=1 — not shipped)
#   - fp16 35B-A3B   : +8.8% .. +11.0% across c1/c2/c4, TTFT better, no c=1 loss
#   - W4A16 512e MoE : +9.0% .. +11.0%, TTFT -28%..-49%
# Evidence:
#   docs/experiments/BENCH_FP16_COMPILE_PORT_0202_2026_06.md (dense 9B, 0.20.2)
#   docs/experiments/BENCH_FP16_COMPILE_MOE_2026_06.md       (fp16 35B-A3B)
#   docs/experiments/BENCH_FP16_COMPILE_QUANT_MOE_2026_06.md (W4A16 512e MoE)
#
# Requires the in-tree Dynamo guard fix (commit 24ac8f64e) in
# vllm/model_executor/kernels/quantization/fused_silu_quant_int8.py, which lets
# mode=3 compile complete on the 0.20.2 build (it crashed before the fix).
#
# Caller must set before sourcing:
#   cell_id        e.g. compile_moe_tp4_c2
#   model_path     absolute path under /models
#   tp             tensor-parallel size (these scripts target TP=4)
#   conc           max concurrency for the bench cell
#   ref_tput       recorded compile-arm reference output_throughput (or "")
#   max_model_len  (optional, default 32768)
#   lm_only        (optional, "1" to add --language-model-only; default off)
#
# Usage (per-cell wrapper):
#   launch_compile_<cell>.sh [--check]        # run + ±5 % gate vs ref (if set)
#   launch_compile_<cell>.sh --serve-only     # start server only
#   launch_compile_<cell>.sh --port 8001 ...  # parallel-safe port override

set -euo pipefail

: "${cell_id:?_compile_cell_common.sh: cell_id must be set}"
: "${model_path:?_compile_cell_common.sh: model_path must be set}"
: "${tp:?_compile_cell_common.sh: tp must be set}"
: "${conc:?_compile_cell_common.sh: conc must be set}"
ref_tput="${ref_tput:-}"
max_model_len="${max_model_len:-32768}"
lm_only="${lm_only:-}"

# Optional TP override for ad-hoc testing (e.g. LAUNCH_TP=2 on a GPU subset to
# avoid thermal hot-spots). The recorded ref_tput is a TP=4 measurement, so any
# override invalidates the ±5 % gate — drop the reference when overridden.
if [[ -n "${LAUNCH_TP:-}" && "${LAUNCH_TP}" != "$tp" ]]; then
  echo "[launch_compile_$cell_id] LAUNCH_TP=$LAUNCH_TP overrides tp=$tp;" \
       "disabling ±5 % gate (reference is TP=$tp)."
  tp="${LAUNCH_TP}"
  ref_tput=""
fi

# ---------------------------------------------------------------------------
# Pinned environment
# ---------------------------------------------------------------------------
export ROCM_PATH="${ROCM_PATH:-/opt/rocm/core-7.12}"
export LD_LIBRARY_PATH="${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="${ROCM_PATH}/bin:$PATH"
export PYTORCH_ROCM_ARCH=gfx908
export HF_HUB_OFFLINE=1

# vLLM must be imported from a *compiled* worktree (one that contains
# vllm/_C.abi3.so). The editable-install .pth can point at an uncompiled
# checkout (PERF_GFX908.md §3), and the emdash worktrees are recreated/cleaned
# over time, so we resolve a compiled tree at launch:
#   1. honour an explicit VLLM_SRC if it is compiled;
#   2. otherwise scan sibling emdash worktrees for one containing _C.abi3.so.
# Override with VLLM_SRC to force a specific checkout.
_find_compiled_worktree() {
  local wt_root
  wt_root="$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)"
  local cand
  for cand in "$wt_root"/*/; do
    if [[ -f "${cand}vllm/_C.abi3.so" ]]; then
      printf '%s' "${cand%/}"
      return 0
    fi
  done
  return 1
}
if [[ -n "${VLLM_SRC:-}" && -f "$VLLM_SRC/vllm/_C.abi3.so" ]]; then
  :
else
  VLLM_SRC="$(_find_compiled_worktree || true)"
fi
if [[ -z "${VLLM_SRC:-}" || ! -f "$VLLM_SRC/vllm/_C.abi3.so" ]]; then
  echo "[launch_compile_$cell_id] FATAL: no compiled vLLM worktree found" \
       "(need vllm/_C.abi3.so). Build one or set VLLM_SRC explicitly." >&2
  exit 3
fi
echo "[launch_compile_$cell_id] using compiled vLLM at $VLLM_SRC"
export PYTHONPATH="$VLLM_SRC${PYTHONPATH:+:$PYTHONPATH}"
cd "$VLLM_SRC"

# torch.compile concurrency-gated win — the whole point of these scripts.
# VLLM_MI100_TORCH_COMPILE=1 lifts rocm.py's default mode=NONE override;
# VLLM_MI100_ALLOW_PIECEWISE=1 lifts the FULL_DECODE_ONLY cudagraph override
# so the --compilation-config below (mode=3 + FULL_AND_PIECEWISE) is honored.
#
# LAUNCH_NO_COMPILE=1 runs the *baseline* arm instead: rocm.py's default
# FULL_DECODE_ONLY (no Inductor), same model/TP/mem, for A/B comparison.
NO_COMPILE="${LAUNCH_NO_COMPILE:-0}"
if [[ "$NO_COMPILE" != "1" ]]; then
  export VLLM_MI100_TORCH_COMPILE=1
  export VLLM_MI100_ALLOW_PIECEWISE=1
fi

# Custom all-reduce stays disabled on gfx908 (IPC buffers go stale on HIP graph
# replay); rocm.py auto-detects but we set it explicitly for reproducibility.
export VLLM_MI100_DISABLE_CUSTOM_AR=1

# Concurrency-gated tuned fused-MoE config (opt-in via LAUNCH_TUNED_MOE=1).
# The int8_w8a16 E=256/N=256 tune is a NET WIN at c>=2 (+15% c2, +24% c4) but
# REGRESSES single-stream (-11.6% c1: its decode-M tile loses to the Triton
# default). So it is *not* installed in the global configs dir (which would load
# unconditionally and hurt c=1). Instead we point VLLM_TUNED_CONFIG_FOLDER at it
# only when LAUNCH_TUNED_MOE=1, mirroring the concurrency-gated torch.compile
# policy. See BENCH_MOE_TUNE_W8A16_E256N256_2026_06.md.
if [[ "${LAUNCH_TUNED_MOE:-0}" == "1" ]]; then
  export VLLM_TUNED_CONFIG_FOLDER="$(dirname "$(readlink -f "$0")")/moe_configs_w8a16"
  echo "[launch_compile_$cell_id] LAUNCH_TUNED_MOE=1: using tuned MoE configs" \
       "from $VLLM_TUNED_CONFIG_FOLDER (win at c>=2, regresses c=1)"
fi

# Inductor compile cache — amortizes the ~40-73 s warmup across server restarts.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/root/.cache/vllm}"

if [[ "$NO_COMPILE" == "1" ]]; then
  COMPILATION_CONFIG='{"cudagraph_mode":"FULL_DECODE_ONLY"}'
else
  COMPILATION_CONFIG='{"mode":3,"cudagraph_mode":"FULL_AND_PIECEWISE"}'
fi

# Inductor raises peak memory; default to 0.90 (vs 0.93 non-compile) for KV
# headroom. Override with LAUNCH_GPU_MEM_UTIL.
GPU_MEM_UTIL="${LAUNCH_GPU_MEM_UTIL:-0.90}"

LM_ONLY_FLAG=()
if [[ -n "$lm_only" ]]; then
  LM_ONLY_FLAG=(--language-model-only)
fi

OUT_ROOT="${OUT_ROOT:-/root/fp16-bench/compile_launch_smoke}"
SERVER_LOG_DIR="${SERVER_LOG_DIR:-$OUT_ROOT/${cell_id}}"
mkdir -p "$SERVER_LOG_DIR"
LOG="$SERVER_LOG_DIR/server.log"

# ---------------------------------------------------------------------------
# CLI arg parsing (no args → mode=--check, port=8000)
# ---------------------------------------------------------------------------
PORT=8000
mode=--check
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="$2"; shift 2 ;;
    --serve-only|--check)
      mode="$1"; shift ;;
    *)
      echo "[launch_compile_$cell_id] unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------
stop_server() {
  if [[ -z "${HIP_VISIBLE_DEVICES:-}" && "${PORT:-8000}" == "8000" ]]; then
    pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
    pkill -9 -f 'VLLM::' 2>/dev/null || true
    sleep 2
  elif [[ -n "${SERVER_PID:-}" ]]; then
    kill -9 "$SERVER_PID" 2>/dev/null || true
    sleep 1
  fi
}

trap stop_server EXIT
stop_server

# ---------------------------------------------------------------------------
# Start server
# ---------------------------------------------------------------------------
/opt/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server \
    --model "$model_path" \
    --dtype float16 \
    --tensor-parallel-size "$tp" \
    --max-model-len "$max_model_len" \
    --block-size 32 \
    --enable-prefix-caching \
    --trust-remote-code \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --compilation-config "$COMPILATION_CONFIG" \
    --port "$PORT" \
    --disable-custom-all-reduce "${LM_ONLY_FLAG[@]}" > "$LOG" 2>&1 &
SERVER_PID=$!
echo "[launch_compile_$cell_id] server PID=$SERVER_PID; log=$LOG"

# Compile warmup makes startup slow — default 900 s (load + Inductor + capture).
HEALTH_WAIT_SECS=${LAUNCH_HEALTH_WAIT_SECS:-900}
poll_count=$(( HEALTH_WAIT_SECS / 5 ))
for i in $(seq 1 "$poll_count"); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
    echo "[launch_compile_$cell_id] healthcheck OK after ${i} x5 s"
    break
  fi
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[launch_compile_$cell_id] FATAL: server exited; tail:"
    tail -n 60 "$LOG"
    exit 2
  fi
  sleep 5
done
if ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; then
  echo "[launch_compile_$cell_id] FATAL: server did not become healthy in ${HEALTH_WAIT_SECS} s"
  tail -n 60 "$LOG"
  exit 2
fi

if [[ "$mode" == "--serve-only" ]]; then
  trap - EXIT
  echo "[launch_compile_$cell_id] server ready; pid=$SERVER_PID (kill it manually)."
  exit 0
fi

# ---------------------------------------------------------------------------
# Canonical bench (synthetic random, NUM_PROMPTS=200, seed=42)
# ---------------------------------------------------------------------------
RAW_DIR="$OUT_ROOT/${cell_id}/bench_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RAW_DIR"

/opt/vllm-env/bin/python3 -m vllm.entrypoints.cli.main bench serve \
    --model "$model_path" \
    --base-url "http://127.0.0.1:${PORT}" \
    --num-prompts 200 \
    --request-rate inf \
    --max-concurrency "$conc" \
    --seed 42 \
    --save-result \
    --result-dir "$RAW_DIR" \
    --result-filename raw.json \
    --trust-remote-code \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 \
    --metadata cell_id=${cell_id} workload=synthetic tp=${tp} concurrency=${conc} \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --ignore-eos

result_json="$RAW_DIR/raw.json"
if [[ ! -f "$result_json" ]]; then
  echo "[launch_compile_$cell_id] FATAL: bench produced no result.json"; exit 2
fi

actual_tput=$(/opt/vllm-env/bin/python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_throughput"])' "$result_json")

echo "[launch_compile_$cell_id] recorded compile-arm reference output_throughput = ${ref_tput:-<none>} tok/s"
echo "[launch_compile_$cell_id] this run                output_throughput        = $actual_tput tok/s"

if [[ -z "$ref_tput" || "$ref_tput" == "None" ]]; then
  echo "[launch_compile_$cell_id] no recorded reference — skipping ±5 % gate"
  exit 0
fi

verdict=$(/opt/vllm-env/bin/python3 -c "
import sys
ref=float(sys.argv[1]); act=float(sys.argv[2])
delta=(act-ref)/ref*100
print(f'delta={delta:+.2f}%')
sys.exit(0 if abs(delta)<=5.0 else 1)
" "$ref_tput" "$actual_tput")
rc=$?
echo "[launch_compile_$cell_id] $verdict (±5 % gate)"
exit $rc
