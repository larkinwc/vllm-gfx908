#!/usr/bin/env bash
# =============================================================================
# /root/bench-int8-w4a16/baseline/run_baseline.sh -- Locked M1 benchmark harness
# =============================================================================
#
# Runs the 24-cell baseline grid for the MI100/gfx908 INT8/W4A16 mission:
#   {model in W8A8, W4A16} x {TP in 1, 4} x {concurrency in 1, 2, 4}
#                       x {workload in synthetic, coding}
#
# Per cell we:
#   1. Stop any running vLLM (manifest-style stop).
#   2. Start the matching service from services.yaml.
#   3. Wait up to 1800 s for /health.
#   4. Run `vllm bench serve` with --num-prompts 200 --seed 42
#      and --max-concurrency = target concurrency (request-rate matched).
#   5. Post-process the raw JSON into a schema-conformant cell file.
#   6. Stop the service.
#
# Reproducibility canary: run this script twice, then diff the two
# harness_manifest.json files. TP=1 c=1 W8A8 throughput must reproduce
# within +/-5%.
#
# Outputs (in $BASELINE_ROOT):
#   harness_manifest.json   -- env, versions, per-cell launch commands
#   synthetic/<id>.json     -- 12 cells, schema-validated
#   coding/<id>.json        -- 12 cells, schema-validated
#   run_baseline_<ts>.log   -- this script's tee'd output
#
# Reruns: idempotent. Existing per-cell JSON files are overwritten.
#
# NEVER pushes to git (per VAL-CROSS-006).
# =============================================================================
set -uo pipefail

# ---------- Paths ----------
REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/loose-rats-clean-phm2k
BASELINE_ROOT=/root/bench-int8-w4a16/m3-f1-baseline
SYN_DIR=$BASELINE_ROOT/synthetic
COD_DIR=$BASELINE_ROOT/coding
PY=/opt/vllm-env/bin/python3
DATASET=/root/bench-int8-w4a16/datasets/coding_agent.jsonl
SERVICES_YAML=/root/.factory/missions/f74e8645-0bfb-462a-963d-84d7196b0f6c/services.yaml

mkdir -p "$SYN_DIR" "$COD_DIR" "$BASELINE_ROOT/profile"

TS=$(date -u +%Y%m%dT%H%M%SZ)
RUN_LOG=$BASELINE_ROOT/run_baseline_${TS}.log
exec > >(tee -a "$RUN_LOG") 2>&1

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

# ---------- Common env (mirrors AGENTS.md required vars) ----------
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PATH=/opt/rocm/core-7.12/bin:${PATH:-/usr/bin:/bin}
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export VLLM_MI100_DISABLE_CUSTOM_AR=0
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1

# ---------- Versions ----------
VLLM_COMMIT=$(cd "$REPO" && git rev-parse HEAD)
ROCM_VERSION=7.12
TORCH_VERSION=$($PY -c "import torch; print(torch.__version__)" 2>/dev/null || echo unknown)
TRITON_VERSION=$($PY -c "import triton; print(triton.__version__)" 2>/dev/null || echo unknown)
VLLM_VERSION=$(cd "$REPO" && $PY -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo unknown)
GPU_TOPO=$(rocm-smi --showtopo 2>/dev/null || true)

log "vLLM commit:    $VLLM_COMMIT ($VLLM_VERSION)"
log "ROCm:           $ROCM_VERSION   torch: $TORCH_VERSION   triton: $TRITON_VERSION"
log "Output root:    $BASELINE_ROOT"

# ---------- Service start helpers ----------
kill_orphans() {
  pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
  pkill -9 -f 'VLLM::' 2>/dev/null || true
  sleep 2
}

start_service() {
  # $1 service id (vllm-w8a8-tp1 | vllm-w8a8-tp4 | vllm-w4a16-tp1 | vllm-w4a16-tp4)
  local svc=$1 model tp eager extra cuda_visible
  case "$svc" in
    vllm-w8a8-tp1)   model=/models/Qwen3.5-9B-w8a8;   tp=1; eager=""; extra=""; cuda_visible=0 ;;
    vllm-w8a8-tp4)   model=/models/Qwen3.5-9B-w8a8;   tp=4; eager=""; extra="--disable-custom-all-reduce"; cuda_visible="" ;;
    vllm-w4a16-tp1)  model=/models/Qwen3.5-9B-w4a16;  tp=1; eager=""; extra=""; cuda_visible=0 ;;
    vllm-w4a16-tp4)  model=/models/Qwen3.5-9B-w4a16;  tp=4; eager=""; extra="--disable-custom-all-reduce"; cuda_visible="" ;;
    *) echo "unknown service $svc" >&2; return 2 ;;
  esac

  kill_orphans
  if [ -n "$cuda_visible" ]; then
    export CUDA_VISIBLE_DEVICES="$cuda_visible"
  else
    unset CUDA_VISIBLE_DEVICES || true
  fi

  if [[ "$svc" == vllm-w4a16-tp4 ]]; then
    export VLLM_MI100_DISABLE_CUSTOM_AR=1
    # M1 pretune: pre-populated Triton cache lets the 4 workers skip JIT
    # compile during profile_run, recovering FULL_DECODE_ONLY graphs.
    export TRITON_CACHE_DIR=/root/bench-int8-w4a16/baseline/triton_cache_w4a16
    export NCCL_TIMEOUT_HOURS=2
    export TORCH_NCCL_BLOCKING_WAIT=1
    export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
  elif [[ "$svc" == vllm-w8a8-tp4 ]]; then
    export VLLM_MI100_DISABLE_CUSTOM_AR=1
    unset TRITON_CACHE_DIR || true
    unset NCCL_TIMEOUT_HOURS TORCH_NCCL_BLOCKING_WAIT TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC || true
  else
    export VLLM_MI100_DISABLE_CUSTOM_AR=0
    unset TRITON_CACHE_DIR || true
    unset NCCL_TIMEOUT_HOURS TORCH_NCCL_BLOCKING_WAIT TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC || true
  fi

  local svc_log=$BASELINE_ROOT/server_${svc}_${TS}.log
  log "starting $svc (model=$model tp=$tp) -> $svc_log"
  cd "$REPO"
  nohup $PY -m vllm.entrypoints.openai.api_server \
    --model "$model" \
    --dtype float16 \
    --tensor-parallel-size "$tp" \
    --max-model-len 32768 \
    --block-size 32 \
    --enable-prefix-caching \
    --language-model-only \
    --trust-remote-code \
    --gpu-memory-utilization 0.93 \
    --port 8000 \
    $extra $eager \
    > "$svc_log" 2>&1 &
  echo $! > "$BASELINE_ROOT/${svc}.pid"
  for i in $(seq 1 360); do
    if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then
      log "  $svc healthy after ${i}*5s"
      return 0
    fi
    if ! kill -0 "$(cat "$BASELINE_ROOT/${svc}.pid")" 2>/dev/null; then
      log "  $svc PID died -- last 30 lines of log:"
      tail -n 30 "$svc_log"
      return 1
    fi
    sleep 5
  done
  log "  healthcheck timeout for $svc"
  tail -n 30 "$svc_log"
  return 1
}

stop_service() {
  kill_orphans
}

# ---------- Per-cell run ----------
# Args: model_short svc tp conc workload
run_cell() {
  local model_short=$1   # w8a8 | w4a16
  local svc=$2           # service id (already started)
  local tp=$3
  local conc=$4
  local workload=$5      # synthetic | coding

  local cell_id="${model_short}_tp${tp}_c${conc}"
  local quant
  if [[ "$model_short" == w8a8 ]]; then quant="w8a8-int8"; model=/models/Qwen3.5-9B-w8a8
  else quant="w4a16"; model=/models/Qwen3.5-9B-w4a16; fi

  local out_dir
  if [[ "$workload" == synthetic ]]; then out_dir="$SYN_DIR"
  else out_dir="$COD_DIR"; fi
  mkdir -p "$out_dir"

  local raw_dir
  raw_dir=$(mktemp -d "$BASELINE_ROOT/raw_${cell_id}_${workload}_XXXXXX")
  local out_file="$out_dir/${cell_id}.json"
  local env_file="$BASELINE_ROOT/env_${cell_id}_${workload}.json"

  log ">> CELL ${model_short} tp=${tp} c=${conc} ${workload} -> ${out_file}"

  # Persist env snapshot for the schema row.
  $PY - <<EOF > "$env_file"
import json, os
keep = [
    "LD_LIBRARY_PATH","ROCM_PATH","PATH","PYTORCH_ROCM_ARCH",
    "VLLM_ROCM_USE_AITER","VLLM_ROCM_USE_SKINNY_GEMM",
    "VLLM_MI100_DISABLE_CUSTOM_AR","TORCH_COMPILE_DISABLE",
    "TRITON_CACHE_DIR","NCCL_TIMEOUT_HOURS","TORCH_NCCL_BLOCKING_WAIT",
    "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC","HF_HUB_OFFLINE",
    "CUDA_VISIBLE_DEVICES",
]
print(json.dumps({k: os.environ.get(k, "") for k in keep}, indent=2))
EOF

  # Build the bench-serve command.
  # Drive at request-rate=inf so the Poisson scheduler does not throttle
  # below the target in-flight depth -- --max-concurrency is the actual
  # concurrency knob. This matches the AGENTS.md "emulate concurrency
  # via observed in-flight depth" guidance and gives a saturation-style
  # measurement for each c=1/2/4 cell.
  local rate=inf
  local n_prompts=${NUM_PROMPTS:-200}
  local bench_args=(
    --model "$model"
    --base-url "http://127.0.0.1:8000"
    --num-prompts "$n_prompts"
    --request-rate "$rate"
    --max-concurrency "$conc"
    --seed 42
    --save-result
    --result-dir "$raw_dir"
    --result-filename "raw.json"
    --trust-remote-code
    --percentile-metrics "ttft,tpot,itl,e2el"
    --metric-percentiles "50,90,99"
    --metadata "cell_id=${cell_id}" "workload=${workload}" "tp=${tp}" "concurrency=${conc}" "num_prompts=${n_prompts}"
  )
  if [[ "$workload" == synthetic ]]; then
    bench_args+=(
      --dataset-name random
      --random-input-len 1024
      --random-output-len 256
      --ignore-eos
    )
  else
    bench_args+=(
      --dataset-name custom
      --dataset-path "$DATASET"
      --custom-output-len 256
      --skip-chat-template
    )
  fi

  local launch_command="$PY -m vllm.entrypoints.cli.main bench serve ${bench_args[*]}"

  log "  launch: $launch_command"
  set +e
  cd "$REPO"
  $PY -m vllm.entrypoints.cli.main bench serve "${bench_args[@]}"
  local rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    log "  ERROR vllm bench serve exit=$rc"
  fi

  # Find the raw JSON. vllm bench serve creates a timestamped file.
  local raw_json
  raw_json=$(ls -1 "$raw_dir"/*.json 2>/dev/null | head -1 || true)
  if [[ -z "${raw_json:-}" ]]; then
    log "  WARNING raw JSON missing under $raw_dir; emitting placeholder"
    # Emit a placeholder schema-conformant file marking this cell FAILED
    # so schema-validate downstream still produces a clear signal.
    cat > "$raw_dir/raw.json" <<EOF
{"output_throughput": 0, "request_throughput": 0, "mean_ttft_ms": 0, "median_ttft_ms": 0, "p99_ttft_ms": 0, "mean_tpot_ms": 0, "median_tpot_ms": 0, "p99_tpot_ms": 0, "FAILED": true, "exit_code": ${rc}}
EOF
    raw_json="$raw_dir/raw.json"
  fi

  cd "$REPO"
  $PY scripts/postprocess_bench_result.py \
    --raw "$raw_json" \
    --cell "${cell_id}_${workload}" \
    --model "$model" \
    --quant "$quant" \
    --tp "$tp" \
    --concurrency "$conc" \
    --request-rate "$rate" \
    --workload "$workload" \
    --num-prompts "$n_prompts" \
    --launch-command "$launch_command" \
    --env-file "$env_file" \
    --kernel-backend "stock" \
    --out "$out_file"
}

# ---------- Manifest ----------
write_manifest() {
  local manifest=$BASELINE_ROOT/harness_manifest.json
  $PY - "$manifest" "$VLLM_COMMIT" "$VLLM_VERSION" "$TORCH_VERSION" \
       "$TRITON_VERSION" "$ROCM_VERSION" "$DATASET" "$SERVICES_YAML" <<'EOF'
import json, os, subprocess, sys
manifest, vllm_sha, vllm_ver, torch_v, triton_v, rocm_v, dataset, services_yaml = sys.argv[1:9]

def shell(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""

env_keys = [
    "LD_LIBRARY_PATH","ROCM_PATH","PATH","PYTORCH_ROCM_ARCH",
    "VLLM_ROCM_USE_AITER","VLLM_ROCM_USE_SKINNY_GEMM",
    "VLLM_MI100_DISABLE_CUSTOM_AR","TORCH_COMPILE_DISABLE",
    "HF_HUB_OFFLINE",
]

cells = []
for ms in ("w8a8","w4a16"):
    for tp in (1,4):
        for c in (1,2,4):
            for wl in ("synthetic","coding"):
                cell = f"{ms}_tp{tp}_c{c}_{wl}"
                cells.append({"id": cell, "model": ms, "tp": tp, "concurrency": c, "workload": wl})

obj = {
    "vllm_commit": vllm_sha,
    "vllm_version": vllm_ver,
    "rocm_version": rocm_v,
    "torch_version": torch_v,
    "triton_version": triton_v,
    "gpu_topo": shell("rocm-smi --showtopo"),
    "rocm_smi_product": shell("rocm-smi --showproductname"),
    "rocm_smi_temp": shell("rocm-smi --showtemp"),
    "env": {k: os.environ.get(k, "") for k in env_keys},
    "dataset_path": dataset,
    "dataset_sha256": shell(f"sha256sum {dataset} | awk '{{print $1}}'").strip(),
    "services_yaml": services_yaml,
    "harness_version": "1.0",
    "num_prompts_per_cell": 200,
    "seed": 42,
    "block_size": 32,
    "max_model_len": 32768,
    "cudagraph_mode": "FULL_DECODE_ONLY",
    "language_model_only": True,
    "enable_prefix_caching": True,
    "cells": cells,
    "timestamp": shell("date -u +%Y-%m-%dT%H:%M:%SZ").strip(),
    "host": shell("hostname").strip(),
}
with open(manifest, "w") as f:
    json.dump(obj, f, indent=2)
print(f"wrote manifest: {manifest}")
EOF
}

# ---------- Main grid execution ----------
write_manifest

# We want all 24 cells. Service is shared across the 6 (concurrency x workload)
# cells of one (model, tp), so start once per service.
SERVICES=(
  "w8a8:vllm-w8a8-tp1:1"
  "w8a8:vllm-w8a8-tp4:4"
  "w4a16:vllm-w4a16-tp1:1"
  "w4a16:vllm-w4a16-tp4:4"
)

# Allow caller to restrict the run via $CELLS_FILTER (regex).
CELLS_FILTER=${CELLS_FILTER:-".*"}

for triple in "${SERVICES[@]}"; do
  IFS=":" read -r model_short svc tp <<<"$triple"

  # Skip whole service if no matching cell
  any_match=0
  for conc in 1 2 4; do
    for wl in synthetic coding; do
      cell="${model_short}_tp${tp}_c${conc}_${wl}"
      if [[ "$cell" =~ $CELLS_FILTER ]]; then any_match=1; break; fi
    done
    [[ $any_match -eq 1 ]] && break
  done
  if [[ $any_match -eq 0 ]]; then
    log "skipping $svc (no cells match filter)"
    continue
  fi

  if ! start_service "$svc"; then
    log "FAILED to start $svc -- skipping its 6 cells"
    stop_service
    continue
  fi

  for conc in 1 2 4; do
    for wl in synthetic coding; do
      cell="${model_short}_tp${tp}_c${conc}_${wl}"
      if [[ ! "$cell" =~ $CELLS_FILTER ]]; then
        log "  skip $cell (filter)"
        continue
      fi
      run_cell "$model_short" "$svc" "$tp" "$conc" "$wl" || log "  cell $cell failed (continuing)"
    done
  done

  stop_service
done

# Re-write manifest at end so timestamp is post-run.
write_manifest

log "=== run_baseline.sh done. Validate with: ==="
log "  $PY $REPO/scripts/validate_results_schema.py $BASELINE_ROOT/"
exit 0
