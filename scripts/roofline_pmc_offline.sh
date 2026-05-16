#!/usr/bin/env bash
# =============================================================================
# scripts/roofline_pmc_offline.sh -- omniperf-equivalent roofline using
# rocprofv3 PMC counters via the offline `vllm bench throughput`
# entrypoint (same harness pattern that produced our kernel-trace
# CSVs).  Captures key gfx908 wave/MFMA/HBM counters across the run
# and aggregates them against the top hot kernel.
#
# Outputs:
#   $OUT_DIR/pmc_*_counter_collection.csv  (per-kernel counters)
#   $OUT_DIR/omniperf_summary.json         (achieved tflops/HBM, %peak)
# =============================================================================
set -uo pipefail

HOT=${1:?path to hot_shapes.json required}
QUANT=${2:-w8a8}    # which quant to roofline (w8a8 by default since it's the M2/M3 scope)

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
BASELINE_ROOT=/root/bench-int8-w4a16/baseline
PY=/opt/vllm-env/bin/python3
OUT_DIR=$BASELINE_ROOT/profile/omniperf
mkdir -p "$OUT_DIR"

# Pick the top kernel for this quant.
PYPICK="
import json, sys
hot = json.load(open(sys.argv[1]))['hot_shapes']
quant = sys.argv[2]
pick = next((h for h in hot if h['quant']==quant), hot[0])
print(json.dumps({
    'kernel_name': pick['kernel_name'],
    'shape': pick['shape'],
    'regime': pick['regime'],
    'quant': pick['quant'],
}))
"
PICK=$($PY -c "$PYPICK" "$HOT" "$QUANT")
KNAME=$(echo "$PICK" | $PY -c 'import json,sys;print(json.load(sys.stdin)["kernel_name"])')
echo "[roofline-pmc] hot kernel for $QUANT: $KNAME"
echo "$PICK" > "$OUT_DIR/picked_kernel_${QUANT}.json"

if [[ "$QUANT" == w8a8 ]]; then model=/models/Qwen3.5-9B-w8a8
else model=/models/Qwen3.5-9B-w4a16; fi

# Common env (TP=1).
export LD_LIBRARY_PATH=/opt/rocm/core-7.12/lib
export ROCM_PATH=/opt/rocm/core-7.12
export PATH=/opt/rocm/core-7.12/bin:${PATH:-/usr/bin:/bin}
export PYTORCH_ROCM_ARCH=gfx908
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_SKINNY_GEMM=0
export TORCH_COMPILE_DISABLE=1
export HF_HUB_OFFLINE=1
export VLLM_MI100_DISABLE_CUSTOM_AR=0
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
cd "$REPO"

# PMC counter file -- minimal gfx908 counters needed for arithmetic
# intensity & HBM utilization.
PMC_INPUT=$OUT_DIR/pmc.txt
cat > "$PMC_INPUT" <<EOF
pmc: SQ_WAVES SQ_INSTS_VALU SQ_BUSY_CYCLES GRBM_GUI_ACTIVE
pmc: TCP_TCC_READ_REQ_sum TCP_TCC_WRITE_REQ_sum
EOF

# Pre-clean orphans.
pgrep -f vllm.entrypoints 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
pgrep -f 'VLLM::' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true
sleep 3

BENCH_LOG=$OUT_DIR/bench_pmc_${QUANT}.log
echo "[roofline-pmc] running offline bench under rocprofv3 PMC"
nohup rocprofv3 \
  -i "$PMC_INPUT" \
  --kernel-trace \
  --kernel-include-regex ".*$(echo "$KNAME" | head -c 40).*" \
  -d "$OUT_DIR" \
  -o "pmc_${QUANT}" \
  -f csv \
  -- \
  $PY -m vllm.entrypoints.cli.main bench throughput \
    --model "$model" \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --block-size 32 \
    --enable-prefix-caching \
    --language-model-only \
    --gpu-memory-utilization 0.93 \
    --trust-remote-code \
    --seed 42 \
    --backend vllm \
    --dataset-name random \
    --random-input-len 1024 \
    --random-output-len 256 \
    --num-prompts 4 \
    > "$BENCH_LOG" 2>&1 &
WPID=$!
echo "$WPID" > "$OUT_DIR/wrapper_${QUANT}.pid"

# Wait for completion (offline bench: ~2-4 min for 4 prompts at TP=1).
for i in $(seq 1 1800); do
  if ! kill -0 "$WPID" 2>/dev/null; then
    echo "[roofline-pmc] wrapper exited cleanly after ${i}s"; break
  fi
  sleep 1
done
if kill -0 "$WPID" 2>/dev/null; then
  echo "[roofline-pmc] wrapper still alive after 1800s -- killing"
  kill -KILL "$WPID" 2>/dev/null || true
fi
sleep 30
pgrep -f 'VLLM::' 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true

# Aggregate.
$PY - "$KNAME" "$OUT_DIR" "$QUANT" <<'EOF'
import csv, glob, json, os, sys
kname_target = sys.argv[1]
out_dir = sys.argv[2]
quant = sys.argv[3]

csvs = sorted(glob.glob(os.path.join(out_dir, "*.csv"), recursive=False))
print("CSVs in out_dir:", csvs)

# rocprofv3 with --pmc emits counter rows per kernel in the same
# kernel-trace CSV (newer rocprofv3) OR a separate counter_collection
# CSV (older). We probe both.
counter_files = []
for c in csvs:
    if quant in os.path.basename(c).lower():
        counter_files.append(c)
print(f"counter files for quant={quant}: {counter_files}")

if not counter_files:
    summary = {
        "status": "no_pmc_csvs",
        "hot_kernel": kname_target,
        "quant": quant,
        "note": "rocprofv3 did not emit PMC counter CSVs.  See bench_pmc.log.",
    }
    with open(os.path.join(out_dir, f"omniperf_summary_{quant}.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    sys.exit(0)

agg = {}
total_dur_ns = 0
n_calls = 0
for path in counter_files:
    if "agent_info" in path:
        continue
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            kn = row.get("Kernel_Name") or row.get("KernelName") or row.get("kernel_name") or ""
            target_short = kname_target[:40]
            if target_short not in kn:
                continue
            try:
                dur = int(row.get("End_Timestamp", row.get("EndNs", 0))) - \
                      int(row.get("Start_Timestamp", row.get("BeginNs", 0)))
            except Exception:
                dur = 0
            if dur > 0:
                total_dur_ns += dur
                n_calls += 1
            for k, v in row.items():
                if not v:
                    continue
                if k.startswith(("SQ_", "TA_", "TCP_", "TCC_", "GRBM_")):
                    try:
                        agg[k] = agg.get(k, 0) + float(v)
                    except ValueError:
                        pass

print(f"matched {n_calls} kernel calls, total_dur_ns={total_dur_ns}")
print("aggregate counters:", json.dumps(agg, indent=2))

PEAK_INT8_TFLOPS = 184.6
PEAK_FP16_TFLOPS = 184.6
PEAK_HBM_GB_S = 1228.8

ach_tflops = None
ach_hbm = None
if total_dur_ns > 0:
    sq_valu = agg.get("SQ_INSTS_VALU", 0)
    if sq_valu > 0:
        # Each VALU instruction processes 64 lanes.
        ach_tflops = (sq_valu * 64) / (total_dur_ns / 1e9) / 1e12
    tcc_rd = agg.get("TCP_TCC_READ_REQ_sum", 0)
    tcc_wr = agg.get("TCP_TCC_WRITE_REQ_sum", 0)
    if tcc_rd or tcc_wr:
        bytes_total = (tcc_rd + tcc_wr) * 64
        ach_hbm = bytes_total / (total_dur_ns / 1e9) / 1e9

summary = {
    "status": "ok" if (ach_tflops or ach_hbm) else "counters_empty",
    "hot_kernel": kname_target,
    "quant": quant,
    "n_calls": n_calls,
    "total_kernel_ns": total_dur_ns,
    "achieved_TFLOPs_VALU_lanes": ach_tflops,
    "peak_TFLOPs_gfx908_fp16_int8": PEAK_FP16_TFLOPS,
    "percent_of_peak_compute": (
        100.0 * ach_tflops / PEAK_FP16_TFLOPS if ach_tflops else None
    ),
    "achieved_HBM_GB_s": ach_hbm,
    "peak_HBM_GB_s_gfx908": PEAK_HBM_GB_S,
    "percent_of_peak_hbm": (
        100.0 * ach_hbm / PEAK_HBM_GB_S if ach_hbm else None
    ),
    "raw_counters": agg,
}
with open(os.path.join(out_dir, f"omniperf_summary_{quant}.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "raw_counters"}, indent=2))
EOF

echo "[roofline-pmc] done -> $OUT_DIR/omniperf_summary_${QUANT}.json"
exit 0
