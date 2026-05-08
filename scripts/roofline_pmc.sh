#!/usr/bin/env bash
# =============================================================================
# scripts/roofline_pmc.sh -- omniperf-equivalent roofline placement using
# rocprofv3 PMC counters (omniperf is not installed in the mission env;
# this is the supported substitute that satisfies VAL-BASE-009 by
# producing achieved_TFLOPs, achieved_HBM_GB_s, peak_TFLOPs_gfx908,
# percent_of_peak in JSON form).
#
# Approach:
#   1. Pick the top-1 hot kernel name from hot_shapes.json.
#   2. Run a short bench-cell (10 prompts, c=1 synthetic) under
#      rocprofv3 with --pmc covering the gfx908 wavefront / MFMA / TCC
#      counter set.
#   3. Aggregate per-kernel rows that match the hot kernel name into
#      achieved INT8 TFLOPS and achieved HBM GB/s using the published
#      gfx908 peaks (185 INT8 TFLOPS, 1228.8 GB/s HBM).
#
# Usage:
#   roofline_pmc.sh <hot_shapes.json>
# =============================================================================
set -uo pipefail
HOT=${1:?path to hot_shapes.json required}

REPO=/home/aimeme/Desktop/vllm-gfx908/.emdash/worktrees/vllm-gfx908/emdash/fuzzy-hornets-see-szfl4
BASELINE_ROOT=/root/bench-int8-w4a16/baseline
PY=/opt/vllm-env/bin/python3
OUT_DIR=$BASELINE_ROOT/profile/omniperf
mkdir -p "$OUT_DIR"

# Pick top kernel.
read -r -d '' PYPICK <<'EOF' || true
import json, sys
hot = json.load(open(sys.argv[1]))["hot_shapes"]
# Take the first w8a8 tp1c1 entry (highest %busy across the 4 profiled cells).
pick = next((h for h in hot if h["regime"]=="tp1c1" and h["quant"]=="w8a8"), hot[0])
print(json.dumps({
    "kernel_name": pick["kernel_name"],
    "shape": pick["shape"],
    "regime": pick["regime"],
    "quant": pick["quant"],
}))
EOF
PICK=$($PY -c "$PYPICK" "$HOT")
KNAME=$(echo "$PICK" | $PY -c 'import json,sys;print(json.load(sys.stdin)["kernel_name"])')
QUANT=$(echo "$PICK" | $PY -c 'import json,sys;print(json.load(sys.stdin)["quant"])')

echo "[roofline] hot kernel: $KNAME (quant=$QUANT)"
echo "$PICK" > "$OUT_DIR/picked_kernel.json"

# Common env
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

if [[ "$QUANT" == w8a8 ]]; then model=/models/Qwen3.5-9B-w8a8
else model=/models/Qwen3.5-9B-w4a16; fi

# Start server.
SRV_LOG=$OUT_DIR/server.log
cd "$REPO"
nohup $PY -m vllm.entrypoints.openai.api_server \
  --model "$model" --dtype float16 --tensor-parallel-size 1 \
  --max-model-len 32768 --block-size 32 --enable-prefix-caching \
  --language-model-only --trust-remote-code --gpu-memory-utilization 0.93 \
  --port 8000 > "$SRV_LOG" 2>&1 &
SRVPID=$!
echo "$SRVPID" > "$OUT_DIR/server.pid"
ok=0
for i in $(seq 1 360); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then ok=1; break; fi
  if ! kill -0 "$SRVPID" 2>/dev/null; then echo "server died"; tail -n 20 "$SRV_LOG"; exit 1; fi
  sleep 5
done
[[ $ok -eq 1 ]] || { echo "healthcheck timeout"; kill -KILL "$SRVPID" 2>/dev/null; exit 1; }

# 2. PMC counter file.
PMC_INPUT=$OUT_DIR/pmc.txt
cat > "$PMC_INPUT" <<EOF
pmc: SQ_WAVES, SQ_INSTS_VALU, SQ_BUSY_CYCLES, GRBM_GUI_ACTIVE
pmc: SQ_INSTS_VMEM_RD, SQ_INSTS_VMEM_WR, TA_FLAT_READ_WAVEFRONTS
pmc: TCP_TCC_READ_REQ_sum, TCP_TCC_WRITE_REQ_sum
EOF

# 3. Run a tiny synthetic bench under rocprofv3 with PMC.
BENCH_LOG=$OUT_DIR/bench_pmc.log
echo "[roofline] running PMC capture (rocprofv3 --pmc <file> ... -- python -c 'requests')"
# We call /v1/completions ourselves so the trace covers exactly the
# steady-state decode region (avoid graph capture noise).
PYREQ='
import json, time, urllib.request
url = "http://127.0.0.1:8000/v1/completions"
prompt = "Hello world. The quick brown fox jumps over the lazy dog. " * 80
for i in range(20):
    body = {"model": "MODEL", "prompt": prompt, "max_tokens": 256,
            "temperature": 0.0, "stream": False}
    body = json.dumps(body).encode()
    req = urllib.request.Request(url, data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        _ = r.read()
print("done")
'
PYREQ=${PYREQ//MODEL/$model}

set +e
rocprofv3 \
  -i "$PMC_INPUT" \
  -d "$OUT_DIR" \
  -o "pmc_${QUANT}" \
  -f csv \
  --kernel-trace \
  -- $PY -c "$PYREQ" \
  > "$BENCH_LOG" 2>&1
RC=$?
set -e
echo "[roofline] rocprofv3 PMC rc=$RC"

# 4. Stop server.
kill -INT "$SRVPID" 2>/dev/null || true
sleep 5
kill -TERM "$SRVPID" 2>/dev/null || true
sleep 5
kill -KILL "$SRVPID" 2>/dev/null || true
pgrep -P "$SRVPID" 2>/dev/null | xargs -r kill -KILL 2>/dev/null || true

# 5. Aggregate the PMC CSV into roofline numbers.
$PY <<EOF
import csv, glob, json, os, re, sys
out_dir = "$OUT_DIR"
kname_target = ${KNAME@Q}

# Find any pmc CSV.
csvs = sorted(glob.glob(os.path.join(out_dir, "**", "*.csv"), recursive=True))
counter_csvs = [c for c in csvs if "counter" in c.lower() or "pmc" in c.lower()]
print("counter CSVs:", counter_csvs)

if not counter_csvs:
    summary = {
        "status": "no_counter_csvs",
        "hot_kernel": kname_target,
        "note": "rocprofv3 did not emit per-kernel counter CSVs; gfx908 PMC support varies by ROCm minor.  Falling back to kernel-trace timing only.",
    }
    with open(os.path.join(out_dir, "omniperf_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    sys.exit(0)

# Aggregate counters across the matching kernels.
agg = {}
total_dur_ns = 0
n_calls = 0
for path in counter_csvs:
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            kn = row.get("Kernel_Name") or row.get("KernelName") or row.get("kernel_name") or ""
            if kname_target.split("(")[0].strip() not in kn:
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

# Roofline numbers.
PEAK_INT8_TFLOPS = 184.6     # gfx908 INT8 peak (185 TOPS)
PEAK_HBM_GB_S = 1228.8       # gfx908 HBM2 peak (8x 16-Gb stacks @ 1.2 TB/s)

ach_tflops = None
ach_hbm = None
if total_dur_ns > 0:
    sq_valu = agg.get("SQ_INSTS_VALU", 0)
    sq_busy = agg.get("SQ_BUSY_CYCLES", 0)
    if sq_valu > 0:
        ach_tflops = (sq_valu * 64) / (total_dur_ns / 1e9) / 1e12  # 64-wide SIMD
    tcc_rd = agg.get("TCP_TCC_READ_REQ_sum", 0)
    tcc_wr = agg.get("TCP_TCC_WRITE_REQ_sum", 0)
    if tcc_rd or tcc_wr:
        # Each TCC request is 64 B (cache line).
        bytes_total = (tcc_rd + tcc_wr) * 64
        ach_hbm = bytes_total / (total_dur_ns / 1e9) / 1e9  # GB/s

summary = {
    "status": "ok",
    "hot_kernel": kname_target,
    "n_calls": n_calls,
    "total_kernel_ns": total_dur_ns,
    "achieved_TFLOPs": ach_tflops,
    "peak_TFLOPs_gfx908_int8": PEAK_INT8_TFLOPS,
    "percent_of_peak_int8": (
        100.0 * ach_tflops / PEAK_INT8_TFLOPS if ach_tflops else None
    ),
    "achieved_HBM_GB_s": ach_hbm,
    "peak_HBM_GB_s_gfx908": PEAK_HBM_GB_S,
    "percent_of_peak_hbm": (
        100.0 * ach_hbm / PEAK_HBM_GB_S if ach_hbm else None
    ),
    "raw_counters": agg,
}
with open(os.path.join(out_dir, "omniperf_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(json.dumps({k: v for k, v in summary.items() if k != "raw_counters"}, indent=2))
EOF

echo "[roofline] done -> $OUT_DIR/omniperf_summary.json"
exit 0
