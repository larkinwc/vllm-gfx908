#!/bin/bash
# Aggregate FP16 TFLOPS + HBM bandwidth across gfx900 GPUs.
# Phase 1: single GPU (GPU0) alone -> uncontended peak.
# Phase 2: all GPUs in a socket concurrently -> realistic aggregate.
# Phase 3: all 16 GPUs concurrently -> full-box aggregate.
set -u
cd ~/gpubench && source venv/bin/activate
PY=/tmp/gpu_peak.py
OUT=~/gpubench/peak_results
mkdir -p "$OUT"
DTYPE="${1:-fp16}"

run_set () {
  local label="$1"; shift
  local gpus=("$@")
  echo "===== $label : GPUs ${gpus[*]} ====="
  pids=()
  for g in "${gpus[@]}"; do
    HIP_VISIBLE_DEVICES=$g python "$PY" --dtype "$DTYPE" --label "$label" \
      > "$OUT/${label}_gpu${g}.json" 2>"$OUT/${label}_gpu${g}.err" &
    pids+=($!)
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  # aggregate
  python - "$OUT" "$label" "${gpus[@]}" <<'PYEOF'
import json, sys
out, label = sys.argv[1], sys.argv[2]
gpus = sys.argv[3:]
tf=bw=0.0; rows=[]
for g in gpus:
    try:
        d=json.load(open(f"{out}/{label}_gpu{g}.json"))
        tf+=d["tflops"]; bw+=d["bw_gbps"]; rows.append((g,d["tflops"],d["bw_gbps"]))
    except Exception as e:
        print(f"  GPU{g}: FAILED ({e})")
for g,t,b in rows:
    print(f"  GPU{g:>2}: {t:7.2f} TFLOPS  {b:7.1f} GB/s")
print(f"  AGG  : {tf:7.2f} TFLOPS  {bw:7.1f} GB/s  (n={len(rows)})")
PYEOF
  echo
}

echo "### dtype=$DTYPE  $(date) ###"
run_set single 0
run_set socket0 0 1 2 3 4 5 6 7
run_set allgpu $(seq 0 15)
echo "### done $(date) ###"
