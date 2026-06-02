#!/bin/bash
# Power-cap efficiency scan on ONE cold GPU.
# For each cap: set it, run sustained FP16 matmul, record TFLOPS + draw + TFLOPS/W.
# Usage: power_scan.sh <gpu_idx> <card_num>
set -u
cd ~/gpubench && source venv/bin/activate
GPU="${1:-1}"
CARD="${2:-2}"
PWR=$(ls /sys/class/drm/card${CARD}/device/hwmon/hwmon*/power1_input 2>/dev/null | head -1)
OUT=~/gpubench/power_scan
mkdir -p "$OUT"
echo "GPU=$GPU card=$CARD power=$PWR"
echo "cap_W  TFLOPS  mean_W  peak_W  TFLOPS/W"
for CAP in 110 100 90 80 70 60 50; do
  sudo -n rocm-smi -d $GPU --setpoweroverdrive $CAP >/dev/null 2>&1
  sleep 2
  HIP_VISIBLE_DEVICES=$GPU python /tmp/power_bench.py \
    --power-path "$PWR" --iters 60 --label "cap${CAP}" \
    > "$OUT/cap${CAP}.json" 2>"$OUT/cap${CAP}.err"
  python - "$OUT/cap${CAP}.json" "$CAP" <<'PYEOF'
import json,sys
d=json.load(open(sys.argv[1]));cap=sys.argv[2]
print(f"{cap:>5}  {d['tflops']:6.2f}  {str(d['mean_w']):>6}  {str(d['peak_w']):>6}  {d['tflops_per_w']}")
PYEOF
  sleep 3   # brief cool between caps
done
# restore default
sudo -n rocm-smi -d $GPU --setpoweroverdrive 110 >/dev/null 2>&1
echo "restored $GPU to 110W"
