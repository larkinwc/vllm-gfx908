#!/bin/bash
# Driver for topo_bench.py: $1=layout $2=graph|eager $3=num_gpus
# Sets HIP_VISIBLE_DEVICES to 0..num_gpus-1 and a hard timeout. Issue #55 topology review.
cd ~/gpubench && source venv/bin/activate
export VLLM_USE_V1=1 PYTHONUNBUFFERED=1
NG=$3; export HIP_VISIBLE_DEVICES=$(seq -s, 0 $((NG-1)))
timeout 1100 python -u ~/vllm-gfx908/bench_scripts/topo_bench.py "$1" "$2" 2>&1 \
  | grep -iE "INIT\[|out_tok/s=|TOPO_DONE|Graph capturing|Error|Traceback \(most|EngineDead|launch failure|RuntimeError|Mamba cache|TimeoutError"
echo "### EXIT $? ###"
