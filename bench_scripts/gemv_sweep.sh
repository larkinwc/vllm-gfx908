#!/bin/bash
source ~/venvs/triton/bin/activate
export PATH=/opt/rocm-7.2.4/bin:$PATH ROCM_PATH=/opt/rocm-7.2.4 LD_LIBRARY_PATH=/opt/rocm-7.2.4/lib:$LD_LIBRARY_PATH TMPDIR=$HOME/buildtmp PYTHONUNBUFFERED=1 HIP_VISIBLE_DEVICES=0,1,2,3
for cfg in 32:64:4:2 32:128:4:2 64:64:4:2 64:128:8:2 16:64:4:2; do
  IFS=: read bn bk nw ns <<< "$cfg"
  echo ">>> CONFIG $cfg"
  GEMV_BN=$bn GEMV_BK=$bk GEMV_NW=$nw GEMV_NS=$ns timeout 500 python -u ~/moe35_quick.py 2>&1 | grep -E "RESULT|Error:|RuntimeError|assert|CUDA out" | tail -3
done
echo SWEEP_DONE
