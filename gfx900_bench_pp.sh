#!/bin/bash
# gfx900 Qwen3.5-9B PIPELINE-PARALLEL grid: TP2xPP{2,4} x concurrency{1,4} x {synthetic,coding}
# Stays within socket0 (GPUs 0-7) to avoid the slow cross-socket QPI link.
# PP2 -> 4 GPUs (0-3), PP4 -> 8 GPUs (0-7). Comparable to the TP4 / TP8 grids.
set -u
cd ~/gpubench && source venv/bin/activate
MODEL="Qwen/Qwen3.5-9B"
SHAREGPT=~/gpubench/sharegpt.json
OUTDIR=~/gpubench/bench_gfx900
mkdir -p "$OUTDIR"
PORT=8120

run_pp () {
  local PP=$1
  local DEVS=$2
  local NGPU=$3
  echo "############ Starting server TP2xPP=$PP ($NGPU GPUs: $DEVS) ############"
  HIP_VISIBLE_DEVICES=$DEVS VLLM_USE_V1=1 nohup vllm serve "$MODEL" \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size $PP \
    --dtype float16 \
    --enforce-eager \
    --language-model-only \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --no-async-scheduling \
    --port $PORT > "$OUTDIR/server_tp2pp${PP}.log" 2>&1 &
  SERVER_PID=$!
  echo "server PID=$SERVER_PID, waiting for ready..."
  for i in $(seq 1 180); do
    if curl -s "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q "$MODEL"; then
      echo "server READY after ${i}0s"; break
    fi
    sleep 10
  done

  for C in 1 4; do
    echo "=== CELL TP2PP=$PP c=$C synthetic ==="
    vllm bench serve \
      --model "$MODEL" --backend vllm --host 127.0.0.1 --port $PORT \
      --dataset-name random --random-input-len 1024 --random-output-len 256 \
      --num-prompts $((C*8)) --max-concurrency $C --ignore-eos \
      --request-rate inf --seed 42 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
      --save-result --result-filename "$OUTDIR/tp2pp${PP}_c${C}_synthetic.json" \
      > "$OUTDIR/cell_tp2pp${PP}_c${C}_synthetic.log" 2>&1
    echo "--- result ---"; grep -iE "Output token throughput|Total Token|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT" "$OUTDIR/cell_tp2pp${PP}_c${C}_synthetic.log"

    echo "=== CELL TP2PP=$PP c=$C coding ==="
    vllm bench serve \
      --model "$MODEL" --backend vllm --host 127.0.0.1 --port $PORT \
      --dataset-name sharegpt --dataset-path "$SHAREGPT" \
      --num-prompts $((C*8)) --max-concurrency $C \
      --request-rate inf --seed 42 \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
      --save-result --result-filename "$OUTDIR/tp2pp${PP}_c${C}_coding.json" \
      > "$OUTDIR/cell_tp2pp${PP}_c${C}_coding.log" 2>&1
    echo "--- result ---"; grep -iE "Output token throughput|Total Token|Request throughput|Median TTFT|P99 TTFT|Median TPOT|P99 TPOT" "$OUTDIR/cell_tp2pp${PP}_c${C}_coding.log"
  done

  echo "############ Stopping server TP2xPP=$PP ############"
  kill $SERVER_PID 2>/dev/null
  sleep 15
  pkill -f "vllm serve" 2>/dev/null
  sleep 10
}

echo "===== PP BENCH START $(date) ====="
run_pp 4 "0,1,2,3,4,5,6,7" 8
run_pp 2 "0,1,2,3" 4
echo "===== PP BENCH END $(date) ====="
