#!/usr/bin/env python3
"""Full-model decode profiler: load a model, run a steady-state decode region,
so rocprofv3 --kernel-trace can rank ALL hot kernels (attention, GEMM, norms,
sampling, all-reduce) — the triage step for finding hand-kernel candidates.

Usage:
  PYTHONPATH=. HIP_VISIBLE_DEVICES=2,3 rocprofv3 --kernel-trace \
    --output-format csv -d /tmp/decprof -- \
    python bench_scripts/decode_profile.py <model_path> <tp>
"""
import os
import sys

import torch

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/models/Qwen3.5-9B"
TP = int(sys.argv[2]) if len(sys.argv) > 2 else 1


def main():
    from vllm import LLM, SamplingParams

    kw = dict(
        model=MODEL,
        dtype="float16",
        tensor_parallel_size=TP,
        gpu_memory_utilization=0.85,
        max_model_len=2048,
        enforce_eager=False,
        disable_log_stats=True,
    )
    if os.environ.get("SLOW_TOK"):
        kw["tokenizer_mode"] = "slow"
    llm = LLM(**kw)
    # warmup: build cudagraphs + caches
    sp = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    llm.generate(["The quick brown fox"], sp, use_tqdm=False)

    # steady-state decode region to be profiled: 1 prompt, many decode steps
    sp = SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True)
    torch.cuda.synchronize()
    llm.generate(["Once upon a time in a distant land,"], sp, use_tqdm=False)
    torch.cuda.synchronize()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
