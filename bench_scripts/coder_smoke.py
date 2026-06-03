"""Qwen3-Coder-Next-AWQ-4bit bring-up smoke test on gfx900.
Usage: coder_smoke.py <layout> <graph|eager>   layout in {tp4pp2, tp8, tp4}
Issue #65. TP4xPP2 on one socket (GPUs 0-7).
"""
import os, sys, time
os.environ.setdefault("VLLM_USE_V1", "1")
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode, CompilationMode

MODEL = "bullpoint/Qwen3-Coder-Next-AWQ-4bit"


def main():
    layout = sys.argv[1] if len(sys.argv) > 1 else "tp4pp2"
    graphs = (len(sys.argv) > 2 and sys.argv[2] == "graph")
    tp, pp = dict(tp4pp2=(4, 2), tp8=(8, 1), tp4=(4, 1))[layout]
    kw = dict(model=MODEL, tensor_parallel_size=tp, pipeline_parallel_size=pp,
              dtype="float16", gpu_memory_utilization=0.92, max_model_len=32768,
              max_num_seqs=16, trust_remote_code=True)
    if graphs:
        kw["enforce_eager"] = False
        kw["compilation_config"] = CompilationConfig(
            mode=CompilationMode.NONE,
            cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
            cudagraph_capture_sizes=[1])
    else:
        kw["enforce_eager"] = True
    t0 = time.time()
    llm = LLM(**kw)
    print("INIT[%s g=%s tp=%d pp=%d] %.0fs" % (layout, graphs, tp, pp, time.time() - t0), flush=True)
    sp = SamplingParams(max_tokens=64, temperature=0)
    prompt = "Write a Python function that returns the nth Fibonacci number.\n"
    o = llm.generate([prompt], sp)
    print("=== OUTPUT ===", flush=True)
    print(o[0].outputs[0].text, flush=True)
    # quick decode tok/s, bs=1
    sp2 = SamplingParams(max_tokens=96, temperature=0, ignore_eos=True)
    llm.generate([prompt], SamplingParams(max_tokens=4, temperature=0, ignore_eos=True))
    t0 = time.time(); o = llm.generate([prompt], sp2); dt = time.time() - t0
    tot = len(o[0].outputs[0].token_ids)
    print("DECODE bs=1: %.1f tok/s (%.1f ms/tok)" % (tot / dt, 1000 * dt / tot), flush=True)
    print("CODER_DONE", flush=True)


if __name__ == "__main__":
    main()
