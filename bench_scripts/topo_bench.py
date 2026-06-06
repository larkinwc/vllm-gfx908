"""Topology sweep for gfx900: TP4/TP8/TP16/TP8xPP2 decode benchmark.

Usage: topo_bench.py <layout> <graph|eager>
  layout in {tp4, tp8, tp16, tp8pp2, tp2pp2}
Driver run_topo.sh sets HIP_VISIBLE_DEVICES to the right GPU count and a hard timeout.
All runs use LLMM1 (default skinny-GEMM) and the FULL_DECODE_ONLY CUDA-graph config.
Issue #55 / topology review.
"""
import os, sys, time
os.environ.setdefault("VLLM_USE_V1", "1")
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode, CompilationMode


def main():
    layout = sys.argv[1]
    graphs = (sys.argv[2] == "graph") if len(sys.argv) > 2 else True
    tp, pp = dict(tp4=(4, 1), tp8=(8, 1), tp16=(16, 1),
                  tp8pp2=(8, 2), tp2pp2=(2, 2))[layout]
    kw = dict(model="Qwen/Qwen3.5-9B", tensor_parallel_size=tp,
              pipeline_parallel_size=pp, dtype="float16",
              gpu_memory_utilization=0.90, max_model_len=2048,
              language_model_only=True, max_num_seqs=32)
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
    base = "The history of computing began long ago and "
    for c in [1, 8, 32]:
        prompts = [base + str(i) for i in range(c)]
        sp = SamplingParams(max_tokens=96, temperature=0, ignore_eos=True)
        llm.generate(prompts[:1], SamplingParams(max_tokens=4, temperature=0, ignore_eos=True))
        t0 = time.time()
        o = llm.generate(prompts, sp)
        dt = time.time() - t0
        tot = sum(len(x.outputs[0].token_ids) for x in o)
        print("[%s g=%s] c=%2d  out_tok/s=%.1f  per-stream=%.2f  tpot_ms=%.1f"
              % (layout, graphs, c, tot / dt, tot / dt / c, 1000 * dt / 96), flush=True)
    print("TOPO_DONE_%s" % layout, flush=True)


if __name__ == "__main__":
    main()
