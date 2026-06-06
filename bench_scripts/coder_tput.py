"""Qwen3-Coder-Next MoE throughput sweep on gfx900 (TP4xPP2, graphs).
Usage: coder_tput.py [kv_dtype]   kv_dtype in {auto, turboquant_k8v4}
Sweeps concurrency c in {1,8,16,32}, reports aggregate + per-stream tok/s.
Issue #65 option B."""
import os, sys, time
os.environ.setdefault("VLLM_USE_V1", "1")
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode, CompilationMode

MODEL = "bullpoint/Qwen3-Coder-Next-AWQ-4bit"


def main():
    kv = sys.argv[1] if len(sys.argv) > 1 else "auto"
    kw = dict(model=MODEL, tensor_parallel_size=4, pipeline_parallel_size=2,
              dtype="float16", gpu_memory_utilization=0.92, max_model_len=32768,
              max_num_seqs=32, trust_remote_code=True, enforce_eager=False,
              compilation_config=CompilationConfig(mode=CompilationMode.NONE,
                  cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                  cudagraph_capture_sizes=[1, 8, 16, 32]))
    if kv != "auto":
        kw["kv_cache_dtype"] = kv
    t0 = time.time()
    llm = LLM(**kw)
    print("INIT[tput kv=%s] %.0fs" % (kv, time.time() - t0), flush=True)
    base = "Write a short note about topic number "
    for c in [1, 8, 16, 32]:
        prompts = [base + str(i) for i in range(c)]
        sp = SamplingParams(max_tokens=96, temperature=0, ignore_eos=True)
        llm.generate(prompts[:1], SamplingParams(max_tokens=4, temperature=0, ignore_eos=True))
        t0 = time.time()
        o = llm.generate(prompts, sp)
        dt = time.time() - t0
        tot = sum(len(x.outputs[0].token_ids) for x in o)
        print("[kv=%s] c=%2d  agg_tok/s=%.1f  per-stream=%.2f  tpot_ms=%.1f"
              % (kv, c, tot / dt, tot / dt / c, 1000 * dt / 96), flush=True)
    print("TPUT_DONE", flush=True)


if __name__ == "__main__":
    main()
