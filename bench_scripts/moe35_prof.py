"""Qwen3.5-35B-A3B-GPTQ-Int4 MoE decode: profile (eager) + throughput (graphs).
Usage: moe35_prof.py prof     -> eager, torch profiler over a decode window
       moe35_prof.py bench    -> graphs, decode tok/s at c=1/8/32
TP4 fits (24.4GB/4=6.1GB/die). language_model_only (multimodal wrapper).
"""
import os, sys, time, glob
os.environ.setdefault("VLLM_USE_V1", "1")
PROF = "/home/larkinwc/moe35_prof"
os.makedirs(PROF, exist_ok=True)
from vllm import LLM, SamplingParams
MODEL = "Qwen/Qwen3.5-35B-A3B-GPTQ-Int4"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bench"
    common = dict(model=MODEL, tensor_parallel_size=4, dtype="float16",
                  gpu_memory_utilization=0.90, max_model_len=2048,
                  language_model_only=True, max_num_seqs=32, trust_remote_code=True)
    if mode == "prof":
        from vllm.config.profiler import ProfilerConfig
        llm = LLM(enforce_eager=True,
                  profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir=PROF,
                                                 torch_profiler_with_stack=False), **common)
        sp = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)
        o = llm.generate(["The capital of France is"], sp)
        print("SANITY:", repr(o[0].outputs[0].text[:60]), flush=True)
        llm.start_profile()
        llm.generate(["Tell me a story."], SamplingParams(max_tokens=64, temperature=0, ignore_eos=True))
        llm.stop_profile(); time.sleep(3)
        print("TRACES:", sorted(glob.glob(PROF + "/rank0*")), flush=True)
        print("PROF_DONE", flush=True)
    else:
        from vllm.config.compilation import CompilationConfig, CUDAGraphMode, CompilationMode
        t0 = time.time()
        llm = LLM(enforce_eager=False,
                  compilation_config=CompilationConfig(mode=CompilationMode.NONE,
                      cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[1]),
                  **common)
        print("RESULT init %.0fs" % (time.time() - t0), flush=True)
        o = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
        print("SANITY:", repr(o[0].outputs[0].text[:60]), flush=True)
        base = "The history of computing began long ago and "
        for c in [1, 8, 32]:
            prompts = [base + str(i) for i in range(c)]
            sp = SamplingParams(max_tokens=96, temperature=0, ignore_eos=True)
            llm.generate(prompts[:1], SamplingParams(max_tokens=4, temperature=0, ignore_eos=True))
            t0 = time.time(); out = llm.generate(prompts, sp); dt = time.time() - t0
            tot = sum(len(x.outputs[0].token_ids) for x in out)
            print("RESULT decode c=%d out_tok_s=%.1f per_stream=%.2f tpot_ms=%.1f"
                  % (c, tot / dt, tot / dt / c, 1000 * dt / 96), flush=True)
        print("RESULT done", flush=True)


if __name__ == "__main__":
    main()
