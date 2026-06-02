import os, time, sys
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode

def main():
    MODE = sys.argv[1]
    common = dict(model="Qwen/Qwen3.5-9B", tensor_parallel_size=4, dtype="float16",
                  gpu_memory_utilization=0.85, max_model_len=2048,
                  max_num_batched_tokens=2048, **{"language_model_only": True})
    if MODE=="eager":
        llm = LLM(enforce_eager=True, **common)
    else:
        cc = CompilationConfig(mode=CompilationMode.NONE,
                               cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                               cudagraph_capture_sizes=[1])
        llm = LLM(compilation_config=cc, **common)
    sp = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True)
    llm.generate(["warmup prompt here"], sp)
    prompt = "Write a detailed explanation of how transformers work in machine learning."
    N=3; tot=0.0; toks=0
    for i in range(N):
        t0=time.time(); o=llm.generate([prompt], sp); dt=time.time()-t0
        n=len(o[0].outputs[0].token_ids); tot+=dt; toks+=n
    print(f"MODE={MODE} avg_total={tot/N:.3f}s tokens={toks//N} tok_s={toks/tot:.2f} ms_per_tok={1000*tot/toks:.1f}", flush=True)

if __name__ == "__main__":
    main()
