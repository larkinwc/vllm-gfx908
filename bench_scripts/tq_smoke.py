import os, time, sys
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
def main():
    preset = sys.argv[1] if len(sys.argv)>1 else "turboquant_k8v4"
    t0=time.time()
    llm = LLM(model="Qwen/Qwen3.5-9B", tensor_parallel_size=4, dtype="float16",
              enforce_eager=True, gpu_memory_utilization=0.90, max_model_len=4096,
              kv_cache_dtype=preset, **{"language_model_only": True})
    print("TQ_INIT[%s] in %.1fs"%(preset, time.time()-t0), flush=True)
    sp=SamplingParams(max_tokens=48, temperature=0)
    for p in ["The capital of France is", "Explain photosynthesis in one sentence:"]:
        o=llm.generate([p], sp)
        print("Q:", p, "\nA:", repr(o[0].outputs[0].text), flush=True)
    print("ALL_DONE", flush=True)
if __name__=="__main__": main()
