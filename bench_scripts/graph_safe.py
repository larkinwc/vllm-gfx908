import os, time
os.environ.setdefault("VLLM_USE_V1","1")
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CompilationMode, CUDAGraphMode
def main():
    cc = CompilationConfig(
        mode=CompilationMode.NONE,
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        cudagraph_capture_sizes=[1],
    )
    t0=time.time()
    llm = LLM(model="Qwen/Qwen3.5-9B",
              tensor_parallel_size=4,
              dtype="float16",
              gpu_memory_utilization=0.85, max_model_len=2048,
              max_num_batched_tokens=2048,
              compilation_config=cc,
              **{"language_model_only": True})
    print("INIT_DONE in %.1fs"%(time.time()-t0), flush=True)
    out = llm.generate(["The capital of France is"], SamplingParams(max_tokens=16, temperature=0))
    for o in out: print("RESULT:", repr(o.outputs[0].text), flush=True)
    print("ALL_DONE", flush=True)
if __name__ == "__main__":
    main()
