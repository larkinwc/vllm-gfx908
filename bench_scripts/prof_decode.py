import os, time, glob
os.environ.setdefault("VLLM_USE_V1","1")
PROF="/home/larkinwc/decode_prof"
os.makedirs(PROF, exist_ok=True)
from vllm import LLM, SamplingParams
from vllm.config.profiler import ProfilerConfig
def main():
    llm = LLM(model="Qwen/Qwen3.5-9B", tensor_parallel_size=4, dtype="float16",
              enforce_eager=True, gpu_memory_utilization=0.90, max_model_len=2048,
              language_model_only=True,
              profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir=PROF,
                                             torch_profiler_with_stack=False))
    # warm up the GDN/FLA autotune + steady state
    sp=SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)
    llm.generate(["The capital of France is"], sp)
    # profile a clean decode-only window: short prompt, many decode steps
    llm.start_profile()
    o=llm.generate(["Tell me a story."], SamplingParams(max_tokens=64,temperature=0,ignore_eos=True))
    llm.stop_profile()
    time.sleep(3)
    print("TRACES:", sorted(glob.glob(PROF+"/*")), flush=True)
    print("PROF_DONE", flush=True)
if __name__=="__main__": main()
