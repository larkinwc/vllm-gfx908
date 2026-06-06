import os, time, glob
os.environ.setdefault("VLLM_USE_V1", "1")
PROF = "/home/larkinwc/coder_prof"
os.makedirs(PROF, exist_ok=True)
from vllm import LLM, SamplingParams
from vllm.config.profiler import ProfilerConfig


def main():
    llm = LLM(model="bullpoint/Qwen3-Coder-Next-AWQ-4bit",
              tensor_parallel_size=4, pipeline_parallel_size=2, dtype="float16",
              enforce_eager=True, gpu_memory_utilization=0.92, max_model_len=32768,
              max_num_seqs=16, trust_remote_code=True,
              profiler_config=ProfilerConfig(profiler="torch", torch_profiler_dir=PROF,
                                             torch_profiler_with_stack=False))
    sp = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)
    llm.generate(["The capital of France is"], sp)  # warm steady state
    llm.start_profile()
    llm.generate(["Tell me a story."], SamplingParams(max_tokens=64, temperature=0, ignore_eos=True))
    llm.stop_profile()
    time.sleep(3)
    print("TRACES:", sorted(glob.glob(PROF + "/*")), flush=True)
    print("PROF_DONE", flush=True)


if __name__ == "__main__":
    main()
