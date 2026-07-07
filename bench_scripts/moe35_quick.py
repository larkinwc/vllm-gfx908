"""Quick c=1 decode bench for the GEMV config sweep. Reads GEMV_BN/BK/NW/NS via env
(consumed inside fused_moe config). Prints one RESULT line."""
import os, time
os.environ.setdefault("VLLM_USE_V1", "1")
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode, CompilationMode


def main():
    llm = LLM(model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", tensor_parallel_size=4, dtype="float16",
              gpu_memory_utilization=0.90, max_model_len=2048, language_model_only=True,
              max_num_seqs=32, trust_remote_code=True, enforce_eager=False,
              compilation_config=CompilationConfig(mode=CompilationMode.NONE,
                  cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[1]))
    o = llm.generate(["The capital of France is"], SamplingParams(max_tokens=8, temperature=0))
    sane = o[0].outputs[0].text[:20]
    p = ["The history of computing began long ago and 0"]
    sp = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True)
    llm.generate(p, SamplingParams(max_tokens=4, temperature=0, ignore_eos=True))
    best = 0.0
    for _ in range(2):
        t0 = time.time(); out = llm.generate(p, sp); dt = time.time() - t0
        n = len(out[0].outputs[0].token_ids)
        best = max(best, n / dt)
    cfg = "BN=%s BK=%s NW=%s NS=%s" % (os.environ.get("GEMV_BN", "32"), os.environ.get("GEMV_BK", "64"),
                                       os.environ.get("GEMV_NW", "4"), os.environ.get("GEMV_NS", "2"))
    print("RESULT %s c1_tok_s=%.1f sane=%r" % (cfg, best, sane), flush=True)


if __name__ == "__main__":
    main()
