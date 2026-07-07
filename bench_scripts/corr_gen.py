"""Decode-path correctness capture. Greedy-generate 128 tokens with per-token
logprobs and dump (token_id, logprob) to $OUT. Run once with GEMV on (default)
and once with VLLM_GFX900_MOE_GEMV=0 (baseline tl.dot), then diff with corr_cmp.py.
"""
import os, json
os.environ.setdefault("VLLM_USE_V1", "1")
from vllm import LLM, SamplingParams
from vllm.config.compilation import CompilationConfig, CUDAGraphMode, CompilationMode


def main():
    llm = LLM(model="Qwen/Qwen3.5-35B-A3B-GPTQ-Int4", tensor_parallel_size=4, dtype="float16",
              gpu_memory_utilization=0.90, max_model_len=2048, language_model_only=True,
              max_num_seqs=32, trust_remote_code=True, enforce_eager=False,
              compilation_config=CompilationConfig(mode=CompilationMode.NONE,
                  cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY, cudagraph_capture_sizes=[1]))
    prompt = ("Explain in detail how a mixture-of-experts transformer routes tokens "
              "to experts during inference, and why sparsity helps.")
    sp = SamplingParams(max_tokens=128, temperature=0.0, logprobs=1, ignore_eos=True)
    o = llm.generate([prompt], sp)[0].outputs[0]
    seq = []
    for tid, lp in zip(o.token_ids, o.logprobs):
        # lp is {token_id: Logprob}; take the chosen token's logprob
        seq.append([int(tid), float(lp[tid].logprob)])
    out = os.environ.get("OUT", "/home/larkinwc/corr_out.json")
    json.dump({"gemv": os.environ.get("VLLM_GFX900_MOE_GEMV", "1"),
               "text": o.text, "seq": seq}, open(out, "w"))
    print("WROTE", out, "ntok", len(seq), "gemv=", os.environ.get("VLLM_GFX900_MOE_GEMV", "1"), flush=True)


if __name__ == "__main__":
    main()
