#include "core/registration.h"
#include "rocm/ops.h"

#ifdef VLLM_BUILD_CK
#include "quantization/w8a8/int8/ck/ck_int8_gemm.h"
#include "quantization/gptq/ck/ck_w4a16_gemm.h"
#endif

// Note on op signatures:
// The X_meta signatures are for the meta functions corresponding to op X.
// They must be kept in sync with the signature for X. Generally, only
// functions that return Tensors require a meta function.
//
// See the following links for detailed docs on op registration and function
// schemas.
// https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/edit#heading=h.ptttacy8y1u9
// https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md#annotations

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, rocm_ops) {
  // vLLM custom ops for rocm

  // Custom gemm op for matrix-vector multiplication
  rocm_ops.def(
      "LLMM1(Tensor in_a, Tensor in_b, int rows_per_block) -> "
      "Tensor");
  rocm_ops.impl("LLMM1", torch::kCUDA, &LLMM1);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitK(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitK", torch::kCUDA, &wvSplitK);

  // Custom gemm op for skinny matrix-matrix multiplication
  rocm_ops.def(
      "wvSplitKrc(Tensor in_a, Tensor in_b, Tensor? in_bias, int CuCount) -> "
      "Tensor");
  rocm_ops.impl("wvSplitKrc", torch::kCUDA, &wvSplitKrc);

  // wvSplitK for fp8
  rocm_ops.def(
      "wvSplitKQ(Tensor in_a, Tensor in_b, Tensor? in_bias, Tensor! out_c, "
      "Tensor scale_a, "
      "          Tensor scale_b, int CuCount) -> ()");
  rocm_ops.impl("wvSplitKQ", torch::kCUDA, &wvSplitKQ);

  // Custom attention op
  // Compute the attention between an input query and the cached
  // keys/values using PagedAttention.
  rocm_ops.def(
      "paged_attention(Tensor! out, Tensor exp_sums,"
      "                Tensor max_logits, Tensor tmp_out,"
      "                Tensor query, Tensor key_cache,"
      "                Tensor value_cache, int num_kv_heads,"
      "                float scale, Tensor block_tables,"
      "                Tensor seq_lens,"
      "                Tensor? query_start_loc,"
      "                int block_size,"
      "                int max_seq_len,"
      "                Tensor? alibi_slopes,"
      "                str kv_cache_dtype,"
      "                Tensor k_scale, Tensor v_scale,"
      "                Tensor? fp8_out_scale,"
      "                str mfma_type) -> ()");
  rocm_ops.impl("paged_attention", torch::kCUDA, &paged_attention);

#ifdef VLLM_BUILD_CK
  // M4 (Path 3) Composable Kernel INT8 GEMM for gfx908.
  // Apply per-token + per-channel scales in a separate fp16 epilogue (gfx908
  // does not support FP-scale-fused MFMA, that's CDNA3+).
  rocm_ops.def(
      "ck_int8_gemm(Tensor a, Tensor b, Tensor scale_a, Tensor scale_b, "
      "Tensor? bias, int tp_rank) -> Tensor");
  rocm_ops.impl("ck_int8_gemm", torch::kCUDA, &vllm::ck_int8::ck_int8_gemm);

  // supports() takes only int args — register on the
  // CompositeExplicitAutograd dispatch key (no tensor → no device dispatch).
  rocm_ops.def(
      "ck_int8_gemm_supports(int M, int N, int K, int tp_rank) -> bool");

  // M4 W4A16 placeholder — see csrc/quantization/gptq/ck/ck_w4a16_gemm.h
  // for why no CK W4A16 instance is registered on gfx908 in ROCm 7.12.
  rocm_ops.def(
      "ck_w4a16_gemm(Tensor a, Tensor b, Tensor scales, Tensor zeros, "
      "int group_size, int tp_rank) -> Tensor");
  rocm_ops.impl("ck_w4a16_gemm", torch::kCUDA,
                &vllm::ck_w4a16::ck_w4a16_gemm);

  rocm_ops.def(
      "ck_w4a16_gemm_supports(int M, int N, int K, int group_size, "
      "int tp_rank) -> bool");
#endif
}

#ifdef VLLM_BUILD_CK
// supports() ops take only int args (no tensors), so they need a backend-
// agnostic registration via CompositeExplicitAutograd. See
// csrc/libtorch_stable/torch_bindings.cpp for the same pattern with
// `cutlass_scaled_mm_supports_fp8`.
TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CompositeExplicitAutograd,
                          rocm_ck_supports) {
  rocm_ck_supports.impl("ck_int8_gemm_supports",
                        &vllm::ck_int8::ck_int8_gemm_supports);
  rocm_ck_supports.impl("ck_w4a16_gemm_supports",
                        &vllm::ck_w4a16::ck_w4a16_gemm_supports);
}
#endif

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
