// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Explicit cuda_runtime.h include forces the hipify preprocessor to rename
// this `.cu` file's output to `.hip` so the cmake HIP build rule resolves.
#include <cuda_runtime.h>

#include "ck_w4a16_gemm.h"

#include <torch/all.h>

namespace vllm::ck_w4a16 {

bool ck_w4a16_gemm_supports(int64_t /*M*/, int64_t /*N*/, int64_t /*K*/,
                            int64_t /*group_size*/, int64_t /*tp_rank*/) {
  // CK ROCm 7.12 does not ship a gfx908-correct W4A16 device template
  // matching vLLM's groupwise (32, 128) scale+zero layout. The dispatcher
  // must fall through to Triton (M3 mi100_w4a16) for these shapes.
  // See BENCH_M4_CK.md "W4A16 deferral".
  return false;
}

torch::Tensor ck_w4a16_gemm(const torch::Tensor& /*a_fp16*/,
                            const torch::Tensor& /*b_packed_int4*/,
                            const torch::Tensor& /*scales*/,
                            const torch::Tensor& /*zeros*/,
                            int64_t /*group_size*/, int64_t /*tp_rank*/) {
  TORCH_CHECK(
      false,
      "ck_w4a16_gemm: no CK W4A16 instance is registered on gfx908 in this "
      "ROCm install. Caller should consult ck_w4a16_gemm_supports() first; "
      "the dispatcher currently falls through to mi100_w4a16 (Triton).");
  return torch::Tensor{};
}

}  // namespace vllm::ck_w4a16
