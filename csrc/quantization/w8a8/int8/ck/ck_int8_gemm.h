// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// M4 (Path 3) Composable Kernel INT8 GEMM for MI100 (gfx908).
//
// Header declares the high-level dispatch entry point that picks one of the
// per-shape DeviceGemm_Xdl_CShuffle instantiations registered from
// instances/*.cpp. Scales are applied in a separate post-processing pass
// because gfx908 does not support FP-scale-fused MFMA epilogues.
#pragma once

#include <torch/all.h>

#include <cstdint>
#include <optional>

namespace vllm::ck_int8 {

// Compute c = (a_int8 @ b_int8) with per-token + per-channel scales applied
// in fp16. b is expected in [N, K] (column-of-A · row-of-B convention) with
// row-major layout (i.e. lhs/rhs both row-major; CK runs `Cijk_Ailk_Bljk`).
//
// Shapes:
//   a:        [M, K] int8
//   b:        [N, K] int8 (will be interpreted as KxN col-major)
//   scale_a:  [M] or [M,1] fp32 (per-token activation scale)
//   scale_b:  [N] or [1,N] fp32 (per-channel weight scale)
//   bias:     optional [N] fp16
// Returns:    [M, N] fp16
//
// `tp_rank`: world size of tensor-parallel group (with TP=1 set 1; with TP=4
// set 4) — matches CK instance registration key in dispatcher. Used to pick
// the right registered instance.
torch::Tensor ck_int8_gemm(const torch::Tensor& a, const torch::Tensor& b,
                           const torch::Tensor& scale_a,
                           const torch::Tensor& scale_b,
                           const std::optional<torch::Tensor>& bias,
                           int64_t tp_rank);

// Returns true if a CK instance is registered for (M, N, K, tp_rank).
bool ck_int8_gemm_supports(int64_t M, int64_t N, int64_t K, int64_t tp_rank);

}  // namespace vllm::ck_int8
