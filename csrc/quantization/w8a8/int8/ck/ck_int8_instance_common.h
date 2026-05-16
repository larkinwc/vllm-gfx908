// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Common include header for CK INT8 GEMM instances on gfx908.
//
// NOTE: We intentionally include CK from a vendored snapshot of
// composable_kernel that has the gfx908-correct `get_warp_size()` (separate
// __host__/__device__ overloads). The system-installed CK at
// /opt/rocm/core-7.12/include/ck has a single __host__ __device__ definition
// that uses the HIP magic constant `warpSize`, which the host-side template
// instantiation cannot resolve and triggers a static_assert error in
// blockwise_gemm_xdlops.hpp.
//
// The exact tile parameters (mission-binding):
//   MPerBlock=128, NPerBlock=128, KPerBlock=64, MPerXdl=16, NPerXdl=16
// chosen to drive the gfx908 INT8 MFMA `v_mfma_i32_16x16x16i8`.
// Forbidden intrinsics (`v_smfmac_*`, `v_mfma_*scale*`) are verified
// absent post-build via llvm-objdump.
#pragma once

#include <hip/hip_runtime.h>

#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

#include "ck_int8_gemm.h"

namespace vllm::ck_int8 {

void register_instance(int64_t m_min, int64_t m_max, int64_t n, int64_t k,
                       int64_t tp_rank,
                       std::function<bool(const at::Tensor&, const at::Tensor&,
                                          at::Tensor&)> run);

namespace detail {

using ALayout = ck::tensor_layout::gemm::RowMajor;
using BLayout = ck::tensor_layout::gemm::ColumnMajor;
using CLayout = ck::tensor_layout::gemm::RowMajor;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// Mission-binding tile descriptor. Chosen so HIPCC emits
// v_mfma_i32_16x16x16i8 (gfx908 INT8 MFMA). FP-scale-fused MFMA epilogues
// are not used (gfx908 does not support them); we apply scales as a
// separate fp16 pass after the int32 GEMM.
using DeviceGemmInt8 = ck::tensor_operation::device::DeviceGemm_Xdl_CShuffle<
    ALayout, BLayout, CLayout,
    int8_t, int8_t, int32_t,
    int32_t, int32_t,
    PassThrough, PassThrough, PassThrough,
    ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    1,                              // NumGemmKPrefetchStage
    256,                            // BlockSize
    128, 128, 64,                   // MPerBlock, NPerBlock, KPerBlock
    16, 16,                         // AK1, BK1
    16, 16,                         // MPerXDL, NPerXDL
    4, 4,                           // MXdlPerWave, NXdlPerWave
    ck::Sequence<4, 64, 1>,         // ABlockTransfer cluster
    ck::Sequence<1, 0, 2>,
    ck::Sequence<1, 0, 2>,
    2, 16, 16, 1,                   // SrcVectorDim, SrcScalar, DstScalarPerVector_AK1, ABlockLdsExtraM
    ck::Sequence<4, 64, 1>,         // BBlockTransfer cluster
    ck::Sequence<1, 0, 2>,
    ck::Sequence<1, 0, 2>,
    2, 8, 8, 1,                     // SrcVectorDim, SrcScalar, DstScalarPerVector_BK1, BBlockLdsExtraN
    1, 1,                           // CShuffleMXdlPerWavePerShuffle, NXdlPerWavePerShuffle
    ck::Sequence<1, 32, 1, 8>,      // CShuffleBlockTransferClusterLengths
    4,                              // CShuffleBlockTransferScalarPerVector_NPerBlock
    ck::LoopScheduler::Default,
    ck::PipelineVersion::v1,
    int8_t, int8_t>;                // ComputeTypeA, ComputeTypeB — drives v_mfma_i32_16x16x16i8

// Run a CK INT8 GEMM with the mission-binding tile descriptor.
// a:    [M, K] int8 row-major, contiguous
// b:    [N, K] int8 row-major, contiguous (treated as [K,N] col-major by CK)
// out:  [M, N] int32 row-major, contiguous
inline bool run_ck_int8_gemm(const at::Tensor& a, const at::Tensor& b,
                             at::Tensor& out) {
  int64_t M = a.size(0);
  int64_t K = a.size(1);
  int64_t N = b.size(0);
  int64_t StrideA = K;
  int64_t StrideB = K;
  int64_t StrideC = N;

  DeviceGemmInt8 gemm;
  auto invoker = gemm.MakeInvoker();
  auto arg = gemm.MakeArgument(
      a.data_ptr<int8_t>(), b.data_ptr<int8_t>(), out.data_ptr<int32_t>(),
      M, N, K, StrideA, StrideB, StrideC,
      PassThrough{}, PassThrough{}, PassThrough{});

  if (!gemm.IsSupportedArgument(arg)) {
    return false;
  }
  StreamConfig sc{at::cuda::getCurrentCUDAStream().stream(), false};
  invoker.Run(arg, sc);
  return true;
}

// Helper for instance TUs to register one (M_bucket, N, K, tp_rank) entry.
inline void register_shape(int64_t m_min, int64_t m_max, int64_t n, int64_t k,
                           int64_t tp_rank) {
  register_instance(m_min, m_max, n, k, tp_rank, &run_ck_int8_gemm);
}

}  // namespace detail
}  // namespace vllm::ck_int8
