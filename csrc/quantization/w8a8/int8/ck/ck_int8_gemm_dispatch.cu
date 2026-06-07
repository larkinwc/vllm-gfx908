// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Dispatch table for CK INT8 GEMM instances on gfx908 (MI100).
//
// Each instance under instances/*.cpp registers itself by appending an entry
// at static-initialization time. The dispatch table is keyed on
// (M_bucket, N, K, tp_rank). M_bucket buckets the activation rows (decode
// vs prefill) so that we don't have to instantiate CK for every M.
//
// IMPORTANT (mission-binding): `MPerBlock=128, NPerBlock=128, KPerBlock=64,
// MPerXdl=16, NPerXdl=16` — chosen so the compiler emits
// `v_mfma_i32_16x16x16i8` (gfx908 INT8 MFMA). FP-scale-fused MFMA epilogues
// (`v_mfma_*scale*`, `v_smfmac_*`) are FORBIDDEN on gfx908; scales are
// therefore applied in a separate fp16 pass after the int32 GEMM completes.
#include "ck_int8_gemm.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace vllm::ck_int8 {

namespace {

struct InstanceEntry {
  int64_t m_min;
  int64_t m_max;
  int64_t n;
  int64_t k;
  int64_t tp_rank;
  std::function<bool(const at::Tensor&, const at::Tensor&, at::Tensor&)> run;
};

std::vector<InstanceEntry>& registry() {
  static std::vector<InstanceEntry> r;
  return r;
}

std::mutex& registry_mutex() {
  static std::mutex m;
  return m;
}

// Apply per-token (M) + per-channel (N) scales and optional bias, casting
// int32 accumulator output to fp16. Done as a small fused HIP kernel so we
// don't pay a global memory roundtrip beyond the single dequant pass.
__global__ void apply_scales_kernel(const int32_t* __restrict__ acc,
                                    const float* __restrict__ scale_a,
                                    const float* __restrict__ scale_b,
                                    const __half* __restrict__ bias,
                                    __half* __restrict__ out, int M, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx >= total) {
    return;
  }
  int m = idx / N;
  int n = idx - m * N;
  float v = static_cast<float>(acc[idx]) * scale_a[m] * scale_b[n];
  if (bias != nullptr) {
    v += static_cast<float>(bias[n]);
  }
  out[idx] = __float2half(v);
}

void apply_scales(at::Tensor& acc_int32, const at::Tensor& scale_a,
                  const at::Tensor& scale_b,
                  const std::optional<at::Tensor>& bias, at::Tensor& out_fp16) {
  int M = acc_int32.size(0);
  int N = acc_int32.size(1);
  int total = M * N;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  const __half* bias_ptr =
      bias.has_value() ? reinterpret_cast<const __half*>(bias->data_ptr())
                       : nullptr;
  hipLaunchKernelGGL(apply_scales_kernel, dim3(blocks), dim3(threads), 0,
                     at::cuda::getCurrentCUDAStream(),
                     acc_int32.data_ptr<int32_t>(), scale_a.data_ptr<float>(),
                     scale_b.data_ptr<float>(), bias_ptr,
                     reinterpret_cast<__half*>(out_fp16.data_ptr()), M, N);
}

}  // namespace

void register_instance(
    int64_t m_min, int64_t m_max, int64_t n, int64_t k, int64_t tp_rank,
    std::function<bool(const at::Tensor&, const at::Tensor&, at::Tensor&)>
        run) {
  std::lock_guard<std::mutex> g(registry_mutex());
  registry().push_back({m_min, m_max, n, k, tp_rank, std::move(run)});
}

bool ck_int8_gemm_supports(int64_t M, int64_t N, int64_t K, int64_t tp_rank) {
  std::lock_guard<std::mutex> g(registry_mutex());
  for (const auto& e : registry()) {
    if (e.n == N && e.k == K && e.tp_rank == tp_rank && M >= e.m_min &&
        M <= e.m_max) {
      return true;
    }
  }
  return false;
}

torch::Tensor ck_int8_gemm(const torch::Tensor& a, const torch::Tensor& b,
                           const torch::Tensor& scale_a,
                           const torch::Tensor& scale_b,
                           const std::optional<torch::Tensor>& bias,
                           int64_t tp_rank) {
  TORCH_CHECK(a.dtype() == at::kChar, "a must be int8");
  TORCH_CHECK(b.dtype() == at::kChar, "b must be int8");
  TORCH_CHECK(a.dim() == 2, "a must be 2D");
  TORCH_CHECK(b.dim() == 2, "b must be 2D");
  TORCH_CHECK(scale_a.dtype() == at::kFloat, "scale_a must be fp32");
  TORCH_CHECK(scale_b.dtype() == at::kFloat, "scale_b must be fp32");
  TORCH_CHECK(a.is_contiguous(), "a must be contiguous");
  TORCH_CHECK(b.is_contiguous(), "b must be contiguous");

  int64_t M = a.size(0);
  int64_t K = a.size(1);
  int64_t N = b.size(0);
  TORCH_CHECK(b.size(1) == K, "b second dim must equal K");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));

  auto out = torch::empty({M, N}, a.options().dtype(at::kHalf));
  auto acc = torch::empty({M, N}, a.options().dtype(at::kInt));

  std::function<bool(const at::Tensor&, const at::Tensor&, at::Tensor&)> picked;
  {
    std::lock_guard<std::mutex> g(registry_mutex());
    for (const auto& e : registry()) {
      if (e.n == N && e.k == K && e.tp_rank == tp_rank && M >= e.m_min &&
          M <= e.m_max) {
        picked = e.run;
        break;
      }
    }
  }
  TORCH_CHECK(picked, "No CK INT8 instance registered for (M=", M, ", N=", N,
              ", K=", K, ", tp_rank=", tp_rank,
              "). Caller should fall back to hipBLASLt/Triton.");

  bool ok = picked(a, b, acc);
  TORCH_CHECK(ok, "CK INT8 GEMM kernel launch failed");

  apply_scales(acc, scale_a, scale_b, bias, out);
  return out;
}

}  // namespace vllm::ck_int8
