// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// CK INT8 instance for Qwen3.5-9B W8A8 prefill shape:
// (M=variable, N=10240, K=4096) — gate_up MLP fused proj.
// See instance_n24576_k4096.cu for why we explicitly include cuda_runtime.h.
#include <cuda_runtime.h>

#include "../ck_int8_instance_common.h"

namespace vllm::ck_int8 {
namespace {
struct Registrar {
  Registrar() {
    detail::register_shape(/*m_min=*/16, /*m_max=*/4096,
                           /*n=*/10240, /*k=*/4096, /*tp_rank=*/1);
  }
};
static Registrar _registrar;
}  // namespace
}  // namespace vllm::ck_int8
