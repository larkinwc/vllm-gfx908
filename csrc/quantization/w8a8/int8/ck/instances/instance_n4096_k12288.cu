// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// CK INT8 instance for Qwen3.5-9B W8A8 second-hottest prefill shape:
// (M=variable, N=4096, K=12288) — MLP down_proj.
// See instance_n24576_k4096.cu for why we explicitly include cuda_runtime.h
// (forces the hipify preprocessor to rename to .hip).
#include <cuda_runtime.h>

#include "../ck_int8_instance_common.h"

namespace vllm::ck_int8 {
namespace {
struct Registrar {
  Registrar() {
    detail::register_shape(/*m_min=*/16, /*m_max=*/4096,
                           /*n=*/4096, /*k=*/12288, /*tp_rank=*/1);
  }
};
static Registrar _registrar;
}  // namespace
}  // namespace vllm::ck_int8
