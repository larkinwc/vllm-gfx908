// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// CK INT8 instance for Qwen3.5-9B W8A8 prefill shape:
// (M=variable, N=4096, K=4096) — attention out_proj.
// TP=4 column-parallel shrinks to N=1024.
// See instance_n24576_k4096.cu for why we explicitly include cuda_runtime.h.
#include <cuda_runtime.h>

#include "../ck_int8_instance_common.h"

namespace vllm::ck_int8 {
namespace {
struct Registrar {
  Registrar() {
    // TP=1 path
    detail::register_shape(/*m_min=*/16, /*m_max=*/4096,
                           /*n=*/4096, /*k=*/4096, /*tp_rank=*/1);
    // TP=4 column-parallel out_proj: N shards by 4 → 1024
    detail::register_shape(/*m_min=*/16, /*m_max=*/4096,
                           /*n=*/1024, /*k=*/4096, /*tp_rank=*/4);
  }
};
static Registrar _registrar;
}  // namespace
}  // namespace vllm::ck_int8
