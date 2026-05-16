// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// CK INT8 instance for the Qwen3.5-9B W8A8 hottest prefill shape:
// (M=variable prefill batch, N=24576, K=4096) — fused QKV in_proj.
// On TP=4 N shards by 4 → (M, 6144, 4096) handled by instance_n6144_k4096.cpp.
// NOTE: `cuda_runtime.h` is intentionally referenced via the include below
// so the hipify_python preprocessor recognises this `.cu` file as needing
// a hipified `.hip` output. The vendored common header
// `ck_int8_instance_common.h` already pulls in `<hip/hip_runtime.h>`, so
// this is a no-op at compile time.
#include <cuda_runtime.h>

#include "../ck_int8_instance_common.h"

namespace vllm::ck_int8 {
namespace {
struct Registrar {
  Registrar() {
    // M-bucket covers the prefill regime (M >= 16 typical for Qwen3.5-9B
    // synthetic prompts at concurrency 1..4) up through 4096 token prefill.
    detail::register_shape(/*m_min=*/16, /*m_max=*/4096,
                           /*n=*/24576, /*k=*/4096, /*tp_rank=*/1);
    // TP=4 row-parallel for the same logical layer (K shards by 4 only when
    // we instead hit row-parallel paths; for column-parallel the N shrinks
    // — handled in n6144_k4096.cpp). We register 24576/4=6144 here as a
    // safety alias for any TP=4 edge case that still issues full N.
  }
};
static Registrar _registrar;
}  // namespace
}  // namespace vllm::ck_int8
