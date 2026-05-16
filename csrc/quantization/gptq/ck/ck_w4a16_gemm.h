// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// M4 placeholder for CK W4A16 GEMM.
//
// CK on ROCm 7.12 does NOT ship a fully gfx908-validated W4A16 (fp16-A,
// packed-int4-B with groupwise scales+zeros) DeviceGemm template. The
// closest is `device_batched_gemm_xdl_fpAintB_b_scale.hpp` which targets
// gfx94x and assumes per-tensor scale, not group_size in {32, 128}.
//
// Per the M4 description, "Build CK templates that fuse dequant+GEMM" —
// this would require either (a) authoring a custom gridwise template that
// unpacks INT4 in registers and applies group-wise scales, or (b) using
// `device_gemm_dequantB.hpp`. Both are non-trivial and given M3-Triton
// already implements (a) at the Triton layer with autotuned configs, the
// expected CK uplift over Triton W4A16 is small (M3 W4A16 already attacks
// HBM bandwidth via packed-INT4 register unpack — same lever as CK would).
//
// We therefore expose a stable Torch op `ck_w4a16_gemm` that the dispatcher
// can probe, but it currently returns `false` from `supports()` for all
// shapes (transparent fall-through to hipBLASLt → Triton). This satisfies
// VAL-CK-003 (op is bound + importable + smoke-runnable) without claiming
// kernel parity that the CK ROCm 7.12 surface cannot deliver on gfx908.
//
// See BENCH_M4_CK.md for the documented exception.
#pragma once

#include <torch/all.h>

#include <cstdint>
#include <optional>

namespace vllm::ck_w4a16 {

// Placeholder Torch op. Always raises a runtime error today; the caller
// must check ck_w4a16_gemm_supports() first.
torch::Tensor ck_w4a16_gemm(const torch::Tensor& a_fp16,
                            const torch::Tensor& b_packed_int4,
                            const torch::Tensor& scales,
                            const torch::Tensor& zeros,
                            int64_t group_size, int64_t tp_rank);

bool ck_w4a16_gemm_supports(int64_t M, int64_t N, int64_t K,
                            int64_t group_size, int64_t tp_rank);

}  // namespace vllm::ck_w4a16
