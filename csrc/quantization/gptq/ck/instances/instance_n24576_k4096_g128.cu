// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// W4A16 placeholder instance for the (N=24576, K=4096, group_size=128)
// shape. See ../ck_w4a16_gemm.h for the rationale on why CK does not
// register a runnable instance for this shape on gfx908 in ROCm 7.12.
// See ../../w8a8/int8/ck/instances/instance_n24576_k4096.cu — explicit
// cuda_runtime.h include forces hipify to rename .cu → .hip.
#include <cuda_runtime.h>

#include "../ck_w4a16_gemm.h"
// Placeholder: see ../ck_w4a16_gemm.h for why this TU has no
// instantiation. Kept as a separate TU to preserve the per-shape file
// layout demanded by the M4 spec, so a future worker dropping in a CK
// W4A16 device template only needs to edit one file per shape.

