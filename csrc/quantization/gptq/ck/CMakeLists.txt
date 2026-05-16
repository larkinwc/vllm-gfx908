# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# CK W4A16 build glue for gfx908. Toggled by VLLM_BUILD_CK (default ON for
# gfx908 builds). The .cpp files in instances/ are header-only stubs today
# — CK on ROCm 7.12 does not ship a gfx908-correct W4A16 device template;
# see ck_w4a16_gemm.h for the documented limitation.

list(APPEND VLLM_ROCM_CK_GPTQ_SRC
  "csrc/quantization/gptq/ck/ck_w4a16_gemm.cu"
  "csrc/quantization/gptq/ck/instances/instance_n24576_k4096_g128.cu"
  "csrc/quantization/gptq/ck/instances/instance_n4096_k12288_g128.cu"
  "csrc/quantization/gptq/ck/instances/instance_n10240_k4096_g128.cu"
  "csrc/quantization/gptq/ck/instances/instance_n4096_k4096_g128.cu"
)
# Note: included via include() from top-level — no PARENT_SCOPE needed.
