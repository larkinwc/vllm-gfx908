# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# CK INT8 GEMM build glue for gfx908. Toggled by VLLM_BUILD_CK (default ON
# for gfx908 builds, see top-level CMakeLists.txt). Source files are
# appended to VLLM_ROCM_CK_INT8_SRC, which the parent CMakeLists.txt mixes
# into VLLM_ROCM_EXT_SRC for the _rocm_C extension.

# Paths are relative to the project source root (vllm/), to match the
# convention in the top-level CMakeLists.txt and to allow hipify_sources_target
# to map them into ${CMAKE_BINARY_DIR}/csrc/... correctly.

list(APPEND VLLM_ROCM_CK_INT8_SRC
  "csrc/quantization/w8a8/int8/ck/ck_int8_gemm_dispatch.cu"
  "csrc/quantization/w8a8/int8/ck/instances/instance_n24576_k4096.cu"
  "csrc/quantization/w8a8/int8/ck/instances/instance_n4096_k12288.cu"
  "csrc/quantization/w8a8/int8/ck/instances/instance_n10240_k4096.cu"
  "csrc/quantization/w8a8/int8/ck/instances/instance_n4096_k4096.cu"
)
# Note: this file is `include()`d directly from the top-level CMakeLists.txt,
# so we do NOT need PARENT_SCOPE — variables set here live in the same scope.
