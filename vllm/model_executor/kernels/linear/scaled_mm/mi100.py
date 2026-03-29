# SPDX-License-Identifier: Apache-2.0
# MI100 (gfx908) FP8 emulation: dequant FP8 -> FP16, then rocBLAS GEMM.
# MI100 lacks native FP8 hardware so we use a software dequant path.

import os

import torch

from vllm.platforms import current_platform

from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

_mi100_debug_logged = False
_MI100_USE_PYTORCH = os.environ.get('MI100_USE_HIP_KERNEL', '0') != '1'


class MI100FP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_rocm():
            return False, 'requires ROCm.'
        from vllm.platforms.rocm import on_mi100
        if not on_mi100():
            return False, 'requires MI100 (gfx908).'
        return True, None

    @classmethod
    def can_implement(
        cls, c: FP8ScaledMMLinearLayerConfig
    ) -> tuple[bool, str | None]:
        per_tensor_a = c.activation_quant_key.scale.group_shape.is_per_tensor()
        per_tensor_w = c.weight_quant_key.scale.group_shape.is_per_tensor()
        if not (per_tensor_a and per_tensor_w):
            return False, 'requires per tensor activation and weight scales.'
        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        global _mi100_debug_logged
        if not _mi100_debug_logged:
            import sys
            print('[MI100_FP8] Using dequant+rocBLAS path (no native FP8)',
                  file=sys.stderr, flush=True)
            _mi100_debug_logged = True

        input_fp16 = A.to(torch.float16) * As.to(torch.float16)
        weight_fp16 = B.to(torch.float16) * Bs.to(torch.float16)
        output = torch.mm(input_fp16, weight_fp16)

        if bias is not None:
            output = output + bias

        output = output.to(out_dtype)
        return torch.narrow(output, 0, 0, output_shape[0]).view(*output_shape)
