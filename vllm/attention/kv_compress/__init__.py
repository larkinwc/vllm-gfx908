# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
SnapKV + PyramidKV: KV Cache Compression via Token Selection.

SnapKV (Li et al., NeurIPS 2024) identifies critical KV positions at prefill
time using an observation window, achieving ~92% compression with zero decode
overhead.

PyramidKV (2024) allocates more cache budget to lower layers (where attention
is diffuse) and less to upper layers, achieving ~88% compression at 12%
retention.

Combined, they can reduce effective KV usage by ~10x.

References:
    - SnapKV: https://arxiv.org/abs/2404.14469
    - PyramidKV: https://arxiv.org/abs/2406.02069
"""

from vllm.attention.kv_compress.combined import SnapPyramidKVCompressor
from vllm.attention.kv_compress.config import KVCompressConfig
from vllm.attention.kv_compress.pyramidkv import PyramidKVBudgetAllocator
from vllm.attention.kv_compress.snapkv import SnapKVSelector

__all__ = [
    "SnapKVSelector",
    "PyramidKVBudgetAllocator",
    "SnapPyramidKVCompressor",
    "KVCompressConfig",
]
