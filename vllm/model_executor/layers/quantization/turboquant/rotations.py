# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-diagonal rotations for TurboQuant (RotorQuant family).

TurboQuant decorrelates KV-cache vectors with a dense ``D x D`` Walsh-Hadamard
rotation before scalar quantization. That rotation costs ``O(D log D)`` (as a
butterfly) or ``O(D^2)`` (as the GEMM this backend actually uses) and stores a
``D x D`` matrix. RotorQuant (Pope, 2026) and the related IsoQuant/PlanarQuant
work (ParaMind2025) observe that real attention vectors live on low-rank
manifolds, so a *block-diagonal* orthonormal rotation made of tiny independent
blocks decorrelates them just as well at ``O(D)`` cost with ``O(D)`` parameters:

    planar : D/2 independent 2x2 Givens rotations  (256 FMAs at D=128)
    iso    : D/4 independent 4x4 quaternion rotations (512 FMAs at D=128)

This module builds the dense ``[D, D]`` forward matrix ``PiT`` (used by the
GEMM/continuation paths and for cache allocation symmetry) together with the
compact per-block parameters consumed by the fused Triton kernels. ``Pi`` is the
inverse rotation and equals ``PiT.T`` because every block is orthonormal.

Convention (matches the existing Hadamard path):
    store  : ``y      = x_hat @ PiT``   (forward rotate normalized keys)
    decode : ``q_rot  = q     @ PiT``   (rotate the query the same way)
    dequant: ``x_hat  = y_q   @ Pi``    (inverse rotate centroid coords)

For Hadamard ``PiT == Pi`` (symmetric); for the block rotations they are a
genuine transpose pair, which the rest of the backend already handles because
it threads both ``Pi`` and ``PiT`` through explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import torch

# Rotation kinds understood by the config / kernels.
ROTATION_HADAMARD = "hadamard"
ROTATION_PLANAR = "planar"
ROTATION_ISO = "iso"

ROTATION_KINDS = (ROTATION_HADAMARD, ROTATION_PLANAR, ROTATION_ISO)

# Block size (coordinates rotated together) for each kind.
_BLOCK_SIZE = {
    ROTATION_PLANAR: 2,
    ROTATION_ISO: 4,
}

# Fixed seed: the rotation must be identical between store and decode, across
# processes (vLLM spawns workers) and across restarts, so it is derived
# deterministically from (kind, D) only — never from a runtime RNG.
_ROT_SEED = 1234567


@dataclass(frozen=True)
class RotationParams:
    """Compact per-block rotation parameters for the fused kernels.

    Attributes:
        kind: One of ``ROTATION_KINDS``.
        block: Coordinates per block (1 for Hadamard, 2 planar, 4 iso).
        cos, sin: ``[D/2]`` tensors for ``planar`` (one angle per pair).
            ``None`` for other kinds.
        quat: ``[D/4, 4]`` unit quaternions for ``iso``. ``None`` otherwise.
    """

    kind: str
    block: int
    cos: torch.Tensor | None = None
    sin: torch.Tensor | None = None
    quat: torch.Tensor | None = None


def _planar_angles(d: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(_ROT_SEED + d)
    return torch.rand(d // 2, generator=g, dtype=torch.float64) * (2.0 * math.pi)


def _iso_quaternions(d: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(_ROT_SEED + 7 * d)
    q = torch.randn(d // 4, 4, generator=g, dtype=torch.float64)
    return q / q.norm(dim=1, keepdim=True)


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """4x4 (left-isoclinic) orthonormal rotation from a unit quaternion."""
    w, x, y, z = q.tolist()
    return torch.tensor(
        [
            [w, -x, -y, -z],
            [x, w, -z, y],
            [y, z, w, -x],
            [z, -y, x, w],
        ],
        dtype=torch.float64,
    )


def _hadamard(d: int) -> torch.Tensor:
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < d:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(d)


def _build_dense_pit(kind: str, d: int) -> torch.Tensor:
    """Dense ``[D, D]`` forward-rotation matrix ``PiT`` for ``y = x_hat @ PiT``.

    The matrix is assembled in float64 for orthonormality, returned float32.
    """
    if kind == ROTATION_HADAMARD:
        return _hadamard(d).float()

    pit = torch.zeros(d, d, dtype=torch.float64)
    if kind == ROTATION_PLANAR:
        if d % 2 != 0:
            raise ValueError(f"planar rotation needs even head_dim, got {d}")
        angles = _planar_angles(d)
        c, s = torch.cos(angles), torch.sin(angles)
        for i in range(d // 2):
            r, cc = 2 * i, 2 * i + 1
            # PiT block = [[c, -s], [s, c]] so y = x_hat @ PiT is a +theta
            # rotation of each (2i, 2i+1) coordinate pair.
            pit[r, r] = c[i]
            pit[cc, r] = s[i]
            pit[r, cc] = -s[i]
            pit[cc, cc] = c[i]
        return pit.float()

    if kind == ROTATION_ISO:
        if d % 4 != 0:
            raise ValueError(f"iso rotation needs head_dim divisible by 4, got {d}")
        quats = _iso_quaternions(d)
        for b in range(d // 4):
            m = _quat_to_matrix(quats[b])  # 4x4 orthonormal
            # PiT block = m.T so that y = x_hat @ PiT applies m on the left,
            # matching the dequant convention x_hat = y_q @ Pi with Pi = PiT.T.
            pit[4 * b : 4 * b + 4, 4 * b : 4 * b + 4] = m.T
        return pit.float()

    raise ValueError(f"unknown rotation kind: {kind!r}")


@lru_cache(maxsize=64)
def _cached_pit(kind: str, d: int, device_str: str) -> torch.Tensor:
    return _build_dense_pit(kind, d).to(torch.device(device_str))


def build_rotation(kind: str, d: int, device: torch.device | str) -> torch.Tensor:
    """Return the dense ``[D, D]`` float32 forward rotation ``PiT`` (cached)."""
    if kind not in ROTATION_KINDS:
        raise ValueError(f"unknown rotation kind {kind!r}; valid: {ROTATION_KINDS}")
    return _cached_pit(kind, d, str(torch.device(device)))


@lru_cache(maxsize=64)
def _cached_params(kind: str, d: int, device_str: str) -> RotationParams:
    device = torch.device(device_str)
    if kind == ROTATION_PLANAR:
        angles = _planar_angles(d)
        return RotationParams(
            kind=kind,
            block=2,
            cos=torch.cos(angles).float().to(device),
            sin=torch.sin(angles).float().to(device),
        )
    if kind == ROTATION_ISO:
        quats = _iso_quaternions(d)
        return RotationParams(
            kind=kind,
            block=4,
            quat=quats.float().to(device),
        )
    # Hadamard has no compact block form; the fused kernels fall back to the
    # dense GEMM path for it.
    return RotationParams(kind=ROTATION_HADAMARD, block=1)


def get_rotation_params(
    kind: str, d: int, device: torch.device | str
) -> RotationParams:
    """Compact per-block rotation parameters for the fused Triton kernels."""
    if kind not in ROTATION_KINDS:
        raise ValueError(f"unknown rotation kind {kind!r}; valid: {ROTATION_KINDS}")
    return _cached_params(kind, d, str(torch.device(device)))


def block_rotate(x: torch.Tensor, params: RotationParams) -> torch.Tensor:
    """Apply the forward block rotation ``y = x @ PiT`` in ``O(D)`` time.

    Vectorized reference/host path used when the fused kernel is not active
    (e.g. the continuation-prefill GEMM replacement). ``x`` is ``[..., D]``.
    """
    if params.kind == ROTATION_PLANAR:
        *lead, d = x.shape
        xp = x.reshape(*lead, d // 2, 2)
        x0, x1 = xp[..., 0], xp[..., 1]
        assert params.cos is not None and params.sin is not None
        c, s = params.cos, params.sin
        y0 = c * x0 + s * x1
        y1 = -s * x0 + c * x1
        return torch.stack((y0, y1), dim=-1).reshape(*lead, d)
    if params.kind == ROTATION_ISO:
        *lead, d = x.shape
        xq = x.reshape(*lead, d // 4, 4)
        # Forward map applies m on the left of each 4-vector: y = m @ x.
        assert params.quat is not None
        w = params.quat[:, 0]
        xx = params.quat[:, 1]
        yy = params.quat[:, 2]
        zz = params.quat[:, 3]
        a, b, c, e = xq[..., 0], xq[..., 1], xq[..., 2], xq[..., 3]
        y0 = w * a - xx * b - yy * c - zz * e
        y1 = xx * a + w * b - zz * c + yy * e
        y2 = yy * a + zz * b + w * c - xx * e
        y3 = zz * a - yy * b + xx * c + w * e
        return torch.stack((y0, y1, y2, y3), dim=-1).reshape(*lead, d)
    raise ValueError(f"block_rotate does not support kind {params.kind!r}")
