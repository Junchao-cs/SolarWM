# SPDX-License-Identifier: Apache-2.0 AND MIT
# Adapted from PRoPE (Projective Positional Encoding for Multiview
# Transformers), MIT, Copyright (c) the PRoPE authors. Modified by SolarWM
# for this backend's attention layout.
"""Torch implementation of the parameter-free LTX-2.5 PRoPE transform.

This module is intentionally heavy.  It is imported only by the embedded
runtime provider after the CUDA/LTX-Core environment has passed preflight.
The arithmetic follows the Stage0.5 ordering: native LTX RoPE
first, then this Q/K/V basis transform, one attention call, and the inverse
output-basis transform.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import torch

WAN_FIXED_FX_NORM = 969.6969696969696 / (960.0 * 2)
WAN_FIXED_FY_NORM = 969.6969696969696 / (540.0 * 2)


def normalize_translation_transform(value: object = "linear") -> str:
    selected = str(value).strip().lower()
    if selected not in {"linear", "logd4"}:
        raise ValueError("camera translation transform must be linear or logd4")
    return selected


def transform_relative_viewmats(
    viewmats: torch.Tensor,
    transform: object = "linear",
) -> torch.Tensor:
    """Apply the model-only linear/logd4 translation transform."""

    selected = normalize_translation_transform(transform)
    if not isinstance(viewmats, torch.Tensor) or tuple(viewmats.shape[-2:]) != (4, 4):
        raise ValueError("camera viewmats must be floating tensors ending in [4,4]")
    if not torch.is_floating_point(viewmats):
        raise TypeError("camera viewmats must be floating point")
    if selected == "linear":
        return viewmats
    translation = viewmats[..., :3, 3]
    compute_dtype = torch.float64 if viewmats.dtype == torch.float64 else torch.float32
    value = translation.to(dtype=compute_dtype)
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(torch.finfo(compute_dtype).tiny)
    scale = torch.where(
        norm > 0,
        torch.log1p(norm) / (4.0 * safe_norm),
        torch.zeros_like(norm),
    )
    output = viewmats.clone()
    output[..., :3, 3] = (value * scale).to(dtype=viewmats.dtype)
    return output


def _fixed_intrinsics_like(intrinsics: torch.Tensor) -> torch.Tensor:
    if not isinstance(intrinsics, torch.Tensor) or tuple(intrinsics.shape[-2:]) != (3, 3):
        raise ValueError("camera intrinsics must be floating tensors ending in [3,3]")
    if not torch.is_floating_point(intrinsics):
        raise TypeError("camera intrinsics must be floating point")
    fixed = torch.zeros_like(intrinsics)
    fixed[..., 0, 0] = WAN_FIXED_FX_NORM
    fixed[..., 1, 1] = WAN_FIXED_FY_NORM
    fixed[..., 0, 2] = 0.5
    fixed[..., 1, 2] = 0.5
    fixed[..., 2, 2] = 1.0
    return fixed


def prope_qkv(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
    camera_translation_transform: str = "linear",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Callable[[torch.Tensor], torch.Tensor],
]:
    """Transform equal ``[B,H,S,D]`` Q/K/V tensors into camera coordinates."""

    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("fused PRoPE requires equal [B,H,S,D] Q/K/V tensors")
    batch, _heads, sequence, head_dim = query.shape
    if tuple(viewmats.shape) != (batch, sequence, 4, 4):
        raise ValueError("PRoPE viewmats must be token-aligned [B,S,4,4]")
    if tuple(intrinsics.shape) != (batch, sequence, 3, 3):
        raise ValueError("PRoPE intrinsics must be token-aligned [B,S,3,3]")
    if head_dim % 4:
        raise ValueError("PRoPE attention head dimension must be divisible by four")
    transformed = transform_relative_viewmats(
        viewmats,
        camera_translation_transform,
    )
    apply_q, apply_kv, apply_output = _prepare_transforms(
        head_dim=head_dim,
        viewmats=transformed,
        intrinsics=intrinsics,
    )
    return apply_q(query), apply_kv(key), apply_kv(value), apply_output


def _prepare_transforms(
    *,
    head_dim: int,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
    Callable[[torch.Tensor], torch.Tensor],
]:
    if head_dim % 4:
        raise ValueError("PRoPE attention head dimension must be divisible by four")
    fixed = _fixed_intrinsics_like(intrinsics)
    normalized = torch.zeros_like(fixed)
    normalized[..., 0, 0] = fixed[..., 0, 0]
    normalized[..., 1, 1] = fixed[..., 1, 1]
    normalized[..., 2, 2] = 1.0
    projection = torch.einsum("...ij,...jk->...ik", _lift_k(normalized), viewmats)
    projection_t = projection.transpose(-1, -2).to(dtype=viewmats.dtype)
    projection_inv = torch.einsum(
        "...ij,...jk->...ik",
        _invert_se3(viewmats),
        _lift_k(_invert_k(normalized)),
    ).to(dtype=viewmats.dtype)
    return (
        partial(_apply_tiled_projection, matrix=projection_t),
        partial(_apply_tiled_projection, matrix=projection_inv),
        partial(_apply_tiled_projection, matrix=projection),
    )


def _apply_tiled_projection(
    features: torch.Tensor,
    *,
    matrix: torch.Tensor,
) -> torch.Tensor:
    batch, heads, sequence, feature_dim = features.shape
    dimension = int(matrix.shape[-1])
    if feature_dim % dimension:
        raise ValueError("PRoPE feature width is not divisible by its basis dimension")
    if tuple(matrix.shape) != (batch, sequence, dimension, dimension):
        raise ValueError("PRoPE projection matrix is not token aligned")
    tiled = features.view(batch, heads, sequence, feature_dim // dimension, dimension)
    return torch.einsum("btij,bntpj->bntpi", matrix, tiled).reshape(features.shape)


def _invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    rotation_inv = transforms[..., :3, :3].transpose(-1, -2)
    output = torch.zeros_like(transforms)
    output[..., :3, :3] = rotation_inv
    output[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i",
        rotation_inv,
        transforms[..., :3, 3],
    )
    output[..., 3, 3] = 1.0
    return output


def _lift_k(intrinsics: torch.Tensor) -> torch.Tensor:
    output = torch.zeros((*intrinsics.shape[:-2], 4, 4), device=intrinsics.device)
    output[..., :3, :3] = intrinsics
    output[..., 3, 3] = 1.0
    return output.to(dtype=intrinsics.dtype)


def _invert_k(intrinsics: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(intrinsics)
    output[..., 0, 0] = 1.0 / intrinsics[..., 0, 0]
    output[..., 1, 1] = 1.0 / intrinsics[..., 1, 1]
    output[..., 0, 2] = -intrinsics[..., 0, 2] / intrinsics[..., 0, 0]
    output[..., 1, 2] = -intrinsics[..., 1, 2] / intrinsics[..., 1, 1]
    output[..., 2, 2] = 1.0
    return output


__all__ = [
    "WAN_FIXED_FX_NORM",
    "WAN_FIXED_FY_NORM",
    "normalize_translation_transform",
    "prope_qkv",
    "transform_relative_viewmats",
]
