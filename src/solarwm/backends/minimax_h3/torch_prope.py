# SPDX-License-Identifier: Apache-2.0 AND MIT
# Adapted from PRoPE (Projective Positional Encoding for Multiview
# Transformers), MIT, Copyright (c) the PRoPE authors. Modified by SolarWM
# for this backend's attention layout.
"""CUDA/torch implementation of the H3 camera-PRoPE suffix."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from .camera import (
    H3_ATTENTION_HEAD_DIM,
    H3_CAMERA_PROPE_DIM_END,
    H3_CAMERA_PROPE_DIM_START,
    WAN_FIXED_FX,
    WAN_FIXED_FY,
)


def logd4_relative_viewmats(viewmats: Any) -> Any:
    import torch

    if not isinstance(viewmats, torch.Tensor):
        raise TypeError("viewmats must be a torch.Tensor")
    if tuple(viewmats.shape[-2:]) != (4, 4) or not viewmats.is_floating_point():
        raise ValueError("viewmats must be floating tensors ending in [4,4]")
    compute_dtype = torch.float64 if viewmats.dtype == torch.float64 else torch.float32
    translation = viewmats[..., :3, 3].to(compute_dtype)
    norm = torch.linalg.vector_norm(translation, dim=-1, keepdim=True)
    safe = norm.clamp_min(torch.finfo(compute_dtype).tiny)
    scale = torch.where(
        norm > 0,
        torch.log1p(norm) / (4.0 * safe),
        torch.zeros_like(norm),
    )
    output = viewmats.clone()
    output[..., :3, 3] = (translation * scale).to(viewmats.dtype)
    return output


def _fixed_focal_K(reference: Any) -> Any:
    output = reference.new_zeros(reference.shape)
    output[..., 0, 0] = WAN_FIXED_FX
    output[..., 1, 1] = WAN_FIXED_FY
    output[..., 2, 2] = 1.0
    return output


def _invert_se3(value: Any) -> Any:
    output = value.new_zeros(value.shape)
    rotation = value[..., :3, :3].transpose(-1, -2)
    output[..., :3, :3] = rotation
    output[..., :3, 3] = -__import__("torch").einsum(
        "...ij,...j->...i", rotation, value[..., :3, 3]
    )
    output[..., 3, 3] = 1.0
    return output


def _lift_K(value: Any) -> Any:
    output = value.new_zeros((*value.shape[:-2], 4, 4))
    output[..., :3, :3] = value
    output[..., 3, 3] = 1.0
    return output


def _invert_K(value: Any) -> Any:
    output = value.new_zeros(value.shape)
    output[..., 0, 0] = 1.0 / value[..., 0, 0]
    output[..., 1, 1] = 1.0 / value[..., 1, 1]
    output[..., 0, 2] = -value[..., 0, 2] / value[..., 0, 0]
    output[..., 1, 2] = -value[..., 1, 2] / value[..., 1, 1]
    output[..., 2, 2] = 1.0
    return output


def _project(features: Any, *, matrix: Any) -> Any:
    import torch

    batch, heads, sequence, width = features.shape
    projective = int(matrix.shape[-1])
    if width % projective or tuple(matrix.shape) != (
        batch,
        sequence,
        projective,
        projective,
    ):
        raise ValueError("camera projection is not aligned with attention rows")
    matrix = matrix.to(device=features.device, dtype=features.dtype)
    tiled = features.reshape(batch, heads, sequence, width // projective, projective)
    return torch.einsum("bsij,bhspj->bhspi", matrix, tiled).reshape(features.shape)


def prope_qkv(
    q: Any,
    k: Any,
    v: Any,
    *,
    viewmats: Any,
    Ks: Any,
    **_unused: object,
) -> tuple[Any, Any, Any, Callable[[Any], Any]]:
    """Apply logd4 camera PRoPE only to H3 head dimensions ``[96:128)``."""

    import torch

    if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must be equal [B,H,S,D] tensors")
    batch, _heads, sequence, head_dim = q.shape
    if head_dim != H3_ATTENTION_HEAD_DIM:
        raise ValueError("H3 camera PRoPE requires head_dim=128")
    if tuple(viewmats.shape) != (batch, sequence, 4, 4):
        raise ValueError("viewmats must be token-aligned [B,S,4,4]")
    if tuple(Ks.shape) != (batch, sequence, 3, 3):
        raise ValueError("K must be token-aligned [B,S,3,3]")
    views = logd4_relative_viewmats(viewmats)
    fixed = _fixed_focal_K(Ks)
    projection = torch.einsum("...ij,...jk->...ik", _lift_K(fixed), views)
    query_matrix = projection.transpose(-1, -2).to(views.dtype)
    kv_matrix = torch.einsum(
        "...ij,...jk->...ik", _invert_se3(views), _lift_K(_invert_K(fixed))
    ).to(views.dtype)

    def suffix(features: Any, transform: Callable[[Any], Any]) -> Any:
        native = features[..., :H3_CAMERA_PROPE_DIM_START]
        camera = features[..., H3_CAMERA_PROPE_DIM_START:H3_CAMERA_PROPE_DIM_END]
        return torch.cat((native, transform(camera)), dim=-1)

    apply_q = partial(_project, matrix=query_matrix)
    apply_kv = partial(_project, matrix=kv_matrix)
    apply_out = partial(_project, matrix=projection.to(views.dtype))
    return (
        suffix(q, apply_q),
        suffix(k, apply_kv),
        suffix(v, apply_kv),
        partial(suffix, transform=apply_out),
    )


__all__ = ["logd4_relative_viewmats", "prope_qkv"]


def _prepare_suffix_transforms(
    *,
    viewmats: Any,
    Ks: Any,
) -> tuple[
    Callable[[Any], Any],
    Callable[[Any], Any],
    Callable[[Any], Any],
]:
    import torch

    fixed_K = _fixed_focal_K(Ks)
    normalized_K = torch.zeros_like(fixed_K)
    normalized_K[..., 0, 0] = fixed_K[..., 0, 0]
    normalized_K[..., 1, 1] = fixed_K[..., 1, 1]
    normalized_K[..., 2, 2] = 1.0

    projection = torch.einsum("...ij,...jk->...ik", _lift_K(normalized_K), viewmats)
    projection_t = projection.transpose(-1, -2).to(dtype=viewmats.dtype)
    projection_inv = torch.einsum(
        "...ij,...jk->...ik",
        _invert_se3(viewmats),
        _lift_K(_invert_K(normalized_K)),
    ).to(dtype=viewmats.dtype)
    return (
        partial(_project, matrix=projection_t),
        partial(_project, matrix=projection_inv),
        partial(_project, matrix=projection),
    )


def prope_qkv_separate(q, k, v, *, query_viewmats, query_K, key_viewmats, key_K):
    """H3's existing head-sliced PRoPE with different query and cached KV rows."""
    import torch

    if q.ndim != 4 or k.ndim != 4 or k.shape != v.shape:
        raise ValueError("H3 cached PRoPE requires BHSD queries and equal BHSD keys/values")
    if q.shape[:2] != k.shape[:2] or q.shape[-1] != 128 or k.shape[-1] != 128:
        raise ValueError("H3 cached PRoPE requires matching batch/heads and head_dim=128")
    apply_q, _, apply_output = _prepare_suffix_transforms(
        viewmats=logd4_relative_viewmats(query_viewmats.to(q.dtype)), Ks=query_K.to(q.dtype)
    )
    _, apply_kv, _ = _prepare_suffix_transforms(
        viewmats=logd4_relative_viewmats(key_viewmats.to(k.dtype)), Ks=key_K.to(k.dtype)
    )

    def suffix(features, transform):
        return torch.cat((features[..., :96], transform(features[..., 96:128])), dim=-1)

    return (
        suffix(q, apply_q),
        suffix(k, apply_kv),
        suffix(v, apply_kv),
        partial(suffix, transform=apply_output),
    )
