"""Torch camera transforms at the Wan PRoPE runtime boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

CAMERA_TRANSLATION_LINEAR = "linear"
CAMERA_TRANSLATION_LOGD4 = "logd4"
CAMERA_TRANSLATION_TRANSFORMS = frozenset({CAMERA_TRANSLATION_LINEAR, CAMERA_TRANSLATION_LOGD4})


def normalize_camera_translation_transform(
    value: object = CAMERA_TRANSLATION_LINEAR,
) -> str:
    """Validate the runtime-only camera translation transform."""

    normalized = str(value).strip().lower()
    if normalized not in CAMERA_TRANSLATION_TRANSFORMS:
        raise ValueError(
            "camera_translation_transform must be one of "
            f"{sorted(CAMERA_TRANSLATION_TRANSFORMS)}, got {value!r}"
        )
    return normalized


def transform_relative_viewmats(
    viewmats: torch.Tensor,
    transform: object = CAMERA_TRANSLATION_LINEAR,
) -> torch.Tensor:
    """Apply linear or logd4 to first-frame-relative W2C translation.

    The stored camera remains authoritative. Compression is applied only when
    PRoPE consumes it, using FP32 norm arithmetic (FP64 for FP64 input).
    """

    import torch

    selected = normalize_camera_translation_transform(transform)
    if not isinstance(viewmats, torch.Tensor):
        raise TypeError("camera viewmats must be a torch.Tensor")
    if tuple(viewmats.shape[-2:]) != (4, 4):
        raise ValueError(f"camera viewmats must end in [4,4], got {tuple(viewmats.shape)}")
    if not torch.is_floating_point(viewmats):
        raise TypeError("camera viewmats must use a floating dtype")
    if selected == CAMERA_TRANSLATION_LINEAR:
        return viewmats

    translation = viewmats[..., :3, 3]
    compute_dtype = torch.float64 if viewmats.dtype == torch.float64 else torch.float32
    working = translation.to(dtype=compute_dtype)
    norm = torch.linalg.vector_norm(working, dim=-1, keepdim=True)
    safe_norm = norm.clamp_min(torch.finfo(compute_dtype).tiny)
    scale = torch.where(
        norm > 0,
        torch.log1p(norm) / (4.0 * safe_norm),
        torch.zeros_like(norm),
    )
    result = viewmats.clone()
    result[..., :3, 3] = (working * scale).to(dtype=viewmats.dtype)
    return result


__all__ = [
    "CAMERA_TRANSLATION_LINEAR",
    "CAMERA_TRANSLATION_LOGD4",
    "CAMERA_TRANSLATION_TRANSFORMS",
    "normalize_camera_translation_transform",
    "transform_relative_viewmats",
]
