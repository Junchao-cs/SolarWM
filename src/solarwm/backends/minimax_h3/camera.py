"""MiniMax-H3 camera normalization and fused-PRoPE suffix arithmetic.

The native MM-RoPE transform is owned by the transformer and must run first.
This module only transforms head dimensions ``[96:128)`` and preserves the
native prefix ``[0:96)`` exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

H3_ATTENTION_HEAD_DIM = 128
H3_NATIVE_MM_ROPE_DIM = 96
H3_CAMERA_PROPE_DIM_START = 96
H3_CAMERA_PROPE_DIM_END = 128
H3_CAMERA_TRANSLATION_TRANSFORM = "logd4"

# H3 uses the shared normalized Wan intrinsics.
WAN_FIXED_FX = 969.6969696969696 / (960.0 * 2.0)
WAN_FIXED_FY = 969.6969696969696 / (540.0 * 2.0)
WAN_FIXED_CX = 0.5
WAN_FIXED_CY = 0.5


@dataclass(frozen=True)
class H3CameraSuffixMatrices:
    """Token-aligned projective transforms for Q, K/V, and attention output."""

    query: np.ndarray
    key_value: np.ndarray
    output: np.ndarray


def h3_fused_prope_contract() -> dict[str, object]:
    """Return the checkpoint-visible H3 fused-PRoPE fingerprint."""

    return {
        "order": "native_mm_rope_then_camera_prope",
        "native_mm_rope_head_slice": [0, H3_NATIVE_MM_ROPE_DIM],
        "camera_prope_head_slice": [
            H3_CAMERA_PROPE_DIM_START,
            H3_CAMERA_PROPE_DIM_END,
        ],
        "relative_translation_transform": H3_CAMERA_TRANSLATION_TRANSFORM,
        "relative_pose": "first_frame_relative_w2c",
        "input_pose": "absolute_c2w",
        "input_intrinsics": "normalized_K",
        "runtime_intrinsics": "wan_fixed_focal_only",
    }


def _floating_array(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if not np.issubdtype(result.dtype, np.floating):
        raise TypeError(f"{name} must use a floating dtype, got {result.dtype}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def validate_absolute_c2w(c2w: object, *, atol: float = 1e-4) -> np.ndarray:
    """Validate authoritative absolute camera-to-world matrices."""

    poses = _floating_array(c2w, name="c2w")
    if poses.ndim not in (3, 4) or poses.shape[-2:] != (4, 4):
        raise ValueError(f"c2w must be [T,4,4] or [B,T,4,4], got {poses.shape}")
    expected_bottom = np.zeros_like(poses[..., 3, :])
    expected_bottom[..., 3] = 1
    if not np.allclose(poses[..., 3, :], expected_bottom, atol=atol, rtol=0):
        raise ValueError("c2w homogeneous bottom row must be [0,0,0,1]")
    rotation = poses[..., :3, :3]
    identity = np.eye(3, dtype=rotation.dtype)
    gram = np.swapaxes(rotation, -1, -2) @ rotation
    if not np.allclose(gram, identity, atol=atol, rtol=0):
        raise ValueError("c2w rotations must be orthonormal")
    determinant = np.linalg.det(rotation)
    if not np.allclose(determinant, 1.0, atol=atol, rtol=0):
        raise ValueError("c2w rotations must be proper rotations with determinant +1")
    return poses


def validate_normalized_intrinsics(Ks: object, *, atol: float = 1e-5) -> np.ndarray:
    """Validate normalized, no-skew pinhole intrinsics."""

    intrinsics = _floating_array(Ks, name="K")
    if intrinsics.ndim not in (3, 4) or intrinsics.shape[-2:] != (3, 3):
        raise ValueError(f"K must be [T,3,3] or [B,T,3,3], got {intrinsics.shape}")
    focal = intrinsics[..., (0, 1), (0, 1)]
    if np.any(focal <= 0) or np.any(focal > 4):
        raise ValueError("normalized K requires 0 < fx,fy <= 4")
    if not (
        np.allclose(intrinsics[..., 0, 1], 0, atol=atol, rtol=0)
        and np.allclose(intrinsics[..., 1, 0], 0, atol=atol, rtol=0)
        and np.allclose(intrinsics[..., 2, :2], 0, atol=atol, rtol=0)
        and np.allclose(intrinsics[..., 2, 2], 1, atol=atol, rtol=0)
    ):
        raise ValueError("normalized K must be no-skew with bottom row [0,0,1]")
    return intrinsics


def invert_se3(transforms: object) -> np.ndarray:
    """Invert rigid transforms without a generic matrix inverse."""

    values = _floating_array(transforms, name="transforms")
    if values.shape[-2:] != (4, 4):
        raise ValueError(f"transforms must end in [4,4], got {values.shape}")
    output = np.zeros_like(values)
    rotation_inv = np.swapaxes(values[..., :3, :3], -1, -2)
    output[..., :3, :3] = rotation_inv
    output[..., :3, 3] = -np.einsum("...ij,...j->...i", rotation_inv, values[..., :3, 3])
    output[..., 3, 3] = 1
    return output


def first_frame_relative_w2c(c2w: object) -> np.ndarray:
    """Convert absolute C2W to first-frame-relative W2C.

    For each frame this computes ``inverse(c2w[t]) @ c2w[0]``.  The first
    result is identity, and there is no axis flip.
    """

    poses = validate_absolute_c2w(c2w)
    if poses.ndim == 3:
        result = invert_se3(poses) @ poses[0]
        result[0] = np.eye(4, dtype=result.dtype)
        return result
    result = invert_se3(poses) @ poses[:, :1]
    result[:, 0] = np.eye(4, dtype=result.dtype)
    return result


def logd4_relative_viewmats(viewmats: object) -> np.ndarray:
    """Apply zero-safe ``t*log1p(||t||)/(4*||t||)`` in FP32 or FP64."""

    values = _floating_array(viewmats, name="viewmats")
    if values.shape[-2:] != (4, 4):
        raise ValueError(f"viewmats must end in [4,4], got {values.shape}")
    output = values.copy()
    compute_dtype = np.float64 if values.dtype == np.float64 else np.float32
    translation = values[..., :3, 3].astype(compute_dtype, copy=False)
    radius = np.linalg.norm(translation, axis=-1, keepdims=True)
    scale = np.zeros_like(radius)
    np.divide(np.log1p(radius), 4.0 * radius, out=scale, where=radius != 0)
    compressed = translation * scale
    output[..., :3, 3] = compressed.astype(values.dtype, copy=False)
    return output


def wan_fixed_intrinsics_like(Ks: object) -> np.ndarray:
    """Return fixed normalized K with the same shape and dtype."""

    reference = _floating_array(Ks, name="K")
    if reference.shape[-2:] != (3, 3):
        raise ValueError(f"K must end in [3,3], got {reference.shape}")
    output = np.zeros_like(reference)
    output[..., 0, 0] = WAN_FIXED_FX
    output[..., 1, 1] = WAN_FIXED_FY
    output[..., 0, 2] = WAN_FIXED_CX
    output[..., 1, 2] = WAN_FIXED_CY
    output[..., 2, 2] = 1
    return output


def _lift_K(Ks: np.ndarray) -> np.ndarray:
    output = np.zeros((*Ks.shape[:-2], 4, 4), dtype=Ks.dtype)
    output[..., :3, :3] = Ks
    output[..., 3, 3] = 1
    return output


def _invert_K(Ks: np.ndarray) -> np.ndarray:
    output = np.zeros_like(Ks)
    output[..., 0, 0] = 1.0 / Ks[..., 0, 0]
    output[..., 1, 1] = 1.0 / Ks[..., 1, 1]
    output[..., 0, 2] = -Ks[..., 0, 2] / Ks[..., 0, 0]
    output[..., 1, 2] = -Ks[..., 1, 2] / Ks[..., 1, 1]
    output[..., 2, 2] = 1
    return output


def prepare_camera_suffix_matrices(
    viewmats: object,
    Ks: object,
) -> H3CameraSuffixMatrices:
    """Prepare H3 logd4/fixed-K projective matrices."""

    views = logd4_relative_viewmats(viewmats)
    intrinsics = validate_normalized_intrinsics(Ks)
    if views.ndim != 4 or views.shape[-2:] != (4, 4):
        raise ValueError(f"viewmats must be token-aligned [B,S,4,4], got {views.shape}")
    if intrinsics.shape != (*views.shape[:2], 3, 3):
        raise ValueError(
            f"K must be token-aligned [B,S,3,3], got {intrinsics.shape} for viewmats {views.shape}"
        )
    fixed = wan_fixed_intrinsics_like(intrinsics)
    # H3 consumes only fx/fy; cx/cy are cleared here.
    focal_only = np.zeros_like(fixed)
    focal_only[..., 0, 0] = fixed[..., 0, 0]
    focal_only[..., 1, 1] = fixed[..., 1, 1]
    focal_only[..., 2, 2] = 1
    projection = _lift_K(focal_only) @ views
    projection_inv = invert_se3(views) @ _lift_K(_invert_K(focal_only))
    return H3CameraSuffixMatrices(
        query=np.swapaxes(projection, -1, -2).astype(views.dtype, copy=False),
        key_value=projection_inv.astype(views.dtype, copy=False),
        output=projection.astype(views.dtype, copy=False),
    )


def _apply_token_projection(features: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    batch, heads, sequence, feature_dim = features.shape
    projective_dim = int(matrix.shape[-1])
    if feature_dim % projective_dim:
        raise ValueError(f"camera PRoPE suffix {feature_dim} is not divisible by {projective_dim}")
    if matrix.shape != (batch, sequence, projective_dim, projective_dim):
        raise ValueError("camera PRoPE projection matrix must be token-aligned")
    tiled = features.reshape(batch, heads, sequence, feature_dim // projective_dim, projective_dim)
    output = np.einsum("bsij,bhspj->bhspi", matrix, tiled)
    return output.reshape(features.shape)


def _apply_camera_suffix(features: object, matrix: np.ndarray) -> np.ndarray:
    values = _floating_array(features, name="attention features")
    if values.ndim != 4 or values.shape[-1] != H3_ATTENTION_HEAD_DIM:
        raise ValueError("H3 fused PRoPE requires [B,H,S,128] attention features")
    native = values[..., :H3_CAMERA_PROPE_DIM_START]
    camera = values[..., H3_CAMERA_PROPE_DIM_START:H3_CAMERA_PROPE_DIM_END]
    return np.concatenate((native, _apply_token_projection(camera, matrix)), axis=-1)


def prope_qkv(
    q: object,
    k: object,
    v: object,
    *,
    viewmats: object,
    Ks: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Callable[[object], np.ndarray]]:
    """Apply camera PRoPE to suffix ``[96:128)`` after native MM-RoPE."""

    query = _floating_array(q, name="q")
    key = _floating_array(k, name="k")
    value = _floating_array(v, name="v")
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("H3 fused PRoPE requires equal [B,H,S,D] q/k/v tensors")
    if query.shape[-1] != H3_ATTENTION_HEAD_DIM:
        raise ValueError(f"H3 fused PRoPE requires attention head_dim=128, got {query.shape[-1]}")
    matrices = prepare_camera_suffix_matrices(viewmats, Ks)
    if matrices.query.shape[:2] != (query.shape[0], query.shape[2]):
        raise ValueError("viewmats/K must align to the q/k/v batch and sequence")

    def apply_output(features: object) -> np.ndarray:
        return _apply_camera_suffix(features, matrices.output)

    return (
        _apply_camera_suffix(query, matrices.query),
        _apply_camera_suffix(key, matrices.key_value),
        _apply_camera_suffix(value, matrices.key_value),
        apply_output,
    )


def select_camera_rows(
    relative_viewmats: object,
    intrinsics: object,
    camera_frame_ids: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand per-latent cameras to explicit anchor/target VisualVAE rows."""

    views = _floating_array(relative_viewmats, name="relative_viewmats")
    Ks = _floating_array(intrinsics, name="K")
    frame_ids = np.asarray(camera_frame_ids, dtype=np.int64)
    if views.ndim == 3:
        views = views[None]
    if Ks.ndim == 3:
        Ks = Ks[None]
    if views.ndim != 4 or views.shape[-2:] != (4, 4):
        raise ValueError("relative_viewmats must be [T,4,4] or [B,T,4,4]")
    if Ks.shape != (*views.shape[:2], 3, 3):
        raise ValueError("K and relative_viewmats must have identical batch/time axes")
    if frame_ids.ndim != 1 or np.any(frame_ids < 0) or np.any(frame_ids >= views.shape[1]):
        raise ValueError("camera_frame_ids are outside the latent camera track")
    return views[:, frame_ids], Ks[:, frame_ids]


__all__ = [
    "H3_ATTENTION_HEAD_DIM",
    "H3_CAMERA_PROPE_DIM_END",
    "H3_CAMERA_PROPE_DIM_START",
    "H3_CAMERA_TRANSLATION_TRANSFORM",
    "H3_NATIVE_MM_ROPE_DIM",
    "WAN_FIXED_CX",
    "WAN_FIXED_CY",
    "WAN_FIXED_FX",
    "WAN_FIXED_FY",
    "H3CameraSuffixMatrices",
    "first_frame_relative_w2c",
    "h3_fused_prope_contract",
    "invert_se3",
    "logd4_relative_viewmats",
    "prepare_camera_suffix_matrices",
    "prope_qkv",
    "select_camera_rows",
    "validate_absolute_c2w",
    "validate_normalized_intrinsics",
    "wan_fixed_intrinsics_like",
]


def fixed_intrinsics_like(reference: object) -> object:
    """Build the released camera intrinsics on a tensor's device and dtype."""
    output = reference.new_zeros(reference.shape)
    output[..., 0, 0] = WAN_FIXED_FX
    output[..., 1, 1] = WAN_FIXED_FY
    output[..., 0, 2] = WAN_FIXED_CX
    output[..., 1, 2] = WAN_FIXED_CY
    output[..., 2, 2] = 1.0
    return output
