"""LTX-2.5 camera preparation and latent/token alignment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solar.data.camera import relative_w2c, transform_translation
from solar.errors import BackendContractError, DataContractError

from .geometry import STABLE_GEOMETRY, CoverResize, transform_intrinsics


def _se3(value: object, *, rows: int, name: str) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float64)
    if matrices.shape != (rows, 4, 4) or not np.isfinite(matrices).all():
        raise BackendContractError(f"{name} must be finite [{rows},4,4]")
    if not np.allclose(matrices[:, 3], (0.0, 0.0, 0.0, 1.0), atol=1e-5, rtol=0.0):
        raise BackendContractError(f"{name} bottom rows must be [0,0,0,1]")
    rotations = matrices[:, :3, :3]
    gram = np.swapaxes(rotations, -1, -2) @ rotations
    if not np.allclose(gram, np.eye(3), atol=2e-3, rtol=0.0):
        raise BackendContractError(f"{name} rotations must be orthonormal")
    if not np.allclose(np.linalg.det(rotations), 1.0, atol=2e-3, rtol=0.0):
        raise BackendContractError(f"{name} rotations must have determinant +1")
    return matrices


def _intrinsics(value: object, *, rows: int) -> np.ndarray:
    matrices = np.asarray(value, dtype=np.float64)
    if matrices.shape != (rows, 3, 3) or not np.isfinite(matrices).all():
        raise BackendContractError(f"camera_K must be finite [{rows},3,3]")
    if np.any(matrices[:, 0, 0] <= 0) or np.any(matrices[:, 1, 1] <= 0):
        raise BackendContractError("camera_K focal lengths must be positive")
    if not np.allclose(matrices[:, 2], (0.0, 0.0, 1.0), atol=1e-5, rtol=0.0):
        raise BackendContractError("camera_K bottom rows must be [0,0,1]")
    if not np.allclose(matrices[:, (0, 1), (1, 0)], 0.0, atol=1e-5, rtol=0.0):
        raise BackendContractError("camera_K must use zero skew")
    if np.any(np.abs(matrices[:, (0, 1), (2, 2)]) > 4.0):
        raise BackendContractError("camera_K principal points are not normalized")
    result = matrices.astype(np.float32)
    if not np.isfinite(result).all():
        raise BackendContractError("camera_K cannot be represented as finite FP32")
    return result


def canonicalize_signed_focal_gauge(
    absolute_c2w: object,
    intrinsics: object,
) -> tuple[np.ndarray, np.ndarray]:
    """Make signed focal entries positive without changing pixel projection."""

    pose = _se3(np.asarray(absolute_c2w)[None], rows=1, name="absolute_c2w")[0]
    K = np.asarray(intrinsics, dtype=np.float64)
    if K.shape != (3, 3) or not np.isfinite(K).all():
        raise BackendContractError("intrinsics must be finite [3,3]")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    if fx == 0.0 or fy == 0.0:
        return pose, K
    sign_x = 1.0 if fx > 0 else -1.0
    sign_y = 1.0 if fy > 0 else -1.0
    if sign_x > 0 and sign_y > 0:
        return pose, K
    scale = sign_x * sign_y
    axis = np.diag((sign_y, sign_x, scale, 1.0))
    K_axis = np.diag((sign_y, sign_x, scale))
    return pose @ axis, scale * K @ K_axis


def validate_latent_camera(relative: object, intrinsics: object) -> tuple[np.ndarray, np.ndarray]:
    """Validate canonical 20-row relative-W2C plus normalized K."""

    poses = _se3(relative, rows=STABLE_GEOMETRY.latent_frames, name="relative_w2c")
    if not np.allclose(poses[0], np.eye(4), atol=1e-5, rtol=0.0):
        raise BackendContractError("relative_w2c row zero must be identity")
    K = _intrinsics(intrinsics, rows=STABLE_GEOMETRY.latent_frames)
    return poses.astype(np.float32), K


def prepare_latent_camera(
    poses: object,
    intrinsics: object,
    *,
    convention: str,
    resize: CoverResize | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert absolute C2W once or pass already-relative W2C through once."""

    pose_rows = np.asarray(poses, dtype=np.float64)
    if pose_rows.shape[0] == STABLE_GEOMETRY.pixel_frames:
        pose_rows = pose_rows[np.asarray(STABLE_GEOMETRY.camera_pixel_indices)]
    pose_rows = _se3(
        pose_rows,
        rows=STABLE_GEOMETRY.latent_frames,
        name=convention,
    )
    K_rows = np.asarray(intrinsics, dtype=np.float64)
    if K_rows.shape == (3, 3):
        K_rows = np.broadcast_to(K_rows, (STABLE_GEOMETRY.latent_frames, 3, 3)).copy()
    elif K_rows.shape[0] == STABLE_GEOMETRY.pixel_frames:
        K_rows = K_rows[np.asarray(STABLE_GEOMETRY.camera_pixel_indices)]
    if K_rows.shape != (STABLE_GEOMETRY.latent_frames, 3, 3):
        raise BackendContractError("intrinsics must be [3,3], [20,3,3], or [153,3,3]")

    if convention == "absolute_c2w":
        canonical = [
            canonicalize_signed_focal_gauge(pose, K)
            for pose, K in zip(pose_rows, K_rows, strict=True)
        ]
        pose_rows = np.stack([item[0] for item in canonical])
        K_rows = np.stack([item[1] for item in canonical])
        try:
            relative = relative_w2c(pose_rows, "absolute_c2w")
        except DataContractError as exc:
            raise BackendContractError(str(exc)) from exc
    elif convention == "relative_w2c":
        relative = pose_rows
    else:
        raise BackendContractError("camera convention must be absolute_c2w or relative_w2c")

    if resize is not None:
        K_rows = np.stack([transform_intrinsics(K, resize) for K in K_rows])
    return validate_latent_camera(relative, K_rows)


@dataclass(frozen=True)
class TokenCamera:
    """Token-aligned camera inputs for fused PRoPE in video self-attention."""

    viewmats: np.ndarray
    intrinsics: np.ndarray

    def __post_init__(self) -> None:
        batch = int(self.viewmats.shape[0]) if self.viewmats.ndim == 4 else -1
        expected_view = (batch, STABLE_GEOMETRY.video_tokens, 4, 4)
        expected_k = (batch, STABLE_GEOMETRY.video_tokens, 3, 3)
        if self.viewmats.shape != expected_view or self.intrinsics.shape != expected_k:
            raise BackendContractError(f"token camera must be {expected_view} and {expected_k}")
        if not np.isfinite(self.viewmats).all() or not np.isfinite(self.intrinsics).all():
            raise BackendContractError("token camera contains NaN or Inf")


def expand_token_camera(relative: object, intrinsics: object) -> TokenCamera:
    """Repeat each latent camera row over its 384 F-H-W tokens."""

    poses = np.asarray(relative)
    K = np.asarray(intrinsics)
    if poses.ndim == 3:
        poses = poses[None]
    if K.ndim == 3:
        K = K[None]
    if poses.shape[0] != K.shape[0]:
        raise BackendContractError("camera pose/intrinsics batch sizes differ")
    validated = [
        validate_latent_camera(pose, intrinsic) for pose, intrinsic in zip(poses, K, strict=True)
    ]
    poses = np.stack([item[0] for item in validated])
    K = np.stack([item[1] for item in validated])
    return TokenCamera(
        viewmats=np.repeat(poses, STABLE_GEOMETRY.tokens_per_latent, axis=1),
        intrinsics=np.repeat(K, STABLE_GEOMETRY.tokens_per_latent, axis=1),
    )


def condition_camera(
    relative: object,
    intrinsics: object,
    *,
    translation_transform: str = "linear",
    max_rel_translation: float = 20.0,
    max_camera_abs: float = 20.0,
) -> TokenCamera:
    """Apply runtime guards, then model-only translation conditioning."""

    poses, K = validate_latent_camera(relative, intrinsics)
    magnitudes = np.linalg.norm(poses[:, :3, 3], axis=-1)
    if float(magnitudes.max(initial=0.0)) > max_rel_translation:
        raise BackendContractError("relative translation exceeds configured guard")
    if float(np.abs(poses).max(initial=0.0)) > max_camera_abs:
        raise BackendContractError("relative camera matrix exceeds configured guard")
    try:
        conditioned = transform_translation(poses, translation_transform)
    except DataContractError as exc:
        raise BackendContractError(str(exc)) from exc
    return expand_token_camera(conditioned, K)


__all__ = [
    "TokenCamera",
    "canonicalize_signed_focal_gauge",
    "condition_camera",
    "expand_token_camera",
    "prepare_latent_camera",
    "validate_latent_camera",
]
