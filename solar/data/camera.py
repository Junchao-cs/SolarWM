"""Explicit camera conventions shared by raw, latent, train, and inference paths."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal

import numpy as np

from solar.errors import DataContractError

CameraStorage = Literal["absolute_c2w", "absolute_w2c", "relative_w2c"]
TranslationTransform = Literal["linear", "logd4"]


class CameraGuardError(DataContractError):
    """A valid camera trajectory rejected by configured magnitude guards."""


def camera_audit_prefix_frames(num_frames: int, fps: float, max_seconds: float = 10.0) -> int:
    """Apply the first-ten-second magnitude-guard window."""

    count = max(0, int(num_frames))
    if count == 0:
        return 0
    rate = float(fps)
    seconds = float(max_seconds)
    if not np.isfinite(rate) or rate <= 0 or not np.isfinite(seconds) or seconds <= 0:
        return count
    return min(count, max(1, int(np.ceil(rate * seconds))))


def load_camera_npz(value: bytes, array_key: str) -> tuple[np.ndarray, CameraStorage]:
    """Load one explicitly declared camera array without changing its dtype."""

    key_contract: dict[str, CameraStorage] = {
        "c2w": "absolute_c2w",
        "w2c": "absolute_w2c",
        "vipe_c2w": "absolute_c2w",
        "vipe_w2c": "absolute_w2c",
        "relative_w2c": "relative_w2c",
    }
    if array_key not in key_contract:
        raise DataContractError(f"unsupported camera array key {array_key!r}")
    try:
        with np.load(io.BytesIO(value), allow_pickle=False) as archive:
            if array_key.startswith("vipe_") and "vipe_status" in archive.files:
                status = str(np.asarray(archive["vipe_status"]).reshape(-1)[0]).lower()
                if status != "ok":
                    raise DataContractError(f"VIPE camera status is {status!r}, expected 'ok'")
            if array_key not in archive.files:
                raise DataContractError(f"camera archive lacks declared array {array_key!r}")
            matrices = np.asarray(archive[array_key])
    except (OSError, ValueError) as exc:
        raise DataContractError("camera member is not a valid non-pickled NPZ") from exc
    return _validate_matrices(matrices, name=array_key), key_contract[array_key]


def _validate_matrices(value: np.ndarray, *, name: str) -> np.ndarray:
    matrices = np.asarray(value)
    if matrices.ndim != 3 or matrices.shape[1:] != (4, 4):
        raise DataContractError(f"{name} must have shape [T,4,4], got {matrices.shape}")
    if not np.isfinite(matrices).all():
        raise DataContractError(f"{name} contains non-finite values")
    return matrices


def invert_se3(matrices: np.ndarray) -> np.ndarray:
    matrices = _validate_matrices(matrices, name="SE3")
    result = np.zeros_like(matrices)
    rotation = matrices[:, :3, :3]
    translation = matrices[:, :3, 3]
    rotation_t = np.swapaxes(rotation, -1, -2)
    result[:, :3, :3] = rotation_t
    result[:, :3, 3] = -np.einsum("tij,tj->ti", rotation_t, translation)
    result[:, 3, 3] = 1
    return result


def absolute_c2w(value: np.ndarray, storage: CameraStorage) -> np.ndarray:
    matrices = _validate_matrices(value, name="camera")
    if storage == "absolute_c2w":
        return matrices
    if storage == "absolute_w2c":
        return invert_se3(matrices)
    raise DataContractError("relative_w2c cannot be converted back to authoritative absolute C2W")


def relative_w2c(value: np.ndarray, storage: CameraStorage) -> np.ndarray:
    matrices = _validate_matrices(value, name="camera")
    if storage == "relative_w2c":
        relative = matrices.copy()
    else:
        c2w = absolute_c2w(matrices, storage)
        relative = np.matmul(invert_se3(c2w), c2w[0])
    identity = np.eye(4, dtype=relative.dtype)
    if not np.allclose(relative[0], identity, rtol=1e-5, atol=1e-5):
        raise DataContractError("first relative W2C row is not identity")
    return relative


def relative_c2w(value: np.ndarray, storage: CameraStorage) -> np.ndarray:
    return invert_se3(relative_w2c(value, storage))


def transform_translation(
    relative: np.ndarray,
    transform: TranslationTransform,
) -> np.ndarray:
    """Apply model-conditioning compression without mutating source camera data."""

    matrices = _validate_matrices(relative, name="relative camera")
    if transform == "linear":
        return matrices
    if transform != "logd4":
        raise DataContractError(f"unknown camera translation transform {transform!r}")
    result = matrices.copy()
    working_dtype = np.float64 if matrices.dtype == np.float64 else np.float32
    translation = matrices[:, :3, 3].astype(working_dtype, copy=False)
    norms = np.linalg.norm(translation, axis=-1)
    scale = np.zeros_like(norms)
    nonzero = norms != 0
    scale[nonzero] = np.log1p(norms[nonzero]) / (4.0 * norms[nonzero])
    result[:, :3, 3] = (translation * scale[:, None]).astype(result.dtype, copy=False)
    return result


@dataclass(frozen=True)
class CameraGuards:
    max_rel_translation: float | None = 20.0
    max_camera_abs: float | None = 20.0

    def apply(self, relative: np.ndarray, *, audit_rows: int | None = None) -> np.ndarray:
        matrices = _validate_matrices(relative, name="relative camera")
        count = matrices.shape[0] if audit_rows is None else int(audit_rows)
        if not 1 <= count <= matrices.shape[0]:
            raise DataContractError("audit_rows is outside the camera trajectory")
        prefix = matrices[:count]
        if self.max_rel_translation is not None:
            magnitude = np.linalg.norm(prefix[:, :3, 3], axis=-1)
            if float(magnitude.max(initial=0.0)) > self.max_rel_translation:
                raise CameraGuardError("relative translation exceeds configured guard")
        result = matrices.copy()
        if self.max_camera_abs is not None and (
            float(np.abs(result[:count]).max(initial=0.0)) > self.max_camera_abs
        ):
            raise CameraGuardError("camera matrix exceeds configured absolute guard")
        return result


def causal_pixel_indices(pixel_frames: int, temporal_stride: int) -> np.ndarray:
    if pixel_frames < 1 or temporal_stride < 1:
        raise DataContractError("pixel_frames and temporal_stride must be positive")
    if pixel_frames == 1:
        return np.asarray([0], dtype=np.int64)
    return np.asarray([0, *range(1, pixel_frames, temporal_stride)], dtype=np.int64)
