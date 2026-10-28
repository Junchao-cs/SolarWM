"""Fixed geometry for the LTX-2.5 153-frame profile."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solar.errors import BackendContractError


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BackendContractError(f"{name} must be a positive integer, got {value!r}")
    return value


def pixel_frames_to_latent_frames(pixel_frames: int, temporal_factor: int = 8) -> int:
    """Map an exact causal-VAE pixel grid to temporal latent slices."""

    frames = _positive_int("pixel_frames", pixel_frames)
    factor = _positive_int("temporal_factor", temporal_factor)
    if (frames - 1) % factor:
        raise BackendContractError(
            "pixel_frames must have exact causal-VAE form 1 + factor*n; "
            f"got frames={frames}, factor={factor}"
        )
    return 1 + (frames - 1) // factor


def causal_camera_pixel_indices(
    pixel_frames: int = 153,
    temporal_factor: int = 8,
) -> tuple[int, ...]:
    """Return causal interval starts ``0,1,9,...,145``."""

    latent_frames = pixel_frames_to_latent_frames(pixel_frames, temporal_factor)
    result = (0, *(1 + index * temporal_factor for index in range(latent_frames - 1)))
    if len(result) != latent_frames or result[-1] >= pixel_frames:
        raise BackendContractError("internal LTX causal camera alignment is inconsistent")
    return result


@dataclass(frozen=True)
class LTX25Geometry:
    """The supported LTX-2.5 geometry."""

    pixel_frames: int = 153
    height: int = 512
    width: int = 768
    latent_channels: int = 128
    latent_frames: int = 20
    latent_height: int = 16
    latent_width: int = 24
    temporal_factor: int = 8
    spatial_factor: int = 32
    model_fps: float = 24.0

    def __post_init__(self) -> None:
        observed = (
            self.pixel_frames,
            self.height,
            self.width,
            self.latent_channels,
            self.latent_frames,
            self.latent_height,
            self.latent_width,
            self.temporal_factor,
            self.spatial_factor,
            float(self.model_fps),
        )
        expected = (153, 512, 768, 128, 20, 16, 24, 8, 32, 24.0)
        if observed != expected:
            raise BackendContractError(
                "LTX-2.5 geometry is fixed to 768x512x153f -> "
                f"[128,20,16,24] at model_fps=24; got {observed!r}"
            )
        if (
            pixel_frames_to_latent_frames(self.pixel_frames, self.temporal_factor)
            != self.latent_frames
        ):
            raise BackendContractError("LTX temporal VAE geometry is inconsistent")
        if (self.height // self.spatial_factor, self.width // self.spatial_factor) != (
            self.latent_height,
            self.latent_width,
        ):
            raise BackendContractError("LTX spatial VAE geometry is inconsistent")

    @property
    def latent_shape(self) -> tuple[int, int, int, int]:
        return self.latent_channels, self.latent_frames, self.latent_height, self.latent_width

    @property
    def first_frame_latent_shape(self) -> tuple[int, int, int, int]:
        return self.latent_channels, 1, self.latent_height, self.latent_width

    @property
    def tokens_per_latent(self) -> int:
        return self.latent_height * self.latent_width

    @property
    def video_tokens(self) -> int:
        return self.latent_frames * self.tokens_per_latent

    @property
    def camera_pixel_indices(self) -> tuple[int, ...]:
        return causal_camera_pixel_indices(self.pixel_frames, self.temporal_factor)


STABLE_GEOMETRY = LTX25Geometry()


@dataclass(frozen=True)
class CoverResize:
    """Realized integer cover-resize and center-crop geometry."""

    source_height: int
    source_width: int
    resized_height: int
    resized_width: int
    crop_top: int
    crop_left: int
    target_height: int = STABLE_GEOMETRY.height
    target_width: int = STABLE_GEOMETRY.width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.source_height

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.source_width


def cover_resize(source_height: int, source_width: int) -> CoverResize:
    """Apply the required bicubic cover-resize integer arithmetic."""

    source_height = _positive_int("source_height", source_height)
    source_width = _positive_int("source_width", source_width)
    target_height, target_width = STABLE_GEOMETRY.height, STABLE_GEOMETRY.width
    aspect = source_width / source_height
    target_aspect = target_width / target_height
    if aspect > target_aspect:
        resized_height = target_height
        resized_width = int(target_height * aspect)
    else:
        resized_height = int(target_width / aspect)
        resized_width = target_width
    return CoverResize(
        source_height=source_height,
        source_width=source_width,
        resized_height=resized_height,
        resized_width=resized_width,
        crop_top=(resized_height - target_height) // 2,
        crop_left=(resized_width - target_width) // 2,
    )


def transform_intrinsics(
    intrinsics: object,
    transform: CoverResize,
    *,
    input_normalized: bool | None = None,
) -> np.ndarray:
    """Resize, crop, and normalize one no-skew pinhole matrix.

    Normalized focal length is intentionally not upper-bounded.
    """

    matrix = np.asarray(intrinsics, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise BackendContractError("intrinsics must be one finite [3,3] matrix")
    if not np.allclose(matrix[2], (0.0, 0.0, 1.0), atol=1e-7, rtol=0.0):
        raise BackendContractError("intrinsics bottom row must be [0,0,1]")
    if abs(matrix[0, 1]) > 1e-7 or abs(matrix[1, 0]) > 1e-7:
        raise BackendContractError("intrinsics must use zero skew")
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    if input_normalized is None:
        input_normalized = bool(max(abs(fx), abs(fy), abs(cx), abs(cy)) <= 4.0)
    if input_normalized:
        fx, cx = fx * transform.source_width, cx * transform.source_width
        fy, cy = fy * transform.source_height, cy * transform.source_height
    result = np.asarray(
        [
            [
                fx * transform.scale_x / transform.target_width,
                0.0,
                (cx * transform.scale_x - transform.crop_left) / transform.target_width,
            ],
            [
                0.0,
                fy * transform.scale_y / transform.target_height,
                (cy * transform.scale_y - transform.crop_top) / transform.target_height,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(result).all() or result[0, 0] <= 0 or result[1, 1] <= 0:
        raise BackendContractError("transformed focal lengths must be finite and positive")
    return result


def validate_contiguous_source_indices(value: object, *, start: int | None = None) -> np.ndarray:
    """Require exactly 153 consecutive source ordinals; FPS is not consulted."""

    indices = np.asarray(value)
    if indices.dtype != np.int64 or indices.shape != (STABLE_GEOMETRY.pixel_frames,):
        raise BackendContractError("source_indices must be int64 [153]")
    if indices[0] < 0 or not np.all(np.diff(indices) == 1):
        raise BackendContractError("source_indices must be nonnegative exact consecutive ordinals")
    if start is not None and int(indices[0]) != int(start):
        raise BackendContractError("source_indices do not begin at the declared start frame")
    return indices


__all__ = [
    "STABLE_GEOMETRY",
    "CoverResize",
    "LTX25Geometry",
    "causal_camera_pixel_indices",
    "cover_resize",
    "pixel_frames_to_latent_frames",
    "transform_intrinsics",
    "validate_contiguous_source_indices",
]
