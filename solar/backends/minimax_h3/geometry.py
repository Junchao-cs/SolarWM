"""MiniMax-H3 video geometry and native three-axis position grids.

This module is intentionally NumPy-only. Importing the H3 plugin must not import
PyTorch, Diffusers, or materialize the 33B transformer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class H3Geometry:
    """MiniMax-H3 VisualVAE and transformer geometry."""

    pixel_frames_per_chunk: int = 17
    latent_frames_per_chunk: int = 5
    latent_prefix_frames: int = 2
    vae_spatial_compression: int = 16
    transformer_patch: tuple[int, int, int] = (1, 2, 2)
    canvas_multiple: int = 32
    semantic_fps: int = 24
    audio_latents_per_second: int = 40
    audio_channels: int = 2
    rope_frame_rescale: float = 5.0 / 3.0
    rope_frames_per_latent: tuple[int, ...] = (1, 4, 4, 4, 4)


DEFAULT_GEOMETRY = H3Geometry()


@dataclass(frozen=True)
class H3Stage0p5Geometry:
    """The supported Stage0.5 SP2 geometry."""

    pixel_frames: int = 158
    encoded_latents: int = 47
    height: int = 768
    width: int = 1344
    latent_channels: int = 24
    latent_height: int = 48
    latent_width: int = 84
    patch_size: tuple[int, int, int] = (1, 2, 2)

    @property
    def rows_per_latent(self) -> int:
        patch_t, patch_h, patch_w = self.patch_size
        if patch_t != 1:
            raise ValueError("MiniMax-H3 requires temporal patch size 1")
        return (self.latent_height // patch_h) * (self.latent_width // patch_w)


STABLE_STAGE0P5_GEOMETRY = H3Stage0p5Geometry()


def align_pixel_frames(
    num_frames: int,
    geometry: H3Geometry = DEFAULT_GEOMETRY,
) -> int:
    """Round up to the next valid ``17*n + 5`` VisualVAE length."""

    if num_frames < 1:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    frames = int(num_frames)
    while frames % geometry.pixel_frames_per_chunk != geometry.latent_frames_per_chunk:
        frames += 1
    return frames


def pixel_frames_to_latent_frames(
    num_frames: int,
    geometry: H3Geometry = DEFAULT_GEOMETRY,
) -> int:
    """Map an exact ``17*n + 5`` pixel length to ``5*n + 2`` latents."""

    fpc = geometry.pixel_frames_per_chunk
    lpc = geometry.latent_frames_per_chunk
    if num_frames < lpc or num_frames % fpc != lpc:
        raise ValueError(f"num_frames must have form {fpc}*n+{lpc}, got {num_frames}")
    return (num_frames - lpc) // fpc * lpc + geometry.latent_prefix_frames


def latent_frames_to_pixel_frames(
    num_latent_frames: int,
    geometry: H3Geometry = DEFAULT_GEOMETRY,
) -> int:
    """Invert :func:`pixel_frames_to_latent_frames` on exact lengths."""

    prefix = geometry.latent_prefix_frames
    lpc = geometry.latent_frames_per_chunk
    if num_latent_frames < prefix or (num_latent_frames - prefix) % lpc:
        raise ValueError(
            f"num_latent_frames must have form {lpc}*n+{prefix}, got {num_latent_frames}"
        )
    chunks = (num_latent_frames - prefix) // lpc
    return chunks * geometry.pixel_frames_per_chunk + lpc


def latent_aligned_pixel_indices(
    num_frames: int,
    geometry: H3Geometry = DEFAULT_GEOMETRY,
) -> np.ndarray:
    """Map encoded latent positions to exact source-pixel frame indices.

    H3 repeats offsets ``[0,1,5,9,13]`` every 17 input frames and retains
    in-range offsets from the final partial period.  For 158 frames this
    returns 47 entries and never retimes the source clip.
    """

    expected_latents = pixel_frames_to_latent_frames(num_frames, geometry)
    offsets = np.asarray((0, 1, 5, 9, 13), dtype=np.int64)
    indices: list[np.ndarray] = []
    for base in range(0, num_frames, geometry.pixel_frames_per_chunk):
        period = base + offsets
        indices.append(period[period < num_frames])
    result = np.concatenate(indices)
    if result.shape != (expected_latents,):
        raise RuntimeError("H3 pixel/latent alignment is internally inconsistent")
    return result


def audio_latents_for_video(
    num_frames: int,
    *,
    fps: float = DEFAULT_GEOMETRY.semantic_fps,
    latents_per_second: int = DEFAULT_GEOMETRY.audio_latents_per_second,
) -> int:
    """Return per-channel audio latents on H3's rounded semantic timeline."""

    if num_frames < 1 or fps <= 0 or latents_per_second <= 0:
        raise ValueError("num_frames, fps, and latents_per_second must be positive")
    return round(num_frames / fps * latents_per_second)


def validate_canvas(height: int, width: int, *, multiple: int = 32) -> tuple[int, int]:
    """Reject rather than round a canvas outside the checkpoint contract."""

    if height <= 0 or width <= 0:
        raise ValueError(f"height and width must be positive, got {(height, width)}")
    if multiple <= 0 or height % multiple or width % multiple:
        raise ValueError(
            f"height and width must both be multiples of {multiple}, got {(height, width)}"
        )
    return int(height), int(width)


def validate_stage0p5_geometry(
    *,
    pixel_frames: int,
    encoded_latents: int,
    height: int,
    width: int,
    latent_channels: int,
    latent_height: int,
    latent_width: int,
) -> H3Stage0p5Geometry:
    """Validate the 158f/47-latent 768x1344 profile."""

    expected = STABLE_STAGE0P5_GEOMETRY
    observed = H3Stage0p5Geometry(
        pixel_frames=int(pixel_frames),
        encoded_latents=int(encoded_latents),
        height=int(height),
        width=int(width),
        latent_channels=int(latent_channels),
        latent_height=int(latent_height),
        latent_width=int(latent_width),
    )
    if observed != expected:
        raise ValueError(
            "MiniMax-H3 Stage0.5 SP2 supports only "
            "158f -> 47 latents at 768x1344 with [24,47,48,84] latents; "
            f"got {observed}"
        )
    if pixel_frames_to_latent_frames(observed.pixel_frames) != observed.encoded_latents:
        raise ValueError("pixel/latent temporal geometry is internally inconsistent")
    validate_canvas(observed.height, observed.width)
    if observed.height // DEFAULT_GEOMETRY.vae_spatial_compression != observed.latent_height:
        raise ValueError("height does not match VisualVAE spatial compression")
    if observed.width // DEFAULT_GEOMETRY.vae_spatial_compression != observed.latent_width:
        raise ValueError("width does not match VisualVAE spatial compression")
    if observed.rows_per_latent != 1008:
        raise ValueError("H3 transformer geometry must have 1008 rows per latent")
    return observed


def spatial_position_axis(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    """Build the provider-compatible float64 endpoint-excluded spatial axis."""

    if dim <= 0 or patch <= 0 or dim % patch:
        raise ValueError(f"dim={dim} must be positive and divisible by patch={patch}")
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * 32


def frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int = 2,
    patch_w: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return flattened ``(h,w)`` coordinates and the width axis for a frame."""

    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = spatial_position_axis(latent_height, patch_h, sqrt_area)
    width_grid = spatial_position_axis(latent_width, patch_w, sqrt_area)
    grid_h, grid_w = np.meshgrid(height_grid, width_grid, indexing="ij")
    return np.stack((grid_h.reshape(-1), grid_w.reshape(-1)), axis=-1), width_grid


def temporal_position_grid(num_latent_frames: int, origin: float = 0.0) -> np.ndarray:
    """Return H3's non-uniform ``5/3*(1,4,4,4,4)`` rotary-time grid."""

    if num_latent_frames < 0:
        raise ValueError(f"num_latent_frames must be non-negative, got {num_latent_frames}")
    spans = np.asarray(
        [
            DEFAULT_GEOMETRY.rope_frame_rescale
            * DEFAULT_GEOMETRY.rope_frames_per_latent[
                index % len(DEFAULT_GEOMETRY.rope_frames_per_latent)
            ]
            for index in range(num_latent_frames)
        ],
        dtype=np.float64,
    )
    if num_latent_frames == 0:
        return spans
    result = np.empty(num_latent_frames, dtype=np.float64)
    result[0] = float(origin)
    if num_latent_frames > 1:
        result[1:] = float(origin) + np.cumsum(spans[:-1])
    return result


def native_video_position_grid(
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    origin: float = 0.0,
) -> np.ndarray:
    """Return flattened native H3 ``(t,h,w)`` IDs in frame-major order."""

    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise ValueError("H3 MM-RoPE layout requires temporal patch_size=1")
    frame_grid, _ = frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    times = temporal_position_grid(num_latent_frames, origin)
    result = np.empty((num_latent_frames, frame_grid.shape[0], 3), dtype=np.float64)
    result[:, :, 0] = times[:, None]
    result[:, :, 1:] = frame_grid[None]
    return result.reshape(-1, 3)


__all__ = [
    "DEFAULT_GEOMETRY",
    "STABLE_STAGE0P5_GEOMETRY",
    "H3Geometry",
    "H3Stage0p5Geometry",
    "align_pixel_frames",
    "audio_latents_for_video",
    "frame_position_grid",
    "latent_aligned_pixel_indices",
    "latent_frames_to_pixel_frames",
    "native_video_position_grid",
    "pixel_frames_to_latent_frames",
    "spatial_position_axis",
    "temporal_position_grid",
    "validate_canvas",
    "validate_stage0p5_geometry",
]
