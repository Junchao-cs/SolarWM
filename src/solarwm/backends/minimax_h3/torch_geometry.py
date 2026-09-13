"""Native torch position grids used by causal H3 execution."""

from __future__ import annotations

import math
from typing import Any

from .geometry import DEFAULT_GEOMETRY


def spatial_position_axis(dim: int, patch: int, sqrt_area: float) -> Any:
    """Build one upstream-compatible float64, endpoint-excluded spatial axis."""

    if dim <= 0 or patch <= 0 or dim % patch:
        raise ValueError(f"dim={dim} must be positive and divisible by patch={patch}")
    import numpy as np
    import torch

    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    values = np.linspace(left, left + ratio, dim // patch, endpoint=False) * 32
    return torch.from_numpy(values).to(torch.float64)


def frame_position_grid(
    latent_height: int,
    latent_width: int,
    patch_h: int = 2,
    patch_w: int = 2,
) -> tuple[Any, Any]:
    """Return flattened ``(h,w)`` coordinates and the width axis for one frame."""

    import torch

    sqrt_area = math.sqrt(latent_height * latent_width)
    height_grid = spatial_position_axis(latent_height, patch_h, sqrt_area)
    width_grid = spatial_position_axis(latent_width, patch_w, sqrt_area)
    grids = torch.meshgrid(height_grid, width_grid, indexing="ij")
    return torch.stack([grid.reshape(-1) for grid in grids], dim=-1), width_grid


def temporal_position_grid(num_latent_frames: int, origin: float = 0.0) -> Any:
    """Return H3's non-uniform ``5/3*(1,4,4,4,4)`` rotary-time grid."""

    if num_latent_frames < 0:
        raise ValueError(f"num_latent_frames must be non-negative, got {num_latent_frames}")
    import torch

    spans = torch.tensor(
        [
            DEFAULT_GEOMETRY.rope_frame_rescale
            * DEFAULT_GEOMETRY.rope_frames_per_latent[
                i % len(DEFAULT_GEOMETRY.rope_frames_per_latent)
            ]
            for i in range(num_latent_frames)
        ],
        dtype=torch.float64,
    )
    if num_latent_frames == 0:
        return spans
    return float(origin) + torch.cat((torch.zeros(1, dtype=torch.float64), spans[:-1].cumsum(0)))


def native_video_position_grid(
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    origin: float = 0.0,
) -> Any:
    """Return flattened native H3 ``(t,h,w)`` IDs in frame-major row order."""

    import torch

    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1:
        raise ValueError("released H3 MM-RoPE layout requires temporal patch_size=1")
    frame_grid, _ = frame_position_grid(latent_height, latent_width, patch_h, patch_w)
    times = temporal_position_grid(num_latent_frames, origin)
    result = torch.empty(num_latent_frames, frame_grid.shape[0], 3, dtype=torch.float64)
    result[:, :, 0] = times[:, None]
    result[:, :, 1:] = frame_grid[None]
    return result.reshape(-1, 3)
