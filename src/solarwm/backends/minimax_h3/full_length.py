"""Source-length geometry for H3 SGF evaluation, independent of GPU libraries."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FullLengthGeometry:
    source_frames: int
    decoded_frames: int
    decode_latents: int
    rollout_latents: int
    camera_indices: tuple[int, ...]
    tail_camera_rows: int


def full_length_geometry(source_frames: int) -> FullLengthGeometry:
    if isinstance(source_frames, bool) or not isinstance(source_frames, int) or source_frames < 22:
        raise ValueError("H3 evaluation requires at least 22 consecutive source frames")
    n = (source_frames - 5 + 16) // 17
    decode_latents = 5 * n + 2
    rollout_latents = (decode_latents + 4) // 5 * 5
    cadence = (1, 4, 4, 4, 4)
    indices = [0]
    tail = 0
    source_index = 0
    for index in range(rollout_latents - 1):
        source_index += cadence[index % 5]
        if source_index < source_frames:
            indices.append(source_index)
        else:
            indices.append(indices[-1])
            tail += 1
    return FullLengthGeometry(
        source_frames, 17 * n + 5, decode_latents, rollout_latents, tuple(indices), tail
    )


def exported_fps(source_fps: float, policy: str) -> float:
    import math

    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Source FPS must be finite and positive")
    if policy not in {"source", "h3"}:
        raise ValueError("FPS policy must be source or h3")
    return float(source_fps) if policy == "source" else 24.0
