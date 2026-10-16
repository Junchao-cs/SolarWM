"""Dependency-light MiniMax-H3 packed-row layouts.

The Stage0.5 document is ordered as::

    [Qwen joint image/text | VisualVAE anchor | visual observations | audio | target video]

A visual observation is any image or video latent, of any size and length, and
carries no type. Only its placement says how it relates to the clip:

* *aligned* with a list of target latent frames: its frames share those frames'
  rotary times and camera poses, and its spatial grid is laid over the target's
  canvas whatever its resolution or aspect ratio. A depth, pose or edge video, the
  source of an edit, a keyframe, a low-resolution preview are all this case. An
  optional box per frame lays it over a part of the canvas instead: a face crop
  that follows a face, a detail patch, an object pasted at a location.
* *free* (no frames): it is not a view of the clip. It sits before the clip on the
  rotary time axis, in order, on its own spatial grid, and never takes part in
  camera PRoPE. A subject, an object, a scene, a style, a motion clip. The clock
  is the one of the released ``ref2va`` layout: it starts where the text ends, an
  image advances it by one slot, a video by its own rotary span, and the clip
  (audio, anchors and target alike) starts where the free observations left it.

Qwen vision rows may carry ``VIDEO_TAG`` while still belonging to the text
segment.  Camera PRoPE must therefore use ``camera_video_indices`` and never
infer camera rows from token tags.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np

from .geometry import frame_position_grid, temporal_position_grid

VIDEO_TAG = 0
TEXT_TAG = 1
AUDIO_TAG = 2

PADDING_ROLE = -1
CONDITION_ROLE = 0
AUDIO_ROLE = 1
CLEAN_VIDEO_ROLE = 2
NOISY_VIDEO_ROLE = 3
TARGET_AUDIO_ROLE = AUDIO_ROLE
TARGET_VIDEO_ROLE = NOISY_VIDEO_ROLE


@dataclass(frozen=True)
class H3VisualObservationSpec:
    """Shape and placement of one visual observation; ``frames=None`` is a free one."""

    num_frames: int
    height: int
    width: int
    frames: tuple[int, ...] | None = None
    # ``(x0, y0, x1, y1)`` in the target's normalized canvas, one per frame or one for all.
    boxes: tuple[tuple[float, float, float, float], ...] | None = None


def _canvas_grid(
    item: H3VisualObservationSpec,
    box: tuple[float, float, float, float],
    target_height: int,
    target_width: int,
    patch_h: int,
    patch_w: int,
) -> np.ndarray:
    """Spatial positions of one observation frame laid over ``box`` of the target canvas."""

    x0, y0, x1, y1 = (float(value) for value in box)
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(f"visual observation box {box} must lie inside the unit canvas")
    sqrt_area = np.sqrt(target_height * target_width)
    axes = []
    for size, low, high, count in (
        (target_height, y0, y1, item.height // patch_h),
        (target_width, x0, x1, item.width // patch_w),
    ):
        ratio = size / sqrt_area
        left = (1.0 - ratio) / 2.0
        steps = low + (high - low) * np.arange(count, dtype=np.float64) / count
        axes.append((left + ratio * steps) * 32)
    grid_h, grid_w = np.meshgrid(axes[0], axes[1], indexing="ij")
    return np.stack((grid_h.reshape(-1), grid_w.reshape(-1)), axis=-1)


def _indices(start: int, stop: int) -> np.ndarray:
    return np.arange(start, stop, dtype=np.int64)


def _empty_indices() -> np.ndarray:
    return np.empty((0,), dtype=np.int64)


@dataclass(frozen=True)
class H3PackedLayout:
    """All structural arrays for one packed H3 attention document."""

    position_ids: np.ndarray
    token_tags: np.ndarray
    video_indices: np.ndarray
    audio_indices: np.ndarray
    text_indices: np.ndarray
    condition_indices: np.ndarray
    target_indices: np.ndarray
    condition_video_indices: np.ndarray
    audio_condition_indices: np.ndarray
    clean_video_indices: np.ndarray
    noisy_video_indices: np.ndarray
    target_video_indices: np.ndarray
    target_audio_indices: np.ndarray
    camera_video_indices: np.ndarray
    camera_frame_ids: np.ndarray
    row_roles: np.ndarray
    target_video_chunk_ids: np.ndarray
    num_condition_video_rows: int
    num_condition_audio_rows: int
    num_clean_video_rows: int
    num_noisy_video_rows: int
    rows_per_video_frame: int
    latent_height: int
    latent_width: int
    patch_size: tuple[int, int, int]
    stage: str
    window_chunks: int | None = None
    chunk_latent_frames: int | None = None
    # Visual observations are the tail of ``condition_video_indices``; row counts per item.
    observation_rows: tuple[int, ...] = ()

    @property
    def num_observation_rows(self) -> int:
        return int(sum(self.observation_rows))

    @property
    def observation_video_indices(self) -> np.ndarray:
        return self.condition_video_indices[
            self.num_condition_video_rows - self.num_observation_rows :
        ]

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])

    @property
    def target_video_output_slice(self) -> slice:
        """Select target rows from an output already filtered to video rows."""

        start = self.num_condition_video_rows + self.num_clean_video_rows
        return slice(start, start + self.num_noisy_video_rows)

    @property
    def noisy_video_output_slice(self) -> slice:
        return self.target_video_output_slice

    @property
    def clean_video_output_slice(self) -> slice:
        start = self.num_condition_video_rows
        return slice(start, start + self.num_clean_video_rows)

    @property
    def condition_video_output_slice(self) -> slice:
        return slice(0, self.num_condition_video_rows)

    @property
    def camera_indices(self) -> np.ndarray:
        return self.camera_video_indices

    def to(self, device: object) -> H3PackedLayout:
        """Transfer structural arrays to torch without changing the public layout."""
        import torch

        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, np.ndarray):
                values[item.name] = torch.from_numpy(value).to(device)
            elif isinstance(value, torch.Tensor):
                values[item.name] = value.to(device)
        return replace(self, **values)

    def transformer_kwargs(self) -> dict[str, np.ndarray]:
        """Return the five native structural arguments expected by H3."""

        return {
            "position_ids": self.position_ids,
            "token_tags": self.token_tags,
            "video_indices": self.video_indices,
            "audio_indices": self.audio_indices,
            "text_indices": self.text_indices,
        }


def _validate_layout_inputs(
    text_token_tags: object,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    patch_size: tuple[int, int, int],
    audio_channels: int,
) -> tuple[np.ndarray, int]:
    tags = np.asarray(text_token_tags, dtype=np.int64)
    if tags.ndim != 1:
        raise ValueError(f"text_token_tags must be one-dimensional, got {tags.shape}")
    patch_t, patch_h, patch_w = patch_size
    if patch_t != 1 or latent_height % patch_h or latent_width % patch_w:
        raise ValueError(
            f"latent grid {(num_latent_frames, latent_height, latent_width)} "
            f"is incompatible with patch {patch_size}"
        )
    if num_latent_frames <= 0 or num_audio_latents < 0 or audio_channels <= 0:
        raise ValueError("latent frame counts/channels are invalid")
    if audio_channels != 2 and num_audio_latents:
        raise ValueError("native H3 audio packing requires two channel-major channels")
    return tags, (latent_height // patch_h) * (latent_width // patch_w)


def _base_layout(
    text_token_tags: object,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int,
    *,
    patch_size: tuple[int, int, int],
    audio_channels: int,
    keyframe_anchors: tuple[str, ...],
    stage: str,
    chunk_latent_frames: int | None,
    window_chunks: int | None,
    observations: tuple[H3VisualObservationSpec, ...] = (),
) -> H3PackedLayout:
    tags, rows_per_frame = _validate_layout_inputs(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_size,
        audio_channels,
    )
    _, patch_h, patch_w = patch_size
    num_text = int(tags.size)
    observation_rows = []
    for item in observations:
        if item.num_frames <= 0 or item.height <= 0 or item.width <= 0:
            raise ValueError(f"visual observation {item} is empty")
        if item.height % patch_h or item.width % patch_w:
            raise ValueError(f"visual observation grid {item} does not fit {patch_size}")
        if item.frames is not None and (
            len(item.frames) != item.num_frames
            or min(item.frames) < 0
            or max(item.frames) >= num_latent_frames
        ):
            raise ValueError(f"visual observation {item} is not aligned with the target frames")
        if item.boxes is not None and (
            item.frames is None or len(item.boxes) not in (1, item.num_frames)
        ):
            raise ValueError(f"visual observation {item} needs frames and one box per frame")
        observation_rows.append(
            item.num_frames * (item.height // patch_h) * (item.width // patch_w)
        )
    num_anchor_video = len(keyframe_anchors) * rows_per_frame
    num_condition_video = num_anchor_video + sum(observation_rows)
    num_audio = num_audio_latents * audio_channels
    num_target_video = num_latent_frames * rows_per_frame
    condition_start = num_text
    audio_start = condition_start + num_condition_video
    target_start = audio_start + num_audio
    sequence_length = target_start + num_target_video

    position_ids = np.zeros((sequence_length, 3), dtype=np.float64)
    position_ids[:num_text, 0] = np.arange(num_text, dtype=np.float64)
    frame_grid, width_grid = frame_position_grid(latent_height, latent_width, patch_h, patch_w)

    # Free observations advance the rotary clock ahead of the clip, like the released ref2va layout.
    origin = float(num_text)
    free_times: list[np.ndarray | None] = []
    for item in observations:
        if item.frames is not None:
            free_times.append(None)
        elif item.num_frames == 1:
            free_times.append(np.asarray([origin]))
            origin += 1.0
        else:
            free_times.append(temporal_position_grid(item.num_frames, origin))
            origin += float(np.diff(temporal_position_grid(item.num_frames + 1, 0.0)).sum())

    anchor_frame_ids: list[int] = []
    temporal = temporal_position_grid(num_latent_frames, origin)
    for index, anchor in enumerate(keyframe_anchors):
        if anchor == "first":
            anchor_time = origin
            anchor_frame_ids.append(0)
        elif anchor == "last":
            anchor_time = float(
                temporal_position_grid(num_latent_frames + 1, origin)[-1] - 5.0 / 3.0
            )
            anchor_frame_ids.append(num_latent_frames - 1)
        else:
            raise ValueError(f"keyframe anchor must be 'first' or 'last', got {anchor!r}")
        start = condition_start + index * rows_per_frame
        rows = slice(start, start + rows_per_frame)
        position_ids[rows, 0] = anchor_time
        position_ids[rows, 1:] = frame_grid

    # Aligned observations take their frames' rotary times and the target's canvas; free
    # ones take their slot on the clock and keep their own spatial grid.
    start = condition_start + num_anchor_video
    observation_camera_frames: list[np.ndarray] = []
    observation_camera_rows: list[np.ndarray] = []
    for item, count, free in zip(observations, observation_rows, free_times, strict=True):
        per_frame = count // item.num_frames
        if free is not None:
            frame_times = free
        else:
            frame_times = temporal[np.asarray(item.frames, dtype=np.int64)]
            observation_camera_rows.append(_indices(start, start + count))
            observation_camera_frames.append(
                np.repeat(np.asarray(item.frames, dtype=np.int64), per_frame)
            )
        rows = slice(start, start + count)
        position_ids[rows, 0] = np.repeat(frame_times, per_frame)
        if item.frames is None:
            grid = frame_position_grid(item.height, item.width, patch_h, patch_w)[0]
            position_ids[rows, 1:] = np.tile(grid, (item.num_frames, 1))
        else:
            boxes = item.boxes or ((0.0, 0.0, 1.0, 1.0),)
            grids = [
                _canvas_grid(item, box, latent_height, latent_width, patch_h, patch_w)
                for box in boxes
            ]
            position_ids[rows, 1:] = (
                np.tile(grids[0], (item.num_frames, 1))
                if len(grids) == 1
                else np.concatenate(grids, axis=0)
            )
        start += count

    if num_audio:
        audio_rows = slice(audio_start, target_start)
        audio_time = origin + np.arange(num_audio_latents, dtype=np.float64)
        position_ids[audio_rows, 0] = np.tile(audio_time, audio_channels)
        position_ids[audio_rows, 2] = np.concatenate(
            (
                np.full(num_audio_latents, width_grid[0], dtype=np.float64),
                np.full(num_audio_latents, width_grid[-1], dtype=np.float64),
            )
        )

    target_positions = np.empty((num_latent_frames, rows_per_frame, 3), dtype=np.float64)
    target_positions[:, :, 0] = temporal[:, None]
    target_positions[:, :, 1:] = frame_grid[None]
    position_ids[target_start:] = target_positions.reshape(-1, 3)

    text_indices = _indices(0, num_text)
    condition_video_indices = _indices(condition_start, audio_start)
    audio_indices = _indices(audio_start, target_start)
    target_video_indices = _indices(target_start, sequence_length)
    video_indices = np.concatenate((condition_video_indices, target_video_indices))

    token_tags = np.empty(sequence_length, dtype=np.int64)
    token_tags[text_indices] = tags
    token_tags[video_indices] = VIDEO_TAG
    token_tags[audio_indices] = AUDIO_TAG
    row_roles = np.full(sequence_length, CONDITION_ROLE, dtype=np.int64)
    row_roles[audio_indices] = AUDIO_ROLE
    row_roles[target_video_indices] = NOISY_VIDEO_ROLE

    condition_indices = np.concatenate((text_indices, condition_video_indices))
    target_indices = np.concatenate((audio_indices, target_video_indices))
    target_chunk_ids = np.full(sequence_length, -1, dtype=np.int64)
    if chunk_latent_frames is not None:
        if chunk_latent_frames <= 0 or num_latent_frames % chunk_latent_frames:
            raise ValueError(
                f"target length {num_latent_frames} must be divisible by "
                f"chunk_latent_frames={chunk_latent_frames}"
            )
        frame_chunks = np.arange(num_latent_frames, dtype=np.int64) // chunk_latent_frames
        target_chunk_ids[target_video_indices] = np.repeat(frame_chunks, rows_per_frame)

    condition_camera_frames = (
        np.repeat(np.asarray(anchor_frame_ids, dtype=np.int64), rows_per_frame)
        if anchor_frame_ids
        else _empty_indices()
    )
    target_camera_frames = np.repeat(np.arange(num_latent_frames, dtype=np.int64), rows_per_frame)
    # Aligned observations are views along the trajectory; free ones are not.
    camera_video_indices = np.concatenate(
        (
            condition_video_indices[:num_anchor_video],
            *observation_camera_rows,
            target_video_indices,
        )
    )
    camera_frame_ids = np.concatenate(
        (condition_camera_frames, *observation_camera_frames, target_camera_frames)
    )

    return H3PackedLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        condition_indices=condition_indices,
        target_indices=target_indices,
        condition_video_indices=condition_video_indices,
        audio_condition_indices=_empty_indices(),
        clean_video_indices=_empty_indices(),
        noisy_video_indices=target_video_indices,
        target_video_indices=target_video_indices,
        target_audio_indices=audio_indices,
        camera_video_indices=camera_video_indices,
        camera_frame_ids=camera_frame_ids,
        row_roles=row_roles,
        target_video_chunk_ids=target_chunk_ids,
        num_condition_video_rows=num_condition_video,
        num_condition_audio_rows=0,
        num_clean_video_rows=0,
        num_noisy_video_rows=num_target_video,
        rows_per_video_frame=rows_per_frame,
        latent_height=latent_height,
        latent_width=latent_width,
        patch_size=patch_size,
        stage=stage,
        window_chunks=window_chunks,
        chunk_latent_frames=chunk_latent_frames,
        observation_rows=tuple(observation_rows),
    )


def build_stage0p5_layout(
    text_token_tags: object,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int = 0,
    *,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_channels: int = 2,
    keyframe_anchors: tuple[str, ...] = ("first",),
    observations: tuple[H3VisualObservationSpec, ...] = (),
) -> H3PackedLayout:
    """Build Stage0.5's full-attention packed document."""

    return _base_layout(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_size=patch_size,
        audio_channels=audio_channels,
        keyframe_anchors=keyframe_anchors,
        stage="stage0p5",
        chunk_latent_frames=None,
        window_chunks=None,
        observations=observations,
    )


def build_stage1_layout(
    text_token_tags: object,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int = 0,
    *,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    audio_channels: int = 2,
    keyframe_anchors: tuple[str, ...] = ("first",),
    chunk_latent_frames: int = 5,
    window_chunks: int = 6,
) -> H3PackedLayout:
    """Build a six-chunk Stage1 row layout."""

    if window_chunks <= 0:
        raise ValueError(f"window_chunks must be positive, got {window_chunks}")
    base = _base_layout(
        text_token_tags,
        num_latent_frames,
        latent_height,
        latent_width,
        num_audio_latents,
        patch_size=patch_size,
        audio_channels=audio_channels,
        keyframe_anchors=keyframe_anchors,
        stage="stage1",
        chunk_latent_frames=chunk_latent_frames,
        window_chunks=window_chunks,
    )
    target_positions = base.position_ids[base.target_video_indices]
    target_start = int(base.target_video_indices[0])
    num_target_rows = int(base.target_video_indices.size)
    clean_start = target_start
    noisy_start = clean_start + num_target_rows
    sequence_length = noisy_start + num_target_rows

    position_ids = np.concatenate(
        (base.position_ids[:target_start], target_positions, target_positions), axis=0
    )
    clean_indices = _indices(clean_start, noisy_start)
    noisy_indices = _indices(noisy_start, sequence_length)
    video_indices = np.concatenate((base.condition_video_indices, clean_indices, noisy_indices))
    token_tags = np.empty(sequence_length, dtype=np.int64)
    token_tags[base.text_indices] = base.token_tags[base.text_indices]
    token_tags[base.audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    row_roles = np.full(sequence_length, CONDITION_ROLE, dtype=np.int64)
    row_roles[base.audio_indices] = AUDIO_ROLE
    row_roles[clean_indices] = CLEAN_VIDEO_ROLE
    row_roles[noisy_indices] = NOISY_VIDEO_ROLE

    frame_chunks = np.repeat(
        np.arange(num_latent_frames, dtype=np.int64) // chunk_latent_frames,
        base.rows_per_video_frame,
    )
    chunk_ids = np.full(sequence_length, -1, dtype=np.int64)
    chunk_ids[clean_indices] = frame_chunks
    chunk_ids[noisy_indices] = frame_chunks
    target_camera_frames = np.repeat(
        np.arange(num_latent_frames, dtype=np.int64), base.rows_per_video_frame
    )
    condition_camera_frames = base.camera_frame_ids[: base.num_condition_video_rows]
    camera_indices = np.concatenate((base.condition_video_indices, clean_indices, noisy_indices))
    camera_frames = np.concatenate(
        (condition_camera_frames, target_camera_frames, target_camera_frames)
    )
    condition_indices = np.concatenate((base.text_indices, base.condition_video_indices))
    target_indices = np.concatenate((base.audio_indices, noisy_indices))

    return H3PackedLayout(
        position_ids=position_ids,
        token_tags=token_tags,
        video_indices=video_indices,
        audio_indices=base.audio_indices,
        text_indices=base.text_indices,
        condition_indices=condition_indices,
        target_indices=target_indices,
        condition_video_indices=base.condition_video_indices,
        audio_condition_indices=_empty_indices(),
        clean_video_indices=clean_indices,
        noisy_video_indices=noisy_indices,
        target_video_indices=noisy_indices,
        target_audio_indices=base.audio_indices,
        camera_video_indices=camera_indices,
        camera_frame_ids=camera_frames,
        row_roles=row_roles,
        target_video_chunk_ids=chunk_ids,
        num_condition_video_rows=base.num_condition_video_rows,
        num_condition_audio_rows=0,
        num_clean_video_rows=num_target_rows,
        num_noisy_video_rows=num_target_rows,
        rows_per_video_frame=base.rows_per_video_frame,
        latent_height=latent_height,
        latent_width=latent_width,
        patch_size=patch_size,
        stage="stage1",
        window_chunks=window_chunks,
        chunk_latent_frames=chunk_latent_frames,
    )


def build_row_timesteps(
    layout: H3PackedLayout,
    video_timestep: object,
    audio_timestep: object,
    *,
    text_timestep: object | None = None,
    condition_video_timestep: float = 0.999,
    clean_video_timestep: float = 1.0,
    observation_timesteps: object | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted distinct times and each row's index into that array.

    ``observation_timesteps`` gives one known degree per visual observation; they
    default to the condition level.
    """

    video_values = np.asarray(video_timestep, dtype=np.float32).reshape(-1)
    if layout.stage == "stage0p5":
        rows = int(layout.noisy_video_indices.size)
        if video_values.size == 1:
            noisy_values = np.full(rows, video_values[0], dtype=np.float32)
            default_text = float(video_values[0])
        else:
            # A known-degree field: one time per latent frame or per target row.
            if video_values.size == rows // layout.rows_per_video_frame:
                noisy_values = np.repeat(video_values, layout.rows_per_video_frame)
            elif video_values.size == rows:
                noisy_values = video_values
            else:
                raise ValueError("Stage0.5 video_timestep must be scalar, per-frame, or per-row")
            if text_timestep is None:
                raise ValueError("a Stage0.5 known-degree field requires an explicit text_timestep")
            default_text = 0.0
    elif layout.stage == "stage1":
        chunk_frames = int(layout.chunk_latent_frames or 0)
        num_frames = layout.noisy_video_indices.size // layout.rows_per_video_frame
        num_chunks = num_frames // chunk_frames
        if video_values.size == 1:
            frame_values = np.full(num_frames, video_values[0], dtype=np.float32)
        elif video_values.size == num_chunks:
            frame_values = np.repeat(video_values, chunk_frames)
        elif video_values.size == num_frames:
            frame_values = video_values
        else:
            raise ValueError("Stage1 video_timestep must be scalar, per-chunk, or per-frame")
        noisy_values = np.repeat(frame_values, layout.rows_per_video_frame)
        default_text = 1.0
    else:
        raise ValueError(f"unsupported layout stage {layout.stage!r}")

    def scalar(value: object, name: str) -> float:
        array = np.asarray(value, dtype=np.float32)
        if array.size != 1:
            raise ValueError(f"{name} must be scalar, got shape {array.shape}")
        return float(array.reshape(()))

    row_times = np.zeros(layout.sequence_length, dtype=np.float32)
    row_times[layout.text_indices] = scalar(
        default_text if text_timestep is None else text_timestep, "text_timestep"
    )
    row_times[layout.condition_video_indices] = scalar(
        condition_video_timestep, "condition_video_timestep"
    )
    if observation_timesteps is not None:
        degrees = np.asarray(observation_timesteps, dtype=np.float32).reshape(-1)
        if degrees.size != len(layout.observation_rows):
            raise ValueError("observation_timesteps needs one degree per visual observation")
        row_times[layout.observation_video_indices] = np.repeat(
            degrees, np.asarray(layout.observation_rows, dtype=np.int64)
        )
    row_times[layout.audio_indices] = scalar(audio_timestep, "audio_timestep")
    row_times[layout.clean_video_indices] = scalar(clean_video_timestep, "clean_video_timestep")
    row_times[layout.noisy_video_indices] = noisy_values
    return np.unique(row_times, return_inverse=True)


def patchify_video(latents: object, patch_size: tuple[int, int, int] = (1, 2, 2)) -> object:
    """Pack ``[B,C,T,H,W]`` latents into ``[B,N,C*pt*ph*pw]`` rows."""

    is_torch = type(latents).__module__.startswith("torch")
    if is_torch:
        import torch

        if not isinstance(latents, torch.Tensor):
            raise TypeError("unrecognized torch-like latent value")
        values = latents
    else:
        values = np.asarray(latents)
    if values.ndim != 5:
        raise ValueError(f"latents must be [B,C,T,H,W], got {values.shape}")
    patch_t, patch_h, patch_w = patch_size
    batch, channels, frames, height, width = values.shape
    if frames % patch_t or height % patch_h or width % patch_w:
        raise ValueError(f"latents shape {values.shape} is not divisible by {patch_size}")
    rows = values.reshape(
        batch,
        channels,
        frames // patch_t,
        patch_t,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    )
    if is_torch:
        rows = rows.permute(0, 2, 4, 6, 1, 3, 5, 7)
        return rows.reshape(batch, -1, channels * patch_t * patch_h * patch_w).contiguous()
    rows = rows.transpose(0, 2, 4, 6, 1, 3, 5, 7)
    return np.ascontiguousarray(rows.reshape(batch, -1, channels * patch_t * patch_h * patch_w))


def unpatchify_video(
    rows: object,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    patch_size: tuple[int, int, int] = (1, 2, 2),
    channels: int | None = None,
) -> object:
    """Invert :func:`patchify_video` into ``[B,C,T,H,W]`` latents."""

    is_torch = type(rows).__module__.startswith("torch")
    if is_torch:
        import torch

        if not isinstance(rows, torch.Tensor):
            raise TypeError("unrecognized torch-like row value")
        values = rows
    else:
        values = np.asarray(rows)
    if values.ndim != 3:
        raise ValueError(f"rows must be [B,N,D], got {values.shape}")
    patch_t, patch_h, patch_w = patch_size
    if num_latent_frames % patch_t or latent_height % patch_h or latent_width % patch_w:
        raise ValueError("requested latent grid is not divisible by patch_size")
    patch_volume = patch_t * patch_h * patch_w
    if channels is None:
        if values.shape[-1] % patch_volume:
            raise ValueError("row width is not divisible by patch volume")
        channels = values.shape[-1] // patch_volume
    expected_rows = (
        (num_latent_frames // patch_t) * (latent_height // patch_h) * (latent_width // patch_w)
    )
    if values.shape[1:] != (expected_rows, channels * patch_volume):
        raise ValueError(
            f"rows shape {values.shape} does not match expected "
            f"N={expected_rows}, D={channels * patch_volume}"
        )
    batch = values.shape[0]
    latents = values.reshape(
        batch,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    if is_torch:
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7)
        return latents.reshape(
            batch, channels, num_latent_frames, latent_height, latent_width
        ).contiguous()
    latents = latents.transpose(0, 4, 1, 5, 2, 6, 3, 7)
    return np.ascontiguousarray(
        latents.reshape(batch, channels, num_latent_frames, latent_height, latent_width)
    )


__all__ = [
    "AUDIO_ROLE",
    "AUDIO_TAG",
    "CLEAN_VIDEO_ROLE",
    "CONDITION_ROLE",
    "NOISY_VIDEO_ROLE",
    "PADDING_ROLE",
    "TARGET_AUDIO_ROLE",
    "TARGET_VIDEO_ROLE",
    "TEXT_TAG",
    "VIDEO_TAG",
    "H3PackedLayout",
    "H3VisualObservationSpec",
    "build_row_timesteps",
    "build_stage0p5_layout",
    "build_stage1_layout",
    "patchify_video",
    "unpatchify_video",
]
