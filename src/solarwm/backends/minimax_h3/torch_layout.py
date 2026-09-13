"""Torch timestep grouping over the shared public H3 row layout."""

from __future__ import annotations

from typing import Any

import torch

from .layout import H3PackedLayout


def build_row_timesteps(
    layout: H3PackedLayout,
    video_timestep: Any,
    audio_timestep: Any,
    *,
    text_timestep: Any | None = None,
    condition_video_timestep: Any = 0.999,
    clean_video_timestep: Any = 1.0,
    device: Any = None,
) -> tuple[Any, Any]:
    """Build H3's ``(distinct_timesteps, row_to_timestep_index)`` pair.

    Stage0.5 requires a scalar ``video_timestep``. Stage1 accepts a scalar, one
    value per chunk, or one value per latent frame; only noisy-video rows use
    those sampled values. Clean rows use ``t=1``, keyframe anchors use their
    fixed augmentation level, and audio uses an independent scalar shift-3
    sample. ``text_timestep`` defaults to video time in Stage0.5 and clean
    ``t=1`` in Stage1.
    """

    import torch

    def tensor(value: Any) -> Any:
        return torch.as_tensor(value, dtype=torch.float32, device=device).flatten()

    def scalar(value: Any, name: str) -> Any:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.numel() != 1:
            raise ValueError(f"{name} must be scalar, got {tuple(tensor.shape)}")
        return tensor.reshape(())

    video_values = tensor(video_timestep)
    if layout.stage == "stage0p5":
        if video_values.numel() != 1:
            raise ValueError("Stage0.5 video_timestep must be scalar")
        noisy_row_values = video_values.expand(layout.noisy_video_indices.numel())
        default_text = video_values[0]
    elif layout.stage == "stage1":
        num_frames = layout.noisy_video_indices.numel() // layout.rows_per_video_frame
        num_chunks = num_frames // int(layout.chunk_latent_frames)
        if video_values.numel() == 1:
            frame_values = video_values.expand(num_frames)
        elif video_values.numel() == num_chunks:
            frame_values = video_values.repeat_interleave(int(layout.chunk_latent_frames))
        elif video_values.numel() == num_frames:
            frame_values = video_values
        else:
            raise ValueError(
                "Stage1 video_timestep must be scalar, [num_chunks] or [num_latent_frames]; "
                f"got {video_values.numel()} values for {num_chunks} chunks/{num_frames} frames"
            )
        noisy_row_values = frame_values.repeat_interleave(layout.rows_per_video_frame)
        default_text = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        raise ValueError(f"unsupported layout stage {layout.stage!r}")

    row_timesteps = torch.zeros(layout.sequence_length, dtype=torch.float32, device=device)
    row_timesteps[layout.text_indices.to(device=device)] = scalar(
        default_text if text_timestep is None else text_timestep, "text_timestep"
    )
    row_timesteps[layout.condition_video_indices.to(device=device)] = scalar(
        condition_video_timestep, "condition_video_timestep"
    )
    row_timesteps[layout.audio_indices.to(device=device)] = scalar(audio_timestep, "audio_timestep")
    if int(layout.clean_video_indices.numel()):
        row_timesteps[layout.clean_video_indices.to(device=device)] = scalar(
            clean_video_timestep, "clean_video_timestep"
        )
    row_timesteps[layout.noisy_video_indices.to(device=device)] = noisy_row_values
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


def _unique_timestep_plan(row_timesteps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if row_timesteps.ndim != 1:
        raise ValueError("row_timesteps must be one-dimensional")
    return torch.unique(row_timesteps.float(), sorted=True, return_inverse=True)


def build_stage1_row_timesteps(
    layout: Any,
    video_chunk_timesteps: torch.Tensor,
    audio_timestep: torch.Tensor,
    *,
    keyframe_timestep: float = 0.999,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Assign per-chunk Stage1 timesteps to the explicit clean/noisy layout."""

    chunks = video_chunk_timesteps.reshape(-1).to(device=device, dtype=torch.float32)
    expected_chunks = int(layout.target_video_chunk_ids.max().item()) + 1
    if chunks.numel() != expected_chunks:
        raise ValueError(
            f"expected {expected_chunks} Stage1 video chunk timesteps, got {chunks.numel()}"
        )
    rows = torch.ones(layout.sequence_length, dtype=torch.float32, device=device)
    rows[layout.condition_video_indices] = float(keyframe_timestep)
    rows[layout.audio_indices] = audio_timestep.reshape(())
    noisy_chunks = layout.target_video_chunk_ids.index_select(0, layout.noisy_video_indices)
    rows[layout.noisy_video_indices] = chunks.index_select(0, noisy_chunks)
    # Text and clean teacher-forcing rows remain clean conditions at t=1.
    return _unique_timestep_plan(rows)


def build_stage1_row_time_pairs(layout, video_t, video_r, audio_t, *, keyframe_timestep, device):
    """Group by the pair, so equal t with different destination r stays distinct."""
    chunks = int(layout.target_video_chunk_ids.max().item()) + 1
    t, t_index = build_stage1_row_timesteps(
        layout,
        video_t.expand(chunks),
        audio_t,
        keyframe_timestep=keyframe_timestep,
        device=device,
    )
    t_rows = t[t_index]
    r_rows = t_rows.clone()
    r_rows[layout.noisy_video_indices] = video_r.reshape(())
    pairs, indices = torch.unique(
        torch.stack((t_rows, r_rows), dim=1), sorted=True, dim=0, return_inverse=True
    )
    return pairs[:, 0], pairs[:, 1], indices


def _unique_row_timestep_plan(
    layout: Any,
    *,
    current_local_chunk: int,
    video_timestep: Any,
    audio_timestep: Any,
    keyframe_timestep: float,
    device: Any,
) -> tuple[Any, Any]:
    import torch

    rows = torch.ones(layout.sequence_length, dtype=torch.float32, device=device)
    rows[layout.condition_video_indices] = float(keyframe_timestep)
    rows[layout.audio_indices] = torch.as_tensor(audio_timestep, device=device).reshape(())
    noisy_chunks = layout.target_video_chunk_ids.index_select(0, layout.noisy_video_indices)
    current_rows = layout.noisy_video_indices[noisy_chunks == int(current_local_chunk)]
    if int(current_rows.numel()) != 5 * int(layout.rows_per_video_frame):
        raise ValueError("current noisy chunk does not contain exactly five latent frames")
    rows[current_rows] = torch.as_tensor(video_timestep, device=device).reshape(())
    return torch.unique(rows, sorted=True, return_inverse=True)


def _unique_row_interval_plan(
    layout: Any,
    *,
    current_local_chunk: int,
    video_timestep: Any,
    video_r_timestep: Any,
    audio_timestep: Any,
    keyframe_timestep: float,
    device: Any,
) -> tuple[Any, Any, Any]:
    """Group native H3 ``(t,r)`` pairs without merging unequal intervals.

    Only current noisy-video rows carry a finite-map interval. Conditions,
    audio and clean history retain their original diagonal conditioning.
    """

    import torch

    times, inverse = _unique_row_timestep_plan(
        layout,
        current_local_chunk=current_local_chunk,
        video_timestep=video_timestep,
        audio_timestep=audio_timestep,
        keyframe_timestep=keyframe_timestep,
        device=device,
    )
    rows = times[inverse]
    r_rows = rows.clone()
    noisy_chunks = layout.target_video_chunk_ids.index_select(0, layout.noisy_video_indices)
    current_rows = layout.noisy_video_indices[noisy_chunks == int(current_local_chunk)]
    r_rows[current_rows] = torch.as_tensor(video_r_timestep, device=device).reshape(())
    pairs, pair_inverse = torch.unique(
        torch.stack((rows, r_rows), dim=1), dim=0, sorted=True, return_inverse=True
    )
    return pairs[:, 0], pairs[:, 1], pair_inverse
