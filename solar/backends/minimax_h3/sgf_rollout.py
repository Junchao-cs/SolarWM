"""H3's two-pass SGF: detached raw-KV rollout, then parallel gradient replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .distributed import broadcast_sp_tensor as broadcast_sequence_parallel_tensor
from .layout import build_stage0p5_layout, build_stage1_layout, patchify_video, unpatchify_video
from .sgf import H3SGFWindow, h3_sgf_schedule, h3_sgf_windows
from .sgf_attention import H3RawKVCache, H3SGFAttention
from .torch_flow import predict_clean_sample, scale_noise
from .torch_layout import build_row_timesteps


@dataclass
class H3SGFInputs:
    prompt: torch.Tensor
    text_tags: torch.Tensor
    anchor_rows: torch.Tensor
    audio_rows: torch.Tensor
    audio_timestep: torch.Tensor
    camera_viewmats: torch.Tensor  # [B,51,4,4], anchor + 50 target cameras
    camera_K: torch.Tensor
    latent_height: int
    latent_width: int
    keyframe_timestep: float = 0.999
    sp_enabled: bool = False
    layouts: dict[str, Any] = field(default_factory=dict)

    def layout(self, mode):
        if mode not in self.layouts:
            builder = build_stage1_layout if mode == "replay" else build_stage0p5_layout
            self.layouts[mode] = builder(
                self.text_tags.cpu(),
                50 if mode == "replay" else 5,
                self.latent_height,
                self.latent_width,
                self.audio_rows.shape[1] // 2,
            ).to(self.prompt.device)
        return self.layouts[mode]


@dataclass(frozen=True)
class H3SGFRollout:
    cache_target: torch.Tensor
    noisy_at_exit: torch.Tensor
    exit_index: int
    native_exit_timestep: torch.Tensor


def _random_like(value, *, inputs, generator=None):
    noise = torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator)
    if inputs.sp_enabled:
        broadcast_sequence_parallel_tensor(noise)
    return noise


def h3_student_forward(
    student,
    inputs: H3SGFInputs,
    noisy,
    timestep,
    *,
    clean=None,
    chunk_index=0,
    cache=None,
    commit_cache=False,
):
    mode = "replay" if clean is not None else "rollout"
    layout = inputs.layout(mode)
    frames = 50 if mode == "replay" else 5
    video_parts = [inputs.anchor_rows]
    if clean is not None:
        video_parts.append(patchify_video(clean.detach()))
    video_parts.append(patchify_video(noisy))
    times, indices = build_row_timesteps(
        layout,
        timestep,
        inputs.audio_timestep,
        text_timestep=1.0,
        condition_video_timestep=inputs.keyframe_timestep,
        device=noisy.device,
    )
    control = H3SGFAttention(
        layout,
        inputs.camera_viewmats,
        inputs.camera_K,
        mode,
        chunk_index=chunk_index,
        cache=cache,
        commit_cache=commit_cache,
    )
    # SGF's camera controls are global immutable metadata. The dedicated
    # attention path selects local query/KV camera rows after Ulysses exchange.
    prediction, _ = student(
        hidden_states=torch.cat(video_parts, dim=1),
        audio_hidden_states=inputs.audio_rows,
        encoder_hidden_states=inputs.prompt,
        timestep=times,
        timestep_indices=indices,
        fused_prope=True,
        packed_sequence_parallel=inputs.sp_enabled,
        sgf_attention=control,
        return_dict=False,
        **layout.transformer_kwargs(),
    )
    velocity = unpatchify_video(
        prediction[:, layout.noisy_video_output_slice],
        frames,
        inputs.latent_height,
        inputs.latent_width,
    )
    return predict_clean_sample(noisy, velocity, timestep)


def h3_sgf_rollout(
    *,
    student,
    inputs: H3SGFInputs,
    noise,
    exit_index=None,
    per_rank_exit_step=True,
    last_step_only=False,
    generator=None,
    video_shift=12.0,
    progress=None,
    inference_full_length=False,
    output_device=None,
):
    if output_device is not None and not inference_full_length:
        raise ValueError("Output offload is inference-only")
    if inference_full_length:
        count = int(noise.shape[2])
        if count <= 0 or count % 5 or exit_index != 3:
            raise ValueError(
                "Full-length H3 inference requires complete five-latent chunks and exit_index=3"
            )
        if inputs.camera_viewmats.shape[1] != count + 1 or inputs.camera_K.shape[1] != count + 1:
            raise ValueError("Full-length H3 inference requires anchor plus every rollout camera")
        windows = tuple(
            H3SGFWindow(i, max(0, i - 5) * 5, i * 5, (i + 1) * 5, (i + 1) * 5)
            for i in range(count // 5)
        )
    elif noise.shape[2] != 50:
        raise ValueError("H3 SGF requires exactly 50 rollout latents")
    else:
        windows = h3_sgf_windows()
    schedule, _ = h3_sgf_schedule(video_shift=video_shift, device=noise.device)
    if exit_index is None:
        selected = (
            torch.full((), 3, device=noise.device, dtype=torch.long)
            if last_step_only
            else torch.randint(0, 4, (), device=noise.device, generator=generator)
        )
        if not per_rank_exit_step and torch.distributed.is_initialized():
            torch.distributed.broadcast(selected, src=0)
        if inputs.sp_enabled:
            broadcast_sequence_parallel_tensor(selected)
        exit_index = int(selected)
    if not 0 <= exit_index < 4:
        raise ValueError("H3 SGF exit index must lie in [0,4)")
    cache = H3RawKVCache()
    committed, noisy_exits = [], []
    try:
        with torch.no_grad():
            for window in windows:
                current = noise[:, :, window.current_start : window.stop]
                exit_x0 = None
                # Keep four forwards on every rank even when exit choices
                # differ across DP: FSDP collectives require identical counts.
                for step_index, timestep in enumerate(schedule.timesteps):
                    if step_index == exit_index:
                        noisy_exits.append(
                            current.detach().to(output_device).clone()
                            if output_device
                            else current.detach().clone()
                        )
                    x0 = h3_student_forward(
                        student,
                        inputs,
                        current,
                        timestep,
                        chunk_index=window.chunk_index,
                        cache=cache,
                    )
                    if not bool(torch.isfinite(x0).all()):
                        raise FloatingPointError(
                            f"non-finite H3 SGF chunk={window.chunk_index}, denoise={step_index}"
                        )
                    if step_index == exit_index:
                        exit_x0 = x0.detach().clone()
                    if step_index < 3:
                        current = scale_noise(
                            x0,
                            _random_like(x0, inputs=inputs, generator=generator),
                            schedule.timesteps[step_index + 1],
                        )
                committed.append(exit_x0.to(output_device) if output_device else exit_x0)
                h3_student_forward(
                    student,
                    inputs,
                    exit_x0,
                    noise.new_tensor(1.0),
                    chunk_index=window.chunk_index,
                    cache=cache,
                    commit_cache=True,
                )
                if progress is not None:
                    progress(window.chunk_index + 1)
        return H3SGFRollout(
            torch.cat(committed, dim=2),
            torch.cat(noisy_exits, dim=2),
            exit_index,
            schedule.timesteps[exit_index],
        )
    finally:
        cache.clear()


def h3_sgf_replay(*, student, inputs, rollout):
    return h3_student_forward(
        student,
        inputs,
        rollout.noisy_at_exit,
        rollout.native_exit_timestep,
        clean=rollout.cache_target,
    )
