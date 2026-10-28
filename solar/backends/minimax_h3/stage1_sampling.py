"""Stage1's absolute-coordinate W6 sampler with four finite-map evaluations."""

from __future__ import annotations

from typing import Any

from .distributed import broadcast_sp_tensor, is_sequence_parallel_enabled
from .layout import build_stage1_layout, patchify_video, unpatchify_video
from .torch_flow import make_shifted_schedule, scale_noise
from .torch_geometry import native_video_position_grid
from .torch_layout import _unique_row_interval_plan
from .torch_mask import build_stage1_w6_block_mask


def generate_stage1(model: Any, batch: Any, silence: Any, *, device: Any, seed: int) -> Any:
    import torch

    if batch.rollout_camera_viewmats is None or batch.rollout_camera_K is None:
        raise ValueError("Stage1 validation requires the real 170-frame source camera trajectory")
    generator = torch.Generator(device=device).manual_seed(int(seed))

    def noise_like(value: Any) -> Any:
        noise = torch.randn(value.shape, device=device, dtype=torch.float32, generator=generator)
        return broadcast_sp_tensor(noise)

    prompt = batch.prompt_embeds.unsqueeze(0).to(device, torch.bfloat16)
    anchor = batch.anchor_latents.unsqueeze(0).to(device, torch.float32)
    tags = batch.text_token_tags.to(device, torch.long)
    views = batch.rollout_camera_viewmats.to(device, torch.float32)
    intrinsics = batch.rollout_camera_K.to(device, torch.float32)
    for value in (prompt, anchor, tags, views, intrinsics):
        broadcast_sp_tensor(value)
    if views.shape != (50, 4, 4) or intrinsics.shape != (50, 3, 3):
        raise ValueError("Stage1 rollout cameras must describe all 50 latent positions")
    anchor_rows = patchify_video(scale_noise(anchor, noise_like(anchor), 0.999))
    audio_clean = silence.to(device, torch.float32).unsqueeze(0)
    audio_clean = audio_clean.permute(0, 1, 3, 2).reshape(1, -1, 32).contiguous()
    audio_noise = noise_like(audio_clean)
    sigmas = torch.linspace(1.0, 0.0, 5, device=device, dtype=torch.float32)
    audio_schedule = make_shifted_schedule(5, shift=3.0, device=device)
    positions = native_video_position_grid(50, 48, 84, origin=float(tags.numel())).to(device)
    generated = []
    masks = {}
    with torch.no_grad():
        for chunk in range(10):
            first = max(0, chunk - 5)
            frames = (chunk - first + 1) * 5
            layout = build_stage1_layout(tags.cpu(), frames, 48, 84, 283).to(device)
            selected_positions = positions[first * 5 * 1008 : (chunk + 1) * 5 * 1008]
            layout.position_ids.index_copy_(0, layout.clean_video_indices, selected_positions)
            layout.position_ids.index_copy_(0, layout.noisy_video_indices, selected_positions)
            if frames not in masks:
                masks[frames] = build_stage1_w6_block_mask(
                    layout, batch_size=1, num_heads=None, device=device
                )
            placeholder = torch.zeros((1, 24, 5, 48, 84), device=device, dtype=torch.float32)
            clean = torch.cat((*generated[first:chunk], placeholder), dim=2)
            noisy = torch.zeros_like(clean)
            current = noise_like(placeholder)
            local_views = torch.cat((views[:1], views[first * 5 : (chunk + 1) * 5]), 0).unsqueeze(0)
            local_K = torch.cat(
                (intrinsics[:1], intrinsics[first * 5 : (chunk + 1) * 5]), 0
            ).unsqueeze(0)
            frame_ids = layout.camera_frame_ids.clone()
            frame_ids[layout.num_condition_video_rows :] += 1
            for step in range(4):
                noisy[:, :, -5:] = current
                audio_t = audio_schedule.timesteps[step]
                audio = scale_noise(audio_clean, audio_noise, audio_t)
                times, targets, indices = _unique_row_interval_plan(
                    layout,
                    current_local_chunk=chunk - first,
                    video_timestep=1.0 - sigmas[step],
                    video_r_timestep=1.0 - sigmas[step + 1],
                    audio_timestep=audio_t,
                    keyframe_timestep=0.999,
                    device=device,
                )
                prediction, _ = model(
                    hidden_states=torch.cat(
                        (anchor_rows, patchify_video(clean), patchify_video(noisy)), 1
                    ),
                    audio_hidden_states=audio,
                    encoder_hidden_states=prompt,
                    timestep=times,
                    r_timestep=targets,
                    timestep_indices=indices,
                    attention_mask=masks[frames],
                    fused_prope=True,
                    prope_token_indices=layout.camera_video_indices,
                    prope_frame_ids=frame_ids,
                    cam_viewmats=local_views,
                    cam_K=local_K,
                    packed_sequence_parallel=is_sequence_parallel_enabled(),
                    return_dict=False,
                    **layout.transformer_kwargs(),
                )
                velocity = unpatchify_video(
                    prediction[:, layout.noisy_video_output_slice][:, -5 * 1008 :], 5, 48, 84
                )
                current = current + (sigmas[step] - sigmas[step + 1]) * velocity.float()
            if not bool(torch.isfinite(current).all()):
                raise FloatingPointError(f"non-finite H3 Stage1 validation chunk {chunk}")
            generated.append(current.contiguous())
    return torch.cat(generated, 2)[:, :, :47]
