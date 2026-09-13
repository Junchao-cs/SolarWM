"""H3 SGF conditioning, bidirectional scoring and causal generation."""

from __future__ import annotations

from typing import Any

from .distributed import broadcast_sp_tensor, is_sequence_parallel_enabled
from .layout import build_stage0p5_layout, patchify_video, unpatchify_video
from .sgf import h3_sgf_score_timesteps
from .sgf_rollout import H3SGFInputs, h3_sgf_rollout
from .stage0p5 import H3Stage0p5Core
from .torch_flow import sample_shifted_timestep, scale_noise
from .torch_layout import build_row_timesteps


class H3SGFCore(H3Stage0p5Core):
    """Share the exact fixed-audio input construction between rollout and validation."""

    def random_like(self, value: Any, *, generator: Any = None) -> Any:
        import torch

        noise = torch.randn(
            value.shape, device=value.device, dtype=value.dtype, generator=generator
        )
        return broadcast_sp_tensor(noise)

    def prepare_inputs(self, batch: Any, *, generator: Any = None) -> H3SGFInputs:
        import torch

        prompt = batch.prompt_embeds.unsqueeze(0).to(self.device, torch.bfloat16)
        tags = batch.text_token_tags.to(self.device, torch.long)
        anchor = batch.anchor_latents.unsqueeze(0).to(self.device, torch.float32)
        views = batch.camera_viewmats.unsqueeze(0).to(self.device, torch.float32)
        intrinsics = batch.camera_K.unsqueeze(0).to(self.device, torch.float32)
        self._broadcast_logical_sample(prompt, tags, anchor, views, intrinsics)
        if views.shape[1] == 47 * 1008:
            for value in (views, intrinsics):
                grouped = value.unflatten(1, (47, 1008))
                if not torch.equal(grouped, grouped[:, :, :1].expand_as(grouped)):
                    raise ValueError("H3 SGF needs one camera per latent; spatial cameras differ")
            views, intrinsics = views[:, ::1008], intrinsics[:, ::1008]
        if views.shape[1] != 47 or intrinsics.shape[1] != 47:
            raise ValueError("H3 SGF requires the exact 47 source latent cameras")
        views = torch.cat((views[:, :1], views, views[:, -1:].expand(-1, 3, -1, -1)), 1)
        intrinsics = torch.cat(
            (intrinsics[:, :1], intrinsics, intrinsics[:, -1:].expand(-1, 3, -1, -1)), 1
        )
        anchor = scale_noise(anchor, self.random_like(anchor, generator=generator), 0.999)
        audio = self.audio_rows()
        audio_time = sample_shifted_timestep(
            1,
            3.0,
            self.device,
            torch.float32,
            generator=generator,
        ).reshape(())
        broadcast_sp_tensor(audio_time)
        audio = scale_noise(audio, self.random_like(audio, generator=generator), audio_time)
        return H3SGFInputs(
            prompt,
            tags,
            patchify_video(anchor),
            audio,
            audio_time,
            views,
            intrinsics,
            48,
            84,
            0.999,
            is_sequence_parallel_enabled(),
        )

    def rollout(
        self,
        inputs: H3SGFInputs,
        *,
        generator: Any = None,
        exit_index: int | None = None,
        progress: Any = None,
    ) -> Any:
        import torch

        noise = torch.empty((1, 24, 50, 48, 84), device=self.device, dtype=torch.float32)
        return h3_sgf_rollout(
            student=self.model,
            inputs=inputs,
            noise=self.random_like(noise, generator=generator),
            exit_index=exit_index,
            generator=generator,
            progress=progress,
        )

    def score_inputs(self, clean: Any) -> tuple[Any, ...]:
        video_time, audio_time = h3_sgf_score_timesteps(device=self.device)
        self._broadcast_logical_sample(video_time, audio_time)
        noise = self.random_like(clean)
        noisy = scale_noise(clean, noise, video_time)
        silence = self.audio_rows()
        audio = scale_noise(silence, self.random_like(silence), audio_time)
        return noisy, noise, video_time, audio, audio_time

    def score_forward(
        self,
        model: Any,
        inputs: H3SGFInputs,
        noisy: Any,
        video_time: Any,
        audio: Any,
        audio_time: Any,
    ) -> Any:
        import torch

        layout = build_stage0p5_layout(inputs.text_tags.cpu(), 47, 48, 84, audio.shape[1] // 2).to(
            self.device
        )
        times, indices = build_row_timesteps(
            layout,
            video_time,
            audio_time,
            condition_video_timestep=0.999,
            device=self.device,
        )
        frame_ids = layout.camera_frame_ids.clone()
        frame_ids[layout.num_condition_video_rows :] += 1
        velocity, _ = model(
            hidden_states=torch.cat((inputs.anchor_rows, patchify_video(noisy)), 1),
            audio_hidden_states=audio,
            encoder_hidden_states=inputs.prompt,
            timestep=times,
            timestep_indices=indices,
            fused_prope=True,
            prope_token_indices=layout.camera_video_indices,
            prope_frame_ids=frame_ids,
            cam_viewmats=inputs.camera_viewmats[:, :48],
            cam_K=inputs.camera_K[:, :48],
            packed_sequence_parallel=is_sequence_parallel_enabled(),
            return_dict=False,
            **layout.transformer_kwargs(),
        )
        return unpatchify_video(velocity[:, layout.noisy_video_output_slice], 47, 48, 84)

    def generate(self, batch: Any, *, noise_seed: int, num_inference_steps: int) -> Any:
        import torch

        if num_inference_steps != 4:
            raise ValueError("H3 SGF validation requires four denoiser evaluations")
        generator = torch.Generator(device=self.device).manual_seed(int(noise_seed))
        with torch.no_grad():
            inputs = self.prepare_inputs(batch, generator=generator)
            return self.rollout(inputs, generator=generator, exit_index=3).cache_target[:, :, :47]
