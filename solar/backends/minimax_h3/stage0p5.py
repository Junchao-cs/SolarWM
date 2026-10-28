"""Exact Stage0.5 data-ward flow-matching forward and Euler sampler."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .artifacts import H3ArtifactBatch
from .distributed import broadcast_sp_tensor, is_sequence_parallel_enabled
from .flow import make_shifted_schedule
from .layout import (
    H3PackedLayout,
    build_row_timesteps,
    build_stage0p5_layout,
    patchify_video,
    unpatchify_video,
)


@dataclass(frozen=True)
class H3TorchLayout:
    source: H3PackedLayout
    position_ids: Any
    token_tags: Any
    video_indices: Any
    audio_indices: Any
    text_indices: Any
    camera_video_indices: Any
    camera_frame_ids: Any

    @property
    def target_video_output_slice(self) -> slice:
        return self.source.target_video_output_slice


class H3Stage0p5Core:
    """One model/silence/device implementation shared by train and inference."""

    def __init__(self, model: Any, silence_latents: Any, device: Any) -> None:
        self.model = model
        self.device = device
        self.silence = silence_latents
        self._layouts: dict[str, H3TorchLayout] = {}

    def layout(self, tags: Any) -> H3TorchLayout:
        import torch

        tags_cpu = tags.detach().cpu().long().numpy()
        key = hashlib.blake2s(tags_cpu.tobytes()).hexdigest()
        cached = self._layouts.get(key)
        if cached is not None:
            return cached
        source = build_stage0p5_layout(tags_cpu, 47, 48, 84, 263)
        layout = H3TorchLayout(
            source=source,
            position_ids=torch.from_numpy(source.position_ids).to(self.device),
            token_tags=torch.from_numpy(source.token_tags).long().to(self.device),
            video_indices=torch.from_numpy(source.video_indices).long().to(self.device),
            audio_indices=torch.from_numpy(source.audio_indices).long().to(self.device),
            text_indices=torch.from_numpy(source.text_indices).long().to(self.device),
            camera_video_indices=torch.from_numpy(source.camera_video_indices)
            .long()
            .to(self.device),
            camera_frame_ids=torch.from_numpy(source.camera_frame_ids).long().to(self.device),
        )
        self._layouts[key] = layout
        while len(self._layouts) > 4:
            self._layouts.pop(next(iter(self._layouts)))
        return layout

    def audio_rows(self) -> Any:
        audio = self.silence.to(
            device=self.device, dtype=__import__("torch").float32, non_blocking=True
        ).unsqueeze(0)
        return audio.permute(0, 1, 3, 2).reshape(1, -1, 32).contiguous()

    @staticmethod
    def _broadcast_logical_sample(*values: Any) -> None:
        """Make SP rank zero authoritative for every model input tensor."""

        if is_sequence_parallel_enabled():
            for value in values:
                broadcast_sp_tensor(value)

    def _camera(self, batch: H3ArtifactBatch, layout: H3TorchLayout) -> tuple[Any, Any, Any]:
        import torch

        views = batch.camera_viewmats.to(
            self.device, dtype=torch.float32, non_blocking=True
        ).unsqueeze(0)
        K = batch.camera_K.to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(0)
        if views.shape[1] == 47:
            return views, K, layout.camera_frame_ids
        if views.shape[1] == 47 * 1008:
            anchor_views = views[:, :1008]
            anchor_K = K[:, :1008]
            return (
                torch.cat((anchor_views, views), dim=1),
                torch.cat((anchor_K, K), dim=1),
                None,
            )
        raise RuntimeError("H3 camera artifact is neither latent nor token aligned")

    @staticmethod
    def shifted_timestep(generator: Any | None, *, shift: float, device: Any) -> Any:
        import torch

        sigma = torch.rand((), generator=generator, device=device, dtype=torch.float32)
        shifted = float(shift) * sigma / (1.0 + (float(shift) - 1.0) * sigma)
        return 1.0 - shifted

    @staticmethod
    def _row_times(
        layout: H3TorchLayout, video_t: Any, audio_t: Any, device: Any
    ) -> tuple[Any, Any]:
        import torch

        times, inverse = build_row_timesteps(
            layout.source,
            float(video_t.item()),
            float(audio_t.item()),
            condition_video_timestep=0.999,
        )
        return (
            torch.from_numpy(times).to(device=device, dtype=torch.float32),
            torch.from_numpy(inverse).to(device=device, dtype=torch.long),
        )

    def _forward(
        self,
        *,
        video_rows: Any,
        audio_rows: Any,
        prompt: Any,
        layout: H3TorchLayout,
        video_t: Any,
        audio_t: Any,
        viewmats: Any,
        K: Any,
        frame_ids: Any,
    ) -> Any:
        times, inverse = self._row_times(layout, video_t, audio_t, self.device)
        video, _audio = self.model(
            hidden_states=video_rows,
            audio_hidden_states=audio_rows,
            encoder_hidden_states=prompt,
            timestep=times,
            timestep_indices=inverse,
            token_tags=layout.token_tags,
            position_ids=layout.position_ids,
            video_indices=layout.video_indices,
            audio_indices=layout.audio_indices,
            text_indices=layout.text_indices,
            attention_mask=None,
            fused_prope=True,
            prope_token_indices=layout.camera_video_indices,
            prope_frame_ids=frame_ids,
            cam_viewmats=viewmats,
            cam_K=K,
            stage0p5_sequence_parallel=is_sequence_parallel_enabled(),
            return_dict=False,
        )
        return video[:, layout.target_video_output_slice]

    def forward_loss(self, batch: H3ArtifactBatch, *, noise_seed: int | None) -> Any:
        import torch
        import torch.nn.functional as F

        generator = (
            None
            if noise_seed is None
            else torch.Generator(device=self.device).manual_seed(int(noise_seed))
        )
        clean = batch.target_latents.to(
            self.device, dtype=torch.float32, non_blocking=True
        ).unsqueeze(0)
        anchor = batch.anchor_latents.to(
            self.device, dtype=torch.float32, non_blocking=True
        ).unsqueeze(0)
        prompt = batch.prompt_embeds.to(
            self.device, dtype=torch.bfloat16, non_blocking=True
        ).unsqueeze(0)
        tags = batch.text_token_tags.to(self.device, dtype=torch.long)
        self._broadcast_logical_sample(tags)
        layout = self.layout(tags)
        viewmats, K, frame_ids = self._camera(batch, layout)
        self._broadcast_logical_sample(clean, anchor, prompt, viewmats, K)
        anchor_noise = torch.randn(
            anchor.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        audio_clean = self.audio_rows()
        audio_noise = torch.randn(
            audio_clean.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        audio_t = self.shifted_timestep(generator, shift=3.0, device=self.device)
        video_noise = torch.randn(
            clean.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        video_t = self.shifted_timestep(generator, shift=12.0, device=self.device)
        if is_sequence_parallel_enabled():
            for value in (anchor_noise, audio_noise, video_noise, video_t, audio_t):
                broadcast_sp_tensor(value)
        anchor_t = torch.tensor(0.999, device=self.device, dtype=anchor.dtype)
        anchor_noisy = anchor_t * anchor + (1.0 - anchor_t) * anchor_noise
        audio_noisy = audio_t * audio_clean + (1.0 - audio_t) * audio_noise
        video_noisy = video_t * clean + (1.0 - video_t) * video_noise
        video_rows = torch.cat((patchify_video(anchor_noisy), patchify_video(video_noisy)), dim=1)
        target_rows = patchify_video(clean - video_noise)
        prediction = self._forward(
            video_rows=video_rows,
            audio_rows=audio_noisy,
            prompt=prompt,
            layout=layout,
            video_t=video_t,
            audio_t=audio_t,
            viewmats=viewmats,
            K=K,
            frame_ids=frame_ids,
        )
        if tuple(prediction.shape) != tuple(target_rows.shape):
            raise RuntimeError(
                f"H3 predicted rows {tuple(prediction.shape)} != {tuple(target_rows.shape)}"
            )
        return F.mse_loss(prediction.float(), target_rows.float())

    def generate(self, batch: H3ArtifactBatch, *, noise_seed: int, num_inference_steps: int) -> Any:
        import torch

        generator = torch.Generator(device=self.device).manual_seed(int(noise_seed))
        anchor = batch.anchor_latents.to(
            self.device, dtype=torch.float32, non_blocking=True
        ).unsqueeze(0)
        prompt = batch.prompt_embeds.to(
            self.device, dtype=torch.bfloat16, non_blocking=True
        ).unsqueeze(0)
        tags = batch.text_token_tags.to(self.device, dtype=torch.long)
        self._broadcast_logical_sample(tags)
        layout = self.layout(tags)
        viewmats, K, frame_ids = self._camera(batch, layout)
        self._broadcast_logical_sample(anchor, prompt, viewmats, K)
        anchor_noise = torch.randn(
            anchor.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        audio_clean = self.audio_rows()
        audio_noise = torch.randn(
            audio_clean.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        current = torch.randn(
            (1, 24, 47, 48, 84),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        if is_sequence_parallel_enabled():
            for value in (anchor_noise, audio_noise, current):
                broadcast_sp_tensor(value)
        anchor_rows = patchify_video(0.999 * anchor + 0.001 * anchor_noise)
        video_schedule = make_shifted_schedule(num_inference_steps, shift=12.0)
        audio_schedule = make_shifted_schedule(num_inference_steps, shift=3.0)
        with torch.no_grad():
            for index, (video_time, audio_time) in enumerate(
                zip(
                    video_schedule.timesteps,
                    audio_schedule.timesteps,
                    strict=True,
                )
            ):
                video_t = torch.tensor(float(video_time), device=self.device)
                audio_t = torch.tensor(float(audio_time), device=self.device)
                audio_rows = audio_t * audio_clean + (1.0 - audio_t) * audio_noise
                prediction_rows = self._forward(
                    video_rows=torch.cat((anchor_rows, patchify_video(current)), dim=1),
                    audio_rows=audio_rows,
                    prompt=prompt,
                    layout=layout,
                    video_t=video_t,
                    audio_t=audio_t,
                    viewmats=viewmats,
                    K=K,
                    frame_ids=frame_ids,
                )
                velocity = unpatchify_video(prediction_rows, 47, 48, 84, channels=24).float()
                if tuple(velocity.shape) != tuple(current.shape):
                    raise RuntimeError(
                        f"H3 inference velocity {tuple(velocity.shape)} != "
                        f"sample {tuple(current.shape)}"
                    )
                if not bool(torch.isfinite(velocity).all().item()):
                    raise FloatingPointError("H3 Stage0.5 inference predicted non-finite velocity")
                sigma = float(video_schedule.sigmas[index])
                sigma_next = float(video_schedule.sigmas[index + 1])
                if sigma == 0:
                    raise RuntimeError("H3 Euler schedule reached sigma zero early")
                denoised = current + sigma * velocity
                ratio = sigma_next / sigma
                current = ratio * current + (1.0 - ratio) * denoised
        if not bool(torch.isfinite(current).all().item()):
            raise FloatingPointError("H3 inference generated non-finite latents")
        return current.contiguous()


__all__ = ["H3Stage0p5Core", "H3TorchLayout"]
