"""Teacher-forced W6 AnyFlow v1.5 with H3's native data-ward velocity."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from solarwm.training.anyflow import apply_timestep_shift, sample_anyflow_time_pairs

from .anyflow_loss import h3_anyflow_v15_loss
from .distributed import get_dp_group, get_dp_rank, get_dp_world_size
from .layout import build_stage1_layout, patchify_video
from .stage0p5 import H3Stage0p5Core
from .torch_layout import build_stage1_row_time_pairs
from .torch_mask import build_stage1_w6_block_mask


class H3Stage1Core(H3Stage0p5Core):
    """Evaluate detached targets before the trainable AnyFlow prediction."""

    def __init__(
        self,
        model: Any,
        silence_latents: Any,
        device: Any,
        train: Any,
        validation_silence: Any = None,
    ) -> None:
        super().__init__(model, silence_latents, device)
        self.train = train
        self.validation_silence = validation_silence
        self._masks: OrderedDict[tuple[int, int, int], Any] = OrderedDict()

    def layout(self, tags: Any) -> Any:
        key = tuple(tags.detach().cpu().tolist())
        if key not in self._layouts:
            self._layouts[key] = build_stage1_layout(
                key,
                45,
                48,
                84,
                self.silence.shape[-1],
            ).to(self.device)
            while len(self._layouts) > 4:
                self._layouts.pop(next(iter(self._layouts)))
        return self._layouts[key]

    def _camera(self, batch: Any, layout: Any) -> tuple[Any, Any, Any]:
        import torch

        views = batch.camera_viewmats.to(self.device, torch.float32).unsqueeze(0)
        intrinsics = batch.camera_K.to(self.device, torch.float32).unsqueeze(0)
        if views.shape[1] == 47:
            return views[:, :45], intrinsics[:, :45], layout.camera_frame_ids
        if views.shape[1] != 47 * 1008:
            raise ValueError("Stage1 camera must describe the complete 47-latent artifact")
        views, intrinsics = views[:, : 45 * 1008], intrinsics[:, : 45 * 1008]
        return (
            torch.cat((views[:, :1008], views, views), dim=1),
            torch.cat((intrinsics[:, :1008], intrinsics, intrinsics), dim=1),
            None,
        )

    def forward_loss(self, batch: Any, *, noise_seed: int | None) -> Any:
        import torch

        generator = (
            None
            if noise_seed is None
            else torch.Generator(device=self.device).manual_seed(int(noise_seed))
        )
        clean = batch.target_latents.to(self.device, torch.float32).unsqueeze(0)
        anchor = batch.anchor_latents.to(self.device, torch.float32).unsqueeze(0)
        prompt = batch.prompt_embeds.to(self.device, torch.bfloat16).unsqueeze(0)
        tags = batch.text_token_tags.to(self.device, torch.long)
        self._broadcast_logical_sample(tags, clean, anchor, prompt)
        clean = clean[:, :, :45].contiguous()
        layout = self.layout(tags)
        views, intrinsics, frame_ids = self._camera(batch, layout)
        self._broadcast_logical_sample(views, intrinsics)

        def noise_like(value: Any) -> Any:
            return torch.randn(
                value.shape,
                device=self.device,
                dtype=torch.float32,
                generator=generator,
            )

        anchor_noise = noise_like(anchor)
        self._broadcast_logical_sample(anchor_noise)
        anchor_t = torch.tensor(0.999, device=self.device, dtype=anchor.dtype)
        anchor_rows = patchify_video(anchor_t * anchor + (1.0 - anchor_t) * anchor_noise)
        audio_clean = self.audio_rows()
        audio_noise = noise_like(audio_clean)
        audio_t = self.shifted_timestep(generator, shift=3.0, device=self.device)
        self._broadcast_logical_sample(audio_noise, audio_t)
        audio = audio_t * audio_clean + (1.0 - audio_t) * audio_noise
        noise = noise_like(clean)
        self._broadcast_logical_sample(noise)
        pairs = sample_anyflow_time_pairs(
            1,
            logical_dp_rank=get_dp_rank(),
            logical_dp_world_size=get_dp_world_size(),
            diffusion_ratio=float(self.train["diffusion_ratio"]),
            consistency_ratio=float(self.train["consistency_ratio"]),
            generator=generator,
            device=self.device,
        )
        self._broadcast_logical_sample(*pairs)
        sigma_t = apply_timestep_shift(pairs.t, 12.0)
        sigma_r = apply_timestep_shift(pairs.r, 12.0)
        clean_rows, noise_rows = patchify_video(clean), patchify_video(noise)
        key = (layout.sequence_length, layout.rows_per_video_frame, layout.window_chunks)
        if key not in self._masks:
            self._masks[key] = build_stage1_w6_block_mask(
                layout,
                batch_size=1,
                num_heads=None,
                device=self.device,
            )
            while len(self._masks) > 4:
                self._masks.popitem(last=False)
        self._masks.move_to_end(key)
        mask = self._masks[key]

        def velocity(noisy: Any, video_t: Any, video_r: Any) -> Any:
            times, targets, indices = build_stage1_row_time_pairs(
                layout,
                video_t,
                video_r,
                audio_t,
                keyframe_timestep=0.999,
                device=self.device,
            )
            prediction, _ = self.model(
                hidden_states=torch.cat((anchor_rows, clean_rows, noisy), dim=1),
                audio_hidden_states=audio,
                encoder_hidden_states=prompt,
                timestep=times,
                r_timestep=targets,
                timestep_indices=indices,
                attention_mask=mask,
                fused_prope=True,
                prope_token_indices=layout.camera_video_indices,
                prope_frame_ids=frame_ids,
                cam_viewmats=views,
                cam_K=intrinsics,
                packed_sequence_parallel=self._sp_enabled(),
                return_dict=False,
                **layout.transformer_kwargs(),
            )
            return prediction[:, layout.noisy_video_output_slice]

        return h3_anyflow_v15_loss(
            velocity,
            clean_rows,
            noise_rows,
            sigma_t,
            sigma_r,
            pairs.is_diffusion,
            epsilon=float(self.train["finite_difference_epsilon"]),
            shift=12.0,
            process_group=get_dp_group(),
        )

    @staticmethod
    def _sp_enabled() -> bool:
        from .distributed import is_sequence_parallel_enabled

        return is_sequence_parallel_enabled()

    def generate(self, batch: Any, *, noise_seed: int, num_inference_steps: int) -> Any:
        from .stage1_sampling import generate_stage1

        if num_inference_steps != 4 or self.validation_silence is None:
            raise ValueError("H3 Stage1 validation needs NFE4 and the encoded 170-frame silence")
        return generate_stage1(
            self.model, batch, self.validation_silence, device=self.device, seed=noise_seed
        )
