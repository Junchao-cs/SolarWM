"""Exact Stage0.5 data-ward flow-matching forward and Euler sampler."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from solarwm.backends.minimax_h3.artifacts import H3ArtifactBatch
from solarwm.backends.minimax_h3.distributed import (
    broadcast_sp_tensor,
    is_sequence_parallel_enabled,
)
from solarwm.backends.minimax_h3.flow import explicit_schedule, make_shifted_schedule
from solarwm.backends.minimax_h3.layout import (
    H3VisualObservationSpec,
    build_row_timesteps,
    build_stage0p5_layout,
    patchify_video,
    unpatchify_video,
)
from solarwm.backends.minimax_h3.stage0p5 import H3Stage0p5Core as _CameraCore
from solarwm.backends.minimax_h3.stage0p5 import H3TorchLayout


def _keep_none(index: int) -> bool:
    return False


@dataclass(frozen=True)
class _Prepared:
    """One training sample's conditioning, drawn by :meth:`H3Stage0p5Core.prepare`."""

    clean: Any
    video_noise: Any
    video_t: Any
    audio_t: Any
    audio_rows: Any
    condition_rows: list
    prompt: Any
    layout: Any
    viewmats: Any
    K: Any
    frame_ids: Any
    observation_degrees: Any
    plan: Any


class H3Stage0p5Core(_CameraCore):
    """One model/silence/device implementation shared by train and inference."""

    def __init__(self, model: Any, silence_latents: Any, device: Any, *, observations: Any = None):
        super().__init__(model, silence_latents, device)
        self.observations = observations

    def layout(
        self,
        tags: Any,
        *,
        anchors: tuple[str, ...] = ("first",),
        observations: tuple[H3VisualObservationSpec, ...] = (),
        grid: tuple[int, int] = (48, 84),
        frames: int = 47,
    ) -> H3TorchLayout:
        import torch

        tags_cpu = tags.detach().cpu().long().numpy()
        key = hashlib.blake2s(
            tags_cpu.tobytes() + f"|{anchors}|{observations}|{grid}|{frames}".encode()
        ).hexdigest()
        cached = self._layouts.get(key)
        if cached is not None:
            return cached
        source = build_stage0p5_layout(
            tags_cpu,
            frames,
            *grid,
            self.audio_latents(frames),
            keyframe_anchors=anchors,
            observations=observations,
        )
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

    def observation_rows(self, batch: H3ArtifactBatch, generator: Any, keep: Any = None) -> Any:
        """``(specs, degrees, [1,N,96] rows)`` of the batch's visual observations.

        Each is noised at its own known degree, like the first-frame anchor. ``keep``
        decides per index whether a sample uses it. The draws come after every other
        draw of the sample, so a batch without observations consumes the same stream.
        """
        import torch

        specs, degrees, rows = ([], [], [])
        for index, item in enumerate(batch.visual_observations or ()):
            if keep is not None and (not keep(index)):
                continue
            clean = item.latents.to(self.device, dtype=torch.float32, non_blocking=True)
            clean = clean.unsqueeze(0)
            if clean.ndim != 5 or clean.shape[1] != 24:
                raise ValueError("H3 visual observations must be [24,f,h,w] latents")
            if not 0.0 < float(item.degree) <= 1.0:
                raise ValueError("a visual observation's degree must lie in (0, 1]")
            noise = torch.randn(
                clean.shape, generator=generator, device=self.device, dtype=torch.float32
            )
            self._broadcast_logical_sample(clean, noise)
            degree = float(item.degree)
            rows.append(patchify_video(degree * clean + (1.0 - degree) * noise))
            degrees.append(degree)
            specs.append(
                H3VisualObservationSpec(
                    num_frames=int(clean.shape[2]),
                    height=int(clean.shape[3]),
                    width=int(clean.shape[4]),
                    frames=None if item.frames is None else tuple(int(f) for f in item.frames),
                    boxes=None
                    if item.boxes is None
                    else tuple(tuple(float(v) for v in box) for box in item.boxes),
                )
            )
        if not rows:
            return ((), None, None)
        return (tuple(specs), tuple(degrees), torch.cat(rows, dim=1))

    @staticmethod
    def latent_grid(batch: H3ArtifactBatch) -> tuple[int, int]:
        """Landscape ``(48, 84)`` or portrait ``(84, 48)``, read from the sample's latents."""
        source = batch.target_latents if batch.anchor_latents is None else batch.anchor_latents
        grid = (int(source.shape[-2]), int(source.shape[-1]))
        if grid not in {(48, 84), (84, 48)}:
            raise ValueError(f"H3 latents must be 48x84 or 84x48, got {grid}")
        return grid

    @staticmethod
    def latent_frames(batch: H3ArtifactBatch) -> int:
        """Latent frames of the clip: 47 for the 158-frame release, ``5n+2`` in general."""
        frames = 47 if batch.target_latents is None else int(batch.target_latents.shape[1])
        if frames < 2 or (frames - 2) % 5:
            raise ValueError(f"H3 clips hold 5n+2 latent frames, got {frames}")
        return frames

    @staticmethod
    def audio_latents(frames: int) -> int:
        """Audio latents covering the clip: 40 per second of its ``17n+5`` frames at 24 fps."""
        return round((17 * ((frames - 2) // 5) + 5) / 24.0 * 40)

    def prompt(self, batch: H3ArtifactBatch, *, use_anchor: bool) -> tuple[Any, Any]:
        """Select embeddings for the explicit anchor setting.

        Reference observations are encoded jointly with the Ref2VA prompt. Removing their
        packed latent rows does not remove their vision features from these embeddings.
        """
        import torch

        embeds, tags = (batch.prompt_embeds, batch.text_token_tags)
        if not use_anchor:
            if batch.text_prompt_embeds is None or batch.text_prompt_tags is None:
                raise ValueError(
                    "a sample without its first-frame anchor needs text-only prompt embeds: "
                    "the joint prompt was encoded with that frame"
                )
            embeds, tags = (batch.text_prompt_embeds, batch.text_prompt_tags)
        embeds = embeds.to(self.device, dtype=torch.bfloat16, non_blocking=True).unsqueeze(0)
        tags = tags.to(self.device, dtype=torch.long)
        self._broadcast_logical_sample(tags, embeds)
        return (embeds, tags)

    def audio_rows(self, frames: int | None = None) -> Any:
        """Silence rows; ``frames`` trims them to a clip shorter than the silence artifact."""
        audio = self.silence.to(
            device=self.device, dtype=__import__("torch").float32, non_blocking=True
        ).unsqueeze(0)
        if frames is not None:
            audio = audio[..., : self.audio_latents(frames)]
        return audio.permute(0, 1, 3, 2).reshape(1, -1, 32).contiguous()

    def _camera(self, batch: H3ArtifactBatch, layout: H3TorchLayout) -> tuple[Any, Any, Any]:
        import torch

        views = batch.camera_viewmats.to(
            self.device, dtype=torch.float32, non_blocking=True
        ).unsqueeze(0)
        K = batch.camera_K.to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(0)
        frames, grid = (self.latent_frames(batch), self.latent_grid(batch))
        rows_per_latent = grid[0] // 2 * (grid[1] // 2)
        if views.shape[1] == frames:
            return (views, K, layout.camera_frame_ids)
        if views.shape[1] == frames * rows_per_latent:
            if layout.source.num_observation_rows:
                raise RuntimeError("token-aligned cameras do not cover visual observations")
            if not layout.source.num_condition_video_rows:
                return (views, K, None)
            anchor_views = views[:, :rows_per_latent]
            anchor_K = K[:, :rows_per_latent]
            return (torch.cat((anchor_views, views), dim=1), torch.cat((anchor_K, K), dim=1), None)
        raise RuntimeError("H3 camera artifact is neither latent nor token aligned")

    @staticmethod
    def _row_times(
        layout: H3TorchLayout,
        video_t: Any,
        audio_t: Any,
        device: Any,
        text_t: Any = None,
        observation_degrees: Any = None,
    ) -> tuple[Any, Any]:
        """``video_t`` is a scalar, or a known-degree field together with its ``text_t``.

        Returns the unique times and the per-row index into them.
        """
        import numpy as np
        import torch

        field = text_t is not None
        times, inverse = build_row_timesteps(
            layout.source,
            video_t.detach().float().cpu().numpy() if field else float(video_t.item()),
            float(audio_t.item()),
            text_timestep=float(text_t.item()) if field else None,
            condition_video_timestep=0.999,
            observation_timesteps=observation_degrees,
        )
        return (
            torch.from_numpy(np.ascontiguousarray(times)).to(device=device, dtype=torch.float32),
            torch.from_numpy(np.asarray(inverse).reshape(-1)).to(device=device, dtype=torch.long),
        )

    def _forward(
        self,
        *,
        video_rows: Any = None,
        audio_rows: Any = None,
        prompt: Any = None,
        layout: H3TorchLayout = None,
        video_t: Any = None,
        audio_t: Any = None,
        viewmats: Any = None,
        K: Any = None,
        frame_ids: Any = None,
        text_t: Any = None,
        observation_degrees: Any = None,
        model: Any = None,
    ) -> Any:
        """``viewmats=None`` runs without a camera, on the backbone's native attention.

        ``model`` evaluates another copy on the same inputs (a DMD teacher or critic).
        """
        times, inverse = self._row_times(
            layout, video_t, audio_t, self.device, text_t, observation_degrees
        )
        camera = viewmats is not None
        video, _audio = (self.model if model is None else model)(
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
            fused_prope=camera,
            prope_token_indices=layout.camera_video_indices if camera else None,
            prope_frame_ids=frame_ids if camera else None,
            cam_viewmats=viewmats,
            cam_K=K if camera else None,
            stage0p5_sequence_parallel=is_sequence_parallel_enabled(),
            return_dict=False,
        )
        return video[:, layout.target_video_output_slice]

    def observation_plan(
        self,
        generator: Any,
        video_t: Any,
        known_degrees: Any = None,
        grid: tuple[int, int] = (48, 84),
        frames: int = 47,
    ) -> Any:
        """Draw this sample's plan after the ordinary noise draws; ``None`` without a profile."""
        if self.observations is None:
            return None
        import numpy as np
        import torch

        from .observations import sample_observation_plan

        seed = torch.randint(
            0, 2**31 - 1, (1,), generator=generator, device=self.device, dtype=torch.int64
        )
        self._broadcast_logical_sample(seed)
        given = None
        if known_degrees is not None:
            given = known_degrees.to(self.device, dtype=torch.float32).reshape(-1)
            if given.numel() != frames * 1008:
                raise ValueError("a batch's known_degrees must cover every target row")
            self._broadcast_logical_sample(given)
            given = given.cpu().numpy()
        return sample_observation_plan(
            self.observations,
            frames=frames,
            grid_height=grid[0] // 2,
            grid_width=grid[1] // 2,
            base_time=float(video_t.item()),
            shift=12.0,
            rng=np.random.default_rng(int(seed.item())),
            given=given,
        )

    def prepare(self, batch: H3ArtifactBatch, generator: Any, draw_time: Any) -> _Prepared:
        """Everything one training forward conditions on, drawn in the fixed order.

        ``draw_time(generator)`` gives the sample's data-ward target time; it is called where the
        ordinary recipe draws it, so every other draw lands exactly where it always has.
        """
        import torch

        clean = batch.target_latents.to(
            self.device, dtype=torch.float32, non_blocking=True
        ).unsqueeze(0)
        frames = self.latent_frames(batch)
        has_anchor, has_camera = (
            batch.anchor_latents is not None,
            batch.camera_viewmats is not None,
        )
        anchor = (
            batch.anchor_latents.to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(
                0
            )
            if has_anchor
            else clean[:, :, :1]
        )
        anchor_noise = torch.randn(
            anchor.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        audio_clean = self.audio_rows(frames)
        audio_noise = torch.randn(
            audio_clean.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        audio_t = self.shifted_timestep(generator, shift=3.0, device=self.device)
        video_noise = torch.randn(
            clean.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        video_t = draw_time(generator)
        if is_sequence_parallel_enabled():
            for value in (anchor_noise, audio_noise, video_noise, video_t, audio_t):
                broadcast_sp_tensor(value)
        grid = self.latent_grid(batch)
        plan = self.observation_plan(generator, video_t, batch.known_degrees, grid, frames)
        use_anchor = has_anchor and (plan is None or plan.use_anchor)
        prompt, tags = self.prompt(batch, use_anchor=use_anchor)
        keep = None if plan is None else plan.keeps_observation
        specs, degrees, observation_rows = self.observation_rows(batch, generator, keep)
        layout = self.layout(
            tags,
            anchors=("first",) if use_anchor else (),
            observations=specs,
            grid=grid,
            frames=frames,
        )
        viewmats = K = frame_ids = None
        if has_camera:
            viewmats, K, frame_ids = self._camera(batch, layout)
            self._broadcast_logical_sample(viewmats, K)
        self._broadcast_logical_sample(clean, anchor)
        if plan is not None and (not plan.use_camera):
            viewmats = K = frame_ids = None
        anchor_t = torch.tensor(0.999, device=self.device, dtype=anchor.dtype)
        anchor_noisy = anchor_t * anchor + (1.0 - anchor_t) * anchor_noise
        parts = [patchify_video(anchor_noisy)] if use_anchor else []
        if observation_rows is not None:
            parts.append(observation_rows)
        return _Prepared(
            clean=clean,
            video_noise=video_noise,
            video_t=video_t,
            audio_t=audio_t,
            audio_rows=audio_t * audio_clean + (1.0 - audio_t) * audio_noise,
            condition_rows=parts,
            prompt=prompt,
            layout=layout,
            viewmats=viewmats,
            K=K,
            frame_ids=frame_ids,
            observation_degrees=degrees,
            plan=plan,
        )

    def forward_loss(self, batch: H3ArtifactBatch, *, noise_seed: int | None) -> Any:
        import torch

        generator = (
            None
            if noise_seed is None
            else torch.Generator(device=self.device).manual_seed(int(noise_seed))
        )
        inputs = self.prepare(
            batch,
            generator,
            lambda draw: self.shifted_timestep(draw, shift=12.0, device=self.device),
        )
        return self.loss_from_inputs(inputs)

    def loss_from_inputs(self, inputs: _Prepared, model: Any = None) -> Any:
        """The flow-matching loss of prepared inputs, on ``model`` (default: this core's)."""
        import torch
        import torch.nn.functional as F

        clean, video_noise, video_t, plan = (
            inputs.clean,
            inputs.video_noise,
            inputs.video_t,
            inputs.plan,
        )
        target_rows = patchify_video(clean - video_noise)
        text_t = row_t = None
        if plan is None:
            noisy_rows = patchify_video(video_t * clean + (1.0 - video_t) * video_noise)
        else:
            row_t = torch.from_numpy(plan.row_times).to(self.device)
            text_t = torch.tensor(plan.base_time, device=self.device)
            degree = row_t.view(1, -1, 1)
            noisy_rows = degree * patchify_video(clean) + (1.0 - degree) * patchify_video(
                video_noise
            )
        prediction = self._forward(
            video_rows=torch.cat((*inputs.condition_rows, noisy_rows), dim=1),
            audio_rows=inputs.audio_rows,
            prompt=inputs.prompt,
            layout=inputs.layout,
            video_t=video_t if plan is None else row_t,
            audio_t=inputs.audio_t,
            viewmats=inputs.viewmats,
            K=inputs.K,
            frame_ids=inputs.frame_ids,
            text_t=text_t,
            observation_degrees=inputs.observation_degrees,
            model=model,
        )
        if tuple(prediction.shape) != tuple(target_rows.shape):
            raise RuntimeError(
                f"H3 predicted rows {tuple(prediction.shape)} != {tuple(target_rows.shape)}"
            )
        if plan is None:
            return F.mse_loss(prediction.float(), target_rows.float())
        weight = torch.from_numpy(plan.loss_rows).to(self.device).view(1, -1, 1).float()
        error = (prediction.float() - target_rows.float()).square() * weight
        return error.sum() / (weight.sum() * prediction.shape[-1])

    def generate(
        self,
        batch: H3ArtifactBatch,
        *,
        noise_seed: int,
        num_inference_steps: int,
        observations: Any = None,
        renoise: bool = False,
        structure_until: float = 1.0,
        video_sigmas: Any = None,
        audio_sigmas: Any = None,
    ) -> Any:
        """Euler sampling; ``observations`` (H3InferenceObservations) makes every input optional.

        ``renoise`` samples as a DMD few-step student was trained to: each step takes the clean
        estimate and noises it afresh to the next time, rather than carrying the old noise on.

        ``video_sigmas`` (and ``audio_sigmas``) replace the shifted grid by a released adapter's own
        sigma list; ``num_inference_steps`` then counts its points.

        ``structure_until`` < 1 keeps the visual observations only for that leading fraction of
        the steps and samples the rest without them, exactly as a training sample whose
        observations were dropped: no rows, no layout entry. The early steps fix layout, motion
        and camera from the structure; the late ones, which shape detail, follow the prompt.
        """
        import torch

        if video_sigmas is not None:
            num_inference_steps = len(video_sigmas)
        generator = torch.Generator(device=self.device).manual_seed(int(noise_seed))
        frames, grid = (self.latent_frames(batch), self.latent_grid(batch))
        has_anchor, has_camera = (
            batch.anchor_latents is not None,
            batch.camera_viewmats is not None,
        )
        anchor = (
            batch.anchor_latents.to(self.device, dtype=torch.float32, non_blocking=True).unsqueeze(
                0
            )
            if has_anchor
            else torch.zeros((1, 24, 1, *grid), device=self.device)
        )
        use_anchor = has_anchor and (observations is None or observations.use_anchor)
        prompt, tags = self.prompt(batch, use_anchor=use_anchor)
        anchor_noise = torch.randn(
            anchor.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        audio_clean = self.audio_rows(frames)
        audio_noise = torch.randn(
            audio_clean.shape, generator=generator, device=self.device, dtype=torch.float32
        )
        current = torch.randn(
            (1, 24, frames, *grid), generator=generator, device=self.device, dtype=torch.float32
        )
        if is_sequence_parallel_enabled():
            for value in (anchor_noise, audio_noise, current):
                broadcast_sp_tensor(value)
        use_items = observations is None or observations.use_visual_observations
        keep = _keep_none if not use_items else None
        specs, degrees, observation_rows = self.observation_rows(batch, generator, keep)
        layout = self.layout(
            tags,
            anchors=("first",) if use_anchor else (),
            observations=specs,
            grid=grid,
            frames=frames,
        )
        viewmats = K = frame_ids = None
        if has_camera:
            viewmats, K, frame_ids = self._camera(batch, layout)
            self._broadcast_logical_sample(viewmats, K)
        self._broadcast_logical_sample(anchor)
        if observations is not None and (not observations.use_camera):
            viewmats = K = frame_ids = None
        anchor_rows = patchify_video(0.999 * anchor + 0.001 * anchor_noise)
        condition_rows = [anchor_rows] if use_anchor else []
        if observation_rows is not None:
            condition_rows.append(observation_rows)
        release_at = num_inference_steps
        free = None
        if structure_until < 1.0 and observation_rows is not None:
            if has_camera:
                raise ValueError("releasing the structure is defined for samples with no camera")
            release_at = max(0, round(float(structure_until) * num_inference_steps))
            free_specs, free_degrees, _ = self.observation_rows(batch, generator, _keep_none)
            free_layout = self.layout(
                tags,
                anchors=("first",) if use_anchor else (),
                observations=free_specs,
                grid=grid,
                frames=frames,
            )
            free = ([anchor_rows] if use_anchor else [], free_layout, free_degrees)
        fixed = held = degree = start = None
        if observations is not None and observations.latents is not None:
            rows = frames * (grid[0] // 2) * (grid[1] // 2)
            given = observations.latents.to(self.device, dtype=torch.float32).unsqueeze(0)
            if tuple(given.shape) != tuple(current.shape):
                raise ValueError("in-place observations must cover the target latent grid")

            def field(value: Any) -> Any:
                if value is None:
                    return torch.zeros(rows, device=self.device)
                result = torch.as_tensor(value, dtype=torch.float32).to(self.device).reshape(-1)
                if result.numel() != rows or bool(((result < 0) | (result > 1)).any().item()):
                    raise ValueError("known degrees cover every target row and lie in [0, 1]")
                return result

            degree, start = (field(observations.row_degrees), field(observations.start_degrees))
            self._broadcast_logical_sample(degree, start, given)
            if bool((degree > 0).all().item()) or bool(((degree > 0) & (start > 0)).any().item()):
                raise ValueError("fixed rows must leave rows to generate and cannot also start")
            fixed = (degree > 0).view(1, -1, 1)
            level = torch.maximum(degree, start).view(1, -1, 1)
            entered = level * patchify_video(given) + (1.0 - level) * patchify_video(current)
            held = entered
            current = unpatchify_video(
                torch.where(level > 0, entered, patchify_video(current)), frames, *grid, channels=24
            )
        if free is not None and fixed is not None:
            raise ValueError("releasing the structure is defined for samples with no in-place rows")
        if renoise and fixed is not None:
            raise ValueError("fresh-noise sampling moves every target row along the plain schedule")
        if video_sigmas is not None:
            video_schedule = explicit_schedule(video_sigmas)
            audio_schedule = (
                explicit_schedule(audio_sigmas)
                if audio_sigmas is not None
                else make_shifted_schedule(len(video_schedule.sigmas), shift=3.0)
            )
        else:
            video_schedule = make_shifted_schedule(num_inference_steps, shift=12.0)
            audio_schedule = make_shifted_schedule(num_inference_steps, shift=3.0)
        with torch.no_grad():
            for index, (video_time, audio_time) in enumerate(
                zip(video_schedule.timesteps, audio_schedule.timesteps, strict=True)
            ):
                video_t = torch.tensor(float(video_time), device=self.device)
                audio_t = torch.tensor(float(audio_time), device=self.device)
                audio_rows = audio_t * audio_clean + (1.0 - audio_t) * audio_noise
                target_rows = patchify_video(current)
                row_t = text_t = None
                if fixed is not None:
                    target_rows = torch.where(fixed, held, target_rows)
                    row_t = torch.where(degree > 0, degree, torch.clamp(start, min=video_t))
                    text_t = video_t
                step_rows, step_layout, step_degrees = (condition_rows, layout, degrees)
                if free is not None and index >= release_at:
                    step_rows, step_layout, step_degrees = free
                prediction_rows = self._forward(
                    video_rows=torch.cat((*step_rows, target_rows), dim=1),
                    audio_rows=audio_rows,
                    prompt=prompt,
                    layout=step_layout,
                    video_t=video_t if row_t is None else row_t,
                    audio_t=audio_t,
                    viewmats=viewmats,
                    K=K,
                    frame_ids=frame_ids,
                    text_t=text_t,
                    observation_degrees=step_degrees,
                )
                velocity = unpatchify_video(prediction_rows, frames, *grid, channels=24).float()
                if tuple(velocity.shape) != tuple(current.shape):
                    raise RuntimeError(
                        f"H3 inference velocity {tuple(velocity.shape)} "
                        f"!= sample {tuple(current.shape)}"
                    )
                if not bool(torch.isfinite(velocity).all().item()):
                    raise FloatingPointError("H3 Stage0.5 inference predicted non-finite velocity")
                sigma = float(video_schedule.sigmas[index])
                sigma_next = float(video_schedule.sigmas[index + 1])
                if sigma == 0:
                    raise RuntimeError("H3 Euler schedule reached sigma zero early")
                if fixed is None and renoise:
                    denoised = current + sigma * velocity
                    fresh = torch.randn(
                        current.shape, generator=generator, device=self.device, dtype=current.dtype
                    )
                    current = (1.0 - sigma_next) * denoised + sigma_next * fresh
                elif fixed is None:
                    denoised = current + sigma * velocity
                    ratio = sigma_next / sigma
                    current = ratio * current + (1.0 - ratio) * denoised
                else:
                    row_sigma = (1.0 - row_t).clamp_min(1e-06).view(1, -1, 1)
                    ratio = (sigma_next / row_sigma).clamp(max=1.0)
                    ratio = torch.where(fixed, torch.ones_like(ratio), ratio)
                    rows_now = target_rows + (1.0 - ratio) * row_sigma * prediction_rows.float()
                    current = unpatchify_video(rows_now, frames, *grid, channels=24)
            if fixed is not None:
                estimate = held + (1.0 - degree.view(1, -1, 1)) * prediction_rows.float()
                current = unpatchify_video(
                    torch.where(fixed, estimate, patchify_video(current)),
                    frames,
                    *grid,
                    channels=24,
                )
        if not bool(torch.isfinite(current).all().item()):
            raise FloatingPointError("H3 inference generated non-finite latents")
        return current.contiguous()
