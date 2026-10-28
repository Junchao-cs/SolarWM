"""SolarRenderer structure-guided distribution matching.

Adapted from SolarRenderer, Copyright 2026 SolarRenderer authors, Apache-2.0.
Modified to use the shared SolarWM H3 runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from solar.backends.minimax_h3.flow import make_shifted_schedule
from solar.backends.minimax_h3.layout import patchify_video, unpatchify_video
from solar.backends.minimax_h3.torch_flow import shift_sigma

from .core import H3Stage0p5Core, _Prepared


@dataclass
class H3DMDInputs:
    prepared: _Prepared
    audio_clean: Any
    frames: int
    grid: tuple[int, int]


def _is_structure(batch: Any, item: Any) -> bool:
    """Whether an observation covers every target latent frame."""
    frames = int(batch.target_latents.shape[1])
    return (
        item.frames is not None
        and int(item.latents.shape[1]) == len(item.frames)
        and (set(range(frames)) <= {int(f) for f in item.frames})
    )


def structure_guided(real: Any, free: Any, guidance: float) -> Any:
    """Apply structure guidance to the teacher's clean prediction."""
    return real + (float(guidance) - 1.0) * (real - free)


class H3OmniDMDCore(H3Stage0p5Core):
    """Stage0.5 omni conditions with a few-step student rollout and score evaluations."""

    def __init__(
        self, model: Any, silence_latents: Any, device: Any, train: Any, *, observations: Any = None
    ) -> None:
        super().__init__(model, silence_latents, device, observations=observations)
        self.train = train

    def prepare_inputs(self, batch: Any, generator: Any = None) -> H3DMDInputs:
        prepared = self.prepare(
            batch,
            generator,
            lambda draw: self.shifted_timestep(draw, shift=12.0, device=self.device),
        )
        plan = prepared.plan
        if plan is not None and plan.pattern != "shared":
            raise ValueError("DMD distillation needs the shared observation pattern alone")
        frames = self.latent_frames(batch)
        return H3DMDInputs(prepared, self.audio_rows(frames), frames, self.latent_grid(batch))

    def _random(self, shape: tuple[int, ...], generator: Any) -> Any:
        import torch

        value = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)
        self._broadcast_logical_sample(value)
        return value

    def _video_forward(
        self,
        model: Any,
        inputs: H3DMDInputs,
        sample: Any,
        video_t: Any,
        audio_t: Any,
        audio_noise: Any,
    ) -> Any:
        import torch

        p = inputs.prepared
        rows = self._forward(
            video_rows=torch.cat((*p.condition_rows, patchify_video(sample)), dim=1),
            audio_rows=audio_t * inputs.audio_clean + (1.0 - audio_t) * audio_noise,
            prompt=p.prompt,
            layout=p.layout,
            video_t=video_t,
            audio_t=audio_t,
            viewmats=p.viewmats,
            K=p.K,
            frame_ids=p.frame_ids,
            observation_degrees=p.observation_degrees,
            model=model,
        )
        return unpatchify_video(rows, inputs.frames, *inputs.grid, channels=24).float()

    def rollout(
        self,
        inputs: H3DMDInputs,
        *,
        model: Any,
        forwards: int,
        exit_index: int,
        generator: Any = None,
    ) -> Any:
        """The student's clean estimate at evaluation ``exit_index`` (0-based), with its graph.

        Evaluations before the exit run without a graph; between evaluations the clean estimate is
        noised afresh to the next level, as the four-step LoRA samples (``generate(renoise=True)``).
        """
        import torch

        if not 0 <= int(exit_index) < int(forwards):
            raise ValueError(f"exit_index {exit_index} outside {forwards} evaluations")
        video = make_shifted_schedule(int(forwards) + 1, shift=12.0)
        audio = make_shifted_schedule(int(forwards) + 1, shift=3.0)
        if len(video.timesteps) != int(forwards) or len(audio.timesteps) != int(forwards):
            raise RuntimeError("the shifted grid collapsed points; too many evaluations")
        shape = (1, 24, inputs.frames, *inputs.grid)
        audio_noise = self._random(tuple(inputs.audio_clean.shape), generator)
        current = self._random(shape, generator)
        for index in range(int(exit_index) + 1):
            video_t = torch.tensor(float(video.timesteps[index]), device=self.device)
            audio_t = torch.tensor(float(audio.timesteps[index]), device=self.device)
            sigma = float(video.sigmas[index])
            with torch.set_grad_enabled(index == int(exit_index) and torch.is_grad_enabled()):
                velocity = self._video_forward(
                    model, inputs, current, video_t, audio_t, audio_noise
                )
                denoised = current + sigma * velocity
            if index == int(exit_index):
                return denoised
            sigma_next = float(video.sigmas[index + 1])
            current = (1.0 - sigma_next) * denoised.detach() + sigma_next * self._random(
                shape, generator
            )
        raise AssertionError("unreachable")

    def score_inputs(self, clean: Any, generator: Any = None) -> tuple[Any, Any, Any, Any]:
        """``clean`` noised at one level: uniform, H3-shifted (video 12 / audio 3), clamped."""
        import torch

        raw = torch.rand((), generator=generator, device=self.device, dtype=torch.float32)
        self._broadcast_logical_sample(raw)
        shift = float(self.train.get("score_shift", 12.0))
        sigma = shift_sigma(raw, shift).clamp(0.02, 0.98)
        audio_t = 1.0 - shift_sigma(raw, 3.0).clamp(0.02, 0.98)
        noise = self._random(tuple(clean.shape), generator)
        noisy = (1.0 - sigma) * clean + sigma * noise
        return (noisy, noise, 1.0 - sigma, audio_t)

    def score_forward(
        self,
        model: Any,
        inputs: H3DMDInputs,
        noisy: Any,
        video_t: Any,
        audio_t: Any,
        audio_noise: Any = None,
    ) -> Any:
        """A score model's data-ward velocity at ``noisy`` (the H3 sign: clean - noise)."""
        if audio_noise is None:
            audio_noise = self._random(tuple(inputs.audio_clean.shape), None)
        return self._video_forward(
            model, inputs, noisy, video_t.reshape(()), audio_t.reshape(()), audio_noise
        )

    def structure_free(self, inputs: H3DMDInputs, batch: Any) -> H3DMDInputs:
        """Remove structure while preserving every retained condition's original noise."""
        import dataclasses

        import torch

        from solar.backends.minimax_h3.layout import H3VisualObservationSpec

        prepared = inputs.prepared
        items = tuple(batch.visual_observations or ())
        planned = None if prepared.plan is None else prepared.plan.keeps_observation
        selected = [item for index, item in enumerate(items) if planned is None or planned(index)]
        use_anchor = batch.anchor_latents is not None and (
            prepared.plan is None or prepared.plan.use_anchor
        )
        parts = list(prepared.condition_rows[:1]) if use_anchor else []
        observation_parts = prepared.condition_rows[1 if use_anchor else 0 :]
        original_rows = torch.cat(observation_parts, dim=1) if observation_parts else None
        counts = prepared.layout.source.observation_rows
        original_degrees = prepared.observation_degrees or ()
        specs, degrees, retained = [], [], []
        offset = 0
        for item, count, degree in zip(selected, counts, original_degrees, strict=True):
            if not _is_structure(batch, item):
                specs.append(
                    H3VisualObservationSpec(
                        num_frames=int(item.latents.shape[1]),
                        height=int(item.latents.shape[2]),
                        width=int(item.latents.shape[3]),
                        frames=None if item.frames is None else tuple(int(f) for f in item.frames),
                        boxes=None
                        if item.boxes is None
                        else tuple(tuple(float(value) for value in box) for box in item.boxes),
                    )
                )
                degrees.append(degree)
                retained.append(original_rows[:, offset : offset + count])
            offset += count
        if retained:
            parts.append(torch.cat(retained, dim=1))
        _prompt, tags = self.prompt(batch, use_anchor=use_anchor)
        layout = self.layout(
            tags,
            anchors=("first",) if use_anchor else (),
            observations=tuple(specs),
            grid=inputs.grid,
            frames=inputs.frames,
        )
        return dataclasses.replace(
            inputs,
            prepared=dataclasses.replace(
                prepared,
                condition_rows=parts,
                layout=layout,
                observation_degrees=tuple(degrees) if degrees else None,
            ),
        )

    def teacher_answer(
        self, teacher, inputs, batch, noisy, video_t, audio_t, audio_noise, *, guidance=1.0
    ):
        """Evaluate the fixed teacher with optional structure guidance."""
        sigma = 1.0 - video_t
        real = noisy + sigma * self.score_forward(
            teacher, inputs, noisy, video_t, audio_t, audio_noise
        )
        if float(guidance) == 1.0:
            return real
        free_inputs = self.structure_free(inputs, batch)
        free = noisy + sigma * self.score_forward(
            teacher, free_inputs, noisy, video_t, audio_t, audio_noise
        )
        return structure_guided(real, free, guidance)
