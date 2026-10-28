"""H3 geometry and native flow math for self-gradient forcing.

The score models see the complete 47-latent physical encode. The causal
student produces ten five-latent chunks, with only the first two latents of
the last chunk supervised. H3's separately encoded image anchor is a
condition: target latent zero remains a generated, supervised video latent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .torch_flow import make_shifted_schedule, predict_clean_sample


@dataclass(frozen=True)
class H3SGFWindow:
    chunk_index: int
    start: int
    current_start: int
    stop: int
    supervised_stop: int

    @property
    def prior_chunks(self) -> tuple[int, ...]:
        return tuple(range(self.start // 5, self.chunk_index))

    @property
    def local_latents(self) -> int:
        return self.stop - self.start

    @property
    def supervised_latents(self) -> int:
        return max(0, self.supervised_stop - self.current_start)


def h3_sgf_windows(
    *,
    rollout_latents: int = 50,
    supervision_latents: int = 47,
    chunk_latents: int = 5,
    window_chunks: int = 6,
) -> tuple[H3SGFWindow, ...]:
    if (rollout_latents, supervision_latents, chunk_latents, window_chunks) != (50, 47, 5, 6):
        raise ValueError("H3 SGF requires 50 rollout / 47 score latents, five-latent chunks and W6")
    return tuple(
        H3SGFWindow(i, max(0, i - 5) * 5, i * 5, (i + 1) * 5, min((i + 1) * 5, 47))
        for i in range(10)
    )


def h3_sgf_schedule(
    *, num_steps: int = 4, video_shift: float = 12.0, audio_shift: float = 3.0, device: Any = None
):
    """Four denoiser evaluations, using H3's video/audio shifted sigma grids.

    H3's upstream API counts grid points; SGF configuration counts evaluations.
    Convert that convention once here for both training and validation.
    """
    if num_steps != 4:
        raise ValueError("H3 SGF requires exactly four denoiser evaluations")
    return (
        make_shifted_schedule(num_steps + 1, shift=video_shift, device=device),
        make_shifted_schedule(num_steps + 1, shift=audio_shift, device=device),
    )


def h3_sgf_score_timesteps(
    *,
    video_shift: float = 12.0,
    audio_shift: float = 3.0,
    min_sigma: float = 0.02,
    max_sigma: float = 0.98,
    device: Any = None,
    generator: Any = None,
):
    """One sampled raw noise level, shifted per H3 modality, in native time."""
    import torch

    from .torch_flow import shift_sigma

    if not 0.0 < min_sigma < max_sigma < 1.0:
        raise ValueError("H3 SGF score sigma bounds must lie strictly within (0,1)")
    raw = torch.randint(0, 1000, (), device=device, generator=generator).float() / 1000.0
    return tuple(
        1.0 - shift_sigma(raw, shift).clamp(min_sigma, max_sigma)
        for shift in (video_shift, audio_shift)
    )


def h3_sgf_critic_loss(velocity, *, noise, clean):
    """H3 predicts data-ward ``clean-noise``, the opposite sign from Wan."""
    import torch.nn.functional as F

    if velocity.shape != noise.shape or velocity.shape != clean.shape:
        raise ValueError("H3 critic prediction, noise and clean shapes must agree")
    return F.mse_loss(velocity.float(), clean.float() - noise.float())


def h3_sgf_clean_prediction(noisy, velocity, timestep):
    return predict_clean_sample(noisy, velocity, timestep)


def h3_sgf_supervised_video(rollout):
    if rollout.ndim != 5 or rollout.shape[2] != 50:
        raise ValueError("H3 SGF rollout must be [B,C,50,H,W]")
    return rollout[:, :, :47]
