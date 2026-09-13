"""H3 TF AnyFlow v1.5 in the shared noise-ward flow-map coordinate.

The model callback accepts native H3 data-ward times and returns data-ward
velocity. Conditions (clean history, anchor, prompt and silence) are captured
by the caller and remain identical across the four model evaluations.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from solarwm.training.anyflow import (
    adaptive_rescale_non_diffusion_losses,
    bounded_difference_timesteps,
    central_difference_derivative,
    gaussian_timestep_weights,
)


def h3_anyflow_v15_loss(
    model_velocity: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    clean: torch.Tensor,
    noise: torch.Tensor,
    sigma_t: torch.Tensor,
    sigma_r: torch.Tensor,
    is_diffusion: torch.Tensor,
    *,
    epsilon: float,
    shift: float,
    num_train_timesteps: int = 1000,
    process_group=None,
) -> torch.Tensor:
    """Evaluate detached targets before retaining the prediction's gradient graph.

    The callback must be deterministic for fixed inputs (H3 AnyFlow requires
    zero adapter dropout). Every rank executes diagonal, plus, minus, then
    prediction, preserving the same four evaluations and loss expression.
    """
    if clean.shape != noise.shape or clean.shape[0] != sigma_t.numel():
        raise ValueError("H3 AnyFlow clean/noise/time batch shapes disagree")
    shape = (clean.shape[0],) + (1,) * (clean.ndim - 1)
    raw_t = sigma_t.float() * num_train_timesteps
    raw_r = sigma_r.float() * num_train_timesteps
    noisy = (1 - sigma_t.view(shape)) * clean + sigma_t.view(shape) * noise

    def noise_velocity(sample, t, r):
        # H3 predicts x0-noise and embeds data-ward time, unlike Wan's sigma.
        return -model_velocity(
            sample, 1 - t / num_train_timesteps, 1 - r / num_train_timesteps
        ).float()

    with torch.no_grad():
        diagonal = noise_velocity(noisy, raw_t, raw_t)
        plus_t, minus_t = bounded_difference_timesteps(
            raw_t,
            raw_r,
            epsilon=epsilon,
            num_train_timesteps=num_train_timesteps,
        )
        plus = noisy + diagonal * ((plus_t - raw_t) / num_train_timesteps).view(shape)
        minus = noisy - diagonal * ((raw_t - minus_t) / num_train_timesteps).view(shape)
        plus_v = noise_velocity(plus, plus_t, raw_r)
        minus_v = noise_velocity(minus, minus_t, raw_r)
        derivative = central_difference_derivative(plus_v, minus_v, plus_t, minus_t)
    del diagonal, plus, minus, plus_v, minus_v
    prediction = noise_velocity(noisy, raw_t, raw_r)
    residual = prediction + (raw_t - raw_r).view(shape) * derivative - (noise - clean)
    per_sample = residual.square().flatten(1).mean(1)
    per_sample = adaptive_rescale_non_diffusion_losses(
        per_sample,
        is_diffusion,
        process_group=process_group,
    )
    weights = gaussian_timestep_weights(
        raw_t,
        shift=shift,
        num_train_timesteps=num_train_timesteps,
    )
    return (per_sample * weights).mean()
