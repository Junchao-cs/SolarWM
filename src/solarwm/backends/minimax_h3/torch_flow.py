"""MiniMax-H3 rectified-flow convention and shifted sigma schedules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class H3FlowSchedule:
    """A shifted H3 sigma grid and its model timesteps (terminal sigma excluded)."""

    sigmas: Any
    timesteps: Any


def _broadcast_time(value: Any, sample: Any) -> Any:
    import torch

    value = torch.as_tensor(value, device=sample.device, dtype=sample.dtype)
    while value.ndim < sample.ndim:
        value = value.unsqueeze(-1)
    return value


def scale_noise(clean_sample: Any, noise: Any, timestep: Any) -> Any:
    """Apply H3 forward flow: ``x_t=t*x0+(1-t)*noise``; ``t=1`` is clean."""

    if clean_sample.shape != noise.shape:
        raise ValueError(
            f"clean/noise shapes must match, got {clean_sample.shape} and {noise.shape}"
        )
    t = _broadcast_time(timestep, clean_sample)
    return t * clean_sample + (1.0 - t) * noise


def add_noise(clean: Any, noise: Any, t: Any) -> Any:
    """Trainer-facing alias for :func:`scale_noise`."""

    return scale_noise(clean, noise, t)


def data_velocity_target(clean_sample: Any, noise: Any) -> Any:
    """Return H3's data-ward velocity target ``x0-noise``."""

    if clean_sample.shape != noise.shape:
        raise ValueError(
            f"clean/noise shapes must match, got {clean_sample.shape} and {noise.shape}"
        )
    return clean_sample - noise


def velocity_target(clean: Any, noise: Any) -> Any:
    """Trainer-facing alias returning H3's ``clean-noise`` target."""

    return data_velocity_target(clean, noise)


def sample_shifted_timestep(
    batch: int | tuple[int, ...],
    shift: float,
    device: Any,
    dtype: Any,
    generator: Any = None,
) -> Any:
    """Sample training timesteps from a uniformly sampled, shifted sigma.

    The random variable is a base ``sigma ~ U(0,1)``.  H3's exponential
    shift is applied to sigma and the transformer convention is returned as
    ``t = 1-sigma_shifted``.
    """

    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    import torch

    shape = (batch,) if isinstance(batch, int) else tuple(batch)
    if not shape or any(int(size) <= 0 for size in shape):
        raise ValueError(f"batch must describe a positive shape, got {batch!r}")
    base_sigma = torch.rand(shape, device=device, dtype=dtype, generator=generator)
    return 1.0 - shift_sigma(base_sigma, shift)


def predict_clean_sample(noisy_sample: Any, velocity: Any, timestep: Any) -> Any:
    """Recover ``x0 = x_t + (1-t)*v`` from H3 data-ward velocity."""

    sigma = 1.0 - _broadcast_time(timestep, noisy_sample)
    return noisy_sample + sigma * velocity


def shift_sigma(sigma: Any, shift: float) -> Any:
    """Apply H3's exponential shift ``s*sigma/(1+(s-1)*sigma)``."""

    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def make_shifted_schedule(
    num_inference_steps: int,
    *,
    shift: float = 12.0,
    device: Any = None,
) -> H3FlowSchedule:
    """Build the exact H3 grid, including terminal sigma zero.

    ``num_inference_steps`` names grid points in upstream H3, therefore the
    transformer runs ``len(sigmas)-1`` evaluations.
    """

    if num_inference_steps < 2:
        raise ValueError("num_inference_steps must be at least 2")
    import torch

    base = torch.linspace(1.0, 0.0, int(num_inference_steps), dtype=torch.float32)
    sigmas = torch.unique_consecutive(shift_sigma(base, shift)).to(device=device)
    return H3FlowSchedule(sigmas=sigmas, timesteps=(1.0 - sigmas[:-1]))


def euler_step(sample: Any, velocity: Any, timestep: Any, sigma: Any, sigma_next: Any) -> Any:
    """Take one deterministic H3 Euler step in float32 for half inputs."""

    import torch

    denoised = predict_clean_sample(sample, velocity, timestep)
    compute_dtype = (
        torch.float32 if sample.dtype in (torch.float16, torch.bfloat16) else sample.dtype
    )
    sigma = torch.as_tensor(sigma, device=sample.device, dtype=compute_dtype)
    sigma_next = torch.as_tensor(sigma_next, device=sample.device, dtype=compute_dtype)
    ratio = sigma_next / sigma
    result = ratio * sample.to(compute_dtype) + (1.0 - ratio) * denoised.to(compute_dtype)
    return result.to(sample.dtype)


__all__ = [
    "H3FlowSchedule",
    "add_noise",
    "data_velocity_target",
    "euler_step",
    "make_shifted_schedule",
    "predict_clean_sample",
    "sample_shifted_timestep",
    "scale_noise",
    "shift_sigma",
    "velocity_target",
]
