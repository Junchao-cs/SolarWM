"""MiniMax-H3 data-ward rectified-flow arithmetic.

The public helpers use NumPy so their signs, schedules, and reconstruction
identity can be tested without a GPU runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class H3FlowSchedule:
    """Shifted sigma grid and model timesteps; terminal sigma has no timestep."""

    sigmas: np.ndarray
    timesteps: np.ndarray


def _pair(clean: object, noise: object) -> tuple[np.ndarray, np.ndarray]:
    clean_array = np.asarray(clean)
    noise_array = np.asarray(noise)
    if clean_array.shape != noise_array.shape:
        raise ValueError(
            f"clean/noise shapes must match, got {clean_array.shape} and {noise_array.shape}"
        )
    if not np.issubdtype(clean_array.dtype, np.floating):
        raise TypeError("clean sample must use a floating dtype")
    if not np.issubdtype(noise_array.dtype, np.floating):
        raise TypeError("noise sample must use a floating dtype")
    return clean_array, noise_array


def _broadcast_time(value: object, sample: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=sample.dtype)
    while result.ndim < sample.ndim:
        result = np.expand_dims(result, axis=-1)
    return result


def scale_noise(clean_sample: object, noise: object, timestep: object) -> np.ndarray:
    """Apply ``x_t = t*x0 + (1-t)*noise`` where ``t=1`` is clean."""

    clean_array, noise_array = _pair(clean_sample, noise)
    time = _broadcast_time(timestep, clean_array)
    return time * clean_array + (1.0 - time) * noise_array


def data_velocity_target(clean_sample: object, noise: object) -> np.ndarray:
    """Return H3's data-ward velocity target, ``x0 - noise``."""

    clean_array, noise_array = _pair(clean_sample, noise)
    return clean_array - noise_array


def predict_clean_sample(
    noisy_sample: object,
    velocity: object,
    timestep: object,
) -> np.ndarray:
    """Recover ``x0 = x_t + (1-t)*v`` under the H3 velocity convention."""

    noisy_array = np.asarray(noisy_sample)
    velocity_array = np.asarray(velocity)
    if noisy_array.shape != velocity_array.shape:
        raise ValueError(
            "noisy sample and velocity must have equal shapes, "
            f"got {noisy_array.shape} and {velocity_array.shape}"
        )
    sigma = 1.0 - _broadcast_time(timestep, noisy_array)
    return noisy_array + sigma * velocity_array


def shift_sigma(sigma: object, shift: float) -> np.ndarray:
    """Apply H3's exponential shift ``s*sigma/(1+(s-1)*sigma)``."""

    if not np.isfinite(shift) or shift <= 0:
        raise ValueError(f"shift must be finite and positive, got {shift}")
    values = np.asarray(sigma)
    return shift * values / (1.0 + (shift - 1.0) * values)


def sample_shifted_timestep(
    shape: int | tuple[int, ...],
    *,
    shift: float,
    generator: np.random.Generator,
    dtype: np.dtype | type = np.float32,
) -> np.ndarray:
    """Sample ``t=1-shift_sigma(U(0,1))`` from an explicit RNG."""

    normalized = (shape,) if isinstance(shape, int) else tuple(shape)
    if not normalized or any(int(size) <= 0 for size in normalized):
        raise ValueError(f"shape must be positive, got {shape!r}")
    if not isinstance(generator, np.random.Generator):
        raise TypeError("generator must be an explicit numpy.random.Generator")
    base_sigma = generator.random(normalized, dtype=np.dtype(dtype))
    return 1.0 - shift_sigma(base_sigma, shift)


def make_shifted_schedule(
    num_inference_steps: int,
    *,
    shift: float = 12.0,
) -> H3FlowSchedule:
    """Build the exact descending shifted grid, including terminal sigma zero."""

    if num_inference_steps < 2:
        raise ValueError("num_inference_steps must be at least 2")
    base = np.linspace(1.0, 0.0, int(num_inference_steps), dtype=np.float32)
    shifted = np.asarray(shift_sigma(base, shift), dtype=np.float32)
    keep = np.concatenate((np.asarray([True]), shifted[1:] != shifted[:-1]))
    sigmas = shifted[keep]
    return H3FlowSchedule(sigmas=sigmas, timesteps=1.0 - sigmas[:-1])


def euler_step(
    sample: object,
    velocity: object,
    timestep: object,
    sigma: object,
    sigma_next: object,
) -> np.ndarray:
    """Take one deterministic H3 Euler step using float32 accumulation."""

    sample_array = np.asarray(sample)
    velocity_array = np.asarray(velocity)
    if sample_array.shape != velocity_array.shape:
        raise ValueError("sample and velocity shapes must match")
    dtype_name = sample_array.dtype.name
    compute_dtype = (
        np.dtype(np.float32) if dtype_name in {"float16", "bfloat16"} else sample_array.dtype
    )
    sigma_array = np.asarray(sigma, dtype=compute_dtype)
    sigma_next_array = np.asarray(sigma_next, dtype=compute_dtype)
    if np.any(sigma_array == 0):
        raise ValueError("Euler step cannot start from sigma zero")
    denoised = predict_clean_sample(sample_array, velocity_array, timestep)
    ratio = sigma_next_array / sigma_array
    result = ratio * sample_array.astype(compute_dtype) + (1.0 - ratio) * denoised.astype(
        compute_dtype
    )
    return result.astype(sample_array.dtype, copy=False)


add_noise = scale_noise
velocity_target = data_velocity_target


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


def explicit_schedule(sigmas: Sequence[float]) -> H3FlowSchedule:
    """A schedule given as its sigma list (a released few-step adapter's own steps), descending
    from at most 1 to 0."""

    values = np.asarray([float(x) for x in sigmas], dtype=np.float32)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.isfinite(values).all()
        or values[0] > 1.0
        or values[-1] != 0.0
        or np.any(np.diff(values) >= 0)
    ):
        raise ValueError(
            f"a schedule descends strictly from at most 1 to exactly 0, got {list(sigmas)}"
        )
    return H3FlowSchedule(sigmas=values, timesteps=1.0 - values[:-1])
