"""Dependency-light LTX native rectified-flow math."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from solar.errors import BackendContractError

from .geometry import STABLE_GEOMETRY


def _floating_pair(clean: object, noise: object) -> tuple[np.ndarray, np.ndarray]:
    clean_array = np.asarray(clean)
    noise_array = np.asarray(noise)
    if clean_array.shape != noise_array.shape:
        raise BackendContractError("clean and noise shapes must match")
    if not np.issubdtype(clean_array.dtype, np.floating) or not np.issubdtype(
        noise_array.dtype, np.floating
    ):
        raise BackendContractError("clean and noise must use floating dtypes")
    return clean_array, noise_array


def _sigma(value: object, sample: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    if not np.isfinite(result).all() or np.any(result < 0) or np.any(result > 1):
        raise BackendContractError("sigma must be finite and in [0,1]")
    while result.ndim < sample.ndim:
        result = np.expand_dims(result, -1)
    try:
        np.broadcast_shapes(result.shape, sample.shape)
    except ValueError as exc:
        raise BackendContractError("sigma is not broadcastable to the sample") from exc
    return result


def scale_noise(clean: object, noise: object, sigma: object) -> np.ndarray:
    """Return ``x_sigma=(1-sigma)*x0 + sigma*eps`` in FP32."""

    clean_array, noise_array = _floating_pair(clean, noise)
    expanded = _sigma(sigma, clean_array)
    return (1.0 - expanded) * clean_array.astype(np.float32) + expanded * noise_array.astype(
        np.float32
    )


def velocity_target(clean: object, noise: object) -> np.ndarray:
    """Return the native noise-ward velocity ``eps-x0``."""

    clean_array, noise_array = _floating_pair(clean, noise)
    return noise_array.astype(np.float32) - clean_array.astype(np.float32)


def predict_clean(noisy: object, velocity: object, sigma: object) -> np.ndarray:
    noisy_array, velocity_array = _floating_pair(noisy, velocity)
    return noisy_array.astype(np.float32) - _sigma(sigma, noisy_array) * velocity_array.astype(
        np.float32
    )


def shifted_logit_normal_mu(
    sequence_length: int = STABLE_GEOMETRY.video_tokens,
    *,
    min_tokens: int = 1024,
    max_tokens: int = 4096,
    min_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    """Return the provider's linearly extrapolated shift (10/3 for 7,680 tokens)."""

    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length <= 0
    ):
        raise BackendContractError("sequence_length must be a positive integer")
    if max_tokens <= min_tokens:
        raise BackendContractError("max_tokens must exceed min_tokens")
    slope = (max_shift - min_shift) / (max_tokens - min_tokens)
    return slope * sequence_length + min_shift - slope * min_tokens


def sample_shifted_logit_normal(
    batch_size: int,
    *,
    generator: np.random.Generator,
    sequence_length: int = STABLE_GEOMETRY.video_tokens,
    std: float = 1.0,
    eps: float = 1e-3,
    uniform_probability: float = 0.1,
) -> np.ndarray:
    """Sample the configured shifted/stretched logit-normal mixture."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise BackendContractError("batch_size must be a positive integer")
    if not isinstance(generator, np.random.Generator):
        raise BackendContractError("generator must be an explicit numpy Generator")
    if not math.isfinite(std) or std <= 0 or not 0 < eps < 0.5:
        raise BackendContractError("std/eps are invalid")
    if not 0 <= uniform_probability <= 1:
        raise BackendContractError("uniform_probability must be in [0,1]")
    mu = shifted_logit_normal_mu(sequence_length)
    normal = generator.standard_normal(batch_size, dtype=np.float32) * std + mu
    logit = 1.0 / (1.0 + np.exp(-normal))
    upper = 1.0 / (1.0 + math.exp(-(mu + 3.0902 * std)))
    lower = 1.0 / (1.0 + math.exp(-(mu - 2.5758 * std)))
    raw = (logit - lower) / (upper - lower)
    stretched = np.clip(np.where(raw >= eps, raw, 2 * eps - raw), 0, 1)
    uniform = (1 - eps) * generator.random(batch_size, dtype=np.float32) + eps
    choose = generator.random(batch_size, dtype=np.float32)
    return np.where(choose > uniform_probability, stretched, uniform).astype(np.float32)


def restore_clean_first_latent(noisy: object, first_frame: object) -> np.ndarray:
    """Copy the clean first latent slice into one ``[B,128,20,16,24]`` sample."""

    sample = np.asarray(noisy)
    first = np.asarray(first_frame)
    expected = (
        STABLE_GEOMETRY.latent_channels,
        STABLE_GEOMETRY.latent_frames,
        STABLE_GEOMETRY.latent_height,
        STABLE_GEOMETRY.latent_width,
    )
    if sample.ndim != 5 or sample.shape[1:] != expected:
        raise BackendContractError(f"noisy sample must be [B,{expected}]")
    if not np.issubdtype(sample.dtype, np.floating) or not np.issubdtype(first.dtype, np.floating):
        raise BackendContractError("latent inputs must use floating dtypes")
    if first.shape != (sample.shape[0], *STABLE_GEOMETRY.first_frame_latent_shape):
        raise BackendContractError("first_frame must be [B,128,1,16,24]")
    result = sample.copy()
    result[:, :, :1] = first
    return result


def first_frame_excluded_mse(prediction: object, target: object) -> float:
    """Compute velocity MSE only over latent slices 1..19."""

    predicted, expected = _floating_pair(prediction, target)
    if predicted.ndim != 5 or predicted.shape[1:] != STABLE_GEOMETRY.latent_shape:
        raise BackendContractError("velocity tensors must be [B,128,20,16,24]")
    delta = predicted[:, :, 1:].astype(np.float32) - expected[:, :, 1:].astype(np.float32)
    return float(np.mean(np.square(delta), dtype=np.float64))


@dataclass(frozen=True)
class NativeFlowContract:
    interpolation: str = "(1-sigma)*x0+sigma*eps"
    target: str = "eps-x0"
    clean_first_sigma: float = 0.0
    loss_latent_slice: str = "1:20"


NATIVE_FLOW_CONTRACT = NativeFlowContract()


__all__ = [
    "NATIVE_FLOW_CONTRACT",
    "NativeFlowContract",
    "first_frame_excluded_mse",
    "predict_clean",
    "restore_clean_first_latent",
    "sample_shifted_logit_normal",
    "scale_noise",
    "shifted_logit_normal_mu",
    "velocity_target",
]
