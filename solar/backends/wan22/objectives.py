"""Allocation-light objective helpers shared by Wan stages.

Torch is imported inside calls so configuration and CLI discovery work in a
minimal installation. The formulas preserve the configured velocity convention:
``x_t = (1-sigma) * clean + sigma * noise`` and target ``noise-clean``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise RuntimeError("Wan objective math requires the 'wan' dependency extra") from exc
    return torch


def apply_timestep_shift_array(normalized: Any, shift: float) -> np.ndarray:
    """NumPy reference for fixture tests without a torch runtime."""

    if not isinstance(shift, (int, float)) or not math.isfinite(float(shift)):
        raise ValueError(f"shift must be finite, got {shift!r}")
    if float(shift) <= 0:
        raise ValueError("shift must be positive")
    value = np.asarray(normalized)
    dtype = np.float64 if value.dtype == np.float64 else np.float32
    value = value.astype(dtype, copy=False)
    return float(shift) * value / (1.0 + (float(shift) - 1.0) * value)


def rectified_interpolate_array(clean: Any, noise: Any, sigma: Any) -> np.ndarray:
    """NumPy reference for ``(1-sigma)*clean + sigma*noise`` in FP32."""

    clean_value = np.asarray(clean)
    noise_value = np.asarray(noise)
    if clean_value.shape != noise_value.shape:
        raise ValueError("clean and noise must have identical shapes")
    sigma_value = np.asarray(sigma, dtype=np.float32)
    while sigma_value.ndim < clean_value.ndim:
        sigma_value = np.expand_dims(sigma_value, -1)
    try:
        sigma_value = np.broadcast_to(sigma_value, clean_value.shape)
    except ValueError as exc:
        raise ValueError("sigma is not prefix-broadcastable to clean") from exc
    result = (1.0 - sigma_value) * clean_value.astype(
        np.float32
    ) + sigma_value * noise_value.astype(np.float32)
    return result.astype(clean_value.dtype, copy=False)


def velocity_target_array(clean: Any, noise: Any) -> np.ndarray:
    clean_value = np.asarray(clean)
    noise_value = np.asarray(noise)
    if clean_value.shape != noise_value.shape:
        raise ValueError("clean and noise must have identical shapes")
    return noise_value.astype(np.float32) - clean_value.astype(np.float32)


def weighted_masked_mse_array(prediction: Any, target: Any, mask: Any, weight: Any) -> np.float32:
    """NumPy loss reference with mask-before-subtraction semantics."""

    prediction_value = np.asarray(prediction)
    target_value = np.asarray(target)
    if prediction_value.shape != target_value.shape:
        raise ValueError("prediction and target must have identical shapes")
    mask_value = np.asarray(mask, dtype=np.bool_)
    weight_value = np.asarray(weight, dtype=np.float32)
    while mask_value.ndim < prediction_value.ndim:
        mask_value = np.expand_dims(mask_value, -1)
    while weight_value.ndim < prediction_value.ndim:
        weight_value = np.expand_dims(weight_value, -1)
    try:
        valid = np.broadcast_to(mask_value, prediction_value.shape)
        weights = np.broadcast_to(weight_value, prediction_value.shape)
    except ValueError as exc:
        raise ValueError("mask and weight must be prefix-broadcastable") from exc
    diff = np.where(
        valid,
        prediction_value.astype(np.float32) - target_value.astype(np.float32),
        np.float32(0.0),
    )
    numerator = np.sum(np.square(diff) * weights, dtype=np.float32)
    denominator = max(float(np.sum(valid.astype(np.float32))), 1.0)
    return np.float32(numerator / denominator)


def apply_timestep_shift(normalized: Any, shift: float) -> Any:
    """Apply the rational FlowMatch timestep shift."""

    torch = _torch()
    value = torch.as_tensor(normalized)
    if not torch.is_floating_point(value):
        value = value.to(torch.float32)
    if not isinstance(shift, (int, float)) or not math.isfinite(float(shift)):
        raise ValueError(f"shift must be finite, got {shift!r}")
    if float(shift) <= 0:
        raise ValueError("shift must be positive")
    if float(shift) == 1.0:
        return value
    return float(shift) * value / (1.0 + (float(shift) - 1.0) * value)


def rectified_interpolate(clean: Any, noise: Any, sigma: Any) -> Any:
    """Return the rectified-flow point while performing arithmetic in FP32."""

    torch = _torch()
    clean_value = torch.as_tensor(clean)
    noise_value = torch.as_tensor(noise, device=clean_value.device)
    if clean_value.shape != noise_value.shape:
        raise ValueError("clean and noise must have identical shapes")
    sigma_value = torch.as_tensor(sigma, device=clean_value.device, dtype=torch.float32)
    while sigma_value.ndim < clean_value.ndim:
        sigma_value = sigma_value.unsqueeze(-1)
    try:
        sigma_value = torch.broadcast_to(sigma_value, clean_value.shape)
    except RuntimeError as exc:
        raise ValueError("sigma is not prefix-broadcastable to clean") from exc
    result = (1.0 - sigma_value) * clean_value.float() + sigma_value * noise_value.float()
    return result.to(clean_value.dtype)


def velocity_target(clean: Any, noise: Any) -> Any:
    """Return the Wan flow-matching target ``noise - clean``."""

    torch = _torch()
    clean_value = torch.as_tensor(clean)
    noise_value = torch.as_tensor(noise, device=clean_value.device)
    if clean_value.shape != noise_value.shape:
        raise ValueError("clean and noise must have identical shapes")
    return noise_value.float() - clean_value.float()


def weighted_masked_mse(prediction: Any, target: Any, mask: Any, weight: Any) -> Any:
    """Mask before subtraction so invalid conditioned values cannot poison loss."""

    torch = _torch()
    prediction_value = torch.as_tensor(prediction)
    target_value = torch.as_tensor(target, device=prediction_value.device)
    if prediction_value.shape != target_value.shape:
        raise ValueError("prediction and target must have identical shapes")
    if prediction_value.ndim == 0:
        raise ValueError("prediction must have at least one dimension")
    mask_value = torch.as_tensor(mask, device=prediction_value.device, dtype=torch.bool)
    weight_value = torch.as_tensor(weight, device=prediction_value.device, dtype=torch.float32)
    while mask_value.ndim < prediction_value.ndim:
        mask_value = mask_value.unsqueeze(-1)
    while weight_value.ndim < prediction_value.ndim:
        weight_value = weight_value.unsqueeze(-1)
    try:
        # The Wan loss builds a full-size contiguous FP32 mask with
        # ones_like(prediction). Materializing the broadcast here preserves its
        # CUDA reduction order for long 153f tensors; a stride-zero bool view can
        # differ by one or two FP32 ULPs even though the formula is equivalent.
        valid = torch.broadcast_to(mask_value, prediction_value.shape).float().contiguous()
        torch.broadcast_shapes(weight_value.shape, prediction_value.shape)
    except RuntimeError as exc:
        raise ValueError("mask and weight must be prefix-broadcastable") from exc
    diff = prediction_value.float() - target_value.float()
    diff = torch.where(valid > 0, diff, torch.zeros_like(diff))
    numerator = ((diff**2) * weight_value).sum()
    denominator = valid.sum().clamp_min(1.0)
    return numerator / denominator


def first_frame_loss_mask(batch_size: int, latent_frames: int, dropped: Any) -> Any:
    """Build the TI2V mask: supervise frame zero only when image condition drops."""

    torch = _torch()
    if batch_size < 1 or latent_frames < 1:
        raise ValueError("batch_size and latent_frames must be positive")
    dropped_value = torch.as_tensor(dropped, dtype=torch.bool)
    if tuple(dropped_value.shape) != (batch_size,):
        raise ValueError(f"dropped must have shape {(batch_size,)}")
    mask = torch.ones((batch_size, latent_frames), dtype=torch.bool, device=dropped_value.device)
    mask[:, 0] = dropped_value
    return mask


__all__ = [
    "apply_timestep_shift",
    "apply_timestep_shift_array",
    "first_frame_loss_mask",
    "rectified_interpolate",
    "rectified_interpolate_array",
    "velocity_target",
    "velocity_target_array",
    "weighted_masked_mse",
    "weighted_masked_mse_array",
]
