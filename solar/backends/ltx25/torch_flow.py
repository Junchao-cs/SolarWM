"""Torch-native LTX-2.5 Stage0.5 rectified-flow objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from solar.errors import BackendContractError

from .geometry import STABLE_GEOMETRY


def shifted_logit_normal_mu(
    sequence_length: int,
    *,
    min_tokens: int = 1024,
    max_tokens: int = 4096,
    min_shift: float = 0.95,
    max_shift: float = 2.05,
) -> float:
    if sequence_length <= 0 or max_tokens <= min_tokens:
        raise BackendContractError("invalid LTX sigma-shift anchors")
    slope = (max_shift - min_shift) / (max_tokens - min_tokens)
    return slope * sequence_length + min_shift - slope * min_tokens


def sample_shifted_logit_normal(
    batch_size: int,
    *,
    device: torch.device,
    std: float,
    epsilon: float,
    uniform_probability: float,
) -> torch.Tensor:
    if batch_size < 1 or std <= 0 or not 0 < epsilon < 0.5:
        raise BackendContractError("invalid LTX shifted-logit-normal parameters")
    if not 0 <= uniform_probability <= 1:
        raise BackendContractError("invalid LTX uniform sigma probability")
    mu = shifted_logit_normal_mu(STABLE_GEOMETRY.video_tokens)
    normal = torch.randn(batch_size, device=device, dtype=torch.float32) * std + mu
    logit_normal = torch.sigmoid(normal)
    upper = torch.sigmoid(torch.tensor(mu + 3.0902 * std, device=device))
    lower = torch.sigmoid(torch.tensor(mu - 2.5758 * std, device=device))
    zero_terminal = (logit_normal - lower) / (upper - lower)
    stretched = torch.where(
        zero_terminal >= epsilon,
        zero_terminal,
        2 * epsilon - zero_terminal,
    ).clamp(0, 1)
    uniform = (1 - epsilon) * torch.rand(
        batch_size,
        device=device,
        dtype=torch.float32,
    ) + epsilon
    probability = torch.rand(batch_size, device=device, dtype=torch.float32)
    return torch.where(probability > uniform_probability, stretched, uniform)


@dataclass(frozen=True)
class Objective:
    noisy: torch.Tensor
    target_velocity: torch.Tensor
    sigma: torch.Tensor


def prepare_objective(
    video_latent: torch.Tensor,
    first_frame_latent: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> Objective:
    expected = (
        STABLE_GEOMETRY.latent_channels,
        STABLE_GEOMETRY.latent_frames,
        STABLE_GEOMETRY.latent_height,
        STABLE_GEOMETRY.latent_width,
    )
    if video_latent.ndim != 5 or tuple(video_latent.shape[1:]) != expected:
        raise BackendContractError("LTX training latent geometry differs")
    if tuple(first_frame_latent.shape) != (
        video_latent.shape[0],
        STABLE_GEOMETRY.latent_channels,
        1,
        STABLE_GEOMETRY.latent_height,
        STABLE_GEOMETRY.latent_width,
    ):
        raise BackendContractError("LTX first-frame latent geometry differs")
    if not torch.equal(first_frame_latent, video_latent[:, :, :1]):
        raise BackendContractError("LTX first-frame latent is not bit equal")
    if noise.shape != video_latent.shape or noise.device != video_latent.device:
        raise BackendContractError("LTX flow noise layout differs")
    sigma = torch.as_tensor(sigma, device=video_latent.device, dtype=torch.float32)
    if tuple(sigma.shape) != (video_latent.shape[0],):
        raise BackendContractError("LTX flow sigma must be [B]")
    if not bool(torch.isfinite(sigma).all()) or not bool(((sigma >= 0) & (sigma <= 1)).all()):
        raise BackendContractError("LTX flow sigma is outside [0,1]")
    clean = video_latent.float().clone()
    first = first_frame_latent.float()
    clean[:, :, :1] = first
    noise_fp32 = noise.float()
    target = noise_fp32 - clean
    expanded = sigma.view(video_latent.shape[0], 1, 1, 1, 1)
    noisy = (1 - expanded) * clean + expanded * noise_fp32
    noisy[:, :, :1] = first
    return Objective(noisy, target, sigma)


def velocity_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise BackendContractError("LTX velocity prediction/target layout differs")
    # Preserve the dropout=0 graph. Building the full tensor
    # and masking frame zero has a different CUDA reduction/backward order
    # from slicing frame zero away before mse_loss.
    per_element = (prediction.float() - target.float()).square()
    loss_mask = torch.ones_like(per_element, dtype=torch.float32)
    loss_mask[:, :, 0] = 0.0
    return (per_element * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


__all__ = [
    "Objective",
    "prepare_objective",
    "sample_shifted_logit_normal",
    "shifted_logit_normal_mu",
    "velocity_loss",
]
