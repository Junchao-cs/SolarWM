"""Shared self-gradient-forcing schedule and detached score surrogate."""

from __future__ import annotations

import torch
from torch import Tensor


def should_update_student(global_step: int, critic_updates_per_student: int) -> bool:
    """Match the reference schedule: warm critic first, then update every N steps."""
    ratio = int(critic_updates_per_student)
    if ratio < 1:
        raise ValueError("train.critic_updates_per_student must be >= 1")
    step = int(global_step)
    return step > 0 and step % ratio == 0


def _expanded_mask(mask: Tensor | None, value: Tensor) -> Tensor | None:
    if mask is None:
        return None
    try:
        return torch.broadcast_to(mask.to(device=value.device, dtype=torch.bool), value.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is not broadcastable to {tuple(value.shape)}"
        ) from exc


def compute_sgf_kl_gradient(
    *,
    fake_x0: Tensor,
    real_x0: Tensor,
    student_output: Tensor,
    mask: Tensor | None = None,
    normalize: bool = True,
) -> Tensor:
    """Compute the reference DMD gradient, with stable masked normalization."""
    fake = fake_x0.float()
    real = real_x0.float()
    output = student_output.float()
    grad = fake - real
    expanded = _expanded_mask(mask, grad)
    if expanded is not None:
        # Mask before every reduction/non-linearity. A conditioned frame can
        # legitimately contain non-finite score output at t=0; NaN * 0 would
        # otherwise poison the normalizer and silently zero every valid frame.
        grad = torch.where(expanded, grad, torch.zeros_like(grad))
    if normalize:
        distance = (output - real).abs()
        reduce_dims = tuple(range(1, distance.ndim))
        if expanded is None:
            normalizer = distance.mean(dim=reduce_dims, keepdim=True)
        else:
            distance = torch.where(expanded, distance, torch.zeros_like(distance))
            weights = expanded.float()
            normalizer = distance.sum(dim=reduce_dims, keepdim=True)
            normalizer = normalizer / weights.sum(dim=reduce_dims, keepdim=True).clamp_min(1.0)
        grad = grad / normalizer.clamp_min(1e-8)
    return torch.nan_to_num(grad)


def sgf_student_loss(
    student_output: Tensor,
    kl_gradient: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Surrogate whose derivative with respect to student output is the KL gradient."""
    output = student_output.float()
    target = (output - kl_gradient.detach()).detach()
    diff = output - target
    expanded = _expanded_mask(mask, diff)
    if expanded is None:
        return 0.5 * diff.square().mean()
    # Mask before square so a non-finite conditioned frame has zero gradient
    # instead of producing NaN through a later multiplication by zero.
    diff = torch.where(expanded, diff, torch.zeros_like(diff))
    return 0.5 * diff.square().sum() / expanded.float().sum().clamp_min(1.0)
