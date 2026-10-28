"""Pure AnyFlow forward-map v1.5 contracts used by the Wan Stage1 route."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any, NamedTuple

import numpy as np

from solar.errors import BackendContractError

from .objectives import apply_timestep_shift

SAMPLE_TYPE_DIFFUSION = 0
SAMPLE_TYPE_CONSISTENCY = 1
SAMPLE_TYPE_FLOW_MAP = 2


class AnyFlowTimePairs(NamedTuple):
    t: Any
    r: Any
    is_diffusion: Any
    is_consistency: Any
    sample_type: Any


def time_pairs_from_uniforms(
    u1: Any,
    u2: Any,
    *,
    logical_dp_rank: int = 0,
    logical_dp_world_size: int = 1,
    diffusion_ratio: float = 0.5,
    consistency_ratio: float = 0.25,
) -> AnyFlowTimePairs:
    """Deterministic NumPy reference after the two RNG draws."""

    first = np.asarray(u1, dtype=np.float32)
    second = np.asarray(u2, dtype=np.float32)
    if first.ndim != 1 or first.shape != second.shape or first.size == 0:
        raise ValueError("u1 and u2 must be non-empty, equal one-dimensional arrays")
    if np.any(first < 0) or np.any(first >= 1) or np.any(second < 0) or np.any(second >= 1):
        raise ValueError("uniform values must be in [0,1)")
    if logical_dp_world_size <= 0 or not 0 <= logical_dp_rank < logical_dp_world_size:
        raise ValueError("invalid logical DP identity")
    if diffusion_ratio < 0 or consistency_ratio < 0 or diffusion_ratio + consistency_ratio > 1:
        raise ValueError("invalid AnyFlow sample ratios")
    batch_size = first.size
    t = first.copy()
    r = second * t
    global_batch = batch_size * logical_dp_world_size
    indices = np.arange(batch_size) + logical_dp_rank * batch_size
    num_diffusion = round(diffusion_ratio * global_batch)
    num_consistency = round(consistency_ratio * global_batch)
    is_diffusion = indices < num_diffusion
    is_consistency = (indices >= num_diffusion) & (indices < num_diffusion + num_consistency)
    is_flow_map = ~(is_diffusion | is_consistency)
    r = np.where(is_diffusion, t, r)
    r = np.where(is_consistency, np.float32(0), r)
    smallest = np.nextafter(np.float32(0), np.float32(1), dtype=np.float32)
    strict_r = np.where(r > 0, r, smallest)
    strict_t = np.where(t > strict_r, t, np.nextafter(strict_r, np.float32(1)))
    r = np.where(is_flow_map, strict_r, r)
    t = np.where(is_flow_map, strict_t, t)
    sample_type = np.full(batch_size, SAMPLE_TYPE_FLOW_MAP, dtype=np.int8)
    sample_type[is_diffusion] = SAMPLE_TYPE_DIFFUSION
    sample_type[is_consistency] = SAMPLE_TYPE_CONSISTENCY
    return AnyFlowTimePairs(t, r, is_diffusion, is_consistency, sample_type)


def build_flowmap_schedule_array(
    num_inference_steps: int, shift: float, num_train_timesteps: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    if num_inference_steps <= 0 or num_train_timesteps <= 0:
        raise ValueError("step counts must be positive")
    normalized = np.linspace(1.0, 0.0, num_inference_steps + 1, dtype=np.float64)
    shifted = float(shift) * normalized / (1.0 + (float(shift) - 1.0) * normalized)
    raw = shifted * num_train_timesteps
    return raw[:-1].copy(), raw[1:].copy()


def bounded_difference_timesteps_array(
    t: Any, r: Any, *, epsilon: float = 5.0, num_train_timesteps: int = 1000
) -> tuple[np.ndarray, np.ndarray]:
    if epsilon <= 0 or num_train_timesteps <= 0:
        raise ValueError("epsilon and num_train_timesteps must be positive")
    try:
        t_value, r_value = np.broadcast_arrays(np.asarray(t), np.asarray(r))
    except ValueError as exc:
        raise ValueError("t and r must be broadcastable") from exc
    return np.minimum(t_value + epsilon, num_train_timesteps), np.maximum(
        t_value - epsilon, r_value
    )


def central_difference_target_array(
    velocity: Any,
    u_plus: Any,
    u_minus: Any,
    t: Any,
    r: Any,
    *,
    epsilon: float = 5.0,
    guidance: float = 1.0,
) -> np.ndarray:
    velocity_value = np.asarray(velocity)
    plus_value = np.asarray(u_plus)
    minus_value = np.asarray(u_minus)
    if velocity_value.shape != plus_value.shape or velocity_value.shape != minus_value.shape:
        raise ValueError("velocity, u_plus and u_minus must have identical shapes")
    if epsilon <= 0 or not math.isfinite(float(guidance)) or guidance == 0:
        raise ValueError("epsilon must be positive and guidance finite/non-zero")
    t_value, r_value = np.broadcast_arrays(np.asarray(t), np.asarray(r))
    while t_value.ndim < velocity_value.ndim:
        t_value = np.expand_dims(t_value, -1)
        r_value = np.expand_dims(r_value, -1)
    derivative = (plus_value - minus_value) / (2.0 * epsilon * guidance)
    target = velocity_value - (t_value - r_value) * derivative
    return np.where(t_value == r_value, velocity_value, target)


def gaussian_timestep_weights_array(
    shifted_raw_timesteps: Any,
    *,
    shift: float,
    num_train_timesteps: int = 1000,
) -> np.ndarray:
    """Mean-normalized Gaussian grid used by AnyFlow v1.5."""

    if num_train_timesteps <= 0:
        raise ValueError("num_train_timesteps must be positive")
    inputs = np.asarray(shifted_raw_timesteps)
    dtype = np.float64 if inputs.dtype == np.float64 else np.float32
    normalized = np.linspace(1.0, 0.0, num_train_timesteps + 1, dtype=dtype)[:-1]
    shifted = float(shift) * normalized / (1.0 + (float(shift) - 1.0) * normalized)
    raw_grid = shifted * num_train_timesteps
    unnormalized = np.exp(
        -2.0 * np.square((raw_grid - num_train_timesteps / 2.0) / num_train_timesteps)
    )
    unnormalized -= np.min(unnormalized)
    total = np.sum(unnormalized)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Gaussian training-grid weights have invalid normalization")
    grid_weights = unnormalized * (num_train_timesteps / total)
    flat = inputs.astype(dtype, copy=False).reshape(-1)
    nearest = np.argmin(np.abs(raw_grid[:, None] - flat[None, :]), axis=0)
    return grid_weights[nearest].reshape(inputs.shape)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("AnyFlow math requires the 'wan' dependency extra") from exc
    return torch


def validate_anyflow_config(train: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    """Validate the supported AnyFlow route."""

    removed = "anyflow_negative_embedding_digest"
    if removed in train:
        raise BackendContractError(
            f"train.{removed} is not supported; provide the embedding path and SolarWM "
            "will validate the loaded tensor"
        )
    checks = {
        "train.objective_variant": (str(train.get("objective_variant", "")) == "v1_5"),
        "train.anyflow_variant": (str(train.get("anyflow_variant", "")) == "v1_5"),
        "train.anyflow_weight_type": (str(train.get("anyflow_weight_type", "")) == "gaussian"),
        "train.anyflow_deltatime_type": (str(train.get("anyflow_deltatime_type", "")) == "r"),
    }
    failed = [field for field, accepted in checks.items() if not accepted]
    if failed:
        raise BackendContractError(f"AnyFlow v1.5 contract mismatch: {failed}")
    gate = float(train.get("anyflow_gate", -1.0))
    epsilon = float(train.get("anyflow_epsilon", 0.0))
    diffusion = float(train.get("anyflow_diffusion_ratio", -1.0))
    consistency = float(train.get("anyflow_consistency_ratio", -1.0))
    guidance = float(train.get("anyflow_fuse_guidance_scale", 0.0))
    if not all(math.isfinite(value) for value in (gate, epsilon, diffusion, consistency, guidance)):
        raise BackendContractError("AnyFlow numeric settings must be finite")
    if not 0.0 <= gate <= 1.0 or epsilon <= 0:
        raise BackendContractError("AnyFlow gate must be in [0,1] and epsilon must be positive")
    if diffusion < 0 or consistency < 0 or diffusion + consistency > 1.0:
        raise BackendContractError(
            "AnyFlow sample ratios must be non-negative and sum to at most one"
        )
    if guidance <= 0:
        raise BackendContractError("AnyFlow guidance must be positive")
    if guidance != 1.0:
        embedding = str(train.get("anyflow_negative_embedding", "")).strip()
        path = PurePosixPath(embedding)
        if (
            not embedding
            or embedding.startswith("/")
            or "://" in embedding
            or "\\" in embedding
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise BackendContractError(
                "guided AnyFlow train.anyflow_negative_embedding must be a "
                "base-model-relative POSIX path"
            )
    passes = validation.get("passes", [])
    if not isinstance(passes, list) or not passes:
        raise BackendContractError("AnyFlow validation.passes must be a non-empty list")
    nfes = {int(item.get("num_inference_steps", 0)) for item in passes if isinstance(item, Mapping)}
    solvers = {str(item.get("solver", "")) for item in passes if isinstance(item, Mapping)}
    if nfes != {4, 50} or solvers != {"flowmap"}:
        raise BackendContractError(
            "AnyFlow validation must contain only flowmap NFE4 and NFE50 passes"
        )


def sample_time_pairs(
    batch_size: int,
    *,
    logical_dp_rank: int = 0,
    logical_dp_world_size: int = 1,
    diffusion_ratio: float = 0.5,
    consistency_ratio: float = 0.25,
    generator: Any = None,
    device: Any = "cpu",
) -> AnyFlowTimePairs:
    """Sample v1.5 pairs using the required two-uniform RNG order."""

    torch = _torch()
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not isinstance(logical_dp_world_size, int) or logical_dp_world_size <= 0:
        raise ValueError("logical_dp_world_size must be positive")
    if not isinstance(logical_dp_rank, int) or not 0 <= logical_dp_rank < logical_dp_world_size:
        raise ValueError("logical_dp_rank is outside the logical DP world")
    if diffusion_ratio < 0 or consistency_ratio < 0 or diffusion_ratio + consistency_ratio > 1:
        raise ValueError("invalid AnyFlow sample ratios")
    u1 = torch.rand(batch_size, dtype=torch.float32, device=device, generator=generator)
    u2 = torch.rand(batch_size, dtype=torch.float32, device=device, generator=generator)
    t = u1
    r = u2 * t
    global_batch = batch_size * logical_dp_world_size
    global_indices = torch.arange(batch_size, device=device) + logical_dp_rank * batch_size
    num_diffusion = round(diffusion_ratio * global_batch)
    num_consistency = round(consistency_ratio * global_batch)
    is_diffusion = global_indices < num_diffusion
    is_consistency = (global_indices >= num_diffusion) & (
        global_indices < num_diffusion + num_consistency
    )
    is_flow_map = ~(is_diffusion | is_consistency)
    r = torch.where(is_diffusion, t, r)
    r = torch.where(is_consistency, torch.zeros_like(r), r)
    zero = torch.zeros((), dtype=r.dtype, device=r.device)
    one = torch.ones((), dtype=r.dtype, device=r.device)
    smallest = torch.nextafter(zero, one)
    strict_r = torch.where(r > 0, r, smallest)
    strict_t = torch.where(t > strict_r, t, torch.nextafter(strict_r, one))
    r = torch.where(is_flow_map, strict_r, r)
    t = torch.where(is_flow_map, strict_t, t)
    sample_type = torch.full((batch_size,), SAMPLE_TYPE_FLOW_MAP, dtype=torch.int8, device=device)
    sample_type = torch.where(
        is_diffusion, torch.full_like(sample_type, SAMPLE_TYPE_DIFFUSION), sample_type
    )
    sample_type = torch.where(
        is_consistency, torch.full_like(sample_type, SAMPLE_TYPE_CONSISTENCY), sample_type
    )
    return AnyFlowTimePairs(t, r, is_diffusion, is_consistency, sample_type)


def build_flowmap_schedule(
    num_inference_steps: int,
    shift: float,
    num_train_timesteps: int = 1000,
    *,
    device: Any = "cpu",
) -> tuple[Any, Any]:
    """Return shifted raw (t,r) pairs used by validation and inference."""

    torch = _torch()
    if num_inference_steps <= 0 or num_train_timesteps <= 0:
        raise ValueError("step counts must be positive")
    normalized = torch.linspace(
        1.0, 0.0, num_inference_steps + 1, dtype=torch.float64, device=device
    )
    raw = apply_timestep_shift(normalized, shift) * num_train_timesteps
    return raw[:-1].contiguous(), raw[1:].contiguous()


def bounded_difference_timesteps(
    t: Any, r: Any, *, epsilon: float = 5.0, num_train_timesteps: int = 1000
) -> tuple[Any, Any]:
    """Return the v1.5 finite-difference window bounded by r and N."""

    torch = _torch()
    if epsilon <= 0 or num_train_timesteps <= 0:
        raise ValueError("epsilon and num_train_timesteps must be positive")
    t_value = torch.as_tensor(t)
    r_value = torch.as_tensor(r, device=t_value.device, dtype=t_value.dtype)
    try:
        t_value, r_value = torch.broadcast_tensors(t_value, r_value)
    except RuntimeError as exc:
        raise ValueError("t and r must be broadcastable") from exc
    return (t_value + epsilon).clamp_max(num_train_timesteps), torch.maximum(
        t_value - epsilon, r_value
    )


def central_difference_target(
    velocity: Any,
    u_plus: Any,
    u_minus: Any,
    t: Any,
    r: Any,
    *,
    epsilon: float = 5.0,
    guidance: float = 1.0,
) -> Any:
    """Build the detached v1.5 forward-map target."""

    torch = _torch()
    velocity_value = torch.as_tensor(velocity)
    plus_value = torch.as_tensor(u_plus, device=velocity_value.device)
    minus_value = torch.as_tensor(u_minus, device=velocity_value.device)
    if velocity_value.shape != plus_value.shape or velocity_value.shape != minus_value.shape:
        raise ValueError("velocity, u_plus and u_minus must have identical shapes")
    if epsilon <= 0 or not math.isfinite(float(guidance)) or guidance == 0:
        raise ValueError("epsilon must be positive and guidance finite/non-zero")
    t_value = torch.as_tensor(t, device=velocity_value.device, dtype=velocity_value.dtype)
    r_value = torch.as_tensor(r, device=velocity_value.device, dtype=velocity_value.dtype)
    t_value, r_value = torch.broadcast_tensors(t_value, r_value)
    while t_value.ndim < velocity_value.ndim:
        t_value = t_value.unsqueeze(-1)
        r_value = r_value.unsqueeze(-1)
    derivative = (plus_value.detach() - minus_value.detach()) / (2.0 * epsilon * guidance)
    target = velocity_value.detach() - (t_value - r_value) * derivative
    return torch.where(t_value == r_value, velocity_value.detach(), target).detach()


__all__ = [
    "SAMPLE_TYPE_CONSISTENCY",
    "SAMPLE_TYPE_DIFFUSION",
    "SAMPLE_TYPE_FLOW_MAP",
    "AnyFlowTimePairs",
    "bounded_difference_timesteps",
    "bounded_difference_timesteps_array",
    "build_flowmap_schedule",
    "build_flowmap_schedule_array",
    "central_difference_target",
    "central_difference_target_array",
    "gaussian_timestep_weights_array",
    "sample_time_pairs",
    "time_pairs_from_uniforms",
    "validate_anyflow_config",
]
