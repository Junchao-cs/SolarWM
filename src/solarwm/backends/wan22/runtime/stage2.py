"""Executable Wan2.2 TI2V-5B Stage2 self-gradient-forcing runtime.

This module keeps the three Stage2 roles explicit and implements two-pass
six-chunk replay, critic/student update cadence, paired transactional checkpoints,
and an injectable generation boundary shared by validation and inference.
"""

from __future__ import annotations

import copy
import gc
import io
import json
import math
import os
import random
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from solarwm.checkpoint import (
    CheckpointContract,
    CheckpointTransaction,
    assert_resume_compatible,
    verify_checkpoint,
)
from solarwm.errors import BackendContractError
from solarwm.runtime.output_layout import checkpoint_model_dir, validation_staging_root
from solarwm.runtime.randomness import model_init_seed
from solarwm.training.ema import ShardedEMA
from solarwm.training.engine import JsonlEventSink, suspend_automatic_cycle_collection
from solarwm.training.schedule import make_warmup_cosine

from ..sgf import (
    RoleInitialization,
    compute_kl_gradient,
    reference_cfg,
    sample_shifted_score_timesteps,
    sgf_critic_flow_loss,
    sgf_student_loss,
    should_update_student,
    student_update_steps,
    validate_checkpoint_transaction,
    warp_denoising_steps,
)
from .assets import WanAssetLayout
from .checkpoint import normalize_model_state
from .codec import Wan5BOnlineCodec
from .components import (
    Wan5BVAE,
    WanDiffusion,
    WanTextEncoder,
    build_diffusion_architecture,
)
from .data import build_raw_dataloader
from .distributed import cleanup_torchrun, initialize_torchrun, wrap_transformer_fsdp
from .readiness import probe_runtime
from .stage0p5 import expand_timesteps_to_tokens

_CHECKPOINT_FORMAT = "solarwm.wan22-stage2-sgf.v1"
_TORCHRUN_OWNER_ENV = "SOLARWM_TORCHRUN_LIFECYCLE_OWNER"
_STREAMING_VAE_LATENT_CHUNK = 60
_FILE_STREAMING_PIXEL_THRESHOLD = 4096
_FILE_STREAMING_VAE_LATENT_CHUNK = 12


class Stage2GenerationRunner(Protocol):
    """One generation callable used by standalone inference and validation."""

    def __call__(
        self,
        config: Mapping[str, Any],
        *,
        provider: Any | None = None,
        cases: Sequence[Any] | None = None,
        weights_ids: Mapping[str, str] | None = None,
        output_dir: str | Path | None = None,
    ) -> Any: ...


@dataclass(frozen=True)
class RoleCheckpointReceipt:
    role: str
    path: Path
    object_bytes: int
    step: int | None
    weights: str
    stage: str
    objective: str
    camera_translation_transform: str


@dataclass(frozen=True)
class Stage2Rollout:
    output: Any
    cache_target: Any
    noisy_at_exit: Any
    loss_mask: Any
    exit_index: int
    denoised_timestep_from: float
    denoised_timestep_to: float


@dataclass(frozen=True)
class RestoredStage2Checkpoint:
    step: int
    student_step: int
    identity: str
    path: Path


def _stage2_initialization_receipt(
    receipts: Mapping[str, RoleCheckpointReceipt],
) -> dict[str, Any]:
    """Aggregate the three verified roles without binding launch-local paths."""

    if set(receipts) != {"student", "teacher", "critic"}:
        raise BackendContractError("Stage2 initialization requires all three role receipts")
    roles: dict[str, dict[str, Any]] = {}
    for role, receipt in sorted(receipts.items()):
        if receipt.step is None or receipt.step <= 0:
            raise BackendContractError(f"Stage2 {role} receipt has no verified source step")
        roles[role] = {
            "step": int(receipt.step),
            "weights": receipt.weights,
            "stage": receipt.stage,
            "objective": receipt.objective,
            "camera_translation_transform": receipt.camera_translation_transform,
        }
    identity = "|".join(
        (
            f"{role}={values['stage']}/{values['objective']}/"
            f"{values['weights']}/step-{values['step']}/"
            f"{values['camera_translation_transform']}"
        )
        for role, values in roles.items()
    )
    return {
        "schema": "solarwm.wan22-stage2-initialization.v1",
        "initialization_id": f"roles:{identity}",
        "roles": roles,
    }


def _distributed_context() -> tuple[Any, int, int]:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist, dist.get_rank(), dist.get_world_size()
    return dist, 0, 1


def _collective_error(local_error: str | None, *, phase: str) -> None:
    dist, rank, world = _distributed_context()
    if world > 1:
        gathered: list[Any] = [None] * world
        dist.all_gather_object(gathered, (rank, local_error))
        failures = [item for item in gathered if item[1] is not None]
    else:
        failures = [(rank, local_error)] if local_error else []
    if failures:
        details = "; ".join(f"rank={item[0]}: {item[1]}" for item in failures)
        raise BackendContractError(f"Wan Stage2 {phase} failed collectively: {details}")


def _broadcast_object(value: Any) -> Any:
    dist, _, world = _distributed_context()
    payload = [value]
    if world > 1:
        dist.broadcast_object_list(payload, src=0)
    return payload[0]


def verify_role_checkpoint(
    initialization: RoleInitialization,
) -> RoleCheckpointReceipt:
    """Verify one role checkpoint path and byte size on rank zero."""

    _, rank, _ = _distributed_context()
    result: dict[str, Any] = {}
    error = ""
    if rank == 0:
        try:
            path = Path(initialization.path).expanduser().resolve()
            if not path.is_file():
                raise BackendContractError(f"Stage2 {initialization.role} checkpoint is missing")
            object_stat = path.stat()
            if object_stat.st_size <= 0:
                raise BackendContractError(f"Stage2 {initialization.role} checkpoint is empty")
            result = {
                "role": initialization.role,
                "path": str(path),
                "object_bytes": int(object_stat.st_size),
                "step": None,
                "weights": initialization.weights,
                "stage": initialization.expected_stage,
                "objective": initialization.expected_objective,
                "camera_translation_transform": initialization.camera_translation_transform,
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    error = str(_broadcast_object(error))
    result = dict(_broadcast_object(result))
    if error:
        raise BackendContractError(
            f"Stage2 {initialization.role} identity verification failed: {error}"
        )
    receipt = RoleCheckpointReceipt(
        role=str(result["role"]),
        path=Path(str(result["path"])),
        object_bytes=int(result["object_bytes"]),
        step=int(result["step"]) if result["step"] is not None else None,
        weights=str(result["weights"]),
        stage=str(result["stage"]),
        objective=str(result["objective"]),
        camera_translation_transform=str(result["camera_translation_transform"]),
    )
    visible = receipt.path.is_file()
    _collective_error(
        None if visible else f"checkpoint is not rank-visible: {receipt.path}",
        phase=f"{initialization.role} visibility",
    )
    return receipt


def _checkpoint_metadata(payload: Mapping[str, Any]) -> tuple[str, bool, str, str, str]:
    config = payload.get("config", {})
    if not isinstance(config, Mapping):
        raise BackendContractError("Stage2 role checkpoint has no config mapping")
    model = config.get("model", {})
    train = config.get("train", {})
    metadata = config.get("metadata", {})
    if not isinstance(model, Mapping) or not isinstance(train, Mapping):
        raise BackendContractError("Stage2 role checkpoint model/train metadata is invalid")
    family = str(model.get("family", model.get("model_family", "")))
    causal = bool(model.get("causal", model.get("is_causal", False)))
    metadata_stage = metadata.get("stage", "") if isinstance(metadata, Mapping) else ""
    stage = str(train.get("stage", metadata_stage))
    objective = str(train.get("objective", train.get("flow_objective", "flow_matching")))
    variant = str(train.get("objective_variant", train.get("anyflow_variant", "")))
    if objective == "anyflow_forward_map" and variant:
        objective = f"{objective}:{variant}"
    camera = str(model.get("camera_translation_transform", "linear"))
    return family, causal, stage, objective, camera


def _load_module_state(
    module: Any,
    state: Mapping[str, Any],
    *,
    role: str,
    allow_anyflow_delta_drop: bool,
) -> tuple[str, ...]:
    """Strict-load one role, optionally dropping exactly four AnyFlow tensors."""

    from .checkpoint import _load_model_state_result

    normalized = normalize_model_state(state, field=role)
    ignored: tuple[str, ...] = ()
    if allow_anyflow_delta_drop:
        ignored = tuple(sorted(key for key in normalized if "delta_embedding" in key.split(".")))
        if len(ignored) != 4:
            raise BackendContractError(
                "Stage2 student conversion must drop exactly four AnyFlow delta tensors"
            )
        normalized = OrderedDict(
            (key, value) for key, value in normalized.items() if key not in ignored
        )
    result = _load_model_state_result(module, normalized, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise BackendContractError(f"Stage2 {role} state load was not exact")
    return ignored


def load_role_checkpoint(
    *,
    initialization: RoleInitialization,
    receipt: RoleCheckpointReceipt,
    diffusion: WanDiffusion,
) -> tuple[tuple[str, ...], RoleCheckpointReceipt]:
    """Load the explicitly selected LIVE/EMA state for one Stage2 role."""

    import torch

    payload: Mapping[str, Any] | None = None
    source_step: int | None = None
    verified_metadata: tuple[str, str, str] | None = None
    error: str | None = None
    try:
        loaded = torch.load(
            receipt.path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(loaded, Mapping):
            raise BackendContractError("Stage2 role checkpoint payload must be a mapping")
        try:
            source_step = int(loaded["global_step"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError(
                f"Stage2 {initialization.role} checkpoint has no valid global_step"
            ) from exc
        if source_step <= 0:
            raise BackendContractError(
                f"Stage2 {initialization.role} checkpoint global_step must be positive"
            )
        family, causal, stage, objective, camera = _checkpoint_metadata(loaded)
        expected_causal = initialization.role == "student"
        expected = (
            "wan22_ti2v_5b",
            expected_causal,
            initialization.expected_stage,
            initialization.expected_objective,
            initialization.camera_translation_transform,
        )
        actual = (family, causal, stage, objective, camera)
        if actual != expected:
            raise BackendContractError(
                f"Stage2 {initialization.role} checkpoint metadata differs: "
                f"actual={actual} expected={expected}"
            )
        verified_metadata = (stage, objective, camera)
        field = "generator_ema" if initialization.weights == "ema" else "generator"
        state = loaded.get(field)
        if not isinstance(state, Mapping) or not state:
            raise BackendContractError(f"Stage2 {initialization.role} checkpoint has no {field}")
        payload = state
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _collective_error(error, phase=f"{initialization.role} role preload")
    assert payload is not None
    assert source_step is not None
    assert verified_metadata is not None
    ignored = _load_module_state(
        diffusion.module,
        payload,
        role=initialization.role,
        allow_anyflow_delta_drop=initialization.allow_anyflow_delta_drop,
    )
    stage, objective, camera = verified_metadata
    return ignored, RoleCheckpointReceipt(
        role=receipt.role,
        path=receipt.path,
        object_bytes=receipt.object_bytes,
        step=source_step,
        weights=receipt.weights,
        stage=stage,
        objective=objective,
        camera_translation_transform=camera,
    )


def _slice_camera(
    camera: Mapping[str, Any],
    *,
    start_frame: int,
    end_frame: int,
    frame_sequence_length: int,
) -> dict[str, Any]:
    start = int(start_frame) * int(frame_sequence_length)
    end = int(end_frame) * int(frame_sequence_length)
    return {
        "viewmats": camera["viewmats"][:, start:end],
        "K": camera["K"][:, start:end],
    }


def _restore_first(value: Any, first: Any, *, start_frame: int) -> Any:
    if int(start_frame) != 0:
        return value
    result = value.clone()
    result[:, :1] = first
    return result


def _sample_exit_index(
    count: int,
    *,
    device: Any,
    per_rank: bool,
    last_step_only: bool,
    forced_exit_index: int | None,
) -> int:
    import torch
    import torch.distributed as dist

    if count < 1:
        raise BackendContractError("Stage2 denoising schedule is empty")
    if forced_exit_index is not None:
        index = int(forced_exit_index)
    elif last_step_only:
        index = count - 1
    else:
        value = torch.randint(0, count, (1,), device=device, dtype=torch.long)
        if not per_rank and dist.is_available() and dist.is_initialized():
            dist.broadcast(value, src=0)
        index = int(value.item())
    if not 0 <= index < count:
        raise BackendContractError("Stage2 exit index is outside the denoising schedule")
    return index


def stage2_camera_rollout(
    *,
    student: WanDiffusion,
    allocate_kv_cache: Callable[..., list[dict[str, Any]]],
    allocate_crossattn_cache: Callable[..., list[dict[str, Any]]],
    noise: Any,
    first_latent: Any,
    condition: Mapping[str, Any],
    camera: Mapping[str, Any],
    denoising_steps: Sequence[Any],
    num_frame_per_block: int,
    frame_sequence_length: int,
    context_timestep: float = 0.0,
    per_rank_exit_step: bool = True,
    match_context: bool = True,
    last_step_only: bool = False,
    require_grad: bool = True,
    validate_finite: bool = False,
    forced_exit_index: int | None = None,
) -> Stage2Rollout:
    """Run the no-grad chunk pass followed by one gradient-carrying replay."""

    import torch

    if noise.ndim != 5:
        raise BackendContractError("Stage2 noise must have shape [B,T,C,H,W]")
    batch_size, num_frames = noise.shape[:2]
    block = int(num_frame_per_block)
    if block != 3 or num_frames % block:
        raise BackendContractError("Stage2 requires a three-latent chunk geometry")
    if tuple(first_latent.shape) != (batch_size, 1, *noise.shape[2:]):
        raise BackendContractError("Stage2 first-latent anchor shape is invalid")
    tokens = num_frames * int(frame_sequence_length)
    if set(camera) != {"viewmats", "K"}:
        raise BackendContractError("Stage2 camera must contain exactly viewmats and K")
    if tuple(camera["viewmats"].shape) != (batch_size, tokens, 4, 4):
        raise BackendContractError("Stage2 camera viewmats shape is invalid")
    if tuple(camera["K"].shape) != (batch_size, tokens, 3, 3):
        raise BackendContractError("Stage2 camera intrinsics shape is invalid")
    steps = tuple(
        torch.as_tensor(value, device=noise.device, dtype=torch.float32).reshape(())
        for value in denoising_steps
    )
    exit_index = _sample_exit_index(
        len(steps),
        device=noise.device,
        per_rank=bool(per_rank_exit_step),
        last_step_only=bool(last_step_only),
        forced_exit_index=forced_exit_index,
    )
    train_timestep = steps[exit_index]

    def require_finite(name: str, value: Any) -> None:
        if validate_finite and not bool(torch.isfinite(value).all().item()):
            raise BackendContractError(f"non-finite Stage2 rollout tensor {name}")

    kv_cache = allocate_kv_cache(batch_size, dtype=noise.dtype, device=noise.device)
    crossattn_cache = allocate_crossattn_cache(batch_size, dtype=noise.dtype, device=noise.device)
    if kv_cache and "_fused_prope_camera_metadata" not in kv_cache[0]:
        raise BackendContractError(
            "Stage2 fused-PRoPE camera metadata must be preallocated before FSDP"
        )
    cache_target = torch.zeros_like(noise)
    matched_context = torch.zeros_like(noise)
    noisy_at_exit = torch.zeros_like(noise)
    with torch.no_grad():
        for start in range(0, num_frames, block):
            end = start + block
            current = _restore_first(noise[:, start:end], first_latent, start_frame=start)
            camera_chunk = _slice_camera(
                camera,
                start_frame=start,
                end_frame=end,
                frame_sequence_length=frame_sequence_length,
            )
            x0_exit = None
            for step_index, step in enumerate(steps):
                timestep = torch.full(
                    (batch_size, block),
                    float(step.item()),
                    device=noise.device,
                    dtype=torch.float32,
                )
                if start == 0:
                    timestep[:, 0] = 0.0
                if step_index == exit_index:
                    noisy_at_exit[:, start:end] = current
                flow = student(
                    current,
                    condition,
                    camera_chunk,
                    expand_timesteps_to_tokens(timestep, frame_sequence_length),
                    sequence_length=block * int(frame_sequence_length),
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=start * int(frame_sequence_length),
                    cache_start=0,
                    cache_update_policy="none",
                )
                x0 = student.flow_to_x0(current, flow, timestep)
                x0 = _restore_first(x0, first_latent, start_frame=start)
                require_finite(f"pass1.chunk{start // block}.step{step_index}", x0)
                if step_index == exit_index:
                    x0_exit = x0
                if step_index < len(steps) - 1:
                    next_timestep = torch.full(
                        (batch_size, block),
                        float(steps[step_index + 1].item()),
                        device=noise.device,
                        dtype=torch.float32,
                    )
                    if start == 0:
                        next_timestep[:, 0] = 0.0
                    current = (
                        student.scheduler.add_noise(
                            x0.flatten(0, 1).float(),
                            torch.randn_like(x0).flatten(0, 1).float(),
                            next_timestep.flatten(0, 1),
                        )
                        .unflatten(0, (batch_size, block))
                        .to(noise.dtype)
                    )
                    current = _restore_first(current, first_latent, start_frame=start)
            if x0_exit is None:
                raise BackendContractError("Stage2 rollout failed to capture the exit x0")
            committed = x0_exit.detach()
            cache_target[:, start:end] = committed
            context_t = torch.full(
                (batch_size, block),
                float(context_timestep),
                device=noise.device,
                dtype=torch.float32,
            )
            if start == 0:
                context_t[:, 0] = 0.0
            if float(context_timestep) == 0.0:
                context = committed.clone()
            else:
                context = (
                    student.scheduler.add_noise(
                        committed.flatten(0, 1).float(),
                        torch.randn_like(committed).flatten(0, 1).float(),
                        context_t.flatten(0, 1),
                    )
                    .unflatten(0, (batch_size, block))
                    .to(noise.dtype)
                )
            context = _restore_first(context, first_latent, start_frame=start)
            matched_context[:, start:end] = context
            student(
                context,
                condition,
                camera_chunk,
                expand_timesteps_to_tokens(context_t, frame_sequence_length),
                sequence_length=block * int(frame_sequence_length),
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=start * int(frame_sequence_length),
                cache_start=0,
                cache_update_policy="commit_detached",
            )
    pass2_clean = matched_context.detach() if match_context else cache_target.detach()
    pass2_timestep = torch.full(
        (batch_size, num_frames),
        float(train_timestep.item()),
        device=noise.device,
        dtype=torch.float32,
    )
    pass2_aug = torch.full_like(
        pass2_timestep,
        float(context_timestep if match_context else 0.0),
    )
    pass2_timestep[:, 0] = 0.0
    pass2_aug[:, 0] = 0.0
    with torch.set_grad_enabled(bool(require_grad)):
        replay_flow = student.forward_train_tf(
            noisy_at_exit,
            pass2_clean,
            condition,
            camera,
            expand_timesteps_to_tokens(pass2_timestep, frame_sequence_length),
            expand_timesteps_to_tokens(pass2_aug, frame_sequence_length),
            num_frame_per_block=block,
            sequence_length=num_frames * int(frame_sequence_length),
        )
        output = student.flow_to_x0(noisy_at_exit, replay_flow, pass2_timestep)
    output = _restore_first(output, first_latent, start_frame=0)
    require_finite("pass2.output", output)
    loss_mask = torch.ones(
        (batch_size, num_frames, 1, 1, 1),
        device=noise.device,
        dtype=torch.bool,
    )
    loss_mask[:, 0] = False
    return Stage2Rollout(
        output=output,
        cache_target=cache_target,
        noisy_at_exit=noisy_at_exit,
        loss_mask=loss_mask,
        exit_index=exit_index,
        denoised_timestep_from=float(train_timestep.item()),
        denoised_timestep_to=(
            0.0 if exit_index == len(steps) - 1 else float(steps[exit_index + 1].item())
        ),
    )


def _generation_steps(provider: Any) -> tuple[Any, ...]:
    configured = getattr(provider, "denoising_steps", None)
    if configured is not None:
        return tuple(configured)
    train = provider.config["train"]
    return warp_denoising_steps(
        train["denoising_step_list"],
        provider.diffusion.scheduler.timesteps.to(provider.device),
        num_train_timesteps=int(train["num_train_timesteps"]),
    )


def _stage2_self_forcing_latents(
    provider: Any,
    generation_pass: Any,
    first_latent: Any,
    condition: Mapping[str, Any],
    camera: Mapping[str, Any],
    generator: Any,
) -> tuple[Any, Mapping[str, Any]]:
    """Run the NFE4 sampler over one persistent rolling six-chunk cache."""

    import torch

    if (
        str(generation_pass.solver) != "self_forcing"
        or int(generation_pass.num_inference_steps) != 4
    ):
        raise BackendContractError("Stage2 generation supports self_forcing NFE4 only")
    latent_frames = int(generation_pass.rollout_latent_frames)
    chunk = int(provider.config["model"]["num_frame_per_block"])
    if chunk != 3 or latent_frames % chunk:
        raise BackendContractError("Stage2 generation horizon must divide into three-latent chunks")
    channels = int(provider.config["model"]["latent_channels"])
    latent_height = int(provider.config["data"]["latent_shape"][-2])
    latent_width = int(provider.config["data"]["latent_shape"][-1])
    frame_tokens = int(provider.config["model"]["frame_sequence_length"])
    output = torch.zeros(
        1,
        latent_frames,
        channels,
        latent_height,
        latent_width,
        device=provider.device,
        dtype=torch.bfloat16,
    )
    output[:, 0] = first_latent[:, 0]
    initial_noise = provider._noise(tuple(output.shape), generator)
    initial_noise[:, 0] = first_latent[:, 0]
    steps = _generation_steps(provider)
    if len(steps) != 4:
        raise BackendContractError("Stage2 shifted generation schedule must have four steps")
    kv_cache = provider.allocate_kv_cache(1, dtype=output.dtype, device=provider.device)
    crossattn_cache = provider.allocate_crossattn_cache(
        1, dtype=output.dtype, device=provider.device
    )
    if kv_cache and "_fused_prope_camera_metadata" not in kv_cache[0]:
        raise BackendContractError("Stage2 generation requires preallocated fused camera metadata")
    try:
        for start in range(0, latent_frames, chunk):
            end = start + chunk
            latents = initial_noise[:, start:end].clone()
            camera_chunk = _slice_camera(
                camera,
                start_frame=start,
                end_frame=end,
                frame_sequence_length=frame_tokens,
            )
            for step_index, step in enumerate(steps):
                timestep = torch.full(
                    (1, chunk),
                    float(step.item()),
                    device=provider.device,
                    dtype=output.dtype,
                )
                if start == 0:
                    timestep[:, 0] = 0.0
                with torch.autocast(
                    device_type=provider.device.type,
                    dtype=torch.bfloat16,
                    enabled=provider.device.type == "cuda",
                ):
                    flow = provider.diffusion(
                        latents,
                        condition,
                        camera_chunk,
                        expand_timesteps_to_tokens(timestep, frame_tokens),
                        sequence_length=chunk * frame_tokens,
                        kv_cache=kv_cache,
                        crossattn_cache=crossattn_cache,
                        current_start=start * frame_tokens,
                        cache_start=0,
                        cache_update_policy="none",
                    )
                    x0 = provider.diffusion.flow_to_x0(latents, flow, timestep)
                if start == 0:
                    x0 = _restore_first(x0, first_latent, start_frame=0)
                if step_index + 1 < len(steps):
                    next_timestep = torch.full(
                        (1, chunk),
                        float(steps[step_index + 1].item()),
                        device=provider.device,
                        dtype=torch.float32,
                    )
                    if start == 0:
                        next_timestep[:, 0] = 0.0
                    renoise = provider._noise(tuple(x0.shape), generator)
                    latents = (
                        provider.diffusion.scheduler.add_noise(
                            x0.flatten(0, 1).float(),
                            renoise.flatten(0, 1).float(),
                            next_timestep.flatten(0, 1),
                        )
                        .unflatten(0, (1, chunk))
                        .to(output.dtype)
                    )
                    if start == 0:
                        latents = _restore_first(latents, first_latent, start_frame=0)
                else:
                    latents = x0
            if not bool(torch.isfinite(latents).all().item()):
                raise BackendContractError(
                    f"Stage2 generation chunk {start // chunk} is non-finite"
                )
            output[:, start:end] = latents
            commit_timestep = torch.zeros((1, chunk), device=provider.device, dtype=output.dtype)
            with (
                torch.no_grad(),
                torch.autocast(
                    device_type=provider.device.type,
                    dtype=torch.bfloat16,
                    enabled=provider.device.type == "cuda",
                ),
            ):
                provider.diffusion(
                    latents,
                    condition,
                    camera_chunk,
                    expand_timesteps_to_tokens(commit_timestep, frame_tokens),
                    sequence_length=chunk * frame_tokens,
                    kv_cache=kv_cache,
                    crossattn_cache=crossattn_cache,
                    current_start=start * frame_tokens,
                    cache_start=0,
                    cache_update_policy="commit_detached",
                )
    finally:
        del kv_cache, crossattn_cache
    return output, {
        "schema": "solarwm.wan22-stage2-self-forcing-schedule.v1",
        "solver": "self_forcing",
        "timesteps": [float(value.item()) for value in steps],
        "chunk_latent_frames": chunk,
        "persistent_kv_cache": True,
        "cache_update_policy": "commit_detached",
        "camera_attention_mode": "fused_prope",
    }


def _stage2_generated_sample(
    provider: Any,
    case: Any,
    *,
    weights_id: str,
) -> Any:
    """Materialize one Stage2 sample for either validation or standalone infer."""

    import torch

    from solarwm.config.loader import canonical_json
    from solarwm.inference import GeneratedSample

    from ..generation import GenerationPass
    from .inference import _encode_compare_mp4, _encode_mp4, _encode_stage2_streaming

    metadata = case.metadata.get("generation_pass")
    if not isinstance(metadata, Mapping):
        raise BackendContractError("Stage2 case lacks generation_pass metadata")
    generation_pass = GenerationPass(
        name=str(metadata["name"]),
        weights=str(metadata["weights"]),
        mode=str(metadata["mode"]),
        solver=str(metadata["solver"]),
        num_inference_steps=int(metadata["num_inference_steps"]),
        rollout_latent_frames=int(metadata["rollout_latent_frames"]),
        min_rollout_latent_frames=int(
            metadata.get(
                "min_rollout_latent_frames",
                metadata["rollout_latent_frames"],
            )
        ),
        fixed_plan_pixel_frames=int(
            metadata.get(
                "fixed_plan_pixel_frames",
                1 + 4 * (int(metadata["rollout_latent_frames"]) - 1),
            )
        ),
        variable_rollout_by_source=bool(metadata.get("variable_rollout_by_source", False)),
    )
    if generation_pass.mode != "autoregressive":
        raise BackendContractError("Stage2 self forcing requires autoregressive mode")
    rng = torch.Generator(device=provider.device)
    rng.manual_seed(int(case.noise_seed))
    first, condition, camera, model_y = provider._conditions(
        case, latent_frames=generation_pass.rollout_latent_frames
    )
    if model_y is not None:
        raise BackendContractError("Stage2 TI2V generation may not receive I2V y")
    with torch.no_grad():
        latents, schedule = _stage2_self_forcing_latents(
            provider,
            generation_pass,
            first,
            condition,
            camera,
            rng,
        )
        output_latent_frames = int(
            metadata.get(
                "output_rollout_latent_frames",
                generation_pass.rollout_latent_frames,
            )
        )
        output_latents = latents[:, :output_latent_frames].contiguous()
        prepared = provider._prepared.get(case.slot)
        if prepared is None:
            raise BackendContractError(
                f"Stage2 adapter has no comparison input for slot {case.slot}"
            )
        model_output_pixel_frames = 1 + 4 * (output_latent_frames - 1)
        publication_pixel_frames = getattr(prepared, "publication_pixel_frames", None)
        target_pixel_frames = (
            int(publication_pixel_frames)
            if publication_pixel_frames is not None
            else model_output_pixel_frames
        )
        if (
            publication_pixel_frames is not None
            and not bool(getattr(prepared, "repeat_first_frame", False))
            and int(prepared.pixels.shape[0]) != target_pixel_frames
        ):
            raise BackendContractError(
                "camera publication GT length differs from the requested output length"
            )

        camera_artifact: bytes | None = None
        publication_provenance: dict[str, Any] = {}
        if publication_pixel_frames is not None:
            c2w = getattr(prepared, "publication_c2w", None)
            if (
                c2w is None
                or c2w.shape != (target_pixel_frames, 4, 4)
                or c2w.dtype != np.float64
                or not np.isfinite(c2w).all()
            ):
                raise BackendContractError(
                    "camera publication requires finite FP64 absolute C2W for every output frame"
                )
            buffer = io.BytesIO()
            np.save(buffer, c2w, allow_pickle=False)
            camera_artifact = buffer.getvalue()
            publication_provenance = {
                "camera_convention": "authoritative_absolute_c2w",
                "model_output_pixel_frames": model_output_pixel_frames,
                "published_pixel_frames": target_pixel_frames,
                "trim_frames": max(model_output_pixel_frames - target_pixel_frames, 0),
                "tail_pad_frames": max(target_pixel_frames - model_output_pixel_frames, 0),
                "tail_padding": (
                    "repeat_last_generated_frame"
                    if target_pixel_frames > model_output_pixel_frames
                    else "none"
                ),
            }

        encoder = getattr(provider, "video_encoder", None)
        if target_pixel_frames > _FILE_STREAMING_PIXEL_THRESHOLD:
            if callable(encoder):
                raise BackendContractError(
                    "hour-scale Stage2 generation requires the released streaming encoder"
                )
            runtime_output = Path(str(provider.config["runtime"]["output_dir"])).expanduser()
            streamed = _encode_stage2_streaming(
                provider.vae,
                output_latents,
                prepared,
                target_pixel_frames=target_pixel_frames,
                fps=float(provider.config["data"].get("fps", 16.0)),
                temporary_root=runtime_output.resolve().parent,
                chunk_latent_frames=_FILE_STREAMING_VAE_LATENT_CHUNK,
            )
            artifacts: dict[str, Any] = {
                "compare.mp4": streamed.compare,
                "video.mp4": streamed.video,
                "schedule.json": canonical_json(dict(schedule)),
            }
            shape = streamed.shape
            finite_fraction = 1.0
            vae_decode = {
                "mode": "continuous_cached_file_tiles",
                "chunk_latent_frames": _FILE_STREAMING_VAE_LATENT_CHUNK,
            }
        else:
            if output_latent_frames > _STREAMING_VAE_LATENT_CHUNK:
                decoded = provider.vae.decode_streaming(
                    output_latents,
                    chunk_latent_frames=_STREAMING_VAE_LATENT_CHUNK,
                )
                vae_decode = {
                    "mode": "continuous_cached_tiles",
                    "chunk_latent_frames": _STREAMING_VAE_LATENT_CHUNK,
                }
            else:
                decoded = provider.vae.decode(output_latents, use_cache=False)
                vae_decode = {
                    "mode": "direct",
                    "chunk_latent_frames": output_latent_frames,
                }
            if int(decoded.shape[1]) != model_output_pixel_frames:
                raise BackendContractError(
                    "Stage2 VAE frame count differs from latent-aligned output: "
                    f"{int(decoded.shape[1])} != {model_output_pixel_frames}"
                )
            finite_fraction = float(torch.isfinite(decoded).float().mean().item())
            if finite_fraction != 1.0:
                raise BackendContractError("Stage2 VAE decode produced non-finite pixels")
            trim_frames = max(int(decoded.shape[1]) - target_pixel_frames, 0)
            tail_pad_frames = max(target_pixel_frames - int(decoded.shape[1]), 0)
            if trim_frames:
                decoded = decoded[:, :target_pixel_frames].contiguous()
            elif tail_pad_frames:
                tail = decoded[:, -1:].expand(
                    -1,
                    tail_pad_frames,
                    -1,
                    -1,
                    -1,
                )
                decoded = torch.cat((decoded, tail), dim=1).contiguous()
            video = (
                encoder(decoded, fps=float(provider.config["data"].get("fps", 16.0)))
                if callable(encoder)
                else _encode_mp4(decoded, fps=float(provider.config["data"].get("fps", 16.0)))
            )
            compare = _encode_compare_mp4(
                decoded,
                prepared,
                fps=float(provider.config["data"].get("fps", 16.0)),
            )
            artifacts = {
                "compare.mp4": compare,
                "video.mp4": video,
                "schedule.json": canonical_json(dict(schedule)),
            }
            shape = tuple(int(value) for value in decoded.shape)

    if camera_artifact is not None:
        artifacts["camera.npy"] = camera_artifact
    return GeneratedSample(
        artifacts=artifacts,
        shape=shape,
        dtype="float32",
        metrics={"finite_fraction": finite_fraction},
        provenance={
            "generation_pass": asdict(generation_pass),
            "weights_id": str(weights_id),
            "denoising_step_list": [
                int(value) for value in provider.config["train"]["denoising_step_list"]
            ],
            "camera_translation_transform": str(
                provider.config["model"]["camera_translation_transform"]
            ),
            "resolved_weights_role": str(
                getattr(provider, "_model_weight_role", generation_pass.weights)
            ),
            "vae_decode": vae_decode,
            **publication_provenance,
        },
    )


def stage2_checkpoint_contract(
    config: Mapping[str, Any],
    receipts: Mapping[str, RoleCheckpointReceipt],
) -> CheckpointContract:
    """Bind all Stage2 algorithm, role, attention, and data semantics."""

    model = config["model"]
    data = config["data"]
    train = config["train"]
    distributed = config["distributed"]
    roles: dict[str, dict[str, Any]] = {}
    for name in ("student", "teacher", "critic"):
        receipt = receipts[name]
        if receipt.step is None or receipt.step <= 0:
            raise BackendContractError(f"Stage2 {name} receipt has no verified source step")
        roles[name] = {
            "weights": receipt.weights,
            "step": int(receipt.step),
            "stage": receipt.stage,
            "objective": receipt.objective,
            "camera_translation_transform": receipt.camera_translation_transform,
        }
    optimizer = train["optimizer"]
    critic_optimizer = train["critic_optimizer"]
    ema = train["ema"]
    return CheckpointContract(
        family=str(model["family"]),
        stage="stage2",
        causal_mode="self_gradient_forcing",
        objective="flow_matching",
        objective_variant="sgf_v1",
        camera_translation_transform=str(model["camera_translation_transform"]),
        parameterization="full-parameter-three-role",
        sp_size=int(distributed["sequence_parallel_size"]),
        data_generation=(
            f"{data['encoding']}:online:{data['pixel_frames']}f:"
            f"{data['height']}x{data['width']}:{data['train_index']}"
        ),
        extras={
            "roles": roles,
            "denoising_step_list": list(train["denoising_step_list"]),
            "critic_updates_per_student": int(train["critic_updates_per_student"]),
            "score_timestep_bounds": [
                float(train["score_min_timestep"]),
                float(train["score_max_timestep"]),
            ],
            "context_timestep": float(train["context_timestep"]),
            "per_rank_exit_step": bool(train["per_rank_exit_step"]),
            "match_context": bool(train["self_gradient_forcing_match_context"]),
            "cache_mode": str(train["self_gradient_forcing_cache_mode"]),
            "last_step_only": bool(train["last_step_only"]),
            "real_guidance_scale": float(train["real_guidance_scale"]),
            "fake_guidance_scale": float(train["fake_guidance_scale"]),
            "negative_prompt": str(train["negative_prompt"]),
            "attention": {
                "local_attn_size": int(model["local_attn_size"]),
                "score_local_attn_size": int(model["score_local_attn_size"]),
                "max_prior_clean_chunks": int(model["max_prior_clean_chunks"]),
                "sink_size": int(model["sink_size"]),
                "rope_train_frames": int(model["rope_train_frames"]),
                "use_echorope": bool(model["use_echorope"]),
                "score_use_echorope": bool(model["score_use_echorope"]),
            },
            "world_size": int(distributed["world_size"]),
            "global_batch_size": int(train["global_batch_size"]),
            "max_steps": int(train["max_steps"]),
            "student_optimizer": {
                "lr": float(optimizer["lr"]),
                "betas": [float(value) for value in optimizer["betas"]],
                "eps": float(optimizer["eps"]),
                "weight_decay": float(optimizer["weight_decay"]),
                "warmup_steps": int(optimizer.get("warmup_steps", 0)),
                "min_lr_ratio": float(optimizer.get("min_lr_ratio", 1.0)),
            },
            "critic_optimizer": {
                "lr": float(critic_optimizer["lr"]),
                "betas": [float(value) for value in critic_optimizer["betas"]],
                "eps": float(critic_optimizer["eps"]),
                "weight_decay": float(critic_optimizer["weight_decay"]),
                "warmup_steps": int(critic_optimizer.get("warmup_steps", 0)),
                "min_lr_ratio": float(critic_optimizer.get("min_lr_ratio", 1.0)),
            },
            "ema": {
                "decay": float(ema["decay"]),
                "start_step": int(ema["start_step"]),
                "update_every": int(ema.get("update_every", 1)),
            },
        },
    )


def _verified_stage2_inference_checkpoint(
    config: Mapping[str, Any],
    source: str | Path,
) -> tuple[Path, str, int]:
    """Verify the portable Stage2 inference transaction before allocation."""

    requested = Path(source).expanduser().resolve()
    if requested.is_dir():
        root = requested
        requested = root / "model.pt"
    else:
        root = requested.parent
    if requested.name != "model.pt":
        raise BackendContractError(
            "Stage2 inference checkpoint must name model.pt or its transaction directory"
        )
    try:
        verified = verify_checkpoint(root)
    except Exception as exc:
        raise BackendContractError(
            f"Stage2 inference checkpoint verification failed: {type(exc).__name__}: {exc}"
        ) from exc
    members = {record.path for record in verified.files}
    if members != {"model.pt"}:
        raise BackendContractError("Stage2 inference checkpoint must contain exactly model.pt")
    metadata = verified.metadata
    if (
        metadata.get("schema") != "solarwm.wan22-stage2-inference.v1"
        or metadata.get("available_weights") != ["live", "ema"]
        or metadata.get("ema_present") is not True
    ):
        raise BackendContractError("Stage2 inference checkpoint metadata differs")
    contract = verified.contract
    model = config["model"]
    train = config["train"]
    expected_header = {
        "family": str(model["family"]),
        "stage": "stage2",
        "causal_mode": "self_gradient_forcing",
        "objective": "flow_matching",
        "objective_variant": "sgf_v1",
        "camera_translation_transform": str(model["camera_translation_transform"]),
        "parameterization": "full-parameter-live-ema",
        "sp_size": 1,
        "data_generation": "inference-portable:v1",
    }
    header_drift = {
        name: {"actual": getattr(contract, name), "expected": expected}
        for name, expected in expected_header.items()
        if getattr(contract, name) != expected
    }
    extras = contract.extras
    expected_extras = {
        "denoising_step_list": list(train["denoising_step_list"]),
        "attention": {
            "local_attn_size": int(model["local_attn_size"]),
            "max_prior_clean_chunks": int(model["max_prior_clean_chunks"]),
            "sink_size": int(model["sink_size"]),
            "rope_train_frames": int(model["rope_train_frames"]),
            "use_echorope": bool(model["use_echorope"]),
        },
    }
    extras_drift = {
        name: {"actual": extras.get(name), "expected": expected}
        for name, expected in expected_extras.items()
        if extras.get(name) != expected
    }
    if header_drift or extras_drift:
        raise BackendContractError(
            "Stage2 inference checkpoint contract differs: "
            f"header={header_drift} extras={extras_drift}"
        )
    if not requested.is_file():
        raise BackendContractError("Stage2 verified transaction has no visible model.pt")
    return requested, verified.manifest_digest, int(verified.step)


def _published_default_weight_role(source: str | Path) -> str:
    """Resolve the release-selected live/EMA role without exposing a CLI choice."""

    requested = Path(source).expanduser().resolve()
    root = requested if requested.is_dir() else requested.parent
    manifest_path = root / "release-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise BackendContractError(
            f"Stage2 camera-length checkpoint lacks a readable release manifest: {exc}"
        ) from exc
    if not isinstance(manifest, Mapping):
        raise BackendContractError("Stage2 release manifest must be a mapping")
    load = manifest.get("load")
    identity = manifest.get("identity")
    model = identity.get("model") if isinstance(identity, Mapping) else None
    if (
        manifest.get("schema") != "solarwm.public-weight-manifest.v1"
        or not isinstance(load, Mapping)
        or load.get("format") != "solarwm_wan_stage2_transaction_v1"
        or load.get("entrypoint") != "."
        or not isinstance(model, Mapping)
    ):
        raise BackendContractError("Stage2 release manifest contract differs")
    role = str(load.get("default_weights", "")).strip().lower()
    available = str(model.get("weight_role", "")).strip().lower().split("+")
    if role not in {"live", "ema"} or role not in available:
        raise BackendContractError("Stage2 release default weight role is invalid")
    return role


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _prefixed_state(state: Mapping[str, Any]) -> OrderedDict[str, Any]:
    result = OrderedDict((f"model.{key}", value) for key, value in state.items())
    if len(result) != len(state):
        raise BackendContractError("Stage2 model prefixing produced duplicate keys")
    return result


def _local_rng_state(rank: int, device: Any, reader: Any) -> dict[str, Any]:
    import torch

    from solarwm.runtime.safe_state import (
        encode_numpy_rng_state,
        encode_python_rng_state,
    )

    state_dict = getattr(reader, "checkpoint_state_dict", None)
    if not callable(state_dict):
        state_dict = getattr(reader, "state_dict", None)
    if not callable(state_dict):
        raise BackendContractError("Stage2 checkpoint requires a stateful data reader")
    return {
        "schema": "solarwm.wan22.stage2-rank-runtime.v1",
        "rank": int(rank),
        "reader": state_dict(),
        "python": encode_python_rng_state(random.getstate()),
        "numpy": encode_numpy_rng_state(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": (
            torch.cuda.get_rng_state(device).cpu() if torch.device(device).type == "cuda" else None
        ),
    }


def _gather_rng_state(rank: int, device: Any, reader: Any) -> list[dict[str, Any]]:
    local = _local_rng_state(rank, device, reader)
    dist, _, world = _distributed_context()
    if world == 1:
        return [local]
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, local)
    return list(gathered)


def _restore_rng_state(states: Any, *, rank: int, device: Any, reader: Any) -> None:
    import torch

    from solarwm.runtime.safe_state import (
        decode_numpy_rng_state,
        decode_python_rng_state,
    )

    if not isinstance(states, list):
        raise BackendContractError("Stage2 checkpoint lacks per-rank RNG state")
    selected = next(
        (
            state
            for state in states
            if isinstance(state, Mapping) and int(state.get("rank", -1)) == int(rank)
        ),
        None,
    )
    if selected is None:
        raise BackendContractError(f"Stage2 checkpoint has no RNG state for rank {rank}")
    if selected.get("schema") != "solarwm.wan22.stage2-rank-runtime.v1":
        raise BackendContractError("Stage2 checkpoint lacks exact rank-local runtime state")
    load_state_dict = getattr(reader, "load_state_dict", None)
    if not callable(load_state_dict):
        raise BackendContractError("Stage2 resume requires a stateful data reader")
    load_state_dict(selected.get("reader", {}))
    random.setstate(decode_python_rng_state(selected.get("python")))
    np.random.set_state(decode_numpy_rng_state(selected.get("numpy")))
    cpu_state = selected.get("torch_cpu")
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
        or cpu_state.numel() == 0
    ):
        raise BackendContractError("Stage2 checkpoint CPU RNG state is invalid")
    torch.set_rng_state(cpu_state)
    if torch.device(device).type == "cuda":
        cuda_state = selected.get("torch_cuda")
        if (
            not isinstance(cuda_state, torch.Tensor)
            or cuda_state.dtype != torch.uint8
            or cuda_state.ndim != 1
            or cuda_state.numel() == 0
        ):
            raise BackendContractError("Stage2 checkpoint CUDA RNG state is invalid")
        torch.cuda.set_rng_state(cuda_state, device)


def _load_optimizer_state(module: Any, optimizer: Any, state: Mapping[str, Any]) -> None:
    from .checkpoint import _is_fsdp

    if not isinstance(state, Mapping) or not state:
        raise BackendContractError("Stage2 optimizer checkpoint state is empty")
    if _is_fsdp(module):
        from torch.distributed.fsdp import FullyShardedDataParallel

        local = FullyShardedDataParallel.optim_state_dict_to_load(
            model=module,
            optim=optimizer,
            optim_state_dict=state,
        )
        optimizer.load_state_dict(local)
    else:
        optimizer.load_state_dict(state)


def _validate_stage2_progress(runtime: Any, step: int) -> None:
    """Bind outer/student/scheduler/EMA progress to one exact resume point."""

    outer_step = int(step)
    if outer_step < 0:
        raise BackendContractError("Stage2 outer step must be non-negative")
    expected_student = len(
        student_update_steps(
            outer_step,
            int(runtime.critic_updates_per_student),
        )
    )
    actual_student = int(runtime.student_step)
    if actual_student != expected_student:
        raise BackendContractError(
            "Stage2 student cadence differs from its outer step: "
            f"actual={actual_student} expected={expected_student}"
        )
    scheduler_progress = {
        "student": (
            int(getattr(runtime.student_scheduler, "last_epoch", -1)),
            expected_student,
        ),
        "critic": (
            int(getattr(runtime.critic_scheduler, "last_epoch", -1)),
            outer_step,
        ),
    }
    drift = {
        name: {"actual": actual, "expected": expected}
        for name, (actual, expected) in scheduler_progress.items()
        if actual != expected
    }
    if drift:
        raise BackendContractError(
            f"Stage2 scheduler progress differs from optimizer cadence: {drift}"
        )

    start = int(runtime.ema_start_step)
    update_every = int(runtime.ema_update_every)
    if start < 0 or update_every < 1:
        raise BackendContractError("Stage2 EMA cadence is invalid")
    expected_ema_updates = (
        0 if expected_student < start else 1 + (expected_student - start) // update_every
    )
    ema = runtime.ema
    if expected_ema_updates == 0:
        if ema is not None:
            raise BackendContractError("Stage2 EMA exists before its hard student-step start")
        return
    if ema is None:
        raise BackendContractError("Stage2 EMA is absent after its hard start")
    actual_ema_updates = int(getattr(ema, "num_updates", -1))
    if actual_ema_updates != expected_ema_updates:
        raise BackendContractError(
            "Stage2 EMA update count differs from student cadence: "
            f"actual={actual_ema_updates} expected={expected_ema_updates}"
        )


def save_stage2_checkpoint(runtime: Any, step: int) -> str:
    """Collectively publish ``critic.pt`` then the ``model.pt`` commit marker."""

    import torch

    from .checkpoint import _gather_full_optimizer, _gather_full_state

    if int(step) != int(runtime.global_step) or int(step) < 1:
        raise BackendContractError("Stage2 checkpoint step is not the completed outer step")
    progress_error: str | None = None
    try:
        _validate_stage2_progress(runtime, int(step))
    except Exception as exc:
        progress_error = f"{type(exc).__name__}: {exc}"
    _collective_error(progress_error, phase="checkpoint progress")
    validate_checkpoint_transaction(("critic.pt", "model.pt"))
    dist, rank, _ = _distributed_context()
    target = checkpoint_model_dir(
        str(runtime.config["runtime"]["output_dir"]),
        step=int(step),
        width=6,
    )
    transaction: CheckpointTransaction | None = None
    setup_error = ""
    if rank == 0:
        try:
            transaction = CheckpointTransaction(target)
            transaction.__enter__()
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"
    setup_error = str(_broadcast_object(setup_error))
    if setup_error:
        raise BackendContractError(f"Stage2 checkpoint transaction setup failed: {setup_error}")

    rng_state = _gather_rng_state(int(runtime.topology.raw_rank), runtime.device, runtime.data)
    student_state = _gather_full_state(runtime.student.module, rank0_only=True)
    student_optimizer = _gather_full_optimizer(runtime.student.module, runtime.student_optimizer)
    ema_state: Mapping[str, Any] | None = None
    if runtime.ema is not None:
        with runtime.ema.swapped_into(runtime.student.module):
            ema_state = _gather_full_state(runtime.student.module, rank0_only=True)
    student_error = ""
    if rank == 0:
        assert transaction is not None
        try:
            model_payload: dict[str, Any] = {
                "checkpoint_format": _CHECKPOINT_FORMAT,
                "generator": _prefixed_state(student_state),
                "optimizer": student_optimizer,
                "scheduler": runtime.student_scheduler.state_dict(),
                "global_step": int(step),
                "student_step": int(runtime.student_step),
                "rng_state_by_rank": rng_state,
                "config": _plain(runtime.config),
                "role_receipts": stage2_checkpoint_contract(
                    runtime.config, runtime.role_receipts
                ).extras["roles"],
            }
            if ema_state is not None:
                model_payload["generator_ema"] = _prefixed_state(ema_state)
                model_payload["ema_num_updates"] = int(runtime.ema.num_updates)
            torch.save(model_payload, transaction.path / "model.pt.pending")
        except Exception as exc:
            student_error = f"{type(exc).__name__}: {exc}"
    student_error = str(_broadcast_object(student_error))
    del student_state, student_optimizer, ema_state
    if student_error:
        raise BackendContractError(f"Stage2 student checkpoint write failed: {student_error}")

    critic_state = _gather_full_state(runtime.critic.module, rank0_only=True)
    critic_optimizer = _gather_full_optimizer(runtime.critic.module, runtime.critic_optimizer)
    identity = ""
    commit_error = ""
    if rank == 0:
        assert transaction is not None
        try:
            torch.save(
                {
                    "checkpoint_format": _CHECKPOINT_FORMAT,
                    "critic": _prefixed_state(critic_state),
                    "critic_optimizer": critic_optimizer,
                    "critic_scheduler": runtime.critic_scheduler.state_dict(),
                    "global_step": int(step),
                },
                transaction.path / "critic.pt",
            )
            os.replace(
                transaction.path / "model.pt.pending",
                transaction.path / "model.pt",
            )
            committed = transaction.commit(
                step=int(step),
                contract=stage2_checkpoint_contract(runtime.config, runtime.role_receipts),
                required_components=("critic.pt", "model.pt"),
                metadata={
                    "schema": "solarwm.wan22-stage2-pair.v1",
                    "publication_order": ["critic.pt", "model.pt"],
                    "commit_marker": "model.pt",
                    "student_step": int(runtime.student_step),
                    "ema_present": runtime.ema is not None,
                },
            )
            identity = committed.manifest_digest
        except Exception as exc:
            commit_error = f"{type(exc).__name__}: {exc}"
        finally:
            transaction.__exit__(None, None, None)
    commit_error = str(_broadcast_object(commit_error))
    identity = str(_broadcast_object(identity))
    del critic_state, critic_optimizer
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    if commit_error:
        raise BackendContractError(f"Stage2 checkpoint commit failed: {commit_error}")
    if not identity:
        raise BackendContractError("Stage2 checkpoint commit returned no identity")
    return identity


def load_stage2_checkpoint(runtime: Any, path: str | Path) -> RestoredStage2Checkpoint:
    """Restore a verified pair, both optimizers/schedulers, EMA, and RNG."""

    import torch

    from .checkpoint import _load_full_ema_state, _load_full_model_state

    _, rank, _ = _distributed_context()
    source = Path(path).expanduser().resolve()
    result: dict[str, Any] = {}
    verify_error = ""
    if rank == 0:
        try:
            verified = verify_checkpoint(source)
            assert_resume_compatible(
                stage2_checkpoint_contract(runtime.config, runtime.role_receipts),
                verified.contract,
            )
            members = {record.path for record in verified.files}
            if not {"critic.pt", "model.pt"}.issubset(members):
                raise BackendContractError("Stage2 checkpoint pair is incomplete")
            result = {
                "step": verified.step,
                "identity": verified.manifest_digest,
                "path": str(verified.path),
            }
        except Exception as exc:
            verify_error = f"{type(exc).__name__}: {exc}"
    verify_error = str(_broadcast_object(verify_error))
    result = dict(_broadcast_object(result))
    if verify_error:
        raise BackendContractError(f"Stage2 resume verification failed: {verify_error}")
    root = Path(str(result["path"]))
    _collective_error(
        None
        if (root / "model.pt").is_file() and (root / "critic.pt").is_file()
        else f"checkpoint pair is not rank-visible: {root}",
        phase="resume visibility",
    )
    model_payload = torch.load(root / "model.pt", map_location="cpu", weights_only=True, mmap=True)
    critic_payload = torch.load(
        root / "critic.pt", map_location="cpu", weights_only=True, mmap=True
    )
    if (
        not isinstance(model_payload, Mapping)
        or not isinstance(critic_payload, Mapping)
        or model_payload.get("checkpoint_format") != _CHECKPOINT_FORMAT
        or critic_payload.get("checkpoint_format") != _CHECKPOINT_FORMAT
    ):
        raise BackendContractError("Stage2 checkpoint pair has an incompatible format")
    step = int(result["step"])
    if (
        int(model_payload.get("global_step", -1)) != step
        or int(critic_payload.get("global_step", -1)) != step
    ):
        raise BackendContractError("Stage2 model/critic/manifest steps differ")
    student_step = int(model_payload.get("student_step", -1))
    student_state = normalize_model_state(model_payload.get("generator", {}), field="generator")
    critic_state = normalize_model_state(critic_payload.get("critic", {}), field="critic")
    _load_full_model_state(runtime.student.module, student_state)
    _load_full_model_state(runtime.critic.module, critic_state)
    _load_optimizer_state(
        runtime.student.module,
        runtime.student_optimizer,
        model_payload.get("optimizer", {}),
    )
    _load_optimizer_state(
        runtime.critic.module,
        runtime.critic_optimizer,
        critic_payload.get("critic_optimizer", {}),
    )
    runtime.student_scheduler.load_state_dict(model_payload.get("scheduler", {}))
    runtime.critic_scheduler.load_state_dict(critic_payload.get("critic_scheduler", {}))
    ema_state = model_payload.get("generator_ema")
    if ema_state is not None:
        ema = runtime.ensure_ema()
        normalized_ema = normalize_model_state(ema_state, field="generator_ema")
        _load_full_ema_state(
            runtime.student.module,
            ema,
            normalized_ema,
            num_updates=int(model_payload.get("ema_num_updates", -1)),
        )
    elif student_step >= int(runtime.ema_start_step):
        raise BackendContractError("Stage2 checkpoint lacks EMA after its hard start")
    runtime._global_step = step
    runtime.student_step = student_step
    progress_error = None
    try:
        _validate_stage2_progress(runtime, step)
    except Exception as exc:
        progress_error = f"{type(exc).__name__}: {exc}"
    _collective_error(progress_error, phase="resume progress")
    runtime_state_error = None
    try:
        _restore_rng_state(
            model_payload.get("rng_state_by_rank"),
            rank=int(runtime.topology.raw_rank),
            device=runtime.device,
            reader=runtime.data,
        )
    except Exception as exc:
        runtime_state_error = f"{type(exc).__name__}: {exc}"
    _collective_error(runtime_state_error, phase="rank runtime restore")
    return RestoredStage2Checkpoint(
        step=step,
        student_step=student_step,
        identity=f"digest:{result['identity']}",
        path=root,
    )


class Wan5BStage2Runtime:
    """Three-role SGF runtime with separate student and critic optimizers."""

    # Validation case materialization delegates to CudaWanGenerationAdapter,
    # which assigns fixed-plan rows to logical DP owners before returning.
    build_cases_returns_partition = True

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        student: WanDiffusion,
        teacher: WanDiffusion,
        critic: WanDiffusion,
        codec: Wan5BOnlineCodec,
        text_encoder: Any,
        batches: Any,
        student_optimizer: Any,
        critic_optimizer: Any,
        student_scheduler: Any,
        critic_scheduler: Any,
        topology: Any,
        role_receipts: Mapping[str, RoleCheckpointReceipt],
        generation_runner: Stage2GenerationRunner | None = None,
        kv_cache_allocator: Callable[..., list[dict[str, Any]]] | None = None,
        crossattn_cache_allocator: Callable[..., list[dict[str, Any]]] | None = None,
        video_encoder: Callable[..., bytes] | None = None,
        device: Any | None = None,
    ) -> None:
        import torch

        self.config = config
        self.student = student
        self.teacher = teacher
        self.critic = critic
        self.codec = codec
        self.text_encoder = text_encoder
        self.data = batches
        self.batches = iter(batches)
        self.student_optimizer = student_optimizer
        self.critic_optimizer = critic_optimizer
        self.student_scheduler = student_scheduler
        self.critic_scheduler = critic_scheduler
        self.topology = topology
        self.role_receipts = dict(role_receipts)
        if set(self.role_receipts) != {"student", "teacher", "critic"}:
            raise BackendContractError("Stage2 runtime requires all three role receipts")
        self.generation_runner = generation_runner
        self.video_encoder = video_encoder
        self.device = torch.device(
            device if device is not None else ("cuda", int(topology.local_rank))
        )
        self.family = str(config["model"]["family"])
        self.is_writer = int(topology.sp_rank) == 0
        self.diffusion = student
        self.vae = codec.vae
        self.critic_updates_per_student = int(config["train"]["critic_updates_per_student"])
        self.ema_start_step = int(config["train"]["ema"]["start_step"])
        self.ema_update_every = int(config["train"]["ema"].get("update_every", 1))
        self.ema: ShardedEMA | None = None
        self._global_step = 0
        self.student_step = 0
        self._unconditional_cache: dict[int, Mapping[str, Any]] = {}
        self._prepared: dict[str, Any] = {}
        self._kv_cache_allocator = kv_cache_allocator
        self._crossattn_cache_allocator = crossattn_cache_allocator
        self.initialization_receipt = _stage2_initialization_receipt(self.role_receipts)
        self.initialization_id = str(self.initialization_receipt["initialization_id"])
        self.checkpoint_id = self.initialization_id

    @property
    def global_step(self) -> int:
        return self._global_step

    def ensure_ema(self) -> ShardedEMA:
        import torch

        if self.ema is None:
            ema_config = self.config["train"]["ema"]
            self.ema = ShardedEMA(
                self.student.module,
                decay=float(ema_config["decay"]),
                device=self.device,
                dtype=torch.float32,
            )
        return self.ema

    def sync(self) -> None:
        dist, _, world = _distributed_context()
        if world > 1:
            dist.barrier()

    def _broadcast(self, tensor: Any) -> None:
        # Stage2 uses SP1. The shared generation adapter must not infer a
        # broader topology.
        if int(self.topology.sp_size) != 1:
            raise BackendContractError("Stage2 generation supports SP1 only")

    def _noise(self, shape: Sequence[int], generator: Any) -> Any:
        import torch

        value = torch.randn(
            tuple(int(item) for item in shape),
            generator=generator,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._broadcast(value)
        return value

    def build_cases(self, plan: Any) -> tuple[Any, ...]:
        # Case materialization is deliberately reused byte-for-byte from the
        # common Wan adapter. Only the diffusion sampler is Stage2-specific.
        from .inference import CudaWanGenerationAdapter

        return CudaWanGenerationAdapter.build_cases(self, plan)

    def _conditions(
        self, case: Any, *, latent_frames: int
    ) -> tuple[Any, Mapping[str, Any], Mapping[str, Any], Any | None]:
        from .inference import CudaWanGenerationAdapter

        return CudaWanGenerationAdapter._conditions(self, case, latent_frames=latent_frames)

    def weight_id(self, role: str) -> str:
        normalized = str(role).strip().lower()
        if normalized not in {"live", "ema"}:
            raise BackendContractError(f"unknown Stage2 weight role {role!r}")
        if normalized == "ema" and self.ema is None:
            raise BackendContractError("Stage2 EMA weights do not exist yet")
        return f"{self.checkpoint_id}#{normalized}"

    def generate(self, case: Any, *, weights_id: str) -> Any:
        metadata = case.metadata.get("generation_pass")
        if not isinstance(metadata, Mapping):
            raise BackendContractError("Stage2 generation case lacks pass metadata")
        role = str(metadata.get("weights", "")).strip().lower()
        expected = self.weight_id(role)
        if str(weights_id) != expected:
            raise BackendContractError(
                f"Stage2 weights identity drift: {weights_id!r} != {expected!r}"
            )
        if role == "ema":
            assert self.ema is not None
            with self.ema.swapped_into(self.student.module):
                return _stage2_generated_sample(self, case, weights_id=weights_id)
        return _stage2_generated_sample(self, case, weights_id=weights_id)

    @staticmethod
    def _root(module: Any) -> Any:
        return getattr(module, "module", module)

    def allocate_kv_cache(
        self, batch_size: int, *, dtype: Any, device: Any
    ) -> list[dict[str, Any]]:
        """Allocate the six-chunk no-sink cache and camera metadata up front."""

        if self._kv_cache_allocator is not None:
            return self._kv_cache_allocator(batch_size, dtype=dtype, device=device)
        import torch

        root = self._root(self.student.module)
        blocks = tuple(getattr(root, "blocks", ()))
        num_heads = int(getattr(root, "num_heads", 0))
        dim = int(getattr(root, "dim", 0))
        if not blocks or num_heads <= 0 or dim <= 0 or dim % num_heads:
            raise BackendContractError("Stage2 cannot infer the Wan KV-cache geometry")
        head_dim = dim // num_heads
        cache_frames = int(self.config["model"]["local_attn_size"])
        cache_tokens = cache_frames * int(self.config["model"]["frame_sequence_length"])
        caches = [
            {
                "k": torch.zeros(
                    batch_size,
                    cache_tokens,
                    num_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    batch_size,
                    cache_tokens,
                    num_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
                "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
            }
            for _ in blocks
        ]
        caches[0]["_fused_prope_camera_metadata"] = {
            "viewmats": torch.zeros(
                batch_size,
                cache_tokens,
                4,
                4,
                dtype=torch.float32,
                device=device,
            ),
            "K": torch.zeros(
                batch_size,
                cache_tokens,
                3,
                3,
                dtype=torch.float32,
                device=device,
            ),
        }
        return caches

    def allocate_crossattn_cache(
        self, batch_size: int, *, dtype: Any, device: Any
    ) -> list[dict[str, Any]]:
        if self._crossattn_cache_allocator is not None:
            return self._crossattn_cache_allocator(batch_size, dtype=dtype, device=device)
        import torch

        root = self._root(self.student.module)
        blocks = tuple(getattr(root, "blocks", ()))
        num_heads = int(getattr(root, "num_heads", 0))
        dim = int(getattr(root, "dim", 0))
        if not blocks or num_heads <= 0 or dim % num_heads:
            raise BackendContractError("Stage2 cannot infer the Wan cross-attention cache geometry")
        head_dim = dim // num_heads
        return [
            {
                "k": torch.zeros(
                    batch_size,
                    512,
                    num_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    batch_size,
                    512,
                    num_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "is_init": False,
            }
            for _ in blocks
        ]

    def _prepare_next_batch(
        self,
    ) -> tuple[Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
        batch = next(self.batches)
        camera = {
            key: value.to(self.device, non_blocking=True) for key, value in batch["camera"].items()
        }
        encoded = self.codec.encode_stage2_batch(
            sample_ids=batch["sample_ids"],
            first_pixels=batch["pixels"][:, :1].to(self.device, non_blocking=True),
            captions=batch["prompts"],
            camera=camera,
        )
        if encoded.get("i2v_y") is not None:
            raise BackendContractError("Stage2 TI2V-5B may not receive official I2V y")
        return (
            encoded["latents"],
            {"prompt_embeds": encoded["prompt_embeds"]},
            encoded["camera"],
            batch,
        )

    def _unconditional(self, batch_size: int, *, dtype: Any) -> Mapping[str, Any]:
        cached = self._unconditional_cache.get(int(batch_size))
        if cached is None:
            import torch

            with torch.no_grad():
                cached = {
                    key: value.detach()
                    for key, value in self.text_encoder(
                        [str(self.config["train"]["negative_prompt"])] * int(batch_size)
                    ).items()
                }
            self._unconditional_cache[int(batch_size)] = cached
        return {key: value.to(device=self.device, dtype=dtype) for key, value in cached.items()}

    def _run_student_rollout(
        self,
        clean: Any,
        condition: Mapping[str, Any],
        camera: Mapping[str, Any],
        *,
        require_grad: bool,
    ) -> Stage2Rollout:
        import torch

        noise = torch.randn_like(clean)
        noise[:, :1] = clean[:, :1]
        return stage2_camera_rollout(
            student=self.student,
            allocate_kv_cache=self.allocate_kv_cache,
            allocate_crossattn_cache=self.allocate_crossattn_cache,
            noise=noise,
            first_latent=clean[:, :1],
            condition=condition,
            camera=camera,
            denoising_steps=self.denoising_steps,
            num_frame_per_block=int(self.config["model"]["num_frame_per_block"]),
            frame_sequence_length=int(self.config["model"]["frame_sequence_length"]),
            context_timestep=float(self.config["train"]["context_timestep"]),
            per_rank_exit_step=bool(self.config["train"]["per_rank_exit_step"]),
            match_context=bool(self.config["train"]["self_gradient_forcing_match_context"]),
            last_step_only=bool(self.config["train"]["last_step_only"]),
            require_grad=require_grad,
            validate_finite=bool(self.config["train"].get("strict_numeric_checks", False)),
        )

    def _sample_score_timestep(self, batch_size: int, num_frames: int) -> Any:
        value = sample_shifted_score_timesteps(
            batch_size=batch_size,
            num_frames=num_frames,
            device=self.device,
            shift=float(self.config["model"]["timestep_shift"]),
            num_train_timesteps=int(self.config["train"]["num_train_timesteps"]),
            min_timestep=float(self.config["train"]["score_min_timestep"]),
            max_timestep=float(self.config["train"]["score_max_timestep"]),
        )
        value[:, 0] = 0.0
        return value

    def _score_forward(
        self,
        model: WanDiffusion,
        noisy: Any,
        condition: Mapping[str, Any],
        camera: Mapping[str, Any],
        timestep: Any,
    ) -> tuple[Any, Any]:
        frame_tokens = int(self.config["model"]["frame_sequence_length"])
        flow = model(
            noisy,
            condition,
            camera,
            expand_timesteps_to_tokens(timestep, frame_tokens),
            sequence_length=noisy.shape[1] * frame_tokens,
        )
        return flow, model.flow_to_x0(noisy, flow, timestep)

    def student_loss_for_batch(
        self, clean: Any, condition: Mapping[str, Any], camera: Mapping[str, Any]
    ) -> tuple[Any, float, Mapping[str, Any]]:
        import torch

        rollout = self._run_student_rollout(clean, condition, camera, require_grad=True)
        generated = rollout.output
        batch_size, frames = generated.shape[:2]
        timestep = self._sample_score_timestep(batch_size, frames)
        noise = torch.randn_like(generated)
        noisy = (
            self.student.scheduler.add_noise(
                generated.detach().flatten(0, 1).float(),
                noise.flatten(0, 1).float(),
                timestep.flatten(0, 1),
            )
            .unflatten(0, (batch_size, frames))
            .to(generated.dtype)
        )
        noisy[:, :1] = clean[:, :1]
        unconditional = self._unconditional(
            batch_size,
            dtype=condition["prompt_embeds"].dtype,
        )
        with torch.no_grad():
            _, fake_cond = self._score_forward(self.critic, noisy, condition, camera, timestep)
            fake_scale = float(self.config["train"]["fake_guidance_scale"])
            if fake_scale:
                _, fake_uncond = self._score_forward(
                    self.critic, noisy, unconditional, camera, timestep
                )
                fake_x0 = reference_cfg(fake_cond, fake_uncond, fake_scale)
            else:
                fake_x0 = fake_cond
            _, real_cond = self._score_forward(self.teacher, noisy, condition, camera, timestep)
            _, real_uncond = self._score_forward(
                self.teacher, noisy, unconditional, camera, timestep
            )
            real_x0 = reference_cfg(
                real_cond,
                real_uncond,
                float(self.config["train"]["real_guidance_scale"]),
            )
            gradient = compute_kl_gradient(
                fake_x0=fake_x0,
                real_x0=real_x0,
                student_output=generated,
                mask=rollout.loss_mask,
                normalize=True,
            )
        loss = sgf_student_loss(generated, gradient, mask=rollout.loss_mask)
        if not bool(torch.isfinite(loss).item()):
            raise BackendContractError("Stage2 student loss is non-finite")
        return loss, float(gradient.abs().mean().item()), {}

    def critic_loss_for_batch(
        self, clean: Any, condition: Mapping[str, Any], camera: Mapping[str, Any]
    ) -> tuple[Any, Mapping[str, Any]]:
        import torch

        with torch.no_grad():
            generated = self._run_student_rollout(clean, condition, camera, require_grad=False)
        output = generated.output.detach()
        batch_size, frames = output.shape[:2]
        timestep = self._sample_score_timestep(batch_size, frames)
        noise = torch.randn_like(output)
        noisy = (
            self.student.scheduler.add_noise(
                output.flatten(0, 1).float(),
                noise.flatten(0, 1).float(),
                timestep.flatten(0, 1),
            )
            .unflatten(0, (batch_size, frames))
            .to(output.dtype)
        )
        noisy[:, :1] = clean[:, :1]
        flow, _ = self._score_forward(self.critic, noisy, condition, camera, timestep)
        loss = sgf_critic_flow_loss(
            flow,
            noise=noise,
            clean=output,
            mask=generated.loss_mask,
        )
        if not bool(torch.isfinite(loss).item()):
            raise BackendContractError("Stage2 critic loss is non-finite")
        return loss, {}

    @staticmethod
    def _clip(module: Any, max_norm: float) -> float:
        import torch

        if hasattr(module, "clip_grad_norm_"):
            norm = module.clip_grad_norm_(float(max_norm))
        else:
            norm = torch.nn.utils.clip_grad_norm_(module.parameters(), float(max_norm))
        value = float(torch.as_tensor(norm).detach().float().item())
        if not math.isfinite(value):
            raise BackendContractError("Stage2 gradient norm is non-finite")
        return value

    def train_outer_step(self) -> Mapping[str, float]:
        """Run one critic update and the scheduled optional student update."""

        update_student = should_update_student(self.global_step, self.critic_updates_per_student)
        student_loss_value = 0.0
        student_grad_norm = 0.0
        dmd_gradient_norm = 0.0
        self.student.module.eval()
        self.teacher.module.eval()
        self.critic.module.eval()
        if update_student:
            self.student_optimizer.zero_grad(set_to_none=True)
            clean, condition, camera, _ = self._prepare_next_batch()
            student_loss, dmd_gradient_norm, _ = self.student_loss_for_batch(
                clean, condition, camera
            )
            student_loss.backward()
            student_grad_norm = self._clip(
                self.student.module,
                float(self.config["train"]["optimizer"]["grad_clip"]),
            )
            self.student_optimizer.step()
            self.student_scheduler.step()
            self.student_step += 1
            student_loss_value = float(student_loss.detach().item())

        self.critic_optimizer.zero_grad(set_to_none=True)
        clean, condition, camera, _ = self._prepare_next_batch()
        critic_loss, _ = self.critic_loss_for_batch(clean, condition, camera)
        critic_loss.backward()
        critic_grad_norm = self._clip(
            self.critic.module,
            float(self.config["train"]["critic_optimizer"]["grad_clip"]),
        )
        self.critic_optimizer.step()
        self.critic_scheduler.step()
        if update_student and self.student_step >= self.ema_start_step:
            ema = self.ensure_ema()
            if (self.student_step - self.ema_start_step) % self.ema_update_every == 0:
                ema.update(self.student.module)
        self._global_step += 1
        critic_loss_value = float(critic_loss.detach().item())
        return {
            "loss": (student_loss_value if update_student else critic_loss_value),
            "loss_student": student_loss_value,
            "loss_critic": critic_loss_value,
            "student_updated": float(update_student),
            "student_step": float(self.student_step),
            "student_grad_norm": student_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "dmdtrain_gradient_norm": dmd_gradient_norm,
            "student_lr": float(self.student_scheduler.get_last_lr()[0]),
            "critic_lr": float(self.critic_scheduler.get_last_lr()[0]),
        }

    def save_checkpoint(self, step: int) -> str:
        identity = save_stage2_checkpoint(self, step)
        self.checkpoint_id = f"digest:{identity}"
        return identity

    def run_generation(
        self,
        *,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        """The single Stage2 generation entry used by infer and validation."""

        if self.generation_runner is None:
            from .inference import run_wan_validation

            runner: Stage2GenerationRunner = run_wan_validation
        else:
            runner = self.generation_runner
        summary = runner(
            self.config,
            provider=self,
            output_dir=output_dir,
        )
        if isinstance(summary, Mapping):
            payload = dict(summary)
        else:
            try:
                payload = asdict(summary)
            except TypeError as exc:
                raise BackendContractError(
                    "Stage2 unified generation runner returned an invalid summary"
                ) from exc
        if isinstance(payload.get("output_dir"), Path):
            payload["output_dir"] = str(payload["output_dir"])
        return payload

    def validate(self, step: int) -> Mapping[str, Any]:
        output_dir = Path(str(self.config["runtime"]["output_dir"]))
        report = self.run_generation(
            output_dir=validation_staging_root(output_dir) / f"step-{int(step):06d}",
        )
        return {
            "schema": "solarwm.wan22-stage2-validation.v1",
            "step": int(step),
            "generation": report,
        }

    def close(self) -> None:
        close = getattr(self.data, "close", None)
        if callable(close):
            close()


def _score_model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    score = copy.deepcopy(dict(config))
    score["model"] = copy.deepcopy(dict(config["model"]))
    score["train"] = copy.deepcopy(dict(config["train"]))
    score["model"]["causal"] = False
    score["model"]["local_attn_size"] = int(config["model"]["score_local_attn_size"])
    score["model"]["sink_size"] = 0
    score["model"]["use_echorope"] = bool(config["model"]["score_use_echorope"])
    score["train"]["objective"] = "flow_matching"
    score["train"].pop("objective_variant", None)
    score["train"].pop("anyflow_variant", None)
    return score


def _adamw(parameters: Any, values: Mapping[str, Any]) -> Any:
    import torch

    return torch.optim.AdamW(
        parameters,
        lr=float(values["lr"]),
        betas=tuple(float(value) for value in values["betas"]),
        eps=float(values["eps"]),
        weight_decay=float(values["weight_decay"]),
    )


def build_stage2_runtime(
    config: Mapping[str, Any],
    *,
    generation_runner: Stage2GenerationRunner | None = None,
) -> Wan5BStage2Runtime:
    """Build all three SGF roles and strictly load their selected initial states."""

    import torch

    from ..sgf import validate_stage2_contract

    roles = validate_stage2_contract(config)
    topology = initialize_torchrun(int(config["distributed"]["sequence_parallel_size"]))
    if int(topology.raw_world_size) != int(config["distributed"]["world_size"]):
        raise BackendContractError("Wan Stage2 torchrun world size differs from config")
    compile_value = os.environ.get("SOLARWM_COMPILE_FLEX", "0").strip().lower()
    if compile_value not in {"1", "true", "yes"}:
        raise BackendContractError(
            "Stage2 requires SOLARWM_COMPILE_FLEX=1 before Python imports the Wan model"
        )

    # Rank zero verifies checkpoint presence and size before every 5B allocation;
    # each role's real payload contract is validated while loading it.
    role_by_name = {role.role: role for role in roles}
    receipts = {role.role: verify_role_checkpoint(role) for role in roles}
    probe_runtime(
        config,
        family="wan22_ti2v_5b",
        require_cuda=True,
        require_transformer_weights=False,
        validate_index_contents=False,
    ).require_ready()
    device = torch.device("cuda", int(topology.local_rank))
    init_seed = model_init_seed("wan22_ti2v_5b", int(config["data"]["seed"]))

    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    layout = WanAssetLayout.from_config(config)
    student = build_diffusion_architecture(config)
    text_encoder = WanTextEncoder(layout.text_encoder, layout.tokenizer)
    vae = Wan5BVAE(layout.vae)
    dropped, receipts["student"] = load_role_checkpoint(
        initialization=role_by_name["student"],
        receipt=receipts["student"],
        diffusion=student,
    )
    if len(dropped) != 4:
        raise BackendContractError(
            "Stage2 student conversion did not drop exactly four AnyFlow tensors"
        )
    student.module.train().requires_grad_(True)
    student.module = wrap_transformer_fsdp(student.module, config, topology)

    score_config = _score_model_config(config)
    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    teacher = build_diffusion_architecture(score_config)
    _, receipts["teacher"] = load_role_checkpoint(
        initialization=role_by_name["teacher"],
        receipt=receipts["teacher"],
        diffusion=teacher,
    )
    teacher.module.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16)

    torch.manual_seed(init_seed)
    torch.cuda.manual_seed_all(init_seed)
    critic = build_diffusion_architecture(score_config)
    _, receipts["critic"] = load_role_checkpoint(
        initialization=role_by_name["critic"],
        receipt=receipts["critic"],
        diffusion=critic,
    )
    critic.module.train().requires_grad_(True)
    critic.module = wrap_transformer_fsdp(critic.module, config, topology)

    text_encoder.to(device)
    vae.to(device)
    student_optimizer = _adamw(student.module.parameters(), config["train"]["optimizer"])
    critic_optimizer = _adamw(critic.module.parameters(), config["train"]["critic_optimizer"])
    max_steps = int(config["train"]["max_steps"])
    ratio = int(config["train"]["critic_updates_per_student"])
    student_steps = max(1, len(student_update_steps(max_steps, ratio)))
    student_optimizer_config = config["train"]["optimizer"]
    critic_optimizer_config = config["train"]["critic_optimizer"]
    student_scheduler = make_warmup_cosine(
        student_optimizer,
        warmup_steps=int(student_optimizer_config.get("warmup_steps", 0)),
        total_steps=student_steps,
        min_lr_ratio=float(student_optimizer_config.get("min_lr_ratio", 1.0)),
    )
    critic_scheduler = make_warmup_cosine(
        critic_optimizer,
        warmup_steps=int(critic_optimizer_config.get("warmup_steps", 0)),
        total_steps=max_steps,
        min_lr_ratio=float(critic_optimizer_config.get("min_lr_ratio", 1.0)),
    )
    codec = Wan5BOnlineCodec(
        vae,
        text_encoder,
        pixel_frames=int(config["data"]["pixel_frames"]),
        height=int(config["data"]["height"]),
        width=int(config["data"]["width"]),
        frame_sequence_length=int(config["model"]["frame_sequence_length"]),
    )
    stream_seed = int(config["data"]["seed"]) * 100003 + int(topology.dp_rank) * 1024
    torch.manual_seed(stream_seed)
    torch.cuda.manual_seed_all(stream_seed)
    loader = build_raw_dataloader(config, topology)
    runtime = Wan5BStage2Runtime(
        config,
        student=student,
        teacher=teacher,
        critic=critic,
        codec=codec,
        text_encoder=text_encoder,
        batches=loader,
        student_optimizer=student_optimizer,
        critic_optimizer=critic_optimizer,
        student_scheduler=student_scheduler,
        critic_scheduler=critic_scheduler,
        topology=topology,
        role_receipts=receipts,
        generation_runner=generation_runner,
        device=device,
    )
    runtime.denoising_steps = warp_denoising_steps(
        config["train"]["denoising_step_list"],
        student.scheduler.timesteps.to(device),
        num_train_timesteps=int(config["train"]["num_train_timesteps"]),
    )
    resume_from = str(config.get("runtime", {}).get("resume_from", "")).strip()
    if resume_from:
        if not resume_from.startswith("/"):
            raise BackendContractError("runtime.resume_from must be absolute")
        restored = load_stage2_checkpoint(runtime, resume_from)
        runtime.checkpoint_id = restored.identity
    return runtime


def run_stage2_training(config: Mapping[str, Any]) -> int:
    """Run the critic-every-step/student-5:1 Stage2 outer loop."""

    owner = os.environ.get(_TORCHRUN_OWNER_ENV, "backend").strip().lower()
    if owner not in {"backend", "caller"}:
        raise BackendContractError(
            f"{_TORCHRUN_OWNER_ENV} must be backend or caller, got {owner!r}"
        )
    runtime = None
    try:
        runtime = build_stage2_runtime(config)
        runtime_config = config.get("runtime", {})
        max_steps = int(runtime_config.get("max_steps_override", config["train"]["max_steps"]))
        if max_steps < runtime.global_step:
            raise BackendContractError("Stage2 max step is behind the restored checkpoint step")
        save_every = int(runtime_config.get("save_every", 0))
        validate_every = int(runtime_config.get("validate_every", 0))
        sink = JsonlEventSink(
            Path(str(runtime_config["output_dir"]))
            / "events"
            / f"rank-{runtime.topology.raw_rank:05d}.jsonl"
        )
        with suspend_automatic_cycle_collection():
            while runtime.global_step < max_steps:
                metrics = dict(runtime.train_outer_step())
                step = runtime.global_step
                sink(
                    {
                        "schema": "solarwm.wan22-stage2-training-step.v1",
                        "event": "outer_step",
                        "step": step,
                        "metrics": metrics,
                    }
                )
                crossed_boundary = False
                if save_every > 0 and step % save_every == 0:
                    checkpoint_id = runtime.save_checkpoint(step)
                    sink(
                        {
                            "schema": "solarwm.training-event.v1",
                            "event": "checkpoint",
                            "step": step,
                            "checkpoint_id": checkpoint_id,
                        }
                    )
                    crossed_boundary = True
                if validate_every > 0 and step % validate_every == 0:
                    report = runtime.validate(step)
                    sink(
                        {
                            "schema": "solarwm.training-event.v1",
                            "event": "validation",
                            "step": step,
                            "report": dict(report),
                        }
                    )
                    crossed_boundary = True
                if crossed_boundary:
                    gc.collect()
        if runtime.global_step != max_steps:
            raise BackendContractError(
                f"Wan Stage2 stopped at {runtime.global_step}, expected {max_steps}"
            )
        return 0
    finally:
        if runtime is not None:
            close = getattr(runtime, "close", None)
            if callable(close):
                close()
        if owner == "backend":
            cleanup_torchrun()


class CudaWanStage2GenerationAdapter:
    """Lazy facade over the common CUDA adapter with the Stage2 sampler."""

    def __new__(cls, config: Mapping[str, Any], plan: Any) -> Any:
        # Delaying the subclass definition avoids pulling CUDA/model imports
        # into CPU-only contract tests that merely import this module.
        from .inference import CudaWanGenerationAdapter

        class _Adapter(CudaWanGenerationAdapter):
            video_encoder = None

            def __init__(self, values: Mapping[str, Any], generation_plan: Any) -> None:
                import torch

                if not torch.cuda.is_available():
                    raise BackendContractError("Wan Stage2 inference requires CUDA")
                self.config = values
                self.plan = generation_plan
                self.family = str(values["model"]["family"])
                self._direct_model = (
                    str(values.get("inference", {}).get("length", "fixed")).strip().lower()
                    == "camera"
                )
                self.topology = initialize_torchrun(
                    int(values["distributed"]["sequence_parallel_size"])
                )
                self.is_writer = int(self.topology.sp_rank) == 0
                self.device = torch.device("cuda", int(self.topology.local_rank))
                (
                    self.checkpoint_path,
                    checkpoint_manifest_id,
                    self.checkpoint_step,
                ) = _verified_stage2_inference_checkpoint(
                    values,
                    str(values["checkpoint"]["path"]),
                )
                self._model_weight_role = (
                    _published_default_weight_role(values["checkpoint"]["path"])
                    if self._direct_model
                    else None
                )
                self.checkpoint_id = f"manifest:{checkpoint_manifest_id}"
                layout = WanAssetLayout.from_config(values)
                self.diffusion = build_diffusion_architecture(values)
                self.text_encoder = WanTextEncoder(layout.text_encoder, layout.tokenizer)
                self.vae = Wan5BVAE(layout.vae)
                inference_dtype = torch.bfloat16
                self.diffusion.module.eval().requires_grad_(False).to(
                    device=self.device, dtype=inference_dtype
                )
                self.text_encoder.to(self.device, dtype=inference_dtype)
                self.vae.to(self.device, dtype=inference_dtype)
                self._loaded_role: str | None = None
                self._prepared: dict[int, Any] = {}
                self._deferred_camera_inputs = None

            @staticmethod
            def _root(module: Any) -> Any:
                return getattr(module, "module", module)

            def weight_id(self, role: str) -> str:
                if role == "model" and self._direct_model:
                    return f"{self.checkpoint_id}#release-default:{self._model_weight_role}"
                return super().weight_id(role)

            def _checkpoint_state_field(self, role: str) -> str:
                if role == "model" and self._direct_model:
                    return super()._checkpoint_state_field(str(self._model_weight_role))
                return super()._checkpoint_state_field(role)

            def allocate_kv_cache(
                self, batch_size: int, *, dtype: Any, device: Any
            ) -> list[dict[str, Any]]:
                self.student = self.diffusion
                self._kv_cache_allocator = None
                return Wan5BStage2Runtime.allocate_kv_cache(
                    self, batch_size, dtype=dtype, device=device
                )

            def allocate_crossattn_cache(
                self, batch_size: int, *, dtype: Any, device: Any
            ) -> list[dict[str, Any]]:
                self.student = self.diffusion
                self._crossattn_cache_allocator = None
                return Wan5BStage2Runtime.allocate_crossattn_cache(
                    self, batch_size, dtype=dtype, device=device
                )

            def generate(self, case: Any, *, weights_id: str) -> Any:
                metadata = case.metadata.get("generation_pass")
                if not isinstance(metadata, Mapping):
                    raise BackendContractError("Stage2 generation case lacks pass metadata")
                role = str(metadata.get("weights", "")).strip().lower()
                expected = self.weight_id(role)
                if str(weights_id) != expected:
                    raise BackendContractError(
                        "Stage2 standalone generation weights identity drift"
                    )
                self._load_role(role)
                camera_length = (
                    str(self.config.get("inference", {}).get("length", "fixed")).strip().lower()
                    == "camera"
                )
                deferred = self._deferred_camera_inputs
                if camera_length and deferred is None:
                    raise BackendContractError("Stage2 camera-length inputs were not prepared")
                if not camera_length:
                    return _stage2_generated_sample(self, case, weights_id=weights_id)
                self._materialize_deferred_camera_case(case)
                try:
                    return _stage2_generated_sample(self, case, weights_id=weights_id)
                finally:
                    self._prepared.pop(case.slot, None)

        return _Adapter(config, plan)


def build_stage2_generation_provider(config: Mapping[str, Any]) -> Any:
    """Build the standalone Stage2 provider used by unified Wan inference."""

    from ..generation import resolve_generation_plan

    if (
        str(config.get("model", {}).get("family", "")) != "wan22_ti2v_5b"
        or str(config.get("train", {}).get("stage", "")) != "stage2"
        or str(config.get("train", {}).get("objective", "")) != "flow_matching"
    ):
        raise BackendContractError("Stage2 inference provider contract differs")
    plan = resolve_generation_plan(config)
    if any(
        item.mode != "autoregressive"
        or item.solver != "self_forcing"
        or item.num_inference_steps != 4
        for item in plan.passes
    ):
        raise BackendContractError("Stage2 inference requires autoregressive self_forcing NFE4")
    return CudaWanStage2GenerationAdapter(config, plan)


def run_stage2_inference(config: Mapping[str, Any]) -> Any:
    """Run standalone Stage2 inference through the validation-identical core."""

    from .inference import run_wan_inference

    provider = build_stage2_generation_provider(config)
    try:
        return run_wan_inference(config, provider=provider)
    finally:
        provider.close()


__all__ = [
    "CudaWanStage2GenerationAdapter",
    "RestoredStage2Checkpoint",
    "RoleCheckpointReceipt",
    "Stage2Rollout",
    "Wan5BStage2Runtime",
    "build_stage2_generation_provider",
    "build_stage2_runtime",
    "load_role_checkpoint",
    "load_stage2_checkpoint",
    "run_stage2_inference",
    "run_stage2_training",
    "save_stage2_checkpoint",
    "stage2_camera_rollout",
    "stage2_checkpoint_contract",
    "verify_role_checkpoint",
]
