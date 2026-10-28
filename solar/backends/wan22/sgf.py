"""Stage2 self-gradient-forcing scheduling, math, and checkpoint contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from solar.errors import BackendContractError

_WEIGHT_SOURCES = frozenset({"live", "ema"})
STAGE2_CHECKPOINT_MEMBERS = ("critic.pt", "model.pt")


@dataclass(frozen=True)
class RoleInitialization:
    role: str
    path: str
    weights: str
    expected_stage: str
    expected_objective: str
    camera_translation_transform: str
    allow_anyflow_delta_drop: bool = False


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise BackendContractError(f"{key} must be a mapping")
    return value


def normalize_weight_source(value: Any, *, role: str) -> str:
    source = str(value).strip().lower()
    if source not in _WEIGHT_SOURCES:
        raise BackendContractError(f"checkpoint.roles.{role}.weights must be live or ema")
    return source


def should_update_student(global_step: int, critic_updates_per_student: int) -> bool:
    """Warm the critic first, then update the student every N outer steps."""

    ratio = int(critic_updates_per_student)
    if ratio < 1:
        raise ValueError("critic_updates_per_student must be at least one")
    step = int(global_step)
    return step > 0 and step % ratio == 0


def student_update_steps(max_outer_steps: int, ratio: int) -> tuple[int, ...]:
    if max_outer_steps < 0:
        raise ValueError("max_outer_steps must be non-negative")
    return tuple(step for step in range(max_outer_steps) if should_update_student(step, ratio))


def validate_checkpoint_transaction(members: Sequence[str]) -> None:
    """Require the paired critic payload followed by the model commit marker."""

    normalized = tuple(str(member) for member in members)
    if normalized != STAGE2_CHECKPOINT_MEMBERS:
        raise BackendContractError(
            "Stage2 checkpoint transaction must publish critic.pt then model.pt; "
            "model.pt is the commit marker"
        )


def _role(role: str, value: Any) -> RoleInitialization:
    if not isinstance(value, Mapping):
        raise BackendContractError(f"checkpoint.roles.{role} must be a mapping")
    path = str(value.get("path", "")).strip()
    if not path.startswith("/"):
        raise BackendContractError(f"checkpoint.roles.{role}.path must be absolute")
    supported = {
        "path",
        "weights",
        "expected_stage",
        "expected_objective",
        "camera_translation_transform",
        "allow_anyflow_delta_drop",
    }
    unknown = sorted(set(value) - supported)
    if unknown:
        raise BackendContractError(f"checkpoint.roles.{role} has unsupported fields: {unknown}")
    camera_transform = str(value.get("camera_translation_transform", "")).strip().lower()
    if camera_transform not in {"linear", "logd4"}:
        raise BackendContractError(
            f"checkpoint.roles.{role}.camera_translation_transform is invalid"
        )
    return RoleInitialization(
        role=role,
        path=path,
        weights=normalize_weight_source(value.get("weights"), role=role),
        expected_stage=str(value.get("expected_stage", "")),
        expected_objective=str(value.get("expected_objective", "")),
        camera_translation_transform=camera_transform,
        allow_anyflow_delta_drop=bool(value.get("allow_anyflow_delta_drop", False)),
    )


def validate_stage2_contract(config: Mapping[str, Any]) -> tuple[RoleInitialization, ...]:
    """Validate the 81f, SP1, six-chunk, three-role SGF contract."""

    action = str(config.get("action", "")).strip().lower()
    training = action != "infer"
    model = _mapping(config, "model")
    data = _mapping(config, "data")
    train = _mapping(config, "train")
    distributed = _mapping(config, "distributed")
    checkpoint = _mapping(config, "checkpoint")
    validation = _mapping(config, "validation")
    inference = _mapping(config, "inference")
    fsdp = _mapping(train, "fsdp")
    optimizer = _mapping(train, "optimizer")
    critic_optimizer = _mapping(train, "critic_optimizer")
    ema = _mapping(train, "ema")

    required = {
        "model.family": str(model.get("family", "")) == "wan22_ti2v_5b",
        "model.causal": bool(model.get("causal")),
        "model.camera_attention_mode": str(model.get("camera_attention_mode", "")) == "fused_prope",
        "model.use_echorope": bool(model.get("use_echorope", True)) is False,
        "model.score_use_echorope": bool(model.get("score_use_echorope", False)) is True,
        "model.local_attn_size": int(model.get("local_attn_size", -1)) == 18,
        "model.score_local_attn_size": int(model.get("score_local_attn_size", -1)) == 21,
        "model.max_prior_clean_chunks": int(model.get("max_prior_clean_chunks", -1)) == 5,
        "model.sink_size": int(model.get("sink_size", -1)) == 0,
        "model.rope_train_frames": int(model.get("rope_train_frames", -1)) == 21,
        "data.pixel_frames": int(data.get("pixel_frames", -1)) == 81,
        "data.latent_frames": int(data.get("latent_frames", -1)) == 21,
        "distributed.sequence_parallel_size": int(distributed.get("sequence_parallel_size", -1))
        == 1,
        "train.grad_accum": int(train.get("grad_accum", -1)) == 1,
        "train.objective": str(train.get("objective", "")) == "flow_matching",
        "train.trainer": str(train.get("trainer", "")) == "self_gradient_forcing",
        "train.i2v_image_condition_dropout": float(train.get("i2v_image_condition_dropout", -1))
        == 0.0,
        "train.self_gradient_forcing_cache_mode": str(
            train.get("self_gradient_forcing_cache_mode", "")
        )
        == "exit",
        "train.save_optimizer": bool(train.get("save_optimizer", False)),
        "train.per_rank_exit_step": bool(train.get("per_rank_exit_step", False)),
        "train.self_gradient_forcing_match_context": bool(
            train.get("self_gradient_forcing_match_context", False)
        ),
        "train.context_timestep": float(train.get("context_timestep", -1)) == 0.0,
        "train.last_step_only": bool(train.get("last_step_only", True)) is False,
        "train.ts_schedule": bool(train.get("ts_schedule", True)) is False,
        "train.real_guidance_scale": float(train.get("real_guidance_scale", -1)) == 3.0,
        "train.fake_guidance_scale": float(train.get("fake_guidance_scale", -1)) == 0.0,
        "train.negative_prompt": bool(str(train.get("negative_prompt", "")).strip()),
        "train.fsdp.cast_root_forward_inputs": bool(fsdp.get("cast_root_forward_inputs", True))
        is False,
        "train.optimizer.warmup_steps": int(optimizer.get("warmup_steps", -1)) == 0,
        "train.optimizer.min_lr_ratio": float(optimizer.get("min_lr_ratio", -1)) == 1.0,
        "train.critic_optimizer.lr": float(critic_optimizer.get("lr", -1)) == 4.0e-7,
        "train.critic_optimizer.betas": list(critic_optimizer.get("betas", [])) == [0.0, 0.999],
        "train.critic_optimizer.warmup_steps": int(critic_optimizer.get("warmup_steps", -1)) == 0,
        "train.critic_optimizer.min_lr_ratio": float(critic_optimizer.get("min_lr_ratio", -1))
        == 1.0,
        "train.ema.update_every": int(ema.get("update_every", -1)) == 1,
        # Training uses FlexAttention for the critic and teacher windows.  The
        # deployment sampler only executes cached three-latent windows and must
        # not pretend that the compiler switch is a checkpoint requirement.
        "runtime.compile_flex": (
            bool(_mapping(config, "runtime").get("compile_flex", False)) if training else True
        ),
    }
    failed = [field for field, accepted in required.items() if not accepted]
    if failed:
        raise BackendContractError(f"Stage2 DMD via SGF contract mismatch: {failed}")

    denoising = tuple(int(value) for value in train.get("denoising_step_list", []))
    if denoising != (1000, 750, 500, 250):
        raise BackendContractError("Stage2 train.denoising_step_list must be [1000,750,500,250]")
    if int(train.get("critic_updates_per_student", 0)) != 5:
        raise BackendContractError("Stage2 critic_updates_per_student must be 5")
    if (
        not 0
        <= float(train.get("score_min_timestep", -1))
        < float(train.get("score_max_timestep", -1))
        <= 1000
    ):
        raise BackendContractError("Stage2 score timestep bounds must be within [0,1000]")

    passes = validation.get("passes", [])
    if not isinstance(passes, list) or not passes:
        raise BackendContractError("Stage2 validation requires at least one pass")
    weights = [str(item.get("weights", "")) for item in passes if isinstance(item, Mapping)]
    camera_length = not training and str(inference.get("length", "fixed")) == "camera"
    allowed_weights = (
        (["live"], ["live", "ema"])
        if training
        else ((["model"],) if camera_length else (["live", "ema"],))
    )
    if weights not in allowed_weights:
        if camera_length:
            raise BackendContractError(
                "camera-length Stage2 inference requires one direct model pass"
            )
        raise BackendContractError(
            "Stage2 training validation must be LIVE-only or LIVE then EMA; "
            "standalone fixed-length inference requires LIVE then EMA"
        )
    for item in passes:
        if (
            not isinstance(item, Mapping)
            or item.get("solver") != "self_forcing"
            or int(item.get("num_inference_steps", 0)) != 4
        ):
            raise BackendContractError("Stage2 validation uses self_forcing NFE4 only")
        if not camera_length and int(item.get("rollout_latent_frames", 0)) != 60:
            raise BackendContractError(
                "Stage2 validation requires the common 60-latent collective horizon"
            )

    # A training launch initializes three independent roles.  Standalone
    # inference instead opens the atomic Stage2 ``model.pt`` commit marker and
    # selects its LIVE/EMA payload, so requiring the initialization roles there
    # would make a valid deployment config impossible to express.
    if not training:
        return ()

    roles_node = checkpoint.get("roles", {})
    if not isinstance(roles_node, Mapping):
        raise BackendContractError("checkpoint.roles must be a mapping")
    roles = tuple(_role(name, roles_node.get(name)) for name in ("student", "teacher", "critic"))
    validate_checkpoint_transaction(checkpoint.get("transaction_members", ()))
    camera_transform = str(model.get("camera_translation_transform", "linear"))
    for role in roles:
        if role.camera_translation_transform != camera_transform:
            raise BackendContractError(
                f"Stage2 {role.role} camera translation transform does not match the run"
            )
    student, teacher, critic = roles
    if (student.weights, student.expected_stage, student.expected_objective) != (
        "ema",
        "stage1",
        "anyflow_forward_map:v1_5",
    ) or not student.allow_anyflow_delta_drop:
        raise BackendContractError(
            "Stage2 student must use Stage1 AnyFlow-v1.5 EMA with the exact "
            "four-tensor delta-drop policy"
        )
    for role in (teacher, critic):
        if (
            role.weights not in {"live", "ema"}
            or role.expected_stage != "stage0p5"
            or role.expected_objective != "flow_matching"
        ):
            raise BackendContractError(
                f"Stage2 {role.role} must use explicit Stage0.5 FM LIVE or EMA weights"
            )
    if teacher.weights != critic.weights:
        raise BackendContractError("Stage2 teacher and critic must select the same weight source")
    return roles


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("SGF math requires the 'wan' dependency extra") from exc
    return torch


def reference_cfg(cond: Any, uncond: Any, guidance_scale: float) -> Any:
    return cond + float(guidance_scale) * (cond - uncond)


def warp_denoising_steps(
    raw_steps: Sequence[int | float],
    scheduler_timesteps: Any,
    *,
    num_train_timesteps: int = 1000,
) -> tuple[Any, ...]:
    """Map public raw Wan labels onto the shifted scheduler table."""

    torch = _torch()
    table = torch.as_tensor(scheduler_timesteps).detach().to(dtype=torch.float32)
    if table.ndim != 1 or table.numel() != int(num_train_timesteps):
        raise ValueError(
            "scheduler_timesteps must contain the complete one-dimensional training table"
        )
    table = torch.cat([table, table.new_zeros(1)])
    result = []
    for raw in raw_steps:
        value = float(raw)
        rounded = round(value)
        if abs(value - rounded) > 1.0e-6 or not 0 <= rounded <= int(num_train_timesteps):
            raise ValueError("SGF denoising labels must be integers in [0,N]")
        result.append(table[int(num_train_timesteps) - rounded].reshape(()))
    if result and float(result[-1].item()) == 0.0:
        result.pop()
    if not result:
        raise ValueError("SGF denoising schedule has no non-zero step")
    return tuple(result)


def sample_shifted_score_timesteps(
    *,
    batch_size: int,
    num_frames: int,
    device: Any,
    shift: float,
    num_train_timesteps: int = 1000,
    min_timestep: float = 20.0,
    max_timestep: float = 980.0,
    generator: Any | None = None,
) -> Any:
    """Sample uniform raw score labels and apply the Wan rational shift."""

    torch = _torch()
    if batch_size < 1 or num_frames < 1:
        raise ValueError("SGF score timestep dimensions must be positive")
    if not 0 <= min_timestep < max_timestep <= num_train_timesteps:
        raise ValueError("SGF score timestep bounds must satisfy 0 <= min < max <= N")
    raw = torch.randint(
        0,
        int(num_train_timesteps),
        (int(batch_size), 1),
        device=device,
        dtype=torch.long,
        generator=generator,
    ).to(torch.float32)
    fraction = raw / float(num_train_timesteps)
    shifted = float(shift) * fraction / (1.0 + (float(shift) - 1.0) * fraction)
    shifted = (shifted * float(num_train_timesteps)).clamp(float(min_timestep), float(max_timestep))
    return shifted.expand(-1, int(num_frames)).clone()


def compute_kl_gradient_array(
    *, fake_x0: Any, real_x0: Any, student_output: Any, mask: Any = None, normalize: bool = True
) -> np.ndarray:
    """NumPy reference for masked SGF gradient fixture tests."""

    fake = np.asarray(fake_x0, dtype=np.float32)
    real = np.asarray(real_x0, dtype=np.float32)
    output = np.asarray(student_output, dtype=np.float32)
    if fake.shape != real.shape or fake.shape != output.shape:
        raise ValueError("fake_x0, real_x0 and student_output must have identical shapes")
    grad = fake - real
    expanded = None
    if mask is not None:
        try:
            expanded = np.broadcast_to(np.asarray(mask, dtype=np.bool_), fake.shape)
        except ValueError as exc:
            raise ValueError("mask must broadcast to the input shape") from exc
        grad = np.where(expanded, grad, np.float32(0))
    if normalize:
        distance = np.abs(output - real)
        dims = tuple(range(1, distance.ndim))
        if expanded is None:
            normalizer = np.mean(distance, axis=dims, keepdims=True)
        else:
            distance = np.where(expanded, distance, np.float32(0))
            denominator = np.maximum(np.sum(expanded, axis=dims, keepdims=True), 1)
            normalizer = np.sum(distance, axis=dims, keepdims=True) / denominator
        grad = grad / np.maximum(normalizer, np.float32(1e-8))
    return np.nan_to_num(grad)


def compute_kl_gradient(
    *, fake_x0: Any, real_x0: Any, student_output: Any, mask: Any = None, normalize: bool = True
) -> Any:
    """Compute the masked reference DMD gradient in FP32."""

    torch = _torch()
    fake = torch.as_tensor(fake_x0).float()
    real = torch.as_tensor(real_x0, device=fake.device).float()
    output = torch.as_tensor(student_output, device=fake.device).float()
    if fake.shape != real.shape or fake.shape != output.shape:
        raise ValueError("fake_x0, real_x0 and student_output must have identical shapes")
    grad = fake - real
    expanded = None
    if mask is not None:
        expanded = torch.as_tensor(mask, device=fake.device, dtype=torch.bool)
        try:
            expanded = torch.broadcast_to(expanded, fake.shape)
        except RuntimeError as exc:
            raise ValueError("mask must broadcast to the input shape") from exc
        grad = torch.where(expanded, grad, torch.zeros_like(grad))
    if normalize:
        distance = (output - real).abs()
        dims = tuple(range(1, distance.ndim))
        if expanded is None:
            normalizer = distance.mean(dim=dims, keepdim=True)
        else:
            distance = torch.where(expanded, distance, torch.zeros_like(distance))
            normalizer = distance.sum(dim=dims, keepdim=True) / expanded.float().sum(
                dim=dims, keepdim=True
            ).clamp_min(1.0)
        grad = grad / normalizer.clamp_min(1e-8)
    return torch.nan_to_num(grad)


def sgf_student_loss(student_output: Any, kl_gradient: Any, *, mask: Any = None) -> Any:
    """Surrogate whose derivative with respect to output is the SGF gradient."""

    torch = _torch()
    output = torch.as_tensor(student_output).float()
    gradient = torch.as_tensor(kl_gradient, device=output.device).float()
    if output.shape != gradient.shape:
        raise ValueError("student_output and kl_gradient must have identical shapes")
    target = (output - gradient.detach()).detach()
    diff = output - target
    if mask is None:
        return 0.5 * diff.square().mean()
    try:
        valid = torch.broadcast_to(
            torch.as_tensor(mask, device=output.device, dtype=torch.bool),
            output.shape,
        )
    except RuntimeError as exc:
        raise ValueError("SGF student mask does not broadcast to output") from exc
    diff = torch.where(valid, diff, torch.zeros_like(diff))
    return 0.5 * diff.square().sum() / valid.float().sum().clamp_min(1.0)


def sgf_critic_flow_loss(
    flow_prediction: Any,
    *,
    noise: Any,
    clean: Any,
    mask: Any = None,
) -> Any:
    """Flow-matching critic loss with mask-before-subtraction semantics."""

    torch = _torch()
    prediction = torch.as_tensor(flow_prediction).float()
    noise_value = torch.as_tensor(noise, device=prediction.device).float()
    clean_value = torch.as_tensor(clean, device=prediction.device).float()
    if prediction.shape != noise_value.shape or prediction.shape != clean_value.shape:
        raise ValueError("SGF critic tensors must have identical shapes")
    diff = prediction - (noise_value - clean_value)
    if mask is None:
        return diff.square().mean()
    try:
        valid = torch.broadcast_to(
            torch.as_tensor(mask, device=prediction.device, dtype=torch.bool),
            prediction.shape,
        )
    except RuntimeError as exc:
        raise ValueError("SGF critic mask does not broadcast to prediction") from exc
    diff = torch.where(valid, diff, torch.zeros_like(diff))
    return diff.square().sum() / valid.float().sum().clamp_min(1.0)


__all__ = [
    "STAGE2_CHECKPOINT_MEMBERS",
    "RoleInitialization",
    "compute_kl_gradient",
    "compute_kl_gradient_array",
    "normalize_weight_source",
    "reference_cfg",
    "sample_shifted_score_timesteps",
    "sgf_critic_flow_loss",
    "sgf_student_loss",
    "should_update_student",
    "student_update_steps",
    "validate_checkpoint_transaction",
    "validate_stage2_contract",
    "warp_denoising_steps",
]
