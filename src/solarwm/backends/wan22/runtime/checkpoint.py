"""Exact Wan full-state checkpoint commit and restore."""

from __future__ import annotations

import random
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from solarwm.checkpoint import (
    CheckpointContract,
    CheckpointTransaction,
    assert_resume_compatible,
    verify_checkpoint,
)
from solarwm.errors import BackendContractError
from solarwm.runtime.output_layout import checkpoint_model_dir
from solarwm.runtime.safe_state import (
    decode_numpy_rng_state,
    decode_python_rng_state,
    encode_numpy_rng_state,
    encode_python_rng_state,
)

_MODEL_PREFIX = "model."


@dataclass(frozen=True)
class RestoredWanCheckpoint:
    step: int
    identity: str
    path: Path
    standalone: bool
    optimizer_parameter_names: tuple[str, ...]


@dataclass(frozen=True)
class RestoredWanWeights:
    """Resolved identity of a weights-only Stage0.5 EMA initialization."""

    source_step: int
    identity: str
    path: Path
    standalone: bool


def _standalone_checkpoint_id(path: Path) -> str:
    size = path.stat().st_size
    if size <= 0:
        raise BackendContractError(f"Wan standalone checkpoint is empty: {path}")
    return f"inventory:file={path.name}:bytes={size}"


def checkpoint_contract(config: Mapping[str, Any]) -> CheckpointContract:
    """Bind every algorithm-bearing Stage0.5 field while allowing corpus refreshes."""

    model = config["model"]
    data = config["data"]
    train = config["train"]
    distributed = config["distributed"]
    optimizer = train["optimizer"]
    ema = train["ema"]
    fsdp = train["fsdp"]
    data_generation = (
        f"{data['encoding']}:{data.get('preencode_schema', 'online')}:"
        f"{data['pixel_frames']}f:{data['height']}x{data['width']}"
    )
    extras: dict[str, Any] = {
        "model_name": str(model["name"]),
        "timestep_shift": float(model["timestep_shift"]),
        "latent_channels": int(model["latent_channels"]),
        "frame_sequence_length": int(model["frame_sequence_length"]),
        "num_output_frames": int(model["num_output_frames"]),
        "local_attn_size": int(model["local_attn_size"]),
        "raw_world_size": int(distributed["world_size"]),
        "global_batch_size": int(train["global_batch_size"]),
        "max_steps": int(train["max_steps"]),
        "warmup_steps": int(train["warmup_steps"]),
        "num_train_timesteps": int(train["num_train_timesteps"]),
        "timestep_mode": str(train["timestep_mode"]),
        "i2v_image_condition_dropout": float(train["i2v_image_condition_dropout"]),
        "optimizer": {
            "lr": float(optimizer["lr"]),
            "betas": [float(value) for value in optimizer["betas"]],
            "eps": float(optimizer["eps"]),
            "weight_decay": float(optimizer["weight_decay"]),
        },
        "ema": {
            "decay": float(ema["decay"]),
            "start_step": int(ema["start_step"]),
            "update_every": int(ema.get("update_every", 1)),
        },
        "activation_checkpointing": bool(fsdp["activation_checkpointing"]),
    }
    if str(train["objective"]) == "anyflow_forward_map":
        anyflow_contract = {
            "variant": str(train["anyflow_variant"]),
            "weight_type": str(train["anyflow_weight_type"]),
            "gate": float(train["anyflow_gate"]),
            "deltatime_type": str(train["anyflow_deltatime_type"]),
            "epsilon": float(train["anyflow_epsilon"]),
            "diffusion_ratio": float(train["anyflow_diffusion_ratio"]),
            "consistency_ratio": float(train["anyflow_consistency_ratio"]),
            "fuse_guidance_scale": float(train["anyflow_fuse_guidance_scale"]),
            "negative_embedding": str(train["anyflow_negative_embedding"]),
        }
        extras["anyflow_v1_5"] = anyflow_contract
    return CheckpointContract(
        family=str(model["family"]),
        stage=str(train["stage"]),
        causal_mode=str(train["causal_mode"]),
        objective=str(train["objective"]),
        objective_variant=str(train.get("objective_variant", "standard")),
        camera_translation_transform=str(model["camera_translation_transform"]),
        parameterization="full-parameter",
        sp_size=int(distributed["sequence_parallel_size"]),
        data_generation=data_generation,
        extras=extras,
    )


def _strip_model_prefix_key(key: str) -> str:
    return key[len(_MODEL_PREFIX) :] if key.startswith(_MODEL_PREFIX) else key


def normalize_model_state(
    state: Mapping[str, Any],
    *,
    field: str,
) -> OrderedDict[str, Any]:
    """Normalize the optional root ``model.`` namespace."""

    if not isinstance(state, Mapping) or not state:
        raise BackendContractError(f"Wan checkpoint {field} must be a non-empty mapping")
    keys = tuple(state)
    if not all(isinstance(key, str) and key for key in keys):
        raise BackendContractError(f"Wan checkpoint {field} keys must be strings")
    prefixed = tuple(key.startswith(_MODEL_PREFIX) for key in keys)
    if any(prefixed) and not all(prefixed):
        raise BackendContractError(
            f"Wan checkpoint {field} mixes model-prefixed and unprefixed keys"
        )
    normalized = OrderedDict((_strip_model_prefix_key(key), value) for key, value in state.items())
    if len(normalized) != len(state):
        raise BackendContractError(f"Wan checkpoint {field} key normalization produced duplicates")
    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        normalized._metadata = {  # type: ignore[attr-defined]
            _strip_model_prefix_key(str(key)): value for key, value in metadata.items()
        }
    return normalized


def normalize_optimizer_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize both FSDP full-state name surfaces, never just model weights."""

    if not isinstance(state, Mapping):
        raise BackendContractError("Wan checkpoint optimizer must be a mapping")
    raw_slots = state.get("state")
    raw_groups = state.get("param_groups")
    if not isinstance(raw_slots, Mapping) or not isinstance(raw_groups, Sequence):
        raise BackendContractError("Wan checkpoint optimizer state is incomplete")
    named_keys = [key for key in raw_slots if isinstance(key, str)]
    group_names: list[str] = []
    for raw_group in raw_groups:
        if isinstance(raw_group, Mapping):
            params = raw_group.get("params", ())
            if isinstance(params, Sequence) and not isinstance(params, (str, bytes)):
                group_names.extend(value for value in params if isinstance(value, str))
    all_names = [*named_keys, *group_names]
    prefix_modes = {name.startswith(_MODEL_PREFIX) for name in all_names}
    if len(prefix_modes) > 1:
        raise BackendContractError(
            "Wan optimizer mixes model-prefixed and unprefixed parameter names"
        )

    slots: dict[Any, Any] = {}
    for key, value in raw_slots.items():
        normalized = _strip_model_prefix_key(key) if isinstance(key, str) else key
        if normalized in slots:
            raise BackendContractError("Wan optimizer state key normalization produced duplicates")
        slots[normalized] = value
    groups: list[dict[str, Any]] = []
    for index, raw_group in enumerate(raw_groups):
        if not isinstance(raw_group, Mapping):
            raise BackendContractError(f"Wan optimizer parameter group {index} is not a mapping")
        params = raw_group.get("params")
        if not isinstance(params, Sequence) or isinstance(params, (str, bytes)):
            raise BackendContractError(
                f"Wan optimizer parameter group {index} has no parameter sequence"
            )
        normalized_params = [
            _strip_model_prefix_key(value) if isinstance(value, str) else value for value in params
        ]
        if len(set(normalized_params)) != len(normalized_params):
            raise BackendContractError(
                f"Wan optimizer parameter group {index} normalization produced duplicates"
            )
        group = dict(raw_group)
        group["params"] = normalized_params
        groups.append(group)
    slot_names = {key for key in slots if isinstance(key, str)}
    parameter_names = {key for group in groups for key in group["params"] if isinstance(key, str)}
    if slot_names and slot_names != parameter_names:
        missing = sorted(parameter_names - slot_names)
        unexpected = sorted(slot_names - parameter_names)
        raise BackendContractError(
            "Wan optimizer slot/parameter names differ after normalization: "
            f"missing_slots={missing[:8]} unexpected_slots={unexpected[:8]}"
        )
    return {"state": slots, "param_groups": groups}


def optimizer_parameter_names(state: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the readable ordered parameter inventory used for exact resume."""

    normalized = normalize_optimizer_state(state)
    return tuple(name for group in normalized["param_groups"] for name in group["params"])


def _strip_parameter_wrapper_segments(name: str) -> str:
    wrappers = {"_fsdp_wrapped_module", "_checkpoint_wrapped_module"}
    return ".".join(part for part in name.split(".") if part not in wrappers)


def _assert_optimizer_names_match_module(state: Mapping[str, Any], module: Any) -> None:
    normalized = normalize_optimizer_state(state)
    checkpoint_names = tuple(
        name for group in normalized["param_groups"] for name in group["params"]
    )
    if not all(isinstance(name, str) and name for name in checkpoint_names):
        raise BackendContractError("Wan checkpoint optimizer parameters must use explicit names")
    runtime_names = tuple(
        _strip_parameter_wrapper_segments(_strip_model_prefix_key(name))
        for name, _ in _named_parameters(module)
    )
    if checkpoint_names != runtime_names:
        missing = sorted(set(runtime_names) - set(checkpoint_names))
        unexpected = sorted(set(checkpoint_names) - set(runtime_names))
        raise BackendContractError(
            "Wan checkpoint optimizer names differ from the runtime model: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _distributed_context() -> tuple[Any, int, int]:
    import torch.distributed as dist

    if dist.is_initialized():
        return dist, dist.get_rank(), dist.get_world_size()
    return dist, 0, 1


def _collective_error(local_error: str | None, *, phase: str) -> None:
    dist, rank, world_size = _distributed_context()
    if world_size > 1:
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, (rank, local_error))
        failures = [item for item in gathered if item[1] is not None]
    else:
        failures = [(rank, local_error)] if local_error is not None else []
    if failures:
        rendered = "; ".join(f"rank={item[0]}: {item[1]}" for item in failures)
        raise BackendContractError(f"Wan checkpoint {phase} failed collectively: {rendered}")


def _local_rank_runtime_state(reader: Any, device: Any) -> dict[str, Any]:
    import torch

    state_dict = getattr(reader, "checkpoint_state_dict", None)
    if not callable(state_dict):
        state_dict = getattr(reader, "state_dict", None)
    if not callable(state_dict):
        raise BackendContractError("Wan checkpoint requires a stateful data reader")
    _, rank, world_size = _distributed_context()
    return {
        "schema": "solarwm.wan22.rank-runtime.v1",
        "rank": int(rank),
        "world_size": int(world_size),
        "reader": state_dict(),
        "python_rng": encode_python_rng_state(random.getstate()),
        "numpy_rng": encode_numpy_rng_state(np.random.get_state()),
        "torch_cpu_rng": torch.get_rng_state().cpu(),
        "torch_cuda_rng": (
            torch.cuda.get_rng_state(device).cpu() if torch.device(device).type == "cuda" else None
        ),
    }


def _gather_rank_runtime_states(reader: Any, device: Any) -> list[dict[str, Any]]:
    local = _local_rank_runtime_state(reader, device)
    dist, _, world_size = _distributed_context()
    if world_size == 1:
        return [local]
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, local)
    return list(gathered)


def _restore_rank_runtime_state(states: Any, *, reader: Any, device: Any) -> None:
    """Restore exact per-rank reader and RNG state."""

    import torch

    if states is None:
        raise BackendContractError("Wan full resume requires rank-local reader and RNG state")
    dist, rank, world_size = _distributed_context()
    del dist
    if not isinstance(states, list) or len(states) != world_size:
        raise BackendContractError("Wan checkpoint raw-rank runtime state count differs")
    selected = states[rank]
    if (
        not isinstance(selected, Mapping)
        or selected.get("schema") != "solarwm.wan22.rank-runtime.v1"
        or int(selected.get("rank", -1)) != rank
        or int(selected.get("world_size", -1)) != world_size
    ):
        raise BackendContractError(f"Wan checkpoint runtime state for rank {rank} is invalid")
    load_state_dict = getattr(reader, "load_state_dict", None)
    if not callable(load_state_dict):
        raise BackendContractError("Wan resume requires a stateful data reader")
    load_state_dict(selected.get("reader", {}))
    random.setstate(decode_python_rng_state(selected.get("python_rng")))
    np.random.set_state(decode_numpy_rng_state(selected.get("numpy_rng")))
    cpu_state = selected.get("torch_cpu_rng")
    if (
        not isinstance(cpu_state, torch.Tensor)
        or cpu_state.dtype != torch.uint8
        or cpu_state.ndim != 1
        or cpu_state.numel() == 0
    ):
        raise BackendContractError("Wan checkpoint CPU RNG state is invalid")
    torch.set_rng_state(cpu_state)
    cuda_state = selected.get("torch_cuda_rng")
    if torch.device(device).type == "cuda":
        if (
            not isinstance(cuda_state, torch.Tensor)
            or cuda_state.dtype != torch.uint8
            or cuda_state.ndim != 1
            or cuda_state.numel() == 0
        ):
            raise BackendContractError("Wan checkpoint CUDA RNG state is invalid")
        torch.cuda.set_rng_state(cuda_state, device)


def _broadcast_text(value: str) -> str:
    dist, _, world_size = _distributed_context()
    payload = [value]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    return str(payload[0])


def _is_fsdp(module: Any) -> bool:
    from torch.distributed.fsdp import FullyShardedDataParallel

    return isinstance(module, FullyShardedDataParallel)


def _load_full_model_state(module: Any, state: Mapping[str, Any]) -> None:
    if _is_fsdp(module):
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        options = FullStateDictConfig(rank0_only=False, offload_to_cpu=False)
        with FullyShardedDataParallel.state_dict_type(
            module, StateDictType.FULL_STATE_DICT, options
        ):
            result = module.load_state_dict(state, strict=True)
    else:
        result = module.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise BackendContractError(
            "Wan full-state load was not exact: "
            f"missing={result.missing_keys[:8]} unexpected={result.unexpected_keys[:8]}"
        )


def _load_model_state_result(
    module: Any,
    state: Mapping[str, Any],
    *,
    strict: bool,
) -> Any:
    """Run one root-aware full-state load and return PyTorch's key report."""

    if _is_fsdp(module):
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        options = FullStateDictConfig(rank0_only=False, offload_to_cpu=False)
        with FullyShardedDataParallel.state_dict_type(
            module, StateDictType.FULL_STATE_DICT, options
        ):
            return module.load_state_dict(state, strict=bool(strict))
    return module.load_state_dict(state, strict=bool(strict))


def _load_stage0p5_state_into_anyflow(module: Any, state: Mapping[str, Any]) -> tuple[str, ...]:
    """Upgrade FM Stage0.5 weights for AnyFlow-v1.5.

    Every shared tensor must match.  The only missing tensors are the four
    objective-only ``delta_embedding`` values, each initialized from its
    corresponding loaded ``time_embedding`` tensor.
    """

    time_keys = tuple(sorted(str(key) for key in state if str(key).startswith("time_embedding.")))
    delta = tuple(key.replace("time_embedding.", "delta_embedding.", 1) for key in time_keys)
    if len(delta) != 4 or any(key in state for key in delta):
        raise BackendContractError(
            "Wan FM -> AnyFlow initialization permits exactly four missing "
            "delta_embedding tensors and no pre-existing delta state"
        )
    upgraded = OrderedDict(state.items())
    metadata = getattr(state, "_metadata", None)
    if metadata is not None:
        upgraded._metadata = metadata  # type: ignore[attr-defined]
    for time_key, delta_key in zip(time_keys, delta, strict=True):
        upgraded[delta_key] = state[time_key].detach().clone()
    try:
        exact = _load_model_state_result(module, upgraded, strict=True)
    except Exception as exc:
        raise BackendContractError(
            "Wan FM -> AnyFlow initialization permits exactly four missing "
            "delta_embedding tensors and no other mismatch"
        ) from exc
    if exact.missing_keys or exact.unexpected_keys:
        raise BackendContractError(
            "Wan upgraded AnyFlow state was not exact after delta initialization"
        )
    return tuple(sorted(delta))


def _named_parameters(module: Any) -> list[tuple[str, Any]]:
    root = getattr(module, "module", module)
    return list(root.named_parameters())


def _load_full_ema_state(
    module: Any,
    ema: Any,
    state: Mapping[str, Any],
    *,
    num_updates: int,
) -> None:
    live = {name: parameter.detach().clone() for name, parameter in _named_parameters(module)}
    try:
        _load_full_model_state(module, state)
        local_ema = {
            name: parameter.detach().clone() for name, parameter in _named_parameters(module)
        }
        ema.load_state_dict(local_ema, num_updates=int(num_updates))
    finally:
        import torch

        with torch.no_grad():
            for name, parameter in _named_parameters(module):
                if name not in live:
                    raise BackendContractError(
                        f"Wan live parameter {name!r} disappeared during EMA restore"
                    )
                parameter.copy_(live[name].to(device=parameter.device, dtype=parameter.dtype))


def _payload_config_contract(payload: Mapping[str, Any]) -> dict[str, str]:
    saved = payload.get("config")
    if not isinstance(saved, Mapping):
        raise BackendContractError("Wan checkpoint config must be a mapping")
    model = saved.get("model")
    train = saved.get("train")
    if not isinstance(model, Mapping) or not isinstance(train, Mapping):
        raise BackendContractError("Wan checkpoint config lacks model or train")
    return {
        "family": str(model.get("family", "")),
        "stage": str(train.get("stage", "")),
        "objective": str(train.get("objective", "")),
        "camera_translation_transform": str(model.get("camera_translation_transform", "linear")),
    }


def _assert_payload_contract(
    payload: Mapping[str, Any],
    *,
    expected: Mapping[str, str],
    label: str,
) -> None:
    actual = _payload_config_contract(payload)
    drift = {
        field: {"actual": actual.get(field), "expected": value}
        for field, value in expected.items()
        if actual.get(field) != value
    }
    if drift:
        raise BackendContractError(f"{label} contract mismatch: {drift}")


def _validate_standalone_full_resume(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> int:
    step = _validate_current_full_resume(payload, config)
    _assert_payload_contract(
        payload,
        expected={
            "family": str(config["model"]["family"]),
            "stage": str(config["train"]["stage"]),
            "objective": str(config["train"]["objective"]),
            "camera_translation_transform": str(config["model"]["camera_translation_transform"]),
        },
        label="Wan standalone full-resume source",
    )
    return step


def _validate_current_full_resume(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> int:
    try:
        step = int(payload["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError("Wan checkpoint has no valid global_step") from exc
    if step != int(config["checkpoint"]["resume_step"]):
        raise BackendContractError("Wan checkpoint global_step differs from checkpoint.resume_step")
    for field in ("generator", "generator_ema", "optimizer", "scheduler"):
        value = payload.get(field)
        if not isinstance(value, Mapping) or not value:
            raise BackendContractError(f"Wan full resume requires non-empty {field}")
    try:
        ema_updates = int(payload["ema_num_updates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError("Wan checkpoint has no EMA update count") from exc
    if ema_updates != step:
        raise BackendContractError("Wan checkpoint EMA update count differs from step")
    return step


def _resolve_input_checkpoint(
    path: str | Path,
    config: Mapping[str, Any],
) -> tuple[Path, str, bool]:
    source = Path(path).expanduser().resolve()
    dist, rank, _ = _distributed_context()
    result = ["", "", False]
    error = [""]
    if rank == 0:
        try:
            if source.is_dir():
                verified = verify_checkpoint(source)
                assert_resume_compatible(checkpoint_contract(config), verified.contract)
                result = [
                    str(verified.path / "model.pt"),
                    f"digest:{verified.manifest_digest}",
                    False,
                ]
            elif source.is_file():
                raise BackendContractError(
                    "Wan full resume requires the complete checkpoint transaction directory"
                )
            else:
                raise BackendContractError(f"Wan checkpoint is missing: {source}")
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
        dist.broadcast_object_list(result, src=0)
    if error[0]:
        raise BackendContractError(f"Wan checkpoint resolution failed: {error[0]}")
    resolved = Path(str(result[0]))
    local_error = None if resolved.is_file() else f"checkpoint is not visible: {resolved}"
    _collective_error(local_error, phase="visibility")
    return resolved, str(result[1]), bool(result[2])


def _weights_source_contract_matches(actual: CheckpointContract, config: Mapping[str, Any]) -> None:
    source = config["checkpoint"]["source_contract"]
    expected = {
        "family": str(config["model"]["family"]),
        "stage": str(source["stage"]),
        "causal_mode": "bidirectional",
        "objective": str(source["objective"]),
        "camera_translation_transform": str(source["camera_translation_transform"]),
        "parameterization": "full-parameter",
    }
    drift = {
        field: {"actual": getattr(actual, field), "expected": value}
        for field, value in expected.items()
        if getattr(actual, field) != value
    }
    if drift:
        raise BackendContractError(f"Wan weights-only source contract mismatch: {drift}")


def _resolve_weights_checkpoint(
    path: str | Path,
    config: Mapping[str, Any],
) -> tuple[Path, str, bool, int]:
    """Resolve a Stage0.5 source without comparing it to the Stage1 contract."""

    source = Path(path).expanduser().resolve()
    dist, rank, _ = _distributed_context()
    result: list[Any] = ["", "", False, 0]
    error = [""]
    if rank == 0:
        try:
            expected_step = int(config["checkpoint"]["source_step"])
            if source.is_dir():
                verified = verify_checkpoint(source)
                _weights_source_contract_matches(verified.contract, config)
                if int(verified.step) != expected_step:
                    raise BackendContractError(
                        "Wan weights-only source step differs from checkpoint.source_step: "
                        f"{verified.step} != {expected_step}"
                    )
                result = [
                    str(verified.path / "model.pt"),
                    f"digest:{verified.manifest_digest}",
                    False,
                    int(verified.step),
                ]
            elif source.is_file():
                result = [
                    str(source),
                    _standalone_checkpoint_id(source),
                    True,
                    expected_step,
                ]
            else:
                raise BackendContractError(f"Wan weights source is missing: {source}")
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
        dist.broadcast_object_list(result, src=0)
    if error[0]:
        raise BackendContractError(f"Wan weights source resolution failed: {error[0]}")
    resolved = Path(str(result[0]))
    local_error = None if resolved.is_file() else f"weights source is not visible: {resolved}"
    _collective_error(local_error, phase="weights-source visibility")
    return resolved, str(result[1]), bool(result[2]), int(result[3])


def _validate_standalone_weights_source(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    source_step: int,
) -> Mapping[str, Any]:
    try:
        actual_step = int(payload["global_step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError("Wan weights source has no valid global_step") from exc
    if actual_step != int(source_step):
        raise BackendContractError(f"Wan weights source step {actual_step} != {source_step}")
    source = config["checkpoint"]["source_contract"]
    _assert_payload_contract(
        payload,
        expected={
            "family": str(config["model"]["family"]),
            "stage": str(source["stage"]),
            "objective": str(source["objective"]),
            "camera_translation_transform": str(source["camera_translation_transform"]),
        },
        label="Wan standalone weights source",
    )
    state = payload.get("generator_ema")
    if not isinstance(state, Mapping) or not state:
        raise BackendContractError("Wan weights-only initialization requires generator_ema")
    try:
        ema_updates = int(payload["ema_num_updates"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BackendContractError("Wan weights source has no EMA update count") from exc
    if ema_updates != actual_step:
        raise BackendContractError("Wan weights source EMA update count differs from global step")
    return state


def load_weights_only_checkpoint(
    *,
    config: Mapping[str, Any],
    path: str | Path,
    diffusion: Any,
) -> RestoredWanWeights:
    """Load only Stage0.5 EMA weights, leaving all Stage1 state fresh."""

    import torch

    resolved, identity, standalone, source_step = _resolve_weights_checkpoint(path, config)
    state: Mapping[str, Any] | None = None
    local_error: str | None = None
    try:
        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(loaded, Mapping):
            raise BackendContractError("Wan weights source payload must be a mapping")
        if standalone:
            raw_state = _validate_standalone_weights_source(
                loaded,
                config,
                source_step=source_step,
            )
        else:
            if int(loaded.get("global_step", -1)) != source_step:
                raise BackendContractError(
                    "Wan transactional weights source step differs from its manifest"
                )
            raw_state = loaded.get("generator_ema")
            if not isinstance(raw_state, Mapping) or not raw_state:
                raise BackendContractError("Wan transactional weights source has no generator_ema")
        state = normalize_model_state(raw_state, field="generator_ema")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(local_error, phase="weights-only preload/contract")
    assert state is not None
    _load_full_model_state(diffusion.module, state)
    return RestoredWanWeights(source_step, identity, resolved, standalone)


def load_anyflow_weights_only_checkpoint(
    *,
    config: Mapping[str, Any],
    path: str | Path,
    diffusion: Any,
) -> tuple[RestoredWanWeights, tuple[str, ...]]:
    """Load Stage0.5 EMA into AnyFlow and initialize only its four deltas."""

    import torch

    resolved, identity, standalone, source_step = _resolve_weights_checkpoint(path, config)
    state: Mapping[str, Any] | None = None
    local_error: str | None = None
    try:
        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(loaded, Mapping):
            raise BackendContractError("Wan AnyFlow source payload must be a mapping")
        if standalone:
            raw_state = _validate_standalone_weights_source(
                loaded,
                config,
                source_step=source_step,
            )
        else:
            if int(loaded.get("global_step", -1)) != source_step:
                raise BackendContractError(
                    "Wan AnyFlow source step differs from its transaction manifest"
                )
            raw_state = loaded.get("generator_ema")
            if not isinstance(raw_state, Mapping) or not raw_state:
                raise BackendContractError("Wan AnyFlow source has no generator_ema")
        state = normalize_model_state(raw_state, field="generator_ema")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(local_error, phase="AnyFlow weights-only preload/contract")
    assert state is not None
    delta_keys: tuple[str, ...] = ()
    try:
        delta_keys = _load_stage0p5_state_into_anyflow(diffusion.module, state)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    else:
        local_error = None
    _collective_error(local_error, phase="AnyFlow objective upgrade")
    return (
        RestoredWanWeights(source_step, identity, resolved, standalone),
        delta_keys,
    )


def load_live_and_ema_weights_checkpoint(
    *,
    config: Mapping[str, Any],
    path: str | Path,
    diffusion: Any,
    ema: Any,
) -> RestoredWanWeights:
    """Initialize Stage0.5 LIVE/EMA selections while resetting training state."""

    import torch

    resolved, identity, standalone, source_step = _resolve_weights_checkpoint(path, config)
    live: Mapping[str, Any] | None = None
    ema_state: Mapping[str, Any] | None = None
    local_error: str | None = None
    try:
        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(loaded, Mapping):
            raise BackendContractError("Wan dual weights source payload must be a mapping")
        if standalone:
            raw_ema = _validate_standalone_weights_source(
                loaded,
                config,
                source_step=source_step,
            )
        else:
            if int(loaded.get("global_step", -1)) != source_step:
                raise BackendContractError(
                    "Wan transactional dual source step differs from its manifest"
                )
            raw_ema = loaded.get("generator_ema")
            if not isinstance(raw_ema, Mapping) or not raw_ema:
                raise BackendContractError("Wan dual source has no generator_ema")
        selections = list(config["checkpoint"].get("weights", []))
        if selections not in (["live", "ema"], ["ema", "ema"]):
            raise BackendContractError(
                "Wan Stage0.5 source must select LIVE/EMA or EMA/EMA weights"
            )
        raw_live = loaded.get("generator" if selections[0] == "live" else "generator_ema")
        if not isinstance(raw_live, Mapping) or not raw_live:
            raise BackendContractError("Wan Stage0.5 source has no selected LIVE weights")
        live = normalize_model_state(raw_live, field=selections[0])
        ema_state = normalize_model_state(raw_ema, field="generator_ema")
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(local_error, phase="dual weights preload/contract")
    assert live is not None and ema_state is not None
    _load_full_model_state(diffusion.module, live)
    _load_full_ema_state(diffusion.module, ema, ema_state, num_updates=0)
    return RestoredWanWeights(source_step, identity, resolved, standalone)


def load_full_checkpoint(
    *,
    config: Mapping[str, Any],
    path: str | Path,
    diffusion: Any,
    optimizer: Any,
    scheduler: Any,
    ema: Any,
    reader: Any,
    device: Any,
) -> RestoredWanCheckpoint:
    """Restore all algorithm-bearing state, including rank-local RNG and data position."""

    import torch

    resolved, identity, standalone = _resolve_input_checkpoint(path, config)
    payload: Mapping[str, Any] | None = None
    local_error: str | None = None
    try:
        loaded = torch.load(
            resolved,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        if not isinstance(loaded, Mapping):
            raise BackendContractError("Wan model.pt payload must be a mapping")
        payload = loaded
        step = (
            _validate_standalone_full_resume(payload, config)
            if standalone
            else _validate_current_full_resume(payload, config)
        )
        generator = normalize_model_state(payload["generator"], field="generator")
        generator_ema = normalize_model_state(payload["generator_ema"], field="generator_ema")
        optimizer_state = normalize_optimizer_state(payload["optimizer"])
        optimizer_names = optimizer_parameter_names(optimizer_state)
        _assert_optimizer_names_match_module(optimizer_state, diffusion.module)
        scheduler_state = payload["scheduler"]
        ema_updates = int(payload["ema_num_updates"])
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _collective_error(local_error, phase="preload/contract")
    assert payload is not None

    _load_full_model_state(diffusion.module, generator)
    _load_full_ema_state(
        diffusion.module,
        ema,
        generator_ema,
        num_updates=ema_updates,
    )
    try:
        if _is_fsdp(diffusion.module):
            from torch.distributed.fsdp import FullyShardedDataParallel

            local_optimizer = FullyShardedDataParallel.optim_state_dict_to_load(
                model=diffusion.module,
                optim=optimizer,
                optim_state_dict=optimizer_state,
            )
            optimizer.load_state_dict(local_optimizer)
        else:
            optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        if int(getattr(scheduler, "last_epoch", -1)) != step:
            raise BackendContractError(
                "restored Wan scheduler epoch differs from checkpoint step: "
                f"{getattr(scheduler, 'last_epoch', None)} != {step}"
            )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    else:
        local_error = None
    _collective_error(local_error, phase="optimizer/scheduler restore")
    try:
        _restore_rank_runtime_state(
            payload.get("rank_runtime_state"),
            reader=reader,
            device=device,
        )
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    else:
        local_error = None
    _collective_error(local_error, phase="rank runtime restore")
    return RestoredWanCheckpoint(step, identity, resolved, standalone, optimizer_names)


def _gather_full_state(module: Any, *, rank0_only: bool) -> Mapping[str, Any]:
    if _is_fsdp(module):
        from torch.distributed.fsdp import (
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        options = FullStateDictConfig(
            rank0_only=bool(rank0_only),
            offload_to_cpu=True,
        )
        with FullyShardedDataParallel.state_dict_type(
            module, StateDictType.FULL_STATE_DICT, options
        ):
            return module.state_dict()
    _, rank, _ = _distributed_context()
    if rank0_only and rank != 0:
        return {}
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _gather_full_optimizer(module: Any, optimizer: Any) -> Mapping[str, Any]:
    if _is_fsdp(module):
        from torch.distributed.fsdp import (
            FullOptimStateDictConfig,
            FullStateDictConfig,
            FullyShardedDataParallel,
            StateDictType,
        )

        with FullyShardedDataParallel.state_dict_type(
            module,
            StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            return FullyShardedDataParallel.optim_state_dict(module, optimizer)
    _, rank, _ = _distributed_context()
    return optimizer.state_dict() if rank == 0 else {}


def save_full_checkpoint(
    *,
    config: Mapping[str, Any],
    step: int,
    diffusion: Any,
    optimizer: Any,
    scheduler: Any,
    ema: Any,
    reader: Any,
    device: Any,
) -> str:
    """Collectively create one full-state payload inside an atomic store."""

    import torch

    dist, rank, _ = _distributed_context()
    if int(step) < 1:
        raise BackendContractError("Wan checkpoint step must be positive")
    target = checkpoint_model_dir(
        str(config["runtime"]["output_dir"]),
        step=int(step),
        width=6,
    )
    rank_runtime_state = _gather_rank_runtime_states(reader, device)
    live = _gather_full_state(diffusion.module, rank0_only=True)
    optimizer_state = _gather_full_optimizer(diffusion.module, optimizer)
    with ema.swapped_into(diffusion.module):
        ema_state = _gather_full_state(diffusion.module, rank0_only=True)
    if dist.is_initialized():
        dist.barrier()

    identity = [""]
    error = [""]
    if rank == 0:
        try:
            with CheckpointTransaction(target) as transaction:
                payload = {
                    "generator": live,
                    "generator_ema": ema_state,
                    "optimizer": normalize_optimizer_state(optimizer_state),
                    "scheduler": scheduler.state_dict(),
                    "global_step": int(step),
                    "config": _plain(config),
                    "ema_num_updates": int(ema.num_updates),
                    "rank_runtime_state": rank_runtime_state,
                }
                torch.save(payload, transaction.path / "model.pt")
                committed = transaction.commit(
                    step=int(step),
                    contract=checkpoint_contract(config),
                    required_components=("model.pt",),
                    metadata={
                        "schema": "solarwm.wan22-full-state.v1",
                        "roles": {
                            "generator": "live_full_state",
                            "generator_ema": "fp32_ema_full_state",
                            "optimizer": "fsdp_full_optimizer_state",
                            "scheduler": "lr_scheduler",
                            "global_step": "completed_optimizer_steps",
                            "rank_runtime_state": "per_raw_rank_rng_and_reader",
                        },
                    },
                )
                identity[0] = committed.manifest_digest
        except Exception as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
        dist.broadcast_object_list(identity, src=0)
        dist.barrier()
    if error[0]:
        raise BackendContractError(f"Wan checkpoint commit failed: {error[0]}")
    if not identity[0]:
        raise BackendContractError("Wan checkpoint commit returned no identity")
    return str(identity[0])


__all__ = [
    "RestoredWanCheckpoint",
    "RestoredWanWeights",
    "checkpoint_contract",
    "load_anyflow_weights_only_checkpoint",
    "load_full_checkpoint",
    "load_live_and_ema_weights_checkpoint",
    "load_weights_only_checkpoint",
    "normalize_model_state",
    "normalize_optimizer_state",
    "optimizer_parameter_names",
    "save_full_checkpoint",
]
