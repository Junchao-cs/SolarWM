"""Strict validation for LTX-2.5 Stage0.5 configs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from solar.data.index import resolve_index_path
from solar.errors import ConfigurationError

from .artifact import PREENCODE_VERSION, READER_CONTRACT
from .checkpoint import EMAContract, LoRACheckpointContract
from .codec import ONLINE_BEHAVIOR_STATUS, ONLINE_CODEC_PROTOCOL
from .distributed import DistributedContract
from .geometry import STABLE_GEOMETRY
from .inference import GuidanceSpec, InferenceSpec, build_inference_plan


@dataclass(frozen=True)
class LTX25RunContract:
    action: str
    input_mode: str
    stage: str
    camera_translation_transform: str
    sequence_parallel_size: int
    global_batch_size: int
    online_behavior_status: str | None = None


def _mapping(config: Mapping[str, Any], key: str, *, required: bool = True) -> Mapping[str, Any]:
    value = config.get(key)
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"{path}.{key} is required")
    return mapping[key]


def _equal(mapping: Mapping[str, Any], key: str, expected: Any, path: str) -> None:
    value = _required(mapping, key, path)
    if value != expected:
        raise ConfigurationError(f"{path}.{key} must be {expected!r}, got {value!r}")


def _path(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = _required(mapping, key, path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path}.{key} must be a non-empty path")
    return value


def _reject_removed(mapping: Mapping[str, Any], keys: tuple[str, ...], path: str) -> None:
    present = sorted(key for key in keys if key in mapping)
    if present:
        raise ConfigurationError(
            f"{path} does not support removed content-digest fields: {present}"
        )


def _integer(mapping: Mapping[str, Any], key: str, path: str) -> int:
    value = _required(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigurationError(f"{path}.{key} must be an integer")
    return value


def _number(mapping: Mapping[str, Any], key: str, path: str) -> float:
    value = _required(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path}.{key} must be numeric")
    return float(value)


def _validate_family(model: Mapping[str, Any]) -> None:
    family = str(_required(model, "family", "model")).strip().lower()
    if family not in {"ltx25_video", "ltx-2.5"}:
        raise ConfigurationError(f"model.family must select LTX-2.5, got {family!r}")


def _validate_adapter(model: Mapping[str, Any]) -> None:
    adapter = _mapping(model, "adapter")
    expected = {
        "type": "lora",
        "target": "block_qkvo_ffn",
        "rank": 384,
        "alpha": 384,
        "dropout": 0.0,
        "bias": "none",
        "dtype": "bfloat16",
        "init_lora_weights": True,
    }
    for key, value in expected.items():
        _equal(adapter, key, value, "model.adapter")
    LoRACheckpointContract(
        rank=int(adapter["rank"]),
        alpha=int(adapter["alpha"]),
        dropout=float(adapter["dropout"]),
    )


def _validate_codec(model: Mapping[str, Any], *, require_gemma: bool) -> None:
    codec = _mapping(model, "codec")
    _reject_removed(codec, ("video_vae_digest", "gemma4_digest"), "model.codec")
    _path(codec, "video_vae_path", "model.codec")
    if require_gemma:
        _path(codec, "gemma4_path", "model.codec")


def _validate_model(model: Mapping[str, Any], *, action: str) -> str:
    _validate_family(model)
    _reject_removed(model, ("checkpoint_digest",), "model")
    if action == "preencode":
        _path(model, "checkpoint_path", "model")
        _validate_codec(model, require_gemma=True)
        return "linear"

    _path(model, "checkpoint_path", "model")
    expected = {
        "architecture": "ltx-2.5-22b-dev-video-only",
        "training_mode": "lora",
        "torch_dtype": "bfloat16",
        "load_conditioners": False,
        "audio_enabled": False,
        "camera_attention_mode": "fused_prope",
        "camera_intrinsics_mode": "wan_fixed",
        "model_fps": 24.0,
        "latent_channels": 128,
        "latent_height": 16,
        "latent_width": 24,
        "caption_cache_stage": "gemma4_feature_extractor_preconnector",
        "attention_backend": "cudnn",
    }
    for key, value in expected.items():
        _equal(model, key, value, "model")
    transform = str(_required(model, "camera_translation_transform", "model")).lower()
    if transform not in {"linear", "logd4"}:
        raise ConfigurationError("model.camera_translation_transform must be linear or logd4")
    _validate_adapter(model)
    _validate_codec(model, require_gemma=False)
    if action == "infer":
        weights = str(_required(model, "adapter_weights", "model")).lower()
        if weights not in {"live", "ema"}:
            raise ConfigurationError("model.adapter_weights must be live or ema")
        checkpoint_format = (
            str(model.get("adapter_checkpoint_format", "transaction_v1")).strip().lower()
        )
        if checkpoint_format not in {
            "transaction_v1",
            "inference_transaction_v1",
        }:
            raise ConfigurationError(
                "model.adapter_checkpoint_format must be transaction_v1, "
                "or inference_transaction_v1"
            )
    return transform


def _validate_geometry(data: Mapping[str, Any]) -> None:
    expected = {
        "pixel_frames": STABLE_GEOMETRY.pixel_frames,
        "encoded_latents": STABLE_GEOMETRY.latent_frames,
        "height": STABLE_GEOMETRY.height,
        "width": STABLE_GEOMETRY.width,
        "latent_channels": STABLE_GEOMETRY.latent_channels,
        "latent_height": STABLE_GEOMETRY.latent_height,
        "latent_width": STABLE_GEOMETRY.latent_width,
        "frame_sampling": "contiguous",
        "source_fps_policy": "provenance_only",
        "random_start": False,
    }
    for key, value in expected.items():
        _equal(data, key, value, "data")


def _validate_transport(data: Mapping[str, Any]) -> None:
    for misplaced_field in ("root", "cache_dir", "cache_max_gib"):
        if misplaced_field in data:
            raise ConfigurationError(
                f"data.{misplaced_field} is unsupported; use data.transport.{misplaced_field}"
            )
    transport = _mapping(data, "transport")
    kind = str(_required(transport, "kind", "data.transport")).strip().lower()
    if kind not in {"local", "gcs"}:
        raise ConfigurationError("data.transport.kind must be local or gcs")
    root = _path(transport, "root", "data.transport")
    index_root = data.get("index_root")
    if kind == "local":
        if not root.startswith("/"):
            raise ConfigurationError("local data.transport.root must be absolute")
        if any(key in transport for key in ("cache_dir", "cache_max_gib")):
            raise ConfigurationError("local data.transport cannot configure a GCS cache")
        if index_root is not None and not _path(data, "index_root", "data").startswith("/"):
            raise ConfigurationError("data.index_root must be absolute")
    else:
        if not root.startswith("gs://"):
            raise ConfigurationError("gcs data.transport.root must be a gs:// URI")
        cache_dir = _path(transport, "cache_dir", "data.transport")
        if not cache_dir.startswith("/"):
            raise ConfigurationError("data.transport.cache_dir must be absolute")
        if _number(transport, "cache_max_gib", "data.transport") <= 0:
            raise ConfigurationError("data.transport.cache_max_gib must be positive")
        if index_root is None or not _path(data, "index_root", "data").startswith("/"):
            raise ConfigurationError(
                "gcs transport requires absolute data.index_root for staged controls"
            )


def _validate_data(data: Mapping[str, Any], *, action: str) -> tuple[str, str | None]:
    input_mode = str(_required(data, "input_mode", "data")).strip().lower()
    allowed = {
        "train": {"preencoded", "raw_online"},
        "infer": {"preencoded"},
        "preencode": {"raw"},
    }[action]
    if input_mode not in allowed:
        raise ConfigurationError(
            f"LTX {action} data.input_mode must be one of {sorted(allowed)}, got {input_mode!r}"
        )
    _validate_transport(data)
    shard_prefetch = data.get("gcs_prefetch_shards", 0)
    if (
        isinstance(shard_prefetch, bool)
        or not isinstance(shard_prefetch, int)
        or shard_prefetch < 0
    ):
        raise ConfigurationError("data.gcs_prefetch_shards must be a non-negative integer")
    _path(data, "index", "data")
    try:
        resolve_index_path(data, "index")
    except Exception as exc:
        raise ConfigurationError(f"cannot resolve data.index: {exc}") from exc
    if action in {"train", "infer"}:
        _path(data, "generation", "data")
    _validate_geometry(data)
    if action in {"train", "infer"}:
        _equal(data, "max_rel_translation", 20.0, "data")
        _equal(data, "max_camera_abs", 20.0, "data")
    status = None
    if input_mode == "preencoded":
        _equal(data, "reader_contract", READER_CONTRACT, "data")
        _equal(data, "artifact_version", PREENCODE_VERSION, "data")
        _path(data, "completion_marker", "data")
    elif input_mode == "raw_online":
        _equal(data, "online_codec_protocol", ONLINE_CODEC_PROTOCOL, "data")
        _equal(data, "behavior_status", ONLINE_BEHAVIOR_STATUS, "data")
        status = ONLINE_BEHAVIOR_STATUS
    return input_mode, status


def _validate_route(train: Mapping[str, Any]) -> None:
    for key, value in {
        "stage": "stage0p5",
        "causal_mode": "bidirectional",
        "objective": "native_rectified_flow",
    }.items():
        _equal(train, key, value, "train")


def _validate_inference(value: Mapping[str, Any], path: str) -> None:
    _path(value, "negative_caption_cache", path)
    guidance = _mapping(value, "guidance")
    spec = InferenceSpec(
        num_inference_steps=_integer(value, "num_inference_steps", path),
        seed=_integer(value, "seed", path),
        fps=_integer(value, "fps", path),
        decoder_mode=str(_required(value, "decoder_mode", path)),
    )
    guide = GuidanceSpec(
        cfg_scale=_number(guidance, "cfg_scale", f"{path}.guidance"),
        stg_scale=_number(guidance, "stg_scale", f"{path}.guidance"),
        rescale_scale=_number(guidance, "rescale_scale", f"{path}.guidance"),
        stg_blocks=tuple(_required(guidance, "stg_blocks", f"{path}.guidance")),
    )
    build_inference_plan(spec, guide)


def _validate_train(config: Mapping[str, Any], train: Mapping[str, Any]) -> tuple[int, int]:
    _validate_route(train)
    for key, value in {
        "precision": "bfloat16",
        "micro_batch_size": 1,
        "first_frame_condition_dropout": 0.0,
    }.items():
        _equal(train, key, value, "train")
    gradient_accumulation = _integer(train, "gradient_accumulation_steps", "train")
    global_batch = _integer(train, "global_batch_size", "train")
    if gradient_accumulation < 1 or global_batch < 1:
        raise ConfigurationError("gradient accumulation and global batch must be positive")
    sampler = _mapping(train, "timestep_sampling")
    for key, value in {
        "mode": "shifted_logit_normal",
        "std": 1.0,
        "epsilon": 1.0e-3,
        "uniform_probability": 0.1,
    }.items():
        _equal(sampler, key, value, "train.timestep_sampling")
    optimizer = _mapping(train, "optimizer")
    for key, value in {
        "name": "fp32_master_adamw",
        "learning_rate": 1.0e-4,
        "betas": [0.9, 0.95],
        "epsilon": 1.0e-8,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "warmup_steps": 500,
    }.items():
        _equal(optimizer, key, value, "train.optimizer")
    fsdp = _mapping(train, "fsdp")
    for key, value in {
        "activation_checkpointing": True,
        "activation_checkpointed_blocks": 48,
        "preserve_checkpoint_dtype": True,
        "param_dtype": None,
        "reduce_dtype": "float32",
        "buffer_dtype": None,
        "ignore_fp32_scale_tables": True,
    }.items():
        _equal(fsdp, key, value, "train.fsdp")
    sharding_strategy = str(_required(fsdp, "sharding_strategy", "train.fsdp"))
    if sharding_strategy not in {"FULL_SHARD", "HYBRID_SHARD"}:
        raise ConfigurationError("train.fsdp.sharding_strategy is unsupported")
    distributed = _mapping(config, "distributed")
    world_size = _integer(distributed, "world_size", "distributed")
    local_world_size = _integer(distributed, "local_world_size", "distributed")
    sp_size = _integer(distributed, "sequence_parallel_size", "distributed")
    _equal(distributed, "rank_partition", "node_shard", "distributed")
    data = _mapping(config, "data")
    _path(data, "test_index", "data")
    try:
        resolve_index_path(data, "test_index")
    except Exception as exc:
        raise ConfigurationError(f"cannot resolve data.test_index: {exc}") from exc
    _equal(data, "partition_mode", "node_shard", "data")
    DistributedContract(
        world_size=world_size,
        local_world_size=local_world_size,
        sp_size=sp_size,
        micro_batch_size=int(train["micro_batch_size"]),
        gradient_accumulation_steps=gradient_accumulation,
        global_batch_size=global_batch,
        sharding_strategy=sharding_strategy,
        activation_checkpointed_blocks=int(fsdp["activation_checkpointed_blocks"]),
    )
    checkpoint = _mapping(config, "checkpoint")
    resume_from = checkpoint.get("resume_from")
    if resume_from is not None and (not isinstance(resume_from, str) or not resume_from.strip()):
        raise ConfigurationError("checkpoint.resume_from must be a non-empty path")
    ema = _mapping(checkpoint, "ema")
    for key, value in {
        "enabled": True,
        "start_step": 0,
        "update_every_steps": 1,
        "decay": 0.999,
        "device": "cuda",
        "dtype": "float32",
        "sharded": True,
        "trainable_only": True,
    }.items():
        _equal(ema, key, value, "checkpoint.ema")
    EMAContract()
    _equal(checkpoint, "save_optimizer", True, "checkpoint")
    validation = _mapping(config, "validation")
    if _integer(validation, "sample_count", "validation") < 1:
        raise ConfigurationError("validation.sample_count must be positive")
    if _integer(validation, "selection_seed", "validation") < 0:
        raise ConfigurationError("validation.selection_seed must be non-negative")
    _validate_inference(_mapping(validation, "inference"), "validation.inference")
    return sp_size, global_batch


def _validate_preencode(config: Mapping[str, Any]) -> None:
    preencode = _mapping(config, "preencode")
    _equal(preencode, "codec_protocol", ONLINE_CODEC_PROTOCOL, "preencode")
    _equal(preencode, "behavior_status", ONLINE_BEHAVIOR_STATUS, "preencode")
    _equal(preencode, "vae_encode_mode", "direct", "preencode")
    _equal(preencode, "automatic_oom_fallback", False, "preencode")
    _path(preencode, "output_root", "preencode")
    if _integer(preencode, "samples_per_shard", "preencode") < 1:
        raise ConfigurationError("preencode.samples_per_shard must be positive")
    index = _path(preencode, "index_relative_path", "preencode")
    if not index.endswith(".jsonl") or index.startswith("/") or ".." in index.split("/"):
        raise ConfigurationError(
            "preencode.index_relative_path must be a safe relative .jsonl path"
        )
    if index.count("{rank}") != 1:
        raise ConfigurationError(
            "preencode.index_relative_path must contain exactly one {rank} placeholder"
        )


def _validate_runtime(runtime: Mapping[str, Any]) -> None:
    _path(runtime, "output_dir", "runtime")
    entrypoint = runtime.get("provider_entrypoint")
    if entrypoint is not None and (
        not isinstance(entrypoint, str)
        or entrypoint.count(":") != 1
        or not all(part.strip() for part in entrypoint.split(":"))
    ):
        raise ConfigurationError("runtime.provider_entrypoint must use 'module.path:factory'")


def validate_ltx25_config(config: Mapping[str, Any]) -> LTX25RunContract:
    """Validate a public LTX config without importing Torch or LTX-Core."""

    if not isinstance(config, Mapping):
        raise ConfigurationError("config must be a mapping")
    action = str(config.get("action", "")).strip().lower()
    if action not in {"train", "infer", "preencode"}:
        raise ConfigurationError("action must be train, infer, or preencode")
    model = _mapping(config, "model")
    transform = _validate_model(model, action=action)
    input_mode, online_status = _validate_data(_mapping(config, "data"), action=action)
    if input_mode == "raw_online":
        _validate_codec(model, require_gemma=True)
    runtime = _mapping(config, "runtime")
    _validate_runtime(runtime)

    sp_size = 0
    global_batch = 0
    if action == "train":
        sp_size, global_batch = _validate_train(config, _mapping(config, "train"))
    elif action == "infer":
        _validate_route(_mapping(config, "train"))
        inference = _mapping(config, "inference")
        _validate_inference(inference, "inference")
        if _integer(inference, "sample_count", "inference") < 1:
            raise ConfigurationError("inference.sample_count must be positive")
        if _integer(inference, "selection_seed", "inference") < 0:
            raise ConfigurationError("inference.selection_seed must be non-negative")
        distributed = _mapping(config, "distributed")
        sp_size = _integer(distributed, "sequence_parallel_size", "distributed")
        if sp_size not in {1, 2}:
            raise ConfigurationError("LTX inference sequence_parallel_size must be 1 or 2")
        _path(model, "adapter_checkpoint_path", "model")
    else:
        _validate_preencode(config)
        online_status = ONLINE_BEHAVIOR_STATUS

    return LTX25RunContract(
        action=action,
        input_mode=input_mode,
        stage="preencode" if action == "preencode" else "stage0p5",
        camera_translation_transform=transform,
        sequence_parallel_size=sp_size,
        global_batch_size=global_batch,
        online_behavior_status=online_status,
    )


__all__ = ["LTX25RunContract", "validate_ltx25_config"]
