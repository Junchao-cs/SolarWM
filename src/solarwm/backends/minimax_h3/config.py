"""Strict validation for the released H3 Stage0.5, AnyFlow and SGF profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from solarwm.data import resolve_index_path
from solarwm.errors import ConfigurationError

from .camera import h3_fused_prope_contract
from .codec import H3_PREENCODE_VERSION
from .geometry import validate_stage0p5_geometry


@dataclass(frozen=True)
class H3RunContract:
    """Resolved fields that identify the supported H3 configuration."""

    action: str
    stage: str
    pixel_frames: int
    encoded_latents: int
    sequence_parallel_size: int
    adapter_rank: int
    camera_translation_transform: str
    data_input_mode: str


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{key} must be a mapping")
    return value


def _required(mapping: Mapping[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise ConfigurationError(f"{path}.{key} is required")
    return mapping[key]


def _equal(mapping: Mapping[str, Any], key: str, expected: Any, path: str) -> None:
    observed = _required(mapping, key, path)
    if isinstance(expected, bool):
        matches = observed is expected
    elif isinstance(expected, int):
        matches = (
            isinstance(observed, int) and not isinstance(observed, bool) and observed == expected
        )
    elif isinstance(expected, float):
        matches = (
            isinstance(observed, (int, float))
            and not isinstance(observed, bool)
            and float(observed) == expected
        )
    else:
        matches = observed == expected
    if not matches:
        raise ConfigurationError(f"{path}.{key} must be {expected!r}, got {observed!r}")


def _positive_int(mapping: Mapping[str, Any], key: str, path: str) -> int:
    value = _required(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigurationError(f"{path}.{key} must be a positive integer")
    return value


def _number(mapping: Mapping[str, Any], key: str, path: str) -> float:
    value = _required(mapping, key, path)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"{path}.{key} must be numeric")
    return float(value)


def _close(mapping: Mapping[str, Any], key: str, expected: float, path: str) -> None:
    observed = _number(mapping, key, path)
    if abs(observed - expected) > 1e-12:
        raise ConfigurationError(f"{path}.{key} must be {expected}, got {observed}")


def _nonempty_path(mapping: Mapping[str, Any], key: str, path: str) -> str:
    value = _required(mapping, key, path)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{path}.{key} must be a non-empty path")
    return value


def _validate_model(
    model: Mapping[str, Any], *, action: str, input_mode: str, stage: str = "stage0p5"
) -> None:
    if "checkpoint_digest" in model:
        raise ConfigurationError(
            "model does not support removed content-digest fields: ['checkpoint_digest']"
        )
    family = str(_required(model, "family", "model")).strip().lower()
    if family not in {"minimax_h3", "minimax-h3"}:
        raise ConfigurationError(f"model.family must select MiniMax-H3, got {family!r}")
    _nonempty_path(model, "checkpoint_path", "model")
    if action == "preencode":
        codec_identity = str(_required(model, "codec_identity", "model")).strip()
        if not codec_identity:
            raise ConfigurationError("model.codec_identity must be non-empty")
    for key, expected in (
        ("architecture", "minimax-h3-33b"),
        ("torch_dtype", "bfloat16"),
        ("camera_attention_mode", "fused_prope"),
        ("camera_translation_transform", "logd4"),
        ("camera_intrinsics_mode", "wan_fixed"),
    ):
        _equal(model, key, expected, "model")
    for key, expected in (
        ("attention_head_dim", 128),
        ("camera_prope_head_dim_start", 96),
        ("camera_prope_head_dim_end", 128),
        ("latent_channels", 24),
        ("latent_height", 48),
        ("latent_width", 84),
        ("rows_per_latent", 1008),
        ("num_frames_per_block", 5),
        ("max_prior_clean_chunks", 5),
    ):
        _equal(model, key, expected, "model")
    if action == "preencode":
        return
    for key, expected in (
        ("training_mode", "lora"),
        ("transformer_subfolder", "transformer"),
        ("transformer_device_map", None),
        ("attention_backend", "flex" if stage in {"stage1", "stage2"} else "flash"),
        ("load_conditioners", False),
    ):
        _equal(model, key, expected, "model")
    adapter = _mapping(model, "adapter")
    for key, expected in (
        ("type", "lora"),
        ("target", "block_qkvo_ffn"),
        ("rank", 384),
        ("alpha", 384),
        ("dropout", 0.0),
        ("bias", "none"),
        ("dtype", "bfloat16"),
        ("expected_target_linear_modules", 312),
        ("expected_trainable_parameters", 2_075_394_048),
    ):
        _equal(adapter, key, expected, "model.adapter")


def _validate_data(data: Mapping[str, Any], *, action: str, stage: str = "stage0p5") -> str:
    input_mode = str(_required(data, "input_mode", "data")).strip().lower()
    allowed = {"raw"} if action == "preencode" else {"preencoded"}
    if input_mode not in allowed:
        raise ConfigurationError(
            f"MiniMax-H3 {action} data.input_mode must be one of {sorted(allowed)!r}; "
            f"got {input_mode!r}"
        )
    transport = _mapping(data, "transport")
    kind = str(_required(transport, "kind", "data.transport")).strip().lower()
    if kind not in {"local", "gcs"}:
        raise ConfigurationError("data.transport.kind must be local or gcs")
    root = _nonempty_path(transport, "root", "data.transport")
    if kind == "local":
        if not root.startswith("/"):
            raise ConfigurationError("local data.transport.root must be absolute")
        if data.get("index_root") is not None:
            index_root = _nonempty_path(data, "index_root", "data")
            if not index_root.startswith("/"):
                raise ConfigurationError("data.index_root must be absolute")
    else:
        if not root.startswith("gs://"):
            raise ConfigurationError("gcs data.transport.root must be a gs:// URI")
        cache_dir = _nonempty_path(transport, "cache_dir", "data.transport")
        if not cache_dir.startswith("/"):
            raise ConfigurationError("data.transport.cache_dir must be absolute")
        if _number(transport, "cache_max_gib", "data.transport") <= 0:
            raise ConfigurationError("data.transport.cache_max_gib must be positive")
        index_root = _nonempty_path(data, "index_root", "data")
        if not index_root.startswith("/"):
            raise ConfigurationError(
                "gcs transport requires absolute data.index_root for staged controls"
            )
    index_fields = ("index",) if action == "preencode" else ("train_index", "test_index")
    for field in index_fields:
        _nonempty_path(data, field, "data")
        try:
            resolve_index_path(data, field)
        except Exception as exc:
            raise ConfigurationError(f"cannot resolve data.{field}: {exc}") from exc
    for key, expected in (
        ("pixel_frames", 158),
        ("encoded_latents", 47),
        ("train_target_latents", 45 if stage == "stage1" else 47),
        ("height", 768),
        ("width", 1344),
        ("latent_channels", 24),
        ("latent_height", 48),
        ("latent_width", 84),
        ("frame_sampling", "contiguous"),
        ("source_fps_policy", "audit_only"),
        ("random_start", False),
        ("resolution_label", "default"),
    ):
        _equal(data, key, expected, "data")
    _close(data, "max_relative_translation", 20.0, "data")
    _close(data, "max_camera_absolute_value", 20.0, "data")
    validate_stage0p5_geometry(
        pixel_frames=int(data["pixel_frames"]),
        encoded_latents=int(data["encoded_latents"]),
        height=int(data["height"]),
        width=int(data["width"]),
        latent_channels=int(data["latent_channels"]),
        latent_height=int(data["latent_height"]),
        latent_width=int(data["latent_width"]),
    )
    if action == "train":
        workers = data.get("num_workers", 1)
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ConfigurationError("data.num_workers must be a positive integer")
        if data.get("prefetch_factor") is not None:
            prefetch = data["prefetch_factor"]
            if isinstance(prefetch, bool) or not isinstance(prefetch, int) or prefetch < 1:
                raise ConfigurationError("data.prefetch_factor must be a positive integer")
        shard_prefetch = data.get("gcs_prefetch_shards", 0)
        if (
            isinstance(shard_prefetch, bool)
            or not isinstance(shard_prefetch, int)
            or shard_prefetch < 0
        ):
            raise ConfigurationError("data.gcs_prefetch_shards must be a non-negative integer")
    if input_mode == "preencoded":
        _equal(data, "preencode_version", H3_PREENCODE_VERSION, "data")
        _equal(data, "dataset_name", "h3_preencoded_wds", "data")
        _nonempty_path(data, "silence_latents_path", "data")
        _nonempty_path(data, "encoder_contract_path", "data")
    else:
        _equal(data, "fixed_start_from_index", True, "data")
    return input_mode


def _validate_route(train: Mapping[str, Any]) -> None:
    from solarwm.config.routes import Route, supported_routes

    route = Route(
        "minimax_h3",
        str(train.get("stage", "")),
        str(train.get("causal_mode", "")),
        str(train.get("objective", "")),
        str(train.get("objective_variant", "")),
    )
    if route not in supported_routes():
        raise ConfigurationError(f"unsupported H3 training route {route.key}")
    if route.stage == "stage1":
        for key, expected in (
            ("anyflow_gate", 0.25),
            ("deltatime_type", "r"),
            ("finite_difference_epsilon", 5.0),
            ("diffusion_ratio", 0.5),
            ("consistency_ratio", 0.25),
            ("weight_type", "gaussian"),
        ):
            _equal(train, key, expected, "train")
    elif route.stage == "stage2":
        for key, expected in (
            ("critic_updates_per_student", 5),
            ("num_denoising_steps", 4),
            ("per_rank_exit_step", True),
            ("last_step_only", False),
            ("match_context", True),
            ("cache_mode", "exit"),
            ("tail_camera_policy", "repeat_last_latent"),
            ("audio_condition_policy", "fixed_noised_silence_per_rollout"),
            ("score_min_sigma", 0.02),
            ("score_max_sigma", 0.98),
        ):
            _equal(train, key, expected, "train")


def _validate_training(
    train: Mapping[str, Any],
    distributed: Mapping[str, Any],
) -> None:
    stage = str(train["stage"])
    sgf = stage == "stage2"
    for key, expected in (
        ("precision", "bfloat16"),
        ("micro_batch_size", 1),
        ("gradient_accumulation_steps", 1),
        ("video_timestep_shift", 12.0),
        ("audio_timestep_shift", 3.0),
        ("keyframe_noise_augmentation", 0.999),
        ("audio_loss_weight", 0.0),
    ):
        _equal(train, key, expected, "train")
    optimizer = _mapping(train, "optimizer")
    for key, expected in (
        ("name", "fp32_master_adamw"),
        ("learning_rate", {"stage0p5": 1e-4, "stage1": 3e-5, "stage2": 2e-6}[stage]),
        ("warmup_steps", {"stage0p5": 500, "stage1": 1000, "stage2": 0}[stage]),
    ):
        _equal(optimizer, key, expected, "train.optimizer")
    if stage != "stage0p5":
        for key, expected in (
            ("betas", [0.0, 0.999] if sgf else [0.9, 0.95]),
            ("epsilon", 1e-8),
            ("weight_decay", 0.0 if sgf else 0.01),
            ("gradient_clip", 10.0 if sgf else 1.0),
            ("min_lr_ratio", 1.0 if sgf else 0.1),
        ):
            _equal(optimizer, key, expected, "train.optimizer")
    if sgf:
        critic = _mapping(train, "critic_optimizer")
        for key, value in optimizer.items():
            _equal(critic, key, 4e-7 if key == "learning_rate" else value, "train.critic_optimizer")
    fsdp = _mapping(train, "fsdp")
    for key, expected in (
        ("sharding_strategy", "HYBRID_SHARD" if sgf else "FULL_SHARD"),
        ("activation_checkpointing", True),
        ("preserve_checkpoint_dtype", True),
        ("param_dtype", None),
        ("reduce_dtype", "float32"),
        ("buffer_dtype", None),
    ):
        _equal(fsdp, key, expected, "train.fsdp")

    world_size = _positive_int(distributed, "world_size", "distributed")
    sequence_parallel = _positive_int(distributed, "sequence_parallel_size", "distributed")
    _equal(distributed, "rank_partition", "node_shard", "distributed")
    _equal(distributed, "context_parallel_size", 1, "distributed")
    _equal(distributed, "sp_peers_share_sample", True, "distributed")
    _equal(distributed, "sp_peers_share_rng", True, "distributed")
    if sequence_parallel != (4 if sgf else 2):
        raise ConfigurationError(f"H3 {stage} requires sequence_parallel_size={4 if sgf else 2}")
    if sgf:
        _equal(fsdp, "frozen_base_shard_size", 8, "train.fsdp")
    if world_size % sequence_parallel:
        raise ConfigurationError("distributed.world_size must be divisible by SP size")
    # Each stage fixes SP, micro batch and accumulation. Changing the world size
    # also changes the global batch and optimization trajectory.
    computed_global_batch = (
        world_size
        // sequence_parallel
        * int(train["micro_batch_size"])
        * int(train["gradient_accumulation_steps"])
    )
    if computed_global_batch != int(train["global_batch_size"]):
        raise ConfigurationError(
            "global batch mismatch: (world_size / SP) * micro_batch * grad_accum "
            f"is {computed_global_batch}, configured {train['global_batch_size']}"
        )


def _validate_validation(
    validation: Mapping[str, Any], *, action: str, stage: str = "stage0p5"
) -> None:
    del action
    _positive_int(validation, "sample_count", "validation")
    for name in ("selection_seed", "noise_seed"):
        value = _required(validation, name, "validation")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ConfigurationError(f"validation.{name} must be a non-negative integer")
    for key, expected in (
        ("pixel_frames", 158),
        ("latent_frames", 47),
        ("fps", 24),
        ("num_inference_steps", 30 if stage == "stage0p5" else 4),
        ("passes", ["live", "ema"]),
    ):
        _equal(validation, key, expected, "validation")
    smoke_step = _required(validation, "smoke_step", "validation")
    if isinstance(smoke_step, bool) or not isinstance(smoke_step, int) or smoke_step < 0:
        raise ConfigurationError("validation.smoke_step must be a non-negative integer")
    _close(validation, "max_relative_translation", 20.0, "validation")
    _close(validation, "max_camera_absolute_value", 20.0, "validation")


def _validate_checkpoint(checkpoint: Mapping[str, Any], *, stage: str = "stage0p5") -> None:
    _equal(checkpoint, "save_optimizer", True, "checkpoint")
    ema = _mapping(checkpoint, "ema")
    for key, expected in (
        ("enabled", True),
        ("dtype", "float32"),
        ("sharded", True),
        ("decay", {"stage0p5": 0.9999, "stage1": 0.999, "stage2": 0.99}[stage]),
        ("start_step", 39 if stage == "stage2" else 0),
        ("update_every_steps", 1),
    ):
        _equal(ema, key, expected, "checkpoint.ema")


def _validate_inference_distributed(
    distributed: Mapping[str, Any], *, stage: str = "stage0p5", full_length: bool = False
) -> None:
    world_size = _positive_int(distributed, "world_size", "distributed")
    sequence_parallel = _positive_int(
        distributed,
        "sequence_parallel_size",
        "distributed",
    )
    expected_sp = 8 if full_length else (4 if stage == "stage2" else 2)
    if full_length and world_size != 8:
        raise ConfigurationError("Source-length H3 inference requires one eight-GPU SP8 worker")
    if sequence_parallel != expected_sp or world_size % sequence_parallel:
        raise ConfigurationError(f"H3 inference requires a world divisible by SP{expected_sp}")
    for key, expected in (
        ("context_parallel_size", 1),
        ("sp_peers_share_sample", True),
        ("sp_peers_share_rng", True),
    ):
        _equal(distributed, key, expected, "distributed")


def validate_h3_config(config: Mapping[str, Any]) -> H3RunContract:
    """Validate one config against the supported H3 profile."""

    if not isinstance(config, Mapping):
        raise ConfigurationError("config must be a mapping")
    action = str(config.get("action", "")).strip().lower()
    if action not in {"train", "infer", "preencode"}:
        raise ConfigurationError("action must be train, infer, or preencode")
    inference = config.get("inference", {})
    if not isinstance(inference, Mapping):
        raise ConfigurationError("inference must be a mapping")
    length_policy = inference.get("length_policy", "fixed")
    if length_policy not in {"fixed", "source"}:
        raise ConfigurationError("inference.length_policy must be fixed or source")
    full_length = length_policy == "source"
    if full_length:
        if action != "infer" or config.get("train", {}).get("stage") != "stage2":
            raise ConfigurationError("Source-length inference is only supported for H3 Stage2 SGF")
        for key in ("plan", "work_dir", "dataset_root"):
            _nonempty_path(inference, key, "inference")
        if inference.get("fps_policy") != "source":
            raise ConfigurationError("Source-length inference requires fps_policy=source")
        if inference.get("phase", "all") not in {"all", "prepare", "generate"}:
            raise ConfigurationError("inference.phase must be all, prepare or generate")
        source = config.get("checkpoint", {}).get("weight_source")
        if source not in {"live", "ema"}:
            raise ConfigurationError(
                "Source-length inference requires explicit live or ema weights"
            )
        _positive_int(inference, "expected_step", "inference")
    model = _mapping(config, "model")
    data = _mapping(config, "data")
    metadata = config.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ConfigurationError("metadata must be a mapping")
    stage = "preencode" if action == "preencode" else str(_mapping(config, "train")["stage"])
    if stage not in {"preencode", "stage0p5", "stage1", "stage2"}:
        raise ConfigurationError(f"unsupported H3 stage {stage!r}")
    declared_input_mode = str(data.get("input_mode", "")).strip().lower()
    _validate_model(model, action=action, input_mode=declared_input_mode, stage=stage)
    input_mode = _validate_data(data, action=action, stage=stage)

    if action != "preencode":
        train = _mapping(config, "train")
        _validate_route(train)
        if stage in {"stage1", "stage2"}:
            validation = _mapping(config, "validation")
            if validation.get("prepared_plan") is not None:
                _nonempty_path(validation, "prepared_plan", "validation")
            elif stage == "stage1":
                _nonempty_path(data, "raw_test_index", "data")
            for key, expected in (
                ("rollout_latents", 50),
                ("decode_latents", 47),
                ("camera_frames", 170 if stage == "stage1" else 158),
            ):
                _equal(validation, key, expected, "validation")
            if stage == "stage2":
                _equal(model, "student_rope_mode", "sliding_local", "model")
                _equal(model, "score_rope_mode", "native_absolute", "model")
            if action == "train":
                initialization = _mapping(_mapping(config, "checkpoint"), "initialization")
                for role in ("student", "teacher", "critic") if stage == "stage2" else ("student",):
                    item = _mapping(initialization, role)
                    _nonempty_path(item, "path", f"checkpoint.initialization.{role}")
                    _equal(item, "weight_source", "ema", f"checkpoint.initialization.{role}")
                    _equal(
                        item,
                        "stage",
                        "stage1" if stage == "stage2" and role == "student" else "stage0p5",
                        f"checkpoint.initialization.{role}",
                    )

        _validate_validation(_mapping(config, "validation"), action=action, stage=stage)
        if action == "train":
            _validate_training(train, _mapping(config, "distributed"))
            _validate_checkpoint(_mapping(config, "checkpoint"), stage=stage)
        else:
            _validate_inference_distributed(
                _mapping(config, "distributed"), stage=stage, full_length=full_length
            )
            _nonempty_path(_mapping(config, "checkpoint"), "resume_from", "checkpoint")
    else:
        preencode = _mapping(config, "preencode")
        _equal(preencode, "codec_protocol", "solarwm.minimax_h3.codec.v1", "preencode")
        _nonempty_path(preencode, "output_root", "preencode")

    runtime = _mapping(config, "runtime")
    _nonempty_path(runtime, "output_dir", "runtime")
    contract = h3_fused_prope_contract()
    if contract["camera_prope_head_slice"] != [96, 128]:
        raise ConfigurationError("internal H3 camera suffix contract is inconsistent")
    return H3RunContract(
        action=action,
        stage=stage,
        pixel_frames=158,
        encoded_latents=47,
        sequence_parallel_size=(
            0
            if action == "preencode"
            else int(_mapping(config, "distributed")["sequence_parallel_size"])
        ),
        adapter_rank=0 if action == "preencode" else 384,
        camera_translation_transform="logd4",
        data_input_mode=input_mode,
    )


__all__ = ["H3RunContract", "validate_h3_config"]
