"""Strict, allocation-free Wan2.2 configuration contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from solarwm.config.routes import Route, validate_route
from solarwm.errors import BackendContractError, ConfigurationError

WAN_TI2V_5B = "wan22_ti2v_5b"
WAN_I2V_A14B = "wan22_i2v_a14b"
WAN_FAMILIES = frozenset({WAN_TI2V_5B, WAN_I2V_A14B})
WAN_TI2V_5B_720P_153F_SCHEMA = "solarwm.wan22_ti2v_5b.720p.153f.v1"


@dataclass(frozen=True)
class FamilyProfile:
    family: str
    model_name: str
    timestep_shift: float
    latent_channels: int
    height: int
    width: int
    latent_height: int
    latent_width: int
    frame_sequence_length: int
    sequence_parallel_size: int
    fsdp_strategy: str


PROFILES = {
    WAN_TI2V_5B: FamilyProfile(
        family=WAN_TI2V_5B,
        model_name="Wan2.2-TI2V-5B-Camera",
        timestep_shift=5.0,
        latent_channels=48,
        height=480,
        width=864,
        latent_height=30,
        latent_width=54,
        frame_sequence_length=405,
        sequence_parallel_size=1,
        fsdp_strategy="HYBRID_SHARD",
    ),
    WAN_I2V_A14B: FamilyProfile(
        family=WAN_I2V_A14B,
        model_name="Wan2.2-I2V-A14B-High-Camera",
        timestep_shift=3.0,
        latent_channels=16,
        height=480,
        width=832,
        latent_height=60,
        latent_width=104,
        frame_sequence_length=1560,
        sequence_parallel_size=2,
        fsdp_strategy="FULL_SHARD",
    ),
}

TI2V_720P_153F_PROFILE = FamilyProfile(
    family=WAN_TI2V_5B,
    model_name="Wan2.2-TI2V-5B-Camera",
    timestep_shift=5.0,
    latent_channels=48,
    height=704,
    width=1280,
    latent_height=44,
    latent_width=80,
    frame_sequence_length=880,
    sequence_parallel_size=1,
    fsdp_strategy="HYBRID_SHARD",
)


def _profile_for_config(config: Mapping[str, Any], family: str) -> FamilyProfile:
    data = _mapping(config, "data")
    if (
        family == WAN_TI2V_5B
        and str(data.get("encoding", "")).strip().lower() == "preencoded"
        and str(data.get("preencode_schema", "")) == WAN_TI2V_5B_720P_153F_SCHEMA
    ):
        return TI2V_720P_153F_PROFILE
    return PROFILES[family]


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key, {})
    if not isinstance(value, Mapping):
        raise BackendContractError(f"{key} must be a mapping")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BackendContractError(message)


def _finite_float(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise BackendContractError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise BackendContractError(f"{field} must be finite")
    return result


def _int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise BackendContractError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise BackendContractError(f"{field} must be an integer") from exc
    if str(value).strip() not in {str(result), f"{result}.0"} and not isinstance(value, int):
        raise BackendContractError(f"{field} must be an integer")
    return result


def _relative_path(value: Any, field: str) -> str:
    text = str(value or "")
    path = PurePosixPath(text)
    if (
        not text
        or text.startswith("/")
        or "://" in text
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BackendContractError(f"{field} must be a portable relative POSIX path")
    return path.as_posix()


def pixel_to_latent_frames(pixel_frames: int) -> int:
    """Map the Wan temporal codec geometry, failing on noncanonical lengths."""

    frames = _int(pixel_frames, "data.pixel_frames")
    if frames < 1 or (frames - 1) % 4:
        raise BackendContractError(
            "Wan pixel frames must satisfy pixel_frames = 1 + 4*(latent_frames-1)"
        )
    return 1 + (frames - 1) // 4


def _validate_transport(data: Mapping[str, Any]) -> None:
    transport = _mapping(data, "transport")
    kind = str(transport.get("kind", "")).strip().lower()
    _require(kind in {"local", "gcs"}, "data.transport.kind must be local or gcs")
    root = str(transport.get("root", "")).strip()
    if kind == "local":
        _require(root.startswith("/"), "local data.transport.root must be absolute")
        if data.get("index_root") is not None:
            _require(
                str(data.get("index_root", "")).startswith("/"),
                "data.index_root must be absolute when provided",
            )
    else:
        _require(root.startswith("gs://"), "gcs data.transport.root must be a gs:// URI")
        _require(
            str(transport.get("cache_dir", "")).startswith("/"),
            "gcs data.transport.cache_dir must be absolute",
        )
        _require(
            _int(transport.get("cache_max_gib", 0), "data.transport.cache_max_gib") > 0,
            "data.transport.cache_max_gib must be positive",
        )
        _require(
            str(data.get("index_root", "")).startswith("/"),
            "gcs transport requires absolute data.index_root for staged controls",
        )


def _validate_runtime_caches(config: Mapping[str, Any]) -> None:
    runtime = _mapping(config, "runtime")
    transport = _mapping(_mapping(config, "data"), "transport")
    is_gcs = str(transport.get("kind", "")).strip().lower() == "gcs"
    data_cache_dir = runtime.get("data_cache_dir")
    data_cache_max_gib = runtime.get("data_cache_max_gib")
    validation_cache_dir = runtime.get("validation_cache_dir")
    validation_cache_max_gib = runtime.get("validation_cache_max_gib")
    _require(
        is_gcs
        or all(
            value is None
            for value in (
                data_cache_dir,
                data_cache_max_gib,
                validation_cache_dir,
                validation_cache_max_gib,
            )
        ),
        "runtime cache overrides are valid only for gcs data transport",
    )
    if data_cache_dir is not None:
        _require(
            str(data_cache_dir).startswith("/"),
            "runtime.data_cache_dir must be absolute",
        )
    if data_cache_max_gib is not None:
        _require(
            _int(data_cache_max_gib, "runtime.data_cache_max_gib") > 0,
            "runtime.data_cache_max_gib must be positive",
        )
    _require(
        (validation_cache_dir is None) == (validation_cache_max_gib is None),
        "runtime validation cache directory and size must be configured together",
    )
    if validation_cache_dir is not None:
        _require(
            str(validation_cache_dir).startswith("/"),
            "runtime.validation_cache_dir must be absolute",
        )
        _require(
            _int(validation_cache_max_gib, "runtime.validation_cache_max_gib") > 0,
            "runtime.validation_cache_max_gib must be positive",
        )


def _validate_data(config: Mapping[str, Any], route: Route, profile: FamilyProfile) -> None:
    data = _mapping(config, "data")
    encoding = str(data.get("encoding", "")).strip().lower()
    _require(encoding in {"online", "preencoded"}, "data.encoding must be online or preencoded")
    pixel_frames = _int(data.get("pixel_frames", 0), "data.pixel_frames")
    latent_frames = pixel_to_latent_frames(pixel_frames)
    _require(
        _int(data.get("latent_frames", 0), "data.latent_frames") == latent_frames,
        f"data.latent_frames must be {latent_frames} for {pixel_frames} pixel frames",
    )
    _require(
        _int(data.get("height", 0), "data.height") == profile.height,
        f"data.height must be {profile.height}",
    )
    _require(
        _int(data.get("width", 0), "data.width") == profile.width,
        f"data.width must be {profile.width}",
    )
    _require(
        _finite_float(data.get("max_rel_translation"), "data.max_rel_translation") == 20.0,
        "ordinary Wan recipes require data.max_rel_translation=20.0",
    )
    _require(
        _finite_float(data.get("max_camera_abs"), "data.max_camera_abs") == 20.0,
        "ordinary Wan recipes require data.max_camera_abs=20.0",
    )
    _require(bool(data.get("streaming_index")) is True, "data.streaming_index must be true")
    inference = _mapping(config, "inference")
    camera_length = (
        str(config.get("action", "")).strip().lower() == "infer"
        and str(inference.get("length", "fixed")).strip().lower() == "camera"
    )
    expected_random_start = pixel_frames == 81 and encoding == "online" and not camera_length
    _require(
        bool(data.get("random_start")) is expected_random_start,
        "online 81f uses random starts except for camera-length inference; "
        "fixed preencoded and 153f routes do not",
    )
    _relative_path(data.get("train_index"), "data.train_index")
    _relative_path(data.get("test_index"), "data.test_index")
    _validate_transport(data)

    from .codec import validate_tensor_data_config

    validate_tensor_data_config(data, family=profile.family)
    _require(
        str(data.get("partition_mode", "")) == "node_shard",
        "data.partition_mode must be node_shard",
    )
    _require(
        _int(data.get("num_workers", 0), "data.num_workers") > 0,
        "data.num_workers must be positive",
    )
    _require(
        _int(data.get("gcs_prefetch_shards", 0), "data.gcs_prefetch_shards") >= 0,
        "data.gcs_prefetch_shards must be non-negative",
    )

    if encoding == "online":
        _require(
            str(data.get("camera_array_key", "")) == "c2w",
            "raw Wan routes require data.camera_array_key=c2w",
        )
    if encoding == "preencoded":
        from .codec import validate_preencoded_data_config

        validate_preencoded_data_config(data, family=profile.family)
    if route.stage in {"stage1", "stage2"}:
        _require(pixel_frames == 81, f"{route.stage} supports 81f only")


def _validate_distributed(config: Mapping[str, Any], profile: FamilyProfile) -> None:
    distributed = _mapping(config, "distributed")
    raw_world = _int(distributed.get("world_size", 0), "distributed.world_size")
    sp_size = _int(
        distributed.get("sequence_parallel_size", 0), "distributed.sequence_parallel_size"
    )
    _require(raw_world > 0, "distributed.world_size must be positive")
    _require(
        sp_size == profile.sequence_parallel_size,
        f"{profile.family} requires sequence_parallel_size={profile.sequence_parallel_size}",
    )
    _require(
        raw_world % sp_size == 0,
        "distributed.world_size must divide evenly by sequence_parallel_size",
    )
    _require(
        str(distributed.get("sample_sharding", "")) == "logical_dp",
        "distributed.sample_sharding must be logical_dp",
    )
    _require(
        str(distributed.get("rng_scope", "")) == "logical_dp",
        "distributed.rng_scope must be logical_dp so SP peers share randomness",
    )

    train = _mapping(config, "train")
    micro = _int(train.get("micro_batch_size", 0), "train.micro_batch_size")
    accum = _int(train.get("grad_accum", 0), "train.grad_accum")
    global_batch = _int(train.get("global_batch_size", 0), "train.global_batch_size")
    expected = raw_world // sp_size * micro * accum
    _require(
        global_batch == expected,
        f"train.global_batch_size must equal DP*micro*grad_accum={expected}",
    )
    if profile.family == WAN_I2V_A14B:
        pixel_frames = _int(_mapping(config, "data").get("pixel_frames"), "data.pixel_frames")
        expected_frames = [11, 10] if pixel_frames == 81 else [20, 19]
        expected_tokens = [value * profile.frame_sequence_length for value in expected_frames]
        _require(
            str(distributed.get("sequence_layout", "")) == "whole_frame_uneven",
            "A14B requires distributed.sequence_layout=whole_frame_uneven",
        )
        _require(
            list(distributed.get("sp_frame_splits", [])) == expected_frames,
            f"A14B SP frame splits must be {expected_frames}",
        )
        _require(
            list(distributed.get("sp_token_splits", [])) == expected_tokens,
            f"A14B SP token splits must be {expected_tokens}",
        )


def _validate_common_model(config: Mapping[str, Any], route: Route, profile: FamilyProfile) -> None:
    model = _mapping(config, "model")
    _validate_model_assets(model)
    _require(
        str(model.get("family", "")) == profile.family, f"model.family must be {profile.family}"
    )
    _require(
        str(model.get("name", "")) == profile.model_name, f"model.name must be {profile.model_name}"
    )
    _require(
        _finite_float(model.get("timestep_shift"), "model.timestep_shift")
        == profile.timestep_shift,
        f"model.timestep_shift must be {profile.timestep_shift:g}",
    )
    _require(
        _int(model.get("latent_channels", 0), "model.latent_channels") == profile.latent_channels,
        f"model.latent_channels must be {profile.latent_channels}",
    )
    _require(
        _int(model.get("frame_sequence_length", 0), "model.frame_sequence_length")
        == profile.frame_sequence_length,
        f"model.frame_sequence_length must be {profile.frame_sequence_length}",
    )
    _require(
        str(model.get("camera_attention_mode", "")) == "fused_prope",
        "model.camera_attention_mode must be fused_prope",
    )
    transform = str(model.get("camera_translation_transform", "linear")).strip().lower()
    _require(
        transform in {"linear", "logd4"},
        "model.camera_translation_transform must be linear or logd4",
    )
    causal = bool(model.get("causal"))
    _require(
        causal is (route.stage in {"stage1", "stage2"}),
        f"model.causal does not match {route.stage}",
    )


def _validate_model_assets(model: Mapping[str, Any]) -> None:
    base = str(model.get("base_path", "")).strip()
    _require(base.startswith("/"), "model.base_path must be absolute")
    assets = _mapping(model, "assets")
    required = {
        "transformer_config",
        "transformer_weights",
        "text_encoder",
        "tokenizer",
        "vae",
    }
    _require(
        set(assets) == required,
        f"model.assets fields must be exactly {sorted(required)}",
    )
    for name in sorted(required):
        value = str(assets.get(name, "")).strip()
        if name == "transformer_config" and value == "builtin":
            continue
        path = PurePosixPath(value)
        _require(
            bool(value)
            and "://" not in value
            and "\\" not in value
            and all(part not in {"", ".", ".."} for part in path.parts if part != "/"),
            f"model.assets.{name} must be an absolute or base-relative POSIX path",
        )


def _validate_optimizer(train: Mapping[str, Any], route: Route) -> None:
    optimizer = _mapping(train, "optimizer")
    expected_lr = 2.0e-6 if route.stage == "stage2" else 5.0e-5
    expected_betas = [0.0, 0.999] if route.stage == "stage2" else [0.9, 0.95]
    _require(
        _finite_float(optimizer.get("lr"), "train.optimizer.lr") == expected_lr,
        f"train.optimizer.lr must be {expected_lr:g}",
    )
    _require(
        list(optimizer.get("betas", [])) == expected_betas,
        f"train.optimizer.betas must be {expected_betas}",
    )


def _validate_ema(train: Mapping[str, Any], route: Route) -> None:
    ema = _mapping(train, "ema")
    _require(bool(ema.get("enabled")), "train.ema.enabled must be true")
    _require(str(ema.get("device", "")) == "cuda", "train.ema.device must be cuda")
    _require(str(ema.get("dtype", "")) == "float32", "train.ema.dtype must be float32")
    _require(bool(ema.get("sharded")), "train.ema.sharded must be true")
    if route.stage == "stage2":
        _require(
            _finite_float(ema.get("decay"), "train.ema.decay") == 0.99,
            "Stage2 train.ema.decay must be 0.99",
        )
        _require(
            _int(ema.get("start_step"), "train.ema.start_step") == 39,
            "Stage2 EMA starts after student update 39",
        )
    elif route.objective == "anyflow_forward_map":
        _require(
            _finite_float(ema.get("decay"), "train.ema.decay") == 0.999,
            "Stage1 AnyFlow train.ema.decay must be 0.999",
        )
        _require(
            _int(ema.get("start_step"), "train.ema.start_step") == 0,
            "Stage1 AnyFlow EMA must exist at step 0",
        )
        _require(
            _int(ema.get("warmup_steps"), "train.ema.warmup_steps") == 1000,
            "Stage1 AnyFlow EMA decay warmup must be 1000",
        )
    elif route.stage == "stage0p5":
        decay = _finite_float(ema.get("decay"), "train.ema.decay")
        _require(
            decay in {0.999, 0.9999},
            "Stage0.5 train.ema.decay must be 0.999 or 0.9999",
        )
        if decay == 0.999:
            _require(
                _int(ema.get("warmup_steps"), "train.ema.warmup_steps") == 500,
                "Stage0.5 EMA decay 0.999 requires the 500-step warmup",
            )
        _require(
            _int(ema.get("start_step"), "train.ema.start_step") == 0,
            "Stage0.5 EMA must exist at step 0",
        )
    else:
        _require(
            _finite_float(ema.get("decay"), "train.ema.decay") == 0.9999,
            "train.ema.decay must be 0.9999",
        )
        _require(
            _int(ema.get("start_step"), "train.ema.start_step") == 0,
            "Wan training initializes EMA at step 0",
        )


def _validate_anyflow(train: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    from .anyflow import validate_anyflow_config

    validate_anyflow_config(train, validation)


def _validate_stage(config: Mapping[str, Any], route: Route, profile: FamilyProfile) -> None:
    train = _mapping(config, "train")
    model = _mapping(config, "model")
    data = _mapping(config, "data")
    validation = _mapping(config, "validation")
    latent_frames = _int(data.get("latent_frames"), "data.latent_frames")
    _require(
        _int(model.get("num_output_frames", 0), "model.num_output_frames") == latent_frames,
        "model.num_output_frames must match data.latent_frames",
    )
    _require(
        _int(model.get("num_frame_per_block", 0), "model.num_frame_per_block") == 3,
        "Wan training requires three latent frames per block",
    )
    _require(
        str(train.get("timestep_mode", ""))
        == ("uniform_per_video" if route.stage == "stage0p5" else "per_block"),
        "train.timestep_mode does not match the stage contract",
    )
    _require(
        _int(train.get("num_train_timesteps", 0), "train.num_train_timesteps") == 1000,
        "train.num_train_timesteps must be 1000",
    )
    _require(
        _finite_float(train.get("i2v_image_condition_dropout"), "train.i2v_image_condition_dropout")
        == (0.0 if route.stage == "stage2" else 0.1),
        "train.i2v_image_condition_dropout does not match the stage contract",
    )
    _validate_optimizer(train, route)
    _validate_ema(train, route)

    fsdp = _mapping(train, "fsdp")
    _require(
        str(fsdp.get("strategy", "")) == profile.fsdp_strategy,
        f"train.fsdp.strategy must be {profile.fsdp_strategy}",
    )
    _require(
        str(fsdp.get("param_dtype", "")) == "bfloat16", "train.fsdp.param_dtype must be bfloat16"
    )
    _require(
        str(fsdp.get("reduce_dtype", "")) == "float32", "train.fsdp.reduce_dtype must be float32"
    )

    if route.stage == "stage0p5":
        _require(route.causal_mode == "bidirectional", "Stage0.5 must be bidirectional")
        _require(route.objective == "flow_matching", "Stage0.5 supports flow_matching only")
        expected_window = latent_frames if profile.family == WAN_I2V_A14B else 21
        _require(
            _int(model.get("local_attn_size", 0), "model.local_attn_size") == expected_window,
            f"Stage0.5 model.local_attn_size must be {expected_window}",
        )
    elif route.stage == "stage1":
        _require(profile.family == WAN_TI2V_5B, "A14B Stage1 is not supported")
        _require(
            model.get("use_echorope") is False,
            "Stage1 requires model.use_echorope=false",
        )
        _require(
            route.causal_mode == "teacher_forcing",
            "Stage1 uses independent teacher forcing",
        )
        _require(
            _int(model.get("max_prior_clean_chunks", -1), "model.max_prior_clean_chunks") == 5,
            "Stage1 six-chunk context requires max_prior_clean_chunks=5",
        )
        _require(
            _int(model.get("local_attn_size", 0), "model.local_attn_size") == 21,
            "Stage1 requires local_attn_size=21",
        )
        if str(config.get("action", "")).strip().lower() == "train":
            _require(
                _mapping(config, "runtime").get("compile_flex") is True,
                "Stage1 training requires runtime.compile_flex=true",
            )
        _require(
            _int(
                train.get("noise_augmentation_max_timestep", -1),
                "train.noise_augmentation_max_timestep",
            )
            == 0,
            "Stage1 clean context requires noise_augmentation_max_timestep=0",
        )
        if route.objective == "anyflow_forward_map":
            _validate_anyflow(train, validation)
    elif route.stage == "stage2":
        from .sgf import validate_stage2_contract

        validate_stage2_contract(config)


def _validate_checkpoint(config: Mapping[str, Any], route: Route) -> None:
    checkpoint = _mapping(config, "checkpoint")
    action = str(config.get("action", "")).strip().lower()
    if action == "infer":
        inference = _mapping(config, "inference")
        camera_length = (
            route.stage == "stage2"
            and str(inference.get("length", "fixed")).strip().lower() == "camera"
        )
        _require(
            str(checkpoint.get("path", "")).startswith("/"),
            "inference checkpoint.path must be absolute",
        )
        if camera_length:
            _require(
                str(checkpoint.get("weights", "")) in {"", "model"},
                "camera-length inference selects the checkpoint model automatically",
            )
        else:
            _require(
                str(checkpoint.get("weights", "")) in {"live", "ema"},
                "inference checkpoint.weights must be live or ema",
            )
        checkpoint_format = str(checkpoint.get("format", "embedded_config_v1")).strip().lower()
        _require(
            checkpoint_format == "embedded_config_v1",
            "inference checkpoint.format must be embedded_config_v1",
        )
        return
    mode = str(checkpoint.get("mode", "")).strip().lower()
    _require(
        mode in {"none", "weights_only", "full_resume", "stage2_roles"},
        "checkpoint.mode is invalid",
    )
    if route.stage == "stage1":
        _require(
            mode in {"weights_only", "full_resume"},
            "Stage1 must start weights-only from Stage0.5 or fully resume Stage1",
        )
    elif route.stage == "stage2":
        _require(mode == "stage2_roles", "Stage2 requires explicit three-role initialization")
    elif _int(_mapping(config, "data").get("pixel_frames"), "data.pixel_frames") == 153:
        _require(
            mode in {"weights_only", "full_resume"},
            "153f training requires an explicit source checkpoint",
        )
    else:
        _require(
            mode in {"none", "weights_only", "full_resume"},
            "81f Stage0.5 requires fresh, explicit weights-only, or full-resume initialization",
        )

    if mode in {"weights_only", "full_resume"}:
        _require(
            str(checkpoint.get("path", "")).startswith("/"),
            "checkpoint.path must be an absolute /path/to placeholder or runtime path",
        )
    source = checkpoint.get("source_contract")
    if mode in {"weights_only", "full_resume"}:
        _require(
            isinstance(source, Mapping),
            "checkpoint.source_contract is required for checkpoint initialization",
        )
    if source is not None:
        _require(isinstance(source, Mapping), "checkpoint.source_contract must be a mapping")
        current = str(_mapping(config, "model").get("camera_translation_transform", "linear"))
        previous = str(source.get("camera_translation_transform", "linear"))
        if current != previous:
            allowed = bool(
                _mapping(config, "train").get(
                    "allow_camera_translation_transform_change_on_init", False
                )
            )
            _require(
                mode == "weights_only" and allowed,
                "camera translation transform mismatch is allowed only for an "
                "explicit weights-only ablation",
            )
        expected_source_stage = route.stage if mode == "full_resume" else "stage0p5"
        expected_source_objective = (
            str(_mapping(config, "train").get("objective"))
            if mode == "full_resume"
            else "flow_matching"
        )
        _require(
            str(source.get("stage", "")) == expected_source_stage,
            f"Wan {mode} initialization requires a {expected_source_stage} source",
        )
        _require(
            str(source.get("objective", "")) == expected_source_objective,
            f"Wan {mode} initialization requires objective {expected_source_objective}",
        )
    if mode == "full_resume":
        resume_step = _int(checkpoint.get("resume_step", 0), "checkpoint.resume_step")
        _require(
            resume_step > 0,
            "full resume requires a positive checkpoint.resume_step",
        )
    if mode == "weights_only" and route.stage == "stage1":
        _require(
            str(checkpoint.get("weights", "")) == "ema",
            "Stage1 weights-only initialization requires EMA weights",
        )
        _require(
            _int(checkpoint.get("source_step", 0), "checkpoint.source_step") > 0,
            "Stage1 weights-only initialization requires checkpoint.source_step",
        )
    if mode == "weights_only" and route.stage == "stage0p5":
        weights = list(checkpoint.get("weights", []))
        _require(
            weights in (["live", "ema"], ["ema", "ema"]),
            "Stage0.5 weights-only initialization must select LIVE/EMA or EMA/EMA",
        )
        _require(
            _int(checkpoint.get("source_step", 0), "checkpoint.source_step") > 0,
            "Stage0.5 weights-only initialization requires checkpoint.source_step",
        )
    if (
        mode == "weights_only"
        and route.family == WAN_I2V_A14B
        and _int(_mapping(config, "data").get("pixel_frames"), "data.pixel_frames") == 153
    ):
        _require(
            list(checkpoint.get("weights", [])) == ["live", "ema"],
            "A14B 153f initialization must inherit source LIVE and EMA weights",
        )
        _require(
            _int(checkpoint.get("source_step", 0), "checkpoint.source_step") > 0,
            "A14B 153f weights-only initialization requires checkpoint.source_step",
        )


def validate_wan_config(config: Mapping[str, Any], *, expected_family: str) -> Route:
    """Validate a train/infer config without allocating a model or device."""

    if expected_family not in WAN_FAMILIES:
        raise BackendContractError(f"unknown Wan family {expected_family!r}")
    action = str(config.get("action", "")).strip().lower()
    _require(action in {"train", "infer"}, "Wan training contract expects action train or infer")
    try:
        route = validate_route(config)
    except ConfigurationError as exc:
        raise BackendContractError(str(exc)) from exc
    _require(
        route.family == expected_family,
        f"backend {expected_family} cannot execute route {route.family}",
    )
    profile = _profile_for_config(config, expected_family)
    _validate_common_model(config, route, profile)
    _validate_data(config, route, profile)
    _validate_distributed(config, profile)
    _validate_stage(config, route, profile)
    _validate_checkpoint(config, route)
    _validate_runtime_caches(config)
    return route


def stable_routes(family: str) -> tuple[str, ...]:
    """Return the explicitly supported route keys for documentation/tools."""

    if family == WAN_TI2V_5B:
        return (
            "wan22_ti2v_5b:stage0p5:bidirectional:flow_matching",
            "wan22_ti2v_5b:stage1:teacher_forcing:flow_matching",
            "wan22_ti2v_5b:stage1:teacher_forcing:anyflow_forward_map:v1_5",
            "wan22_ti2v_5b:stage2:self_gradient_forcing:flow_matching",
        )
    if family == WAN_I2V_A14B:
        return ("wan22_i2v_a14b:stage0p5:bidirectional:flow_matching",)
    raise BackendContractError(f"unknown Wan family {family!r}")
