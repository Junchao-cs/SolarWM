"""Base-weight, LoRA, EMA, and full-resume identity contracts."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from solar.errors import BackendContractError

from .adapter import lora_target_modules
from .artifact import PREENCODE_VERSION, READER_CONTRACT
from .geometry import STABLE_GEOMETRY

VIDEO_CORE_TENSORS: Final = 1362
VIDEO_CORE_PARAMETERS: Final = 13_123_337_344
VIDEO_CONNECTOR_TENSORS: Final = 129
VIDEO_CONNECTOR_PARAMETERS: Final = 1_612_546_304
FP32_SCALE_TABLES: Final = 97
LORA_TRAINABLE_PARAMETERS_R384: Final = 1_962_934_272
BASE_RETAINED_TENSORS: Final = VIDEO_CORE_TENSORS + VIDEO_CONNECTOR_TENSORS
BASE_RETAINED_PARAMETERS: Final = VIDEO_CORE_PARAMETERS + VIDEO_CONNECTOR_PARAMETERS
BASE_AUDIO_TENSORS: Final = 1_586
BASE_AV_CROSS_TENSORS: Final = 1_272
BASE_AUDIO_PARAMETERS: Final = 3_690_028_416
BASE_AV_CROSS_PARAMETERS: Final = 2_578_113_536
MAX_SAFETENSORS_HEADER_BYTES: Final = 64 * 1024 * 1024
STATE_DICT_PREFIX: Final = "model.diffusion_model."

_DTYPE_BYTES: Final = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "C64": 8,
    "I64": 8,
    "U64": 8,
    "F64": 8,
    "C128": 16,
}

_VIDEO_TOP_LEVEL_PREFIXES: Final = (
    "patchify_proj.",
    "adaln_single.",
    "prompt_adaln_single.",
    "proj_out.",
)
_VIDEO_TOP_LEVEL_EXACT: Final = frozenset({"keyframes_abs_pos_embedding", "scale_shift_table"})
_VIDEO_BLOCK_MODULES: Final = frozenset({"attn1", "attn2", "ff"})
_VIDEO_BLOCK_EXACT: Final = frozenset({"scale_shift_table", "prompt_scale_shift_table"})
_AUDIO_TOP_LEVEL_PREFIXES: Final = (
    "audio_patchify_proj.",
    "audio_proj_out.",
    "audio_adaln_single.",
    "audio_prompt_adaln_single.",
    "audio_embeddings_connector.",
)
_AUDIO_TOP_LEVEL_EXACT: Final = frozenset({"audio_scale_shift_table"})
_AUDIO_BLOCK_MODULES: Final = frozenset({"audio_attn1", "audio_attn2", "audio_ff"})
_AUDIO_BLOCK_EXACT: Final = frozenset({"audio_scale_shift_table", "audio_prompt_scale_shift_table"})
_AV_TOP_LEVEL_PREFIXES: Final = (
    "av_ca_video_scale_shift_adaln_single.",
    "av_ca_audio_scale_shift_adaln_single.",
    "av_ca_a2v_gate_adaln_single.",
    "av_ca_v2a_gate_adaln_single.",
)
_AV_BLOCK_MODULES: Final = frozenset({"audio_to_video_attn", "video_to_audio_attn"})
_AV_BLOCK_EXACT: Final = frozenset(
    {"scale_shift_table_a2v_ca_audio", "scale_shift_table_a2v_ca_video"}
)


@dataclass(frozen=True)
class SafetensorHeaderEntry:
    """One structurally validated tensor entry from a safetensors header."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    data_offsets: tuple[int, int]

    @property
    def parameter_count(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


@dataclass(frozen=True)
class SafetensorsHeader:
    metadata: Mapping[str, str]
    tensors: tuple[SafetensorHeaderEntry, ...]
    header_size: int
    file_size: int
    header_digest: str


@dataclass(frozen=True)
class BaseCheckpointInspection:
    """Header-only authority passed to the heavy runtime before allocation."""

    path: Path
    file_size: int
    header_size: int
    header_digest: str
    retained_layout_digest: str
    video_core_tensors: int
    video_core_parameters: int
    video_connector_tensors: int
    video_connector_parameters: int
    fp32_scale_tables: int
    audio_tensors: int
    audio_parameters: int
    av_cross_tensors: int
    av_cross_parameters: int

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["path"] = str(self.path)
        return result


@dataclass(frozen=True)
class InferenceAdapterCheckpoint:
    """Verified adapter authority for a standard checkpoint transaction."""

    path: Path
    tensor_path: Path
    manifest_digest: str
    weights: str
    checkpoint_format: str


def _json_object_without_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BackendContractError(f"duplicate JSON key in safetensors header: {key!r}")
        result[key] = value
    return result


def _nonnegative_integer(value: Any, *, field: str, tensor: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendContractError(
            f"safetensors {tensor!r} {field} must contain nonnegative integers"
        )
    return value


def _header_entry(name: str, value: Any) -> SafetensorHeaderEntry:
    if not name or not isinstance(value, dict):
        raise BackendContractError(f"invalid safetensors tensor entry {name!r}")
    if set(value) != {"dtype", "shape", "data_offsets"}:
        raise BackendContractError(f"safetensors entry {name!r} has unknown fields")
    dtype = value["dtype"]
    shape = value["shape"]
    offsets = value["data_offsets"]
    if dtype not in _DTYPE_BYTES or not isinstance(shape, list):
        raise BackendContractError(f"safetensors entry {name!r} has invalid dtype/shape")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise BackendContractError(f"safetensors entry {name!r} has invalid offsets")
    parsed_shape = tuple(_nonnegative_integer(item, field="shape", tensor=name) for item in shape)
    start, end = (_nonnegative_integer(item, field="data_offsets", tensor=name) for item in offsets)
    if end < start:
        raise BackendContractError(f"safetensors entry {name!r} has reversed offsets")
    entry = SafetensorHeaderEntry(name, dtype, parsed_shape, (start, end))
    if end - start != entry.parameter_count * _DTYPE_BYTES[dtype]:
        raise BackendContractError(
            f"safetensors entry {name!r} byte range disagrees with dtype/shape"
        )
    return entry


def read_safetensors_header(
    path: str | os.PathLike[str],
    *,
    max_header_bytes: int = MAX_SAFETENSORS_HEADER_BYTES,
) -> SafetensorsHeader:
    """Read only the bounded header and validate every tensor byte range."""

    source = Path(path)
    if (
        isinstance(max_header_bytes, bool)
        or not isinstance(max_header_bytes, int)
        or max_header_bytes < 2
    ):
        raise BackendContractError("max_header_bytes must be an integer >= 2")
    try:
        file_size = source.stat().st_size
        with source.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise BackendContractError("safetensors length prefix is truncated")
            header_size = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_size <= max_header_bytes:
                raise BackendContractError(
                    f"safetensors header size {header_size} is outside the accepted range"
                )
            raw_header = handle.read(header_size)
    except OSError as exc:
        raise BackendContractError(f"cannot read checkpoint header {source}: {exc}") from exc
    if len(raw_header) != header_size or 8 + header_size > file_size:
        raise BackendContractError("safetensors JSON header is truncated")
    try:
        value = json.loads(
            raw_header.decode("utf-8").rstrip(" "),
            object_pairs_hook=_json_object_without_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BackendContractError(f"invalid safetensors JSON header in {source}") from exc
    if not isinstance(value, dict):
        raise BackendContractError("safetensors header must be a JSON object")
    metadata = value.pop("__metadata__", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(item, str) for key, item in metadata.items()
    ):
        raise BackendContractError("safetensors metadata must contain only strings")
    tensors = tuple(_header_entry(name, item) for name, item in value.items())
    cursor = 0
    for tensor in sorted(tensors, key=lambda item: (*item.data_offsets, item.name)):
        if tensor.data_offsets[0] != cursor:
            raise BackendContractError(f"safetensors data is not contiguous before {tensor.name!r}")
        cursor = tensor.data_offsets[1]
    if cursor != file_size - 8 - header_size:
        raise BackendContractError("safetensors data ranges do not cover the file")
    return SafetensorsHeader(
        metadata=dict(metadata),
        tensors=tensors,
        header_size=header_size,
        file_size=file_size,
        header_digest=hashlib.blake2s(prefix + raw_header).hexdigest(),
    )


def _classify_tensor(name: str) -> tuple[str, str | None]:
    if not name.startswith(STATE_DICT_PREFIX):
        return "unknown", None
    stripped = name[len(STATE_DICT_PREFIX) :]
    if stripped.startswith("video_embeddings_connector."):
        return "video_connector", "video_connector." + stripped.removeprefix(
            "video_embeddings_connector."
        )
    if stripped in _VIDEO_TOP_LEVEL_EXACT or stripped.startswith(_VIDEO_TOP_LEVEL_PREFIXES):
        return "video_core", stripped
    if stripped in _AUDIO_TOP_LEVEL_EXACT or stripped.startswith(_AUDIO_TOP_LEVEL_PREFIXES):
        return "audio", None
    if stripped.startswith(_AV_TOP_LEVEL_PREFIXES):
        return "av_cross", None
    parts = stripped.split(".")
    if (
        len(parts) < 3
        or parts[0] != "transformer_blocks"
        or not parts[1].isdigit()
        or (len(parts[1]) > 1 and parts[1].startswith("0"))
        or not 0 <= int(parts[1]) < 48
    ):
        return "unknown", None
    component = parts[2]
    suffix = parts[3:]
    if (component in _VIDEO_BLOCK_MODULES and suffix) or (
        component in _VIDEO_BLOCK_EXACT and not suffix
    ):
        return "video_core", stripped
    if (component in _AUDIO_BLOCK_MODULES and suffix) or (
        component in _AUDIO_BLOCK_EXACT and not suffix
    ):
        return "audio", None
    if (component in _AV_BLOCK_MODULES and suffix) or (component in _AV_BLOCK_EXACT and not suffix):
        return "av_cross", None
    return "unknown", None


def _metadata_object(metadata: Mapping[str, str], key: str) -> Mapping[str, Any]:
    try:
        result = json.loads(metadata[key], object_pairs_hook=_json_object_without_duplicates)
    except (KeyError, json.JSONDecodeError) as exc:
        raise BackendContractError(f"checkpoint metadata {key!r} is missing or invalid") from exc
    if not isinstance(result, dict):
        raise BackendContractError(f"checkpoint metadata {key!r} is not an object")
    return result


def _validate_base_metadata(metadata: Mapping[str, str]) -> None:
    if metadata.get("model_version") != "2.5.0":
        raise BackendContractError("LTX checkpoint model_version must be 2.5.0")
    config = _metadata_object(metadata, "config")
    transformer = config.get("transformer")
    scheduler = config.get("scheduler")
    if not isinstance(transformer, Mapping) or not isinstance(scheduler, Mapping):
        raise BackendContractError("LTX checkpoint lacks transformer/scheduler metadata")
    expected_transformer = {
        "_class_name": "AVTransformer3DModel",
        "num_layers": 48,
        "num_attention_heads": 32,
        "attention_head_dim": 128,
        "in_channels": 128,
        "out_channels": 128,
        "cross_attention_dim": 4096,
        "caption_channels": 3840,
        "connector_num_layers": 8,
        "connector_num_attention_heads": 32,
        "connector_attention_head_dim": 128,
        "use_audio_video_cross_attention": True,
        "use_embeddings_connector": True,
    }
    for key, expected in expected_transformer.items():
        if transformer.get(key) != expected:
            raise BackendContractError(
                f"LTX checkpoint transformer.{key} differs from the required model layout"
            )
    if (
        scheduler.get("_class_name") != "RectifiedFlowScheduler"
        or scheduler.get("num_train_timesteps") != 1000
        or scheduler.get("sampler") != "LinearQuadratic"
    ):
        raise BackendContractError("LTX checkpoint scheduler metadata drifted")
    gemma = _metadata_object(metadata, "gemma_source_checkpoint")
    if dict(gemma) != {
        "ltx_version": "2.5.0",
        "gemma_version": "gemma4-12b-ltx-v1",
    }:
        raise BackendContractError("LTX checkpoint Gemma connector identity drifted")


def inspect_base_checkpoint(path: str | os.PathLike[str]) -> BaseCheckpointInspection:
    """Fail closed on any base header, key, dtype, or layout drift."""

    source = Path(path)
    header = read_safetensors_header(source)
    _validate_base_metadata(header.metadata)
    grouped: dict[str, list[SafetensorHeaderEntry]] = {
        "video_core": [],
        "video_connector": [],
        "audio": [],
        "av_cross": [],
    }
    retained: list[tuple[str, SafetensorHeaderEntry]] = []
    unknown: list[str] = []
    for tensor in header.tensors:
        category, target = _classify_tensor(tensor.name)
        if category == "unknown":
            unknown.append(tensor.name)
            continue
        grouped[category].append(tensor)
        if target is not None:
            retained.append((target, tensor))
    if unknown:
        raise BackendContractError(f"unknown LTX checkpoint keys: {unknown[:8]!r}")
    stats = {
        key: (len(items), sum(item.parameter_count for item in items))
        for key, items in grouped.items()
    }
    expected_stats = {
        "video_core": (VIDEO_CORE_TENSORS, VIDEO_CORE_PARAMETERS),
        "video_connector": (VIDEO_CONNECTOR_TENSORS, VIDEO_CONNECTOR_PARAMETERS),
        "audio": (BASE_AUDIO_TENSORS, BASE_AUDIO_PARAMETERS),
        "av_cross": (BASE_AV_CROSS_TENSORS, BASE_AV_CROSS_PARAMETERS),
    }
    if stats != expected_stats:
        raise BackendContractError(f"LTX checkpoint category statistics drifted: {stats!r}")
    core_dtypes = Counter(item.dtype for item in grouped["video_core"])
    connector_dtypes = Counter(item.dtype for item in grouped["video_connector"])
    if core_dtypes != {"BF16": 1265, "F32": 97} or connector_dtypes != {"BF16": 129}:
        raise BackendContractError("LTX retained checkpoint dtype policy drifted")
    fp32_targets = {target for target, tensor in retained if tensor.dtype == "F32"}
    expected_fp32 = {"scale_shift_table"}
    for index in range(48):
        expected_fp32.add(f"transformer_blocks.{index}.scale_shift_table")
        expected_fp32.add(f"transformer_blocks.{index}.prompt_scale_shift_table")
    if fp32_targets != expected_fp32:
        raise BackendContractError("LTX FP32 scale-table inventory drifted")
    if len({target for target, _ in retained}) != BASE_RETAINED_TENSORS:
        raise BackendContractError("LTX retained target keys are not unique")
    layout = hashlib.blake2s()
    for target, tensor in sorted(retained):
        layout.update(
            json.dumps(
                {
                    "dtype": tensor.dtype,
                    "shape": list(tensor.shape),
                    "source": tensor.name,
                    "target": target,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        layout.update(b"\n")
    return BaseCheckpointInspection(
        path=source.resolve(),
        file_size=header.file_size,
        header_size=header.header_size,
        header_digest=header.header_digest,
        retained_layout_digest=layout.hexdigest(),
        video_core_tensors=stats["video_core"][0],
        video_core_parameters=stats["video_core"][1],
        video_connector_tensors=stats["video_connector"][0],
        video_connector_parameters=stats["video_connector"][1],
        fp32_scale_tables=len(fp32_targets),
        audio_tensors=stats["audio"][0],
        audio_parameters=stats["audio"][1],
        av_cross_tensors=stats["av_cross"][0],
        av_cross_parameters=stats["av_cross"][1],
    )


@dataclass(frozen=True)
class BaseCheckpointContract:
    video_core_tensors: int = VIDEO_CORE_TENSORS
    video_core_parameters: int = VIDEO_CORE_PARAMETERS
    video_connector_tensors: int = VIDEO_CONNECTOR_TENSORS
    video_connector_parameters: int = VIDEO_CONNECTOR_PARAMETERS
    fp32_scale_tables: int = FP32_SCALE_TABLES
    dropped_streams: tuple[str, ...] = ("audio", "audio_video_cross_attention")
    strict_load: bool = True

    def __post_init__(self) -> None:
        observed = (
            self.video_core_tensors,
            self.video_core_parameters,
            self.video_connector_tensors,
            self.video_connector_parameters,
            self.fp32_scale_tables,
            self.dropped_streams,
            self.strict_load,
        )
        expected = (
            VIDEO_CORE_TENSORS,
            VIDEO_CORE_PARAMETERS,
            VIDEO_CONNECTOR_TENSORS,
            VIDEO_CONNECTOR_PARAMETERS,
            FP32_SCALE_TABLES,
            ("audio", "audio_video_cross_attention"),
            True,
        )
        if observed != expected:
            raise BackendContractError("LTX base checkpoint contract drifted")


BASE_CHECKPOINT_CONTRACT = BaseCheckpointContract()


@dataclass(frozen=True)
class LoRACheckpointContract:
    rank: int = 384
    alpha: int = 384
    dropout: float = 0.0
    target_count: int = 480
    trainable_parameters: int = LORA_TRAINABLE_PARAMETERS_R384
    base_scale_tables_trainable: bool = False

    def __post_init__(self) -> None:
        expected = (384, 384, 0.0, 480, LORA_TRAINABLE_PARAMETERS_R384, False)
        observed = (
            self.rank,
            self.alpha,
            float(self.dropout),
            self.target_count,
            self.trainable_parameters,
            self.base_scale_tables_trainable,
        )
        if observed != expected:
            raise BackendContractError("LTX LoRA-384 contract drifted")


LORA_CHECKPOINT_CONTRACT = LoRACheckpointContract()


@dataclass(frozen=True)
class EMAContract:
    enabled: bool = True
    start_step: int = 0
    update_every_steps: int = 1
    decay: float = 0.999
    device: str = "cuda"
    dtype: str = "float32"
    sharded: bool = True
    trainable_only: bool = True

    def __post_init__(self) -> None:
        expected = (True, 0, 1, 0.999, "cuda", "float32", True, True)
        observed = (
            self.enabled,
            self.start_step,
            self.update_every_steps,
            float(self.decay),
            self.device,
            self.dtype,
            self.sharded,
            self.trainable_only,
        )
        if observed != expected:
            raise BackendContractError("LTX EMA must be CUDA FP32 sharded trainable-only")


EMA_CONTRACT = EMAContract()


@dataclass(frozen=True)
class StrictModelLoadReceipt:
    """Record that the inspected base was loaded into the exact model layout."""

    provider_identity: str
    ltx_core_version: str
    header_digest: str
    retained_layout_digest: str
    strict_state_dict: bool
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    video_core_tensors: int
    video_core_parameters: int
    video_connector_tensors: int
    video_connector_parameters: int
    fp32_scale_tables: int
    dropped_streams: tuple[str, ...]
    adapter_target_count: int
    adapter_targets: tuple[str, ...]
    adapter_trainable_parameters: int
    adapter_mode: str
    adapter_checkpoint_manifest_digest: str = ""
    fused_prope_parameter_free: bool = False
    schema: str = "solarwm.ltx25.strict-model-load.v1"

    def validate(
        self,
        inspection: BaseCheckpointInspection,
        *,
        adapter_checkpoint_manifest_digest: str = "",
    ) -> None:
        expected = (
            inspection.header_digest,
            inspection.retained_layout_digest,
            VIDEO_CORE_TENSORS,
            VIDEO_CORE_PARAMETERS,
            VIDEO_CONNECTOR_TENSORS,
            VIDEO_CONNECTOR_PARAMETERS,
            FP32_SCALE_TABLES,
            ("audio", "audio_video_cross_attention"),
            LORA_CHECKPOINT_CONTRACT.target_count,
            lora_target_modules(),
            LORA_CHECKPOINT_CONTRACT.trainable_parameters,
        )
        observed = (
            self.header_digest,
            self.retained_layout_digest,
            self.video_core_tensors,
            self.video_core_parameters,
            self.video_connector_tensors,
            self.video_connector_parameters,
            self.fp32_scale_tables,
            tuple(self.dropped_streams),
            self.adapter_target_count,
            tuple(self.adapter_targets),
            self.adapter_trainable_parameters,
        )
        if self.schema != "solarwm.ltx25.strict-model-load.v1":
            raise BackendContractError("unknown LTX strict model-load receipt schema")
        if not self.provider_identity.strip() or not self.ltx_core_version.strip():
            raise BackendContractError("LTX model-load receipt lacks provider/LTX identity")
        if observed != expected:
            raise BackendContractError("LTX model-load receipt differs from strict inspection")
        if (
            not self.strict_state_dict
            or self.missing_keys
            or self.unexpected_keys
            or not self.fused_prope_parameter_free
        ):
            raise BackendContractError("LTX model load was not strict and parameter-free")
        if self.adapter_mode not in {"initialized", "checkpoint"}:
            raise BackendContractError("LTX adapter_mode must be initialized or checkpoint")
        if adapter_checkpoint_manifest_digest:
            if (
                self.adapter_mode != "checkpoint"
                or self.adapter_checkpoint_manifest_digest != adapter_checkpoint_manifest_digest
            ):
                raise BackendContractError(
                    "LTX adapter load is not bound to the verified shared checkpoint"
                )
        elif self.adapter_mode != "initialized" or self.adapter_checkpoint_manifest_digest:
            raise BackendContractError("fresh LTX training must initialize a new LoRA adapter")

    def as_dict(self) -> dict[str, Any]:
        """Return a readable, JSON-compatible receipt for checkpoint metadata."""

        value = asdict(self)
        for name in ("missing_keys", "unexpected_keys", "dropped_streams", "adapter_targets"):
            value[name] = list(value[name])
        return value


@dataclass(frozen=True)
class StrictCodecLoadReceipt:
    """Official DiffVAE/Gemma asset and implementation identity."""

    provider_identity: str
    video_vae_class: str
    diffvae_mode: str
    gemma_feature_extractor_class: str = ""
    caption_cache_stage: str = ""
    video_vae_operation: str = "diffvae_decode"
    video_vae_encoder_class: str = ""
    schema: str = "solarwm.ltx25.strict-codec-load.v1"

    def validate(
        self,
        *,
        require_gemma: bool,
    ) -> None:
        if self.schema != "solarwm.ltx25.strict-codec-load.v1":
            raise BackendContractError("unknown LTX strict codec-load receipt schema")
        if not self.provider_identity.strip():
            raise BackendContractError("LTX codec receipt lacks provider identity")

        official_video_vae_classes = (
            "ltx_core.",
            "ltx_pipelines.utils.blocks.VideoDecoder",
        )
        decode_valid = (
            self.video_vae_operation == "diffvae_decode"
            and self.video_vae_class.startswith(official_video_vae_classes)
            and self.diffvae_mode == "chunked_eager"
        )
        encode_valid = (
            self.video_vae_operation == "direct_encode"
            and self.video_vae_class.startswith("ltx_core.")
            and self.diffvae_mode == "direct"
        )
        if not (decode_valid or encode_valid):
            raise BackendContractError("LTX codec receipt does not identify official DiffVAE")
        if require_gemma and (
            not self.gemma_feature_extractor_class.startswith("ltx_core.")
            or self.caption_cache_stage != "gemma4_feature_extractor_preconnector"
        ):
            raise BackendContractError(
                "LTX codec receipt does not identify the official Gemma4 preconnector extractor"
            )
        if (
            require_gemma
            and self.video_vae_operation == "diffvae_decode"
            and (not self.video_vae_encoder_class.startswith("ltx_core."))
        ):
            raise BackendContractError(
                "LTX online training receipt does not bind the official direct VAE encoder"
            )

    @property
    def digest(self) -> str:
        return fingerprint_digest(asdict(self))


def runtime_fingerprint(
    *,
    camera_translation_transform: str,
    data_generation: str,
) -> dict[str, Any]:
    if camera_translation_transform not in {"linear", "logd4"}:
        raise BackendContractError("camera translation transform must be linear or logd4")
    if not data_generation.strip():
        raise BackendContractError("data_generation must be non-empty")
    base_checkpoint = asdict(BASE_CHECKPOINT_CONTRACT)
    base_checkpoint["dropped_streams"] = list(base_checkpoint["dropped_streams"])
    return {
        "schema": "solarwm.ltx25.checkpoint.v1",
        "family": "ltx25_video",
        "stage": "stage0p5",
        "objective": "native_rectified_flow",
        "geometry": asdict(STABLE_GEOMETRY),
        "reader_contract": READER_CONTRACT,
        "artifact_version": PREENCODE_VERSION,
        "data_generation": data_generation,
        "camera_translation_transform": camera_translation_transform,
        "base_checkpoint": base_checkpoint,
        "lora": asdict(LORA_CHECKPOINT_CONTRACT),
        "ema": asdict(EMA_CONTRACT),
    }


def fingerprint_digest(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2s(payload.encode()).hexdigest()


@dataclass(frozen=True)
class TrainingCheckpointManifest:
    """Logical contents required for a same-stage full resume."""

    global_step: int
    runtime_fingerprint: Mapping[str, Any]
    adapter_targets: Sequence[str]
    ema_targets: Sequence[str]
    optimizer_present: bool
    scheduler_present: bool

    def validate(self) -> None:
        if isinstance(self.global_step, bool) or self.global_step < 0:
            raise BackendContractError("global_step must be nonnegative")
        expected = lora_target_modules()
        if tuple(self.adapter_targets) != expected:
            raise BackendContractError("checkpoint adapter target inventory is not exact")
        if tuple(self.ema_targets) != expected:
            raise BackendContractError("EMA must contain exactly trainable LoRA targets")
        if not self.optimizer_present or not self.scheduler_present:
            raise BackendContractError("full resume requires optimizer and scheduler state")
        expected_schema = self.runtime_fingerprint.get("schema")
        if expected_schema != "solarwm.ltx25.checkpoint.v1":
            raise BackendContractError("checkpoint runtime fingerprint schema is missing")


def validate_full_resume(
    saved: TrainingCheckpointManifest,
    current_fingerprint: Mapping[str, Any],
) -> None:
    saved.validate()
    if dict(saved.runtime_fingerprint) != dict(current_fingerprint):
        differing = sorted(
            key
            for key in set(saved.runtime_fingerprint) | set(current_fingerprint)
            if saved.runtime_fingerprint.get(key) != current_fingerprint.get(key)
        )
        raise BackendContractError(
            f"LTX full resume requires identical runtime semantics; differing={differing}"
        )


__all__ = [
    "BASE_CHECKPOINT_CONTRACT",
    "BASE_RETAINED_PARAMETERS",
    "BASE_RETAINED_TENSORS",
    "EMA_CONTRACT",
    "FP32_SCALE_TABLES",
    "LORA_CHECKPOINT_CONTRACT",
    "LORA_TRAINABLE_PARAMETERS_R384",
    "BaseCheckpointInspection",
    "InferenceAdapterCheckpoint",
    "SafetensorHeaderEntry",
    "SafetensorsHeader",
    "StrictCodecLoadReceipt",
    "StrictModelLoadReceipt",
    "TrainingCheckpointManifest",
    "fingerprint_digest",
    "inspect_base_checkpoint",
    "read_safetensors_header",
    "runtime_fingerprint",
    "validate_full_resume",
]
