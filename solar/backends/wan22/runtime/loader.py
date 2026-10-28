"""Strict Wan transformer construction and shard-wise weight loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from solar.errors import BackendContractError

from .assets import WanAssetLayout


@dataclass(frozen=True)
class WeightLoadReport:
    shards: tuple[str, ...]
    source_keys: int
    target_keys: int
    stripped_model_prefix: bool
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    architecture_inventory: tuple[str, int] | None = None
    shard_inventory: tuple[tuple[str, int], ...] = ()
    initialized_extension_keys: tuple[str, ...] = ()
    initialization_id: str = ""

    def initialization_receipt(self) -> dict[str, Any]:
        if not self.initialization_id:
            raise BackendContractError("Wan weight report has no bound initialization receipt")
        return {
            "schema": "solarwm.wan22-initialization.v2",
            "initialization_id": self.initialization_id,
            "architecture": (
                {
                    "name": self.architecture_inventory[0],
                    "bytes": self.architecture_inventory[1],
                }
                if self.architecture_inventory is not None
                else None
            ),
            "initialized_extension_keys": list(self.initialized_extension_keys),
            "shards": [{"name": name, "bytes": size} for name, size in self.shard_inventory],
            "source_keys": self.source_keys,
            "target_keys": self.target_keys,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
        }


def _bind_initialization_receipt(
    *,
    architecture: Path,
    report: WeightLoadReport,
) -> WeightLoadReport:
    architecture_inventory = (architecture.name, architecture.stat().st_size)
    shard_inventory = tuple(
        (path.name, path.stat().st_size) for path in (Path(raw_path) for raw_path in report.shards)
    )
    shard_bytes = sum(size for _, size in shard_inventory)
    initialization_id = (
        f"inventory:architecture={architecture_inventory[0]}:{architecture_inventory[1]}:"
        f"shards={len(shard_inventory)}:{shard_bytes}:"
        f"source-keys={report.source_keys}:target-keys={report.target_keys}:"
        f"extensions={len(report.missing_keys)}"
    )
    return replace(
        report,
        architecture_inventory=architecture_inventory,
        shard_inventory=shard_inventory,
        initialized_extension_keys=tuple(sorted(report.missing_keys)),
        initialization_id=initialization_id,
    )


def _allowed_base_missing_key(key: str) -> bool:
    return (key.startswith("blocks.") and ".cam_self_attn." in key) or key.startswith(
        "delta_embedding."
    )


def _weight_shards(directory: Path) -> tuple[Path, ...]:
    if not directory.is_dir():
        raise BackendContractError(f"Wan transformer directory is missing: {directory}")
    shards = tuple(sorted(directory.glob("*.safetensors")))
    if not shards:
        raise BackendContractError(
            f"Wan transformer directory contains no .safetensors shards: {directory}"
        )
    return shards


def _source_keys(shards: tuple[Path, ...]) -> set[str]:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise BackendContractError("safetensors is required to load Wan weights") from exc

    keys: set[str] = set()
    for shard in shards:
        try:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                # ``safe_open`` is a context-managed handle, not an iterable
                # in the safetensors 0.8 API used by this runtime.
                # Use its explicit inventory surface so this remains valid
                # across the supported safetensors versions.
                for key in handle.keys():  # noqa: SIM118 - safe_open is not a Mapping
                    if key in keys:
                        raise BackendContractError(
                            f"Wan tensor key {key!r} occurs in multiple shards"
                        )
                    keys.add(key)
        except BackendContractError:
            raise
        except Exception as exc:
            raise BackendContractError(
                f"cannot inspect Wan safetensors shard {shard}: {exc}"
            ) from exc
    return keys


def inspect_base_weights(model: Any, directory: str | Path) -> WeightLoadReport:
    """Compare the complete source inventory with the constructed model."""

    root = Path(directory)
    shards = _weight_shards(root)
    source = _source_keys(shards)
    target = set(model.state_dict())
    strip_prefix = not bool(source & target) and any(key.startswith("model.") for key in source)

    def normalize(key: str) -> str:
        return key.removeprefix("model.") if strip_prefix else key

    normalized = {normalize(key) for key in source}
    if len(normalized) != len(source):
        raise BackendContractError("Wan key normalization produced duplicate tensor names")
    missing = tuple(sorted(target - normalized))
    unexpected = tuple(sorted(normalized - target))
    disallowed = tuple(key for key in missing if not _allowed_base_missing_key(key))
    if unexpected or disallowed:
        raise BackendContractError(
            "Wan base checkpoint does not match the camera transformer: "
            f"disallowed_missing={len(disallowed)} unexpected={len(unexpected)} "
            f"first_disallowed_missing={list(disallowed[:8])} "
            f"first_unexpected={list(unexpected[:8])}"
        )
    return WeightLoadReport(
        shards=tuple(str(path) for path in shards),
        source_keys=len(source),
        target_keys=len(target),
        stripped_model_prefix=strip_prefix,
        missing_keys=missing,
        unexpected_keys=unexpected,
    )


def load_base_weights(model: Any, directory: str | Path) -> WeightLoadReport:
    """Load one shard at a time after a complete fail-closed inventory check."""

    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise BackendContractError("safetensors is required to load Wan weights") from exc

    report = inspect_base_weights(model, directory)
    for raw_path in report.shards:
        path = Path(raw_path)
        try:
            state = load_file(path, device="cpu")
        except Exception as exc:
            raise BackendContractError(f"cannot read Wan shard {path}: {exc}") from exc
        if report.stripped_model_prefix:
            state = {key.removeprefix("model."): value for key, value in state.items()}
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise BackendContractError(
                f"unexpected Wan keys while applying {path}: {result.unexpected_keys[:8]}"
            )
        del state
    if getattr(model, "uses_anyflow", False):
        model.initialize_anyflow_delta_from_time()
    return report


def build_camera_transformer_architecture(config: Mapping[str, Any]) -> Any:
    """Construct the configured camera transformer without loading weights.

    Stage2 has three complete role checkpoints. Loading the base
    shards before replacing every tensor would multiply startup I/O and
    transient memory by three, so that route intentionally uses this builder.
    Other stages continue through :func:`build_camera_transformer`.
    """

    try:
        from .modeling.causal_model import CausalWanModel
    except ImportError as exc:
        raise BackendContractError(f"cannot import the Wan transformer runtime: {exc}") from exc

    model_config = config.get("model", {})
    train = config.get("train", {})
    if not isinstance(model_config, Mapping) or not isinstance(train, Mapping):
        raise BackendContractError("model and train must be mappings")
    layout = WanAssetLayout.from_config(config)
    objective = str(train.get("objective", "flow_matching"))
    overrides: dict[str, Any] = {
        "local_attn_size": int(model_config["local_attn_size"]),
        "sink_size": int(model_config.get("sink_size", 0)),
        "attn_compress": 4 if bool(model_config.get("eprope", False)) else 1,
        "frame_seq_length": int(model_config["frame_sequence_length"]),
        "use_echorope": bool(model_config.get("use_echorope", True)),
        "camera_attention_mode": str(model_config["camera_attention_mode"]),
        "camera_translation_transform": str(
            model_config.get("camera_translation_transform", "linear")
        ),
        "flow_objective": objective,
    }
    optional = ("max_prior_clean_chunks", "rope_train_frames")
    for name in optional:
        if model_config.get(name) is not None:
            overrides[name] = int(model_config[name])
    if objective == "anyflow_forward_map":
        overrides["anyflow_gate"] = float(train["anyflow_gate"])
        overrides["anyflow_deltatime_type"] = str(train["anyflow_deltatime_type"])

    try:
        model = CausalWanModel.from_config(layout.transformer_config, **overrides)
    except Exception as exc:
        raise BackendContractError(
            f"cannot construct Wan transformer from {layout.transformer_config}: {exc}"
        ) from exc
    model.num_frame_per_block = int(model_config.get("num_frame_per_block", 3))
    return model


def build_camera_transformer(
    config: Mapping[str, Any],
) -> tuple[Any, WeightLoadReport]:
    """Construct the configured camera transformer and load exact base weights."""

    layout = WanAssetLayout.from_config(config)
    model = build_camera_transformer_architecture(config)
    report = load_base_weights(model, layout.transformer_weights)
    report = _bind_initialization_receipt(
        architecture=layout.transformer_config,
        report=report,
    )
    return model, report


__all__ = [
    "WeightLoadReport",
    "build_camera_transformer",
    "build_camera_transformer_architecture",
    "inspect_base_weights",
    "load_base_weights",
]
