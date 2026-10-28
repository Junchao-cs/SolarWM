"""Dependency, asset, index, CUDA, and topology readiness probes."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Literal

from solar.data.index import inventory, read_index, resolve_index_path
from solar.errors import BackendContractError

from .assets import WanAssetLayout, is_placeholder_path

Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class ReadinessIssue:
    code: str
    message: str
    severity: Severity = "error"
    path: str | None = None


@dataclass(frozen=True)
class ReadinessReport:
    family: str
    ready: bool
    assets: Mapping[str, str]
    dependencies: Mapping[str, str | None]
    weight_inventory: Mapping[str, Any]
    index_inventories: Mapping[str, Mapping[str, Any]]
    issues: tuple[ReadinessIssue, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "solarwm.wan22-readiness.v1",
            "family": self.family,
            "ready": self.ready,
            "assets": dict(self.assets),
            "dependencies": dict(self.dependencies),
            "weight_inventory": dict(self.weight_inventory),
            "index_inventories": {
                key: dict(value) for key, value in self.index_inventories.items()
            },
            "issues": [asdict(issue) for issue in self.issues],
        }

    def require_ready(self) -> None:
        errors = [issue for issue in self.issues if issue.severity == "error"]
        if errors:
            rendered = "; ".join(
                f"{issue.code}: {issue.message}" + (f" [{issue.path}]" if issue.path else "")
                for issue in errors
            )
            raise BackendContractError(f"Wan runtime is not ready: {rendered}")


def _dependency_versions(
    *, online: bool, transformer: bool = True
) -> tuple[dict[str, str | None], list[ReadinessIssue]]:
    required = [
        ("torch", "torch"),
        ("safetensors", "safetensors"),
        ("einops", "einops"),
        ("transformers", "transformers"),
        ("ftfy", "ftfy"),
        ("regex", "regex"),
    ]
    if transformer:
        required.append(("diffusers", "diffusers"))
    if online:
        required.append(("decord", "decord"))
    versions: dict[str, str | None] = {}
    issues: list[ReadinessIssue] = []
    for module_name, distribution in required:
        available = importlib.util.find_spec(module_name) is not None
        try:
            version = metadata.version(distribution) if available else None
        except metadata.PackageNotFoundError:
            version = "unknown" if available else None
        versions[module_name] = version
        if not available:
            issues.append(
                ReadinessIssue(
                    code="dependency_missing",
                    message=f"Python module {module_name!r} is not installed",
                )
            )
    if transformer:
        flash_available = any(
            importlib.util.find_spec(name) is not None
            for name in ("flash_attn", "flash_attn_interface")
        )
        versions["flash_attention"] = "available" if flash_available else None
        if not flash_available:
            issues.append(
                ReadinessIssue(
                    code="flash_attention_missing",
                    message=(
                        "FlashAttention is required for GPU training; "
                        "the PyTorch attention fallback is for diagnostics only"
                    ),
                )
            )
    return versions, issues


def _check_file(
    path: Path,
    *,
    label: str,
    issues: list[ReadinessIssue],
) -> None:
    if is_placeholder_path(path):
        issues.append(
            ReadinessIssue(
                code="placeholder_path",
                message=f"{label} still uses a documentation placeholder",
                path=str(path),
            )
        )
    elif not path.is_file():
        issues.append(
            ReadinessIssue(
                code="asset_missing",
                message=f"{label} is not a readable file",
                path=str(path),
            )
        )


def _check_assets(
    layout: WanAssetLayout,
    *,
    online: bool,
    transformer: bool = True,
    transformer_weights: bool = True,
    issues: list[ReadinessIssue],
) -> None:
    if transformer:
        _check_file(
            layout.transformer_config,
            label="transformer config",
            issues=issues,
        )
    if transformer_weights:
        weights = layout.transformer_weights
        if is_placeholder_path(weights):
            issues.append(
                ReadinessIssue(
                    code="placeholder_path",
                    message="transformer weights still use a documentation placeholder",
                    path=str(weights),
                )
            )
        elif not weights.is_dir():
            issues.append(
                ReadinessIssue(
                    code="asset_missing",
                    message="transformer weight directory is missing",
                    path=str(weights),
                )
            )
        elif not tuple(sorted(weights.glob("*.safetensors"))):
            issues.append(
                ReadinessIssue(
                    code="weights_missing",
                    message="transformer directory contains no .safetensors shards",
                    path=str(weights),
                )
            )
    if online:
        _check_file(layout.text_encoder, label="UMT5 encoder", issues=issues)
        _check_file(layout.vae, label="Wan VAE", issues=issues)
        if is_placeholder_path(layout.tokenizer):
            issues.append(
                ReadinessIssue(
                    code="placeholder_path",
                    message="tokenizer still uses a documentation placeholder",
                    path=str(layout.tokenizer),
                )
            )
        elif not layout.tokenizer.is_dir():
            issues.append(
                ReadinessIssue(
                    code="asset_missing",
                    message="UMT5 tokenizer directory is missing",
                    path=str(layout.tokenizer),
                )
            )
    negative = layout.anyflow_negative_embedding
    if negative is not None:
        _check_file(negative, label="AnyFlow negative embedding", issues=issues)


def _check_transformer_config(
    path: Path,
    *,
    family: str,
    issues: list[ReadinessIssue],
) -> None:
    if not path.is_file():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(
            ReadinessIssue(
                code="transformer_config_invalid",
                message=f"transformer config is not valid UTF-8 JSON: {exc}",
                path=str(path),
            )
        )
        return
    expected = (
        {"model_type": "ti2v", "in_dim": 48, "out_dim": 48, "dim": 3072}
        if family == "wan22_ti2v_5b"
        else {"model_type": "i2v", "in_dim": 36, "out_dim": 16, "dim": 5120}
    )
    drift = {
        key: {"actual": payload.get(key), "expected": value}
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if drift:
        issues.append(
            ReadinessIssue(
                code="transformer_config_mismatch",
                message=f"transformer architecture differs from {family}: {drift}",
                path=str(path),
            )
        )


def _probe_weights(
    directory: Path,
    issues: list[ReadinessIssue],
) -> Mapping[str, Any]:
    if is_placeholder_path(directory) or not directory.is_dir():
        return {}
    shards = tuple(sorted(directory.glob("*.safetensors")))
    if not shards or importlib.util.find_spec("safetensors") is None:
        return {}
    try:
        from safetensors import safe_open

        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for shard in shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                keys = tuple(handle.keys())
            duplicate = sorted(seen.intersection(keys))
            if duplicate:
                raise BackendContractError(
                    f"tensor keys occur in multiple Wan shards: {duplicate[:8]}"
                )
            seen.update(keys)
            rows.append(
                {
                    "name": shard.name,
                    "bytes": shard.stat().st_size,
                    "tensor_keys": len(keys),
                }
            )
        return {
            "shards": rows,
            "total_bytes": sum(int(row["bytes"]) for row in rows),
            "tensor_keys": len(seen),
            "inspection": "safetensors-headers",
        }
    except Exception as exc:
        issues.append(
            ReadinessIssue(
                code="weights_invalid",
                message=f"cannot inventory Wan safetensors: {exc}",
                path=str(directory),
            )
        )
        return {}


def _index_paths(
    config: Mapping[str, Any],
) -> tuple[dict[str, Path], list[ReadinessIssue]]:
    data = config.get("data", {})
    if not isinstance(data, Mapping):
        return {}, [ReadinessIssue("data_invalid", "data must be a mapping")]
    action = str(config.get("action", ""))
    if action == "preencode":
        names = ("index",)
    elif action == "infer":
        names = ("test_index",)
    else:
        names = ("train_index", "test_index")
    paths: dict[str, Path] = {}
    issues: list[ReadinessIssue] = []
    for name in names:
        try:
            paths[name] = resolve_index_path(data, name)
        except Exception as exc:
            issues.append(
                ReadinessIssue(
                    code="index_resolution_failed",
                    message=f"cannot resolve {name}: {exc}",
                )
            )
    return paths, issues


def _probe_indexes(
    config: Mapping[str, Any],
    issues: list[ReadinessIssue],
    *,
    validate_contents: bool = True,
) -> dict[str, Mapping[str, Any]]:
    paths, path_issues = _index_paths(config)
    issues.extend(path_issues)
    inventories: dict[str, Mapping[str, Any]] = {}
    for name, path in paths.items():
        if is_placeholder_path(path) or not path.is_file():
            issues.append(
                ReadinessIssue(
                    code="index_missing",
                    message=f"{name} is not a readable local index",
                    path=str(path),
                )
            )
            continue
        if not validate_contents:
            inventories[name] = {
                "bytes": path.stat().st_size,
                "validation": "reader",
            }
            continue
        try:
            rows = read_index(path)
            inventories[name] = asdict(inventory(path, rows))
        except Exception as exc:
            issues.append(
                ReadinessIssue(
                    code="index_invalid",
                    message=f"{name} failed canonical index validation: {exc}",
                    path=str(path),
                )
            )
    return inventories


def _probe_inference_controls(config: Mapping[str, Any], issues: list[ReadinessIssue]) -> None:
    if str(config.get("action", "")) != "infer":
        return
    checkpoint = config.get("checkpoint", {})
    raw_path = checkpoint.get("path", "") if isinstance(checkpoint, Mapping) else ""
    source = Path(str(raw_path)).expanduser().resolve()
    if is_placeholder_path(source) or not (source.is_file() or source.is_dir()):
        issues.append(
            ReadinessIssue(
                code="checkpoint_missing",
                message="inference checkpoint is not a readable file or transaction",
                path=str(source),
            )
        )
    elif source.is_dir() or (source.parent / "COMPLETE.json").is_file():
        transaction = source if source.is_dir() else source.parent
        try:
            from solar.checkpoint import verify_checkpoint

            verify_checkpoint(transaction)
        except Exception as exc:
            issues.append(
                ReadinessIssue(
                    code="checkpoint_invalid",
                    message=f"inference checkpoint transaction is invalid: {exc}",
                    path=str(transaction),
                )
            )
    if shutil.which("ffmpeg") is None:
        issues.append(
            ReadinessIssue(
                code="ffmpeg_missing",
                message="ffmpeg is required to encode Wan inference MP4 artifacts",
            )
        )


def _probe_cuda(issues: list[ReadinessIssue], *, required: bool) -> None:
    if not required or importlib.util.find_spec("torch") is None:
        return
    try:
        import torch

        if not torch.cuda.is_available():
            issues.append(ReadinessIssue("cuda_unavailable", "CUDA is unavailable"))
            return
        device_count = torch.cuda.device_count()
        local_world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        if device_count < min(local_world, 8):
            issues.append(
                ReadinessIssue(
                    "cuda_device_count",
                    f"visible CUDA devices={device_count}, LOCAL_WORLD_SIZE={local_world}",
                )
            )
    except Exception as exc:
        issues.append(ReadinessIssue("cuda_probe_failed", f"CUDA probe failed: {exc}"))


def probe_runtime(
    config: Mapping[str, Any],
    *,
    family: str,
    require_cuda: bool = False,
    require_transformer_weights: bool = True,
    validate_index_contents: bool = True,
) -> ReadinessReport:
    """Return every actionable runtime issue in one deterministic report."""

    data = config.get("data", {})
    action = str(config.get("action", ""))
    online = bool(
        action in {"infer", "preencode"}
        or (isinstance(data, Mapping) and str(data.get("encoding", "online")) == "online")
        or (
            action == "train"
            and isinstance(config.get("validation"), Mapping)
            and bool(config["validation"].get("passes"))
        )
    )
    transformer = str(config.get("action", "")) != "preencode"
    layout = WanAssetLayout.from_config(config)
    dependencies, dependency_issues = _dependency_versions(online=online, transformer=transformer)
    issues = list(dependency_issues)
    _check_assets(
        layout,
        online=online,
        transformer=transformer,
        transformer_weights=transformer and bool(require_transformer_weights),
        issues=issues,
    )
    if transformer:
        _check_transformer_config(
            layout.transformer_config,
            family=family,
            issues=issues,
        )
        weight_inventory = (
            _probe_weights(layout.transformer_weights, issues)
            if require_transformer_weights
            else {"source": "complete-role-checkpoints"}
        )
    else:
        weight_inventory = {}
    inventories = _probe_indexes(
        config,
        issues,
        validate_contents=validate_index_contents,
    )
    _probe_inference_controls(config, issues)
    _probe_cuda(issues, required=require_cuda)
    return ReadinessReport(
        family=family,
        ready=not any(issue.severity == "error" for issue in issues),
        assets=layout.as_dict(),
        dependencies=dependencies,
        weight_inventory=weight_inventory,
        index_inventories=inventories,
        issues=tuple(issues),
    )


def require_training_runtime(
    config: Mapping[str, Any],
    *,
    family: str,
    require_transformer_weights: bool = True,
) -> ReadinessReport:
    """Require the normal Wan training runtime and all configured assets."""

    checkpoint = config.get("checkpoint", {})
    full_resume = isinstance(checkpoint, Mapping) and str(checkpoint.get("mode")) == "full_resume"
    report = probe_runtime(
        config,
        family=family,
        require_cuda=True,
        require_transformer_weights=require_transformer_weights and not full_resume,
        validate_index_contents=False,
    )
    report.require_ready()
    return report


__all__ = [
    "ReadinessIssue",
    "ReadinessReport",
    "probe_runtime",
    "require_training_runtime",
]
