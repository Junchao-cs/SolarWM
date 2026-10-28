"""Machine-readable preflight for the LTX-2.5 runtime."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from solar.checkpoint import verify_checkpoint
from solar.data import resolve_index_path
from solar.errors import BackendContractError
from solar.runtime.serialization import canonical_json_bytes

from .checkpoint import inspect_base_checkpoint
from .runtime import (
    ProviderCheck,
    ProviderResolution,
    resolve_runtime_provider,
    validate_ltx_checkpoint,
    validate_ltx_inference_checkpoint,
)


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str
    required: bool = True
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip() or self.status not in {"pass", "fail", "warning"}:
            raise BackendContractError("LTX readiness check is malformed")
        if not self.detail.strip():
            raise BackendContractError("LTX readiness check lacks detail")


@dataclass(frozen=True)
class ReadinessReport:
    action: str
    provider_entrypoint: str
    provider_identity: str
    checks: tuple[ReadinessCheck, ...]
    schema: str = "solarwm.ltx25.readiness.v1"

    @property
    def ready(self) -> bool:
        return all(not check.required or check.status == "pass" for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        failed = [check.name for check in self.checks if check.required and check.status != "pass"]
        return {
            "schema": self.schema,
            "family": "ltx25_video",
            "action": self.action,
            "status": "ready" if self.ready else "blocked",
            "ready": self.ready,
            "provider": {
                "entrypoint": self.provider_entrypoint,
                "identity": self.provider_identity,
            },
            "summary": {
                "checks": len(self.checks),
                "required_failures": len(failed),
                "failed_checks": failed,
            },
            "checks": [asdict(check) for check in self.checks],
        }

    def require_ready(self, *, report_path: Path | None = None) -> None:
        if self.ready:
            return
        failed = ", ".join(self.as_dict()["summary"]["failed_checks"])
        suffix = f"; report={report_path}" if report_path is not None else ""
        raise BackendContractError(f"LTX-2.5 {self.action} readiness failed: {failed}{suffix}")


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _dependency_check(name: str, *, required: bool = True) -> ReadinessCheck:
    available = _module_available(name)
    version = ""
    if available:
        package = name.replace("_", "-")
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return ReadinessCheck(
        name=f"dependency.{name}",
        status="pass" if available else "fail",
        detail=(f"importable version {version}" if available else "module is not importable"),
        required=required,
        evidence={"version": version or None},
    )


def _environment_checks(action: str, input_mode: str) -> list[ReadinessCheck]:
    checks = [
        _dependency_check("torch"),
        _dependency_check("ltx_core"),
        _dependency_check("safetensors"),
    ]
    if action == "train":
        checks.append(_dependency_check("peft"))
    if action == "preencode" or input_mode == "raw_online":
        checks.append(_dependency_check("av"))
    if _module_available("torch"):
        try:
            torch = importlib.import_module("torch")
            version = str(torch.__version__)
            cuda_version = str(torch.version.cuda)
            cuda_available = bool(torch.cuda.is_available())
            device_count = int(torch.cuda.device_count()) if cuda_available else 0
            cuda_build = cuda_version not in {"", "None"}
            checks.extend(
                (
                    ReadinessCheck(
                        "runtime.torch_cuda_build",
                        "pass" if cuda_build else "fail",
                        (
                            f"Torch {version} reports CUDA build {cuda_version}"
                            if cuda_build
                            else f"Torch {version} is not a CUDA build"
                        ),
                        evidence={"torch": version, "cuda": cuda_version},
                    ),
                    ReadinessCheck(
                        "runtime.cuda",
                        "pass" if cuda_available and device_count >= 1 else "fail",
                        (
                            f"CUDA available with {device_count} visible device(s)"
                            if cuda_available
                            else "CUDA is unavailable"
                        ),
                        evidence={"available": cuda_available, "devices": device_count},
                    ),
                )
            )
        except Exception as exc:  # environment probe must become data
            checks.append(
                ReadinessCheck(
                    "runtime.torch_probe",
                    "fail",
                    f"Torch probe failed: {type(exc).__name__}: {exc}",
                )
            )
    return checks


def _file_check(name: str, raw_path: Any) -> ReadinessCheck:
    value = str(raw_path or "").strip()
    if not value:
        return ReadinessCheck(name, "fail", "path is not configured")
    if "://" in value:
        return ReadinessCheck(
            name,
            "pass",
            "remote URI existence/identity is delegated to the provider",
            evidence={"path": value, "delegated": True},
        )
    path = Path(value).expanduser()
    ready = path.is_file() and not path.is_symlink()
    return ReadinessCheck(
        name,
        "pass" if ready else "fail",
        "regular non-symlink file exists" if ready else "file is missing or is a symlink",
        evidence={"path": str(path), "size": path.stat().st_size if ready else None},
    )


def _asset_checks(config: Mapping[str, Any], action: str, input_mode: str) -> list[ReadinessCheck]:
    model = config["model"]
    checks: list[ReadinessCheck] = []
    if not isinstance(model, Mapping):
        return [ReadinessCheck("assets.model", "fail", "model is not a mapping")]
    if action in {"train", "infer", "preencode"}:
        base_path = model.get("checkpoint_path")
        base = _file_check("assets.base_transformer", base_path)
        checks.append(base)
        if base.status == "pass" and "://" not in str(base_path):
            try:
                inspection = inspect_base_checkpoint(str(base_path))
            except Exception as exc:
                checks.append(
                    ReadinessCheck(
                        "assets.base_transformer_header",
                        "fail",
                        f"strict header inspection failed: {type(exc).__name__}: {exc}",
                    )
                )
            else:
                checks.append(
                    ReadinessCheck(
                        "assets.base_transformer_header",
                        "pass",
                        "all checkpoint keys were exhaustively classified",
                        evidence=inspection.as_dict(),
                    )
                )
    codec = model.get("codec", {})
    if not isinstance(codec, Mapping):
        codec = {}
    if action in {"train", "infer", "preencode"}:
        checks.append(_file_check("assets.video_vae", codec.get("video_vae_path")))
    if action == "preencode" or input_mode == "raw_online":
        checks.append(_file_check("assets.gemma4", codec.get("gemma4_path")))
    if action == "infer":
        adapter_path = model.get("adapter_checkpoint_path")
        path = Path(str(adapter_path or "")).expanduser()
        checkpoint_format = str(model.get("adapter_checkpoint_format", "transaction_v1")).lower()
        try:
            if checkpoint_format == "inference_transaction_v1":
                verified = verify_checkpoint(path)
                validate_ltx_inference_checkpoint(
                    verified,
                    config,
                    weights=str(model.get("adapter_weights") or ""),
                )
                evidence = {
                    "path": str(verified.path),
                    "manifest_digest": verified.manifest_digest,
                    "step": verified.step,
                    "format": "inference_transaction_v1",
                }
                detail = "optimizer-free inference checkpoint is complete and verified"
            else:
                verified = verify_checkpoint(path)
                validate_ltx_checkpoint(verified, config)
                evidence = {
                    "path": str(verified.path),
                    "manifest_digest": verified.manifest_digest,
                    "step": verified.step,
                    "format": "transaction_v1",
                }
                detail = "shared checkpoint is complete and resume-compatible"
        except Exception as exc:
            checks.append(
                ReadinessCheck(
                    "assets.adapter_checkpoint",
                    "fail",
                    f"shared checkpoint verification failed: {type(exc).__name__}: {exc}",
                    evidence={"path": str(path)},
                )
            )
        else:
            checks.append(
                ReadinessCheck(
                    "assets.adapter_checkpoint",
                    "pass",
                    detail,
                    evidence=evidence,
                )
            )
    if action == "train":
        checkpoint = config.get("checkpoint", {})
        resume_from = (
            str(checkpoint.get("resume_from") or "").strip()
            if isinstance(checkpoint, Mapping)
            else ""
        )
        if resume_from:
            path = Path(resume_from).expanduser()
            try:
                verified = verify_checkpoint(path)
                validate_ltx_checkpoint(verified, config)
            except Exception as exc:
                checks.append(
                    ReadinessCheck(
                        "assets.resume_checkpoint",
                        "fail",
                        f"shared resume checkpoint structure failed: {type(exc).__name__}: {exc}",
                        evidence={"path": str(path)},
                    )
                )
            else:
                checks.append(
                    ReadinessCheck(
                        "assets.resume_checkpoint",
                        "pass",
                        "shared resume checkpoint structure and manifest are compatible",
                        evidence={
                            "path": str(verified.path),
                            "manifest_digest": verified.manifest_digest,
                            "step": verified.step,
                        },
                    )
                )
    data = config.get("data", {})
    if isinstance(data, Mapping):
        controls: list[tuple[str, str]] = [("index", "controls.data_index")]
        if action == "train":
            controls.append(("test_index", "controls.test_index"))
        if input_mode == "preencoded":
            controls.append(("completion_marker", "controls.completion_marker"))
        for field, name in controls:
            try:
                control_path = resolve_index_path(data, field)
            except Exception as exc:
                checks.append(
                    ReadinessCheck(
                        name,
                        "fail",
                        f"control path resolution failed: {type(exc).__name__}: {exc}",
                    )
                )
            else:
                ready = control_path.is_file() and not control_path.is_symlink()
                checks.append(
                    ReadinessCheck(
                        name,
                        "pass" if ready else "fail",
                        (
                            "staged local control exists"
                            if ready
                            else "staged local control is missing or a symlink"
                        ),
                        evidence={"path": str(control_path)},
                    )
                )
    inference = config.get("inference" if action == "infer" else "validation", {})
    if action in {"train", "infer"} and isinstance(inference, Mapping):
        if action == "train":
            inference = inference.get("inference", {})
        if isinstance(inference, Mapping):
            checks.append(
                _file_check(
                    "assets.negative_caption_cache",
                    inference.get("negative_caption_cache"),
                )
            )
    return checks


def _provider_checks(
    resolution: ProviderResolution,
    config: Mapping[str, Any],
    action: str,
) -> list[ReadinessCheck]:
    if resolution.provider is None:
        return [
            ReadinessCheck(
                "provider.discovery",
                "fail",
                resolution.error or "no provider was discovered",
                evidence={"entrypoint": resolution.entrypoint},
            )
        ]
    provider = resolution.provider
    checks = [
        ReadinessCheck(
            "provider.discovery",
            "pass",
            "runtime provider loaded with the configured contract",
            evidence={
                "entrypoint": resolution.entrypoint,
                "identity": provider.identity,
                "api_version": provider.api_version,
            },
        )
    ]
    try:
        provided: Sequence[ProviderCheck] = provider.readiness(config, action)
        for item in provided:
            if not isinstance(item, ProviderCheck):
                raise BackendContractError("provider returned a non-ProviderCheck value")
            checks.append(ReadinessCheck(**asdict(item)))
    except Exception as exc:
        checks.append(
            ReadinessCheck(
                "provider.readiness",
                "fail",
                f"provider readiness failed: {type(exc).__name__}: {exc}",
            )
        )
    return checks


def probe_ltx25_runtime(
    config: Mapping[str, Any],
    *,
    resolution: ProviderResolution | None = None,
    enforce_environment: bool = True,
) -> ReadinessReport:
    action = str(config.get("action", "")).strip().lower()
    data = config.get("data", {})
    input_mode = str(data.get("input_mode", "")) if isinstance(data, Mapping) else ""
    selected = resolution or resolve_runtime_provider(config)
    checks: list[ReadinessCheck] = []
    if enforce_environment:
        checks.extend(_environment_checks(action, input_mode))
    checks.extend(_asset_checks(config, action, input_mode))
    checks.extend(_provider_checks(selected, config, action))
    identity = selected.provider.identity if selected.provider is not None else ""
    return ReadinessReport(action, selected.entrypoint, identity, tuple(checks))


def write_readiness_report(path: str | Path, report: ReadinessReport) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(report.as_dict()))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


__all__ = [
    "ReadinessCheck",
    "ReadinessReport",
    "probe_ltx25_runtime",
    "write_readiness_report",
]
