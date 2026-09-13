"""Atomic, fixed-plan generation used by both inference and validation."""

from __future__ import annotations

import hashlib
import stat
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from solarwm.errors import BackendContractError
from solarwm.runtime.create_only import (
    link_file_create_only,
    publish_directory_no_replace,
    write_file_create_only,
)
from solarwm.runtime.serialization import canonical_json_bytes

_RESERVED_SAMPLE_PATHS = frozenset({"manifest.json"})


def _canonical_artifact_path(name: object) -> PurePosixPath:
    if not isinstance(name, str):
        raise BackendContractError(f"invalid generated artifact {name!r}")
    raw_parts = name.split("/")
    if (
        not name
        or "\\" in name
        or "\x00" in name
        or any(part in {"", ".", ".."} for part in raw_parts)
    ):
        raise BackendContractError(f"invalid generated artifact {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or path.as_posix() != name:
        raise BackendContractError(f"invalid generated artifact {name!r}")
    if path.parts[0] in _RESERVED_SAMPLE_PATHS:
        raise BackendContractError(f"generated artifact path is reserved: {name!r}")
    return path


@dataclass(frozen=True)
class InferenceCase:
    slot: int
    sample_id: str
    prompt: str
    start_frame: int
    noise_seed: int
    camera_fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.slot < 0 or self.start_frame < 0 or self.noise_seed < 0:
            raise BackendContractError("inference slot/start/noise values must be non-negative")
        if not self.sample_id or not self.camera_fingerprint:
            raise BackendContractError("inference case lacks sample/camera identity")


@dataclass(frozen=True)
class GeneratedFile:
    """A generated artifact already materialized on the output filesystem.

    Large videos use this path-backed form so the inference engine never has to
    hold the complete encoded artifact in memory. The producer must place the
    file on the same filesystem as ``runtime.output_dir``; the engine publishes
    it with a create-only hard link and then removes the private source name.
    """

    path: Path
    delete_after_publish: bool = True

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser()
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise BackendContractError(f"generated file is not readable: {path}: {exc}") from exc
        if not stat.S_ISREG(mode) or path.is_symlink():
            raise BackendContractError(f"generated file must be a regular non-symlink file: {path}")
        object.__setattr__(self, "path", path.resolve())


@dataclass(frozen=True)
class GeneratedSample:
    artifacts: Mapping[str, bytes | GeneratedFile]
    shape: tuple[int, ...]
    dtype: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.artifacts or not self.shape or not self.dtype:
            raise BackendContractError("generated sample lacks artifacts/shape/dtype")
        paths: list[PurePosixPath] = []
        for name, value in self.artifacts.items():
            path = _canonical_artifact_path(name)
            if not isinstance(value, (bytes, GeneratedFile)):
                raise BackendContractError(f"invalid generated artifact {name!r}")
            if any(
                path == other or path in other.parents or other in path.parents for other in paths
            ):
                raise BackendContractError(f"generated artifact paths collide: {name!r}")
            paths.append(path)


class InferenceAdapter(Protocol):
    family: str

    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample: ...


@dataclass(frozen=True)
class InferenceSummary:
    output_dir: Path
    weights_id: str
    cases: int
    ordered_manifest_digest: str


def _write_file(path: Path, value: bytes) -> None:
    write_file_create_only(
        path,
        value,
        error_type=BackendContractError,
        label="inference output file",
    )


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.blake2s()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise BackendContractError(f"cannot read generated file {path}: {exc}") from exc
    return size, digest.hexdigest()


class InferenceEngine:
    """One implementation for standalone generation and training validation."""

    def __init__(self, adapter: InferenceAdapter) -> None:
        self.adapter = adapter

    def run(
        self,
        cases: Sequence[InferenceCase],
        *,
        weights_id: str,
        output_dir: str | Path,
        collective_error: Callable[[str | None, str], None] | None = None,
        collective_case_waves: int | None = None,
    ) -> InferenceSummary:
        if not weights_id:
            raise BackendContractError("inference requires a stable weights_id")
        if not cases and collective_case_waves is None:
            raise BackendContractError("inference requires cases")
        slots = [case.slot for case in cases]
        if len(slots) != len(set(slots)):
            raise BackendContractError("inference case slots are duplicated")
        waves = len(cases) if collective_case_waves is None else int(collective_case_waves)
        if waves < 1:
            raise BackendContractError("inference collective case waves must be positive")
        if waves < len(cases):
            raise BackendContractError(
                "inference collective case waves are shorter than local cases"
            )
        target = Path(output_dir).resolve()
        staging: Path | None = None
        setup_error: str | None = None
        try:
            if target.exists():
                raise BackendContractError(f"inference output already exists: {target}")
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
            staging.mkdir()
        except Exception as exc:
            setup_error = f"{type(exc).__name__}: {exc}"
        if collective_error is not None:
            collective_error(setup_error, "output setup")
        if setup_error is not None or staging is None:
            raise BackendContractError(f"inference output setup failed: {setup_error}")
        manifests: list[dict[str, Any]] = []
        for case_index in range(waves):
            case_error: str | None = None
            manifest: dict[str, Any] | None = None
            if case_index < len(cases):
                case = cases[case_index]
                generated: GeneratedSample | None = None
                try:
                    generated = self.adapter.generate(case, weights_id=weights_id)
                    sample_dir = staging / f"slot-{case.slot:06d}"
                    sample_dir.mkdir()
                    artifact_records: list[dict[str, Any]] = []
                    for name, value in sorted(generated.artifacts.items()):
                        artifact = sample_dir / name
                        if isinstance(value, GeneratedFile):
                            size, digest = _file_identity(value.path)
                            link_file_create_only(
                                value.path,
                                artifact,
                                error_type=BackendContractError,
                                label="inference generated file",
                            )
                        else:
                            _write_file(artifact, value)
                            size = len(value)
                            digest = hashlib.blake2s(value).hexdigest()
                        artifact_records.append(
                            {
                                "path": f"slot-{case.slot:06d}/{name}",
                                "bytes": size,
                                "digest": digest,
                            }
                        )
                    manifest = {
                        "schema": "solarwm.inference-sample.v1",
                        "family": self.adapter.family,
                        "weights_id": weights_id,
                        "case": asdict(case),
                        "shape": list(generated.shape),
                        "dtype": generated.dtype,
                        "metrics": dict(generated.metrics),
                        "provenance": dict(generated.provenance),
                        "artifacts": artifact_records,
                    }
                    _write_file(
                        sample_dir / "manifest.json",
                        canonical_json_bytes(manifest),
                    )
                except Exception as exc:
                    case_error = f"{type(exc).__name__}: {exc}"
                finally:
                    if generated is not None:
                        for value in generated.artifacts.values():
                            if isinstance(value, GeneratedFile) and value.delete_after_publish:
                                value.path.unlink(missing_ok=True)
            if collective_error is not None:
                collective_error(case_error, f"case wave {case_index}")
            if case_error is not None:
                raise BackendContractError(f"inference case wave {case_index} failed: {case_error}")
            if case_index < len(cases):
                if manifest is None:
                    raise BackendContractError(
                        f"inference case wave {case_index} returned no manifest"
                    )
                manifests.append(manifest)

        ordered_digest = ""
        commit_error: str | None = None
        try:
            ordered_bytes = b"".join(canonical_json_bytes(item) for item in manifests)
            ordered_digest = hashlib.blake2s(ordered_bytes).hexdigest()
            _write_file(staging / "ordered-manifest.jsonl", ordered_bytes)
            _write_file(
                staging / "COMPLETE.json",
                canonical_json_bytes(
                    {
                        "schema": "solarwm.inference-complete.v1",
                        "family": self.adapter.family,
                        "weights_id": weights_id,
                        "cases": len(cases),
                        "ordered_manifest_digest": ordered_digest,
                    }
                ),
            )
            publish_directory_no_replace(
                staging,
                target,
                error_type=BackendContractError,
                label="inference output",
            )
        except Exception as exc:
            commit_error = f"{type(exc).__name__}: {exc}"
        if collective_error is not None:
            collective_error(commit_error, "partition commit")
        if commit_error is not None:
            raise BackendContractError(f"inference partition commit failed: {commit_error}")
        return InferenceSummary(target, weights_id, len(cases), ordered_digest)


def run_validation(
    adapter: InferenceAdapter,
    cases: Sequence[InferenceCase],
    *,
    weights_id: str,
    output_dir: str | Path,
) -> InferenceSummary:
    """Training validation is deliberately only a named call to inference."""

    return InferenceEngine(adapter).run(cases, weights_id=weights_id, output_dir=output_dir)
