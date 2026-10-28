"""Local checkpoint commit protocol with explicit component inventories."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from solar.errors import CheckpointError
from solar.runtime.serialization import canonical_json_bytes

_MANIFEST = "checkpoint-manifest.json"
_COMPLETE = "COMPLETE.json"


@dataclass(frozen=True)
class CheckpointContract:
    """Algorithm state that must match for an exact full resume."""

    family: str
    stage: str
    causal_mode: str
    objective: str
    objective_variant: str
    camera_translation_transform: str
    parameterization: str
    sp_size: int
    data_generation: str
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "family",
            "stage",
            "causal_mode",
            "objective",
            "camera_translation_transform",
            "parameterization",
            "data_generation",
        ):
            if not str(getattr(self, name)).strip():
                raise CheckpointError(f"checkpoint contract {name} must be non-empty")
        if self.sp_size < 1:
            raise CheckpointError("checkpoint contract sp_size must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointContract:
        try:
            return cls(
                family=str(value["family"]),
                stage=str(value["stage"]),
                causal_mode=str(value["causal_mode"]),
                objective=str(value["objective"]),
                objective_variant=str(value.get("objective_variant", "")),
                camera_translation_transform=str(value["camera_translation_transform"]),
                parameterization=str(value["parameterization"]),
                sp_size=int(value["sp_size"]),
                data_generation=str(value["data_generation"]),
                extras=dict(value.get("extras", {})),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointError("invalid checkpoint contract") from exc


@dataclass(frozen=True)
class ComponentFile:
    path: str
    size: int

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise CheckpointError("checkpoint component path must be non-empty")
        if type(self.size) is not int or self.size < 1:
            raise CheckpointError(f"checkpoint component size must be positive: {self.path!r}")


@dataclass(frozen=True)
class VerifiedCheckpoint:
    path: Path
    step: int
    contract: CheckpointContract
    files: tuple[ComponentFile, ...]
    metadata: Mapping[str, Any]
    manifest_digest: str


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _component_files(root: Path, required: Sequence[str]) -> tuple[ComponentFile, ...]:
    required_set = set(required)
    if not required_set or "" in required_set:
        raise CheckpointError("required checkpoint components must be non-empty")
    if len(required_set) != len(required):
        raise CheckpointError("required checkpoint components contain duplicates")
    for component in sorted(required_set):
        if "/" in component or component in {".", "..", _MANIFEST, _COMPLETE}:
            raise CheckpointError(f"invalid checkpoint component {component!r}")
        target = root / component
        if not target.exists():
            raise CheckpointError(f"checkpoint component is missing: {component}")
        if target.is_symlink():
            raise CheckpointError(f"checkpoint component may not be a symlink: {component}")

    records: list[ComponentFile] = []
    for component in sorted(required_set):
        target = root / component
        candidates = (target,) if target.is_file() else tuple(sorted(target.rglob("*")))
        for candidate in candidates:
            if candidate.is_symlink():
                raise CheckpointError(
                    f"checkpoint contents may not be symlinks: {candidate.relative_to(root)}"
                )
            if not candidate.is_file():
                continue
            records.append(
                ComponentFile(
                    path=candidate.relative_to(root).as_posix(),
                    size=candidate.stat().st_size,
                )
            )
    if not records:
        raise CheckpointError("checkpoint contains no component files")
    return tuple(records)


class CheckpointTransaction:
    """Build a checkpoint privately and publish it with one atomic rename.

    Version 2 manifests record the exact component inventory and byte sizes.
    They deliberately avoid rereading large payloads to compute content digests.
    """

    def __init__(self, target: str | Path) -> None:
        self.target = Path(target).resolve()
        self.path = self.target.with_name(f".{self.target.name}.{uuid.uuid4().hex}.partial")
        self._committed = False

    def __enter__(self) -> CheckpointTransaction:
        if self.target.exists():
            raise CheckpointError(f"checkpoint target already exists: {self.target}")
        self.target.parent.mkdir(parents=True, exist_ok=True)
        self.path.mkdir(mode=0o755)
        return self

    def commit(
        self,
        *,
        step: int,
        contract: CheckpointContract,
        required_components: Sequence[str],
        metadata: Mapping[str, Any],
    ) -> VerifiedCheckpoint:
        if self._committed:
            raise CheckpointError("checkpoint transaction was already committed")
        if step < 0:
            raise CheckpointError("checkpoint step must be non-negative")
        files = _component_files(self.path, required_components)
        manifest = {
            "schema": "solarwm.checkpoint.v2",
            "step": int(step),
            "contract": contract.as_dict(),
            "required_components": sorted(required_components),
            "files": [{"path": record.path, "size": record.size} for record in files],
            "metadata": dict(metadata),
        }
        manifest_bytes = canonical_json_bytes(manifest)
        manifest_digest = hashlib.blake2s(manifest_bytes).hexdigest()
        _atomic_write(self.path / _MANIFEST, manifest_bytes)
        _atomic_write(
            self.path / _COMPLETE,
            canonical_json_bytes(
                {
                    "schema": "solarwm.checkpoint-complete.v2",
                    "step": int(step),
                    "manifest_digest": manifest_digest,
                }
            ),
        )
        if self.target.exists():
            raise CheckpointError(f"checkpoint target appeared during commit: {self.target}")
        os.replace(self.path, self.target)
        directory_fd = os.open(self.target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        self._committed = True
        return verify_checkpoint(self.target)

    def __exit__(self, *_: object) -> None:
        # An incomplete directory is intentionally retained for diagnosis. It
        # has a .partial suffix and is never discoverable as a valid checkpoint.
        return None


def verify_checkpoint(path: str | Path) -> VerifiedCheckpoint:
    """Validate a v2 checkpoint's small metadata, inventory, paths, and sizes."""
    root = Path(path).resolve()
    complete_path = root / _COMPLETE
    manifest_path = root / _MANIFEST
    if not complete_path.is_file() or not manifest_path.is_file():
        raise CheckpointError(f"checkpoint is not complete: {root}")
    try:
        complete = json.loads(complete_path.read_bytes())
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"checkpoint metadata is invalid: {root}") from exc
    if (
        complete.get("schema") != "solarwm.checkpoint-complete.v2"
        or manifest.get("schema") != "solarwm.checkpoint.v2"
    ):
        raise CheckpointError("unknown or mismatched checkpoint schemas")
    manifest_digest = hashlib.blake2s(manifest_bytes).hexdigest()
    if complete.get("manifest_digest") != manifest_digest:
        raise CheckpointError("checkpoint manifest identity differs from COMPLETE")
    if complete.get("step") != manifest.get("step"):
        raise CheckpointError("checkpoint step differs between manifest and COMPLETE")

    try:
        files = tuple(ComponentFile(**value) for value in manifest["files"])
        contract = CheckpointContract.from_dict(manifest["contract"])
        metadata = dict(manifest["metadata"])
        step = int(manifest["step"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CheckpointError("checkpoint manifest fields are invalid") from exc
    if not files:
        raise CheckpointError("checkpoint manifest has no files")
    if len({record.path for record in files}) != len(files):
        raise CheckpointError("checkpoint manifest has duplicate file paths")
    for record in files:
        relative = Path(record.path)
        if relative.is_absolute() or ".." in relative.parts:
            raise CheckpointError(f"checkpoint manifest has unsafe path {record.path!r}")
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise CheckpointError(f"checkpoint file is missing: {record.path}")
        if target.stat().st_size != record.size:
            raise CheckpointError(f"checkpoint file size differs: {record.path}")
    return VerifiedCheckpoint(root, step, contract, files, metadata, manifest_digest)


def assert_resume_compatible(expected: CheckpointContract, actual: CheckpointContract) -> None:
    """Reject full resume unless every algorithm-bearing field is identical."""

    if expected.as_dict() == actual.as_dict():
        return
    differences = {
        key: {"expected": expected.as_dict()[key], "actual": actual.as_dict()[key]}
        for key in expected.as_dict()
        if expected.as_dict()[key] != actual.as_dict()[key]
    }
    raise CheckpointError(f"exact-resume contract mismatch: {differences}")
