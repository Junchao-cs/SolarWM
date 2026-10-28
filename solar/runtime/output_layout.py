"""Canonical output paths for training, validation, and standalone inference."""

from __future__ import annotations

import os
import re
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from solar.errors import BackendContractError

CAMERA_DATASET_TRIPLET_LAYOUT = "dataset_triplet_v1"
CAMERA_TRANSACTION_LAYOUT = "transaction_v1"
_PORTABLE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True)
class CameraInferenceOutputLayout:
    """Resolved standalone camera-inference publication and provenance roots."""

    publish_root: Path
    run_root: Path
    run_id: str
    layout: str


def portable_output_component(value: object, *, field: str) -> str:
    """Validate one collision-free, cross-platform output path component."""

    component = str(value or "")
    if not _PORTABLE_COMPONENT.fullmatch(component) or component in {".", ".."}:
        raise BackendContractError(
            f"{field} must be a portable ASCII path component, got {component!r}"
        )
    return component


def camera_inference_output_layout(
    config: Mapping[str, Any],
) -> CameraInferenceOutputLayout | None:
    """Resolve the opt-out dataset-triplet layout for camera-length inference.

    All non-camera routes return ``None`` and therefore preserve their existing
    output semantics. Camera-length inference may explicitly select
    ``transaction_v1`` to retain the original generation transaction layout.
    """

    inference = config.get("inference", {})
    runtime = config.get("runtime", {})
    if not isinstance(inference, Mapping) or not isinstance(runtime, Mapping):
        raise BackendContractError("inference and runtime must be mappings")
    camera_length = (
        str(config.get("action", "")).strip().lower() == "infer"
        and str(inference.get("length", "fixed")).strip().lower() == "camera"
    )
    if not camera_length:
        return None
    layout = str(inference.get("output_layout", CAMERA_DATASET_TRIPLET_LAYOUT)).strip().lower()
    if layout not in {CAMERA_DATASET_TRIPLET_LAYOUT, CAMERA_TRANSACTION_LAYOUT}:
        raise BackendContractError(
            "camera-length inference.output_layout must be "
            f"{CAMERA_DATASET_TRIPLET_LAYOUT} or {CAMERA_TRANSACTION_LAYOUT}"
        )
    if layout == CAMERA_TRANSACTION_LAYOUT:
        return None
    raw_root = str(runtime.get("output_dir", "")).strip()
    publish_root = Path(raw_root).expanduser()
    if not raw_root or not publish_root.is_absolute():
        raise BackendContractError(
            "camera-length dataset publication requires an absolute runtime.output_dir"
        )
    run_id = portable_output_component(
        inference.get("run_id", config.get("name")),
        field="inference.run_id",
    )
    publish_root = publish_root.resolve()
    return CameraInferenceOutputLayout(
        publish_root=publish_root,
        run_root=publish_root / "runs" / run_id,
        run_id=run_id,
        layout=layout,
    )


def invocation_output_dir(config: Mapping[str, Any]) -> Path:
    """Return the directory that owns one invocation's provenance files."""

    layout = camera_inference_output_layout(config)
    if layout is not None:
        return layout.run_root
    runtime = config.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise BackendContractError("runtime must be a mapping")
    raw_root = str(runtime.get("output_dir", "")).strip()
    if not raw_root:
        raise BackendContractError("runtime.output_dir is required")
    return Path(raw_root).expanduser()


def _run_root(output_dir: str | Path) -> Path:
    root = Path(output_dir).expanduser()
    if not root.is_absolute():
        raise BackendContractError("training output directory must be absolute")
    return root.resolve()


def checkpoint_model_dir(
    output_dir: str | Path,
    *,
    step: int,
    width: int,
) -> Path:
    """Return the stable top-level public checkpoint directory."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise BackendContractError("checkpoint step must be a positive integer")
    if width not in {6, 8}:
        raise BackendContractError("checkpoint step width must be 6 or 8")
    return _run_root(output_dir) / f"checkpoint_model_{step:0{width}d}"


def validation_pass_component(value: object) -> str:
    """Validate and return one validation-pass path component."""

    pass_name = str(value or "").strip()
    path = PurePosixPath(pass_name)
    if (
        not pass_name
        or pass_name in {".", ".."}
        or "/" in pass_name
        or "\\" in pass_name
        or path.parts != (pass_name,)
    ):
        raise BackendContractError("validation pass names must be portable path components")
    return pass_name


def public_validation_dir(
    output_dir: str | Path,
    *,
    step: int,
    pass_name: str,
) -> Path:
    """Return the one public validation directory for a step/pass pair."""

    if isinstance(step, bool) or not isinstance(step, int) or step < 0:
        raise BackendContractError("validation step must be a non-negative integer")
    pass_name = validation_pass_component(pass_name)
    return _run_root(output_dir) / "validation" / f"step_{step:06d}_{pass_name}"


def validation_staging_root(output_dir: str | Path) -> Path:
    """Private, non-publishable staging root for rank-local validation output."""

    return _run_root(output_dir) / "validation" / ".staging"


def cleanup_validation_staging(path: str | Path, *, output_dir: str | Path) -> None:
    """Remove one successful staging subtree, never anything outside ``.staging``.

    Callers invoke this only after their distributed output and completion gates
    pass.  Failed validations deliberately retain their staging tree for diagnosis.
    """

    staging_root = validation_staging_root(output_dir)
    target = Path(os.path.abspath(Path(path).expanduser()))
    if target == staging_root or staging_root not in target.parents:
        raise BackendContractError(
            f"refusing to clean validation path outside private staging: {target}"
        )

    current = target
    while current != staging_root:
        if current.is_symlink():
            raise BackendContractError(
                f"refusing to clean validation path containing a symlink: {target}"
            )
        current = current.parent
    if not target.exists():
        raise BackendContractError(f"validation staging path is missing: {target}")
    if not target.is_dir():
        raise BackendContractError(f"validation staging path is not a directory: {target}")
    shutil.rmtree(target)

    parent = target.parent
    while parent == staging_root or staging_root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        if parent == staging_root:
            break
        parent = parent.parent


__all__ = [
    "CAMERA_DATASET_TRIPLET_LAYOUT",
    "CAMERA_TRANSACTION_LAYOUT",
    "CameraInferenceOutputLayout",
    "camera_inference_output_layout",
    "checkpoint_model_dir",
    "cleanup_validation_staging",
    "invocation_output_dir",
    "portable_output_component",
    "public_validation_dir",
    "validation_pass_component",
    "validation_staging_root",
]
