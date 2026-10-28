"""Create-only dataset-triplet publication for standalone camera inference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from solar.errors import BackendContractError
from solar.inference import InferenceCase
from solar.runtime.create_only import link_file_create_only, write_file_create_only
from solar.runtime.output_layout import (
    CameraInferenceOutputLayout,
    portable_output_component,
)
from solar.runtime.serialization import canonical_json_bytes

_REQUIRED_ARTIFACTS = {
    "video.mp4": "generate",
    "compare.mp4": "compare",
    "camera.npy": "camera",
}


@dataclass(frozen=True)
class CameraPublicationSummary:
    cases: int
    ordered_manifest_digest: str
    complete_digest: str
    complete_path: Path


@dataclass(frozen=True)
class _PublicationTarget:
    case: InferenceCase
    physical_dataset: str
    stem: str
    published_pixel_frames: int
    destinations: Mapping[str, Path]


def _file_identity(path: Path) -> tuple[int, str]:
    digest = hashlib.blake2s()
    size = 0
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise BackendContractError(f"cannot read camera publication source {path}: {exc}") from exc
    return size, digest.hexdigest()


def _publication_targets(
    layout: CameraInferenceOutputLayout,
    cases: Sequence[InferenceCase],
) -> tuple[_PublicationTarget, ...]:
    targets: list[_PublicationTarget] = []
    seen_slots: set[int] = set()
    seen_paths: set[Path] = set()
    for case in cases:
        slot = int(case.slot)
        if slot in seen_slots:
            raise BackendContractError(f"camera publication contains duplicate slot {slot}")
        seen_slots.add(slot)
        physical_dataset = portable_output_component(
            case.metadata.get("physical_dataset"),
            field=f"camera publication slot {slot} physical_dataset",
        )
        stem = portable_output_component(
            case.metadata.get("publish_stem"),
            field=f"camera publication slot {slot} publish_stem",
        )
        try:
            published_pixel_frames = int(case.metadata["publication_pixel_frames"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError(
                f"camera publication slot {slot} lacks publication_pixel_frames"
            ) from exc
        if published_pixel_frames < 1:
            raise BackendContractError(
                f"camera publication slot {slot} has invalid publication_pixel_frames"
            )
        if (
            str(case.metadata.get("camera_publication_convention", ""))
            != "authoritative_absolute_c2w"
        ):
            raise BackendContractError(
                f"camera publication slot {slot} lacks authoritative absolute C2W provenance"
            )
        destinations = {
            artifact: layout.publish_root
            / public_kind
            / physical_dataset
            / f"{stem}{Path(artifact).suffix}"
            for artifact, public_kind in _REQUIRED_ARTIFACTS.items()
        }
        for destination in destinations.values():
            if destination in seen_paths:
                raise BackendContractError(
                    f"camera publication destination is duplicated: {destination}"
                )
            seen_paths.add(destination)
        targets.append(
            _PublicationTarget(
                case=case,
                physical_dataset=physical_dataset,
                stem=stem,
                published_pixel_frames=published_pixel_frames,
                destinations=destinations,
            )
        )
    if not targets:
        raise BackendContractError("camera publication requires at least one case")
    return tuple(targets)


def preflight_camera_publication(
    layout: CameraInferenceOutputLayout,
    cases: Sequence[InferenceCase],
) -> None:
    """Reject invalid or occupied public names before any generation starts."""

    targets = _publication_targets(layout, cases)
    occupied = [
        str(destination)
        for target in targets
        for destination in target.destinations.values()
        if destination.exists() or destination.is_symlink()
    ]
    if occupied:
        raise BackendContractError(
            f"camera publication targets already exist; first={occupied[:8]}"
        )
    publication_root = layout.run_root / "publication"
    if publication_root.exists() or publication_root.is_symlink():
        raise BackendContractError(
            f"camera publication provenance already exists: {publication_root}"
        )


def _artifact_records(
    sample_dir: Path,
    manifest: Mapping[str, Any],
    *,
    published_pixel_frames: int,
) -> dict[str, tuple[Path, int, str]]:
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list):
        raise BackendContractError(f"camera publication manifest has no artifacts: {sample_dir}")
    records: dict[str, tuple[Path, int, str]] = {}
    prefix = f"{sample_dir.name}/"
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise BackendContractError(
                f"camera publication artifact record is invalid: {sample_dir}"
            )
        raw_path = str(raw_record.get("path", ""))
        if not raw_path.startswith(prefix):
            raise BackendContractError(
                f"camera publication artifact path escaped its slot: {raw_path!r}"
            )
        name = raw_path.removeprefix(prefix)
        if name not in _REQUIRED_ARTIFACTS:
            continue
        if name in records:
            raise BackendContractError(f"camera publication manifest duplicates artifact {name!r}")
        try:
            expected_size = int(raw_record["bytes"])
            expected_digest = str(raw_record["digest"])
        except (KeyError, TypeError, ValueError) as exc:
            raise BackendContractError(
                f"camera publication artifact identity is invalid: {raw_path!r}"
            ) from exc
        source = sample_dir / name
        actual_size, actual_digest = _file_identity(source)
        if (actual_size, actual_digest) != (expected_size, expected_digest):
            raise BackendContractError(f"camera publication artifact identity drifted: {source}")
        records[name] = (source, actual_size, actual_digest)
    missing = sorted(set(_REQUIRED_ARTIFACTS).difference(records))
    if missing:
        raise BackendContractError(
            f"camera publication sample {sample_dir.name} lacks artifacts {missing}"
        )
    camera_path = records["camera.npy"][0]
    try:
        c2w = np.load(camera_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise BackendContractError(
            f"camera publication artifact is not a valid NPY: {camera_path}"
        ) from exc
    if (
        not isinstance(c2w, np.ndarray)
        or c2w.shape != (published_pixel_frames, 4, 4)
        or c2w.dtype != np.float64
        or not np.isfinite(c2w).all()
    ):
        raise BackendContractError(
            "camera publication artifact must be finite float64 [published_frames,4,4]"
        )
    return records


def publish_camera_triplets(
    generation_root: Path,
    layout: CameraInferenceOutputLayout,
    *,
    generation_pass: str,
    cases: Sequence[InferenceCase],
    family: str,
    weights_id: str,
    generation_complete_digest: str,
) -> CameraPublicationSummary:
    """Publish one completed camera run and commit its provenance last.

    Each public file is an atomic create-only hard link. The three links are
    followed by a per-sample receipt, while the publication ``COMPLETE.json``
    is the visibility gate for the complete batch.
    """

    targets = _publication_targets(layout, cases)
    preflight_camera_publication(layout, cases)
    pass_root = generation_root / generation_pass
    if (
        not (generation_root / "COMPLETE.json").is_file()
        or not (pass_root / "COMPLETE.json").is_file()
    ):
        raise BackendContractError("camera publication source transaction is incomplete")

    prepared: list[
        tuple[_PublicationTarget, Mapping[str, Any], dict[str, tuple[Path, int, str]]]
    ] = []
    for target in targets:
        sample_dir = pass_root / f"slot-{target.case.slot:06d}"
        manifest_path = sample_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BackendContractError(
                f"camera publication sample manifest is invalid: {manifest_path}: {exc}"
            ) from exc
        case_manifest = manifest.get("case", {})
        if (
            not isinstance(case_manifest, Mapping)
            or int(case_manifest.get("slot", -1)) != target.case.slot
            or str(case_manifest.get("sample_id", "")) != target.case.sample_id
            or str(manifest.get("family", "")) != family
            or str(manifest.get("weights_id", "")) != weights_id
        ):
            raise BackendContractError(
                f"camera publication sample identity drifted for slot {target.case.slot}"
            )
        prepared.append(
            (
                target,
                manifest,
                _artifact_records(
                    sample_dir,
                    manifest,
                    published_pixel_frames=target.published_pixel_frames,
                ),
            )
        )

    publication_root = layout.run_root / "publication"
    records: list[dict[str, Any]] = []
    for target, manifest, artifacts in prepared:
        public_artifacts: dict[str, dict[str, Any]] = {}
        for artifact_name in _REQUIRED_ARTIFACTS:
            source, size, digest = artifacts[artifact_name]
            destination = target.destinations[artifact_name]
            link_file_create_only(
                source,
                destination,
                error_type=BackendContractError,
                label="camera publication artifact",
            )
            public_artifacts[artifact_name] = {
                "path": destination.relative_to(layout.publish_root).as_posix(),
                "bytes": size,
                "digest": digest,
            }
        record = {
            "schema": "solarwm.wan22-camera-publication-sample.v1",
            "slot": int(target.case.slot),
            "sample_id": target.case.sample_id,
            "physical_dataset": target.physical_dataset,
            "stem": target.stem,
            "published_pixel_frames": target.published_pixel_frames,
            "source_manifest_digest": hashlib.blake2s(canonical_json_bytes(manifest)).hexdigest(),
            "artifacts": public_artifacts,
        }
        write_file_create_only(
            publication_root / "samples" / f"slot-{target.case.slot:06d}.json",
            canonical_json_bytes(record),
            error_type=BackendContractError,
            label="camera publication sample receipt",
        )
        records.append(record)

    ordered_bytes = b"".join(canonical_json_bytes(record) for record in records)
    ordered_digest = hashlib.blake2s(ordered_bytes).hexdigest()
    write_file_create_only(
        publication_root / "ordered-manifest.jsonl",
        ordered_bytes,
        error_type=BackendContractError,
        label="camera publication ordered manifest",
    )
    complete_payload = canonical_json_bytes(
        {
            "schema": "solarwm.wan22-camera-publication-complete.v1",
            "layout": layout.layout,
            "run_id": layout.run_id,
            "family": family,
            "weights_id": weights_id,
            "generation_pass": generation_pass,
            "generation_complete_digest": generation_complete_digest,
            "cases": len(records),
            "ordered_manifest_digest": ordered_digest,
        }
    )
    complete_path = publication_root / "COMPLETE.json"
    write_file_create_only(
        complete_path,
        complete_payload,
        error_type=BackendContractError,
        label="camera publication completion marker",
    )
    return CameraPublicationSummary(
        cases=len(records),
        ordered_manifest_digest=ordered_digest,
        complete_digest=hashlib.blake2s(complete_payload).hexdigest(),
        complete_path=complete_path,
    )


__all__ = [
    "CameraPublicationSummary",
    "preflight_camera_publication",
    "publish_camera_triplets",
]
