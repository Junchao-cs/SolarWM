"""Bind source-length conditions and completed outputs to their actual inputs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import tarfile
import uuid
from pathlib import Path
from typing import Any

from solarwm.config.loader import canonical_json
from solarwm.data.index import IndexRow, validate_relative_key
from solarwm.data.transport import GCSResolver, LocalResolver

_SOURCE_MEMBERS = {
    "video_member": "source.mp4",
    "camera_member": "source.camera.npz",
    "intrinsics_member": "source.intrinsics.npy",
    "manifest_member": "source.manifest.json",
}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        part.write_bytes(canonical_json(value))
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(row: dict[str, Any], args: Any) -> dict[str, Any]:
    root = str(args.dataset_root)
    return {
        "root": root.rstrip("/") if root.startswith("gs://") else str(Path(root).resolve()),
        "row": row,
    }


def preparation_identity(row: dict[str, Any], args: Any) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "schema": "solarwm.h3-source-condition-inputs.v1",
                "source": source_identity(row, args),
                "model": args.config["model"],
            }
        )
    ).hexdigest()


def source_files(row: dict[str, Any], args: Any) -> Path:
    resolver = (
        GCSResolver(args.dataset_root, Path(args.work_dir) / "shard-cache", 48 << 30)
        if args.dataset_root.startswith("gs://")
        else LocalResolver(Path(args.dataset_root))
    )
    local = resolver.resolve(IndexRow.from_mapping(0, row))
    case = Path(args.work_dir) / "cases" / row["key"]
    case.mkdir(parents=True, exist_ok=True)
    identity = source_identity(row, args)
    receipt_path = case / "SOURCE.json"
    with (case / ".source.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads(receipt_path.read_text()) if receipt_path.is_file() else None
        if previous is not None and previous.get("identity") != identity:
            raise ValueError("Source cache belongs to another dataset or row; use a new work_dir")
        sizes = {}
        with tarfile.open(local, "r:") as archive:
            for field, name in _SOURCE_MEMBERS.items():
                member = archive.getmember(validate_relative_key(row[field], field=field))
                if not member.isfile():
                    raise ValueError(f"Source member is not a regular file: {field}")
                sizes[name] = member.size
                target = case / name
                if (
                    previous is not None
                    and previous.get("sizes", {}).get(name) == member.size
                    and target.is_file()
                    and target.stat().st_size == member.size
                ):
                    continue
                part = target.with_name(f".{name}.{os.getpid()}.tmp")
                try:
                    with archive.extractfile(member) as source, part.open("wb") as output:
                        shutil.copyfileobj(source, output, 8 << 20)
                    if part.stat().st_size != member.size:
                        raise ValueError(f"Incomplete source member: {field}")
                    part.replace(target)
                finally:
                    part.unlink(missing_ok=True)
        atomic_json(receipt_path, {"identity": identity, "sizes": sizes})
    return case


def prepared_case_receipt(case: Path, row: dict[str, Any], args: Any) -> dict[str, Any]:
    receipt = json.loads((case / "READY.json").read_text())
    if receipt.get("preparation_identity") != preparation_identity(row, args):
        raise ValueError("Condition cache belongs to another source or model; use a new work_dir")
    if receipt.get("condition_sha256") != file_sha256(case / "condition.safetensors"):
        raise ValueError("Frozen condition changed after preparation")
    if row.get("input_kind") == "demo":
        from .demo_conditions import source_digest

        digest = source_digest(case)
    else:
        digest = file_sha256(case / "source.mp4")
    if receipt.get("source_sha256") != digest:
        raise ValueError("Cached source video changed after preparation")
    return receipt


def result_identity(
    args: Any, condition: dict[str, Any], *, checkpoint_sha256: str, silence_sha256: str, seed: int
) -> str:
    return hashlib.sha256(
        canonical_json(
            {
                "schema": "solarwm.h3-source-result-inputs.v1",
                "preparation_identity": condition["preparation_identity"],
                "condition_sha256": condition["condition_sha256"],
                "checkpoint_receipt_sha256": checkpoint_sha256,
                "weight_source": args.weight_source,
                "silence_sha256": silence_sha256,
                "noise_seed": seed,
                "fps_policy": args.fps_policy,
                "save_latents": bool(args.config["inference"].get("save_latents", False)),
                **(
                    {"stream_decode": True}
                    if args.config["inference"].get("stream_decode", False)
                    else {}
                ),
            }
        )
    ).hexdigest()


def verify_completed_result(manifest: Path, *, expected_identity: str, save_latents: bool) -> None:
    receipt = json.loads(manifest.read_text())
    if receipt.get("result_identity") != expected_identity:
        raise ValueError("Existing result belongs to another condition or inference configuration")
    suffixes = {
        ".mp4": "output_sha256",
        ".compare.mp4": "compare_sha256",
        ".storyboard.jpg": "storyboard_sha256",
    }
    if receipt.get("source_row", {}).get("input_kind") == "demo":
        suffixes.pop(".compare.mp4")
    if save_latents:
        suffixes[".latents.safetensors"] = "latents_sha256"
    for suffix, field in suffixes.items():
        path = manifest.with_name(manifest.stem + suffix)
        if not path.is_file() or file_sha256(path) != receipt.get(field):
            raise ValueError(
                f"Existing result file differs from its completion receipt: {path.name}"
            )
