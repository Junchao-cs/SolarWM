"""Deterministic uncompressed tar and relative-index production."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import tarfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solar.config.loader import canonical_json
from solar.data.index import validate_relative_key
from solar.errors import DataContractError

from .contracts import EncodedPayload


@dataclass(frozen=True)
class ShardReceipt:
    relative_path: str
    samples: int
    size: int
    digest: str
    md5_b64: str
    rows: tuple[Mapping[str, Any], ...]


def _digest(path: Path) -> tuple[str, str]:
    digest = hashlib.blake2s()
    md5 = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
            md5.update(chunk)
    return digest.hexdigest(), base64.b64encode(md5.digest()).decode()


def _tar_member(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mtime = 0
    info.mode = 0o644
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(value))


def write_shard(
    root: str | Path,
    relative_path: str,
    samples: Sequence[EncodedPayload],
) -> ShardReceipt:
    """Write one shard without overwriting any existing output."""

    relative = validate_relative_key(relative_path)
    if not relative.endswith(".tar"):
        raise DataContractError("preencoded shards must use the .tar suffix")
    if not samples:
        raise DataContractError("cannot write an empty preencoded shard")
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise DataContractError("preencoded shard contains duplicate sample IDs")

    target = Path(root).resolve() / relative
    if target.exists():
        raise DataContractError(f"preencoded shard already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    rows: list[dict[str, Any]] = []
    try:
        with tarfile.open(temporary, "w:", format=tarfile.USTAR_FORMAT) as archive:
            for sample in samples:
                prefix = f"samples/{hashlib.blake2s(sample.sample_id.encode()).hexdigest()[:24]}"
                members: dict[str, str] = {}
                for suffix, value in sorted(sample.members.items()):
                    member = f"{prefix}/{suffix}"
                    _tar_member(archive, member, value)
                    members[suffix] = member
                provenance = {
                    "schema": "solarwm.preencoded-sample.v1",
                    "sample_id": sample.sample_id,
                    "key": sample.key,
                    "source_sample_id": sample.source_sample_id,
                    "start_frame": sample.start_frame,
                    "source_frame_indices": list(sample.source_frame_indices),
                    "encoder_contract_digest": sample.encoder_contract_digest,
                    "members": members,
                    "metadata": dict(sample.metadata),
                }
                provenance_member = f"{prefix}/provenance.json"
                _tar_member(archive, provenance_member, canonical_json(provenance))
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "key": sample.key,
                        "source_sample_id": sample.source_sample_id,
                        "shard": relative,
                        "epoch_repeats": 1,
                        "start_frame": sample.start_frame,
                        "source_frame_indices": list(sample.source_frame_indices),
                        "encoder_contract_digest": sample.encoder_contract_digest,
                        "members": members,
                        "provenance_member": provenance_member,
                        "metadata": dict(sample.metadata),
                    }
                )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    digest, md5_b64 = _digest(target)
    size = target.stat().st_size
    for row in rows:
        row["shard_size"] = size
        row["shard_digest"] = digest
        row["shard_md5_b64"] = md5_b64
    return ShardReceipt(relative, len(samples), size, digest, md5_b64, tuple(rows))


def write_index(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Atomically write ordered JSONL and return its content digest."""

    if not rows:
        raise DataContractError("cannot write an empty preencoded index")
    sample_ids: set[str] = set()
    payload = bytearray()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id or sample_id in sample_ids:
            raise DataContractError(f"missing or duplicate sample_id {sample_id!r}")
        sample_ids.add(sample_id)
        validate_relative_key(str(row.get("shard") or ""))
        payload.extend(canonical_json(dict(row)))
    target = Path(path)
    if target.exists():
        raise DataContractError(f"preencoded index already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.blake2s(payload).hexdigest()
