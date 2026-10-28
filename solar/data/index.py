"""Ordered, transport-neutral WebDataset index contracts."""

from __future__ import annotations

import base64
import binascii
import gzip
import hashlib
import json
import random
import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, TextIO

from solar.errors import DataContractError


def _open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


_URI_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:")


def validate_relative_key(value: Any, *, field: str = "shard") -> str:
    """Validate a portable path stored in a published index.

    Runtime roots may be absolute or object-store URIs. Index keys may not be;
    keeping that distinction is what makes one index portable between them.
    """

    if not isinstance(value, str) or not value or "\\" in value:
        raise DataContractError(f"{field} must be a non-empty POSIX path: {value!r}")
    if value.startswith("/") or _URI_SCHEME.match(value):
        raise DataContractError(f"{field} must be relative, got {value!r}")
    # PurePosixPath normalizes repeated separators and dot segments. Inspect
    # the authoritative bytes first so non-canonical spellings cannot alias.
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise DataContractError(f"{field} contains a non-portable segment: {value!r}")
    path = PurePosixPath(value)
    return path.as_posix()


def resolve_index_path(data: Mapping[str, Any], field: str) -> Path:
    """Resolve a portable index key independently from shard transport.

    Recipe controls are normally staged locally even when shards stream from
    object storage. ``index_root`` states that control-plane location. A local
    all-in-one tree may omit it and use its payload root for both purposes.
    """

    key = validate_relative_key(data.get(field), field=f"data.{field}")
    raw_index_root = str(data.get("index_root") or "").strip()
    transport = data.get("transport", {})
    if not isinstance(transport, Mapping):
        raise DataContractError("data.transport must be a mapping")
    kind = str(transport.get("kind") or "").strip().lower()
    if raw_index_root:
        root = Path(raw_index_root).expanduser()
        if not root.is_absolute():
            raise DataContractError("data.index_root must be an absolute local path")
        return root / key
    if kind == "local":
        root = Path(str(transport.get("root") or "")).expanduser()
        if not root.is_absolute():
            raise DataContractError("local data.transport.root must be absolute")
        return root / key
    raise DataContractError(
        "bucket transport requires an absolute local data.index_root for staged recipe controls"
    )


@dataclass(frozen=True)
class IndexRow:
    """A lossless normalized view over one canonical JSONL row."""

    ordinal: int
    sample_id: str
    key: str
    shard: str
    epoch_repeats: int
    values: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, ordinal: int, values: Mapping[str, Any]) -> IndexRow:
        if not isinstance(values, Mapping):
            raise DataContractError(f"index row {ordinal} is not an object")
        identities: dict[str, str] = {}
        for field in ("sample_id", "key"):
            value = values.get(field)
            if not isinstance(value, str) or not value:
                raise DataContractError(f"index row {ordinal} {field} must be a non-empty string")
            identities[field] = value
        sample_id = identities["sample_id"]
        key = identities["key"]
        shard = validate_relative_key(values.get("shard"))
        raw_repeats = values.get("epoch_repeats", 1)
        if type(raw_repeats) is not int:
            raise DataContractError(f"index row {ordinal} epoch_repeats must be an integer")
        if raw_repeats < 1:
            raise DataContractError(f"index row {ordinal} epoch_repeats must be >= 1")
        return cls(
            ordinal=ordinal,
            sample_id=sample_id,
            key=key,
            shard=shard,
            epoch_repeats=raw_repeats,
            values=dict(values),
        )


@dataclass(frozen=True)
class IndexInventory:
    rows: int
    virtual_occurrences: int
    ordered_sample_id_digest: str
    ordered_row_digest: str
    decompressed_digest: str


@dataclass(frozen=True)
class ShardIdentity:
    """Canonical transport metadata declared by one index row."""

    generation: str | None
    size: int
    md5_b64: str | None
    digest: str | None


def shard_identity(row: IndexRow) -> ShardIdentity | None:
    """Parse and validate optional transport metadata on an index row.

    A size is required once any identity field is present. Digests remain
    optional publication metadata; runtime transport does not require them.
    Generation may be omitted for a local tree, while bucket resolvers impose
    it as an additional requirement.
    """

    values = row.values
    names = (
        "shard_generation",
        "shard_size",
        "shard_md5_b64",
        "shard_digest",
    )
    if not any(values.get(name) is not None and values.get(name) != "" for name in names):
        return None

    raw_size = values.get("shard_size")
    if raw_size is None or raw_size == "":
        raise DataContractError(
            f"sample {row.sample_id!r} has a partial shard identity: shard_size is required"
        )
    if isinstance(raw_size, bool):
        raise DataContractError(f"sample {row.sample_id!r} has invalid shard_size")
    if isinstance(raw_size, float):
        raise DataContractError(f"sample {row.sample_id!r} has invalid shard_size")
    try:
        size = int(raw_size)
    except (TypeError, ValueError) as exc:
        raise DataContractError(f"sample {row.sample_id!r} has invalid shard_size") from exc
    if size <= 0:
        raise DataContractError(
            f"sample {row.sample_id!r} has a partial shard identity: shard_size is required"
        )

    generation_value = values.get("shard_generation")
    generation_text = (
        str(generation_value).strip()
        if generation_value is not None and generation_value != ""
        else ""
    )
    generation = generation_text or None

    raw_md5 = values.get("shard_md5_b64")
    md5_b64 = str(raw_md5).strip() if raw_md5 is not None and raw_md5 != "" else None
    if md5_b64 is not None:
        try:
            decoded_md5 = base64.b64decode(md5_b64, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise DataContractError(f"sample {row.sample_id!r} has invalid shard_md5_b64") from exc
        if len(decoded_md5) != hashlib.md5().digest_size:
            raise DataContractError(f"sample {row.sample_id!r} has invalid shard_md5_b64")
        md5_b64 = base64.b64encode(decoded_md5).decode("ascii")

    raw_digest = values.get("shard_digest")
    digest = (
        str(raw_digest).strip().lower() if raw_digest is not None and raw_digest != "" else None
    )
    if digest is not None:
        try:
            decoded_digest = bytes.fromhex(digest)
        except ValueError as exc:
            raise DataContractError(f"sample {row.sample_id!r} has invalid shard_digest") from exc
        if len(digest) != 64 or len(decoded_digest) != hashlib.blake2s().digest_size:
            raise DataContractError(f"sample {row.sample_id!r} has invalid shard_digest")

    return ShardIdentity(
        generation=generation,
        size=size,
        md5_b64=md5_b64,
        digest=digest,
    )


def iter_index(path: str | Path) -> Iterator[IndexRow]:
    source = Path(path)
    with _open_text(source) as handle:
        ordinal = 0
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                values = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataContractError(f"invalid JSON at {source}:{line_number}: {exc}") from exc
            yield IndexRow.from_mapping(ordinal, values)
            ordinal += 1


def read_index(path: str | Path) -> tuple[IndexRow, ...]:
    rows = tuple(iter_index(path))
    if not rows:
        raise DataContractError(f"empty index: {Path(path)}")
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        if row.sample_id in seen:
            duplicates.append(row.sample_id)
        seen.add(row.sample_id)
    if duplicates:
        preview = sorted(set(duplicates))[:8]
        raise DataContractError(f"duplicate sample_id values: {preview}")
    _validate_shard_identities(rows)
    return rows


def select_index_rows(
    rows: tuple[IndexRow, ...],
    *,
    sample_count: int,
    seed: int,
) -> tuple[IndexRow, ...]:
    """Choose one reproducible validation subset from a recipe test index."""

    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise DataContractError("validation sample_count must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DataContractError("validation selection seed must be a non-negative integer")
    if len(rows) < sample_count:
        raise DataContractError(
            "recipe test index has fewer rows than validation.sample_count: "
            f"rows={len(rows)} sample_count={sample_count}"
        )
    indices = random.Random(seed).sample(range(len(rows)), sample_count)
    return tuple(rows[index] for index in indices)


def _validate_shard_identities(rows: Iterable[IndexRow]) -> None:
    identities: dict[str, ShardIdentity | None] = {}
    for row in rows:
        identity = shard_identity(row)
        if row.shard in identities and identities[row.shard] != identity:
            raise DataContractError(f"shard identity drift inside index: {row.shard}")
        identities[row.shard] = identity


def inventory(path: str | Path, rows: Iterable[IndexRow] | None = None) -> IndexInventory:
    source = Path(path)
    materialized = tuple(rows) if rows is not None else read_index(source)
    ordered = hashlib.blake2s()
    ordered_rows = hashlib.blake2s()
    for row in materialized:
        ordered.update(row.sample_id.encode("utf-8"))
        ordered.update(b"\n")
        ordered_rows.update(
            (
                json.dumps(
                    row.values,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode()
        )
    decompressed = hashlib.blake2s()
    with _open_text(source) as handle:
        for line in handle:
            decompressed.update(line.encode("utf-8"))
    return IndexInventory(
        rows=len(materialized),
        virtual_occurrences=sum(row.epoch_repeats for row in materialized),
        ordered_sample_id_digest=ordered.hexdigest(),
        ordered_row_digest=ordered_rows.hexdigest(),
        decompressed_digest=decompressed.hexdigest(),
    )


def ensure_disjoint(train: Iterable[IndexRow], test: Iterable[IndexRow]) -> None:
    train_ids = {row.sample_id for row in train}
    test_ids = {row.sample_id for row in test}
    overlap = train_ids & test_ids
    if overlap:
        raise DataContractError(
            f"train/test sample_id overlap ({len(overlap)}): {sorted(overlap)[:8]}"
        )
