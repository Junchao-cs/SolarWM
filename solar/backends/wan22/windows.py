"""Deterministic Wan window geometry and 153f index materialization."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solar.data.index import IndexRow, iter_index, shard_identity
from solar.errors import DataContractError

PIXEL_FRAMES_81F = 81
PIXEL_FRAMES_153F = 153
WINDOW_HASH_NAMESPACE_81F = "solarwm-latent-81f-v1"
WINDOW_HASH_NAMESPACE_153F = "solarwm-latent-153f-v1"
SIX_WINDOW_153F_DATASETS = frozenset({"abot", "miradata", "sekai_game"})
_WINDOW_SAMPLE_ID = re.compile(r"^(?P<source>.+)/latent-153f-w(?P<index>[0-9]{2})$")


@dataclass(frozen=True)
class WindowMaterializationSummary:
    """Auditable result of one create-only fixed-window index build."""

    output: Path
    train_sources: int
    test_sources: int
    train_windows: int
    test_windows: int
    decompressed_sha256: str
    compressed_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "solarwm.wan22-153f-window-index.v1",
            "output": str(self.output),
            "train_sources": self.train_sources,
            "test_sources": self.test_sources,
            "train_windows": self.train_windows,
            "test_windows": self.test_windows,
            "total_windows": self.train_windows + self.test_windows,
            "window_hash_namespace": WINDOW_HASH_NAMESPACE_153F,
            "decompressed_sha256": self.decompressed_sha256,
            "compressed_sha256": self.compressed_sha256,
        }


def expected_81f_window_count(num_frames: int) -> int:
    """Return the release-defined number of five-second windows at 16 fps."""

    if num_frames < PIXEL_FRAMES_81F:
        raise DataContractError("81f preencoding source is shorter than 81 frames")
    # Round half up so a 160-frame source deterministically carries two windows.
    return max(1, (int(num_frames) - 1 + 40) // 80)


def expected_81f_window_start(preencoding: Mapping[str, Any], num_frames: int) -> int:
    """Return an endpoint-inclusive, uniformly spaced 81f window start."""

    try:
        window_index = int(preencoding["window_index"])
        window_count = int(preencoding["window_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError("81f preencoding window metadata is incomplete") from exc
    expected_count = expected_81f_window_count(num_frames)
    if window_count != expected_count or not 0 <= window_index < window_count:
        raise DataContractError(
            f"81f source requires {expected_count} window ordinals, got "
            f"index={window_index} count={window_count}"
        )
    max_start = int(num_frames) - PIXEL_FRAMES_81F
    if window_count == 1:
        return max_start // 2
    denominator = window_count - 1
    return (2 * window_index * max_start + denominator) // (2 * denominator)


def expected_153f_window_start(preencoding: Mapping[str, Any], num_frames: int) -> int:
    """Compute the release-defined deterministic 153-frame window assignment."""

    try:
        source_sample_id = str(preencoding["source_sample_id"])
        dataset = str(preencoding["source_dataset"])
        window_index = int(preencoding["window_index"])
        window_count = int(preencoding["window_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError("153f preencoding window metadata is incomplete") from exc
    if num_frames < PIXEL_FRAMES_153F:
        raise DataContractError("153f preencoding source is shorter than 153 frames")
    if dataset in SIX_WINDOW_153F_DATASETS:
        if window_count != 6 or not 0 <= window_index < 6:
            raise DataContractError("long-form 153f source must carry six valid window ordinals")
        max_start = num_frames - PIXEL_FRAMES_153F
        return (window_index * max_start + 2) // 5
    if window_count != 1 or window_index != 0:
        raise DataContractError("ordinary 153f source must carry exactly one window")
    namespace = str(preencoding.get("window_hash_namespace", WINDOW_HASH_NAMESPACE_153F))
    payload = f"{namespace}\0{source_sample_id}".encode()
    value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little")
    return value % (num_frames - PIXEL_FRAMES_153F + 1)


def _source_role(row: IndexRow, expected: str) -> None:
    declared = set()
    for name in ("split", "role", "recipe_role"):
        raw = row.values.get(name)
        if raw is None or raw == "":
            continue
        if not isinstance(raw, str):
            raise DataContractError(f"source row {row.sample_id!r} field {name!r} must be a string")
        declared.add(raw.strip().lower())
    if declared != {expected}:
        raise DataContractError(
            f"source row {row.sample_id!r} must declare only role {expected!r}, "
            f"got {sorted(declared)}"
        )


def _source_geometry(row: IndexRow) -> tuple[str, int]:
    dataset = str(row.values.get("dataset") or "").strip().lower()
    if not dataset:
        raise DataContractError(f"source row {row.sample_id!r} lacks dataset")
    if dataset not in row.sample_id.split("/")[:2]:
        raise DataContractError(
            f"source row {row.sample_id!r} does not contain dataset {dataset!r} in its identity"
        )
    manifest = row.values.get("manifest", {})
    video = manifest.get("video", {}) if isinstance(manifest, Mapping) else {}
    try:
        num_frames = int(row.values.get("num_frames") or video["num_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(f"source row {row.sample_id!r} lacks num_frames") from exc
    if num_frames < PIXEL_FRAMES_153F:
        raise DataContractError(
            f"source row {row.sample_id!r} has {num_frames} frames, needs at least 153"
        )
    return dataset, num_frames


def _reject_materialized_source(row: IndexRow) -> None:
    fixed_fields = (
        "source_sample_id",
        "source_dataset",
        "window_index",
        "window_count",
        "start_frame",
        "source_frame_indices",
    )
    if row.sample_id.rsplit("/", 1)[-1].startswith("latent-153f-w") or any(
        name in row.values for name in fixed_fields
    ):
        raise DataContractError(f"source row {row.sample_id!r} is already window-materialized")


def _authority_geometry(row: IndexRow) -> tuple[str, int, int, tuple[int, ...]]:
    match = _WINDOW_SAMPLE_ID.fullmatch(row.sample_id)
    if match is None:
        raise DataContractError(
            f"window authority sample_id must end in /latent-153f-wNN: {row.sample_id!r}"
        )
    try:
        start = int(row.values["start_frame"])
        indices = tuple(int(value) for value in row.values["source_frame_indices"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(
            f"window authority row {row.sample_id!r} lacks fixed frame geometry"
        ) from exc
    expected_indices = tuple(range(start, start + PIXEL_FRAMES_153F))
    if indices != expected_indices:
        raise DataContractError(
            f"window authority row {row.sample_id!r} is not one contiguous 153f window"
        )
    return match.group("source"), int(match.group("index")), start, indices


def _materialize_authority_row(
    raw_row: IndexRow,
    authority_row: IndexRow,
    role: str,
) -> tuple[dict[str, Any], int, int]:
    source_sample_id, window_index, start, indices = _authority_geometry(authority_row)
    if raw_row.sample_id != source_sample_id:
        raise DataContractError(
            f"window authority source {source_sample_id!r} differs from its raw row"
        )
    _source_role(raw_row, role)
    _reject_materialized_source(raw_row)
    dataset, num_frames = _source_geometry(raw_row)
    window_count = 6 if dataset in SIX_WINDOW_153F_DATASETS else 1
    if not 0 <= window_index < window_count:
        raise DataContractError(
            f"window authority row {authority_row.sample_id!r} has an invalid ordinal"
        )
    if start < 0 or indices[-1] >= num_frames:
        raise DataContractError(
            f"window authority row {authority_row.sample_id!r} lies outside {num_frames} frames"
        )
    expected_key = f"{raw_row.key}__latent153f_w{window_index:02d}"
    if authority_row.key != expected_key:
        raise DataContractError(
            f"window authority key {authority_row.key!r} != expected {expected_key!r}"
        )
    if dataset in SIX_WINDOW_153F_DATASETS:
        expected_start = expected_153f_window_start(
            {
                "source_sample_id": source_sample_id,
                "source_dataset": dataset,
                "window_index": window_index,
                "window_count": window_count,
            },
            num_frames,
        )
        if start != expected_start:
            raise DataContractError(
                f"long-form authority start {start} != expected {expected_start} "
                f"for {authority_row.sample_id!r}"
            )
    values = dict(raw_row.values)
    values.update(
        {
            "sample_id": authority_row.sample_id,
            "key": authority_row.key,
            "epoch_repeats": 1,
            "split": role,
            "source_sample_id": source_sample_id,
            "source_dataset": dataset,
            "window_index": window_index,
            "window_count": window_count,
            "window_hash_namespace": WINDOW_HASH_NAMESPACE_153F,
            "start_frame": start,
            "source_frame_indices": list(indices),
        }
    )
    return values, window_index, window_count


def _jsonl_line(row: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(row),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _link_create_only(path: Path, temporary: Path) -> None:
    if path.exists():
        raise DataContractError(f"window index output already exists: {path}")
    try:
        os.link(temporary, path)
    except FileExistsError as exc:
        raise DataContractError(f"window index output already exists: {path}") from exc


def _iter_source_rows(
    path: str | Path,
    role: str,
    *,
    seen_sources: set[str],
    identities: dict[str, object],
) -> Iterator[IndexRow]:
    count = 0
    for row in iter_index(path):
        if row.sample_id in seen_sources:
            raise DataContractError(
                f"duplicate or train/test-overlapping source sample_id: {row.sample_id!r}"
            )
        identity = shard_identity(row)
        if row.shard in identities and identities[row.shard] != identity:
            raise DataContractError(f"shard identity drift inside source indexes: {row.shard}")
        identities[row.shard] = identity
        _source_role(row, role)
        _reject_materialized_source(row)
        _source_geometry(row)
        seen_sources.add(row.sample_id)
        count += 1
        yield row
    if count == 0:
        raise DataContractError(f"empty {role} index: {Path(path)}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_source_database(
    database: sqlite3.Connection,
    train_index: str | Path,
    test_index: str | Path,
) -> dict[str, int]:
    database.execute(
        "CREATE TABLE sources ("
        "sample_id TEXT PRIMARY KEY, role TEXT NOT NULL, row_json TEXT NOT NULL, "
        "expected_mask INTEGER NOT NULL, seen_mask INTEGER NOT NULL DEFAULT 0)"
    )
    seen_sources: set[str] = set()
    identities: dict[str, object] = {}
    counts = {"train": 0, "test": 0}
    for source, role in ((train_index, "train"), (test_index, "test")):
        for row in _iter_source_rows(
            source,
            role,
            seen_sources=seen_sources,
            identities=identities,
        ):
            dataset, _ = _source_geometry(row)
            window_count = 6 if dataset in SIX_WINDOW_153F_DATASETS else 1
            database.execute(
                "INSERT INTO sources(sample_id, role, row_json, expected_mask) VALUES (?, ?, ?, ?)",
                (
                    row.sample_id,
                    role,
                    json.dumps(
                        dict(row.values),
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                    (1 << window_count) - 1,
                ),
            )
            counts[role] += 1
    database.commit()
    return counts


def _iter_authority_rows(
    database: sqlite3.Connection,
    path: str | Path,
    role: str,
) -> Iterator[dict[str, Any]]:
    count = 0
    seen_windows: set[str] = set()
    for authority in iter_index(path):
        if authority.sample_id in seen_windows:
            raise DataContractError(
                f"duplicate window authority sample_id: {authority.sample_id!r}"
            )
        seen_windows.add(authority.sample_id)
        source_sample_id, _, _, _ = _authority_geometry(authority)
        record = database.execute(
            "SELECT role, row_json, seen_mask FROM sources WHERE sample_id = ?",
            (source_sample_id,),
        ).fetchone()
        if record is None:
            raise DataContractError(
                f"window authority source is absent from raw indexes: {source_sample_id!r}"
            )
        raw_role, raw_json, seen_mask = record
        if raw_role != role:
            raise DataContractError(
                f"window authority role {role!r} differs from raw role {raw_role!r} "
                f"for {source_sample_id!r}"
            )
        raw = IndexRow.from_mapping(0, json.loads(raw_json))
        materialized, window_index, _ = _materialize_authority_row(raw, authority, role)
        bit = 1 << window_index
        if int(seen_mask) & bit:
            raise DataContractError(
                f"duplicate window ordinal {window_index} for source {source_sample_id!r}"
            )
        database.execute(
            "UPDATE sources SET seen_mask = seen_mask | ? WHERE sample_id = ?",
            (bit, source_sample_id),
        )
        count += 1
        yield materialized
    if count == 0:
        raise DataContractError(f"empty {role} window authority index: {Path(path)}")


def write_wan153f_window_index(
    train_index: str | Path,
    test_index: str | Path,
    train_window_index: str | Path,
    test_window_index: str | Path,
    output: str | Path,
) -> WindowMaterializationSummary:
    """Join released window authority to raw rows for ``solar preencode``."""

    target = Path(output).expanduser().resolve()
    if target.exists():
        raise DataContractError(f"window index output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.partial")
    database_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.sqlite3")
    database: sqlite3.Connection | None = None
    source_counts = {"train": 0, "test": 0}
    window_counts = {"train": 0, "test": 0}
    decompressed = hashlib.sha256()
    try:
        database = sqlite3.connect(database_path)
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("PRAGMA temp_store=MEMORY")
        source_counts = _build_source_database(database, train_index, test_index)
        with temporary.open("xb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                fileobj=raw_handle,
                compresslevel=9,
                mtime=0,
            ) as compressed_handle:
                for authority, role in (
                    (train_window_index, "train"),
                    (test_window_index, "test"),
                ):
                    for materialized in _iter_authority_rows(database, authority, role):
                        line = _jsonl_line(materialized)
                        decompressed.update(line)
                        compressed_handle.write(line)
                        window_counts[role] += 1
            database.commit()
            missing = database.execute(
                "SELECT sample_id FROM sources WHERE seen_mask != expected_mask LIMIT 1"
            ).fetchone()
            if missing is not None:
                raise DataContractError(
                    f"raw source lacks complete window authority: {missing[0]!r}"
                )
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        compressed_sha256 = _file_sha256(temporary)
        _link_create_only(target, temporary)
        return WindowMaterializationSummary(
            output=target,
            train_sources=source_counts["train"],
            test_sources=source_counts["test"],
            train_windows=window_counts["train"],
            test_windows=window_counts["test"],
            decompressed_sha256=decompressed.hexdigest(),
            compressed_sha256=compressed_sha256,
        )
    finally:
        if database is not None:
            database.close()
        temporary.unlink(missing_ok=True)
        database_path.unlink(missing_ok=True)


__all__ = [
    "PIXEL_FRAMES_153F",
    "SIX_WINDOW_153F_DATASETS",
    "WINDOW_HASH_NAMESPACE_153F",
    "WindowMaterializationSummary",
    "expected_153f_window_start",
    "write_wan153f_window_index",
]
