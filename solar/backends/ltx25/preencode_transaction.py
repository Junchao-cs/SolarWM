"""Create-only, multi-rank local publication for LTX-2.5 preencoding.

Every raw rank owns one private subtree and commits its receipt last.  Rank
zero trusts neither Python objects nor directory naming: it rereads and
byte-validates every rank index, receipt, and shard before constructing the
canonical corpus controls.  ``COMPLETE.json`` is the final staging write and
the whole tree becomes visible through one atomic rename.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from solar.data import IndexRow, read_index
from solar.data.index import validate_relative_key
from solar.errors import DataContractError
from solar.preencode import EncoderContract, ShardReceipt, write_index
from solar.runtime.create_only import publish_directory_no_replace
from solar.runtime.serialization import canonical_json_bytes

LTX25_CORPUS_INDEX_PATH = "index.jsonl"
LTX25_ENCODER_CONTRACT_PATH = "encoder-contract.json"
LTX25_CORPUS_CONTROL_PATH = "corpus-control.json"
LTX25_COMPLETE_PATH = "COMPLETE.json"
LTX25_RANK_ROOT = "ranks"


def _exists(path: Path) -> bool:
    return os.path.lexists(path)


def _digest_bytes(value: bytes) -> str:
    return hashlib.blake2s(value).hexdigest()


def _validated_digest(value: Any, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DataContractError(f"{field} must be a 64-character lowercase hex content digest")
    return value


def _ordered_sample_id_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    return _digest_bytes(b"".join(f"{row['sample_id']!s}\n".encode() for row in rows))


def _atomic_write(path: Path, value: bytes) -> str:
    if _exists(path):
        raise DataContractError(f"LTX preencode control already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return _digest_bytes(value)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> str:
    return _atomic_write(path, canonical_json_bytes(value))


def _file_identity(path: Path) -> tuple[int, str, str]:
    if not path.is_file() or path.is_symlink():
        raise DataContractError(f"LTX preencode artifact is not a regular file: {path}")
    digest = hashlib.blake2s()
    md5 = hashlib.md5(usedforsecurity=False)
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
            md5.update(chunk)
    return size, digest.hexdigest(), base64.b64encode(md5.digest()).decode("ascii")


def _safe_relative(value: str, *, field: str) -> str:
    try:
        return validate_relative_key(value, field=field)
    except DataContractError:
        raise
    except Exception as exc:
        raise DataContractError(f"invalid {field}: {value!r}") from exc


def rank_prefix(rank: int) -> str:
    if rank < 0:
        raise DataContractError("LTX preencode rank must be nonnegative")
    return f"{LTX25_RANK_ROOT}/rank-{rank:05d}"


def rank_shard_relative(rank: int, shard_index: int) -> str:
    if shard_index < 0:
        raise DataContractError("LTX preencode shard index must be nonnegative")
    return f"{rank_prefix(rank)}/shards/part-{shard_index:08d}.tar"


def create_staging(target: str | Path) -> Path:
    destination = Path(target).expanduser().resolve()
    if _exists(destination):
        raise DataContractError(f"LTX preencode output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    staging.mkdir(mode=0o755)
    return staging


def write_rank_publication(
    staging: str | Path,
    *,
    rank: int,
    world_size: int,
    rows: Sequence[Mapping[str, Any]],
    shards: Sequence[ShardReceipt],
    index_relative_path: str,
    provider_identity: str,
    codec_identity: str,
    codec_load_receipt_digest: str,
    encoder_contract_digest: str,
) -> Mapping[str, Any]:
    """Commit one private rank index and receipt after all of its shards."""

    if world_size < 1 or not 0 <= rank < world_size:
        raise DataContractError("LTX preencode rank is outside world_size")
    if not rows or not shards:
        raise DataContractError("every LTX preencode rank must publish samples and shards")
    if not all(
        isinstance(value, str) and value.strip()
        for value in (
            provider_identity,
            codec_identity,
            codec_load_receipt_digest,
            encoder_contract_digest,
        )
    ):
        raise DataContractError("LTX rank receipt lacks provider/codec/contract identity")
    _validated_digest(
        codec_load_receipt_digest,
        field="codec_load_receipt_digest",
    )
    _validated_digest(
        encoder_contract_digest,
        field="encoder_contract_digest",
    )

    root = Path(staging).resolve()
    prefix = rank_prefix(rank)
    relative_index = _safe_relative(index_relative_path, field="rank index")
    if not relative_index.endswith(".jsonl"):
        raise DataContractError("LTX rank index must use the .jsonl suffix")
    full_index_relative = f"{prefix}/{relative_index}"
    index_path = root / full_index_relative
    index_digest = write_index(index_path, rows)
    index_bytes = index_path.stat().st_size

    shard_records: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    for receipt in shards:
        _validated_digest(receipt.digest, field="rank shard digest")
        relative = _safe_relative(receipt.relative_path, field="rank shard")
        if not relative.startswith(f"{prefix}/shards/"):
            raise DataContractError(f"rank {rank} shard escaped its private subtree: {relative!r}")
        if relative in declared_paths:
            raise DataContractError(f"rank {rank} declared duplicate shard {relative!r}")
        declared_paths.add(relative)
        shard_records.append(
            {
                "path": relative,
                "samples": int(receipt.samples),
                "bytes": int(receipt.size),
                "digest": str(receipt.digest),
                "md5_b64": str(receipt.md5_b64),
                "local_generation": f"local-digest:{receipt.digest}",
            }
        )
    indexed_paths = {str(row.get("shard") or "") for row in rows}
    if indexed_paths != declared_paths:
        raise DataContractError("LTX rank index and shard receipt inventories differ")

    receipt = {
        "schema": "solarwm.ltx25.preencode-rank.v1",
        "rank": rank,
        "world_size": world_size,
        "provider_identity": provider_identity,
        "codec_identity": codec_identity,
        "codec_load_receipt_digest": codec_load_receipt_digest,
        "encoder_contract_digest": encoder_contract_digest,
        "samples": len(rows),
        "shards": shard_records,
        "index": full_index_relative,
        "index_bytes": index_bytes,
        "index_digest": index_digest,
        "ordered_sample_id_digest": _ordered_sample_id_digest(rows),
    }
    _atomic_json(root / prefix / "receipt.json", receipt)
    return receipt


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise DataContractError(f"LTX rank receipt is missing: {path}")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataContractError(f"LTX rank receipt is invalid JSON: {path}") from exc
    if not isinstance(value, dict) or raw != canonical_json_bytes(value):
        raise DataContractError(f"LTX rank receipt bytes are not canonical: {path}")
    return value, raw


def _validate_expected_row(published: IndexRow, expected: IndexRow) -> None:
    values = published.values
    if published.key != expected.key:
        raise DataContractError(f"LTX published key differs for sample {published.sample_id!r}")
    try:
        expected_start = int(expected.values["start_frame"])
        expected_indices = tuple(int(item) for item in expected.values["source_frame_indices"])
        actual_start = int(values["start_frame"])
        actual_indices = tuple(int(item) for item in values["source_frame_indices"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataContractError(
            f"LTX sample {published.sample_id!r} lacks frozen source-window identity"
        ) from exc
    if (actual_start, actual_indices) != (expected_start, expected_indices):
        raise DataContractError(
            f"LTX published source window differs for sample {published.sample_id!r}"
        )


def _restore_source_logical_fields(
    published: Mapping[str, Any],
    expected: IndexRow,
) -> dict[str, Any]:
    """Restore source scheduling/geometry without leaking raw transport fields."""

    result = dict(published)
    indices = tuple(int(item) for item in result["source_frame_indices"])
    minimum_frames = max(indices) + 1
    raw_num_frames = expected.values.get("num_frames", result.get("num_frames"))
    if isinstance(raw_num_frames, bool):
        raise DataContractError(f"LTX source sample {expected.sample_id!r} has invalid num_frames")
    try:
        num_frames = int(raw_num_frames)
    except (TypeError, ValueError) as exc:
        raise DataContractError(
            f"LTX source sample {expected.sample_id!r} has invalid num_frames"
        ) from exc
    if num_frames < minimum_frames:
        raise DataContractError(
            f"LTX source sample {expected.sample_id!r} is shorter than its frozen window"
        )
    result["num_frames"] = num_frames

    raw_expected_fps = expected.values.get("fps")
    raw_published_fps = result.get("fps")
    raw_fps = raw_expected_fps if raw_expected_fps is not None else raw_published_fps
    if isinstance(raw_fps, bool):
        raise DataContractError(f"LTX source sample {expected.sample_id!r} has invalid fps")
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError) as exc:
        raise DataContractError(
            f"LTX source sample {expected.sample_id!r} has invalid fps"
        ) from exc
    if not math.isfinite(fps) or fps <= 0:
        raise DataContractError(f"LTX source sample {expected.sample_id!r} has invalid fps")
    if raw_expected_fps is not None and raw_published_fps is not None:
        try:
            published_fps = float(raw_published_fps)
        except (TypeError, ValueError) as exc:
            raise DataContractError(
                f"LTX published sample {expected.sample_id!r} has invalid fps"
            ) from exc
        if not math.isclose(published_fps, fps, rel_tol=0.0, abs_tol=1e-9):
            raise DataContractError(f"LTX published fps differs for sample {expected.sample_id!r}")
    result["fps"] = raw_expected_fps if raw_expected_fps is not None else fps
    result["epoch_repeats"] = expected.epoch_repeats
    return result


def finalize_local_preencode(
    staging: str | Path,
    target: str | Path,
    *,
    expected_rows: Sequence[IndexRow],
    source_index_path: str | Path,
    world_size: int,
    provider_identity: str,
    codec_identity: str,
    codec_load_receipt_digest: str,
    encoder_contract: EncoderContract,
) -> Mapping[str, Any]:
    """Validate the complete rank universe and atomically publish one local tree."""

    root = Path(staging).resolve()
    destination = Path(target).expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise DataContractError("LTX preencode staging root is missing or is a symlink")
    if _exists(destination):
        raise DataContractError(f"LTX preencode output already exists: {destination}")
    if world_size < 1:
        raise DataContractError("LTX preencode world_size must be positive")
    if not expected_rows:
        raise DataContractError("LTX preencode expected source index is empty")
    expected_by_id = {row.sample_id: row for row in expected_rows}
    if len(expected_by_id) != len(expected_rows):
        raise DataContractError("LTX expected source index contains duplicate sample IDs")
    _validated_digest(
        codec_load_receipt_digest,
        field="codec_load_receipt_digest",
    )
    _validated_digest(encoder_contract.digest, field="encoder_contract_digest")
    if set(path.name for path in root.iterdir()) != {LTX25_RANK_ROOT}:
        raise DataContractError("LTX staging root contains unexpected pre-finalization entries")

    ranks_root = root / LTX25_RANK_ROOT
    expected_rank_names = {f"rank-{rank:05d}" for rank in range(world_size)}
    if not ranks_root.is_dir() or ranks_root.is_symlink():
        raise DataContractError("LTX preencode rank root is missing or is a symlink")
    actual_rank_names = {path.name for path in ranks_root.iterdir()}
    if actual_rank_names != expected_rank_names:
        raise DataContractError(
            "LTX preencode rank coverage differs: "
            f"got {sorted(actual_rank_names)}, expected {sorted(expected_rank_names)}"
        )

    published_by_id: dict[str, dict[str, Any]] = {}
    shard_records: list[dict[str, Any]] = []
    rank_receipts: list[dict[str, Any]] = []
    for rank in range(world_size):
        prefix = rank_prefix(rank)
        rank_root = root / prefix
        if not rank_root.is_dir() or rank_root.is_symlink():
            raise DataContractError(f"LTX rank {rank} private root is invalid")
        receipt_path = rank_root / "receipt.json"
        receipt, receipt_bytes = _read_canonical_json(receipt_path)
        required = {
            "schema": "solarwm.ltx25.preencode-rank.v1",
            "rank": rank,
            "world_size": world_size,
            "provider_identity": provider_identity,
            "codec_identity": codec_identity,
            "codec_load_receipt_digest": codec_load_receipt_digest,
            "encoder_contract_digest": encoder_contract.digest,
        }
        for field, expected in required.items():
            if receipt.get(field) != expected:
                raise DataContractError(
                    f"LTX rank {rank} receipt {field} differs: {receipt.get(field)!r}"
                )
        for field in (
            "codec_load_receipt_digest",
            "encoder_contract_digest",
            "index_digest",
            "ordered_sample_id_digest",
        ):
            _validated_digest(
                receipt.get(field),
                field=f"rank {rank} receipt {field}",
            )
        index_relative = _safe_relative(str(receipt.get("index") or ""), field="rank index")
        if not index_relative.startswith(f"{prefix}/"):
            raise DataContractError(f"LTX rank {rank} index escaped its private subtree")
        index_path = root / index_relative
        index_size, index_digest, _index_md5 = _file_identity(index_path)
        if (
            int(receipt.get("index_bytes", -1)) != index_size
            or receipt.get("index_digest") != index_digest
        ):
            raise DataContractError(f"LTX rank {rank} index bytes differ from its receipt")
        rank_rows = read_index(index_path)
        if int(receipt.get("samples", -1)) != len(rank_rows):
            raise DataContractError(f"LTX rank {rank} sample count differs from its index")
        if receipt.get("ordered_sample_id_digest") != _ordered_sample_id_digest(
            [row.values for row in rank_rows]
        ):
            raise DataContractError(f"LTX rank {rank} ordered sample identity differs")

        raw_shards = receipt.get("shards")
        if not isinstance(raw_shards, list) or not raw_shards:
            raise DataContractError(f"LTX rank {rank} receipt has no shards")
        rank_shards: dict[str, dict[str, Any]] = {}
        for raw_record in raw_shards:
            if not isinstance(raw_record, Mapping):
                raise DataContractError(f"LTX rank {rank} shard receipt is invalid")
            record = dict(raw_record)
            _validated_digest(
                record.get("digest"),
                field=f"rank {rank} shard digest",
            )
            relative = _safe_relative(str(record.get("path") or ""), field="rank shard")
            if not relative.startswith(f"{prefix}/shards/") or relative in rank_shards:
                raise DataContractError(f"LTX rank {rank} shard ownership/uniqueness differs")
            size, digest, md5_b64 = _file_identity(root / relative)
            expected_identity = (
                int(record.get("bytes", -1)),
                str(record.get("digest") or ""),
                str(record.get("md5_b64") or ""),
            )
            if expected_identity != (size, digest, md5_b64):
                raise DataContractError(
                    f"LTX rank {rank} shard bytes differ from receipt: {relative}"
                )
            if record.get("local_generation") != f"local-digest:{digest}":
                raise DataContractError(f"LTX rank {rank} shard local identity differs")
            record["path"] = relative
            rank_shards[relative] = record

        row_counts: dict[str, int] = {path: 0 for path in rank_shards}
        for row in rank_rows:
            if row.sample_id in published_by_id:
                raise DataContractError(f"duplicate LTX sample_id across ranks: {row.sample_id!r}")
            expected = expected_by_id.get(row.sample_id)
            if expected is None:
                raise DataContractError(
                    f"published LTX sample is outside the source index: {row.sample_id!r}"
                )
            _validate_expected_row(row, expected)
            record = rank_shards.get(row.shard)
            if record is None:
                raise DataContractError(
                    f"LTX rank {rank} index references an undeclared shard {row.shard!r}"
                )
            metadata = row.values.get("metadata")
            if not isinstance(metadata, Mapping):
                raise DataContractError(f"LTX sample {row.sample_id!r} metadata is not a mapping")
            if (
                row.values.get("encoder_contract_digest") != encoder_contract.digest
                or metadata.get("codec_identity") != codec_identity
            ):
                raise DataContractError(
                    f"LTX sample {row.sample_id!r} codec/encoder contract differs"
                )
            shard_digest = str(row.values.get("shard_digest") or "")
            _validated_digest(
                shard_digest,
                field=f"sample {row.sample_id!r} shard_digest",
            )
            if (
                shard_digest != record["digest"]
                or int(row.values.get("shard_size", -1)) != record["bytes"]
                or row.values.get("shard_md5_b64") != record["md5_b64"]
                or row.values.get("shard_generation") != f"local-digest:{shard_digest}"
            ):
                raise DataContractError(
                    f"LTX sample {row.sample_id!r} local shard identity differs"
                )
            row_counts[row.shard] += 1
            published_by_id[row.sample_id] = _restore_source_logical_fields(
                row.values,
                expected,
            )
        if any(
            row_counts[path] != int(record.get("samples", -1))
            for path, record in rank_shards.items()
        ):
            raise DataContractError(f"LTX rank {rank} shard sample counts differ")

        expected_files = {
            receipt_path.relative_to(root).as_posix(),
            index_relative,
            *rank_shards,
        }
        actual_files: set[str] = set()
        for path in rank_root.rglob("*"):
            if path.is_symlink():
                raise DataContractError(f"LTX rank {rank} subtree contains a symlink")
            if path.is_file():
                actual_files.add(path.relative_to(root).as_posix())
        if actual_files != expected_files:
            raise DataContractError(
                f"LTX rank {rank} private file inventory differs: "
                f"extra={sorted(actual_files - expected_files)}, "
                f"missing={sorted(expected_files - actual_files)}"
            )
        shard_records.extend(rank_shards.values())
        rank_receipts.append(
            {
                "rank": rank,
                "path": receipt_path.relative_to(root).as_posix(),
                "bytes": len(receipt_bytes),
                "digest": _digest_bytes(receipt_bytes),
                "index": index_relative,
                "index_bytes": index_size,
                "index_digest": index_digest,
                "samples": len(rank_rows),
                "shards": len(rank_shards),
            }
        )

    missing = [sample_id for sample_id in expected_by_id if sample_id not in published_by_id]
    if missing or len(published_by_id) != len(expected_rows):
        raise DataContractError(
            "LTX published sample coverage differs from the source index: "
            f"missing={missing[:8]}, published={len(published_by_id)}, "
            f"expected={len(expected_rows)}"
        )
    ordered_rows = [published_by_id[row.sample_id] for row in expected_rows]
    corpus_index_digest = write_index(root / LTX25_CORPUS_INDEX_PATH, ordered_rows)
    corpus_index_bytes = (root / LTX25_CORPUS_INDEX_PATH).stat().st_size
    encoder_contract_digest = _atomic_json(
        root / LTX25_ENCODER_CONTRACT_PATH,
        encoder_contract.as_dict(),
    )
    if encoder_contract_digest != encoder_contract.digest:
        raise DataContractError("LTX serialized encoder contract content digest differs")

    source_path = Path(source_index_path).expanduser().resolve()
    source_size, source_digest, _source_md5 = _file_identity(source_path)
    ordered_ids_digest = _ordered_sample_id_digest(ordered_rows)
    control = {
        "schema": "solarwm.ltx25.preencode-corpus-control.v1",
        "family": "ltx25_video",
        "format_version": encoder_contract.format_version,
        "provider_identity": provider_identity,
        "codec_identity": codec_identity,
        "codec_load_receipt_digest": codec_load_receipt_digest,
        "encoder_contract": LTX25_ENCODER_CONTRACT_PATH,
        "encoder_contract_digest": encoder_contract.digest,
        "source_index_name": source_path.name,
        "source_index_bytes": source_size,
        "source_index_digest": source_digest,
        "index": LTX25_CORPUS_INDEX_PATH,
        "index_bytes": corpus_index_bytes,
        "index_digest": corpus_index_digest,
        "samples": len(ordered_rows),
        "shards": len(shard_records),
        "world_size": world_size,
        "ordered_sample_id_digest": ordered_ids_digest,
        "rank_receipts": rank_receipts,
        "shard_records": sorted(shard_records, key=lambda item: str(item["path"])),
    }
    control_digest = _atomic_json(root / LTX25_CORPUS_CONTROL_PATH, control)
    complete = {
        "schema": "solarwm.ltx25.preencode-complete.v1",
        "family": "ltx25_video",
        "samples": len(ordered_rows),
        "shards": len(shard_records),
        "world_size": world_size,
        "index": LTX25_CORPUS_INDEX_PATH,
        "index_bytes": corpus_index_bytes,
        "index_digest": corpus_index_digest,
        "encoder_contract": LTX25_ENCODER_CONTRACT_PATH,
        "encoder_contract_digest": encoder_contract.digest,
        "corpus_control": LTX25_CORPUS_CONTROL_PATH,
        "corpus_control_digest": control_digest,
        "ordered_sample_id_digest": ordered_ids_digest,
    }
    complete_digest = _atomic_json(root / LTX25_COMPLETE_PATH, complete)
    publish_directory_no_replace(
        root,
        destination,
        error_type=DataContractError,
        label="LTX preencode output",
    )
    return {
        **complete,
        "complete_digest": complete_digest,
        "output_root": str(destination),
    }


__all__ = [
    "LTX25_COMPLETE_PATH",
    "LTX25_CORPUS_CONTROL_PATH",
    "LTX25_CORPUS_INDEX_PATH",
    "LTX25_ENCODER_CONTRACT_PATH",
    "create_staging",
    "finalize_local_preencode",
    "rank_prefix",
    "rank_shard_relative",
    "write_rank_publication",
]
