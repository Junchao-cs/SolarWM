from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from solar.data.index import (
    IndexRow,
    ensure_disjoint,
    inventory,
    read_index,
    resolve_index_path,
    select_index_rows,
)
from solar.errors import DataContractError


def _write(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def test_order_and_repeats_are_authoritative(tmp_path: Path) -> None:
    path = tmp_path / "index.jsonl.gz"
    _write(
        path,
        [
            {"sample_id": "a", "key": "sample-a", "shard": "data/a.tar", "epoch_repeats": 6},
            {"sample_id": "b", "key": "sample-b", "shard": "data/b.tar"},
        ],
    )
    rows = read_index(path)
    report = inventory(path, rows)
    assert [row.sample_id for row in rows] == ["a", "b"]
    assert [row.key for row in rows] == ["sample-a", "sample-b"]
    assert report.virtual_occurrences == 7
    assert len(report.ordered_row_digest) == 64


@pytest.mark.parametrize(
    "shard",
    [
        "gs://bucket/key.tar",
        "file:/absolute/key.tar",
        "urn:solarwm:shard",
        "/abs/key.tar",
        "//host/key.tar",
        "../key.tar",
        "data/../key.tar",
        "data/./key.tar",
        "data//key.tar",
        "data/key.tar/",
        "a\\b.tar",
    ],
)
def test_published_shards_must_be_relative(tmp_path: Path, shard: str) -> None:
    path = tmp_path / "index.jsonl.gz"
    _write(path, [{"sample_id": "a", "key": "a", "shard": shard}])
    with pytest.raises(DataContractError, match="shard"):
        read_index(path)


@pytest.mark.parametrize("field", ["sample_id", "key", "shard"])
@pytest.mark.parametrize("value", [None, "", 7, True, ["value"], {"value": "x"}])
def test_authoritative_index_identities_must_be_strict_strings(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path = tmp_path / f"invalid-{field}.jsonl.gz"
    row: dict[str, object] = {
        "sample_id": "sample",
        "key": "key",
        "shard": "data/shard.tar",
    }
    row[field] = value
    _write(path, [row])
    with pytest.raises(DataContractError, match=field):
        read_index(path)


@pytest.mark.parametrize("field", ["sample_id", "key", "shard"])
def test_authoritative_index_identities_may_not_be_omitted(
    tmp_path: Path,
    field: str,
) -> None:
    path = tmp_path / f"missing-{field}.jsonl.gz"
    row = {"sample_id": "sample", "key": "key", "shard": "data/shard.tar"}
    row.pop(field)
    _write(path, [row])
    with pytest.raises(DataContractError, match=field):
        read_index(path)


def test_train_test_overlap_fails(tmp_path: Path) -> None:
    train_path = tmp_path / "train.jsonl.gz"
    test_path = tmp_path / "test.jsonl.gz"
    _write(train_path, [{"sample_id": "same", "key": "same", "shard": "a.tar"}])
    _write(test_path, [{"sample_id": "same", "key": "same", "shard": "b.tar"}])
    with pytest.raises(DataContractError, match="overlap"):
        ensure_disjoint(read_index(train_path), read_index(test_path))


def test_shard_identity_must_be_complete_and_consistent(tmp_path: Path) -> None:
    partial = tmp_path / "partial.jsonl.gz"
    _write(
        partial,
        [
            {
                "sample_id": "a",
                "key": "a",
                "shard": "s.tar",
                "shard_generation": "1",
            }
        ],
    )
    with pytest.raises(DataContractError, match="partial shard identity"):
        read_index(partial)

    drift = tmp_path / "drift.jsonl.gz"
    rows = [
        {
            "sample_id": sample,
            "key": sample,
            "shard": "s.tar",
            "shard_generation": "1",
            "shard_size": 10,
            "shard_md5_b64": md5,
        }
        for sample, md5 in (
            ("a", "AAAAAAAAAAAAAAAAAAAAAA=="),
            ("b", "BBBBBBBBBBBBBBBBBBBBBB=="),
        )
    ]
    _write(drift, rows)
    with pytest.raises(DataContractError, match="identity drift"):
        read_index(drift)


def test_digest_is_part_of_identity_and_cannot_hide_as_a_partial_field(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "digest-only-partial.jsonl.gz"
    _write(
        partial,
        [{"sample_id": "a", "key": "a", "shard": "s.tar", "shard_digest": "a" * 64}],
    )
    with pytest.raises(DataContractError, match="partial shard identity"):
        read_index(partial)

    complete = tmp_path / "size-and-digest.jsonl.gz"
    _write(
        complete,
        [
            {
                "sample_id": "a",
                "key": "a",
                "shard": "s.tar",
                "shard_size": 10,
                "shard_digest": "a" * 64,
            }
        ],
    )
    assert read_index(complete)[0].sample_id == "a"

    mixed = tmp_path / "mixed-identity.jsonl.gz"
    _write(
        mixed,
        [
            {"sample_id": "a", "key": "a", "shard": "s.tar"},
            {
                "sample_id": "b",
                "key": "b",
                "shard": "s.tar",
                "shard_size": 10,
                "shard_digest": "a" * 64,
            },
        ],
    )
    with pytest.raises(DataContractError, match="identity drift"):
        read_index(mixed)


def test_generation_and_size_do_not_require_a_runtime_digest(tmp_path: Path) -> None:
    path = tmp_path / "generation-size.jsonl.gz"
    _write(
        path,
        [
            {
                "sample_id": "a",
                "key": "a",
                "shard": "s.tar",
                "shard_generation": "1",
                "shard_size": 10,
            }
        ],
    )

    assert read_index(path)[0].sample_id == "a"


@pytest.mark.parametrize("value", [True, False, "2", 2.0, None, [2]])
def test_epoch_repeats_requires_an_exact_integer(tmp_path: Path, value: object) -> None:
    path = tmp_path / "invalid-repeats.jsonl.gz"
    _write(
        path,
        [{"sample_id": "a", "key": "a", "shard": "a.tar", "epoch_repeats": value}],
    )
    with pytest.raises(DataContractError, match="integer"):
        read_index(path)


@pytest.mark.parametrize("value", [0, -1])
def test_epoch_repeats_must_be_positive(tmp_path: Path, value: int) -> None:
    path = tmp_path / "non-positive-repeats.jsonl.gz"
    _write(
        path,
        [{"sample_id": "a", "key": "a", "shard": "a.tar", "epoch_repeats": value}],
    )
    with pytest.raises(DataContractError, match=">= 1"):
        read_index(path)


def test_index_control_root_is_independent_from_shard_transport(tmp_path: Path) -> None:
    relative = "indexes/dataset/train-index.jsonl.gz"
    local = {
        "train_index": relative,
        "transport": {"kind": "local", "root": str(tmp_path / "payloads")},
    }
    bucket = {
        "train_index": relative,
        "index_root": str(tmp_path / "controls"),
        "transport": {"kind": "gcs", "root": "gs://example-bucket"},
    }

    assert resolve_index_path(local, "train_index") == tmp_path / "payloads" / relative
    assert resolve_index_path(bucket, "train_index") == tmp_path / "controls" / relative


def test_validation_subset_is_seeded_from_the_recipe_test_index() -> None:
    rows = tuple(
        IndexRow.from_mapping(
            index,
            {"sample_id": f"sample-{index}", "key": f"key-{index}", "shard": "data.tar"},
        )
        for index in range(20)
    )
    first = select_index_rows(rows, sample_count=6, seed=42)
    repeated = select_index_rows(rows, sample_count=6, seed=42)
    changed = select_index_rows(rows, sample_count=6, seed=43)
    assert [row.sample_id for row in first] == [row.sample_id for row in repeated]
    assert [row.sample_id for row in first] != [row.sample_id for row in changed]
    assert len({row.sample_id for row in first}) == 6


@pytest.mark.parametrize(("count", "seed"), ((0, 42), (21, 42), (2, -1)))
def test_validation_subset_rejects_invalid_selection(count: int, seed: int) -> None:
    rows = tuple(
        IndexRow.from_mapping(
            index,
            {"sample_id": f"sample-{index}", "key": f"key-{index}", "shard": "data.tar"},
        )
        for index in range(20)
    )
    with pytest.raises(DataContractError):
        select_index_rows(rows, sample_count=count, seed=seed)


def test_bucket_index_requires_staged_local_control_root() -> None:
    with pytest.raises(DataContractError, match="index_root"):
        resolve_index_path(
            {
                "train_index": "indexes/train.jsonl.gz",
                "transport": {"kind": "gcs", "root": "gs://example-bucket"},
            },
            "train_index",
        )
