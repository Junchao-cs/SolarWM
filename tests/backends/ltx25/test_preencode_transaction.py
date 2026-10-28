from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from solar.backends.ltx25 import preencode_transaction as transaction
from solar.backends.ltx25.backend import _preencode_expected_rows
from solar.backends.ltx25.torch_raw import RawIndexedStream
from solar.data import read_index
from solar.errors import DataContractError
from solar.preencode import (
    EncodedPayload,
    EncoderContract,
    TensorSpec,
    write_index,
    write_shard,
)

PROVIDER = "test-provider"
CODEC = "test-codec"
CODEC_RECEIPT = "a" * 64


def _contract() -> EncoderContract:
    return EncoderContract(
        schema="solarwm.encoder.v1",
        family="ltx25_video",
        format_version="test-ltx.v1",
        pixel_frames=2,
        latent_frames=1,
        height=2,
        width=2,
        camera_convention="relative_w2c+normalized_K",
        tensors=(TensorSpec("latent", (1,), "float32"),),
    )


def _source_row(
    sample_id: str,
    *,
    ordinal: int,
    repeats: int,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "key": f"key-{sample_id}",
        "shard": f"raw/source-{ordinal}.tar",
        "epoch_repeats": repeats,
        "start_frame": ordinal,
        "source_frame_indices": [ordinal, ordinal + 1],
        "num_frames": 20 + ordinal,
        "fps": 24.0,
    }


def _publish_rank(
    staging: Path,
    *,
    rank: int,
    world_size: int,
    sample_id: str,
    start_frame: int,
    contract: EncoderContract,
) -> None:
    payload = EncodedPayload(
        sample_id=sample_id,
        key=f"key-{sample_id}",
        source_sample_id=sample_id,
        start_frame=start_frame,
        source_frame_indices=(start_frame, start_frame + 1),
        encoder_contract_digest=contract.digest,
        members={"payload.bin": sample_id.encode()},
        metadata={"codec_identity": CODEC, "source_fps": 24.0},
    )
    receipt = write_shard(
        staging,
        transaction.rank_shard_relative(rank, 0),
        (payload,),
    )
    rows = []
    for raw in receipt.rows:
        row = dict(raw)
        row.update(
            {
                "shard_generation": f"local-digest:{receipt.digest}",
                "num_frames": start_frame + 2,
                "fps": 24.0,
            }
        )
        rows.append(row)
    transaction.write_rank_publication(
        staging,
        rank=rank,
        world_size=world_size,
        rows=rows,
        shards=(receipt,),
        index_relative_path="index.jsonl",
        provider_identity=PROVIDER,
        codec_identity=CODEC,
        codec_load_receipt_digest=CODEC_RECEIPT,
        encoder_contract_digest=contract.digest,
    )


def _source_index(tmp_path: Path) -> tuple[Path, tuple[Any, ...]]:
    path = tmp_path / "private" / "source.jsonl"
    # Deliberately order b before a. Rank order must not control corpus order.
    write_index(
        path,
        (
            _source_row("b", ordinal=1, repeats=4),
            _source_row("a", ordinal=0, repeats=2),
        ),
    )
    return path, read_index(path)


def _finalize(
    staging: Path,
    target: Path,
    source_path: Path,
    rows: tuple[Any, ...],
    contract: EncoderContract,
) -> dict[str, Any]:
    return dict(
        transaction.finalize_local_preencode(
            staging,
            target,
            expected_rows=rows,
            source_index_path=source_path,
            world_size=2,
            provider_identity=PROVIDER,
            codec_identity=CODEC,
            codec_load_receipt_digest=CODEC_RECEIPT,
            encoder_contract=contract,
        )
    )


def test_two_rank_merge_preserves_source_order_repeats_and_portability(
    tmp_path: Path,
) -> None:
    contract = _contract()
    source_path, source_rows = _source_index(tmp_path)
    target = tmp_path / "published"
    staging = transaction.create_staging(target)
    _publish_rank(
        staging,
        rank=0,
        world_size=2,
        sample_id="a",
        start_frame=0,
        contract=contract,
    )
    _publish_rank(
        staging,
        rank=1,
        world_size=2,
        sample_id="b",
        start_frame=1,
        contract=contract,
    )

    complete = _finalize(staging, target, source_path, source_rows, contract)

    published = read_index(target / transaction.LTX25_CORPUS_INDEX_PATH)
    assert [row.sample_id for row in published] == ["b", "a"]
    assert [row.epoch_repeats for row in published] == [4, 2]
    assert [row.values["num_frames"] for row in published] == [21, 20]
    assert len(published) == 2  # repeats weight scheduling, not physical encoding
    assert all(str(row.values["shard_generation"]).startswith("local-digest:") for row in published)
    control_bytes = (target / transaction.LTX25_CORPUS_CONTROL_PATH).read_bytes()
    control = json.loads(control_bytes)
    assert control["source_index_name"] == source_path.name
    assert str(source_path).encode() not in control_bytes
    assert complete["samples"] == 2
    assert (target / transaction.LTX25_COMPLETE_PATH).is_file()


def test_physical_preencode_stream_does_not_expand_epoch_repeats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = tmp_path / "physical-source.jsonl"
    write_index(
        source_path,
        tuple(
            {
                "sample_id": sample_id,
                "key": f"key-{sample_id}",
                "shard": f"raw/{sample_id}.tar",
                "epoch_repeats": repeats,
                "start_frame": start,
                "source_frame_indices": list(range(start, start + 153)),
                "num_frames": start + 200,
                "fps": 24.0,
            }
            for sample_id, repeats, start in (("a", 4, 0), ("b", 2, 1))
        ),
    )
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    stream = RawIndexedStream(
        {
            "data": {
                "index_root": str(source_path.parent),
                "index": source_path.name,
                "transport": {
                    "kind": "local",
                    "root": str(tmp_path / "raw-payloads"),
                },
                "seed": 7,
                "num_workers": 1,
                "shuffle_buffer": 1,
                "partition_mode": "global_occurrence",
            }
        },
        logical_dp=False,
        physical_once=True,
    )
    try:
        assert [row.epoch_repeats for row in stream.rows] == [4, 2]
        assert len(stream.streams) == 1
        assert len(stream.streams[0].plans) == 2
        assert {plan.repeat_ordinal for plan in stream.streams[0].plans} == {0}
    finally:
        stream.close()


def test_preencode_finalizer_uses_the_raw_stream_window_policy(tmp_path: Path) -> None:
    source_path = tmp_path / "generic-raw.jsonl"
    write_index(
        source_path,
        (
            {
                "sample_id": "generic",
                "key": "generic",
                "shard": "raw/generic.tar",
                "num_frames": 200,
                "fps": 24.0,
            },
        ),
    )

    expected = _preencode_expected_rows(source_path)

    assert expected[0].values["start_frame"] == 0
    assert expected[0].values["source_frame_indices"] == list(range(153))


def test_duplicate_cross_rank_sample_fails_before_corpus_complete(tmp_path: Path) -> None:
    contract = _contract()
    source_path, source_rows = _source_index(tmp_path)
    target = tmp_path / "published"
    staging = transaction.create_staging(target)
    for rank in range(2):
        _publish_rank(
            staging,
            rank=rank,
            world_size=2,
            sample_id="a",
            start_frame=0,
            contract=contract,
        )
    with pytest.raises(DataContractError, match="duplicate LTX sample_id across ranks"):
        _finalize(staging, target, source_path, source_rows, contract)
    assert not target.exists()
    assert not (staging / transaction.LTX25_COMPLETE_PATH).exists()


def test_rank_receipt_rejects_non_hex_digest(tmp_path: Path) -> None:
    contract = _contract()
    staging = transaction.create_staging(tmp_path / "published")
    payload = EncodedPayload(
        sample_id="a",
        key="key-a",
        source_sample_id="a",
        start_frame=0,
        source_frame_indices=(0, 1),
        encoder_contract_digest=contract.digest,
        members={"payload.bin": b"a"},
        metadata={"codec_identity": CODEC},
    )
    receipt = write_shard(
        staging,
        transaction.rank_shard_relative(0, 0),
        (payload,),
    )
    with pytest.raises(DataContractError, match="lowercase hex content digest"):
        transaction.write_rank_publication(
            staging,
            rank=0,
            world_size=1,
            rows=receipt.rows,
            shards=(receipt,),
            index_relative_path="index.jsonl",
            provider_identity=PROVIDER,
            codec_identity=CODEC,
            codec_load_receipt_digest="g" * 64,
            encoder_contract_digest=contract.digest,
        )


def test_last_moment_target_is_never_overwritten(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    source_path, source_rows = _source_index(tmp_path)
    target = tmp_path / "published"
    staging = transaction.create_staging(target)
    _publish_rank(
        staging,
        rank=0,
        world_size=2,
        sample_id="a",
        start_frame=0,
        contract=contract,
    )
    _publish_rank(
        staging,
        rank=1,
        world_size=2,
        sample_id="b",
        start_frame=1,
        contract=contract,
    )
    real_publish = transaction.publish_directory_no_replace

    def race(source: Path, destination: Path, **kwargs: Any) -> None:
        destination.mkdir()
        (destination / "owner.txt").write_text("other writer")
        real_publish(source, destination, **kwargs)

    monkeypatch.setattr(transaction, "publish_directory_no_replace", race)
    with pytest.raises(DataContractError, match="appeared during publication"):
        _finalize(staging, target, source_path, source_rows, contract)
    assert (target / "owner.txt").read_text() == "other writer"
    assert not (target / transaction.LTX25_COMPLETE_PATH).exists()
    assert (staging / transaction.LTX25_COMPLETE_PATH).is_file()
