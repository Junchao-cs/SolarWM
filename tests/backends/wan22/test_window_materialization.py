from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from solar.backends.wan22.windows import (
    WINDOW_HASH_NAMESPACE_153F,
    expected_153f_window_start,
    write_wan153f_window_index,
)
from solar.data.index import read_index
from solar.errors import DataContractError


def _row(
    sample_id: str,
    *,
    dataset: str,
    role: str,
    num_frames: int,
    **updates: object,
) -> dict[str, object]:
    key = sample_id.replace("/", "__")
    value: dict[str, object] = {
        "sample_id": sample_id,
        "key": key,
        "shard": f"raw-wds/{dataset}/shards/kept-high-000000.tar",
        "epoch_repeats": 6,
        "dataset": dataset,
        "recipe_role": role,
        "num_frames": num_frames,
        "fps": 16.0,
        "video_member": f"{key}.video.mp4",
        "camera_member": f"{key}.camera.npz",
        "manifest_member": f"{key}.manifest.json",
        "kept_tier": "high",
    }
    value.update(updates)
    return value


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    raw = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows
    )
    path.write_bytes(gzip.compress(raw, mtime=0))


def _read_values(path: Path) -> list[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _authority_rows(
    source: dict[str, object],
    starts: list[int],
) -> list[dict[str, object]]:
    sample_id = str(source["sample_id"])
    key = str(source["key"])
    return [
        {
            "sample_id": f"{sample_id}/latent-153f-w{index:02d}",
            "key": f"{key}__latent153f_w{index:02d}",
            "shard": "latent-wds/authority/shards/kept-high-000000.tar",
            "start_frame": start,
            "source_frame_indices": list(range(start, start + 153)),
        }
        for index, start in enumerate(starts)
    ]


def test_materializer_combines_roles_and_freezes_release_window_policy(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl.gz"
    test = tmp_path / "test.jsonl.gz"
    train_windows = tmp_path / "train-windows.jsonl.gz"
    test_windows = tmp_path / "test-windows.jsonl.gz"
    train_row = _row("abot/train-source", dataset="abot", role="train", num_frames=960)
    test_row = _row("dl3dv-10s/test-source", dataset="dl3dv-10s", role="test", num_frames=200)
    _write(train, [train_row])
    _write(test, [test_row])
    long_starts = [0, 161, 323, 484, 646, 807]
    current_hash_start = expected_153f_window_start(
        {
            "source_sample_id": "dl3dv-10s/test-source",
            "source_dataset": "dl3dv-10s",
            "window_index": 0,
            "window_count": 1,
        },
        200,
    )
    authority_start = (current_hash_start + 1) % 48
    _write(train_windows, _authority_rows(train_row, long_starts))
    _write(test_windows, _authority_rows(test_row, [authority_start]))

    first = tmp_path / "first" / "preencode-window-index.jsonl.gz"
    second = tmp_path / "second" / "preencode-window-index.jsonl.gz"
    summary = write_wan153f_window_index(train, test, train_windows, test_windows, first)
    write_wan153f_window_index(train, test, train_windows, test_windows, second)

    assert first.read_bytes() == second.read_bytes()
    assert summary.train_sources == summary.test_sources == 1
    assert summary.train_windows == 6
    assert summary.test_windows == 1
    assert len(read_index(first)) == 7

    values = _read_values(first)
    assert [int(row["start_frame"]) for row in values[:6]] == long_starts
    for index, row in enumerate(values[:6]):
        assert row["sample_id"] == f"abot/train-source/latent-153f-w{index:02d}"
        assert row["key"] == f"abot__train-source__latent153f_w{index:02d}"
        assert row["source_sample_id"] == "abot/train-source"
        assert row["source_dataset"] == "abot"
        assert row["window_count"] == 6
        assert row["window_hash_namespace"] == WINDOW_HASH_NAMESPACE_153F
        assert row["split"] == "train"
        assert row["epoch_repeats"] == 1
        assert row["video_member"] == "abot__train-source.video.mp4"
        start = int(row["start_frame"])
        assert row["source_frame_indices"] == list(range(start, start + 153))

    ordinary = values[-1]
    assert ordinary["start_frame"] == authority_start
    assert ordinary["start_frame"] != current_hash_start
    assert ordinary["split"] == "test"


def test_materializer_is_create_only(tmp_path: Path) -> None:
    train = tmp_path / "train.jsonl.gz"
    test = tmp_path / "test.jsonl.gz"
    train_windows = tmp_path / "train-windows.jsonl.gz"
    test_windows = tmp_path / "test-windows.jsonl.gz"
    output = tmp_path / "windows.jsonl.gz"
    train_row = _row("abot/train", dataset="abot", role="train", num_frames=153)
    test_row = _row("abot/test", dataset="abot", role="test", num_frames=153)
    _write(train, [train_row])
    _write(test, [test_row])
    _write(train_windows, _authority_rows(train_row, [0] * 6))
    _write(test_windows, _authority_rows(test_row, [0] * 6))

    write_wan153f_window_index(train, test, train_windows, test_windows, output)
    original = output.read_bytes()
    with pytest.raises(DataContractError, match="already exists"):
        write_wan153f_window_index(train, test, train_windows, test_windows, output)
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    ("role", "num_frames", "updates", "message"),
    [
        ("test", 153, {}, "must declare only role 'train'"),
        ("train", 152, {}, "needs at least 153"),
        ("train", 153, {"start_frame": 0}, "already window-materialized"),
    ],
)
def test_materializer_rejects_invalid_raw_train_rows(
    tmp_path: Path,
    role: str,
    num_frames: int,
    updates: dict[str, object],
    message: str,
) -> None:
    train = tmp_path / "train.jsonl.gz"
    test = tmp_path / "test.jsonl.gz"
    train_windows = tmp_path / "train-windows.jsonl.gz"
    test_windows = tmp_path / "test-windows.jsonl.gz"
    train_row = _row("abot/train", dataset="abot", role=role, num_frames=num_frames, **updates)
    test_row = _row("abot/test", dataset="abot", role="test", num_frames=153)
    _write(
        train,
        [train_row],
    )
    _write(test, [test_row])
    _write(train_windows, _authority_rows(train_row, [0] * 6))
    _write(test_windows, _authority_rows(test_row, [0] * 6))

    with pytest.raises(DataContractError, match=message):
        write_wan153f_window_index(
            train,
            test,
            train_windows,
            test_windows,
            tmp_path / "windows.jsonl.gz",
        )
