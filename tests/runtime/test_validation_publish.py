from __future__ import annotations

import json
from pathlib import Path

import pytest

from solarwm.cli import main
from solarwm.runtime.gcs_upload import ObjectInfo, PublicationError
from solarwm.runtime.validation_publish import ValidationPublisher

PASS = "validation/step_000200_live"


class MemoryStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.writes: list[str] = []
        self.fail: str | None = None

    def put_file(self, key, path, expected) -> None:
        value = path.read_bytes()
        assert ObjectInfo.from_bytes(value) == expected
        self.put_bytes(key, value)

    def put_bytes(self, key, payload) -> None:
        if key == self.fail:
            raise PublicationError("injected_transport_failure", retryable=True)
        if key in self.objects:
            if self.objects[key] != payload:
                raise PublicationError("object_content_conflict")
            return
        self.objects[key] = payload
        self.writes.append(key)

    def read(self, key):
        return self.objects[key]

    def list(self, prefix):
        return {
            key: ObjectInfo.from_bytes(value)
            for key, value in self.objects.items()
            if key.startswith(prefix)
        }


def partition(root: Path, slots, *, total=10, failures=(), pass_dir=PASS) -> Path:
    directory = root / pass_dir
    directory.mkdir(parents=True)
    for slot in slots:
        video = directory / f"compare/rank{slot:03d}_sample.mp4"
        video.parent.mkdir(exist_ok=True)
        video.write_bytes(b"synthetic-video-" + str(slot).encode())
        manifest = directory / f"manifests/rank_{slot:03d}.json"
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_text(
            json.dumps(
                {
                    "logical_validation_rank": slot,
                    "step": 200,
                    "active_pass_name": pass_dir.rsplit("_", 1)[-1],
                    "compare_mp4": video.relative_to(root).as_posix(),
                    "artifact_valid": True,
                    "error": None,
                }
            )
        )
    for slot in failures:
        error = directory / f"errors/rank_{slot:03d}.json"
        error.parent.mkdir(exist_ok=True)
        error.write_text(
            json.dumps(
                {
                    "logical_validation_rank": slot,
                    "step": 200,
                    "active_pass_name": "live",
                    "error": "synthetic_decode_failure",
                }
            )
        )
    (directory / "COMPLETE.json").write_text(
        json.dumps(
            {
                "schema": "solarwm.validation-comparison.v1",
                "step": 200,
                "pass": pass_dir.rsplit("_", 1)[-1],
                "compare_layout": "ground_truth_left_generated_right",
                "local_slots": len(slots),
                "global_slots": total,
                "failed_slots": list(failures),
            }
        )
    )
    return directory


def publisher(root, store, rank=0, nodes=1):
    return ValidationPublisher(root, store, node_rank=rank, num_nodes=nodes)


def test_eight_nodes_ten_outputs_empty_partitions_and_late_receipts(tmp_path):
    store = MemoryStore()
    assignments = [range(4), range(4, 8), range(8, 10), [], [], [], [], []]
    publishers = []
    for rank, slots in enumerate(assignments):
        root = tmp_path / f"node-{rank}"
        partition(root, slots)
        publishers.append(publisher(root, store, rank, 8))
    assert publishers[0].scan()["pending"] == 1
    assert f"{PASS}/COMPLETE.json" not in store.objects
    for worker in publishers[1:]:
        assert worker.scan()["errors"] == 0
    assert f"{PASS}/COMPLETE.json" not in store.objects
    assert publishers[0].scan()["pending"] == 0
    complete = json.loads(store.objects[f"{PASS}/COMPLETE.json"])
    assert complete["global_slots"] == complete["successful_slots"] == 10
    assert complete["status"] == "complete"
    assert sum("/compare/" in key for key in store.objects) == 10
    assert sum("/manifests/" in key for key in store.objects) == 10
    assert sum("/receipts/" in key for key in store.objects) == 8
    assert store.writes[-1] == f"{PASS}/COMPLETE.json"
    assert not any(key.startswith("nodes/") for key in store.objects)
    writes = store.writes.copy()
    for rank in range(8):
        assert publisher(tmp_path / f"node-{rank}", store, rank, 8).scan()["errors"] == 0
    assert writes == store.writes
    receipt = store.objects[f"{PASS}/receipts/node_000.json"]
    assert str(tmp_path).encode() not in receipt


def test_upload_failure_preserves_other_pass_and_can_resume(tmp_path):
    store = MemoryStore()
    partition(tmp_path, [0], total=1)
    partition(tmp_path, [0], total=1, pass_dir=PASS.replace("live", "ema"))
    store.fail = f"{PASS}/compare/rank000_sample.mp4"
    worker = publisher(tmp_path, store)
    assert worker.scan()["errors"] == 1
    assert f"{PASS}/receipts/node_000.json" not in store.objects
    assert f"{PASS}/COMPLETE.json" not in store.objects
    assert f"{PASS.replace('live', 'ema')}/COMPLETE.json" in store.objects
    store.fail = None
    assert worker.scan()["errors"] == 0
    assert f"{PASS}/COMPLETE.json" in store.objects


@pytest.mark.parametrize("corruption", ["missing", "corrupt", "extra", "duplicate", "wrong_step"])
def test_global_completion_rejects_bad_inventory(tmp_path, corruption):
    store = MemoryStore()
    first, second = tmp_path / "first", tmp_path / "second"
    partition(first, [0], total=2)
    partition(second, [1], total=2)
    coordinator = publisher(first, store, 0, 2)
    coordinator.scan()
    publisher(second, store, 1, 2).scan()
    video = f"{PASS}/compare/rank001_sample.mp4"
    if corruption == "missing":
        del store.objects[video]
    elif corruption == "corrupt":
        store.objects[video] = b"different-video"
    elif corruption == "extra":
        store.objects[f"{PASS}/compare/rank099_extra.mp4"] = b"extra"
    else:
        key = f"{PASS}/receipts/node_001.json"
        receipt = json.loads(store.objects[key])
        if corruption == "wrong_step":
            receipt["step"] = 400
        else:
            receipt["successful_slots"] = [0]
        store.objects[key] = json.dumps(receipt).encode()
    assert coordinator.scan()["errors"] == 1
    assert f"{PASS}/COMPLETE.json" not in store.objects


def test_partial_sample_failures_are_not_reported_as_success(tmp_path):
    store = MemoryStore()
    first, second = tmp_path / "first", tmp_path / "second"
    partition(first, [0], total=2)
    partition(second, [], total=2, failures=[1])
    publisher(second, store, 1, 2).scan()
    report = publisher(first, store, 0, 2).scan()
    assert report["partial"] == 1
    complete = json.loads(store.objects[f"{PASS}/COMPLETE.json"])
    assert complete["status"] == "partial"
    assert complete["failed_slots"] == [1]


@pytest.mark.parametrize("kind", ["video", "manifest", "group", "step"])
def test_symlinks_cannot_export_files_outside_run(tmp_path, kind):
    root, outside = tmp_path / "run", tmp_path / "outside"
    directory = partition(root, [0], total=1)
    target = {
        "video": directory / "compare/rank000_sample.mp4",
        "manifest": directory / "manifests/rank_000.json",
        "group": directory / "compare",
        "step": directory,
    }[kind]
    target.rename(outside)
    target.symlink_to(outside)
    store = MemoryStore()
    assert publisher(root, store).scan()["errors"] == 1
    assert not store.objects


def test_allowlist_excludes_configs_logs_staging_and_unknown_files(tmp_path):
    directory = partition(tmp_path, [0], total=1)
    for name in ["config.json", "train.log", ".env", "upload-session.json"]:
        (tmp_path / name).write_text("synthetic-private-data")
        (directory / name).write_text("synthetic-private-data")
        (directory / "manifests" / name).write_text("synthetic-private-data")
    staging = tmp_path / "validation/.staging/run"
    staging.mkdir(parents=True)
    (staging / "COMPLETE.json").write_text("{}")
    store = MemoryStore()
    assert publisher(tmp_path, store).scan()["errors"] == 0
    assert len(store.objects) == 4
    assert not any(b"synthetic-private-data" in data for data in store.objects.values())


@pytest.mark.parametrize("field", ["access_token", "private_key", "authorization", "session_url"])
def test_credentials_in_manifest_block_publication_without_echo(tmp_path, field):
    directory = partition(tmp_path, [0], total=1)
    path = directory / "manifests/rank_000.json"
    value = json.loads(path.read_text())
    value["metadata"] = {field: "synthetic-credential-must-not-appear"}
    path.write_text(json.dumps(value))
    store = MemoryStore()
    report = publisher(tmp_path, store).scan()
    assert report["errors"] == 1
    assert report["partitions"][0]["error"] == "credential_in_publication_json"
    assert "synthetic-credential" not in json.dumps(report)
    assert not store.objects


def test_same_key_different_contents_is_a_conflict(tmp_path):
    partition(tmp_path, [0], total=1)
    store = MemoryStore()
    store.objects[f"{PASS}/compare/rank000_sample.mp4"] = b"previous-video"
    assert publisher(tmp_path, store).scan()["errors"] == 1
    assert store.objects[f"{PASS}/compare/rank000_sample.mp4"] == b"previous-video"
    assert f"{PASS}/COMPLETE.json" not in store.objects


def test_local_marker_is_required_and_cannot_change_after_upload(tmp_path):
    directory = partition(tmp_path, [0], total=2)
    marker = directory / "COMPLETE.json"
    payload = marker.read_bytes()
    marker.unlink()
    store = MemoryStore()
    worker = publisher(tmp_path, store, 0, 2)
    assert worker.scan()["pending"] == 1
    assert not store.objects
    marker.write_bytes(payload)
    worker.scan()
    marker.write_bytes(payload + b" ")
    assert worker.scan()["errors"] == 1


def test_cli_once_exit_codes_and_no_gpu_requirement(tmp_path, monkeypatch, capsys):
    store = MemoryStore()
    monkeypatch.setattr("solarwm.runtime.gcs_upload.GCSStore", lambda _destination: store)
    args = [
        "publish-validation",
        "--run-dir",
        str(tmp_path),
        "--destination",
        "gs://example-bucket/example-run",
        "--node-rank",
        "0",
        "--num-nodes",
        "1",
    ]
    assert main(args) == 1
    partition(tmp_path, [0], total=1)
    assert main(args) == 0
    store.objects[f"{PASS}/compare/rank000_sample.mp4"] = b"conflict"
    assert main(args) == 2
    output = capsys.readouterr().out
    assert str(tmp_path) not in output
    assert "example-bucket" not in output


def test_cli_watch_retries_without_repeating_status(tmp_path, monkeypatch, capsys):
    store = MemoryStore()
    monkeypatch.setattr("solarwm.runtime.gcs_upload.GCSStore", lambda _destination: store)
    calls = []

    def sleep(_seconds):
        calls.append(1)
        if len(calls) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr("time.sleep", sleep)
    assert (
        main(
            [
                "publish-validation",
                "--run-dir",
                str(tmp_path),
                "--destination",
                "gs://example-bucket/example-run",
                "--node-rank",
                "0",
                "--num-nodes",
                "1",
                "--watch",
            ]
        )
        == 130
    )
    assert capsys.readouterr().out.count('"pending"') == 1
