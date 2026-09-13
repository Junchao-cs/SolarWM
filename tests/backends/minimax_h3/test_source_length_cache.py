from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from solarwm.backends.minimax_h3 import source_length_cache as cache


def _source(root: Path, payload: bytes):
    root.mkdir()
    members = {
        "video_member": "video.mp4",
        "camera_member": "camera.npz",
        "intrinsics_member": "K.npy",
        "manifest_member": "manifest.json",
    }
    with tarfile.open(root / "part.tar", "w") as archive:
        for name in members.values():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return {"sample_id": "dataset/case", "key": "case", "shard": "part.tar", **members}


def _args(tmp_path):
    return SimpleNamespace(
        dataset_root=str(tmp_path / "source"),
        work_dir=str(tmp_path / "cache"),
        config={"model": {"checkpoint_path": "/models/base-a"}, "inference": {}},
        weight_source="ema",
        fps_policy="source",
    )


def _prepared(tmp_path):
    args = _args(tmp_path)
    row = _source(Path(args.dataset_root), b"source video")
    case = cache.source_files(row, args)
    (case / "condition.safetensors").write_bytes(b"encoded condition")
    receipt = {
        "row": row,
        "preparation_identity": cache.preparation_identity(row, args),
        "condition_sha256": cache.file_sha256(case / "condition.safetensors"),
        "source_sha256": cache.file_sha256(case / "source.mp4"),
    }
    cache.atomic_json(case / "READY.json", receipt)
    return args, row, case, receipt


def test_same_size_source_from_another_root_is_not_reused(tmp_path):
    args = _args(tmp_path)
    row = _source(Path(args.dataset_root), b"AAAA")
    case = cache.source_files(row, args)
    args.dataset_root = str(tmp_path / "other-source")
    _source(Path(args.dataset_root), b"BBBB")
    with pytest.raises(ValueError, match="another dataset"):
        cache.source_files(row, args)
    assert (case / "source.mp4").read_bytes() == b"AAAA"


def test_members_without_source_receipt_are_replaced_even_at_same_size(tmp_path):
    args = _args(tmp_path)
    row = _source(Path(args.dataset_root), b"NEW!")
    case = Path(args.work_dir) / "cases/case"
    case.mkdir(parents=True)
    (case / "source.mp4").write_bytes(b"OLD!")
    cache.source_files(row, args)
    assert (case / "source.mp4").read_bytes() == b"NEW!"


def test_interrupted_copy_is_not_published_and_can_retry(tmp_path, monkeypatch):
    args = _args(tmp_path)
    row = _source(Path(args.dataset_root), b"complete payload")
    original = cache.shutil.copyfileobj

    def fail(source, output, length):
        output.write(source.read(3))
        raise OSError("injected interruption")

    monkeypatch.setattr(cache.shutil, "copyfileobj", fail)
    with pytest.raises(OSError, match="injected"):
        cache.source_files(row, args)
    case = Path(args.work_dir) / "cases/case"
    assert not (case / "source.mp4").exists()
    assert not (case / "SOURCE.json").exists()
    monkeypatch.setattr(cache.shutil, "copyfileobj", original)
    cache.source_files(row, args)
    assert (case / "source.mp4").read_bytes() == b"complete payload"


@pytest.mark.parametrize("change", ["root", "model", "condition", "video"])
def test_prepared_condition_rejects_changed_inputs(tmp_path, change):
    args, row, case, receipt = _prepared(tmp_path)
    assert cache.prepared_case_receipt(case, row, args) == receipt
    if change == "root":
        args.dataset_root = str(tmp_path / "another-root")
    elif change == "model":
        args.config["model"]["checkpoint_path"] = "/models/base-b"
    elif change == "condition":
        (case / "condition.safetensors").write_bytes(b"another condition")
    else:
        (case / "source.mp4").write_bytes(b"another video")
    with pytest.raises(ValueError):
        cache.prepared_case_receipt(case, row, args)


def test_result_reuse_requires_current_condition_and_all_outputs(tmp_path):
    args, _row, _case, condition = _prepared(tmp_path)
    inputs = dict(checkpoint_sha256="checkpoint", silence_sha256="silence", seed=42)
    identity = cache.result_identity(args, condition, **inputs)
    receipt = {"result_identity": identity}
    for suffix, field in {
        ".mp4": "output_sha256",
        ".compare.mp4": "compare_sha256",
        ".storyboard.jpg": "storyboard_sha256",
    }.items():
        path = tmp_path / ("case" + suffix)
        path.write_bytes(suffix.encode())
        receipt[field] = cache.file_sha256(path)
    manifest = tmp_path / "case.json"
    manifest.write_text(json.dumps(receipt))
    cache.verify_completed_result(manifest, expected_identity=identity, save_latents=False)
    changed = {**condition, "condition_sha256": "reencoded"}
    with pytest.raises(ValueError, match="another condition"):
        cache.verify_completed_result(
            manifest,
            expected_identity=cache.result_identity(args, changed, **inputs),
            save_latents=False,
        )
    with pytest.raises(ValueError, match="latents"):
        cache.verify_completed_result(manifest, expected_identity=identity, save_latents=True)
    (tmp_path / "case.storyboard.jpg").unlink()
    with pytest.raises(ValueError, match="storyboard"):
        cache.verify_completed_result(manifest, expected_identity=identity, save_latents=False)
