from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest

from solar.backends.wan22.runtime import publication
from solar.backends.wan22.runtime.publication import (
    preflight_camera_publication,
    publish_camera_triplets,
)
from solar.errors import BackendContractError
from solar.inference import GeneratedSample, InferenceCase, InferenceEngine
from solar.runtime.output_layout import CameraInferenceOutputLayout
from solar.runtime.serialization import canonical_json_bytes


class _Adapter:
    family = "wan22_ti2v_5b"

    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
        del weights_id
        c2w = np.repeat(np.eye(4, dtype=np.float64)[None], 960, axis=0)
        c2w[:, 0, 3] = float(case.slot)
        buffer = io.BytesIO()
        np.save(buffer, c2w, allow_pickle=False)
        return GeneratedSample(
            artifacts={
                "video.mp4": f"video-{case.slot}".encode(),
                "compare.mp4": f"compare-{case.slot}".encode(),
                "camera.npy": buffer.getvalue(),
                "schedule.json": b"{}\n",
            },
            shape=(1, 960, 3, 1, 1),
            dtype="bfloat16",
        )


def _case(slot: int, *, stem: str | None = None) -> InferenceCase:
    return InferenceCase(
        slot=slot,
        sample_id=f"sample-{slot}",
        prompt="prompt",
        start_frame=0,
        noise_seed=42 + slot,
        camera_fingerprint=f"{slot:064x}",
        metadata={
            "physical_dataset": "sekai_game-fix",
            "publish_stem": stem or f"clip-{slot}",
            "publication_pixel_frames": 960,
            "camera_publication_convention": "authoritative_absolute_c2w",
        },
    )


def _source_transaction(
    tmp_path: Path,
    cases: tuple[InferenceCase, ...],
) -> tuple[Path, CameraInferenceOutputLayout]:
    publish_root = tmp_path / "release"
    run_root = publish_root / "runs/run-1"
    generation_root = run_root / "generation"
    generation_root.mkdir(parents=True)
    InferenceEngine(_Adapter()).run(
        cases,
        weights_id="checkpoint#ema",
        output_dir=generation_root / "model",
    )
    (generation_root / "COMPLETE.json").write_bytes(
        canonical_json_bytes({"schema": "test-generation-complete.v1"})
    )
    return generation_root, CameraInferenceOutputLayout(
        publish_root=publish_root,
        run_root=run_root,
        run_id="run-1",
        layout="dataset_triplet_v1",
    )


def test_camera_publication_creates_exact_triplets_and_receipts(tmp_path: Path) -> None:
    cases = (_case(0), _case(1))
    generation_root, layout = _source_transaction(tmp_path, cases)

    summary = publish_camera_triplets(
        generation_root,
        layout,
        generation_pass="model",
        cases=cases,
        family="wan22_ti2v_5b",
        weights_id="checkpoint#ema",
        generation_complete_digest="a" * 64,
    )

    for slot in range(2):
        stem = f"clip-{slot}"
        assert (layout.publish_root / f"generate/sekai_game-fix/{stem}.mp4").read_bytes() == (
            f"video-{slot}".encode()
        )
        assert (layout.publish_root / f"compare/sekai_game-fix/{stem}.mp4").read_bytes() == (
            f"compare-{slot}".encode()
        )
        published_c2w = np.load(
            layout.publish_root / f"camera/sekai_game-fix/{stem}.npy",
            allow_pickle=False,
        )
        assert published_c2w.dtype == np.float64
        assert np.all(published_c2w[:, 0, 3] == float(slot))
    assert summary.cases == 2
    assert summary.complete_path == layout.run_root / "publication/COMPLETE.json"
    complete = json.loads(summary.complete_path.read_text(encoding="utf-8"))
    assert complete["ordered_manifest_digest"] == summary.ordered_manifest_digest
    assert sorted(path.name for path in (layout.run_root / "publication/samples").iterdir()) == [
        "slot-000000.json",
        "slot-000001.json",
    ]


def test_camera_publication_rejects_duplicate_or_existing_destinations(tmp_path: Path) -> None:
    layout = CameraInferenceOutputLayout(
        publish_root=tmp_path,
        run_root=tmp_path / "runs/run-1",
        run_id="run-1",
        layout="dataset_triplet_v1",
    )
    with pytest.raises(BackendContractError, match="destination is duplicated"):
        preflight_camera_publication(layout, (_case(0, stem="same"), _case(1, stem="same")))

    occupied = tmp_path / "generate/sekai_game-fix/clip-0.mp4"
    occupied.parent.mkdir(parents=True)
    occupied.write_bytes(b"old")
    with pytest.raises(BackendContractError, match="already exist"):
        preflight_camera_publication(layout, (_case(0),))
    assert occupied.read_bytes() == b"old"


def test_camera_publication_failure_never_writes_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = (_case(0),)
    generation_root, layout = _source_transaction(tmp_path, cases)
    real_link = publication.link_file_create_only
    calls = 0

    def fail_second(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise BackendContractError("injected link failure")
        real_link(*args, **kwargs)

    monkeypatch.setattr(publication, "link_file_create_only", fail_second)
    with pytest.raises(BackendContractError, match="injected link failure"):
        publish_camera_triplets(
            generation_root,
            layout,
            generation_pass="model",
            cases=cases,
            family="wan22_ti2v_5b",
            weights_id="checkpoint#ema",
            generation_complete_digest="a" * 64,
        )

    assert not (layout.run_root / "publication/COMPLETE.json").exists()
