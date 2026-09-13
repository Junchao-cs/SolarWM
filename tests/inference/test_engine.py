from __future__ import annotations

import json
from pathlib import Path

import pytest

import solarwm.inference.engine as inference_engine
from solarwm.errors import BackendContractError
from solarwm.inference import (
    GeneratedFile,
    GeneratedSample,
    InferenceCase,
    InferenceEngine,
    run_validation,
)


class FakeAdapter:
    family = "test_family"

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
        self.calls.append((case.slot, weights_id))
        return GeneratedSample(
            artifacts={"video.mp4": f"slot={case.slot}".encode()},
            shape=(1, 3, 81, 16, 16),
            dtype="bfloat16",
            metrics={"finite_fraction": 1.0},
        )


def _case(slot: int) -> InferenceCase:
    return InferenceCase(slot, f"sample-{slot}", "prompt", 3, 100 + slot, "cam-digest")


def test_inference_writes_ordered_identity_and_completion(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    summary = InferenceEngine(adapter).run(
        [_case(1), _case(0)], weights_id="weights-digest", output_dir=tmp_path / "infer"
    )
    assert summary.cases == 2
    records = [
        json.loads(line)
        for line in (summary.output_dir / "ordered-manifest.jsonl").read_text().splitlines()
    ]
    assert [record["case"]["slot"] for record in records] == [1, 0]
    assert all(record["weights_id"] == "weights-digest" for record in records)
    assert (summary.output_dir / "COMPLETE.json").is_file()


def test_validation_calls_the_same_adapter_and_engine(tmp_path: Path) -> None:
    adapter = FakeAdapter()
    summary = run_validation(
        adapter, [_case(0)], weights_id="ema-digest", output_dir=tmp_path / "validation"
    )
    assert adapter.calls == [(0, "ema-digest")]
    assert summary.weights_id == "ema-digest"


def test_collective_case_waves_keep_shorter_dp_partitions_synchronized(tmp_path: Path) -> None:
    waves: list[tuple[str | None, str]] = []
    adapter = FakeAdapter()

    summary = InferenceEngine(adapter).run(
        [_case(0)],
        weights_id="weights-digest",
        output_dir=tmp_path / "infer",
        collective_case_waves=2,
        collective_error=lambda error, phase: waves.append((error, phase)),
    )

    assert summary.cases == 1
    assert adapter.calls == [(0, "weights-digest")]
    assert (None, "case wave 1") in waves


def test_collective_case_waves_allow_an_empty_dp_partition(tmp_path: Path) -> None:
    waves: list[tuple[str | None, str]] = []
    adapter = FakeAdapter()

    summary = InferenceEngine(adapter).run(
        [],
        weights_id="weights-digest",
        output_dir=tmp_path / "infer",
        collective_case_waves=1,
        collective_error=lambda error, phase: waves.append((error, phase)),
    )

    assert summary.cases == 0
    assert adapter.calls == []
    assert (None, "case wave 0") in waves
    complete = json.loads((summary.output_dir / "COMPLETE.json").read_text())
    assert complete["cases"] == 0


def test_empty_inference_without_collective_waves_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(BackendContractError, match="requires cases"):
        InferenceEngine(FakeAdapter()).run(
            [],
            weights_id="weights-digest",
            output_dir=tmp_path / "infer",
        )


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(BackendContractError, match="already exists"):
        InferenceEngine(FakeAdapter()).run([_case(0)], weights_id="weights", output_dir=output)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "/absolute/video.mp4",
        "../video.mp4",
        "frames/../video.mp4",
        "./video.mp4",
        "frames/./video.mp4",
        "frames//video.mp4",
        "frames/",
        "frames\\video.mp4",
        "gs://bucket/video.mp4",
        "nul\x00video.mp4",
    ],
)
def test_generated_artifact_paths_must_be_raw_canonical_posix(name: str) -> None:
    with pytest.raises(BackendContractError, match="invalid generated artifact"):
        GeneratedSample(artifacts={name: b"video"}, shape=(1,), dtype="uint8")


@pytest.mark.parametrize("name", ["manifest.json", "manifest.json/payload.bin"])
def test_generated_artifacts_cannot_claim_the_sample_manifest(name: str) -> None:
    with pytest.raises(BackendContractError, match="reserved"):
        GeneratedSample(artifacts={name: b"not-the-manifest"}, shape=(1,), dtype="uint8")


def test_generated_artifact_paths_cannot_overlap_as_file_and_directory() -> None:
    with pytest.raises(BackendContractError, match="collide"):
        GeneratedSample(
            artifacts={"frames": b"file", "frames/000000.png": b"frame"},
            shape=(1,),
            dtype="uint8",
        )


def test_generated_artifact_accepts_a_canonical_nested_path(tmp_path: Path) -> None:
    class NestedAdapter(FakeAdapter):
        def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
            return GeneratedSample(
                artifacts={"frames/000000.png": b"frame"},
                shape=(1,),
                dtype="uint8",
            )

    summary = InferenceEngine(NestedAdapter()).run(
        [_case(0)], weights_id="weights", output_dir=tmp_path / "nested"
    )
    assert (summary.output_dir / "slot-000000/frames/000000.png").read_bytes() == b"frame"


def test_inference_links_large_generated_files_without_loading_them(
    tmp_path: Path,
) -> None:
    source = tmp_path / "generated.mp4"
    source.write_bytes(b"streamed-video")

    class FileAdapter(FakeAdapter):
        def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
            return GeneratedSample(
                artifacts={"video.mp4": GeneratedFile(source)},
                shape=(1, 1, 3, 1, 1),
                dtype="float32",
            )

    summary = InferenceEngine(FileAdapter()).run(
        [_case(0)], weights_id="weights", output_dir=tmp_path / "file-backed"
    )

    published = summary.output_dir / "slot-000000/video.mp4"
    assert published.read_bytes() == b"streamed-video"
    assert not source.exists()
    manifest = json.loads((published.parent / "manifest.json").read_text())
    assert manifest["artifacts"] == [
        {
            "bytes": len(b"streamed-video"),
            "digest": "b7dff0ced259c44af07974f373c50a2b4d5565e4be64d650ef7a6c31d9e54730",
            "path": "slot-000000/video.mp4",
        }
    ]


def test_inference_output_file_writes_are_create_only(tmp_path: Path) -> None:
    target = tmp_path / "artifact.bin"
    inference_engine._write_file(target, b"first")
    with pytest.raises(BackendContractError, match="already exists"):
        inference_engine._write_file(target, b"second")
    assert target.read_bytes() == b"first"


def test_output_race_winner_is_never_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "raced"
    real_publish = inference_engine.publish_directory_no_replace

    def race(source: Path, destination: Path, **kwargs: object) -> None:
        destination.mkdir()
        (destination / "owner.txt").write_bytes(b"race-winner")
        real_publish(source, destination, **kwargs)

    monkeypatch.setattr(inference_engine, "publish_directory_no_replace", race)
    with pytest.raises(BackendContractError, match="target appeared"):
        InferenceEngine(FakeAdapter()).run([_case(0)], weights_id="weights", output_dir=output)
    assert (output / "owner.txt").read_bytes() == b"race-winner"
