from __future__ import annotations

import io
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from solarwm.backends.wan22.generation import GenerationPass, resolve_generation_plan
from solarwm.backends.wan22.runtime import inference as wan_inference
from solarwm.backends.wan22.runtime.inference import (
    TrainingWanGenerationAdapter,
    WanGenerationSummary,
    _authoritative_publication_c2w,
    _camera_length_rollout_latents,
    _camera_publication_identity,
    _camera_publication_source_frame_indices,
    _camera_window,
    _checkpoint_file,
    _comparison_view_name,
    _flowmap_schedule,
    _merge_generation_partitions,
    _partition_generation_cases,
    _plan_case,
    _seed_for,
    _source_variable_rollout_latents,
    run_wan_generation,
    run_wan_inference,
    run_wan_validation,
)
from solarwm.config import load_config
from solarwm.data.index import IndexRow, select_index_rows
from solarwm.errors import BackendContractError, DataContractError
from solarwm.inference import (
    GeneratedSample,
    InferenceCase,
    InferenceEngine,
    publish_comparison_complete,
    publish_comparison_partition,
)


def test_anyflow_comparison_view_matches_reference_nfe_directory() -> None:
    generation_pass = GenerationPass(
        name="live_nfe4_autoregressive",
        weights="live",
        mode="autoregressive",
        solver="flowmap",
        num_inference_steps=4,
        rollout_latent_frames=60,
        min_rollout_latent_frames=21,
        fixed_plan_pixel_frames=237,
        variable_rollout_by_source=True,
    )
    assert _comparison_view_name(50, generation_pass) == (
        "step_000050_flowmap_nfe0004_live_nfe4_autoregressive"
    )


ROOT = Path(__file__).resolve().parents[3]


class FakeProvider:
    def __init__(self, family: str, *, fail_pass: str = "") -> None:
        self.family = family
        self.fail_pass = fail_pass
        self.calls: list[tuple[str, int, str]] = []

    def weight_id(self, role: str) -> str:
        return f"checkpoint-digest#{role}"

    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
        generation_pass = case.metadata["generation_pass"]
        name = str(generation_pass["name"])
        self.calls.append((name, case.slot, weights_id))
        if name == self.fail_pass:
            raise BackendContractError("injected generation failure")
        return GeneratedSample(
            artifacts={"video.mp4": f"{name}:{case.slot}".encode()},
            shape=(1, int(generation_pass["rollout_latent_frames"]), 3, 8, 8),
            dtype="float32",
            provenance={"solver": generation_pass["solver"]},
        )


class ComparisonFakeProvider(FakeProvider):
    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
        sample = super().generate(case, weights_id=weights_id)
        return GeneratedSample(
            artifacts={
                **sample.artifacts,
                "compare.mp4": f"compare:{case.slot}".encode(),
            },
            shape=sample.shape,
            dtype=sample.dtype,
            provenance=sample.provenance,
        )


class CameraPublicationFakeProvider(FakeProvider):
    def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
        sample = super().generate(case, weights_id=weights_id)
        buffer = io.BytesIO()
        np.save(
            buffer,
            np.repeat(np.eye(4, dtype=np.float64)[None], 960, axis=0),
            allow_pickle=False,
        )
        return GeneratedSample(
            artifacts={
                **sample.artifacts,
                "compare.mp4": f"compare:{case.slot}".encode(),
                "camera.npy": buffer.getvalue(),
            },
            shape=(1, 960, 3, 8, 8),
            dtype=sample.dtype,
            provenance=sample.provenance,
        )


def _case() -> InferenceCase:
    return InferenceCase(
        slot=0,
        sample_id="sample-0",
        prompt="move forward",
        start_frame=17,
        noise_seed=1234,
        camera_fingerprint="a" * 64,
    )


def _slotted_case(slot: int) -> InferenceCase:
    return InferenceCase(
        slot=slot,
        sample_id=f"sample-{slot}",
        prompt="move forward",
        start_frame=slot,
        noise_seed=1200 + slot,
        camera_fingerprint=f"{slot:064x}",
    )


def _config(relative: str) -> dict:
    return load_config(ROOT / relative).mutable_copy()


def _validation_rows(count: int = 8) -> tuple[IndexRow, ...]:
    return tuple(
        IndexRow.from_mapping(
            index,
            {
                "sample_id": f"sample-{index}",
                "key": f"key-{index}",
                "shard": "raw/shard.tar",
                "fps": 16.0,
                "num_frames": 160,
                "start_frame": 0,
                "video_member": f"key-{index}.mp4",
                "camera_member": f"key-{index}.camera.npz",
                "manifest": {"metadata": {}, "prompt": {"text": "move forward"}},
            },
        )
        for index in range(count)
    )


def _build_fixed_validation_cases(
    monkeypatch: pytest.MonkeyPatch,
    rows: tuple[IndexRow, ...],
    *,
    rejected: set[str],
    sample_count: int,
    selection_seed: int,
    dp_world_size: int = 1,
    dp_rank: int = 0,
    distributed_context: tuple[object, int, int] | None = None,
    validation_plan_path: Path | None = None,
) -> tuple[tuple[InferenceCase, ...], list[int]]:
    config = _config("configs/examples/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml")
    config["validation"]["sample_count"] = sample_count
    config["validation"]["selection_seed"] = selection_seed
    read_ordinals: list[int] = []

    class FakeShards:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeShards:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, row: IndexRow, name: str) -> bytes:
            read_ordinals.append(row.ordinal)
            return name.encode()

    def fake_camera(payload: bytes, *_args: object, **_kwargs: object) -> dict[str, np.ndarray]:
        sample_id = payload.decode().removesuffix(".camera.npz").replace("key-", "sample-")
        if sample_id in rejected:
            raise wan_inference.CameraGuardError("camera guard rejected candidate")
        return {"viewmats": np.zeros((81, 4, 4), dtype=np.float32)}

    monkeypatch.setattr(wan_inference, "read_index", lambda _path: rows)
    monkeypatch.setattr(wan_inference, "resolver_from_config", lambda *_a, **_k: object())
    monkeypatch.setattr(wan_inference, "TarShardReader", FakeShards)
    monkeypatch.setattr(
        wan_inference,
        "decode_video",
        lambda *_a, **_k: np.zeros((81, 3, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(wan_inference, "build_camera_tokens", fake_camera)
    if distributed_context is not None:
        monkeypatch.setattr(
            wan_inference,
            "_distributed_generation_context",
            lambda: distributed_context,
        )

    adapter = object.__new__(wan_inference.CudaWanGenerationAdapter)
    adapter.config = config
    adapter.topology = SimpleNamespace(dp_world_size=dp_world_size, dp_rank=dp_rank)
    adapter._prepared = {}
    if validation_plan_path is not None:
        adapter.validation_plan_path = validation_plan_path
    return adapter.build_cases(resolve_generation_plan(config)), read_ordinals


def test_source_variable_recipe_start_uses_deterministic_adjustment() -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample-variable",
            "key": "video-variable",
            "shard": "part-000.tar",
            "fps": 16.0,
            "num_frames": 160,
            "start_frame": 28,
        },
    )
    plan, noise_seed, adjusted = _plan_case(
        row,
        slot=137,
        pixel_frames=153,
        output_fps=16.0,
        base_noise_seed=42,
        variable_rollout_by_source=True,
    )
    expected_start_seed = _seed_for("variable-rollout-start:video-variable", 0, 0, 137)
    expected_start = int(np.random.RandomState(expected_start_seed).randint(0, 8))
    assert adjusted is True
    assert plan.start_frame == expected_start
    assert noise_seed == _seed_for("validation-noise", 0, 137, 42)


def test_recipe_row_without_start_derives_a_repeatable_window() -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample-random-start",
            "key": "video-random-start",
            "shard": "part-000.tar",
            "fps": 16.0,
            "num_frames": 320,
        },
    )
    plan, _, adjusted = _plan_case(
        row,
        slot=11,
        pixel_frames=153,
        output_fps=16.0,
        base_noise_seed=42,
        variable_rollout_by_source=True,
    )
    seed = _seed_for("video-random-start", 0, 0, 11)
    expected = int(np.random.RandomState(seed).randint(0, 168))
    assert adjusted is False
    assert plan.start_frame == expected


def test_recipe_row_can_pin_historical_validation_noise_seed() -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample-historical-seed",
            "key": "video-historical-seed",
            "shard": "part-000.tar",
            "fps": 16.0,
            "num_frames": 160,
            "validation_noise_seed": 546,
        },
    )

    _, noise_seed, _ = _plan_case(
        row,
        slot=14,
        pixel_frames=153,
        output_fps=16.0,
        base_noise_seed=42,
        variable_rollout_by_source=True,
    )

    assert noise_seed == 546


@pytest.mark.parametrize("invalid", [-1, 2**63, 1.5, True, "546"])
def test_recipe_row_rejects_invalid_validation_noise_seed(invalid: object) -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample-invalid-seed",
            "key": "video-invalid-seed",
            "shard": "part-000.tar",
            "fps": 16.0,
            "num_frames": 160,
            "validation_noise_seed": invalid,
        },
    )

    with pytest.raises(DataContractError, match="invalid validation_noise_seed"):
        _plan_case(
            row,
            slot=0,
            pixel_frames=153,
            output_fps=16.0,
            base_noise_seed=42,
            variable_rollout_by_source=True,
        )


def test_camera_length_recipe_starts_at_the_first_camera_frame() -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample-full-camera",
            "key": "video-full-camera",
            "shard": "part-000.tar",
            "fps": 16.0,
            "num_frames": 960,
            "start_frame": 28,
        },
    )
    plan, _, adjusted = _plan_case(
        row,
        slot=0,
        pixel_frames=957,
        output_fps=16.0,
        base_noise_seed=42,
        variable_rollout_by_source=True,
        start_at_first_frame=True,
    )
    assert plan.start_frame == 0
    assert plan.source_frame_indices[-1] == 956
    assert adjusted is False


def test_camera_publication_maps_full_source_duration() -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample-full-camera",
            "key": "video-full-camera",
            "shard": "part-000.tar",
            "fps": 16.0,
            "num_frames": 960,
        },
    )

    indices = _camera_publication_source_frame_indices(row, output_fps=16.0)

    assert len(indices) == 960
    assert indices == tuple(range(960))


def test_camera_publication_c2w_is_source_equal_with_float64_cast_only() -> None:
    source = np.repeat(np.eye(4, dtype=np.float64)[None], 5, axis=0)
    source[:, :3, 3] = np.asarray(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]],
        dtype=np.float64,
    )
    source[2, 0, 3] = np.nextafter(source[2, 0, 3], np.inf)
    buffer = io.BytesIO()
    np.savez(buffer, c2w=source)

    selected = _authoritative_publication_c2w(
        buffer.getvalue(),
        array_key="c2w",
        source_frame_indices=(0, 2, 4),
    )

    assert selected.dtype == np.float64
    assert selected.flags.c_contiguous
    np.testing.assert_array_equal(selected, source[[0, 2, 4]])
    assert not np.array_equal(selected[0], np.eye(4, dtype=np.float64))


def test_selected_recipe_row_materializes_against_the_complete_test_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("configs/examples/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml")
    config["validation"]["sample_count"] = 1
    config["validation"]["selection_seed"] = 0
    rows = tuple(
        IndexRow.from_mapping(
            index,
            {
                "sample_id": f"sample-{index}",
                "key": f"key-{index}",
                "shard": "raw/shard.tar",
                "fps": 16.0,
                "num_frames": 160,
                "start_frame": 0,
                "video_member": f"key-{index}.mp4",
                "camera_member": f"key-{index}.camera.npz",
                "manifest": {"metadata": {}, "prompt": {"text": "move forward"}},
            },
        )
        for index in range(4)
    )

    read_ordinals: list[int] = []

    class FakeShards:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeShards:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self, _row: IndexRow, name: str) -> bytes:
            read_ordinals.append(_row.ordinal)
            return name.encode()

    monkeypatch.setattr(wan_inference, "read_index", lambda _path: rows)
    monkeypatch.setattr(wan_inference, "resolver_from_config", lambda *_a, **_k: object())
    monkeypatch.setattr(wan_inference, "TarShardReader", FakeShards)
    monkeypatch.setattr(
        wan_inference,
        "decode_video",
        lambda *_a, **_k: np.zeros((81, 3, 2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        wan_inference,
        "build_camera_tokens",
        lambda *_a, **_k: {"viewmats": np.zeros((81, 4, 4), dtype=np.float32)},
    )

    adapter = object.__new__(wan_inference.CudaWanGenerationAdapter)
    adapter.config = config
    adapter.topology = SimpleNamespace(dp_world_size=1, dp_rank=0)
    adapter._prepared = {}
    cases = adapter.build_cases(resolve_generation_plan(config))

    assert [case.sample_id for case in cases] == ["sample-3"]
    assert cases[0].metadata["artifact_valid"] is True
    assert read_ordinals == [3, 3]
    assert set(adapter._prepared) == {0}
    assert getattr(adapter, "_deferred_camera_inputs", None) is None


def test_camera_length_defers_video_decode_and_releases_each_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml")
    config["inference"]["output_layout"] = "transaction_v1"
    config["validation"]["sample_count"] = 2
    rows = _validation_rows(2)
    reads: list[str] = []
    decodes: list[bytes] = []
    closes: list[int] = []
    camera_drift = [False]

    class FakeShards:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeShards:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def close(self) -> None:
            closes.append(1)

        def read(self, _row: IndexRow, name: str) -> bytes:
            reads.append(name)
            if camera_drift[0] and name.endswith(".camera.npz"):
                return b"changed-camera"
            return name.encode()

    def fake_decode(payload: bytes, *_args: object, **_kwargs: object) -> np.ndarray:
        decodes.append(payload)
        return np.zeros((153, 3, 2, 2), dtype=np.float32)

    monkeypatch.setattr(wan_inference, "read_index", lambda _path: rows)
    monkeypatch.setattr(wan_inference, "resolver_from_config", lambda *_a, **_k: object())
    monkeypatch.setattr(wan_inference, "TarShardReader", FakeShards)
    monkeypatch.setattr(wan_inference, "decode_video", fake_decode)
    monkeypatch.setattr(
        wan_inference,
        "build_camera_tokens",
        lambda *_a, **_k: {
            "viewmats": np.zeros((1, 4, 4), dtype=np.float32),
            "K": np.zeros((1, 3, 3), dtype=np.float32),
        },
    )

    adapter = object.__new__(wan_inference.CudaWanGenerationAdapter)
    adapter.config = config
    adapter.topology = SimpleNamespace(dp_world_size=1, dp_rank=0)
    adapter._prepared = {}
    adapter._deferred_camera_inputs = None
    cases = adapter.build_cases(resolve_generation_plan(config))

    assert len(cases) == 2
    assert adapter._prepared == {}
    assert set(adapter._deferred_camera_inputs.cases) == {0, 1}
    assert decodes == []
    assert len(reads) == 2
    assert all(name.endswith(".camera.npz") for name in reads)

    adapter._materialize_deferred_camera_case(cases[0])
    assert set(adapter._prepared) == {cases[0].slot}
    assert len(decodes) == 1
    key = cases[0].metadata["key"]
    assert reads[-2:] == [f"{key}.mp4", f"{key}.camera.npz"]

    first_deferred = adapter._deferred_camera_inputs
    first_reader = first_deferred.reader
    repeated = adapter.build_cases(resolve_generation_plan(config))
    assert repeated == cases
    assert adapter._deferred_camera_inputs is not first_deferred
    assert first_deferred.cases == {}
    assert first_deferred.reader is None
    assert first_deferred.shards is None
    assert first_reader is not None
    assert adapter._prepared == {}
    assert len(decodes) == 1

    camera_drift[0] = True
    with pytest.raises(DataContractError, match="payload drifted"):
        adapter._materialize_deferred_camera_case(repeated[1])
    assert adapter._prepared == {}
    assert len(decodes) == 1

    from solarwm.backends.wan22.runtime import distributed as wan_distributed

    distributed_closes: list[int] = []
    monkeypatch.setattr(wan_distributed, "cleanup_torchrun", lambda: distributed_closes.append(1))
    second_deferred = adapter._deferred_camera_inputs
    adapter.close()
    adapter.close()
    assert adapter._deferred_camera_inputs is None
    assert second_deferred.cases == {}
    assert second_deferred.reader is None
    assert second_deferred.shards is None
    assert len(closes) == 4
    assert distributed_closes == [1, 1]


def test_camera_length_static_condition_decodes_only_the_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml")
    config["inference"]["output_layout"] = "transaction_v1"
    config["validation"]["sample_count"] = 1
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample-static",
            "key": "static",
            "shard": "raw/shard.tar",
            "clip_id": "static",
            "dataset": "demo-hour",
            "fps": 16.0,
            "num_frames": 160,
            "start_frame": 0,
            "video_member": "static.mp4",
            "camera_member": "static.camera.npz",
            "manifest": {
                "metadata": {},
                "prompt": {"text": "move forward"},
                "video": {"frame_content": "static_first_frame"},
            },
        },
    )
    decoded_indices: list[tuple[int, ...]] = []

    class FakeShards:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> FakeShards:
            return self

        def __exit__(self, *_args: object) -> None:
            self.close()

        def close(self) -> None:
            return None

        def read(self, _row: IndexRow, name: str) -> bytes:
            return name.encode()

    def fake_decode(
        _payload: bytes, frame_indices: tuple[int, ...], **_kwargs: object
    ) -> np.ndarray:
        decoded_indices.append(tuple(frame_indices))
        return np.zeros((len(frame_indices), 3, 2, 2), dtype=np.float32)

    monkeypatch.setattr(wan_inference, "read_index", lambda _path: (row,))
    monkeypatch.setattr(wan_inference, "resolver_from_config", lambda *_a, **_k: object())
    monkeypatch.setattr(wan_inference, "TarShardReader", FakeShards)
    monkeypatch.setattr(wan_inference, "decode_video", fake_decode)
    monkeypatch.setattr(
        wan_inference,
        "build_camera_tokens",
        lambda *_a, **_k: {
            "viewmats": np.zeros((1, 4, 4), dtype=np.float32),
            "K": np.zeros((1, 3, 3), dtype=np.float32),
        },
    )

    adapter = object.__new__(wan_inference.CudaWanGenerationAdapter)
    adapter.config = config
    adapter.topology = SimpleNamespace(dp_world_size=1, dp_rank=0)
    adapter._prepared = {}
    adapter._deferred_camera_inputs = None
    cases = adapter.build_cases(resolve_generation_plan(config))
    adapter._materialize_deferred_camera_case(cases[0])

    assert decoded_indices == [(0,)]
    assert adapter._prepared[0].repeat_first_frame is True
    assert adapter._prepared[0].pixels.shape[0] == 1


@pytest.mark.parametrize(
    ("target_pixel_frames", "expected_chunk_sizes"),
    ((8, (5, 3)), (12, (5, 4, 3))),
)
def test_stage2_streaming_encoder_bounds_tiles_and_adjusts_the_tail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    target_pixel_frames: int,
    expected_chunk_sizes: tuple[int, ...],
) -> None:
    torch = pytest.importorskip("torch")
    writers: list[object] = []

    class FakeWriter:
        def __init__(self, path: Path, *, fps: float, height: int, width: int) -> None:
            self.path = path
            self.fps = fps
            self.height = height
            self.width = width
            self.frames: list[np.ndarray] = []
            writers.append(self)

        def write(self, frames: np.ndarray) -> None:
            self.frames.append(frames.copy())

        def finish(self) -> None:
            self.path.write_bytes(b"encoded")

        def abort(self) -> None:
            return None

    class FakeVAE:
        @staticmethod
        def decode_streaming_chunks(_latents: object, *, chunk_latent_frames: int) -> object:
            assert chunk_latent_frames == 2
            yield torch.zeros((1, 5, 3, 2, 2), dtype=torch.float32)
            yield torch.ones((1, 4, 3, 2, 2), dtype=torch.float32)

    monkeypatch.setattr(wan_inference, "_RawVideoFileWriter", FakeWriter)
    prepared = wan_inference._PreparedCase(
        pixels=torch.full((1, 3, 2, 2), -1.0, dtype=torch.float32),
        camera={},
        source_pixel_frames=12,
        repeat_first_frame=True,
    )
    result = wan_inference._encode_stage2_streaming(
        FakeVAE(),
        torch.zeros((1, 3, 1, 1, 1)),
        prepared,
        target_pixel_frames=target_pixel_frames,
        fps=16.0,
        temporary_root=tmp_path,
        chunk_latent_frames=2,
    )

    assert len(writers) == 2
    generated_writer, compare_writer = writers
    assert tuple(len(chunk) for chunk in generated_writer.frames) == expected_chunk_sizes
    assert tuple(len(chunk) for chunk in compare_writer.frames) == expected_chunk_sizes
    assert generated_writer.width == 2
    assert compare_writer.width == 4
    np.testing.assert_array_equal(compare_writer.frames[0][..., :2, :], 0)
    assert result.shape == (1, target_pixel_frames, 3, 2, 2)
    result.video.path.unlink()
    result.compare.path.unlink()


def test_compare_encoder_repeats_static_first_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    captured: dict[str, object] = {}

    def fake_encode(reference: object, generated: object, **kwargs: object) -> bytes:
        captured.update(reference=reference, generated=generated, kwargs=kwargs)
        return b"encoded"

    monkeypatch.setattr(wan_inference, "encode_compare_mp4", fake_encode)
    prepared = wan_inference._PreparedCase(
        pixels=torch.full((1, 3, 2, 2), -0.5),
        camera={},
        source_pixel_frames=12,
        repeat_first_frame=True,
    )
    generated = torch.zeros((1, 12, 3, 2, 2))

    assert wan_inference._encode_compare_mp4(generated, prepared, fps=16.0) == b"encoded"
    reference = captured["reference"]
    assert isinstance(reference, torch.Tensor)
    assert reference.shape == generated.shape
    torch.testing.assert_close(reference, torch.full_like(generated, -0.5))
    assert captured["generated"] is generated
    assert captured["kwargs"] == {
        "fps": 16.0,
        "layout": "btchw",
        "value_range": "minus_one_one",
    }


def test_fixed_validation_skips_and_refills_from_the_full_seeded_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _validation_rows()
    seeded = select_index_rows(rows, sample_count=len(rows), seed=42)
    rejected = {seeded[0].sample_id, seeded[2].sample_id}

    cases, _ = _build_fixed_validation_cases(
        monkeypatch,
        rows,
        rejected=rejected,
        sample_count=4,
        selection_seed=42,
    )

    expected = [row.sample_id for row in seeded if row.sample_id not in rejected][:4]
    assert [case.sample_id for case in cases] == expected
    assert [case.slot for case in cases] == list(range(4))
    assert len({case.sample_id for case in cases}) == 4
    assert all(case.metadata["artifact_valid"] is True for case in cases)


def test_fixed_validation_freezes_then_materializes_only_loaded_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _validation_rows()
    plan_path = tmp_path / "validation/frozen-plan.json"
    rejected = {select_index_rows(rows, sample_count=len(rows), seed=42)[0].sample_id}
    first, first_reads = _build_fixed_validation_cases(
        monkeypatch,
        rows,
        rejected=rejected,
        sample_count=4,
        selection_seed=42,
        validation_plan_path=plan_path,
    )
    second, second_reads = _build_fixed_validation_cases(
        monkeypatch,
        rows,
        rejected=rejected,
        sample_count=4,
        selection_seed=42,
        validation_plan_path=plan_path,
    )

    assert second == first
    assert plan_path.is_file()
    assert len(second_reads) == 2 * len(second)
    assert len(first_reads) > len(second_reads)


def test_fixed_validation_is_stable_and_partitions_complete_waves(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _validation_rows()
    seeded = select_index_rows(rows, sample_count=len(rows), seed=17)
    rejected = {seeded[1].sample_id}

    def build_partition(dp_rank: int) -> tuple[InferenceCase, ...]:
        cases, _ = _build_fixed_validation_cases(
            monkeypatch,
            rows,
            rejected=rejected,
            sample_count=4,
            selection_seed=17,
            dp_world_size=2,
            dp_rank=dp_rank,
        )
        return cases

    first = tuple(sorted((*build_partition(0), *build_partition(1)), key=lambda case: case.slot))
    repeated = tuple(sorted((*build_partition(0), *build_partition(1)), key=lambda case: case.slot))

    assert [case.slot for case in first] == list(range(4))
    assert [case.slot for case in build_partition(0)] == [0, 2]
    assert [case.slot for case in build_partition(1)] == [1, 3]
    assert [(case.slot, case.sample_id, case.start_frame, case.noise_seed) for case in first] == [
        (case.slot, case.sample_id, case.start_frame, case.noise_seed) for case in repeated
    ]


def test_distributed_selector_evaluates_each_candidate_on_one_dp_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _validation_rows()
    seeded = select_index_rows(rows, sample_count=len(rows), seed=23)
    rejected = {seeded[0].sample_id}

    class FakeDistributed:
        def __init__(self) -> None:
            self.candidate_ordinal = 0

        def all_gather_object(self, output: list[object], local: object) -> None:
            ordinal = self.candidate_ordinal
            row = seeded[ordinal]
            report = local or {
                "candidate_ordinal": ordinal,
                "sample_id": row.sample_id,
                "accepted": row.sample_id not in rejected,
                "error": (
                    "CameraGuardError: camera guard rejected candidate"
                    if row.sample_id in rejected
                    else None
                ),
            }
            output[:] = [None] * 4
            output[ordinal % 4] = report
            self.candidate_ordinal += 1

    distributed = FakeDistributed()
    cases, read_ordinals = _build_fixed_validation_cases(
        monkeypatch,
        rows,
        rejected=rejected,
        sample_count=4,
        selection_seed=23,
        dp_world_size=4,
        dp_rank=0,
        distributed_context=(distributed, 0, 4),
    )

    assert [case.slot for case in cases] == [0]
    assert cases[0].sample_id == seeded[1].sample_id
    # DP0 evaluates candidate ordinals 0 and 4, then materializes only its
    # selected slot. It does not decode the complete five-candidate prefix.
    assert len(read_ordinals) == 6


def test_fixed_validation_fails_after_exhausting_unique_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _validation_rows(4)
    rejected = {row.sample_id for row in rows[1:]}

    with pytest.raises(DataContractError, match="too few camera-safe validation candidates"):
        _build_fixed_validation_cases(
            monkeypatch,
            rows,
            rejected=rejected,
            sample_count=2,
            selection_seed=5,
        )


def test_nonvariable_recipe_start_remains_a_contract_error() -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample-fixed",
            "key": "video-fixed",
            "shard": "part-000.tar",
            "fps": 16.0,
            "num_frames": 160,
            "start_frame": 28,
        },
    )
    with pytest.raises(DataContractError, match="outside max 7"):
        _plan_case(
            row,
            slot=0,
            pixel_frames=153,
            output_fps=16.0,
            base_noise_seed=42,
        )


def test_validation_rejects_an_unmaterialized_input(
    tmp_path: Path,
) -> None:
    config = _config("configs/examples/wan22_ti2v_5b/train_stage1_tf_anyflow_v1_5_81f.yaml")
    invalid = InferenceCase(
        slot=0,
        sample_id="unmaterialized",
        prompt="move forward",
        start_frame=0,
        noise_seed=42,
        camera_fingerprint="0" * 64,
        metadata={
            "artifact_valid": False,
        },
    )
    provider = FakeProvider("wan22_ti2v_5b")
    output = tmp_path / "strict-validation"
    with pytest.raises(BackendContractError, match="requires materialized inputs"):
        run_wan_generation(
            config,
            provider=provider,
            cases=(invalid,),
            output_dir=output,
        )
    assert provider.calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    ("relative", "family", "frames"),
    (
        (
            "configs/examples/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml",
            "wan22_ti2v_5b",
            21,
        ),
        (
            "configs/examples/wan22_ti2v_5b/train_stage0p5_fm_153f.yaml",
            "wan22_ti2v_5b",
            39,
        ),
        (
            "configs/examples/wan22_i2v_a14b/train_stage0p5_fm_81f.yaml",
            "wan22_i2v_a14b",
            21,
        ),
        (
            "configs/examples/wan22_i2v_a14b/train_stage0p5_fm_153f.yaml",
            "wan22_i2v_a14b",
            39,
        ),
    ),
)
def test_fake_provider_routes_both_families_and_frame_lengths(
    tmp_path: Path,
    relative: str,
    family: str,
    frames: int,
) -> None:
    config = _config(relative)
    provider = FakeProvider(family)
    summary = run_wan_generation(
        config,
        provider=provider,
        cases=(_case(),),
        output_dir=tmp_path / family / f"{frames}f",
    )
    assert summary.family == family
    records = [
        json.loads(line)
        for line in (summary.output_dir / summary.passes[0] / "ordered-manifest.jsonl")
        .read_text()
        .splitlines()
    ]
    assert records[0]["case"]["metadata"]["generation_pass"]["rollout_latent_frames"] == frames


def test_inference_and_validation_use_the_same_runner_contract(tmp_path: Path) -> None:
    config = _config("configs/examples/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml")
    left = FakeProvider("wan22_i2v_a14b")
    right = FakeProvider("wan22_i2v_a14b")
    inference = run_wan_inference(
        config,
        provider=left,
        cases=(_case(),),
        output_dir=tmp_path / "inference",
    )
    validation = run_wan_validation(
        config,
        provider=right,
        cases=(_case(),),
        output_dir=tmp_path / "validation",
    )
    assert left.calls == right.calls
    assert inference.complete_digest == validation.complete_digest
    assert (
        json.loads((inference.output_dir / "COMPLETE.json").read_text())[
            "shared_inference_validation_implementation"
        ]
        is True
    )


def test_validation_publishes_compare_layout(tmp_path: Path) -> None:
    from solarwm.backends.wan22.generation import resolve_generation_plan

    config = _config("configs/examples/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml")
    plan = resolve_generation_plan(config)
    target = tmp_path / "validation" / "step-000020"
    generation_pass = plan.passes[0]
    sample = target / generation_pass.name / "slot-000000"
    sample.mkdir(parents=True)
    (sample / "compare.mp4").write_bytes(b"gt-left-gen-right")
    manifest = {
        "case": {
            "slot": 0,
            "sample_id": "dataset/sample-0",
            "prompt": "",
            "start_frame": 17,
            "noise_seed": 42,
            "camera_fingerprint": "a" * 64,
            "metadata": {
                "key": "sample-0",
                "artifact_valid": True,
                "generation_pass": {
                    **generation_pass.__dict__,
                    "output_rollout_latent_frames": 21,
                },
            },
        },
        "shape": [1, 81, 3, 480, 864],
        "metrics": {"finite_fraction": 1.0},
        "provenance": {"denoising_step_list": [1000, 625, 250, 125]},
    }
    (sample / "manifest.json").write_text(json.dumps(manifest))

    output = tmp_path / "validation" / f"step_000020_{generation_pass.name}"
    records = publish_comparison_partition(
        target / generation_pass.name,
        output,
        step=20,
        pass_name=generation_pass.name,
        cases=1,
        dp_world_size=1,
        sp_size=1,
        run_root=tmp_path,
    )
    publish_comparison_complete(
        output,
        step=20,
        pass_name=generation_pass.name,
        local_slots=len(records),
        global_slots=1,
    )

    compare = next((output / "compare").glob("rank*.mp4"))
    assert compare.read_bytes() == b"gt-left-gen-right"
    manifest = json.loads((output / "manifests/rank_000.json").read_text())
    assert manifest["compare_mp4"].endswith(compare.name)
    assert manifest["artifact_valid"] is True
    assert manifest["T_pix"] == 81
    assert manifest["denoising_step_list"] == [1000, 625, 250, 125]
    assert json.loads((output / "COMPLETE.json").read_text())["global_slots"] == 1


def test_training_validation_exposes_only_public_views_after_success(tmp_path: Path) -> None:
    config = _config("configs/examples/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml")
    run_root = tmp_path / "run"
    config["runtime"]["output_dir"] = str(run_root)
    target = run_root / "validation/.staging/step-000020"

    summary = run_wan_validation(
        config,
        provider=ComparisonFakeProvider("wan22_i2v_a14b"),
        cases=(_case(),),
        output_dir=target,
    )

    assert summary.output_dir == run_root / "validation"
    assert summary.output_dir.is_dir()
    assert not (run_root / "validation/.staging").exists()
    public = tuple((run_root / "validation").glob("step_000020_*"))
    assert public
    assert all((directory / "COMPLETE.json").is_file() for directory in public)
    assert not tuple((run_root / "validation").glob("step-*"))


def test_failed_training_validation_retains_private_staging(tmp_path: Path) -> None:
    config = _config("configs/examples/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml")
    run_root = tmp_path / "run"
    config["runtime"]["output_dir"] = str(run_root)
    target = run_root / "validation/.staging/step-000020"
    first_pass = resolve_generation_plan(config).passes[0].name

    with pytest.raises(BackendContractError, match="injected generation failure"):
        run_wan_validation(
            config,
            provider=ComparisonFakeProvider(
                "wan22_i2v_a14b",
                fail_pass=first_pass,
            ),
            cases=(_case(),),
            output_dir=target,
        )

    assert (run_root / "validation/.staging").is_dir()
    assert not tuple((run_root / "validation").glob("step_000020_*"))


def test_comparison_manifest_preserves_multiwave_round_identity(tmp_path: Path) -> None:
    source = tmp_path / "source"
    sample = source / "slot-000009"
    sample.mkdir(parents=True)
    (sample / "compare.mp4").write_bytes(b"gt-left-gen-right")
    manifest = {
        "case": {
            "slot": 9,
            "sample_id": "dataset/sample-9",
            "prompt": "",
            "start_frame": 0,
            "noise_seed": 51,
            "camera_fingerprint": "b" * 64,
            "metadata": {"key": "sample-9", "artifact_valid": True},
        },
        "shape": [1, 153, 3, 512, 768],
        "metrics": {"finite_fraction": 1.0},
    }
    (sample / "manifest.json").write_text(json.dumps(manifest))

    destination = tmp_path / "validation" / "step_000050_live"
    records = publish_comparison_partition(
        source,
        destination,
        step=50,
        pass_name="live",
        cases=16,
        dp_world_size=8,
        sp_size=2,
        run_root=tmp_path,
        logical_world_size_per_round=8,
    )

    assert "round01" in records[0].compare_path.name
    payload = json.loads(records[0].manifest_path.read_text())
    assert payload["logical_rank_within_round"] == 1
    assert payload["logical_world_size_per_round"] == 8
    assert payload["validation_round_index"] == 1
    assert payload["validation_num_rounds"] == 2


def test_failed_pass_never_commits_generation_complete(tmp_path: Path) -> None:
    config = _config("configs/examples/wan22_ti2v_5b/infer_stage1_tf_anyflow_v1_5_81f.yaml")
    provider = FakeProvider("wan22_ti2v_5b", fail_pass="ema_nfe50_autoregressive")
    target = tmp_path / "failed"
    with pytest.raises(BackendContractError, match="injected generation failure"):
        run_wan_generation(
            config,
            provider=provider,
            cases=(_case(),),
            output_dir=target,
        )
    assert not target.exists()
    assert not list(tmp_path.glob(".failed.*.partial/COMPLETE.json"))


def test_sampler_cannot_drift_from_training_stage(tmp_path: Path) -> None:
    config = _config("configs/examples/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml")
    config["validation"]["passes"][0]["mode"] = "autoregressive"
    with pytest.raises(BackendContractError, match="does not match stage0p5"):
        run_wan_generation(
            config,
            provider=FakeProvider("wan22_i2v_a14b"),
            cases=(_case(),),
            output_dir=tmp_path / "drift",
        )


@pytest.mark.parametrize("name", ("../escape", "/absolute", "nested/pass", r"win\\pass"))
def test_generation_pass_name_cannot_escape_output_tree(tmp_path: Path, name: str) -> None:
    config = _config("configs/examples/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml")
    config["validation"]["passes"][0]["name"] = name
    with pytest.raises(BackendContractError, match="portable path components"):
        run_wan_generation(
            config,
            provider=FakeProvider("wan22_i2v_a14b"),
            cases=(_case(),),
            output_dir=tmp_path / "output",
        )
    assert not (tmp_path / "escape").exists()


def test_existing_generation_output_is_never_overwritten(tmp_path: Path) -> None:
    config = _config("configs/examples/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml")
    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(BackendContractError, match="already exists"):
        run_wan_generation(
            config,
            provider=FakeProvider("wan22_i2v_a14b"),
            cases=(_case(),),
            output_dir=target,
        )


def test_checkpoint_directory_requires_a_complete_transaction(tmp_path: Path) -> None:
    config = _config("configs/examples/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.pt").write_bytes(b"uncommitted")
    config["checkpoint"]["path"] = str(checkpoint)
    with pytest.raises(BackendContractError, match="transaction is invalid"):
        _checkpoint_file(config)


def test_cli_runtime_controls_and_generation_tree_do_not_collide(tmp_path: Path) -> None:
    config = _config("configs/examples/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml")
    runtime = tmp_path / "run"
    runtime.mkdir()
    (runtime / "launch-manifest.json").write_text("{}\n")
    config["runtime"]["output_dir"] = str(runtime)
    summary = run_wan_generation(
        config,
        provider=FakeProvider("wan22_i2v_a14b"),
        cases=(_case(),),
    )
    assert summary.output_dir == runtime / "generation"
    assert (runtime / "launch-manifest.json").is_file()
    assert (runtime / "generation/COMPLETE.json").is_file()


def test_camera_inference_default_publishes_dataset_triplets_and_keeps_run_transaction(
    tmp_path: Path,
) -> None:
    config = _config("configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml")
    config["runtime"]["output_dir"] = str(tmp_path)
    config["inference"]["run_id"] = "sekai-fix-canary"
    config["validation"]["sample_count"] = 1
    case = InferenceCase(
        **{
            **_case().__dict__,
            "metadata": {
                "physical_dataset": "sekai_game-fix",
                "publish_stem": "clip-0001",
                "publication_pixel_frames": 960,
                "camera_publication_convention": "authoritative_absolute_c2w",
                "rollout_length_source": "camera",
                "rollout_latent_frames_by_pass": {
                    "model_self_forcing_nfe4": 240,
                },
            },
        }
    )

    summary = run_wan_generation(
        config,
        provider=CameraPublicationFakeProvider("wan22_ti2v_5b"),
        cases=(case,),
    )

    assert summary.output_dir == tmp_path / "runs/sekai-fix-canary/generation"
    assert summary.publication_layout == "dataset_triplet_v1"
    assert (
        (tmp_path / "generate/sekai_game-fix/clip-0001.mp4")
        .read_bytes()
        .startswith(b"model_self_forcing_nfe4")
    )
    assert (tmp_path / "compare/sekai_game-fix/clip-0001.mp4").read_bytes() == b"compare:0"
    published_c2w = np.load(
        tmp_path / "camera/sekai_game-fix/clip-0001.npy",
        allow_pickle=False,
    )
    assert published_c2w.dtype == np.float64
    assert published_c2w.shape == (960, 4, 4)
    assert (tmp_path / "runs/sekai-fix-canary/publication/COMPLETE.json").is_file()
    assert (tmp_path / "runs/sekai-fix-canary/generation/COMPLETE.json").is_file()


def test_camera_dataset_layout_rejects_node_local_multinode_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config("configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml")
    config["runtime"]["output_dir"] = str(tmp_path)
    monkeypatch.setenv("NNODES", "2")
    monkeypatch.setattr(
        wan_inference,
        "_generation_topology",
        lambda _provider: (0, 2, 0, 2, 0),
    )

    with pytest.raises(BackendContractError, match="node-local multi-node output is unsupported"):
        run_wan_generation(
            config,
            provider=CameraPublicationFakeProvider("wan22_ti2v_5b"),
            cases=(_case(),),
        )


@pytest.mark.parametrize("shift", (3.0, 5.0))
def test_release_unipc_matches_wan_flow_grid(shift: float) -> None:
    from solarwm.backends.wan22.runtime.scheduler import (
        build_wan_flow_unipc_scheduler,
    )

    scheduler = build_wan_flow_unipc_scheduler(
        num_train_timesteps=1000,
        shift=shift,
        num_inference_steps=50,
        device="cpu",
    )
    base = np.linspace(0.999, 0.0, 51, dtype=np.float64)[:-1]
    expected_sigmas = shift * base / (1.0 + (shift - 1.0) * base)
    expected_timesteps = (expected_sigmas * 1000).astype(np.int64)
    np.testing.assert_array_equal(scheduler.timesteps.numpy(), expected_timesteps)
    np.testing.assert_allclose(
        scheduler.sigmas[:-1].numpy(), expected_sigmas.astype(np.float32), atol=1e-7
    )


@pytest.mark.parametrize(
    ("num_frames", "expected"),
    ((81, 21), (153, 39), (160, 39), (960, 60)),
)
def test_source_variable_rollout_matches_mapping(num_frames: int, expected: int) -> None:
    assert (
        _source_variable_rollout_latents(
            num_frames=num_frames,
            max_latent_frames=60,
            min_latent_frames=21,
            num_frame_per_block=3,
        )
        == expected
    )


@pytest.mark.parametrize(
    ("num_frames", "source_fps", "expected"),
    (
        (81, 16.0, 21),
        (153, 16.0, 39),
        (160, 16.0, 42),
        (237, 16.0, 60),
        (960, 16.0, 243),
        (960, 30.0, 129),
    ),
)
def test_camera_length_rollout_uses_smallest_complete_covering_chunk(
    num_frames: int,
    source_fps: float,
    expected: int,
) -> None:
    assert (
        _camera_length_rollout_latents(
            num_frames=num_frames,
            source_fps=source_fps,
            output_fps=16.0,
            num_frame_per_block=3,
        )
        == expected
    )


def test_camera_length_rollout_holds_the_tail_for_a_short_source() -> None:
    assert (
        _camera_length_rollout_latents(
            num_frames=8,
            source_fps=16.0,
            output_fps=16.0,
            num_frame_per_block=3,
        )
        == 3
    )


def test_camera_length_plan_holds_only_the_chunk_alignment_tail() -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample-tail-hold",
            "key": "video-tail-hold",
            "shard": "part-000.tar",
            "fps": 16.0,
            "num_frames": 160,
        },
    )

    plan, _, _ = _plan_case(
        row,
        slot=0,
        pixel_frames=165,
        output_fps=16.0,
        base_noise_seed=42,
        variable_rollout_by_source=True,
        start_at_first_frame=True,
        hold_last_source_frame=True,
    )

    assert plan.source_frame_indices[:160] == tuple(range(160))
    assert plan.source_frame_indices[160:] == (159,) * 5


def test_source_variable_pass_keeps_collective_max_and_records_trim() -> None:
    from solarwm.backends.wan22.generation import GenerationPass
    from solarwm.backends.wan22.runtime.inference import _pass_case

    generation_pass = GenerationPass(
        name="source_length",
        weights="live",
        mode="autoregressive",
        solver="unipc",
        num_inference_steps=50,
        rollout_latent_frames=60,
        min_rollout_latent_frames=21,
        fixed_plan_pixel_frames=237,
        variable_rollout_by_source=True,
    )
    case = _slotted_case(0)
    case = InferenceCase(
        **{
            **case.__dict__,
            "metadata": {"rollout_latent_frames_by_pass": {"source_length": 21}},
        }
    )
    active = _pass_case(case, generation_pass)
    metadata = active.metadata["generation_pass"]
    assert metadata["rollout_latent_frames"] == 60
    assert metadata["output_rollout_latent_frames"] == 21


def test_camera_length_pass_runs_the_resolved_chunk_aligned_horizon() -> None:
    from solarwm.backends.wan22.runtime.inference import _pass_case

    generation_pass = GenerationPass(
        name="model_self_forcing_nfe4",
        weights="model",
        mode="autoregressive",
        solver="self_forcing",
        num_inference_steps=4,
        rollout_latent_frames=3,
        min_rollout_latent_frames=3,
        fixed_plan_pixel_frames=9,
        variable_rollout_by_source=False,
    )
    case = _slotted_case(0)
    case = InferenceCase(
        **{
            **case.__dict__,
            "metadata": {
                "rollout_length_source": "camera",
                "rollout_latent_frames_by_pass": {"model_self_forcing_nfe4": 240},
            },
        }
    )
    active = _pass_case(case, generation_pass)
    metadata = active.metadata["generation_pass"]
    assert metadata["rollout_latent_frames"] == 240
    assert metadata["min_rollout_latent_frames"] == 240
    assert metadata["fixed_plan_pixel_frames"] == 957
    assert "output_rollout_latent_frames" not in metadata


@pytest.mark.parametrize("shift", (3.0, 5.0))
def test_flowmap_inference_schedule_is_float32_exact(shift: float) -> None:
    torch = pytest.importorskip("torch")
    actual_t, actual_r = _flowmap_schedule(
        50,
        shift=shift,
        num_train_timesteps=1000,
        device="cpu",
    )
    base = torch.linspace(1.0, 0.0, 51, dtype=torch.float32)
    shifted = shift * base / (1.0 + (shift - 1.0) * base)
    shifted[-1] = 0.0
    expected = shifted * 1000.0
    assert actual_t.dtype == torch.float32
    assert torch.equal(actual_t, expected[:-1])
    assert torch.equal(actual_r, expected[1:])


def test_logical_dp_partition_is_wave_aligned() -> None:
    cases = tuple(_slotted_case(slot) for slot in range(7))
    assert [
        case.slot for case in _partition_generation_cases(cases, dp_rank=0, dp_world_size=3)
    ] == [0, 3, 6]
    assert [
        case.slot for case in _partition_generation_cases(cases, dp_rank=2, dp_world_size=3)
    ] == [2, 5]


def test_logical_dp_partition_allows_idle_ranks() -> None:
    cases = tuple(_slotted_case(slot) for slot in range(3))

    assert _partition_generation_cases(cases, dp_rank=7, dp_world_size=8) == ()


def test_camera_publication_identity_falls_back_to_dataset() -> None:
    row = IndexRow(
        ordinal=0,
        sample_id="SolarWM/abot/clip-1",
        key="abot__abot__clip-1",
        shard="shards/test-000000.tar",
        epoch_repeats=1,
        values={"dataset": "abot", "clip_id": "clip-1"},
    )

    assert _camera_publication_identity(row) == ("abot", "clip-1")


def test_cuda_adapter_admits_multiple_logical_dp_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime import components, inference
    from solarwm.backends.wan22.runtime import distributed as wan_distributed

    class _Movable:
        def eval(self) -> _Movable:
            return self

        def requires_grad_(self, value: bool) -> _Movable:
            assert value is False
            return self

        def to(self, *args: object, **kwargs: object) -> _Movable:
            del args, kwargs
            return self

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    topology = SimpleNamespace(
        raw_rank=2,
        raw_world_size=4,
        local_rank=0,
        dp_rank=2,
        dp_world_size=4,
        sp_rank=0,
        sp_size=1,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(wan_distributed, "initialize_torchrun", lambda _: topology)
    monkeypatch.setattr(inference, "_checkpoint_file", lambda _: checkpoint)
    movable = _Movable()
    monkeypatch.setattr(
        components,
        "build_online_components",
        lambda _: (SimpleNamespace(module=movable), movable, movable, object()),
    )
    adapter = inference.CudaWanGenerationAdapter(
        {
            "model": {"family": "wan22_ti2v_5b"},
            "distributed": {"sequence_parallel_size": 1},
        },
        SimpleNamespace(),
    )
    assert adapter.topology.dp_world_size == 4
    assert adapter.topology.dp_rank == 2
    assert adapter.checkpoint_id == (
        f"inventory:file=checkpoint.pt:bytes={checkpoint.stat().st_size}"
    )


def test_logical_dp_partitions_merge_into_one_ordered_transaction(
    tmp_path: Path,
) -> None:
    cases = tuple(_slotted_case(slot) for slot in range(4))

    class PartitionProvider:
        family = "wan22_ti2v_5b"

        def generate(self, case: InferenceCase, *, weights_id: str) -> GeneratedSample:
            del weights_id
            return GeneratedSample(
                artifacts={"video.mp4": str(case.slot).encode()},
                shape=(1, 1, 3, 1, 1),
                dtype="float32",
            )

    provider = PartitionProvider()
    parts = tmp_path / "parts"
    for dp_rank in range(2):
        InferenceEngine(provider).run(
            _partition_generation_cases(cases, dp_rank=dp_rank, dp_world_size=2),
            weights_id="checkpoint-digest#live",
            output_dir=parts / f"dp-{dp_rank:06d}",
        )
    merged = tmp_path / "merged"
    count, digest = _merge_generation_partitions(
        parts,
        merged,
        cases=cases,
        family="wan22_ti2v_5b",
        weights_id="checkpoint-digest#live",
        dp_world_size=2,
    )
    manifests = [
        json.loads(line) for line in (merged / "ordered-manifest.jsonl").read_text().splitlines()
    ]
    assert count == 4
    assert len(digest) == 64
    assert [item["case"]["slot"] for item in manifests] == [0, 1, 2, 3]


def test_sliding_camera_window_is_rebased_to_its_first_latent() -> None:
    import torch

    frame_sequence_length = 2
    frame_viewmats = torch.eye(4).repeat(1, 3, 1, 1)
    frame_viewmats[0, 1, 0, 3] = -1.0
    frame_viewmats[0, 2, 0, 3] = -3.0
    camera = {
        "viewmats": frame_viewmats.repeat_interleave(frame_sequence_length, dim=1),
        "K": torch.eye(3).repeat(1, 3 * frame_sequence_length, 1, 1),
    }
    window = _camera_window(
        camera,
        frame_sequence_length=frame_sequence_length,
        start=1,
        end=3,
    )
    torch.testing.assert_close(window["viewmats"][0, 0], torch.eye(4))
    torch.testing.assert_close(window["viewmats"][0, 2, 0, 3], torch.tensor(-2.0))


def _runtime_case(weights: str) -> InferenceCase:
    return InferenceCase(
        slot=0,
        sample_id="runtime-sample",
        prompt="move forward",
        start_frame=0,
        noise_seed=17,
        camera_fingerprint="b" * 64,
        metadata={
            "generation_pass": {
                "name": f"{weights}-pass",
                "weights": weights,
                "mode": "bidirectional",
                "solver": "unipc",
                "num_inference_steps": 2,
                "rollout_latent_frames": 21,
            }
        },
    )


def test_training_adapter_swaps_ema_and_restores_live_training_state() -> None:
    torch = pytest.importorskip("torch")
    from solarwm.training.ema import ShardedEMA

    config = _config("configs/examples/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml")
    module = torch.nn.Linear(1, 1, bias=False)
    module.weight.data.fill_(2.0)
    module.train()
    ema = ShardedEMA(
        module,
        decay=0.9,
        device="cpu",
        dtype=torch.float32,
    )
    ema.shadow["weight"].fill_(5.0)
    runtime = SimpleNamespace(
        codec=SimpleNamespace(text_encoder=object(), vae=object()),
        config=config,
        family="wan22_ti2v_5b",
        topology=SimpleNamespace(raw_rank=0),
        device=torch.device("cpu"),
        diffusion=SimpleNamespace(module=module),
        global_step=7,
        checkpoint_id="digest:runtime-checkpoint",
        initialization_receipt={
            "schema": "solarwm.wan22-initialization.v1",
            "initialization_id": "digest:initial",
        },
        ema=ema,
    )
    adapter = TrainingWanGenerationAdapter(runtime)
    observed: list[tuple[str, float, bool]] = []

    def fake_generate_loaded(
        self: TrainingWanGenerationAdapter,
        case: InferenceCase,
        generation_pass: object,
        *,
        weights_id: str,
    ) -> GeneratedSample:
        del case, weights_id
        observed.append(
            (
                generation_pass.weights,
                float(self.diffusion.module.weight.item()),
                bool(self.diffusion.module.training),
            )
        )
        return GeneratedSample(
            artifacts={"video.mp4": b"fake"},
            shape=(1, 1, 3, 1, 1),
            dtype="float32",
        )

    adapter._generate_loaded = MethodType(fake_generate_loaded, adapter)
    adapter.generate(_runtime_case("live"), weights_id=adapter.weight_id("live"))
    adapter.generate(_runtime_case("ema"), weights_id=adapter.weight_id("ema"))

    assert observed == [("live", 2.0, False), ("ema", 5.0, False)]
    assert float(module.weight.item()) == 2.0
    assert module.training is True


def test_preencoded_runtime_cannot_claim_inline_raw_validation() -> None:
    config = _config("configs/examples/wan22_ti2v_5b/train_stage0p5_fm_153f.yaml")
    runtime = SimpleNamespace(codec=None, config=config)
    with pytest.raises(BackendContractError, match="preencoded-only training"):
        TrainingWanGenerationAdapter(runtime)


def test_stage0p5_runtime_validate_calls_the_common_generation_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import solarwm.backends.wan22.runtime.inference as inference_module
    from solarwm.backends.wan22.runtime.stage0p5 import Wan5BStage0p5Runtime

    config = _config("configs/examples/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml")
    config["runtime"]["output_dir"] = str(tmp_path / "run")
    runtime = Wan5BStage0p5Runtime.__new__(Wan5BStage0p5Runtime)
    runtime.config = config
    runtime.codec = object()
    runtime._global_step = 12
    provider = object()
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        inference_module,
        "TrainingWanGenerationAdapter",
        lambda value: provider if value is runtime else None,
    )

    def fake_runner(
        runner_config: object,
        *,
        provider: object,
        output_dir: Path,
    ) -> WanGenerationSummary:
        captured.update(
            config=runner_config,
            provider=provider,
            output_dir=output_dir,
        )
        return WanGenerationSummary(
            output_dir=output_dir,
            family="wan22_ti2v_5b",
            cases=1,
            passes=("live", "ema"),
            weights_ids={"live": "live-id", "ema": "ema-id"},
            complete_digest="c" * 64,
        )

    monkeypatch.setattr(inference_module, "run_wan_validation", fake_runner)
    report = runtime.validate(12)

    assert captured == {
        "config": config,
        "provider": provider,
        "output_dir": tmp_path / "run/validation/.staging/step-000012",
    }
    assert report["schema"] == "solarwm.wan22-training-validation.v1"
    assert report["generation"]["output_dir"] == str(
        tmp_path / "run/validation/.staging/step-000012"
    )
