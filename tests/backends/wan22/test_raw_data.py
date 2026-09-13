from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from solarwm.backends.wan22.runtime import data as raw_data
from solarwm.data.archive import RawSample
from solarwm.data.index import IndexRow
from solarwm.data.sampling import SamplePlan, SamplingConfig
from solarwm.errors import DataContractError


@pytest.mark.parametrize("count", [1, 8, 9, 81])
@pytest.mark.parametrize(
    ("source", "target"),
    [
        ((12, 16), (12, 16)),
        ((24, 32), (12, 16)),
        ((9, 13), (15, 21)),
        ((19, 31), (10, 12)),
        ((31, 19), (12, 10)),
        ((19, 31), (19, 20)),
    ],
)
def test_video_preprocessing_is_pixel_exact_across_frame_chunks(
    count: int,
    source: tuple[int, int],
    target: tuple[int, int],
) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    frames = torch.randint(
        0, 256, (count, *source, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(42)
    )
    original = frames.clone()
    # The full-clip path is the reference: do not duplicate chunking here.
    expected = frames.permute(0, 3, 1, 2).contiguous().float() / 255.0
    height, width = target
    if source != target:
        scale = max(height / source[0], width / source[1])
        resized = (round(source[0] * scale), round(source[1] * scale))
        if resized != source:
            expected = functional.interpolate(
                expected, size=resized, mode="bilinear", align_corners=False
            )
        top, left = (resized[0] - height) // 2, (resized[1] - width) // 2
        expected = expected[:, :, top : top + height, left : left + width]
    expected = (expected * 2.0 - 1.0).to(torch.bfloat16)

    actual = raw_data._preprocess_video_frames(frames, height=height, width=width)

    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)
    assert torch.equal(frames, original)


def test_decode_video_preserves_selected_frame_order_and_uint8_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    torch = pytest.importorskip("torch")
    frames = torch.arange(10 * 6 * 8 * 3, dtype=torch.int16).reshape(10, 6, 8, 3)
    selected = [9, 1, 1, 3, 0, 7, 2, 4, 6]
    calls = []

    class Reader:
        def __init__(self, payload: Any, *, num_threads: int) -> None:
            assert payload.read() == b"video"
            assert num_threads == 1

        def get_batch(self, indices: list[int]) -> Any:
            calls.append(indices)
            return frames[indices]

    monkeypatch.setitem(
        sys.modules,
        "decord",
        SimpleNamespace(
            bridge=SimpleNamespace(set_bridge=lambda value: calls.append(value)),
            VideoReader=Reader,
        ),
    )
    actual = raw_data.decode_video(b"video", selected, height=6, width=8)
    expected = frames[selected].to(torch.uint8).permute(0, 3, 1, 2).contiguous().float() / 255.0
    expected = (expected * 2.0 - 1.0).to(torch.bfloat16)
    assert torch.equal(actual, expected)
    assert calls == ["torch", selected]


def test_video_preprocessing_bounds_full_resolution_interpolation_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    batches = []
    interpolate = functional.interpolate

    def record(frames: Any, **kwargs: Any) -> Any:
        batches.append(frames.shape[0])
        return interpolate(frames, **kwargs)

    monkeypatch.setattr(functional, "interpolate", record)
    result = raw_data._preprocess_video_frames(
        torch.zeros((81, 24, 32, 3), dtype=torch.uint8),
        height=12,
        width=16,
    )
    assert batches == [8] * 10 + [1]
    assert result.shape == (81, 3, 12, 16)


def _plan() -> SamplePlan:
    return SamplePlan(
        sample_id="dataset/bad-camera",
        key="bad-camera",
        shard="raw/shard.tar",
        row_ordinal=0,
        repeat_ordinal=0,
        epoch=0,
        start_frame=0,
        source_frame_indices=tuple(range(81)),
        reader_rank=7,
        worker_id=0,
    )


class _Reader:
    def __init__(self, result: RawSample | Exception) -> None:
        self.result = result

    def materialize(self, _plan: SamplePlan) -> RawSample:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _raw_sample(*, finite: bool) -> RawSample:
    plan = _plan()
    return RawSample(
        plan=plan,
        index_values={"fps": 16.0},
        caption="caption",
        scene="scene",
        manifest={
            "video": {"fps": 16.0},
            "camera": {
                "array_key": "c2w",
                "convention": raw_data.CAMERA_CONVENTION,
                "dtype": "float32",
                "finite": finite,
                "magnitude_audit_seconds": 10.0,
                "max_camera_abs": 20.0,
                "max_rel_translation": 20.0,
                "shape": [81, 4, 4],
            },
        },
        members={"video_member": b"video", "camera_member": b"camera"},
    )


def _config() -> dict[str, Any]:
    return {
        "data": {
            "height": 480,
            "width": 832,
            "fps": 16.0,
            "max_rel_translation": 20.0,
            "max_camera_abs": 20.0,
            "camera_array_key": "c2w",
        },
        "model": {"frame_sequence_length": 1560},
    }


def _batch_config(*, micro_batch_size: int = 1) -> dict[str, Any]:
    config = _config()
    config["data"].update(
        train_index="train.jsonl",
        pixel_frames=81,
        random_start=True,
        seed=42,
        shuffle_buffer=1,
        partition_mode="global_occurrence",
        transport={"root": "/read-only-data"},
    )
    config["train"] = {"micro_batch_size": micro_batch_size}
    return config


class _ShardContext:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def __enter__(self) -> _ShardContext:
        return self

    def __exit__(self, *_args: Any) -> None:
        pass


class _OnePlanPerEpochSampler:
    def __init__(self, plan: SamplePlan) -> None:
        self.plan = plan

    def iter_epoch(self, _epoch: int) -> Any:
        yield self.plan


def _patch_raw_iterator(
    monkeypatch: pytest.MonkeyPatch,
    materialize: Any,
) -> None:
    plan = _plan()
    monkeypatch.setattr(raw_data, "resolve_index_path", lambda *_args: "train.jsonl")
    monkeypatch.setattr(raw_data, "read_index", lambda *_args: (object(),))
    monkeypatch.setattr(raw_data, "resolver_from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(raw_data, "TarShardReader", _ShardContext)
    monkeypatch.setattr(raw_data, "RawSampleReader", lambda *_args: object())
    monkeypatch.setattr(
        raw_data,
        "CanonicalSampler",
        lambda *_args: _OnePlanPerEpochSampler(plan),
    )
    monkeypatch.setattr(raw_data, "_materialize_raw_sample", materialize)


def _topology() -> SimpleNamespace:
    return SimpleNamespace(
        dp_rank=0,
        dp_world_size=1,
        node_id=0,
        node_count=1,
        local_dp_rank=0,
        local_dp_world_size=1,
    )


def test_materialize_raw_sample_skips_nonfinite_camera_manifest(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(
        raw_data,
        "decode_video",
        lambda *_args, **_kwargs: torch.zeros((81, 3, 1, 1)),
    )

    result = raw_data._materialize_raw_sample(
        _Reader(_raw_sample(finite=False)),
        _plan(),
        _config(),
    )

    assert result is None
    assert (
        "[wds][rank7] skip bad-camera: DataContractError: "
        "raw Wan manifest must attest camera finite=true"
    ) in capsys.readouterr().out


@pytest.mark.parametrize(
    ("failure_site", "error"),
    [
        ("materialize", OSError("broken tar member")),
        ("decode", ValueError("broken decoded sample")),
    ],
)
def test_materialize_raw_sample_skips_any_sample_local_exception(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_site: str,
    error: Exception,
) -> None:
    sample = _raw_sample(finite=True)
    reader = _Reader(error if failure_site == "materialize" else sample)
    if failure_site == "decode":

        def fail_decode(*_args: Any) -> raw_data.DecodedWanSample:
            raise error

        monkeypatch.setattr(raw_data, "decode_raw_sample", fail_decode)

    assert raw_data._materialize_raw_sample(reader, _plan(), _config()) is None
    output = capsys.readouterr().out
    assert "[wds][rank7] skip bad-camera:" in output
    assert f"{type(error).__name__}: {error}" in output


def test_materialize_raw_sample_skips_a_prefetch_failure(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_prepare(_plan: SamplePlan) -> None:
        raise OSError("temporary object download failure")

    assert (
        raw_data._materialize_raw_sample(
            _Reader(_raw_sample(finite=True)),
            _plan(),
            _config(),
            prepare=fail_prepare,
        )
        is None
    )
    assert (
        "[wds][rank7] skip bad-camera: OSError: temporary object download failure"
        in capsys.readouterr().out
    )


def test_raw_iterator_fails_only_after_a_complete_epoch_has_no_healthy_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_raw_iterator(monkeypatch, lambda *_args, **_kwargs: None)
    batches = raw_data.iter_raw_batches(_batch_config(), _topology())

    with pytest.raises(
        RuntimeError,
        match=r"reader rank=0 worker=0 emitted no samples in epoch 0",
    ):
        next(batches)


def test_raw_iterator_does_not_treat_an_incomplete_batch_as_an_empty_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    healthy = object()
    _patch_raw_iterator(monkeypatch, lambda *_args, **_kwargs: healthy)
    monkeypatch.setattr(raw_data, "collate_raw_samples", lambda samples: tuple(samples))
    batches = raw_data.iter_raw_batches(
        _batch_config(micro_batch_size=2),
        _topology(),
    )

    assert next(batches) == (healthy, healthy)


def _multiplexer_rows() -> tuple[IndexRow, ...]:
    return tuple(
        IndexRow.from_mapping(
            ordinal,
            {
                "sample_id": f"dataset/sample-{ordinal:02d}",
                "key": f"sample-{ordinal:02d}",
                "shard": f"raw/shard-{ordinal // 4:02d}.tar",
                "num_frames": 121,
                "fps": 16.0,
            },
        )
        for ordinal in range(24)
    )


def _multiplexer(*, rows: tuple[IndexRow, ...] | None = None) -> raw_data.WanPlanMultiplexer:
    return raw_data.WanPlanMultiplexer(
        _multiplexer_rows() if rows is None else rows,
        SamplingConfig(
            seed=42,
            pixel_frames=81,
            random_start=True,
            clip_seconds=5.0,
            output_fps=16.0,
            shuffle_buffer=4,
            partition_mode="global_occurrence",
        ),
        _topology(),
        num_workers=3,
    )


def _next_plan_identity(
    value: raw_data.WanPlanMultiplexer,
) -> tuple[int, str, int, int]:
    worker = value.next_worker
    plan = value.next_plan(worker)
    value.finish_batch(worker)
    return worker, plan.sample_id, plan.epoch, plan.start_frame


def test_wan_plan_multiplexer_resume_replays_the_exact_next_sample() -> None:
    uninterrupted = _multiplexer()
    for _ in range(11):
        _next_plan_identity(uninterrupted)
    state = uninterrupted.state_dict()
    expected = [_next_plan_identity(uninterrupted) for _ in range(40)]

    resumed = _multiplexer()
    resumed.load_state_dict(state)
    assert [_next_plan_identity(resumed) for _ in range(40)] == expected


def test_wan_plan_fingerprint_is_not_recomputed_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = _multiplexer()
    state = reader.state_dict()
    monkeypatch.setattr(
        raw_data,
        "plan_fingerprint",
        lambda _plans: pytest.fail("plan fingerprint was recomputed for a batch"),
    )

    reader.record_worker_progress(state["workers"][0])
    assert reader.state_dict()["workers"][0] == state["workers"][0]


def test_wan_plan_multiplexer_rejects_changed_sample_plan() -> None:
    original = _multiplexer()
    state = original.state_dict()
    changed_rows = list(_multiplexer_rows())
    changed_rows[0] = IndexRow.from_mapping(
        0,
        {
            "sample_id": "dataset/replaced",
            "key": "replaced",
            "shard": "raw/shard-00.tar",
            "num_frames": 121,
            "fps": 16.0,
        },
    )
    changed = _multiplexer(rows=tuple(changed_rows))
    with pytest.raises(DataContractError, match="sample plan changed"):
        changed.load_state_dict(state)


def test_stateful_raw_reader_keeps_multiworker_prefetch_and_resumes_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    rows = _multiplexer_rows()
    monkeypatch.setattr(raw_data, "resolve_index_path", lambda *_args: "train.jsonl")
    monkeypatch.setattr(raw_data, "read_index", lambda *_args: rows)
    monkeypatch.setattr(raw_data, "resolver_from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(raw_data, "TarShardReader", _ShardContext)
    monkeypatch.setattr(raw_data, "RawSampleReader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(raw_data, "build_shard_prefetcher", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        raw_data,
        "_materialize_raw_sample",
        lambda _reader, plan, _config, **_kwargs: plan,
    )
    monkeypatch.setattr(
        raw_data,
        "collate_raw_samples",
        lambda plans: {
            "sample_ids": tuple(plan.sample_id for plan in plans),
            "start_frames": tuple(plan.start_frame for plan in plans),
        },
    )
    config = _batch_config()
    config["data"].update(num_workers=2, prefetch_factor=3)

    uninterrupted = raw_data.build_raw_dataloader(config, _topology())
    cpu_rng = torch.random.get_rng_state().clone()
    uninterrupted._start_loader()
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    prefix = [next(uninterrupted) for _ in range(9)]
    assert uninterrupted._loader.num_workers == 2
    assert uninterrupted._loader.prefetch_factor == 3
    assert uninterrupted._loader.persistent_workers is True
    state = uninterrupted.checkpoint_state_dict()
    assert uninterrupted._iterator is None
    expected = [next(uninterrupted) for _ in range(15)]
    uninterrupted.close()

    resumed = raw_data.build_raw_dataloader(config, _topology())
    resumed.load_state_dict(state)
    actual = [next(resumed) for _ in range(15)]
    resumed.close()

    assert actual == expected
    reference = raw_data.WanPlanMultiplexer(
        rows,
        SamplingConfig(
            seed=42,
            pixel_frames=81,
            random_start=True,
            clip_seconds=5.0,
            output_fps=16.0,
            shuffle_buffer=1,
            partition_mode="global_occurrence",
        ),
        _topology(),
        num_workers=2,
    )
    reference_identity = []
    for _ in range(24):
        worker = reference.next_worker
        plan = reference.next_plan(worker)
        reference.finish_batch(worker)
        reference_identity.append((plan.sample_id, plan.start_frame))
    observed_identity = [
        (str(batch["sample_ids"][0]), int(batch["start_frames"][0]))
        for batch in (*prefix, *expected)
    ]
    assert observed_identity == reference_identity


def test_stateful_raw_resume_with_microbatch_skips_and_epoch_crossing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    rows = _multiplexer_rows()
    monkeypatch.setattr(raw_data, "resolve_index_path", lambda *_args: "train.jsonl")
    monkeypatch.setattr(raw_data, "read_index", lambda *_args: rows)
    monkeypatch.setattr(raw_data, "resolver_from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(raw_data, "TarShardReader", _ShardContext)
    monkeypatch.setattr(raw_data, "RawSampleReader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(raw_data, "build_shard_prefetcher", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        raw_data,
        "_materialize_raw_sample",
        lambda _reader, plan, _config, **_kwargs: None if plan.row_ordinal % 11 == 0 else plan,
    )
    monkeypatch.setattr(
        raw_data,
        "collate_raw_samples",
        lambda plans: {
            "sample_ids": tuple(plan.sample_id for plan in plans),
            "epochs": tuple(plan.epoch for plan in plans),
            "start_frames": tuple(plan.start_frame for plan in plans),
        },
    )
    config = _batch_config(micro_batch_size=2)
    config["data"].update(num_workers=2, prefetch_factor=3, shuffle_buffer=4)

    uninterrupted = raw_data.build_raw_dataloader(config, _topology())
    for _ in range(9):
        next(uninterrupted)
    state = uninterrupted.checkpoint_state_dict()
    expected = [next(uninterrupted) for _ in range(15)]
    uninterrupted.close()

    resumed = raw_data.build_raw_dataloader(config, _topology())
    resumed.load_state_dict(state)
    actual = [next(resumed) for _ in range(15)]
    resumed.close()

    assert actual == expected
    assert any(len(set(batch["epochs"])) > 1 for batch in actual)


@pytest.mark.parametrize("tail_bad", [False, True])
def test_stateful_raw_resume_at_epoch_tail_preserves_prior_epoch_health(
    monkeypatch: pytest.MonkeyPatch,
    tail_bad: bool,
) -> None:
    rows = _multiplexer_rows()
    plan = _multiplexer()
    worker_state = plan.state_dict()["workers"][0]
    worker_state["cursor"] = len(plan.workers[0].plans) - int(tail_bad)
    worker_state["healthy_samples"] = max(1, int(worker_state["cursor"]))
    monkeypatch.setattr(raw_data, "resolver_from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(raw_data, "TarShardReader", _ShardContext)
    monkeypatch.setattr(raw_data, "RawSampleReader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(raw_data, "build_shard_prefetcher", lambda *_args, **_kwargs: None)
    calls = 0

    def materialize(_reader: object, candidate: SamplePlan, *_args: object, **_kwargs: object):
        nonlocal calls
        calls += 1
        return None if tail_bad and calls == 1 else candidate

    monkeypatch.setattr(raw_data, "_materialize_raw_sample", materialize)
    monkeypatch.setattr(
        raw_data,
        "collate_raw_samples",
        lambda samples: {"epochs": tuple(sample.epoch for sample in samples)},
    )
    config = _batch_config()
    config["data"]["shuffle_buffer"] = 4
    stream = raw_data._iter_stateful_raw_batches(
        config,
        _topology(),
        rows=rows,
        worker_id=0,
        num_workers=3,
        worker_state=worker_state,
    )

    assert next(stream)["epochs"] == (1,)
    stream.close()


def test_stateful_raw_reader_preserves_empty_epoch_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("torch")
    rows = _multiplexer_rows()
    monkeypatch.setattr(raw_data, "resolve_index_path", lambda *_args: "train.jsonl")
    monkeypatch.setattr(raw_data, "read_index", lambda *_args: rows)
    monkeypatch.setattr(raw_data, "resolver_from_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(raw_data, "TarShardReader", _ShardContext)
    monkeypatch.setattr(raw_data, "RawSampleReader", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(raw_data, "build_shard_prefetcher", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(raw_data, "_materialize_raw_sample", lambda *_args, **_kwargs: None)
    config = _batch_config()
    config["data"].update(num_workers=2, prefetch_factor=2)
    reader = raw_data.build_raw_dataloader(config, _topology())

    with pytest.raises(RuntimeError, match="emitted no samples in epoch 0"):
        next(reader)
    reader.close()
