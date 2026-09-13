from __future__ import annotations

import io
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest

from solarwm.backends.minimax_h3.artifacts import H3ArtifactBatch, H3CameraFilterError
from solarwm.backends.minimax_h3.raw_data import decode_camera
from solarwm.backends.minimax_h3.validation_stream import (
    H3IndexedValidationStream,
    source_video_id,
    validation_candidates,
)
from solarwm.data import IndexRow
from solarwm.errors import DataContractError


def rows_by_source(count=12):
    rows = []
    for source in ("a", "b", "c"):
        for video in range(count):
            for occurrence in range(2):
                sample = f"{source}/{video}/h3-158f-occ-{occurrence:02d}"
                rows.append(
                    IndexRow.from_mapping(
                        len(rows),
                        dict(
                            sample_id=sample,
                            key=sample,
                            shard=f"{source}.tar",
                            start_frame=occurrence,
                        ),
                    )
                )
    return rows


def test_candidates_balance_uneven_quotas_and_keep_all_reserves_disjoint():
    rows = rows_by_source()
    first = validation_candidates(rows, sample_count=8, seed=42, balance=True)
    assert first == validation_candidates(rows, sample_count=8, seed=42, balance=True)
    assert first != validation_candidates(rows, sample_count=8, seed=43, balance=True)
    assert Counter(pool[0].sample_id.split("/")[0] for pool in first) == dict(a=3, b=3, c=2)
    source_ids = [source_video_id(row) for pool in first for row in pool]
    assert len(source_ids) == len(set(source_ids)) == 36
    assert all(len({row.sample_id.split("/")[0] for row in pool}) == 1 for pool in first)


def test_candidates_reject_too_few_distinct_source_videos():
    with pytest.raises(DataContractError, match="distinct source videos"):
        validation_candidates(rows_by_source(count=1), sample_count=6, seed=42, balance=True)


def bare_stream(*, frozen=False):
    stream = object.__new__(H3IndexedValidationStream)
    stream.stage = "stage2"
    stream.cursor = 1
    stream.count = 4
    stream.noise_seed = 42
    stream.digest = "fixed-index"
    stream.topology = SimpleNamespace(dp_rank=1, dp_world_size=2)
    stream.candidates = validation_candidates(
        rows_by_source(), sample_count=4, seed=42, balance=True
    )
    stream.frozen = frozen
    return stream


def test_camera_rejection_keeps_the_slot_dataset_and_noise_identity():
    stream = bare_stream()
    rejected, accepted = stream.candidates[1][:2]
    visited = []

    def read(row, plan):
        visited.append(row.sample_id)
        if row.sample_id == rejected.sample_id:
            raise H3CameraFilterError("camera guard")
        return H3ArtifactBatch(
            row.sample_id,
            plan.start_frame,
            "train-plan",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            dataset_source="b",
        )

    stream.reader = SimpleNamespace(read_artifact=read)
    batch = stream.next()
    assert visited == [rejected.sample_id, accepted.sample_id]
    assert batch.sample_id == accepted.sample_id
    assert batch.validation_slot == 1 and batch.validation_noise_seed == 1051
    assert batch.plan_fingerprint == "fixed-index"
    assert stream.cursor == 3


def test_frozen_camera_failure_is_not_replaced():
    stream = bare_stream(frozen=True)

    def fail(*args):
        raise H3CameraFilterError("camera guard")

    stream.reader = SimpleNamespace(read_artifact=fail)
    with pytest.raises(H3CameraFilterError, match="camera guard"):
        stream.next()
    assert stream.cursor == 1


def test_raw_camera_170_frame_selection_preserves_the_encoded_prefix():
    count = 200
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (count, 4, 4)).copy()
    poses[:, 0, 3] = np.arange(count) / 100
    intrinsic = np.array([0.8, 0.9, 0.5, 0.5], dtype=np.float32)
    payload = io.BytesIO()
    np.savez(payload, c2w=poses, intrinsics=intrinsic)
    transform = dict(
        source_h=1080,
        source_w=1920,
        resized_h=768,
        resized_w=1365,
        crop_left=10,
        crop_top=0,
        target_h=768,
        target_w=1344,
    )
    first = decode_camera(payload.getvalue(), range(7, 165), transform)
    extended = decode_camera(payload.getvalue(), range(7, 177), transform)
    assert extended[0].shape == (170, 4, 4)
    assert extended[1].shape == (170, 3, 3)
    for short, full in zip(first, extended, strict=True):
        np.testing.assert_array_equal(short, full[:158])
    np.testing.assert_array_equal(extended[0][-1], poses[176])


def test_collective_refill_excludes_all_previously_selected_videos():
    stream = bare_stream()
    stream.cursor = 0
    stream.count = 6
    stream.topology = SimpleNamespace(dp_rank=0, dp_world_size=1)
    stream.config = {"validation": {"selection_seed": 42, "balance_by_dataset": True}}
    stream.candidates = validation_candidates(
        rows_by_source(), sample_count=6, seed=42, balance=True
    )
    rejected = {row.sample_id for row in stream.candidates[1]}

    def read(row, plan):
        if row.sample_id in rejected:
            raise H3CameraFilterError("camera guard")
        return H3ArtifactBatch(
            row.sample_id,
            plan.start_frame,
            "plan",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    stream.reader = SimpleNamespace(read_artifact=read)
    stream.prepare(SimpleNamespace(is_initialized=lambda: False))
    batches = [stream.next() for _ in range(6)]
    assert len({batch.sample_id for batch in batches}) == 6
    assert Counter(batch.sample_id.split("/")[0] for batch in batches) == dict(a=2, b=2, c=2)
    assert batches[1].sample_id not in rejected
    assert batches[1].sample_id != batches[4].sample_id
