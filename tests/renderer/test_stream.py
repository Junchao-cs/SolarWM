from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from solar.errors import DataContractError
from solar.renderer.data import H3OmniStream

CONTRACT = (
    Path(__file__).resolve().parents[2] / "solar" / "renderer" / "assets" / "encoder_contract.json"
)


def rows():
    return [
        {"id": f"clip-{index}", "member": f"clip-{index}.safetensors", "latent_frames": 47}
        for index in range(4)
    ]


def make_stream(root, *, samples=None, seed=23, world=1, length_buckets=False, shuffle=True):
    index = root / "train.jsonl"
    if samples is not None or not index.exists():
        index.write_text(
            "".join(json.dumps(row) + "\n" for row in (rows() if samples is None else samples))
        )
    return H3OmniStream(
        root=str(root),
        index=str(index),
        topology=SimpleNamespace(dp_rank=0, dp_world_size=world),
        seed=seed,
        encoder_contract_path=str(CONTRACT),
        shuffle=shuffle,
        length_buckets=length_buckets,
    )


def test_same_name_and_size_index_with_changed_members_has_different_identity(tmp_path):
    original = make_stream(tmp_path)
    index_bytes = (tmp_path / "train.jsonl").stat().st_size
    changed_rows = rows()
    changed_rows[0]["member"] = "alt--0.safetensors"
    changed = make_stream(tmp_path, samples=changed_rows)
    assert (tmp_path / "train.jsonl").stat().st_size == index_bytes
    assert original.current_plan_fingerprint != changed.current_plan_fingerprint
    with pytest.raises(DataContractError):
        changed.load_state_dict(original.state_dict())


@pytest.mark.parametrize(
    "options",
    [
        {"seed": 24},
        {"world": 2},
        {"length_buckets": True},
        {"shuffle": False},
    ],
)
def test_reader_schedule_changes_invalidate_resume(tmp_path, options):
    original = make_stream(tmp_path)
    changed = make_stream(tmp_path, **options)
    assert original.current_plan_fingerprint != changed.current_plan_fingerprint
    with pytest.raises(DataContractError):
        changed.load_state_dict(original.state_dict())


def test_resume_restores_exact_sample_sequence_across_epoch_boundary(tmp_path, monkeypatch):
    original = make_stream(tmp_path, length_buckets=True)
    monkeypatch.setattr(original, "read", lambda row: row["id"])
    for _ in range(3):
        original.next()
    state = original.state_dict()
    expected = [original.next() for _ in range(9)]
    restored = make_stream(tmp_path, length_buckets=True)
    monkeypatch.setattr(restored, "read", lambda row: row["id"])
    restored.load_state_dict(state)
    assert [restored.next() for _ in range(9)] == expected


@pytest.mark.parametrize(
    "updates",
    [
        {"epoch": -1},
        {"epoch": True},
        {"epoch": 1.5},
        {"cursor": -1},
        {"cursor": 5},
        {"cursor": True},
        {"cursor": 1.5},
        {"plan_fingerprint": "different-data"},
    ],
)
def test_resume_rejects_invalid_reader_state_without_changing_cursor(tmp_path, updates):
    stream = make_stream(tmp_path)
    original = stream.state_dict()
    state = {**original, **updates}
    with pytest.raises(DataContractError):
        stream.load_state_dict(state)
    assert stream.state_dict() == original


def test_duplicate_sample_ids_are_rejected(tmp_path):
    samples = rows()
    samples[1]["id"] = samples[0]["id"]
    with pytest.raises(DataContractError):
        make_stream(tmp_path, samples=samples)


def test_failed_member_is_not_skipped_or_committed(tmp_path, monkeypatch):
    stream = make_stream(tmp_path, shuffle=False)
    calls = []

    def broken_member(row):
        calls.append(row["id"])
        raise DataContractError("corrupt sample")

    monkeypatch.setattr(stream, "read", broken_member)
    before = stream.state_dict()
    with pytest.raises(DataContractError, match="corrupt sample"):
        stream.next()
    assert calls == ["clip-0"]
    assert stream.state_dict() == before
    monkeypatch.setattr(stream, "read", lambda row: row["id"])
    assert stream.next() == "clip-0"
