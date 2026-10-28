from __future__ import annotations

import json
from collections import Counter
from contextlib import ExitStack

import pytest

from solar.backends.minimax_h3.artifacts import H3PreencodedStream, h3_encoder_contract
from solar.backends.minimax_h3.runtime import _reader
from solar.errors import DataContractError
from solar.runtime import Topology


@pytest.fixture
def readers(tmp_path):
    # Source-ordered rows revisit 64 physical shards non-consecutively, so the
    # 12,000 groups exceed the old buffer despite the small shard inventory.
    rows = [
        {
            "sample_id": f"{'first' if index < 6000 else 'second'}-{index}",
            "key": f"sample-{index}",
            "shard": f"shards/{index % 64}.tar",
            "start_frame": index % 3,
            "epoch_repeats": 2 if index % 11 == 0 else 1,
            "shard_size": 1,
            "shard_generation": "1",
            "h3_preencoded_member": f"sample-{index}.safetensors",
            "h3_provenance_member": f"sample-{index}.provenance.json",
            "manifest_member": f"sample-{index}.json",
        }
        for index in range(12000)
    ]
    for name, values in (
        ("train", rows),
        ("test", [dict(row, epoch_repeats=1) for row in rows[:8]]),
    ):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))
    contract = tmp_path / "encoder.json"
    contract.write_text(json.dumps(h3_encoder_contract().as_dict()))
    topology = Topology(1, 0, 1, 0)

    with ExitStack() as stack:

        def make(*, direct=False, fixed_validation=False, **overrides):
            data = {
                "transport": {"kind": "local", "root": str(tmp_path)},
                "train_index": "train.jsonl",
                "test_index": "test.jsonl",
                "encoder_contract_path": str(contract),
                "train_target_latents": 47,
                "seed": 42,
                "num_workers": 2,
                **overrides,
            }
            if direct:
                stream = H3PreencodedStream(
                    root=str(tmp_path),
                    index=str(tmp_path / "train.jsonl"),
                    topology=topology,
                    seed=42,
                    encoder_contract_path=str(contract),
                    num_workers=2,
                    **overrides,
                )
            else:
                stream = _reader(
                    {
                        "data": data,
                        "train": {"stage": "stage0p5"},
                        "validation": {"selection_seed": 42, "noise_seed": 42},
                    },
                    topology,
                    index_field="test_index" if fixed_validation else "train_index",
                    fixed_validation=fixed_validation,
                    fixed_validation_sample_count=8 if fixed_validation else None,
                )
            stack.callback(stream.close)
            return stream

        yield make


def _occurrences(stream):
    return Counter(
        (plan.sample_id, plan.repeat_ordinal, plan.source_frame_indices)
        for worker in stream.plan.workers
        for plan in worker.plans
    )


@pytest.mark.parametrize("direct", [False, True])
def test_default_mixes_source_ordered_groups_without_changing_occurrences(readers, direct):
    fixed = readers(direct=direct)
    legacy = readers(direct=direct, shuffle_buffer=4096)
    assert fixed.sampling.shuffle_buffer == 32768
    assert _occurrences(fixed) == _occurrences(legacy)

    # These are candidate plans, before tensor reads and camera filtering.
    def second_source_fraction(stream):
        return (
            sum(stream.plan.next_plan().sample_id.startswith("second-") for _ in range(1024)) / 1024
        )

    assert second_source_fraction(legacy) == 0
    assert 0.45 < second_source_fraction(fixed) < 0.55


def test_strict_resume_rejects_changed_plan_and_accepts_original_buffer(readers):
    legacy = readers(shuffle_buffer=4096)
    for _ in range(19):
        legacy.plan.next_plan()
    state = legacy.state_dict()

    with pytest.raises(DataContractError, match="worker plan changed at resume"):
        readers().load_state_dict(state)

    resumed = readers(shuffle_buffer=4096)
    resumed.load_state_dict(state)
    assert resumed.state_dict() == state
    assert [resumed.plan.next_plan() for _ in range(100)] == [
        legacy.plan.next_plan() for _ in range(100)
    ]


def test_default_resume_preserves_worker_cursors_across_epoch_boundary(readers):
    original = readers()
    count = sum(len(worker.plans) for worker in original.plan.workers)
    for _ in range(count - 3):
        original.plan.next_plan()
    state = original.state_dict()

    resumed = readers()
    resumed.load_state_dict(state)
    assert resumed.state_dict() == state
    assert [resumed.plan.next_plan() for _ in range(100)] == [
        original.plan.next_plan() for _ in range(100)
    ]


def test_training_buffer_does_not_change_fixed_validation(readers):
    fixed = readers(fixed_validation=True)
    legacy = readers(fixed_validation=True, shuffle_buffer=4096)
    assert fixed.sampling.shuffle_buffer == legacy.sampling.shuffle_buffer == 1
    assert fixed.state_dict() == legacy.state_dict()
    assert fixed.rows == legacy.rows
