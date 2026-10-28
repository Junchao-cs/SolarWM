from __future__ import annotations

import random

from solar.data.index import IndexRow
from solar.data.sampling import (
    CanonicalSampler,
    ReaderIdentity,
    SamplingConfig,
    plan_fingerprint,
)
from solar.runtime.topology import Topology


def _rows() -> tuple[IndexRow, ...]:
    raw = [
        {"sample_id": "a", "key": "ka", "shard": "x/0.tar", "epoch_repeats": 2, "num_frames": 100},
        {"sample_id": "b", "key": "kb", "shard": "x/0.tar", "num_frames": 100},
        {"sample_id": "c", "key": "kc", "shard": "x/1.tar", "epoch_repeats": 3, "num_frames": 100},
        {"sample_id": "d", "key": "kd", "shard": "x/2.tar", "num_frames": 100},
    ]
    return tuple(IndexRow.from_mapping(i, row) for i, row in enumerate(raw))


def _global_reference(rows, *, seed: int, reader: int, total: int):
    rng = random.Random(seed + reader * 100003)
    selected = []
    virtual_index = 0
    for row in rows:
        for repeat in range(row.epoch_repeats):
            if virtual_index % total == reader:
                selected.append((row, repeat))
            virtual_index += 1
    buffer = []
    output = []
    for occurrence in selected:
        buffer.append(occurrence)
        if len(buffer) >= 2:
            row, repeat = buffer.pop(rng.randrange(len(buffer)))
            output.append((row.sample_id, repeat, rng.randint(0, 19)))
    while buffer:
        row, repeat = buffer.pop(rng.randrange(len(buffer)))
        output.append((row.sample_id, repeat, rng.randint(0, 19)))
    return output


def test_global_occurrence_matches_reference_rng_order() -> None:
    rows = _rows()
    config = SamplingConfig(seed=17, pixel_frames=81, shuffle_buffer=2)
    identity = ReaderIdentity(rank=0, world_size=2)
    plans = list(CanonicalSampler(rows, config, identity).iter_epoch(0))
    actual = [(plan.sample_id, plan.repeat_ordinal, plan.start_frame) for plan in plans]
    assert actual == _global_reference(rows, seed=17, reader=0, total=2)


def test_shuffle_buffer_one_does_not_consume_a_shuffle_draw() -> None:
    rows = _rows()
    seed = 73
    plans = list(
        CanonicalSampler(
            rows,
            SamplingConfig(seed=seed, pixel_frames=81, shuffle_buffer=1),
            ReaderIdentity(rank=0, world_size=1),
        ).iter_epoch(0)
    )
    rng = random.Random(seed)
    expected = []
    for row in rows:
        for repeat in range(row.epoch_repeats):
            expected.append((row.sample_id, repeat, rng.randint(0, 19)))

    assert [(plan.sample_id, plan.repeat_ordinal, plan.start_frame) for plan in plans] == expected


def test_one_sampler_preserves_rng_stream_across_epochs() -> None:
    rows = _rows()
    seed = 91
    sampler = CanonicalSampler(
        rows,
        SamplingConfig(seed=seed, pixel_frames=81, shuffle_buffer=1),
        ReaderIdentity(rank=0, world_size=1),
    )
    first = list(sampler.iter_epoch(0))
    second = list(sampler.iter_epoch(1))
    rng = random.Random(seed)
    starts = [rng.randint(0, 19) for _ in range(2 * sum(row.epoch_repeats for row in rows))]

    assert [plan.start_frame for plan in [*first, *second]] == starts


def test_sampler_is_transport_independent_and_repeatable() -> None:
    rows = _rows()
    config = SamplingConfig(seed=41, pixel_frames=81, shuffle_buffer=3)
    identity = ReaderIdentity(rank=1, world_size=2)
    left = list(CanonicalSampler(rows, config, identity).iter_epoch(0))
    right = list(CanonicalSampler(rows, config, identity).iter_epoch(0))
    assert plan_fingerprint(left) == plan_fingerprint(right)
    assert all(not plan.shard.startswith("/") for plan in left)


def test_node_shard_sp_peers_share_reader_identity() -> None:
    rows = _rows()
    config = SamplingConfig(
        seed=29,
        pixel_frames=81,
        shuffle_buffer=2,
        partition_mode="node_shard",
    )
    # Raw ranks 0 and 1 in an SP2 group both map to logical local rank 0.
    identity = ReaderIdentity(
        rank=0,
        world_size=2,
        local_rank=0,
        local_world_size=1,
        node_id=0,
        node_count=2,
    )
    left = list(CanonicalSampler(rows, config, identity).iter_epoch(0))
    right = list(CanonicalSampler(rows, config, identity).iter_epoch(0))
    assert left == right


def test_raw_sp_peers_map_to_same_logical_reader() -> None:
    peer0 = ReaderIdentity.from_topology(
        Topology(8, 2, 8, 2, sp_size=2), worker_id=3, num_workers=4
    )
    peer1 = ReaderIdentity.from_topology(
        Topology(8, 3, 8, 3, sp_size=2), worker_id=3, num_workers=4
    )
    assert peer0 == peer1
    assert peer0.rank == 1
    assert peer0.world_size == 4


def test_non_random_raw_start_is_zero_but_fixed_indexes_are_explicit() -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "fixed",
            "key": "fixed",
            "shard": "x/0.tar",
            "num_frames": 100,
            "start_frame": 7,
        },
    )
    identity = ReaderIdentity(rank=0, world_size=1)
    raw = list(
        CanonicalSampler(
            (row,), SamplingConfig(seed=1, pixel_frames=81, random_start=False), identity
        ).iter_epoch(0)
    )
    fixed = list(
        CanonicalSampler(
            (row,),
            SamplingConfig(
                seed=1,
                pixel_frames=81,
                random_start=False,
                fixed_start_from_index=True,
            ),
            identity,
        ).iter_epoch(0)
    )
    assert raw[0].start_frame == 0
    assert fixed[0].start_frame == 7


def test_node_shard_ownership_uses_complete_index_shard_identity() -> None:
    def rows(prefix: str) -> tuple[IndexRow, ...]:
        return tuple(
            IndexRow.from_mapping(
                index,
                {
                    "sample_id": f"sample-{index}",
                    "key": f"sample-{index}",
                    "shard": f"{prefix}abot/shards/part-{index:03d}.tar",
                    "num_frames": 100,
                },
            )
            for index in range(64)
        )

    config = SamplingConfig(
        seed=42,
        pixel_frames=81,
        shuffle_buffer=7,
        partition_mode="node_shard",
    )
    for node_id in range(4):
        identity = ReaderIdentity(
            rank=node_id,
            world_size=4,
            node_id=node_id,
            node_count=4,
        )
        direct = CanonicalSampler(rows(""), config, identity).node_shard_groups(0)
        relocated = CanonicalSampler(
            rows("latent-wds/model-generation/"), config, identity
        ).node_shard_groups(0)
        assert [group_rows[0].sample_id for _, group_rows in direct] != [
            group_rows[0].sample_id for _, group_rows in relocated
        ]


def test_node_shard_collection_relative_scope_ignores_release_root() -> None:
    def rows(prefix: str) -> tuple[IndexRow, ...]:
        return tuple(
            IndexRow.from_mapping(
                index,
                {
                    "sample_id": f"sample-{index}",
                    "key": f"sample-{index}",
                    "shard": f"{prefix}abot/shards/part-{index:03d}.tar",
                    "num_frames": 100,
                },
            )
            for index in range(64)
        )

    config = SamplingConfig(
        seed=42,
        pixel_frames=81,
        shuffle_buffer=7,
        partition_mode="node_shard",
        shard_partition_scope="collection_relative",
    )
    for node_id in range(4):
        identity = ReaderIdentity(
            rank=node_id,
            world_size=4,
            node_id=node_id,
            node_count=4,
        )
        direct = CanonicalSampler(rows(""), config, identity).node_shard_groups(0)
        relocated = CanonicalSampler(
            rows("latent-wds/model-generation/"), config, identity
        ).node_shard_groups(0)
        assert [group_rows[0].sample_id for _, group_rows in direct] == [
            group_rows[0].sample_id for _, group_rows in relocated
        ]
