from __future__ import annotations

import pytest

from solar.errors import ConfigurationError
from solar.runtime.randomness import (
    model_init_seed,
    objective_seed,
    rng_identity,
    stable_validation_seed,
)
from solar.runtime.topology import Topology


def test_model_initialization_namespaces_match_reference_values() -> None:
    assert model_init_seed("wan22_ti2v_5b", 42) == 3_509_493_760 + 42
    assert model_init_seed("minimax_h3", 42) == 1_211_301_888 + 42
    assert model_init_seed("ltx25_video", 42) == 1_280_596_005 + 42


def test_sp_peers_share_objective_rng_but_dp_ranks_do_not() -> None:
    left = rng_identity("wan22_ti2v_5b", 42, Topology(8, 2, 8, 2, sp_size=2))
    peer = rng_identity("wan22_ti2v_5b", 42, Topology(8, 3, 8, 3, sp_size=2))
    other = rng_identity("wan22_ti2v_5b", 42, Topology(8, 4, 8, 4, sp_size=2))
    assert left.objective_seed == peer.objective_seed == objective_seed(42, 1)
    assert left.objective_seed != other.objective_seed
    assert left.sp_rank == 0 and peer.sp_rank == 1


def test_validation_seed_matches_frozen_algorithm() -> None:
    assert stable_validation_seed("validation-noise", 20, 7, 42) == 1679897218


def test_topology_loads_torchrun_coordinates() -> None:
    topology = Topology.from_environ(
        2,
        {
            "WORLD_SIZE": "16",
            "RANK": "11",
            "LOCAL_WORLD_SIZE": "8",
            "LOCAL_RANK": "3",
        },
    )
    assert (topology.dp_world_size, topology.dp_rank, topology.sp_rank) == (8, 5, 1)
    assert (topology.node_count, topology.node_id) == (2, 1)
    with pytest.raises(ConfigurationError, match="missing"):
        Topology.from_environ(1, {})
