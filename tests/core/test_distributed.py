from __future__ import annotations

import sys
import time
from datetime import timedelta
from pathlib import Path

import pytest

from solarwm.errors import BackendContractError
from solarwm.runtime.distributed import (
    DataReadiness,
    assert_peer_fingerprints,
    gather_and_assert_sp_identity,
    identity_fingerprint,
)


def test_identity_hash_is_mapping_order_independent() -> None:
    assert identity_fingerprint({"sample": "a", "noise": 7}) == identity_fingerprint(
        {"noise": 7, "sample": "a"}
    )


def test_peer_mismatch_fails_closed() -> None:
    assert assert_peer_fingerprints(["same", "same"]) == "same"
    with pytest.raises(BackendContractError, match="different"):
        assert_peer_fingerprints(["left", "right"])


def test_sp1_does_not_require_torch_distributed() -> None:
    assert gather_and_assert_sp_identity({"sample": "a"}, sp_size=1)


def test_disabled_data_readiness_preserves_values_and_errors() -> None:
    ready = DataReadiness(enabled=False)
    value = object()
    assert ready.read(lambda: value) is value

    def fail() -> None:
        raise ValueError("original read error")

    with pytest.raises(ValueError, match="original read error"):
        ready.read(fail)


def _data_readiness_worker(rank: int, rendezvous: str) -> None:
    import torch
    import torch.distributed as dist

    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        ready = DataReadiness(enabled=True)
        # Model communication has a shorter deadline than the injected read.
        model_group = dist.new_group(backend="gloo", timeout=timedelta(seconds=1))
        rng = torch.get_rng_state().clone()
        expected = {"sample_id": f"sample-{rank}"}

        def slow_read() -> dict[str, str]:
            if rank == 1:
                time.sleep(1.5)
            return expected

        assert ready.read(slow_read) is expected
        assert torch.equal(torch.get_rng_state(), rng)
        value = torch.tensor([rank + 1])
        dist.all_reduce(value, group=model_group)
        assert value.item() == 3

        def broken_read() -> dict[str, str]:
            if rank == 1:
                raise ValueError("injected reader failure")
            return expected

        with pytest.raises(
            BackendContractError, match="rank 1: ValueError: injected reader failure"
        ):
            ready.read(broken_read)
    finally:
        dist.destroy_process_group()


def test_data_readiness_isolates_slow_reads_and_propagates_failures(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.distributed.is_gloo_available():
        pytest.skip("Gloo is unavailable")
    # Pytest's importlib mode deliberately leaves the repository root off
    # sys.path. Multiprocessing spawn must still be able to import this worker
    # module in each fresh interpreter.
    project_root = str(Path(__file__).resolve().parents[2])
    added_to_path = project_root not in sys.path
    if added_to_path:
        sys.path.insert(0, project_root)
    try:
        torch.multiprocessing.spawn(
            _data_readiness_worker,
            args=((tmp_path / "rendezvous").as_uri(),),
            nprocs=2,
            join=True,
        )
    finally:
        if added_to_path:
            sys.path.remove(project_root)
