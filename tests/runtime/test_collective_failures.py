from __future__ import annotations

from typing import Any

import pytest

from solar.errors import BackendContractError
from solar.runtime.distributed import (
    collective_call,
    collective_rank_zero_call,
    propagate_collective_error,
)


class FakeDist:
    def __init__(
        self,
        *,
        rank: int = 0,
        world_size: int = 2,
        gathered: list[tuple[int, str]] | None = None,
        broadcast_value: Any = None,
        initialized: bool = True,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.gathered = gathered
        self.broadcast_value = broadcast_value
        self.initialized = initialized
        self.exchanged: list[tuple[int, str]] = []

    def is_initialized(self) -> bool:
        return self.initialized

    def get_rank(self) -> int:
        return self.rank

    def get_world_size(self) -> int:
        return self.world_size

    def all_gather_object(self, output: list[Any], local: tuple[int, str]) -> None:
        assert not isinstance(local[1], BaseException)
        self.exchanged.append(local)
        records = self.gathered
        if records is None:
            records = [local, *((peer, "") for peer in range(1, self.world_size))]
        output[:] = records

    def broadcast_object_list(self, values: list[Any], *, src: int) -> None:
        assert src == 0
        if self.rank != src:
            values[0] = self.broadcast_value


def test_single_process_collective_call_is_a_noop() -> None:
    assert collective_call(lambda: "value", dist=None, label="single") == "value"


def test_explicit_multi_rank_never_degrades_without_process_group() -> None:
    with pytest.raises(BackendContractError, match="requires initialized"):
        propagate_collective_error(
            None,
            dist=None,
            rank=0,
            world_size=2,
            label="missing group",
        )


def test_peer_failures_are_deterministic_and_rank_sorted() -> None:
    fake = FakeDist(gathered=[(1, "second"), (0, "first")])
    with pytest.raises(BackendContractError) as caught:
        propagate_collective_error("first", dist=fake, label="phase")
    assert str(caught.value).endswith("rank 0: first | rank 1: second")


def test_callback_exception_crosses_as_text_not_exception_object() -> None:
    fake = FakeDist(gathered=[(0, "ValueError: local boom"), (1, "peer boom")])

    def fail() -> None:
        raise ValueError("local boom")

    with pytest.raises(BackendContractError, match="rank 0: ValueError: local boom"):
        collective_call(fail, dist=fake, label="reader")
    assert fake.exchanged == [(0, "ValueError: local boom")]


def test_nonzero_rank_does_not_run_rank_zero_callback() -> None:
    called = False
    fake = FakeDist(
        rank=1,
        gathered=[(0, ""), (1, "")],
        broadcast_value="rank-zero-value",
    )

    def forbidden() -> str:
        nonlocal called
        called = True
        return "wrong"

    assert collective_rank_zero_call(forbidden, dist=fake, label="rank zero") == "rank-zero-value"
    assert called is False
