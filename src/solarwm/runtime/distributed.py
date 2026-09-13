"""Small fail-closed checks around model-family distributed process groups."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from typing import Any, TypeVar

from solarwm.config.loader import canonical_json
from solarwm.errors import BackendContractError

T = TypeVar("T")


class DataReadiness:
    """Finish remote batch reads on every rank before entering GPU collectives.

    All raw ranks construct and call this at the same reader boundary. The
    separate CPU group keeps storage latency out of the NCCL watchdog window;
    it does not select samples, advance RNGs, or move batch tensors.
    """

    def __init__(self, *, enabled: bool) -> None:
        self.group: Any = None
        if not enabled:
            return
        import torch
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() == 1:
            return
        self.torch = torch
        self.dist = dist
        self.group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))

    def read(self, call: Callable[[], T]) -> T:
        if self.group is None:
            return call()
        started = time.monotonic()
        result: T | None = None
        local_error: Exception | None = None
        try:
            result = call()
        except Exception as exc:
            local_error = exc
        rank = self.dist.get_rank()
        world = self.dist.get_world_size()
        status = self.torch.tensor(
            [rank if local_error is not None else world], dtype=self.torch.int64, device="cpu"
        )
        self.dist.all_reduce(status, op=self.dist.ReduceOp.MIN, group=self.group)
        failed_rank = int(status.item())
        if failed_rank != world:
            messages = [
                f"{type(local_error).__name__}: {local_error}" if rank == failed_rank else ""
            ]
            self.dist.broadcast_object_list(messages, src=failed_rank, group=self.group)
            raise BackendContractError(
                f"training data read failed on rank {failed_rank}: {messages[0]}"
            ) from local_error
        elapsed = time.monotonic() - started
        if elapsed >= 30:
            print(
                f"[data-readiness][rank{rank}] all batches ready after {elapsed:.1f}s", flush=True
            )
        return result  # type: ignore[return-value]


def _collective_coordinates(
    dist: Any | None,
    *,
    rank: int | None,
    world_size: int | None,
) -> tuple[int, int, bool]:
    """Resolve one raw-world collective without importing Torch.

    Callers pass ``torch.distributed`` in heavy runtimes and a small fake in
    CPU tests.  An explicitly multi-rank call may never silently degrade to a
    local check when the process group is unavailable.
    """

    initialized = bool(dist is not None and dist.is_initialized())
    resolved_rank = (
        int(dist.get_rank()) if rank is None and initialized else int(0 if rank is None else rank)
    )
    resolved_world = (
        int(dist.get_world_size())
        if world_size is None and initialized
        else int(1 if world_size is None else world_size)
    )
    if resolved_world < 1 or not 0 <= resolved_rank < resolved_world:
        raise BackendContractError(
            "collective failure propagation received an invalid rank/world_size"
        )
    if resolved_world > 1 and not initialized:
        raise BackendContractError(
            "multi-rank failure propagation requires initialized torch.distributed"
        )
    if initialized:
        actual_rank = int(dist.get_rank())
        actual_world = int(dist.get_world_size())
        if (resolved_rank, resolved_world) != (actual_rank, actual_world):
            raise BackendContractError(
                "collective failure propagation rank/world_size differs from the process group"
            )
    return resolved_rank, resolved_world, initialized


def propagate_collective_error(
    local_error: str | None,
    *,
    dist: Any | None,
    label: str,
    rank: int | None = None,
    world_size: int | None = None,
    error_type: type[Exception] = BackendContractError,
) -> None:
    """Raise the same rank-ordered failure on every raw-world peer.

    Only strings cross the process-group boundary.  Exception instances are
    deliberately never pickled or broadcast.  Every peer must call this at
    the same phase boundary.
    """

    resolved_rank, resolved_world, initialized = _collective_coordinates(
        dist,
        rank=rank,
        world_size=world_size,
    )
    message = str(local_error or "")
    local = (resolved_rank, message)
    gathered: list[Any] = [local]
    if initialized and resolved_world > 1:
        gathered = [None] * resolved_world
        dist.all_gather_object(gathered, local)

    normalized: list[tuple[int, str]] = []
    for item in gathered:
        if not isinstance(item, (tuple, list)) or len(item) != 2 or isinstance(item[0], bool):
            raise error_type(f"{label} failure exchange returned a malformed rank record")
        try:
            peer_rank = int(item[0])
        except (TypeError, ValueError) as exc:
            raise error_type(f"{label} failure exchange returned a malformed rank value") from exc
        peer_error = str(item[1] or "")
        if not 0 <= peer_rank < resolved_world:
            raise error_type(f"{label} failure exchange returned an out-of-range rank")
        normalized.append((peer_rank, peer_error))
    if len(normalized) != resolved_world or {item[0] for item in normalized} != set(
        range(resolved_world)
    ):
        raise error_type(f"{label} failure exchange did not cover every raw rank")
    failures = [(peer_rank, error) for peer_rank, error in sorted(normalized) if error]
    if failures:
        detail = " | ".join(f"rank {peer_rank}: {error}" for peer_rank, error in failures)
        raise error_type(f"{label} failed collectively: {detail}")


def collective_call(
    call: Callable[[], T],
    *,
    dist: Any | None,
    label: str,
    rank: int | None = None,
    world_size: int | None = None,
    error_type: type[Exception] = BackendContractError,
) -> T:
    """Run rank-local work, then exchange errors before any later collective."""

    result: T | None = None
    local_exception: Exception | None = None
    local_error = ""
    try:
        result = call()
    except Exception as exc:
        local_exception = exc
        local_error = f"{type(exc).__name__}: {exc}"
    try:
        propagate_collective_error(
            local_error,
            dist=dist,
            label=label,
            rank=rank,
            world_size=world_size,
            error_type=error_type,
        )
    except Exception as exc:
        if local_exception is not None:
            raise exc from local_exception
        raise
    return result  # type: ignore[return-value]


def collective_rank_zero_call(
    call: Callable[[], T],
    *,
    dist: Any | None,
    label: str,
    rank: int | None = None,
    world_size: int | None = None,
    error_type: type[Exception] = BackendContractError,
) -> T:
    """Run work on raw rank zero and broadcast only its successful value."""

    resolved_rank, resolved_world, initialized = _collective_coordinates(
        dist,
        rank=rank,
        world_size=world_size,
    )
    result = collective_call(
        call if resolved_rank == 0 else lambda: None,  # type: ignore[arg-type]
        dist=dist,
        label=label,
        rank=resolved_rank,
        world_size=resolved_world,
        error_type=error_type,
    )
    values: list[Any] = [result if resolved_rank == 0 else None]
    if initialized and resolved_world > 1:
        dist.broadcast_object_list(values, src=0)
    return values[0]  # type: ignore[return-value]


def identity_fingerprint(value: Mapping[str, Any]) -> str:
    return hashlib.blake2s(canonical_json(value)).hexdigest()


def assert_peer_fingerprints(values: Sequence[str]) -> str:
    if not values or any(not value for value in values):
        raise BackendContractError("SP peer fingerprints are missing")
    if len(set(values)) != 1:
        raise BackendContractError(f"SP peers loaded different sample/RNG identities: {values}")
    return values[0]


def gather_and_assert_sp_identity(
    value: Mapping[str, Any], *, sp_size: int, group: Any = None
) -> str:
    """Collect identity hashes in the already-created SP group and compare."""

    local = identity_fingerprint(value)
    if sp_size == 1:
        return local
    try:
        import torch.distributed as dist
    except ImportError as exc:
        raise BackendContractError("torch is required for sequence parallelism") from exc
    if not dist.is_available() or not dist.is_initialized():
        raise BackendContractError("SP identity check requires initialized torch.distributed")
    peers: list[str | None] = [None] * sp_size
    dist.all_gather_object(peers, local, group=group)
    return assert_peer_fingerprints([str(value or "") for value in peers])
