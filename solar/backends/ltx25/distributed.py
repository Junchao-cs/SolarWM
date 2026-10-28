"""Sequence-parallel and SP-aware HSDP topology contracts."""

from __future__ import annotations

from dataclasses import dataclass

from solar.errors import BackendContractError

from .geometry import STABLE_GEOMETRY


def contiguous_token_bounds(total_tokens: int, *, sp_size: int, sp_rank: int) -> tuple[int, int]:
    """Return an equal contiguous token shard without implicit padding."""

    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (total_tokens, sp_size, sp_rank)
    ):
        raise BackendContractError("token topology values must be integers")
    if total_tokens <= 0 or sp_size <= 0 or not 0 <= sp_rank < sp_size:
        raise BackendContractError("invalid token topology")
    if total_tokens % sp_size:
        raise BackendContractError("video token count must be divisible by SP size")
    local = total_tokens // sp_size
    return sp_rank * local, (sp_rank + 1) * local


@dataclass(frozen=True)
class HSDPTopology:
    """All rank groups for LTX's SP-column-preserving hybrid sharding."""

    raw_world_size: int
    local_world_size: int
    sp_size: int
    shard_groups: tuple[tuple[int, ...], ...]
    replica_groups: tuple[tuple[int, ...], ...]

    @property
    def node_count(self) -> int:
        return self.raw_world_size // self.local_world_size

    @property
    def local_dp_size(self) -> int:
        return self.local_world_size // self.sp_size


def build_hsdp_topology(
    raw_world_size: int,
    *,
    local_world_size: int = 8,
    sp_size: int = 2,
) -> HSDPTopology:
    """Build ``shard(node,sp)`` and ``replica(local_dp,sp)`` groups."""

    values = (raw_world_size, local_world_size, sp_size)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in values):
        raise BackendContractError("HSDP topology values must be positive integers")
    if raw_world_size % local_world_size or local_world_size % sp_size:
        raise BackendContractError("raw/local world sizes must be divisible by the SP size")
    nodes = raw_world_size // local_world_size
    local_dp = local_world_size // sp_size
    shard_groups = tuple(
        tuple(node * local_world_size + dp * sp_size + sp for dp in range(local_dp))
        for node in range(nodes)
        for sp in range(sp_size)
    )
    replica_groups = tuple(
        tuple(node * local_world_size + dp * sp_size + sp for node in range(nodes))
        for dp in range(local_dp)
        for sp in range(sp_size)
    )
    flattened_shards = sorted(rank for group in shard_groups for rank in group)
    if flattened_shards != list(range(raw_world_size)):
        raise BackendContractError("HSDP shard groups do not partition raw ranks")
    return HSDPTopology(
        raw_world_size=raw_world_size,
        local_world_size=local_world_size,
        sp_size=sp_size,
        shard_groups=shard_groups,
        replica_groups=replica_groups,
    )


@dataclass(frozen=True)
class DistributedContract:
    world_size: int
    local_world_size: int
    sp_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    sharding_strategy: str
    activation_checkpointed_blocks: int

    def __post_init__(self) -> None:
        if self.sp_size != 2:
            raise BackendContractError("LTX Stage0.5 requires SP2")
        if self.world_size % self.sp_size or self.local_world_size != 8:
            raise BackendContractError(
                "LTX topology requires eight-GPU nodes and SP-divisible world"
            )
        computed = (
            self.world_size
            // self.sp_size
            * self.micro_batch_size
            * self.gradient_accumulation_steps
        )
        if computed != self.global_batch_size:
            raise BackendContractError(
                "global batch mismatch: world/SP * micro * GA "
                f"is {computed}, configured {self.global_batch_size}"
            )
        if self.sharding_strategy not in {"FULL_SHARD", "HYBRID_SHARD"}:
            raise BackendContractError("LTX sharding must be FULL_SHARD or HYBRID_SHARD")
        if self.activation_checkpointed_blocks != 48:
            raise BackendContractError("LTX Stage0.5 activation-checkpoints all 48 blocks")
        if self.sharding_strategy == "HYBRID_SHARD":
            topology = build_hsdp_topology(
                self.world_size,
                local_world_size=self.local_world_size,
                sp_size=self.sp_size,
            )
            if topology.node_count < 2:
                raise BackendContractError("HYBRID_SHARD requires at least two nodes")

    @property
    def local_token_bounds(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            contiguous_token_bounds(
                STABLE_GEOMETRY.video_tokens,
                sp_size=self.sp_size,
                sp_rank=rank,
            )
            for rank in range(self.sp_size)
        )


__all__ = [
    "DistributedContract",
    "HSDPTopology",
    "build_hsdp_topology",
    "contiguous_token_bounds",
]
