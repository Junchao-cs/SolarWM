"""Torch distributed primitives for the embedded LTX-2.5 runtime.

The topology is ``raw_rank = dp_rank * sp_size + sp_rank``.  SP peers own
equal contiguous token ranges and exchange sequence for attention heads.  FSDP
shards down an SP column; HYBRID_SHARD additionally replicates that shard
position across nodes.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from solar.errors import BackendContractError

from .distributed import build_hsdp_topology, contiguous_token_bounds


@dataclass
class DistributedState:
    rank: int
    world_size: int
    local_rank: int
    local_world_size: int
    sp_size: int
    sp_rank: int
    dp_rank: int
    dp_world_size: int
    sp_group: dist.ProcessGroup | None
    dp_group: dist.ProcessGroup | None
    sp_group_ranks: tuple[int, ...]
    dp_group_ranks: tuple[int, ...]
    fsdp_process_group: Any
    fsdp_shard_group: dist.ProcessGroup | None
    sequence_length: int = 0


_STATE: DistributedState | None = None


def initialize(config: dict[str, Any] | Any) -> DistributedState:
    """Initialize NCCL and every deterministic SP/DP/FSDP group exactly once."""

    global _STATE
    distributed = config["distributed"]
    train = config.get("train", {})
    fsdp_config = train.get("fsdp", {})
    sp_size = int(distributed["sequence_parallel_size"])
    expected_world = int(distributed["world_size"])
    expected_local_world = int(distributed["local_world_size"])
    if _STATE is not None:
        if (
            _STATE.world_size,
            _STATE.local_world_size,
            _STATE.sp_size,
        ) != (expected_world, expected_local_world, sp_size):
            raise BackendContractError(
                "initialized LTX topology differs from the requested configuration"
            )
        return _STATE
    if not dist.is_initialized():
        if expected_world != 1:
            dist.init_process_group(backend="nccl")
        else:
            rank = int(os.environ.get("RANK", "0"))
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            _STATE = DistributedState(
                rank=rank,
                world_size=1,
                local_rank=local_rank,
                local_world_size=1,
                sp_size=1,
                sp_rank=0,
                dp_rank=0,
                dp_world_size=1,
                sp_group=None,
                dp_group=None,
                sp_group_ranks=(0,),
                dp_group_ranks=(0,),
                fsdp_process_group=None,
                fsdp_shard_group=None,
            )
            return _STATE
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    if (world_size, local_world_size) != (expected_world, expected_local_world):
        raise BackendContractError("torchrun topology differs from the validated LTX configuration")
    if world_size % sp_size:
        raise BackendContractError("raw world size must be divisible by LTX SP size")
    torch.cuda.set_device(local_rank)
    dp_world_size = world_size // sp_size
    sp_rank = rank % sp_size
    dp_rank = rank // sp_size
    sp_group = None
    dp_group = None
    sp_ranks: tuple[int, ...] = ()
    dp_ranks: tuple[int, ...] = ()
    for dp_index in range(dp_world_size):
        ranks = tuple(range(dp_index * sp_size, (dp_index + 1) * sp_size))
        group = dist.new_group(list(ranks), backend="nccl")
        if rank in ranks:
            sp_group, sp_ranks = group, ranks
    for sp_index in range(sp_size):
        ranks = tuple(range(sp_index, world_size, sp_size))
        group = dist.new_group(list(ranks), backend="nccl")
        if rank in ranks:
            dp_group, dp_ranks = group, ranks
    if sp_size > 1 and (sp_group is None or dp_group is None):
        raise BackendContractError("failed to construct the LTX SP/DP process groups")

    strategy = str(fsdp_config.get("sharding_strategy", "FULL_SHARD"))
    fsdp_process_group: Any = dp_group if sp_size > 1 else None
    fsdp_shard_group = dp_group if sp_size > 1 else None
    if strategy == "HYBRID_SHARD":
        topology = build_hsdp_topology(
            world_size,
            local_world_size=local_world_size,
            sp_size=sp_size,
        )
        selected_shard = None
        selected_replica = None
        for ranks in topology.shard_groups:
            group = dist.new_group(list(ranks), backend="nccl")
            if rank in ranks:
                selected_shard = group
        for ranks in topology.replica_groups:
            group = dist.new_group(list(ranks), backend="nccl")
            if rank in ranks:
                selected_replica = group
        if selected_shard is None or selected_replica is None:
            raise BackendContractError("failed to construct SP-aware HSDP groups")
        fsdp_process_group = (selected_shard, selected_replica)
        fsdp_shard_group = selected_shard

    _STATE = DistributedState(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        sp_size=sp_size,
        sp_rank=sp_rank,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        sp_group=sp_group,
        dp_group=dp_group,
        sp_group_ranks=sp_ranks or (rank,),
        dp_group_ranks=dp_ranks or (rank,),
        fsdp_process_group=fsdp_process_group,
        fsdp_shard_group=fsdp_shard_group,
    )
    return _STATE


def state() -> DistributedState:
    if _STATE is None:
        raise BackendContractError("LTX distributed runtime is not initialized")
    return _STATE


def register_sequence_length(local_length: int) -> tuple[int, ...]:
    runtime = state()
    runtime.sequence_length = int(local_length)
    if runtime.sp_size == 1:
        return (runtime.sequence_length,)
    value = torch.tensor([runtime.sequence_length], device="cuda", dtype=torch.int64)
    gathered = [torch.empty_like(value) for _ in range(runtime.sp_size)]
    dist.all_gather(gathered, value, group=runtime.sp_group)
    lengths = tuple(int(item.item()) for item in gathered)
    if lengths != (runtime.sequence_length,) * runtime.sp_size:
        raise BackendContractError(
            f"LTX Stage0.5 requires equal SP token shards, observed {lengths}"
        )
    return lengths


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input_: torch.Tensor,
        world_size: int,
        dimension: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.world_size = world_size
        ctx.dimension = dimension if dimension >= 0 else input_.dim() + dimension
        ctx.input_shape = input_.shape
        output_size = (input_.shape[0] * world_size, *input_.shape[1:])
        output = torch.empty(output_size, device=input_.device, dtype=input_.dtype)
        dist.all_gather_into_tensor(output, input_.contiguous(), group=group)
        output = output.reshape((world_size, *input_.shape)).movedim(0, ctx.dimension)
        shape = (
            *input_.shape[: ctx.dimension],
            world_size * input_.shape[ctx.dimension],
            *input_.shape[ctx.dimension + 1 :],
        )
        return output.reshape(shape)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[Any, ...]:
        size = gradient.shape[ctx.dimension] // ctx.world_size
        chunks = (
            gradient.reshape(
                *gradient.shape[: ctx.dimension],
                ctx.world_size,
                size,
                *gradient.shape[ctx.dimension + 1 :],
            )
            .movedim(ctx.dimension, 0)
            .contiguous()
        )
        result = torch.empty(ctx.input_shape, device=gradient.device, dtype=gradient.dtype)
        dist.reduce_scatter_tensor(result, chunks, group=ctx.group)
        return None, result, None, None


class _AllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input_: torch.Tensor,
        world_size: int,
        scatter_dimension: int,
        gather_dimension: int,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.world_size = world_size
        ctx.scatter_dimension = scatter_dimension
        ctx.gather_dimension = gather_dimension
        if world_size == 1:
            return input_
        if input_.ndim != 4:
            raise BackendContractError("LTX SP all-to-all requires a rank-four tensor")
        if (scatter_dimension, gather_dimension) == (2, 1):
            batch, local_sequence, heads, head_dim = input_.shape
            if heads % world_size:
                raise BackendContractError("LTX attention heads must divide SP size")
            local_heads = heads // world_size
            exchanged = input_.transpose(0, 2).contiguous()
            output = torch.empty_like(exchanged)
            dist.all_to_all_single(output, exchanged, group=group)
            output = torch.cat(output.split(local_heads), dim=1)
            return output.transpose(0, 2).contiguous()
        if (scatter_dimension, gather_dimension) == (1, 2):
            batch, sequence, local_heads, head_dim = input_.shape
            if sequence % world_size:
                raise BackendContractError("LTX full token sequence must divide SP size")
            local_sequence = sequence // world_size
            exchanged = (
                input_.transpose(0, 2)
                .contiguous()
                .reshape(local_heads, world_size, local_sequence, batch, head_dim)
                .transpose(0, 1)
                .reshape(local_heads * world_size, local_sequence, batch, head_dim)
                .contiguous()
            )
            output = torch.empty_like(exchanged)
            dist.all_to_all_single(output, exchanged, group=group)
            return output.transpose(0, 2).contiguous()
        raise BackendContractError("unsupported LTX SP all-to-all dimensions")

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[Any, ...]:
        if ctx.world_size == 1:
            return None, gradient, None, None, None
        result = _AllToAll4D.apply(
            ctx.group,
            gradient,
            ctx.world_size,
            ctx.gather_dimension,
            ctx.scatter_dimension,
        )
        return None, result, None, None, None


def all_gather_sequence(input_: torch.Tensor, dimension: int = 1) -> torch.Tensor:
    runtime = state()
    if runtime.sp_size == 1:
        return input_
    return _AllGather.apply(
        runtime.sp_group,
        input_,
        runtime.sp_size,
        dimension,
    )


def all_to_all_4d(
    input_: torch.Tensor,
    *,
    scatter_dimension: int,
    gather_dimension: int,
) -> torch.Tensor:
    runtime = state()
    if runtime.sp_size == 1:
        return input_
    return _AllToAll4D.apply(
        runtime.sp_group,
        input_,
        runtime.sp_size,
        scatter_dimension,
        gather_dimension,
    )


def broadcast_sp_tensor(input_: torch.Tensor) -> torch.Tensor:
    runtime = state()
    if runtime.sp_size == 1:
        return input_
    source = runtime.sp_group_ranks[0]
    staging = input_ if input_.is_contiguous() else input_.contiguous()
    dist.broadcast(staging, src=source, group=runtime.sp_group)
    if staging is not input_:
        input_.copy_(staging)
    return input_


def assert_sp_object_identity(value: Any) -> None:
    runtime = state()
    if runtime.sp_size == 1:
        return
    peers: list[Any] = [None] * runtime.sp_size
    dist.all_gather_object(peers, value, group=runtime.sp_group)
    if any(peer != peers[0] for peer in peers[1:]):
        raise BackendContractError(f"SP peers have different batch identities: {peers!r}")


@torch.no_grad()
def _sync_replicated_gradient_group(
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    group: dist.ProcessGroup | None,
    average: bool,
) -> None:
    """Reduce FSDP-ignored LoRA tensors in bounded FP32 buckets."""

    if not parameters or not dist.is_initialized():
        return
    world_size = dist.get_world_size(group=group)
    if world_size == 1:
        return
    flag_device = next(
        (parameter.grad.device for parameter in parameters if parameter.grad is not None),
        parameters[0].device,
    )
    flags = torch.tensor(
        [int(parameter.grad is not None) for parameter in parameters],
        device=flag_device,
        dtype=torch.int32,
    )
    dist.all_reduce(flags, group=group)
    inconsistent = (flags != 0) & (flags != world_size)
    if bool(inconsistent.any().item()):
        indices = inconsistent.nonzero(as_tuple=False).flatten()[:16].cpu().tolist()
        raise BackendContractError(
            "LoRA gradient presence differs across replicated peers; "
            f"first parameter indices={indices}"
        )

    limit_mib = max(1, int(os.environ.get("SOLARWM_REPLICATED_GRAD_BUCKET_MB", "16")))
    limit_bytes = limit_mib * 1024 * 1024
    buckets: list[list[torch.Tensor]] = []
    current: list[torch.Tensor] = []
    current_bytes = 0
    current_key: tuple[torch.device, torch.dtype] | None = None
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        key = (gradient.device, gradient.dtype)
        size_bytes = gradient.numel() * gradient.element_size()
        if current and (key != current_key or current_bytes + size_bytes > limit_bytes):
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(gradient)
        current_bytes += size_bytes
        current_key = key
    if current:
        buckets.append(current)

    scale = 1.0 / world_size if average else 1.0
    for bucket in buckets:
        flattened = torch.cat([gradient.detach().reshape(-1).float() for gradient in bucket])
        dist.all_reduce(flattened, group=group)
        if average:
            flattened.mul_(scale)
        offset = 0
        for gradient in bucket:
            stop = offset + gradient.numel()
            gradient.copy_(flattened[offset:stop].view(gradient.shape))
            offset = stop


@torch.no_grad()
def synchronize_replicated_gradients(parameters: tuple[torch.nn.Parameter, ...]) -> None:
    """Reconstruct SP gradients, then average LoRA gradients over logical DP in FP32."""

    runtime = state()
    _sync_replicated_gradient_group(
        parameters,
        group=runtime.sp_group,
        average=False,
    )
    _sync_replicated_gradient_group(
        parameters,
        group=runtime.dp_group,
        average=True,
    )


@torch.no_grad()
def clip_replicated_gradient_norm(
    parameters: tuple[torch.nn.Parameter, ...],
    max_norm: float,
) -> torch.Tensor:
    """Clip one replicated LoRA norm without overflowing FP32 squares."""

    limit = float(max_norm)
    if not math.isfinite(limit) or limit <= 0:
        raise BackendContractError("LTX gradient clip must be finite and positive")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.zeros((), dtype=torch.float32)
    device = gradients[0].device
    squared = torch.zeros((), device=device, dtype=torch.float64)
    for gradient in gradients:
        if gradient.device != device or gradient.is_sparse:
            raise BackendContractError("LTX replicated gradients must be dense on one device")
        scale = gradient.detach().abs().amax().double()
        safe_scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        norm = torch.linalg.vector_norm(
            gradient.detach() / safe_scale.to(dtype=gradient.dtype),
            ord=2,
            dtype=(
                torch.float64
                if gradient.dtype in {torch.float64, torch.complex128}
                else torch.float32
            ),
        )
        squared.add_((scale * norm.double()).square())
    total = squared.sqrt()
    if bool(torch.isfinite(total).item()):
        coefficient = min(1.0, limit / (float(total.item()) + 1e-6))
        if coefficient < 1.0:
            for gradient in gradients:
                gradient.mul_(coefficient)
    return total.float()


def token_bounds(total_tokens: int) -> tuple[int, int]:
    runtime = state()
    return contiguous_token_bounds(
        total_tokens,
        sp_size=runtime.sp_size,
        sp_rank=runtime.sp_rank,
    )


__all__ = [
    "DistributedState",
    "all_gather_sequence",
    "all_to_all_4d",
    "assert_sp_object_identity",
    "broadcast_sp_tensor",
    "clip_replicated_gradient_norm",
    "initialize",
    "register_sequence_length",
    "state",
    "synchronize_replicated_gradients",
    "token_bounds",
]
