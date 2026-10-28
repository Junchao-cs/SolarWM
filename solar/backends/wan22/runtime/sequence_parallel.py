# SPDX-License-Identifier: Apache-2.0
# ruff: noqa
"""Sequence-parallel runtime for SolarWM.

The runtime uses a DP x SP topology:
  global_rank = dp_rank * sp_size + sp_rank

SP ranks in one group process different token chunks of the same logical
sample. Most backbones shard FSDP over the logical-DP group and explicitly sum
the complementary SP gradients. H3 full-parameter SP2 instead shards FSDP over
the raw world group: its FSDP reduce-scatter already includes SP, which both
fits a 33B optimizer state on one 8-GPU node and changes the loss scaling rule.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional, Sequence

import torch
import torch.distributed as dist


@dataclass
class _SPState:
    initialized: bool = False
    sp_size: int = 1
    sp_rank: int = 0
    dp_rank: int = 0
    dp_world_size: int = 1
    sp_group: Optional[dist.ProcessGroup] = None
    dp_group: Optional[dist.ProcessGroup] = None
    sp_group_ranks: tuple[int, ...] = (0,)
    dp_group_ranks: tuple[int, ...] = (0,)
    sequence_lengths: tuple[int, ...] = ()
    fsdp_over_raw_world: bool = False
    fsdp_hybrid_shard: bool = False
    fsdp_shard_group: Optional[dist.ProcessGroup] = None
    fsdp_replica_group: Optional[dist.ProcessGroup] = None
    fsdp_shard_group_ranks: tuple[int, ...] = ()
    fsdp_replica_group_ranks: tuple[int, ...] = ()


_STATE = _SPState()


def build_sp_hybrid_fsdp_group_ranks(
    *,
    world_size: int,
    local_world_size: int,
    sp_size: int,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Build SP-column-preserving HSDP shard and replica groups.

    ``torchrun`` assigns each node a contiguous block of raw ranks while the
    SolarWM SP layout is ``global_rank = dp_rank * sp_size + sp_rank``.  Each
    HSDP shard group therefore contains one SP column inside one node.  Ranks
    at the same local-DP position form the corresponding cross-node replica
    group.
    """

    world_size = int(world_size)
    local_world_size = int(local_world_size)
    sp_size = int(sp_size)
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if local_world_size < 1:
        raise ValueError(f"LOCAL_WORLD_SIZE must be >= 1, got {local_world_size}")
    if sp_size < 1:
        raise ValueError(f"sp_size must be >= 1, got {sp_size}")
    if world_size % local_world_size:
        raise ValueError(
            f"world_size={world_size} must be divisible by LOCAL_WORLD_SIZE={local_world_size}"
        )
    if local_world_size % sp_size:
        raise ValueError(
            f"LOCAL_WORLD_SIZE={local_world_size} must be divisible by sp_size={sp_size}"
        )

    num_nodes = world_size // local_world_size
    local_dp_world_size = local_world_size // sp_size
    if num_nodes < 2:
        raise ValueError("SP-aware HSDP requires at least two nodes")
    if local_dp_world_size < 2:
        raise ValueError("SP-aware HSDP requires at least two logical-DP ranks per node")

    shard_groups = tuple(
        tuple(
            node_idx * local_world_size + local_dp_idx * sp_size + sp_idx
            for local_dp_idx in range(local_dp_world_size)
        )
        for node_idx in range(num_nodes)
        for sp_idx in range(sp_size)
    )
    replica_groups = tuple(
        tuple(
            node_idx * local_world_size + local_dp_idx * sp_size + sp_idx
            for node_idx in range(num_nodes)
        )
        for local_dp_idx in range(local_dp_world_size)
        for sp_idx in range(sp_size)
    )
    return shard_groups, replica_groups


def init_sequence_parallel(
    sp_size: int = 1,
    *,
    fsdp_over_raw_world: bool = False,
    fsdp_hybrid_shard: bool = False,
) -> _SPState:
    """Create SP and DP process groups after torch.distributed init."""
    global _STATE

    sp_size = int(sp_size)
    if sp_size < 1:
        raise ValueError(f"sp_size must be >= 1, got {sp_size}")
    if fsdp_over_raw_world and fsdp_hybrid_shard:
        raise ValueError("fsdp_over_raw_world and fsdp_hybrid_shard are mutually exclusive")

    if not dist.is_initialized():
        if fsdp_over_raw_world or fsdp_hybrid_shard:
            raise RuntimeError(
                "distributed FSDP topology requires torch.distributed to be initialized"
            )
        if sp_size != 1:
            raise RuntimeError("SP_SIZE > 1 requires torch.distributed to be initialized")
        _STATE = _SPState(initialized=True)
        return _STATE

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size % sp_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by sp_size={sp_size}")
    if fsdp_hybrid_shard and sp_size == 1:
        raise ValueError("SP-aware HSDP requires sp_size > 1")

    dp_world_size = world_size // sp_size
    sp_rank = rank % sp_size
    dp_rank = rank // sp_size

    if sp_size == 1:
        _STATE = _SPState(
            initialized=True,
            sp_size=1,
            sp_rank=0,
            dp_rank=rank,
            dp_world_size=world_size,
            sp_group=None,
            dp_group=None,
            sp_group_ranks=(rank,),
            dp_group_ranks=tuple(range(world_size)),
            fsdp_over_raw_world=bool(fsdp_over_raw_world),
        )
        return _STATE

    sp_group = None
    dp_group = None
    sp_group_ranks: Sequence[int] = ()
    dp_group_ranks: Sequence[int] = ()
    fsdp_shard_group = None
    fsdp_replica_group = None
    fsdp_shard_group_ranks: Sequence[int] = ()
    fsdp_replica_group_ranks: Sequence[int] = ()

    # Every rank must create groups in identical order.
    for dp_idx in range(dp_world_size):
        ranks = list(range(dp_idx * sp_size, (dp_idx + 1) * sp_size))
        group = dist.new_group(ranks=ranks, backend="nccl")
        if rank in ranks:
            sp_group = group
            sp_group_ranks = tuple(ranks)

    for sp_idx in range(sp_size):
        ranks = list(range(sp_idx, world_size, sp_size))
        group = dist.new_group(ranks=ranks, backend="nccl")
        if rank in ranks:
            dp_group = group
            dp_group_ranks = tuple(ranks)

    if fsdp_hybrid_shard:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
        shard_groups, replica_groups = build_sp_hybrid_fsdp_group_ranks(
            world_size=world_size,
            local_world_size=local_world_size,
            sp_size=sp_size,
        )
        for ranks in shard_groups:
            group = dist.new_group(ranks=list(ranks), backend="nccl")
            if rank in ranks:
                fsdp_shard_group = group
                fsdp_shard_group_ranks = ranks
        for ranks in replica_groups:
            group = dist.new_group(ranks=list(ranks), backend="nccl")
            if rank in ranks:
                fsdp_replica_group = group
                fsdp_replica_group_ranks = ranks
        assert fsdp_shard_group is not None
        assert fsdp_replica_group is not None

    assert sp_group is not None
    assert dp_group is not None
    _STATE = _SPState(
        initialized=True,
        sp_size=sp_size,
        sp_rank=sp_rank,
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        sp_group=sp_group,
        dp_group=dp_group,
        sp_group_ranks=tuple(sp_group_ranks),
        dp_group_ranks=tuple(dp_group_ranks),
        fsdp_over_raw_world=bool(fsdp_over_raw_world),
        fsdp_hybrid_shard=bool(fsdp_hybrid_shard),
        fsdp_shard_group=fsdp_shard_group,
        fsdp_replica_group=fsdp_replica_group,
        fsdp_shard_group_ranks=tuple(fsdp_shard_group_ranks),
        fsdp_replica_group_ranks=tuple(fsdp_replica_group_ranks),
    )
    return _STATE


def get_sequence_parallel_state() -> _SPState:
    if not _STATE.initialized:
        # Single-process helpers should still be safe before distributed init.
        return _SPState(initialized=True)
    return _STATE


def is_sequence_parallel_enabled() -> bool:
    return get_sequence_parallel_state().sp_size > 1


def get_sp_size() -> int:
    return get_sequence_parallel_state().sp_size


def get_sp_rank() -> int:
    return get_sequence_parallel_state().sp_rank


def get_dp_rank() -> int:
    return get_sequence_parallel_state().dp_rank


def get_dp_world_size() -> int:
    return get_sequence_parallel_state().dp_world_size


def get_sp_group() -> Optional[dist.ProcessGroup]:
    return get_sequence_parallel_state().sp_group


def get_dp_group() -> Optional[dist.ProcessGroup]:
    return get_sequence_parallel_state().dp_group


def get_sp_group_ranks() -> tuple[int, ...]:
    return get_sequence_parallel_state().sp_group_ranks


def get_fsdp_process_group():
    """Return the FSDP group, including a 2-D tuple for SP-aware HSDP."""
    state = get_sequence_parallel_state()
    if state.fsdp_over_raw_world:
        return None
    if state.fsdp_hybrid_shard:
        assert state.fsdp_shard_group is not None
        assert state.fsdp_replica_group is not None
        return state.fsdp_shard_group, state.fsdp_replica_group
    return state.dp_group if state.sp_size > 1 else None


def get_fsdp_shard_process_group() -> Optional[dist.ProcessGroup]:
    """Return the one-dimensional group used to gather FSDP optimizer state."""
    state = get_sequence_parallel_state()
    if state.fsdp_over_raw_world:
        return None
    if state.fsdp_hybrid_shard:
        return state.fsdp_shard_group
    return state.dp_group if state.sp_size > 1 else None


def fsdp_shards_across_sequence_parallel() -> bool:
    """Whether FSDP reduction already includes the SP dimension."""
    return bool(get_sequence_parallel_state().fsdp_over_raw_world)


def register_sequence_parallel_sequence_length(local_length: int) -> tuple[int, ...]:
    """Register the token length owned by every rank in the current SP group.

    Whole-frame sharding can be uneven when the number of latent frames is not
    divisible by ``sp_size`` (for example 39 frames with SP2).  The Ulysses
    all-to-all then needs the per-rank lengths in order to exchange variable
    sequence chunks without introducing attention-visible padding.
    """
    state = get_sequence_parallel_state()
    local_length = int(local_length)
    if local_length < 0:
        raise ValueError(f"local SP sequence length must be non-negative, got {local_length}")
    if state.sp_size == 1:
        state.sequence_lengths = (local_length,)
        return state.sequence_lengths
    assert state.sp_group is not None
    value = torch.tensor([local_length], dtype=torch.int64, device="cuda")
    gathered = [torch.empty_like(value) for _ in range(state.sp_size)]
    dist.all_gather(gathered, value, group=state.sp_group)
    state.sequence_lengths = tuple(int(item.item()) for item in gathered)
    return state.sequence_lengths


def get_sequence_parallel_sequence_lengths() -> tuple[int, ...]:
    return get_sequence_parallel_state().sequence_lengths


def broadcast_sequence_parallel_tensor(input_: torch.Tensor, src_sp_rank: int = 0) -> torch.Tensor:
    """Broadcast a model input from one SP rank to its sequence-parallel peers."""
    state = get_sequence_parallel_state()
    if state.sp_size == 1:
        return input_
    if src_sp_rank < 0 or src_sp_rank >= state.sp_size:
        raise ValueError(f"src_sp_rank must be in [0, {state.sp_size}), got {src_sp_rank}")
    assert state.sp_group is not None
    src_global_rank = state.sp_group_ranks[src_sp_rank]
    if input_.is_contiguous():
        dist.broadcast(input_, src=src_global_rank, group=state.sp_group)
    else:
        staging = input_.contiguous()
        dist.broadcast(staging, src=src_global_rank, group=state.sp_group)
        input_.copy_(staging)
    return input_


class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, group: dist.ProcessGroup, input_: torch.Tensor, world_size: int, dim: int):
        ctx.group = group
        ctx.world_size = world_size
        ctx.dim = dim
        ctx.input_shape = input_.shape

        if dim < 0:
            dim += input_.dim()
            ctx.dim = dim

        state = get_sequence_parallel_state()
        lengths = state.sequence_lengths
        variable_sequence_gather = (
            dim == 1
            and len(lengths) == world_size
            and input_.size(dim) == lengths[state.sp_rank]
            and len(set(lengths)) > 1
        )
        ctx.variable_sequence_gather = variable_sequence_gather
        if variable_sequence_gather:
            ctx.sequence_lengths = lengths
            ctx.sp_rank = state.sp_rank
            max_length = max(lengths)
            padded_shape = list(input_.shape)
            padded_shape[dim] = max_length
            padded = input_.new_zeros(padded_shape)
            padded.narrow(dim, 0, input_.size(dim)).copy_(input_)
            gathered = [torch.empty_like(padded) for _ in range(world_size)]
            dist.all_gather(gathered, padded.contiguous(), group=group)
            return torch.cat(
                [tensor.narrow(dim, 0, length) for tensor, length in zip(gathered, lengths)],
                dim=dim,
            )

        input_size = input_.size()
        output_size = (input_size[0] * world_size,) + input_size[1:]
        output = torch.empty(output_size, dtype=input_.dtype, device=input_.device)
        dist.all_gather_into_tensor(output, input_.contiguous(), group=group)
        output = output.reshape((world_size,) + input_size)
        output = output.movedim(0, dim)
        return output.reshape(
            input_size[:dim] + (world_size * input_size[dim],) + input_size[dim + 1 :]
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.variable_sequence_gather:
            # Every SP rank continues from the identical gathered full sequence.
            # Sum those full-sequence gradients, then return the frame-aligned
            # slice owned by this rank.  This is the exact variable-length
            # analogue of reduce-scatter.
            reduced = grad_output.contiguous()
            dist.all_reduce(reduced, group=ctx.group)
            start = sum(ctx.sequence_lengths[: ctx.sp_rank])
            length = ctx.sequence_lengths[ctx.sp_rank]
            return None, reduced.narrow(ctx.dim, start, length).contiguous(), None, None
        dim_size = grad_output.size(ctx.dim) // ctx.world_size
        chunks = grad_output.reshape(
            grad_output.shape[: ctx.dim]
            + (ctx.world_size, dim_size)
            + grad_output.shape[ctx.dim + 1 :]
        )
        chunks = chunks.movedim(ctx.dim, 0).contiguous()
        grad_input = torch.empty(
            ctx.input_shape, dtype=grad_output.dtype, device=grad_output.device
        )
        dist.reduce_scatter_tensor(grad_input, chunks, group=ctx.group)
        return None, grad_input, None, None


class _AllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        group: dist.ProcessGroup,
        input_: torch.Tensor,
        world_size: int,
        scatter_dim: int,
        gather_dim: int,
    ):
        ctx.group = group
        ctx.world_size = world_size
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        if world_size == 1:
            return input_
        if input_.dim() != 4:
            raise RuntimeError(f"all_to_all_4D expects 4D input, got shape={tuple(input_.shape)}")

        state = get_sequence_parallel_state()
        sequence_lengths = state.sequence_lengths
        if len(sequence_lengths) != world_size:
            raise RuntimeError(
                "SP sequence lengths were not registered before all-to-all: "
                f"got {sequence_lengths}, expected {world_size} entries"
            )

        if scatter_dim == 2 and gather_dim == 1:
            bs, shard_seqlen, hn, hd = input_.shape
            if hn % world_size != 0:
                raise RuntimeError(f"num_heads={hn} must be divisible by sp_size={world_size}")
            if shard_seqlen != sequence_lengths[state.sp_rank]:
                raise RuntimeError(
                    "local SP sequence length disagrees with the registered layout: "
                    f"rank={state.sp_rank} tensor={shard_seqlen} registered={sequence_lengths}"
                )
            shard_hn = hn // world_size

            inp = input_.transpose(0, 2).contiguous()  # hn, shard_seq, bs, hd
            if len(set(sequence_lengths)) == 1:
                out = torch.empty_like(inp)
                dist.all_to_all_single(out, inp, group=group)
                out = torch.cat(out.split(shard_hn), dim=1)
                return out.transpose(0, 2).contiguous()  # bs, full_seq, shard_hn, hd

            element_factor = shard_hn * bs * hd
            input_split_sizes = [element_factor * shard_seqlen] * world_size
            output_split_sizes = [element_factor * length for length in sequence_lengths]
            inp_flat = inp.reshape(-1)
            out_flat = torch.empty(
                sum(output_split_sizes), dtype=input_.dtype, device=input_.device
            )
            dist.all_to_all_single(
                out_flat,
                inp_flat,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
            )
            source_chunks = []
            offset = 0
            for length, flat_size in zip(sequence_lengths, output_split_sizes):
                source_chunks.append(
                    out_flat.narrow(0, offset, flat_size).reshape(shard_hn, length, bs, hd)
                )
                offset += flat_size
            out = torch.cat(source_chunks, dim=1)
            return out.transpose(0, 2).contiguous()  # bs, full_seq, shard_hn, hd

        if scatter_dim == 1 and gather_dim == 2:
            bs, seqlen, shard_hn, hd = input_.shape
            if seqlen != sum(sequence_lengths):
                raise RuntimeError(
                    "full SP sequence length disagrees with the registered layout: "
                    f"tensor={seqlen} registered={sequence_lengths}"
                )
            shard_seqlen = sequence_lengths[state.sp_rank]
            inp = input_.transpose(0, 2).contiguous()  # shard_hn, full_seq, bs, hd
            if len(set(sequence_lengths)) == 1:
                inp = (
                    inp.reshape(shard_hn, world_size, shard_seqlen, bs, hd)
                    .transpose(0, 1)
                    .reshape(shard_hn * world_size, shard_seqlen, bs, hd)
                    .contiguous()
                )
                out = torch.empty_like(inp)
                dist.all_to_all_single(out, inp, group=group)
                return out.transpose(0, 2).contiguous()  # bs, shard_seq, full_hn, hd

            sequence_starts = [sum(sequence_lengths[:rank]) for rank in range(world_size)]
            destination_chunks = [
                inp.narrow(1, start, length).contiguous().reshape(-1)
                for start, length in zip(sequence_starts, sequence_lengths)
            ]
            inp_flat = torch.cat(destination_chunks)
            element_factor = shard_hn * bs * hd
            input_split_sizes = [element_factor * length for length in sequence_lengths]
            output_split_sizes = [element_factor * shard_seqlen] * world_size
            out_flat = torch.empty(
                sum(output_split_sizes), dtype=input_.dtype, device=input_.device
            )
            dist.all_to_all_single(
                out_flat,
                inp_flat,
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
            )
            source_chunks = []
            offset = 0
            for flat_size in output_split_sizes:
                source_chunks.append(
                    out_flat.narrow(0, offset, flat_size).reshape(shard_hn, shard_seqlen, bs, hd)
                )
                offset += flat_size
            out = torch.cat(source_chunks, dim=0)
            return out.transpose(0, 2).contiguous()  # bs, shard_seq, full_hn, hd

        raise RuntimeError(
            f"unsupported all_to_all_4D scatter_dim={scatter_dim}, gather_dim={gather_dim}"
        )

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.world_size == 1:
            return None, grad_output, None, None, None
        grad_input = _AllToAll4D.apply(
            ctx.group,
            grad_output,
            ctx.world_size,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return None, grad_input, None, None, None


def sequence_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    state = get_sequence_parallel_state()
    if state.sp_size == 1:
        return input_
    return _AllGather.apply(state.sp_group, input_, state.sp_size, dim)


def sequence_model_parallel_all_to_all_4D(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
) -> torch.Tensor:
    state = get_sequence_parallel_state()
    if state.sp_size == 1:
        return input_
    return _AllToAll4D.apply(state.sp_group, input_, state.sp_size, scatter_dim, gather_dim)


def sync_sequence_parallel_gradients(module: torch.nn.Module) -> None:
    """Sum sharded-token gradients across SP ranks after FSDP DP reduction.

    Loss is scaled by 1 / sp_size before backward. Summing the resulting partial
    token/head gradients across SP reconstructs the exact logical-sample
    gradient while keeping all SP replicas' optimizer states identical.
    """
    state = get_sequence_parallel_state()
    if state.sp_size == 1:
        return
    group = state.sp_group
    assert group is not None

    params = list(module.parameters())
    if not params:
        return

    # Check the entire grad/non-grad pattern with one collective instead of one
    # scalar all-reduce per parameter.
    flag_device = next((p.grad.device for p in params if p.grad is not None), params[0].device)
    grad_flags = torch.tensor(
        [1 if p.grad is not None else 0 for p in params],
        device=flag_device,
        dtype=torch.int32,
    )
    dist.all_reduce(grad_flags, op=dist.ReduceOp.SUM, group=group)
    bad_flags = (grad_flags != 0) & (grad_flags != state.sp_size)
    if bool(bad_flags.any().item()):
        bad_indices = bad_flags.nonzero(as_tuple=False).flatten()[:16].cpu().tolist()
        raise RuntimeError(
            "SP gradient mismatch: every SP rank must have the same grad/non-grad pattern; "
            f"first parameter indices={bad_indices}"
        )

    grads = [p.grad for p in params if p.grad is not None]
    bucket_limit = max(1, int(os.environ.get("SOLARWM_SP_GRAD_BUCKET_MB", "256"))) * 1024 * 1024
    buckets: list[list[torch.Tensor]] = []
    current: list[torch.Tensor] = []
    current_bytes = 0
    current_key = None
    for grad in grads:
        key = (grad.device, grad.dtype)
        grad_bytes = grad.numel() * grad.element_size()
        if current and (key != current_key or current_bytes + grad_bytes > bucket_limit):
            buckets.append(current)
            current = []
            current_bytes = 0
        current.append(grad)
        current_bytes += grad_bytes
        current_key = key
    if current:
        buckets.append(current)

    for bucket in buckets:
        flat = torch.cat([grad.reshape(-1) for grad in bucket])
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        offset = 0
        for grad in bucket:
            next_offset = offset + grad.numel()
            grad.copy_(flat[offset:next_offset].view_as(grad))
            offset = next_offset


def contiguous_sp_bounds(length: int, sp_size: int, sp_rank: int) -> tuple[int, int]:
    """Return the equal contiguous token shard owned by one SP rank."""

    length = int(length)
    sp_size = int(sp_size)
    sp_rank = int(sp_rank)
    if sp_size < 1:
        raise ValueError(f"sp_size must be >= 1, got {sp_size}")
    if not 0 <= sp_rank < sp_size:
        raise ValueError(f"sp_rank must be in [0, {sp_size}), got {sp_rank}")
    if length % sp_size:
        raise ValueError(f"padded sequence length={length} must be divisible by sp_size={sp_size}")
    shard_length = length // sp_size
    return sp_rank * shard_length, shard_length
