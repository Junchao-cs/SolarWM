"""MiniMax-H3 packed sequence parallelism.

The public topology helpers stay importable without PyTorch.  CUDA and
``torch.distributed`` are resolved only when a process group or collective is
actually requested.  H3 uses contiguous SP groups and strided logical-DP
groups::

    global_rank = dp_rank * sp_size + sp_rank

Stage0.5 shards the *already packed* heterogeneous sequence.  Uneven caption
lengths are supported without attention-visible padding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class H3DistributedState:
    initialized: bool = False
    sp_size: int = 1
    sp_rank: int = 0
    dp_rank: int = 0
    dp_world_size: int = 1
    sp_group: Any = None
    dp_group: Any = None
    sp_group_ranks: tuple[int, ...] = (0,)
    dp_group_ranks: tuple[int, ...] = (0,)
    sequence_lengths: tuple[int, ...] = ()


_STATE = H3DistributedState()


def contiguous_packed_bounds(total_tokens: int, *, sp_size: int, sp_rank: int) -> tuple[int, int]:
    """Return the balanced contiguous ``[start, stop)`` shard for one rank."""

    for name, value in (("total_tokens", total_tokens), ("sp_size", sp_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if isinstance(sp_rank, bool) or not isinstance(sp_rank, int):
        raise TypeError("sp_rank must be an integer")
    if not 0 <= sp_rank < sp_size:
        raise ValueError(f"sp_rank must be in [0,{sp_size})")
    if total_tokens < sp_size:
        raise ValueError("every H3 SP rank must own at least one packed row")
    base, remainder = divmod(total_tokens, sp_size)
    count = base + int(sp_rank < remainder)
    start = sp_rank * base + min(sp_rank, remainder)
    return start, start + count


def logical_node_data_world_size(*, world_size: int, local_world_size: int, sp_size: int) -> int:
    """Return logical readers per node and prove SP groups cannot cross nodes."""

    for name, value in {
        "world_size": world_size,
        "local_world_size": local_world_size,
        "sp_size": sp_size,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if world_size % local_world_size:
        raise ValueError("world_size must be divisible by LOCAL_WORLD_SIZE")
    if world_size % sp_size or local_world_size % sp_size:
        raise ValueError("global and node-local worlds must be divisible by SP size")
    return local_world_size // sp_size


def init_sequence_parallel(sp_size: int = 1) -> H3DistributedState:
    """Create H3 SP and logical-DP groups after process-group initialization."""

    global _STATE
    import torch.distributed as dist

    sp_size = int(sp_size)
    if sp_size < 1:
        raise ValueError("sp_size must be positive")
    if not dist.is_available() or not dist.is_initialized():
        if sp_size != 1:
            raise RuntimeError("SP2 requires initialized torch.distributed")
        _STATE = H3DistributedState(initialized=True)
        return _STATE

    rank = dist.get_rank()
    world = dist.get_world_size()
    if world % sp_size:
        raise ValueError(f"world_size={world} is not divisible by sp_size={sp_size}")
    dp_world = world // sp_size
    sp_group = None
    dp_group = None
    sp_ranks: tuple[int, ...] = ()
    dp_ranks: tuple[int, ...] = ()
    # Every process creates groups in exactly the same order.
    for dp_index in range(dp_world):
        ranks = tuple(range(dp_index * sp_size, (dp_index + 1) * sp_size))
        group = dist.new_group(ranks=list(ranks))
        if rank in ranks:
            sp_group, sp_ranks = group, ranks
    for sp_index in range(sp_size):
        ranks = tuple(range(sp_index, world, sp_size))
        group = dist.new_group(ranks=list(ranks))
        if rank in ranks:
            dp_group, dp_ranks = group, ranks
    if sp_size > 1 and (sp_group is None or dp_group is None):
        raise RuntimeError("failed to construct H3 distributed groups")
    _STATE = H3DistributedState(
        initialized=True,
        sp_size=sp_size,
        sp_rank=rank % sp_size,
        dp_rank=rank // sp_size,
        dp_world_size=dp_world,
        sp_group=sp_group,
        dp_group=dp_group,
        sp_group_ranks=sp_ranks or (rank,),
        dp_group_ranks=dp_ranks or tuple(range(world)),
    )
    return _STATE


def state() -> H3DistributedState:
    return _STATE if _STATE.initialized else H3DistributedState(initialized=True)


def get_sp_size() -> int:
    return state().sp_size


def get_sp_rank() -> int:
    return state().sp_rank


def get_dp_rank() -> int:
    return state().dp_rank


def get_dp_world_size() -> int:
    return state().dp_world_size


def get_sp_group() -> Any:
    return state().sp_group


def get_dp_group() -> Any:
    return state().dp_group


def is_sequence_parallel_enabled() -> bool:
    return get_sp_size() > 1


def register_sequence_length(local_length: int) -> tuple[int, ...]:
    import torch
    import torch.distributed as dist

    current = state()
    local_length = int(local_length)
    if local_length < 0:
        raise ValueError("local sequence length must be non-negative")
    if current.sp_size == 1:
        current.sequence_lengths = (local_length,)
        return current.sequence_lengths
    value = torch.tensor([local_length], dtype=torch.int64, device="cuda")
    values = [torch.empty_like(value) for _ in range(current.sp_size)]
    dist.all_gather(values, value, group=current.sp_group)
    current.sequence_lengths = tuple(int(item.item()) for item in values)
    return current.sequence_lengths


def broadcast_sp_tensor(tensor: Any, src_sp_rank: int = 0) -> Any:
    import torch.distributed as dist

    current = state()
    if current.sp_size == 1:
        return tensor
    if not 0 <= src_sp_rank < current.sp_size:
        raise ValueError("src_sp_rank is outside the SP group")
    source = current.sp_group_ranks[src_sp_rank]
    if tensor.is_contiguous():
        dist.broadcast(tensor, src=source, group=current.sp_group)
    else:
        staging = tensor.contiguous()
        dist.broadcast(staging, src=source, group=current.sp_group)
        tensor.copy_(staging)
    return tensor


def _all_gather_apply(input_: Any, dim: int) -> Any:
    import torch
    import torch.distributed as dist

    current = state()

    class _AllGather(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, value: Any) -> Any:
            normalized_dim = dim if dim >= 0 else value.dim() + dim
            ctx.dim = normalized_dim
            ctx.input_shape = value.shape
            ctx.lengths = current.sequence_lengths
            ctx.sp_rank = current.sp_rank
            variable = (
                normalized_dim == 1
                and len(ctx.lengths) == current.sp_size
                and value.size(normalized_dim) == ctx.lengths[current.sp_rank]
                and len(set(ctx.lengths)) > 1
            )
            ctx.variable = variable
            if variable:
                maximum = max(ctx.lengths)
                shape = list(value.shape)
                shape[normalized_dim] = maximum
                padded = value.new_zeros(shape)
                padded.narrow(normalized_dim, 0, value.size(normalized_dim)).copy_(value)
                gathered = [torch.empty_like(padded) for _ in range(current.sp_size)]
                dist.all_gather(gathered, padded.contiguous(), group=current.sp_group)
                return torch.cat(
                    [
                        item.narrow(normalized_dim, 0, length)
                        for item, length in zip(gathered, ctx.lengths, strict=True)
                    ],
                    dim=normalized_dim,
                )
            input_shape = value.size()
            output_shape = (input_shape[0] * current.sp_size, *input_shape[1:])
            output = torch.empty(output_shape, dtype=value.dtype, device=value.device)
            dist.all_gather_into_tensor(output, value.contiguous(), group=current.sp_group)
            output = output.reshape((current.sp_size, *input_shape))
            output = output.movedim(0, normalized_dim)
            return output.reshape(
                (
                    *input_shape[:normalized_dim],
                    current.sp_size * input_shape[normalized_dim],
                    *input_shape[normalized_dim + 1 :],
                )
            )

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any]:
            if ctx.variable:
                # Uneven documents cannot use fixed-size reduce-scatter.
                reduced = gradient.contiguous()
                dist.all_reduce(reduced, group=current.sp_group)
                start = sum(ctx.lengths[: ctx.sp_rank])
                length = ctx.lengths[ctx.sp_rank]
                return (reduced.narrow(ctx.dim, start, length).contiguous(),)
            length = gradient.size(ctx.dim) // current.sp_size
            chunks = gradient.reshape(
                (
                    *gradient.shape[: ctx.dim],
                    current.sp_size,
                    length,
                    *gradient.shape[ctx.dim + 1 :],
                )
            )
            chunks = chunks.movedim(ctx.dim, 0).contiguous()
            output = torch.empty(
                ctx.input_shape,
                dtype=gradient.dtype,
                device=gradient.device,
            )
            dist.reduce_scatter_tensor(output, chunks, group=current.sp_group)
            return (output,)

    return _AllGather.apply(input_)


def sequence_all_gather(input_: Any, dim: int = 1) -> Any:
    if get_sp_size() == 1:
        return input_
    return _all_gather_apply(input_, dim)


def _all_to_all_raw(input_: Any, scatter_dim: int, gather_dim: int) -> Any:
    import torch
    import torch.distributed as dist

    current = state()
    world = current.sp_size
    if world == 1:
        return input_
    if input_.dim() != 4:
        raise RuntimeError("H3 Ulysses all-to-all requires [B,S,H,D]")
    lengths = current.sequence_lengths
    if len(lengths) != world:
        raise RuntimeError("SP sequence lengths were not registered")

    if scatter_dim == 2 and gather_dim == 1:
        batch, local_sequence, heads, head_dim = input_.shape
        if heads % world:
            raise RuntimeError("attention heads must be divisible by SP size")
        if local_sequence != lengths[current.sp_rank]:
            raise RuntimeError("local sequence length differs from registered SP layout")
        local_heads = heads // world
        transposed = input_.transpose(0, 2).contiguous()
        if len(set(lengths)) == 1:
            output = torch.empty_like(transposed)
            dist.all_to_all_single(output, transposed, group=current.sp_group)
            output = torch.cat(output.split(local_heads), dim=1)
            return output.transpose(0, 2).contiguous()
        factor = local_heads * batch * head_dim
        input_splits = [factor * local_sequence] * world
        output_splits = [factor * length for length in lengths]
        output = torch.empty(sum(output_splits), dtype=input_.dtype, device=input_.device)
        dist.all_to_all_single(
            output,
            transposed.reshape(-1),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=current.sp_group,
        )
        chunks = []
        offset = 0
        for length, size in zip(lengths, output_splits, strict=True):
            chunks.append(
                output.narrow(0, offset, size).reshape(local_heads, length, batch, head_dim)
            )
            offset += size
        return torch.cat(chunks, dim=1).transpose(0, 2).contiguous()

    if scatter_dim == 1 and gather_dim == 2:
        batch, sequence, local_heads, head_dim = input_.shape
        if sequence != sum(lengths):
            raise RuntimeError("full sequence length differs from registered SP layout")
        local_sequence = lengths[current.sp_rank]
        transposed = input_.transpose(0, 2).contiguous()
        if len(set(lengths)) == 1:
            transposed = (
                transposed.reshape(local_heads, world, local_sequence, batch, head_dim)
                .transpose(0, 1)
                .reshape(local_heads * world, local_sequence, batch, head_dim)
                .contiguous()
            )
            output = torch.empty_like(transposed)
            dist.all_to_all_single(output, transposed, group=current.sp_group)
            return output.transpose(0, 2).contiguous()
        starts = [sum(lengths[:rank]) for rank in range(world)]
        pieces = [
            transposed.narrow(1, start, length).contiguous().reshape(-1)
            for start, length in zip(starts, lengths, strict=True)
        ]
        factor = local_heads * batch * head_dim
        input_splits = [factor * length for length in lengths]
        output_splits = [factor * local_sequence] * world
        output = torch.empty(sum(output_splits), dtype=input_.dtype, device=input_.device)
        dist.all_to_all_single(
            output,
            torch.cat(pieces),
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=current.sp_group,
        )
        chunks = []
        offset = 0
        for size in output_splits:
            chunks.append(
                output.narrow(0, offset, size).reshape(local_heads, local_sequence, batch, head_dim)
            )
            offset += size
        return torch.cat(chunks, dim=0).transpose(0, 2).contiguous()
    raise RuntimeError(f"unsupported H3 all-to-all dimensions {scatter_dim}->{gather_dim}")


def sequence_all_to_all(input_: Any, scatter_dim: int = 2, gather_dim: int = 1) -> Any:
    if get_sp_size() == 1:
        return input_
    import torch

    class _AllToAll(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, value: Any) -> Any:
            ctx.scatter_dim = scatter_dim
            ctx.gather_dim = gather_dim
            return _all_to_all_raw(value, scatter_dim, gather_dim)

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any]:
            return (_all_to_all_raw(gradient, ctx.gather_dim, ctx.scatter_dim),)

    return _AllToAll.apply(input_)


def _sync_gradient_bucket(parameters: tuple[Any, ...], *, group: Any, average: bool) -> None:
    import os

    import torch
    import torch.distributed as dist

    if not parameters:
        return
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    flag_device = gradients[0].device if gradients else parameters[0].device
    flags = torch.tensor(
        [int(parameter.grad is not None) for parameter in parameters],
        device=flag_device,
        dtype=torch.int32,
    )
    group_size = dist.get_world_size(group=group)
    dist.all_reduce(flags, group=group)
    if bool(((flags != 0) & (flags != group_size)).any().item()):
        raise RuntimeError("H3 distributed ranks have different gradient patterns")
    if not gradients:
        return
    bucket_limit = max(
        1,
        int(os.environ.get("SOLARWM_H3_GRAD_BUCKET_MB", "16")),
    ) * (1 << 20)
    buckets: list[list[Any]] = []
    current_bucket: list[Any] = []
    current_bytes = 0
    current_key: tuple[Any, Any] | None = None
    for gradient in gradients:
        key = (gradient.device, gradient.dtype)
        size = int(gradient.numel()) * int(gradient.element_size())
        if current_bucket and (key != current_key or current_bytes + size > bucket_limit):
            buckets.append(current_bucket)
            current_bucket = []
            current_bytes = 0
        current_bucket.append(gradient)
        current_bytes += size
        current_key = key
    if current_bucket:
        buckets.append(current_bucket)
    for bucket in buckets:
        # Match FSDP's configured FP32 reduction. LoRA parameters are BF16
        # and deliberately ignored by FSDP, so a native-dtype all-reduce here
        # would silently lower both SP-sum and logical-DP-average precision.
        flat = torch.cat([gradient.detach().reshape(-1).float() for gradient in bucket])
        dist.all_reduce(flat, group=group)
        if average:
            flat.div_(group_size)
        offset = 0
        for gradient in bucket:
            count = gradient.numel()
            gradient.copy_(flat[offset : offset + count].view_as(gradient))
            offset += count


def sync_lora_gradients(parameters: tuple[Any, ...]) -> None:
    """Sum complementary SP work, then average replicated LoRA over logical DP."""

    current = state()
    if current.sp_size > 1:
        _sync_gradient_bucket(parameters, group=current.sp_group, average=False)
    if current.dp_world_size > 1:
        _sync_gradient_bucket(parameters, group=current.dp_group, average=True)


@dataclass(frozen=True)
class H3PackedSequenceShard:
    hidden_states: Any
    position_ids: Any
    token_tags: Any
    timestep_indices: Any
    prope_token_indices: Any
    prope_frame_ids: Any
    camera_viewmats: Any
    camera_K: Any
    start: int
    stop: int
    total_tokens: int
    sequence_lengths: tuple[int, ...]


def shard_stage0p5_packed_sequence(
    *,
    hidden_states: Any,
    position_ids: Any,
    token_tags: Any,
    timestep_indices: Any,
    prope_token_indices: Any,
    prope_frame_ids: Any,
    camera_viewmats: Any,
    camera_K: Any,
) -> H3PackedSequenceShard:
    """Shard the full packed document and localize camera row controls."""

    import torch

    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise ValueError("hidden_states must be [B,S,D]")
    total = int(hidden_states.shape[1])
    for name, value in (
        ("position_ids", position_ids),
        ("token_tags", token_tags),
        ("timestep_indices", timestep_indices),
    ):
        if not isinstance(value, torch.Tensor) or int(value.shape[0]) != total:
            raise ValueError(f"{name} must have one row per packed token")
    start, stop = contiguous_packed_bounds(total, sp_size=get_sp_size(), sp_rank=get_sp_rank())
    lengths = tuple(
        contiguous_packed_bounds(total, sp_size=get_sp_size(), sp_rank=rank)[1]
        - contiguous_packed_bounds(total, sp_size=get_sp_size(), sp_rank=rank)[0]
        for rank in range(get_sp_size())
    )
    if register_sequence_length(stop - start) != lengths:
        raise RuntimeError("SP ranks disagree on packed sequence lengths")

    controls = (prope_token_indices, prope_frame_ids, camera_viewmats, camera_K)
    if all(value is None for value in controls):
        local_indices = local_frames = local_views = local_K = None
    else:
        if prope_token_indices is None or camera_viewmats is None or camera_K is None:
            raise ValueError("camera PRoPE requires indices, viewmats, and K")
        indices = prope_token_indices.to(dtype=torch.long)
        if indices.ndim != 1 or (
            indices.numel() and (int(indices.min()) < 0 or int(indices.max()) >= total)
        ):
            raise ValueError("camera PRoPE indices are outside the packed sequence")
        selected = (indices >= start) & (indices < stop)
        local_indices = (indices[selected] - start).contiguous()
        if prope_frame_ids is not None:
            if tuple(prope_frame_ids.shape) != tuple(indices.shape):
                raise ValueError("camera frame IDs must align with PRoPE row indices")
            local_frames = prope_frame_ids[selected].contiguous()
            local_views, local_K = camera_viewmats, camera_K
        else:
            if camera_viewmats.shape[1] != indices.numel() or camera_K.shape[1] != indices.numel():
                raise ValueError("token-aligned camera tensors must align with PRoPE indices")
            local_frames = None
            local_views = camera_viewmats[:, selected].contiguous()
            local_K = camera_K[:, selected].contiguous()

    return H3PackedSequenceShard(
        hidden_states=hidden_states[:, start:stop].contiguous(),
        position_ids=position_ids[start:stop].contiguous(),
        token_tags=token_tags[start:stop].contiguous(),
        timestep_indices=timestep_indices[start:stop].contiguous(),
        prope_token_indices=local_indices,
        prope_frame_ids=local_frames,
        camera_viewmats=local_views,
        camera_K=local_K,
        start=start,
        stop=stop,
        total_tokens=total,
        sequence_lengths=lengths,
    )


__all__ = [
    "H3DistributedState",
    "H3PackedSequenceShard",
    "broadcast_sp_tensor",
    "contiguous_packed_bounds",
    "get_dp_group",
    "get_dp_rank",
    "get_dp_world_size",
    "get_sp_group",
    "get_sp_rank",
    "get_sp_size",
    "init_sequence_parallel",
    "is_sequence_parallel_enabled",
    "logical_node_data_world_size",
    "register_sequence_length",
    "sequence_all_gather",
    "sequence_all_to_all",
    "shard_stage0p5_packed_sequence",
    "state",
    "sync_lora_gradients",
]


def prepare_flex_attention_input(tensor: Any) -> Any:
    """Convert BSHD to contiguous BHSD without changing values or autograd.

    Ulysses returns sequence-major storage. PyTorch 2.6 Flex cannot order its
    symbolic transposed strides; the head-major layout matches native H3 SP1.
    An already head-major input view needs no copy.
    """

    return tensor.transpose(1, 2).contiguous()


def validate_packed_attention_mask(attention_mask: Any, *, total_tokens: int) -> None:
    """Require a global, head-broadcast Flex mask before Ulysses sharding.

    Ulysses restores the full packed sequence while distributing heads. H3's
    W6 visibility is head-independent, so the original global mask is reused
    on every SP rank; neither its rows nor its head dimension may be sharded.
    """

    if attention_mask is None:
        return
    if type(attention_mask).__name__ != "BlockMask":
        raise ValueError("H3 packed SP accepts only a global Flex BlockMask or no mask")
    if tuple(attention_mask.seq_lengths) != (total_tokens, total_tokens):
        raise ValueError("H3 packed SP BlockMask must cover the complete global sequence")
    for name in (
        "kv_num_blocks",
        "kv_indices",
        "full_kv_num_blocks",
        "full_kv_indices",
        "q_num_blocks",
        "q_indices",
        "full_q_num_blocks",
        "full_q_indices",
    ):
        metadata = getattr(attention_mask, name, None)
        if metadata is not None and (metadata.ndim < 2 or metadata.shape[1] != 1):
            raise ValueError("H3 packed SP requires head-broadcast BlockMask metadata")
