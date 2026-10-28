"""Balanced sequence partitions and unified variable-length collectives."""

from ..compatibility import expected_digest
from typing import Any
from pathlib import Path
import hashlib
import torch
import torch.distributed as dist
import solar.backends.minimax_h3.distributed as d
from . import all_gather as canonical


def contiguous_packed_bounds(total_tokens: int, *, sp_size: int, sp_rank: int):
    for name, value in (("total_tokens", total_tokens), ("sp_size", sp_size)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if isinstance(sp_rank, bool) or not isinstance(sp_rank, int):
        raise TypeError("sp_rank must be an integer")
    if not 0 <= sp_rank < sp_size:
        raise ValueError("invalid rank")
    if total_tokens < sp_size:
        raise ValueError("every rank needs one row")
    count = (total_tokens + sp_size - 1 - sp_rank) // sp_size
    start = sum((total_tokens + sp_size - 1 - rank) // sp_size
                for rank in range(sp_rank))
    stop = start + count
    return start, total_tokens if sp_rank == sp_size - 1 else stop


state = d.state
SOURCE_BINDINGS = {d.__file__: expected_digest("distributed"), canonical.__file__: expected_digest("canonical_all_gather")}

def validate_balanced():
    lengths = d.state().sequence_lengths
    if len(lengths) != 8 or any(type(x) is not int or x < 1 for x in lengths):
        raise RuntimeError('eight actual positive SP lengths required')
    if tuple(sorted(lengths, reverse=True)) != lengths or lengths[0]-lengths[-1] > 1:
        raise RuntimeError('balanced descending native SP partition required')

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

class StaticAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: Any, dim: int) -> Any:
        current = d.state()
        ctx.current = current
        normalized_dim = dim if dim >= 0 else value.dim() + dim
        ctx.dim = normalized_dim
        ctx.input_shape = value.shape
        ctx.lengths = current.sequence_lengths
        ctx.sp_rank = current.sp_rank
        variable = (
            normalized_dim == 1
            and len(ctx.lengths) == current.sp_size
            and value.size(normalized_dim) == ctx.lengths[current.sp_rank]
        )
        ctx.variable = variable
        if variable:
            maximum = ctx.lengths[0]
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
        current = ctx.current
        if ctx.variable:
            # Uneven documents cannot use fixed-size reduce-scatter.
            reduced = gradient.contiguous()
            dist.all_reduce(reduced, group=current.sp_group)
            start = sum(ctx.lengths[: ctx.sp_rank])
            length = ctx.lengths[ctx.sp_rank]
            return (reduced.narrow(ctx.dim, start, length).contiguous(), None)
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
        return (output, None)

class Installed:
    def __init__(self):
        for path, digest in SOURCE_BINDINGS.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError('frozen native collective source changed')
        self.raw = d._all_to_all_raw
        self.gather = canonical.StaticAllGather
        self.closed = False
        d._all_to_all_raw = _all_to_all_raw
        canonical.StaticAllGather = StaticAllGather

    def close(self):
        if self.closed:
            return
        if d._all_to_all_raw is not _all_to_all_raw or canonical.StaticAllGather is not StaticAllGather:
            raise RuntimeError('collective metadata patch ownership changed')
        d._all_to_all_raw = self.raw
        canonical.StaticAllGather = self.gather
        self.closed = True

def install():
    return Installed()
