"""Default-OFF all-gather class hoist; native equal/unequal communication preserved."""
from contextlib import contextmanager
from typing import Any
import torch
import torch.distributed as dist
import solarwm.backends.minimax_h3.distributed as d

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

def all_gather_apply(input_: Any, dim: int) -> Any:
    return StaticAllGather.apply(input_, dim)

_NATIVE=d._all_gather_apply
_ACTIVE=False

class GatherOwner:
 def __init__(self,model):
  self.model=model;self.native=d._all_gather_apply;self.entry=d.sequence_all_gather
  if self.native is not _NATIVE or model.sequence_model_parallel_all_gather is not self.entry:
   raise RuntimeError("unrecognized all-gather entry")
  s=d.state()
  if not s.initialized or s.sp_size!=8 or s.sp_group is None:raise RuntimeError("initialized SP8 required")
  self.topology=self.signature()
 def signature(self):
  s=d.state()
  return id(s),s.sp_size,s.sp_rank,id(s.sp_group),s.sp_group_ranks
 def validate(self):
  if self.signature()!=self.topology:raise RuntimeError("all-gather topology changed")
  if d._all_gather_apply is not all_gather_apply or d.sequence_all_gather is not self.entry or self.model.sequence_model_parallel_all_gather is not self.entry:
   raise RuntimeError("all-gather owner changed")

@contextmanager
def canonical_all_gather(*,enabled=False):
 global _ACTIVE
 if not enabled:
  yield None
  return
 if _ACTIVE:raise RuntimeError("overlapping all-gather owner")
 import solarwm.backends.minimax_h3.model as model
 owner=GatherOwner(model);_ACTIVE=True
 try:
  d._all_gather_apply=all_gather_apply;owner.validate()
  yield owner
  owner.validate()
 finally:
  d._all_gather_apply=owner.native;_ACTIVE=False
