"""Default-OFF SP all-to-all class hoist; native raw collectives unchanged."""

from contextlib import contextmanager
import torch
import solarwm.backends.minimax_h3.distributed as d


_ACTIVE=False
class StaticAllToAll(torch.autograd.Function):
 @staticmethod
 def forward(ctx,value,scatter_dim,gather_dim):
  ctx.scatter_dim=scatter_dim;ctx.gather_dim=gather_dim
  return d._all_to_all_raw(value,scatter_dim,gather_dim)
 @staticmethod
 def backward(ctx,gradient):
  return d._all_to_all_raw(gradient,ctx.gather_dim,ctx.scatter_dim),None,None
def sequence_all_to_all(input_,scatter_dim=2,gather_dim=1):
 if d.get_sp_size()==1:return input_
 return StaticAllToAll.apply(input_,scatter_dim,gather_dim)
class AllToAllOwner:
 def __init__(self,model_module):
  self.model_module=model_module;self.state=d.state()
  self.native=d.sequence_all_to_all;self.raw=d._all_to_all_raw
  if model_module.sequence_model_parallel_all_to_all_4D is not self.native:
   raise RuntimeError("unrecognized model all-to-all alias")
  if not self.state.initialized or self.state.sp_size!=8 or self.state.sp_group is None:
   raise RuntimeError("explicit initialized SP8 required")
  self.topology=self.signature()
 def signature(self):
  s=d.state()
  return id(s),s.sp_size,s.sp_rank,id(s.sp_group),s.sp_group_ranks
 def validate(self):
  if self.signature()!=self.topology or d._all_to_all_raw is not self.raw:
   raise RuntimeError("native topology/raw collective changed")
  if d.sequence_all_to_all is not sequence_all_to_all or self.model_module.sequence_model_parallel_all_to_all_4D is not sequence_all_to_all:
   raise RuntimeError("all-to-all owner changed")
@contextmanager
def canonical_all_to_all(*,enabled=False):
 global _ACTIVE
 if not enabled:
  yield None
  return
 if _ACTIVE:raise RuntimeError("overlapping all-to-all owner")
 import solarwm.backends.minimax_h3.model as model
 owner=AllToAllOwner(model);_ACTIVE=True
 try:
  d.sequence_all_to_all=sequence_all_to_all
  model.sequence_model_parallel_all_to_all_4D=sequence_all_to_all
  owner.validate()
  yield owner
  owner.validate()
 finally:
  d.sequence_all_to_all=owner.native
  model.sequence_model_parallel_all_to_all_4D=owner.native
  _ACTIVE=False
