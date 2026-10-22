"""Default-OFF, request-owned backend representation normalization.

Only native H3 token-refiner processors are touched. The same registered backend
and its checks/arguments remain in use. No attention kernel selection, dispatch
replacement, compile fallback, parameter cast or official source modification."""

from contextlib import contextmanager


_ACTIVE=set()
class BackendRepresentationOwner:
 def __init__(self,refiner,expected_backend):
  from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3TokenRefiner, MiniMaxH3AttnProcessor
  from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
  if type(refiner) is not MiniMaxH3TokenRefiner or refiner.training:
   raise ValueError("native eval H3 token refiner required")
  if type(expected_backend) is not str:raise ValueError("explicit backend string required")
  key=AttentionBackendName(expected_backend)
  self.fn=_AttentionBackendRegistry._backends[key];self.key=key
  self.refiner=refiner;self.entries=[]
  for block in refiner.refiner_blocks:
   p=block.attn.processor
   if type(p) is not MiniMaxH3AttnProcessor or p._parallel_config is not None:
    raise ValueError("only native non-parallel token-refiner dispatch supported")
   old=p._attention_backend
   if not isinstance(old,(str,AttentionBackendName)) or AttentionBackendName(old)!=key:
    raise ValueError("backend identity differs; do not choose a new backend")
   self.entries.append((p,"_attention_backend" in p.__dict__,old))
  self.ids={id(p) for p,_,_ in self.entries}
  if not self.entries or len(self.ids)!=len(self.entries):raise ValueError("nonempty distinct processors required")
  if self.ids & _ACTIVE:raise RuntimeError("overlapping backend owner")
 def validate(self):
  from diffusers.models.attention_dispatch import _AttentionBackendRegistry
  if _AttentionBackendRegistry._backends[self.key] is not self.fn:
   raise RuntimeError("attention registry changed")
  if [id(b.attn.processor) for b in self.refiner.refiner_blocks]!=[id(p) for p,_,_ in self.entries]:
   raise RuntimeError("token-refiner processor changed")
  if any(type(p._attention_backend) is not str or p._attention_backend!=self.key.value or p._parallel_config is not None for p,_,_ in self.entries):
   raise RuntimeError("owned backend representation changed")
@contextmanager
def canonical_refiner_backend(refiner,*,enabled=False,expected_backend="flash"):
 if not enabled:
  yield None
  return
 owner=BackendRepresentationOwner(refiner,expected_backend)
 _ACTIVE.update(owner.ids)
 try:
  for p,_,_ in owner.entries:p._attention_backend=owner.key.value
  owner.validate()
  yield owner
  owner.validate()
 finally:
  for p,had,old in owner.entries:
   if had:p._attention_backend=old
   else:p.__dict__.pop("_attention_backend",None)
  _ACTIVE.difference_update(owner.ids)
