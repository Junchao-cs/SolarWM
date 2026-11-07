"""Default-OFF PRoPE dynamic-import normalization; native math and positions intact."""

from contextlib import contextmanager
from typing import Any
import torch
import solar.backends.minimax_h3.torch_prope as p


_NATIVE=p._invert_se3
_ACTIVE=False

def _invert_se3(value: Any) -> Any:
    output = value.new_zeros(value.shape)
    rotation = value[..., :3, :3].transpose(-1, -2)
    output[..., :3, :3] = rotation
    output[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i", rotation, value[..., :3, 3]
    )
    output[..., 3, 3] = 1.0
    return output

class PropeImportOwner:
 def __init__(self):
  if p._invert_se3 is not _NATIVE:raise RuntimeError("unrecognized PRoPE inverse function")
  self.native=_NATIVE
 def validate(self):
  if p._invert_se3 is not _invert_se3:raise RuntimeError("PRoPE inverse owner changed")

@contextmanager
def canonical_prope_import(*,enabled=False):
 global _ACTIVE
 if not enabled:
  yield None
  return
 if _ACTIVE:raise RuntimeError("overlapping PRoPE inverse owner")
 owner=PropeImportOwner();_ACTIVE=True
 try:
  p._invert_se3=_invert_se3;owner.validate()
  yield owner
  owner.validate()
 finally:
  p._invert_se3=owner.native;_ACTIVE=False
