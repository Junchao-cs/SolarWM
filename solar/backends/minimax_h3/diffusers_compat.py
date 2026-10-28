"""Narrow compatibility patch for the H3 Diffusers build."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_KNOWN_ERRORS = (
    "local variable 'tuple' referenced before assignment",
    "cannot access local variable 'tuple' where it is not associated with a value",
)


def _tensor_attribute_dtype(module: Any) -> Any:
    import torch

    def tensor_attributes(owner: Any) -> list[tuple[str, Any]]:
        return [(name, value) for name, value in owner.__dict__.items() if torch.is_tensor(value)]

    last = None
    for _name, tensor in module._named_members(get_members_fn=tensor_attributes):
        last = tensor.dtype
        if tensor.is_floating_point():
            return tensor.dtype
    return last


def patch_minimax_h3_parameter_dtype() -> Callable[[Any], Any]:
    """Patch the Python-3.10 local-``tuple`` FSDP failure.

    FSDP temporarily exposes leaf parameters as plain tensor attributes.  The
    H3 Diffusers implementation reaches a fallback whose annotation shadows
    ``tuple`` on Python 3.10.  No other exception or normal code path is hidden.
    """

    import diffusers.models.transformers.transformer_minimax_h3 as module

    current = module.get_parameter_dtype
    if getattr(current, "_solarwm_h3_tuple_compat", False):
        return current

    def compatible(owner: Any) -> Any:
        try:
            return current(owner)
        except UnboundLocalError as exc:
            if not any(message in str(exc) for message in _KNOWN_ERRORS):
                raise
            if any(owner.named_parameters()) or any(owner.buffers()):
                raise
            return _tensor_attribute_dtype(owner)

    compatible.__name__ = current.__name__
    compatible.__doc__ = current.__doc__
    compatible._solarwm_h3_tuple_compat = True  # type: ignore[attr-defined]
    module.get_parameter_dtype = compatible
    return compatible


__all__ = ["patch_minimax_h3_parameter_dtype"]
