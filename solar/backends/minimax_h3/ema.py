"""Rank-local FP32 EMA aligned with H3 FSDP/LoRA parameter ownership."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


def _named_parameters(module: Any, *, trainable_only: bool) -> list[tuple[str, Any]]:
    root = module.module if hasattr(module, "module") else module
    return [
        (name, parameter)
        for name, parameter in root.named_parameters()
        if not trainable_only or parameter.requires_grad
    ]


class H3ShardedEMA:
    """FP32 EMA over each rank's ``use_orig_params`` parameter view.

    Updates require no full-state gather. Ignored adapter parameters are
    replicated and identically updated; FSDP-owned
    parameters would retain their local shard shape.
    """

    def __init__(
        self,
        module: Any,
        *,
        decay: float = 0.9999,
        device: str | torch.device = "cuda",
        trainable_only: bool = True,
    ) -> None:
        self.decay = float(decay)
        if not 0 <= self.decay <= 1:
            raise ValueError("EMA decay must be in [0,1]")
        self.device = torch.device(device)
        self.trainable_only = bool(trainable_only)
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {}
        self.reset_from(module)

    @property
    def initialized(self) -> bool:
        return bool(self.shadow)

    def _validate(self, module: Any) -> list[tuple[str, Any]]:
        named = _named_parameters(module, trainable_only=self.trainable_only)
        if {name for name, _ in named} != set(self.shadow):
            raise RuntimeError("EMA parameter names differ from the live model")
        for name, parameter in named:
            shadow = self.shadow[name]
            if tuple(shadow.shape) != tuple(parameter.shape) or shadow.dtype != torch.float32:
                raise RuntimeError(f"EMA shard layout differs for {name!r}")
        return named

    @torch.no_grad()
    def reset_from(self, module: Any) -> None:
        self.shadow = {
            name: parameter.detach().to(device=self.device, dtype=torch.float32).clone()
            for name, parameter in _named_parameters(module, trainable_only=self.trainable_only)
        }
        self.num_updates = 0

    @torch.no_grad()
    def update(self, module: Any) -> None:
        for name, parameter in self._validate(module):
            value = parameter.detach().to(device=self.shadow[name].device, dtype=torch.float32)
            self.shadow[name].mul_(self.decay).add_(value, alpha=1.0 - self.decay)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "solarwm.minimax-h3-ema.v1",
            "decay": self.decay,
            "num_updates": self.num_updates,
            "trainable_only": self.trainable_only,
            "shadow": {name: value.detach().cpu() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, values: dict[str, Any], module: Any) -> None:
        if values.get("schema") != "solarwm.minimax-h3-ema.v1":
            raise ValueError("unsupported H3 EMA checkpoint schema")
        if bool(values.get("trainable_only")) != self.trainable_only:
            raise ValueError("EMA trainable-only policy differs")
        shadow = values.get("shadow")
        if not isinstance(shadow, dict):
            raise ValueError("EMA checkpoint lacks shadow tensors")
        self.shadow = {
            name: value.detach().to(device=self.device, dtype=torch.float32).clone()
            for name, value in shadow.items()
        }
        self.decay = float(values["decay"])
        self.num_updates = int(values["num_updates"])
        self._validate(module)

    @torch.no_grad()
    def copy_to(self, module: Any) -> None:
        for name, parameter in self._validate(module):
            parameter.copy_(self.shadow[name].to(parameter.device, parameter.dtype))

    @contextmanager
    def swapped_into(self, module: Any) -> Iterator[None]:
        backup = {name: parameter.detach().clone() for name, parameter in self._validate(module)}
        try:
            self.copy_to(module)
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self._validate(module):
                    parameter.copy_(backup[name])


__all__ = ["H3ShardedEMA"]
