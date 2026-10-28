"""Collective-free rank-local EMA for FSDP-exposed parameter shards."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from solar.errors import BackendContractError


def ema_decay_for_step(*, target_decay: float, global_step: int, warmup_steps: int = 0) -> float:
    decay = float(target_decay)
    step = int(global_step)
    warmup = int(warmup_steps)
    if not 0 <= decay <= 1 or step < 0 or warmup < 0:
        raise BackendContractError("invalid EMA decay/step/warmup")
    return 0.0 if step < warmup else decay


class ShardedEMA:
    """Track only locally exposed shards; save-time gathering belongs to the backend."""

    def __init__(
        self,
        module: Any,
        *,
        decay: float,
        device: Any,
        dtype: Any,
        trainable_only: bool = False,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise BackendContractError("torch is required for EMA") from exc
        if not 0 <= float(decay) <= 1:
            raise BackendContractError("EMA decay must be in [0,1]")
        if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
            raise BackendContractError("EMA dtype must be floating point")
        self.decay = float(decay)
        self.device = torch.device(device)
        self.dtype = dtype
        self.trainable_only = bool(trainable_only)
        self.num_updates = 0
        self.shadow = {
            name: parameter.detach().to(device=self.device, dtype=dtype).clone()
            for name, parameter in self._named(module)
        }
        if not self.shadow:
            raise BackendContractError("EMA selected no parameters")

    def _named(self, module: Any) -> list[tuple[str, Any]]:
        root = getattr(module, "module", module)
        return [
            (name, parameter)
            for name, parameter in root.named_parameters()
            if not self.trainable_only or parameter.requires_grad
        ]

    def _validate(self, module: Any) -> list[tuple[str, Any]]:
        named = self._named(module)
        if {name for name, _ in named} != set(self.shadow):
            raise BackendContractError("EMA parameter names differ from the live model")
        for name, parameter in named:
            shadow = self.shadow[name]
            if shadow.shape != parameter.shape or shadow.dtype != self.dtype:
                raise BackendContractError(f"EMA shard layout differs for {name}")
        return named

    def update(self, module: Any, *, decay: float | None = None) -> None:
        import torch

        value = self.decay if decay is None else float(decay)
        if not 0 <= value <= 1:
            raise BackendContractError("EMA update decay must be in [0,1]")
        with torch.no_grad():
            for name, parameter in self._validate(module):
                live = parameter.detach().to(device=self.device, dtype=self.dtype)
                self.shadow[name].mul_(value).add_(live, alpha=1 - value)
        self.num_updates += 1

    def state_dict(self) -> dict[str, Any]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def load_state_dict(self, values: Mapping[str, Any], *, num_updates: int) -> None:
        if set(values) != set(self.shadow):
            raise BackendContractError("EMA checkpoint names differ from the live layout")
        restored = {}
        for name, current in self.shadow.items():
            value = values[name]
            if value.shape != current.shape or not value.is_floating_point():
                raise BackendContractError(f"EMA checkpoint layout differs for {name}")
            restored[name] = value.detach().to(device=self.device, dtype=self.dtype, copy=True)
        if int(num_updates) < 0:
            raise BackendContractError("EMA update count must be non-negative")
        self.shadow = restored
        self.num_updates = int(num_updates)

    @contextmanager
    def swapped_into(self, module: Any) -> Iterator[None]:
        import torch

        named = self._validate(module)
        live = {name: parameter.detach().clone() for name, parameter in named}
        try:
            with torch.no_grad():
                for name, parameter in named:
                    parameter.copy_(
                        self.shadow[name].to(device=parameter.device, dtype=parameter.dtype)
                    )
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self._named(module):
                    parameter.copy_(live[name])
