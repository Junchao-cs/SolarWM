"""Logical data-parallel topology shared by data and model execution."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from solar.errors import ConfigurationError


@dataclass(frozen=True)
class Topology:
    raw_world_size: int
    raw_rank: int
    local_world_size: int
    local_rank: int
    sp_size: int = 1

    @classmethod
    def from_environ(cls, sp_size: int, environ: Mapping[str, str] | None = None) -> Topology:
        values = os.environ if environ is None else environ
        required = ("WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK")
        missing = [name for name in required if name not in values]
        if missing:
            raise ConfigurationError(f"torchrun environment is missing {missing}")
        try:
            return cls(
                raw_world_size=int(values["WORLD_SIZE"]),
                raw_rank=int(values["RANK"]),
                local_world_size=int(values["LOCAL_WORLD_SIZE"]),
                local_rank=int(values["LOCAL_RANK"]),
                sp_size=int(sp_size),
            )
        except ValueError as exc:
            raise ConfigurationError("torchrun rank variables must be integers") from exc

    def __post_init__(self) -> None:
        for name in ("raw_world_size", "local_world_size", "sp_size"):
            if getattr(self, name) < 1:
                raise ConfigurationError(f"{name} must be positive")
        if not 0 <= self.raw_rank < self.raw_world_size:
            raise ConfigurationError("raw_rank is outside raw_world_size")
        if not 0 <= self.local_rank < self.local_world_size:
            raise ConfigurationError("local_rank is outside local_world_size")
        if self.raw_world_size % self.sp_size:
            raise ConfigurationError("raw_world_size must be divisible by sp_size")
        if self.local_world_size % self.sp_size:
            raise ConfigurationError("local_world_size must be divisible by sp_size")

    @property
    def dp_world_size(self) -> int:
        return self.raw_world_size // self.sp_size

    @property
    def dp_rank(self) -> int:
        return self.raw_rank // self.sp_size

    @property
    def sp_rank(self) -> int:
        return self.raw_rank % self.sp_size

    @property
    def node_count(self) -> int:
        if self.raw_world_size % self.local_world_size:
            raise ConfigurationError("raw_world_size must be divisible by local_world_size")
        return self.raw_world_size // self.local_world_size

    @property
    def node_id(self) -> int:
        return self.raw_rank // self.local_world_size

    @property
    def local_dp_world_size(self) -> int:
        return self.local_world_size // self.sp_size

    @property
    def local_dp_rank(self) -> int:
        return self.local_rank // self.sp_size
