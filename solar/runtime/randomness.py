"""Named RNG domains shared by data-parallel and sequence-parallel execution."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from solar.errors import ConfigurationError

from .topology import Topology

_MODEL_INIT_NAMESPACES = {
    "wan22_ti2v_5b": 3_509_493_760,
    "wan22_i2v_a14b": 3_509_493_760,
    "minimax_h3": 1_211_301_888,
    "ltx25_video": 1_280_596_005,
}


@dataclass(frozen=True)
class RNGIdentity:
    """Seeds recorded beside every training microbatch."""

    base_seed: int
    model_init_seed: int
    objective_seed: int
    dp_rank: int
    sp_rank: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def model_init_seed(family: str, base_seed: int) -> int:
    try:
        namespace = _MODEL_INIT_NAMESPACES[family]
    except KeyError as exc:
        raise ConfigurationError(f"unknown RNG namespace for {family!r}") from exc
    return namespace + int(base_seed)


def objective_seed(base_seed: int, dp_rank: int) -> int:
    """Objective seed; SP peers intentionally receive the same value."""

    if dp_rank < 0:
        raise ConfigurationError("dp_rank must be non-negative")
    return int(base_seed) * 100003 + int(dp_rank) * 1024


def rng_identity(family: str, base_seed: int, topology: Topology) -> RNGIdentity:
    return RNGIdentity(
        base_seed=int(base_seed),
        model_init_seed=model_init_seed(family, base_seed),
        objective_seed=objective_seed(base_seed, topology.dp_rank),
        dp_rank=topology.dp_rank,
        sp_rank=topology.sp_rank,
    )


def seed_process(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and torch when installed."""

    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def stable_validation_seed(video_id: str, step: int, slot: int, base_seed: int) -> int:
    """Derive a stable Wan generation seed from public case identity."""

    payload = f"{video_id}|{int(step)}|{int(slot)}|{int(base_seed)}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def torch_generator(seed: int, *, device: Any) -> Any:
    """Construct a torch generator lazily so core inspection needs no torch."""

    try:
        import torch
    except ImportError as exc:
        raise ConfigurationError("torch is required to construct a model RNG") from exc
    return torch.Generator(device=device).manual_seed(int(seed))
