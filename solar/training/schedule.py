"""Pure learning-rate schedule math plus an optional torch adapter."""

from __future__ import annotations

import math
from typing import Any

from solar.errors import BackendContractError


def warmup_cosine_factor(
    step: int, *, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1
) -> float:
    # A bounded training run may intentionally stop before the warmup inherited
    # from the full training recipe completes.  In that case
    # ``total_steps < warmup_steps`` is well-defined: every executed step uses
    # the same warmup prefix as the full run, and the cosine phase is never
    # reached.
    if step < 0 or warmup_steps < 0 or total_steps < 1:
        raise BackendContractError("invalid warmup/cosine step bounds")
    if not 0 <= min_lr_ratio <= 1:
        raise BackendContractError("min_lr_ratio must be in [0,1]")
    if step < warmup_steps:
        return float(step) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return cosine * (1 - min_lr_ratio) + min_lr_ratio


def make_warmup_cosine(
    optimizer: Any,
    *,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise BackendContractError("torch is required to construct a scheduler") from exc

    def factor(step: int) -> float:
        return warmup_cosine_factor(
            step,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio,
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
