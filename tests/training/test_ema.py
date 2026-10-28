from __future__ import annotations

import pytest

from solar.errors import BackendContractError
from solar.training.ema import ema_decay_for_step


def test_ema_warmup_is_copy_then_target_decay() -> None:
    assert ema_decay_for_step(target_decay=0.9999, global_step=9, warmup_steps=10) == 0
    assert ema_decay_for_step(target_decay=0.9999, global_step=10, warmup_steps=10) == 0.9999


def test_ema_decay_validation() -> None:
    with pytest.raises(BackendContractError):
        ema_decay_for_step(target_decay=1.1, global_step=0)
