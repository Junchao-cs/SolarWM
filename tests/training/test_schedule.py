from __future__ import annotations

import pytest

from solar.errors import BackendContractError
from solar.training.schedule import warmup_cosine_factor


def test_warmup_cosine_matches_formula() -> None:
    assert warmup_cosine_factor(0, warmup_steps=10, total_steps=100) == 0.0
    assert warmup_cosine_factor(5, warmup_steps=10, total_steps=100) == 0.5
    assert warmup_cosine_factor(10, warmup_steps=10, total_steps=100) == 1.0
    assert warmup_cosine_factor(100, warmup_steps=10, total_steps=100) == pytest.approx(0.1)
    assert warmup_cosine_factor(200, warmup_steps=10, total_steps=100) == pytest.approx(0.1)


def test_schedule_rejects_invalid_bounds() -> None:
    for values in (
        {"step": -1, "warmup_steps": 10, "total_steps": 100},
        {"step": 0, "warmup_steps": -1, "total_steps": 100},
        {"step": 0, "warmup_steps": 10, "total_steps": 0},
    ):
        with pytest.raises(BackendContractError):
            warmup_cosine_factor(**values)


def test_bounded_run_can_end_during_full_recipe_warmup() -> None:
    assert warmup_cosine_factor(0, warmup_steps=1000, total_steps=100) == 0.0
    assert warmup_cosine_factor(50, warmup_steps=1000, total_steps=100) == 0.05
    assert warmup_cosine_factor(100, warmup_steps=1000, total_steps=100) == 0.1
