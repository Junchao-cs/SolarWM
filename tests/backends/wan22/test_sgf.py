from __future__ import annotations

import numpy as np
import pytest

from solar.backends.wan22.sgf import (
    compute_kl_gradient_array,
    should_update_student,
    student_update_steps,
    validate_checkpoint_transaction,
)
from solar.errors import BackendContractError


def test_student_update_schedule_warms_critic_first() -> None:
    assert not should_update_student(0, 5)
    assert should_update_student(5, 5)
    assert not should_update_student(6, 5)
    assert student_update_steps(20, 5) == (5, 10, 15)


def test_checkpoint_pair_uses_model_as_commit_marker() -> None:
    validate_checkpoint_transaction(["critic.pt", "model.pt"])
    with pytest.raises(BackendContractError, match=r"critic.pt then model.pt"):
        validate_checkpoint_transaction(["model.pt", "critic.pt"])


def test_masked_kl_gradient_ignores_nonfinite_anchor() -> None:
    fake = np.array([[np.nan, 3.0]], dtype=np.float32)
    real = np.array([[np.nan, 1.0]], dtype=np.float32)
    student = np.array([[np.nan, 5.0]], dtype=np.float32)
    value = compute_kl_gradient_array(
        fake_x0=fake,
        real_x0=real,
        student_output=student,
        mask=np.array([[False, True]]),
    )
    np.testing.assert_array_equal(value, [[0.0, 0.5]])
