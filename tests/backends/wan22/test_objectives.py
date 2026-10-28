from __future__ import annotations

import numpy as np

from solar.backends.wan22.objectives import (
    apply_timestep_shift_array,
    rectified_interpolate_array,
    velocity_target_array,
    weighted_masked_mse_array,
)


def test_velocity_and_interpolation_convention() -> None:
    clean = np.array([[[1.0, 3.0]]], dtype=np.float32)
    noise = np.array([[[5.0, -1.0]]], dtype=np.float32)
    sigma = np.array([[0.25]], dtype=np.float32)
    np.testing.assert_array_equal(velocity_target_array(clean, noise), [[[4.0, -4.0]]])
    np.testing.assert_array_equal(rectified_interpolate_array(clean, noise, sigma), [[[2.0, 2.0]]])


def test_rational_timestep_shift() -> None:
    value = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    actual = apply_timestep_shift_array(value, 5.0)
    expected = np.array([0.0, 5.0 / 6.0, 1.0], dtype=np.float64)
    np.testing.assert_array_equal(actual, expected)


def test_mask_happens_before_subtraction() -> None:
    prediction = np.array([[np.nan, 3.0]], dtype=np.float32)
    target = np.array([[np.nan, 1.0]], dtype=np.float32)
    loss = weighted_masked_mse_array(prediction, target, [[False, True]], [[1.0, 1.0]])
    assert float(loss) == 4.0


def test_weight_scales_numerator_but_not_denominator() -> None:
    prediction = np.array([[0.0, 2.0]], dtype=np.float32)
    target = np.zeros_like(prediction)
    loss = weighted_masked_mse_array(prediction, target, [[True, True]], [[1.0, 3.0]])
    assert float(loss) == 6.0
