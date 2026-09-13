import numpy as np
import pytest

from solarwm.backends.minimax_h3.raw_data import _normalise_K
from solarwm.errors import DataContractError

TRANSFORM = dict(
    source_w=1280,
    source_h=720,
    resized_w=1344,
    resized_h=768,
    target_w=1344,
    target_h=768,
    crop_left=0,
    crop_top=0,
)


def test_full_evaluation_retains_unused_source_focal_values():
    source = np.array([[-4938.0, -4946.0, 640.0, 360.0], [28157.0, 28193.0, 640.0, 360.0]])
    with pytest.raises(DataContractError, match="PRoPE guards"):
        _normalise_K(source, (0, 1), TRANSFORM)
    result = _normalise_K(source, (0, 1), TRANSFORM, enforce_focal_guard=False)
    np.testing.assert_allclose(result[:, 0, 0], source[:, 0] / 1280, rtol=1e-6)
    np.testing.assert_allclose(result[:, 1, 1], source[:, 1] / 720, rtol=1e-6)


def test_valid_conditions_are_unchanged():
    source = np.array([1000.0, 1000.0, 640.0, 360.0])
    np.testing.assert_array_equal(
        _normalise_K(source, (0, 1), TRANSFORM),
        _normalise_K(source, (0, 1), TRANSFORM, enforce_focal_guard=False),
    )


def test_full_evaluation_still_rejects_nonfinite_intrinsics():
    with pytest.raises(DataContractError, match="PRoPE guards"):
        _normalise_K(
            np.array([1000.0, np.inf, 640.0, 360.0]), (0,), TRANSFORM, enforce_focal_guard=False
        )


def test_full_evaluation_still_rejects_missing_camera_rows():
    with pytest.raises(DataContractError, match="cannot align"):
        _normalise_K(np.ones((2, 4)), (0, 1, 2), TRANSFORM, enforce_focal_guard=False)
