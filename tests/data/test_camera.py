from __future__ import annotations

import numpy as np
import pytest

from solar.data.camera import (
    CameraGuards,
    camera_audit_prefix_frames,
    causal_pixel_indices,
    load_camera_npz,
    relative_c2w,
    relative_w2c,
    transform_translation,
)
from solar.errors import DataContractError


def _trajectory() -> np.ndarray:
    camera = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
    camera[:, 0, 3] = [2.0, 3.0, 6.0]
    return camera


def test_relative_conventions_are_inverse() -> None:
    c2w = _trajectory()
    rel_w2c = relative_w2c(c2w, "absolute_c2w")
    rel_c2w = relative_c2w(c2w, "absolute_c2w")
    np.testing.assert_allclose(rel_w2c[0], np.eye(4))
    np.testing.assert_allclose(rel_c2w[0], np.eye(4))
    np.testing.assert_allclose(rel_w2c[:, 0, 3], [0.0, -1.0, -4.0])
    np.testing.assert_allclose(rel_c2w[:, 0, 3], [0.0, 1.0, 4.0])


def test_logd4_is_zero_safe_and_direction_preserving() -> None:
    relative = relative_c2w(_trajectory(), "absolute_c2w")
    transformed = transform_translation(relative, "logd4")
    assert transformed[0, 0, 3] == 0.0
    assert transformed[1, 0, 3] > 0.0
    assert transformed[2, 0, 3] > transformed[1, 0, 3]
    np.testing.assert_array_equal(transformed[:, :3, :3], relative[:, :3, :3])


def test_guards_use_unscaled_translation() -> None:
    relative = relative_c2w(_trajectory(), "absolute_c2w")
    with pytest.raises(DataContractError, match="translation"):
        CameraGuards(max_rel_translation=3.0).apply(relative)
    np.testing.assert_array_equal(CameraGuards().apply(relative), relative)


def test_causal_indices_match_documented_profiles() -> None:
    assert causal_pixel_indices(81, 4).tolist() == [0, *range(1, 81, 4)]
    assert causal_pixel_indices(153, 8).tolist() == [0, *range(1, 153, 8)]
    assert len(causal_pixel_indices(153, 8)) == 20


def test_camera_audit_prefix_matches_first_ten_seconds() -> None:
    assert camera_audit_prefix_frames(953, 15.0) == 150
    assert camera_audit_prefix_frames(81, 15.0) == 81
    assert camera_audit_prefix_frames(81, 0.0) == 81


def test_npz_camera_key_and_storage_are_explicit() -> None:
    import io

    stream = io.BytesIO()
    np.savez(stream, c2w=_trajectory())
    matrices, storage = load_camera_npz(stream.getvalue(), "c2w")
    assert storage == "absolute_c2w"
    np.testing.assert_array_equal(matrices, _trajectory())
    with pytest.raises(DataContractError, match="lacks declared"):
        load_camera_npz(stream.getvalue(), "w2c")


def test_npz_camera_loader_preserves_source_precision() -> None:
    import io

    camera = _trajectory().astype(np.float64)
    camera[1, 0, 3] = np.nextafter(3.0, 4.0)
    stream = io.BytesIO()
    np.savez(stream, c2w=camera)

    matrices, storage = load_camera_npz(stream.getvalue(), "c2w")

    assert storage == "absolute_c2w"
    assert matrices.dtype == np.float64
    np.testing.assert_array_equal(matrices, camera)
