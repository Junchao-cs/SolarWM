from __future__ import annotations

import io

import numpy as np
import pytest

from solar.backends.minimax_h3.raw_data import decode_camera
from solar.errors import DataContractError

TRANSFORM = {
    "source_h": 768,
    "source_w": 1344,
    "resized_h": 768,
    "resized_w": 1344,
    "crop_top": 0,
    "crop_left": 0,
    "target_h": 768,
    "target_w": 1344,
}


def _camera_bytes(c2w: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.savez(output, c2w=c2w)
    return output.getvalue()


def _intrinsics_bytes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    np.save(output, values)
    return output.getvalue()


def _poses(count: int) -> np.ndarray:
    values = np.repeat(np.eye(4, dtype=np.float32)[None], count, axis=0)
    values[:, 0, 3] = np.arange(count, dtype=np.float32)
    return values


def test_h3_camera_track_must_cover_a_nonzero_source_window() -> None:
    with pytest.raises(DataContractError, match="shorter than the selected window"):
        decode_camera(
            _camera_bytes(_poses(158)),
            tuple(range(7, 165)),
            TRANSFORM,
        )


@pytest.mark.parametrize("layout", ("matrix", "vector"))
def test_h3_per_frame_intrinsics_must_cover_a_nonzero_source_window(layout: str) -> None:
    if layout == "matrix":
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 158, axis=0)
        message = "camera K cannot align"
    else:
        intrinsics = np.repeat(
            np.asarray([[1.0, 1.0, 0.5, 0.5]], dtype=np.float32),
            158,
            axis=0,
        )
        message = "intrinsics vectors cannot align"
    with pytest.raises(DataContractError, match=message):
        decode_camera(
            _camera_bytes(_poses(165)),
            tuple(range(7, 165)),
            TRANSFORM,
            intrinsics_bytes=_intrinsics_bytes(intrinsics),
        )


def test_h3_full_camera_tracks_select_the_exact_source_window() -> None:
    poses = _poses(165)
    selected, _ = decode_camera(
        _camera_bytes(poses),
        tuple(range(7, 165)),
        TRANSFORM,
    )
    np.testing.assert_array_equal(selected, poses[7:165])
