from __future__ import annotations

import numpy as np
import pytest

from solar.backends.ltx25.camera import (
    canonicalize_signed_focal_gauge,
    condition_camera,
    expand_token_camera,
    prepare_latent_camera,
    validate_latent_camera,
)
from solar.backends.ltx25.geometry import (
    STABLE_GEOMETRY,
    causal_camera_pixel_indices,
    cover_resize,
    pixel_frames_to_latent_frames,
    transform_intrinsics,
    validate_contiguous_source_indices,
)
from solar.errors import BackendContractError


def _identity_poses(rows: int) -> np.ndarray:
    return np.broadcast_to(np.eye(4), (rows, 4, 4)).copy()


def _intrinsics(rows: int, focal: float = 1.0) -> np.ndarray:
    value = np.asarray([[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]])
    return np.broadcast_to(value, (rows, 3, 3)).copy()


def test_fixed_geometry_and_causal_rows() -> None:
    assert pixel_frames_to_latent_frames(153) == 20
    assert STABLE_GEOMETRY.latent_shape == (128, 20, 16, 24)
    assert STABLE_GEOMETRY.video_tokens == 7680
    assert causal_camera_pixel_indices() == (
        0,
        1,
        9,
        17,
        25,
        33,
        41,
        49,
        57,
        65,
        73,
        81,
        89,
        97,
        105,
        113,
        121,
        129,
        137,
        145,
    )
    with pytest.raises(BackendContractError, match="causal-VAE"):
        pixel_frames_to_latent_frames(152)


def test_cover_resize_uses_realized_integer_geometry() -> None:
    transform = cover_resize(720, 1280)
    assert (transform.resized_height, transform.resized_width) == (512, 910)
    assert (transform.crop_top, transform.crop_left) == (0, 71)


def test_normalized_optical_zoom_above_four_is_preserved() -> None:
    transform = cover_resize(512, 768)
    result = transform_intrinsics(
        [[11.0, 0.0, 0.5], [0.0, 8.0, 0.5], [0.0, 0.0, 1.0]],
        transform,
        input_normalized=True,
    )
    assert result[0, 0] == pytest.approx(11.0)
    assert result[1, 1] == pytest.approx(8.0)


def test_signed_focal_gauge_preserves_projective_camera() -> None:
    pose = np.eye(4)
    pose[:3, 3] = (1.0, 2.0, 3.0)
    K = np.asarray([[-2.0, 0.0, 0.5], [0.0, 3.0, 0.5], [0.0, 0.0, 1.0]])
    canonical_pose, canonical_K = canonicalize_signed_focal_gauge(pose, K)
    original_projection = K @ np.linalg.inv(pose)[:3]
    canonical_projection = canonical_K @ np.linalg.inv(canonical_pose)[:3]
    assert canonical_K[0, 0] > 0 and canonical_K[1, 1] > 0
    assert np.allclose(canonical_projection, -original_projection)


def test_absolute_c2w_converts_once_to_relative_w2c() -> None:
    poses = _identity_poses(153)
    poses[:, 0, 3] = np.arange(153)
    relative, K = prepare_latent_camera(
        poses,
        _intrinsics(153),
        convention="absolute_c2w",
        resize=cover_resize(512, 768),
    )
    assert np.array_equal(relative[0], np.eye(4))
    assert relative[1, 0, 3] == pytest.approx(-1.0)
    assert relative[-1, 0, 3] == pytest.approx(-145.0)
    assert K.shape == (20, 3, 3)


def test_relative_w2c_is_not_rebased_twice() -> None:
    poses = _identity_poses(20)
    poses[:, 1, 3] = -np.arange(20)
    relative, _ = prepare_latent_camera(
        poses,
        _intrinsics(20),
        convention="relative_w2c",
    )
    assert np.array_equal(relative, poses.astype(np.float32))


def test_token_camera_repeats_each_latent_row_384_times() -> None:
    poses = _identity_poses(20)
    poses[1, 0, 3] = -1.0
    camera = expand_token_camera(poses, _intrinsics(20))
    assert camera.viewmats.shape == (1, 7680, 4, 4)
    assert np.array_equal(camera.viewmats[0, 383], poses[0])
    assert np.array_equal(camera.viewmats[0, 384], poses[1])


def test_logd4_is_model_only_and_zero_safe_after_runtime_guards() -> None:
    poses = _identity_poses(20)
    poses[1, 0, 3] = 4.0
    conditioned = condition_camera(
        poses,
        _intrinsics(20),
        translation_transform="logd4",
    )
    assert conditioned.viewmats[0, 0, 0, 3] == 0.0
    assert conditioned.viewmats[0, 384, 0, 3] == pytest.approx(np.log1p(4.0) / 4.0)
    assert poses[1, 0, 3] == 4.0
    poses[1, 0, 3] = 21.0
    with pytest.raises(BackendContractError, match="translation"):
        condition_camera(poses, _intrinsics(20))


def test_camera_and_source_validation_fail_closed() -> None:
    poses = _identity_poses(20)
    poses[0, 0, 3] = 1.0
    with pytest.raises(BackendContractError, match="row zero"):
        validate_latent_camera(poses, _intrinsics(20))
    indices = np.arange(153, dtype=np.int64)
    indices[5] += 1
    with pytest.raises(BackendContractError, match="consecutive"):
        validate_contiguous_source_indices(indices)
