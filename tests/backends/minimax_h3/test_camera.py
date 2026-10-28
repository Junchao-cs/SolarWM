from __future__ import annotations

import numpy as np
import pytest

from solar.backends.minimax_h3.camera import (
    H3_CAMERA_PROPE_DIM_START,
    first_frame_relative_w2c,
    h3_fused_prope_contract,
    logd4_relative_viewmats,
    prepare_camera_suffix_matrices,
    prope_qkv,
    select_camera_rows,
    validate_absolute_c2w,
)
from solar.backends.minimax_h3.torch_prope import _project


def _poses(translations: list[tuple[float, float, float]]) -> np.ndarray:
    result = np.repeat(np.eye(4, dtype=np.float32)[None], len(translations), axis=0)
    result[:, :3, 3] = np.asarray(translations, dtype=np.float32)
    return result


def _intrinsics(length: int, focal: float = 1.0) -> np.ndarray:
    result = np.repeat(np.eye(3, dtype=np.float32)[None], length, axis=0)
    result[:, 0, 0] = focal
    result[:, 1, 1] = focal
    result[:, 0, 2] = 0.5
    result[:, 1, 2] = 0.5
    return result


def test_absolute_c2w_to_relative_w2c_preserves_translation_scale() -> None:
    c2w = _poses([(10, 0, 0), (12, 0, 0), (12, 3, 0)])
    relative = first_frame_relative_w2c(c2w)
    np.testing.assert_array_equal(relative[0], np.eye(4, dtype=np.float32))
    np.testing.assert_allclose(relative[1, :3, 3], [-2, 0, 0])
    np.testing.assert_allclose(relative[2, :3, 3], [-2, -3, 0])

    angle = np.float32(np.pi / 7)
    rotated = _poses([(0, 0, 0), (1, 0, 0)])
    rotated[:, :2, :2] = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(first_frame_relative_w2c(rotated)[0], np.eye(4, dtype=np.float32))


def test_logd4_is_zero_safe_and_rotation_invariant() -> None:
    views = _poses([(0, 0, 0), (3, 4, 0)])
    compressed = logd4_relative_viewmats(views)
    np.testing.assert_array_equal(compressed[0, :3, 3], np.zeros(3, np.float32))
    expected_scale = np.log1p(5.0) / (4.0 * 5.0)
    np.testing.assert_allclose(compressed[1, :3, 3], np.asarray([3, 4, 0]) * expected_scale)
    assert np.isfinite(compressed).all()


def test_camera_suffix_preserves_native_prefix_and_projective_pairing() -> None:
    rng = np.random.default_rng(7)
    q = rng.standard_normal((1, 2, 3, 128), dtype=np.float32)
    k = rng.standard_normal((1, 2, 3, 128), dtype=np.float32)
    v = rng.standard_normal((1, 2, 3, 128), dtype=np.float32)
    views = first_frame_relative_w2c(_poses([(0, 0, 0), (1, 0, 0), (1, 2, 0)]))[None]
    Ks = _intrinsics(3)[None]
    q_camera, k_camera, v_camera, apply_output = prope_qkv(q, k, v, viewmats=views, Ks=Ks)
    start = H3_CAMERA_PROPE_DIM_START
    np.testing.assert_array_equal(q_camera[..., :start], q[..., :start])
    np.testing.assert_array_equal(k_camera[..., :start], k[..., :start])
    np.testing.assert_array_equal(v_camera[..., :start], v[..., :start])
    output = apply_output(v_camera)
    np.testing.assert_array_equal(output[..., :start], v[..., :start])
    np.testing.assert_allclose(output[..., start:], v[..., start:], rtol=2e-5, atol=2e-5)

    # P^T q and P^-1 k preserve every 4D projective inner product.
    before = (q[..., start:].reshape(1, 2, 3, 8, 4) * k[..., start:].reshape(1, 2, 3, 8, 4)).sum(-1)
    after = (
        q_camera[..., start:].reshape(1, 2, 3, 8, 4) * k_camera[..., start:].reshape(1, 2, 3, 8, 4)
    ).sum(-1)
    np.testing.assert_allclose(after, before, rtol=2e-5, atol=2e-5)


def test_torch_camera_projection_matches_bfloat16_attention_dtype() -> None:
    torch = pytest.importorskip("torch")
    features = torch.randn(1, 2, 3, 8, dtype=torch.bfloat16)
    matrix = torch.eye(4, dtype=torch.float32).expand(1, 3, 4, 4)

    projected = _project(features, matrix=matrix)

    assert projected.dtype == torch.bfloat16
    torch.testing.assert_close(projected, features)


def test_wan_fixed_runtime_ignores_input_intrinsics_values() -> None:
    views = first_frame_relative_w2c(_poses([(0, 0, 0), (1, 0, 0)]))[None]
    left = prepare_camera_suffix_matrices(views, _intrinsics(2, 0.7)[None])
    right = prepare_camera_suffix_matrices(views, _intrinsics(2, 1.7)[None])
    np.testing.assert_array_equal(left.query, right.query)
    np.testing.assert_array_equal(left.key_value, right.key_value)
    np.testing.assert_array_equal(left.output, right.output)


def test_camera_row_selection_and_contract_fingerprint() -> None:
    views = first_frame_relative_w2c(_poses([(0, 0, 0), (1, 0, 0)]))
    rows, Ks = select_camera_rows(views, _intrinsics(2), [0, 0, 1])
    assert rows.shape == (1, 3, 4, 4)
    assert Ks.shape == (1, 3, 3, 3)
    assert h3_fused_prope_contract()["camera_prope_head_slice"] == [96, 128]


def test_camera_validation_rejects_reflection() -> None:
    c2w = _poses([(0, 0, 0)])
    c2w[0, 0, 0] = -1
    with pytest.raises(ValueError, match="determinant"):
        validate_absolute_c2w(c2w)
