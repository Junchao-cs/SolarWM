from __future__ import annotations

import numpy as np

from solar.backends.wan22.anyflow import (
    SAMPLE_TYPE_CONSISTENCY,
    SAMPLE_TYPE_DIFFUSION,
    SAMPLE_TYPE_FLOW_MAP,
    bounded_difference_timesteps_array,
    build_flowmap_schedule_array,
    central_difference_target_array,
    gaussian_timestep_weights_array,
    time_pairs_from_uniforms,
)


def test_v15_pair_type_assignment_uses_logical_global_batch() -> None:
    pairs = time_pairs_from_uniforms(
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        logical_dp_rank=1,
        logical_dp_world_size=2,
        diffusion_ratio=0.5,
        consistency_ratio=0.25,
    )
    assert pairs.sample_type.tolist() == [
        SAMPLE_TYPE_CONSISTENCY,
        SAMPLE_TYPE_CONSISTENCY,
        SAMPLE_TYPE_FLOW_MAP,
        SAMPLE_TYPE_FLOW_MAP,
    ]
    assert bool(np.all(pairs.r[pairs.is_consistency] == 0))
    flow = ~(pairs.is_diffusion | pairs.is_consistency)
    assert bool(np.all(pairs.r[flow] > 0))
    assert bool(np.all(pairs.t[flow] > pairs.r[flow]))


def test_first_logical_rank_contains_diffusion_samples() -> None:
    pairs = time_pairs_from_uniforms(
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        logical_dp_rank=0,
        logical_dp_world_size=2,
    )
    assert pairs.sample_type.tolist() == [SAMPLE_TYPE_DIFFUSION] * 4
    np.testing.assert_array_equal(pairs.t, pairs.r)


def test_flowmap_schedule_has_shared_endpoints() -> None:
    t, r = build_flowmap_schedule_array(4, shift=5.0)
    assert t.shape == r.shape == (4,)
    assert t[0].item() == 1000.0
    assert r[-1].item() == 0.0
    np.testing.assert_array_equal(t[1:], r[:-1])


def test_bounded_difference_and_diagonal_target() -> None:
    plus, minus = bounded_difference_timesteps_array([998.0, 7.0], [0.0, 6.0])
    assert plus.tolist() == [1000.0, 12.0]
    assert minus.tolist() == [993.0, 6.0]
    velocity = np.array([[2.0, 4.0]], dtype=np.float32)
    target = central_difference_target_array(
        velocity,
        np.full_like(velocity, np.nan),
        np.full_like(velocity, np.nan),
        np.array([5.0]),
        np.array([5.0]),
    )
    np.testing.assert_array_equal(target, velocity)


def test_v15_gaussian_grid_is_mean_normalized() -> None:
    normalized = np.linspace(1.0, 0.0, 1001, dtype=np.float64)[:-1]
    raw = 5.0 * normalized / (1.0 + 4.0 * normalized) * 1000
    weights = gaussian_timestep_weights_array(raw, shift=5.0)
    assert np.isclose(np.mean(weights), 1.0)
    assert weights[0] == 0.0
