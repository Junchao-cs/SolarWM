from __future__ import annotations

import numpy as np
import pytest

from solar.backends.minimax_h3.flow import (
    data_velocity_target,
    euler_step,
    make_shifted_schedule,
    predict_clean_sample,
    sample_shifted_timestep,
    scale_noise,
)
from solar.backends.minimax_h3.geometry import (
    STABLE_STAGE0P5_GEOMETRY,
    align_pixel_frames,
    audio_latents_for_video,
    latent_aligned_pixel_indices,
    latent_frames_to_pixel_frames,
    native_video_position_grid,
    pixel_frames_to_latent_frames,
    temporal_position_grid,
    validate_stage0p5_geometry,
)


def test_exact_stable_geometry_and_pixel_alignment() -> None:
    profile = validate_stage0p5_geometry(
        pixel_frames=158,
        encoded_latents=47,
        height=768,
        width=1344,
        latent_channels=24,
        latent_height=48,
        latent_width=84,
    )
    assert profile == STABLE_STAGE0P5_GEOMETRY
    assert profile.rows_per_latent == 1008
    assert pixel_frames_to_latent_frames(158) == 47
    assert latent_frames_to_pixel_frames(47) == 158
    assert align_pixel_frames(155) == 158
    assert audio_latents_for_video(158) == 263
    aligned = latent_aligned_pixel_indices(158)
    assert aligned.shape == (47,)
    np.testing.assert_array_equal(aligned[:7], [0, 1, 5, 9, 13, 17, 18])
    np.testing.assert_array_equal(aligned[-2:], [153, 154])


def test_invalid_geometry_fails_closed() -> None:
    with pytest.raises(ValueError, match="supports only"):
        validate_stage0p5_geometry(
            pixel_frames=158,
            encoded_latents=47,
            height=480,
            width=832,
            latent_channels=24,
            latent_height=30,
            latent_width=52,
        )
    with pytest.raises(ValueError, match="form"):
        pixel_frames_to_latent_frames(157)


def test_native_position_grid_uses_nonuniform_time() -> None:
    times = temporal_position_grid(7)
    np.testing.assert_allclose(
        times,
        [0, 5 / 3, 25 / 3, 15, 65 / 3, 85 / 3, 30],
    )
    grid = native_video_position_grid(2, 4, 4)
    assert grid.shape == (8, 3)
    np.testing.assert_allclose(grid[:4, 0], 0)
    np.testing.assert_allclose(grid[4:, 0], 5 / 3)


def test_flow_reconstruction_sign_and_broadcast() -> None:
    clean = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    noise = np.full_like(clean, -2)
    timestep = np.asarray([0.25, 0.75], dtype=np.float32)
    noisy = scale_noise(clean, noise, timestep)
    velocity = data_velocity_target(clean, noise)
    reconstructed = predict_clean_sample(noisy, velocity, timestep)
    np.testing.assert_allclose(reconstructed, clean, rtol=1e-6, atol=1e-6)


def test_shifted_rng_schedule_and_euler_are_deterministic() -> None:
    left = sample_shifted_timestep(8, shift=12.0, generator=np.random.default_rng(1234))
    right = sample_shifted_timestep(8, shift=12.0, generator=np.random.default_rng(1234))
    np.testing.assert_array_equal(left, right)
    assert np.all((left >= 0) & (left <= 1))

    schedule = make_shifted_schedule(30, shift=12.0)
    assert schedule.sigmas[0] == pytest.approx(1.0)
    assert schedule.sigmas[-1] == pytest.approx(0.0)
    assert schedule.timesteps.shape == (schedule.sigmas.size - 1,)
    assert np.all(np.diff(schedule.sigmas) < 0)

    sample = np.asarray([3.0], dtype=np.float16)
    velocity = np.asarray([2.0], dtype=np.float16)
    result = euler_step(sample, velocity, 0.5, 0.5, 0.0)
    np.testing.assert_array_equal(result, np.asarray([4.0], dtype=np.float16))

    result64 = euler_step(
        np.asarray([3.0], dtype=np.float64),
        np.asarray([2.0], dtype=np.float64),
        0.5,
        0.5,
        0.25,
    )
    assert result64.dtype == np.float64
