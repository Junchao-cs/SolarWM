from __future__ import annotations

import numpy as np

from solar.backends.minimax_h3.layout import (
    VIDEO_TAG,
    build_row_timesteps,
    build_stage0p5_layout,
    build_stage1_layout,
    patchify_video,
    unpatchify_video,
)
from solar.backends.minimax_h3.mask import (
    build_stage1_window_mask,
    stage0p5_full_attention,
    stage1_window_allows,
)


def test_stage0p5_packing_and_explicit_camera_rows() -> None:
    # Row zero is a Qwen vision row: VIDEO_TAG alone must not select it for camera PRoPE.
    layout = build_stage0p5_layout(
        np.asarray([VIDEO_TAG, 1], dtype=np.int64),
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
    )
    assert layout.rows_per_video_frame == 4
    assert layout.sequence_length == 2 + 4 + 4 + 8
    assert 0 not in layout.camera_video_indices
    np.testing.assert_array_equal(layout.camera_frame_ids, [0] * 8 + [1] * 4)
    np.testing.assert_array_equal(
        layout.camera_video_indices,
        np.concatenate((layout.condition_video_indices, layout.target_video_indices)),
    )
    attention = stage0p5_full_attention(layout)
    assert attention.shape == (layout.sequence_length, layout.sequence_length)
    assert attention.allows(0, layout.sequence_length - 1)


def test_row_timesteps_preserve_condition_audio_and_video_roles() -> None:
    layout = build_stage0p5_layout([1], 2, 4, 4, num_audio_latents=1)
    distinct, inverse = build_row_timesteps(layout, 0.2, 0.7)
    row_times = distinct[inverse]
    np.testing.assert_allclose(row_times[layout.text_indices], 0.2)
    np.testing.assert_allclose(row_times[layout.condition_video_indices], 0.999)
    np.testing.assert_allclose(row_times[layout.audio_indices], 0.7)
    np.testing.assert_allclose(row_times[layout.noisy_video_indices], 0.2)


def test_patchify_round_trip() -> None:
    latents = np.arange(2 * 3 * 2 * 4 * 6, dtype=np.float32).reshape(2, 3, 2, 4, 6)
    rows = patchify_video(latents)
    assert rows.shape == (2, 2 * 2 * 3, 3 * 4)
    restored = unpatchify_video(rows, 2, 4, 6, channels=3)
    np.testing.assert_array_equal(restored, latents)


def test_stage1_dense_mask_matches_scalar_window_contract() -> None:
    layout = build_stage1_layout(
        [1],
        num_latent_frames=4,
        latent_height=2,
        latent_width=2,
        num_audio_latents=1,
        chunk_latent_frames=2,
        window_chunks=2,
    )
    dense = build_stage1_window_mask(layout)
    for query in range(layout.sequence_length):
        for key in range(layout.sequence_length):
            assert bool(dense[query, key]) == stage1_window_allows(
                int(layout.row_roles[query]),
                int(layout.target_video_chunk_ids[query]),
                int(layout.row_roles[key]),
                int(layout.target_video_chunk_ids[key]),
                window_chunks=2,
            )


def test_six_chunk_window_excludes_old_clean_chunk_but_keeps_own_noisy_chunk() -> None:
    from solar.backends.minimax_h3.layout import CLEAN_VIDEO_ROLE, NOISY_VIDEO_ROLE

    assert not stage1_window_allows(NOISY_VIDEO_ROLE, 6, CLEAN_VIDEO_ROLE, 0, window_chunks=6)
    assert stage1_window_allows(NOISY_VIDEO_ROLE, 6, CLEAN_VIDEO_ROLE, 1, window_chunks=6)
    assert stage1_window_allows(NOISY_VIDEO_ROLE, 6, NOISY_VIDEO_ROLE, 6, window_chunks=6)
