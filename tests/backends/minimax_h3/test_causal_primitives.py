from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from solarwm.backends.minimax_h3.artifacts import H3CameraFilterError, align_h3_camera
from solarwm.backends.minimax_h3.layout import build_stage1_layout
from solarwm.backends.minimax_h3.mask import build_stage1_window_mask
from solarwm.backends.minimax_h3.sgf import h3_sgf_schedule, h3_sgf_windows
from solarwm.backends.minimax_h3.sgf_attention import (
    H3RawKVCache,
    H3SGFAttention,
    prepare_sgf_rotaries,
)
from solarwm.backends.minimax_h3.torch_mask import build_stage1_w6_block_mask
from solarwm.training.sgf import compute_sgf_kl_gradient, sgf_student_loss, should_update_student


def test_stage1_direct_sparse_mask_preserves_visibility_across_sliding_boundary() -> None:
    source = build_stage1_layout([0, 1, 1], 50, 2, 2, 3)
    layout = source.to("cpu")
    mask = build_stage1_w6_block_mask(layout, device="cpu")
    rows = torch.arange(layout.sequence_length)
    actual = mask.mask_mod(torch.tensor(0), torch.tensor(0), rows[:, None], rows[None, :])
    expected = torch.from_numpy(build_stage1_window_mask(source))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_sgf_raw_cache_keeps_five_detached_chunks_and_checks_continuity() -> None:
    cache = H3RawKVCache()
    with torch.no_grad():
        for chunk in range(8):
            value = torch.full((1, 5, 2, 128), float(chunk))
            cache.commit(0, chunk, value, value)
            value.zero_()
    history = cache.history(0, 8)
    assert [entry[0] for entry in history] == [3, 4, 5, 6, 7]
    assert history[-1][1].mean().item() == 7
    with pytest.raises(RuntimeError, match="history differs"):
        cache.history(0, 9)
    with pytest.raises(RuntimeError, match="no-grad"):
        cache.commit(0, 8, value, value)
    cache.clear()
    assert cache.layers == {}


def test_sgf_local_rope_is_independent_of_global_chunk_index() -> None:
    layout = build_stage1_layout([1, 1], 50, 2, 2, 3).to("cpu")
    views = torch.eye(4).repeat(1, 51, 1, 1)
    intrinsics = torch.eye(3).repeat(1, 51, 1, 1)
    control = H3SGFAttention(layout, views, intrinsics, "rollout", chunk_index=5)
    first = prepare_sgf_rotaries(control, lambda positions: (positions.clone(),))
    last = prepare_sgf_rotaries(
        replace(control, chunk_index=9), lambda positions: (positions.clone(),)
    )
    torch.testing.assert_close(
        first.window_rotaries[30][0], last.window_rotaries[30][0], rtol=0, atol=0
    )
    assert first.window_rotaries[30][0][0, 0].item() == 2.0


def test_sgf_supervision_and_schedule_do_not_count_the_padded_tail() -> None:
    windows = h3_sgf_windows()
    assert len(windows) == 10
    assert sum(window.supervised_latents for window in windows) == 47
    assert windows[-1].supervised_latents == 2
    assert windows[-1].prior_chunks == (4, 5, 6, 7, 8)
    video, audio = h3_sgf_schedule(device="cpu")
    assert video.timesteps.numel() == audio.timesteps.numel() == 4
    assert video.sigmas[-1].item() == audio.sigmas[-1].item() == 0
    assert [index + 1 for index in range(16) if should_update_student(index, 5)] == [6, 11, 16]


def test_sgf_surrogate_has_exact_normalized_detached_gradient() -> None:
    output = torch.zeros(2, 3, 47, 2, 2, requires_grad=True)
    real, fake = torch.full_like(output, 2.0), torch.full_like(output, 3.0)
    gradient = compute_sgf_kl_gradient(fake_x0=fake, real_x0=real, student_output=output)
    loss = sgf_student_loss(output, gradient)
    loss.backward()
    torch.testing.assert_close(
        output.grad, torch.full_like(output, 0.5 / output.numel()), rtol=0, atol=0
    )


def test_stage1_camera_guard_only_filters_the_45_trained_latents() -> None:
    views = torch.eye(4).repeat(47, 1, 1)
    views[-2:, 0, 3] = 21
    K = torch.tensor([[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]])
    values = {"camera_c2w": views, "camera_K": K}
    accepted, _ = align_h3_camera(values, audit_latents=45)
    assert accepted.shape[0] == 47
    with pytest.raises(H3CameraFilterError):
        align_h3_camera(values, audit_latents=47)
    np.testing.assert_array_equal(accepted[0].numpy(), np.eye(4))
