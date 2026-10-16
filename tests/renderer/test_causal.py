from __future__ import annotations

import numpy as np
import pytest
import torch

from solarwm.backends.minimax_h3.distributed import validate_packed_attention_mask
from solarwm.backends.minimax_h3.layout import H3VisualObservationSpec, build_stage0p5_layout
from solarwm.backends.minimax_h3.segmented_attention import (
    H3ChunkKVCache,
    H3SegmentPlan,
    segment_attention,
)
from solarwm.renderer.causal import causal_rows, training_plan


@pytest.mark.parametrize("teacher", [False, True])
def test_segment_attention_matches_dense_attention_and_gradients(teacher):
    layout = build_stage0p5_layout(
        [1, 1],
        12,
        2,
        2,
        2,
        keyframe_anchors=(),
        observations=(H3VisualObservationSpec(12, 2, 2, tuple(range(12))),),
    )
    rows = causal_rows(layout, 0, 5)
    plan, _ = training_plan(rows, window=2, teacher=teacher)
    size = plan.sequence_length
    mask = torch.zeros(size, size, dtype=torch.bool)
    for start, stop, spans, _, _ in plan.segments:
        for a, b in spans:
            mask[start:stop, a:b] = True
    torch.manual_seed(42)
    tensors = [torch.randn(1, size, 2, 8, requires_grad=True) for _ in range(3)]
    actual = segment_attention(*tensors, plan, 0)
    expected = torch.nn.functional.scaled_dot_product_attention(
        *(tensor.transpose(1, 2) for tensor in tensors), attn_mask=mask
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.square().sum(), tensors, retain_graph=True)
    expected_grads = torch.autograd.grad(expected.square().sum(), tensors)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad)
    for index, (start, stop) in enumerate(rows.target):
        assert not mask[start:stop, rows.target[index][1] : rows.target[-1][1]].any()
        if index >= 2:
            expired_start, expired_stop = rows.structure[index - 2]
            assert not mask[start:stop, expired_start:expired_stop].any()
    validate_packed_attention_mask(plan, total_tokens=size)
    with pytest.raises(ValueError, match="complete global"):
        validate_packed_attention_mask(plan, total_tokens=size + 1)


def test_cache_detaches_owns_storage_and_evicts_old_chunks():
    cache = H3ChunkKVCache(window=2)
    value = torch.ones(1, 2, 1, 4, requires_grad=True)
    with pytest.raises(RuntimeError, match="without a graph"):
        cache.put(0, ("clean", 0), value, value)
    with torch.no_grad():
        for chunk in range(4):
            cache.put(0, ("clean", chunk), value, value)
        value.zero_()
    assert set(cache.layers[0]) == {("clean", 2), ("clean", 3)}
    key, _ = cache.get(0, ("clean", 3))
    assert key.sum() == 8 and key.grad_fn is None


def test_aligned_observations_share_target_positions_and_free_ones_do_not_use_camera():
    layout = build_stage0p5_layout(
        [1],
        7,
        4,
        4,
        2,
        keyframe_anchors=(),
        observations=(
            H3VisualObservationSpec(7, 4, 4, tuple(range(7))),
            H3VisualObservationSpec(1, 2, 2),
        ),
    )
    aligned = layout.observation_video_indices[:28]
    np.testing.assert_array_equal(
        layout.position_ids[aligned], layout.position_ids[layout.target_video_indices]
    )
    assert layout.observation_video_indices[-1] not in layout.camera_video_indices


def test_segment_plan_rejects_gaps():
    with pytest.raises(ValueError, match="tile"):
        H3SegmentPlan(((1, 2, ((0, 2),), (), None),), 2)
