from __future__ import annotations

import pytest
import torch

from solar.renderer.silence import extend_silence


@pytest.mark.parametrize("length", [255, 263, 283])
def test_silence_shorter_clips_keep_the_original_encoded_prefix(length):
    silence = torch.arange(2 * 32 * 283).reshape(2, 32, 283)
    result = extend_silence(silence, length)
    torch.testing.assert_close(result, silence[..., :length])
    if length == 283:
        assert result is silence


@pytest.mark.parametrize("edge", [20, 7])
def test_long_silence_preserves_both_transients_and_repeats_the_steady_middle(edge):
    silence = torch.arange(2 * 32 * 283).reshape(2, 32, 283).to(torch.bfloat16)
    original = silence.clone()
    result = extend_silence(silence, 401, edge=edge)
    assert result.shape == (2, 32, 401)
    assert result.dtype == silence.dtype and result.device == silence.device
    assert result.is_contiguous()
    torch.testing.assert_close(result[..., :edge], original[..., :edge])
    torch.testing.assert_close(result[..., -edge:], original[..., -edge:])
    torch.testing.assert_close(
        result[..., edge:-edge], original[..., 141:142].expand(2, 32, 401 - 2 * edge)
    )
    torch.testing.assert_close(silence, original)
