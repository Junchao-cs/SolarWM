from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from solar.backends.minimax_h3.codec import (
    H3EncodedPayload,
    H3PreencodedSample,
    H3RawSample,
    assemble_model_inputs,
    encode_raw_sample,
    validate_encoded_payload,
)


@dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


def _raw_sample() -> H3RawSample:
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 158, axis=0)
    c2w[:, 0, 3] = np.arange(158, dtype=np.float32) / 10
    K = np.repeat(np.eye(3, dtype=np.float32)[None], 158, axis=0)
    K[:, 0, 2] = 0.5
    K[:, 1, 2] = 0.5
    return H3RawSample(
        sample_id="sample-0001",
        frames=tuple(range(158)),
        caption="drive forward",
        source_frame_indices=np.arange(100, 258, dtype=np.int64),
        camera_c2w=c2w,
        camera_intrinsics=K,
        source_fps=29.97,
    )


class FakeOfficialCodec:
    identity = "test-codec@digest:1234"

    def encode_target_video(self, sample: H3RawSample) -> object:
        return TensorSpec((24, 47, 48, 84), "bfloat16")

    def encode_visual_anchor(self, sample: H3RawSample) -> object:
        return TensorSpec((24, 1, 48, 84), "bfloat16")

    def encode_joint_prompt(self, sample: H3RawSample) -> tuple[object, object]:
        return TensorSpec((3, 5120), "bfloat16"), np.asarray([0, 1, 1], np.int64)

    def encode_silence_audio(self, sample: H3RawSample) -> object:
        return TensorSpec((2, 32, 263), "bfloat16")


def test_injected_codec_builds_explicit_model_inputs() -> None:
    inputs = encode_raw_sample(_raw_sample(), FakeOfficialCodec())
    assert inputs.preencode_version == "h3.158f.v1"
    assert inputs.codec_identity.startswith("test-codec")
    assert inputs.layout.rows_per_video_frame == 1008
    assert inputs.camera_viewmats.shape == (48 * 1008, 4, 4)
    assert inputs.camera_intrinsics.shape == (48 * 1008, 3, 3)
    np.testing.assert_array_equal(inputs.source_frame_indices, np.arange(100, 258, dtype=np.int64))


def test_preencoded_sample_and_shared_silence_assemble_separately() -> None:
    raw = _raw_sample()
    sample = H3PreencodedSample(
        sample_id=raw.sample_id,
        encoder_identity="test-codec@digest:abcd",
        target_latents=TensorSpec((24, 47, 48, 84), "bfloat16"),
        anchor_latent=TensorSpec((24, 1, 48, 84), "bfloat16"),
        prompt_embeddings=TensorSpec((2, 5120), "bfloat16"),
        text_token_tags=np.asarray([0, 1], dtype=np.int64),
        source_frame_indices=raw.source_frame_indices,
        camera_c2w=raw.camera_c2w,
        camera_intrinsics=raw.camera_intrinsics,
    )
    inputs = assemble_model_inputs(sample, TensorSpec((2, 32, 263), "bfloat16"))
    assert inputs.sample_id == raw.sample_id
    assert inputs.audio_latents.shape == (2, 32, 263)


def test_codec_payload_requires_bfloat16_and_official_silence_shape() -> None:
    payload = H3EncodedPayload(
        target_latents=TensorSpec((24, 47, 48, 84), "float16"),
        anchor_latent=TensorSpec((24, 1, 48, 84), "bfloat16"),
        prompt_embeddings=TensorSpec((2, 5120), "bfloat16"),
        text_token_tags=np.asarray([1, 1], dtype=np.int64),
        audio_latents=TensorSpec((2, 32, 263), "bfloat16"),
    )
    with pytest.raises(TypeError, match="target_latents dtype"):
        validate_encoded_payload(payload)


def test_raw_sample_rejects_noncontiguous_source_frames() -> None:
    sample = _raw_sample()
    bad = H3RawSample(
        sample_id=sample.sample_id,
        frames=sample.frames,
        caption=sample.caption,
        source_frame_indices=np.arange(158, dtype=np.int64) * 2,
        camera_c2w=sample.camera_c2w,
        camera_intrinsics=sample.camera_intrinsics,
    )
    with pytest.raises(ValueError, match="contiguous"):
        encode_raw_sample(bad, FakeOfficialCodec())


def test_incomplete_codec_is_rejected_instead_of_stubbed() -> None:
    class IncompleteCodec:
        identity = "incomplete"

    with pytest.raises(TypeError, match="complete H3Codec"):
        encode_raw_sample(_raw_sample(), IncompleteCodec())  # type: ignore[arg-type]
