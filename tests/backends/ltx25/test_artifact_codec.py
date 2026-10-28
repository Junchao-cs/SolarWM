from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from solar.backends.ltx25.artifact import (
    PREENCODE_VERSION,
    TENSOR_SPECS,
    PreencodedSample,
    TensorArtifact,
    artifact_contract,
)
from solar.backends.ltx25.codec import (
    ONLINE_BEHAVIOR_STATUS,
    RawSample,
    encode_online,
    online_behavior_contract,
)
from solar.errors import BackendContractError


def _bf16(name: str, *, fill: int = 0) -> TensorArtifact:
    spec = TENSOR_SPECS[name]
    return TensorArtifact(spec.dtype, spec.shape, bytes([fill]) * spec.nbytes)


def _valid_tensors(*, focal: float = 1.0) -> dict[str, TensorArtifact]:
    video = _bf16("video_latent")
    channels, frames, height, width = video.shape
    plane_bytes = height * width * 2
    channel_bytes = frames * plane_bytes
    first_bytes = b"".join(
        video.data[channel * channel_bytes : channel * channel_bytes + plane_bytes]
        for channel in range(channels)
    )
    poses = np.broadcast_to(np.eye(4), (20, 4, 4)).copy()
    K = np.broadcast_to(
        np.asarray([[focal, 0.0, 0.5], [0.0, focal, 0.5], [0.0, 0.0, 1.0]]),
        (20, 3, 3),
    ).copy()
    source = np.arange(10, 163, dtype=np.int64)
    camera_indices = source[
        np.asarray(
            (0, 1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 105, 113, 121, 129, 137, 145)
        )
    ]
    mask = np.concatenate((np.zeros(100, dtype=np.int64), np.ones(924, dtype=np.int64)))
    return {
        "video_latent": video,
        "first_frame_latent": TensorArtifact("BF16", (128, 1, 16, 24), first_bytes),
        "video_prompt_embeds": _bf16("video_prompt_embeds"),
        "prompt_attention_mask": TensorArtifact.from_numpy(mask, dtype="I64"),
        "relative_w2c": TensorArtifact.from_numpy(poses, dtype="F32"),
        "camera_K": TensorArtifact.from_numpy(K, dtype="F32"),
        "source_indices": TensorArtifact.from_numpy(source, dtype="I64"),
        "camera_source_indices": TensorArtifact.from_numpy(camera_indices, dtype="I64"),
    }


def _valid_sample(*, focal: float = 1.0) -> PreencodedSample:
    return PreencodedSample(
        sample_id="sample-10",
        key="sample-key-10",
        start_frame=10,
        source_fps=29.97,
        tensors=_valid_tensors(focal=focal),
    )


def test_artifact_schema_preserves_tensor_contract() -> None:
    contract = artifact_contract()
    assert contract["version"] == PREENCODE_VERSION
    assert contract["tensors"]["video_latent"] == {
        "dtype": "BF16",
        "shape": [128, 20, 16, 24],
    }
    assert set(contract["tensors"]) == set(TENSOR_SPECS)
    _valid_sample()


def test_optical_zoom_above_four_is_valid_in_artifact() -> None:
    sample = _valid_sample(focal=11.0)
    assert sample.tensors["camera_K"].numpy()[0, 0, 0] == pytest.approx(11.0)


def test_first_frame_must_be_bit_equal() -> None:
    tensors = _valid_tensors()
    first = tensors["first_frame_latent"]
    tensors["first_frame_latent"] = TensorArtifact(
        first.dtype,
        first.shape,
        b"\x01" + first.data[1:],
    )
    with pytest.raises(BackendContractError, match="bit-equal"):
        PreencodedSample("sample", "key", 10, 24.0, tensors)


def test_caption_mask_must_be_left_padded() -> None:
    tensors = _valid_tensors()
    mask = np.ones(1024, dtype=np.int64)
    mask[100] = 0
    tensors["prompt_attention_mask"] = TensorArtifact.from_numpy(mask, dtype="I64")
    with pytest.raises(BackendContractError, match="left padding"):
        PreencodedSample("sample", "key", 10, 24.0, tensors)


def test_tensor_shape_and_key_drift_fail_closed() -> None:
    with pytest.raises(BackendContractError, match="byte length"):
        TensorArtifact("F32", (2, 2), b"short")
    tensors = _valid_tensors()
    tensors.pop("camera_K")
    with pytest.raises(BackendContractError, match="tensor keys"):
        PreencodedSample("sample", "key", 10, 24.0, tensors)


def test_bfloat16_nan_or_inf_bits_fail_closed() -> None:
    tensors = _valid_tensors()
    prompt = tensors["video_prompt_embeds"]
    tensors["video_prompt_embeds"] = TensorArtifact(
        prompt.dtype,
        prompt.shape,
        b"\x80\x7f" + prompt.data[2:],
    )
    with pytest.raises(BackendContractError, match="NaN or Inf"):
        PreencodedSample("sample", "key", 10, 24.0, tensors)


class _FakeCodec:
    identity = "official-ltx-assets@test"

    def __init__(self) -> None:
        tensors = _valid_tensors()
        self.video = tensors["video_latent"]
        self.first = tensors["first_frame_latent"]
        self.prompt = tensors["video_prompt_embeds"]
        self.mask = tensors["prompt_attention_mask"]

    def encode_video(self, frames: object) -> tuple[TensorArtifact, TensorArtifact]:
        assert len(frames) == 153  # type: ignore[arg-type]
        return self.video, self.first

    def encode_prompt(self, caption: str) -> tuple[TensorArtifact, TensorArtifact]:
        assert caption == "caption"
        return self.prompt, self.mask


def test_online_codec_reuses_exact_artifact_contract_but_is_unpaired() -> None:
    poses = np.broadcast_to(np.eye(4), (153, 4, 4)).copy()
    intrinsics = np.broadcast_to(
        np.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]),
        (153, 3, 3),
    ).copy()
    raw = RawSample(
        sample_id="sample-10",
        key="sample-key-10",
        frames=[object()] * 153,
        caption="caption",
        source_indices=np.arange(10, 163, dtype=np.int64),
        camera_poses=poses,
        camera_intrinsics=intrinsics,
        camera_convention="absolute_c2w",
        source_height=512,
        source_width=768,
        source_fps=24.0,
    )
    encoded = encode_online(raw, _FakeCodec())
    assert encoded.sample_id == raw.sample_id
    assert encoded.tensors["camera_source_indices"].numpy()[1] == 11
    assert online_behavior_contract()["status"] == ONLINE_BEHAVIOR_STATUS


def test_online_codec_requires_nonempty_identity() -> None:
    codec = _FakeCodec()
    codec.identity = ""
    poses = np.broadcast_to(np.eye(4), (153, 4, 4)).copy()
    raw = RawSample(
        sample_id="sample",
        key="key",
        frames=[object()] * 153,
        caption="caption",
        source_indices=np.arange(153, dtype=np.int64),
        camera_poses=poses,
        camera_intrinsics=np.eye(3),
        camera_convention="absolute_c2w",
        source_height=512,
        source_width=768,
        source_fps=24.0,
    )
    with pytest.raises(BackendContractError, match="identity"):
        encode_online(raw, codec)


def test_frozen_sample_cannot_change_declared_version() -> None:
    sample = _valid_sample()
    with pytest.raises(BackendContractError, match="version"):
        replace(sample, version="ltx25.v2")
