from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from solar.errors import DataContractError
from solar.preencode import (
    EncodedPayload,
    EncoderContract,
    TensorSpec,
    validate_encoded_tensors,
    write_index,
    write_shard,
)


def _contract() -> EncoderContract:
    return EncoderContract(
        schema="solarwm.encoder.v1",
        family="wan22_ti2v_5b",
        format_version="wan-latent-v1",
        pixel_frames=81,
        latent_frames=21,
        height=480,
        width=864,
        camera_convention="first-frame-relative-w2c",
        tensors=(
            TensorSpec("latents", (16, 21, 60, 108), "float32"),
            TensorSpec("prompt_embeds", (None, 4096), "float32"),
        ),
    )


def _payload(sample_id: str) -> EncodedPayload:
    return EncodedPayload(
        sample_id=sample_id,
        key=f"key-{sample_id}",
        source_sample_id=f"source-{sample_id}",
        start_frame=3,
        source_frame_indices=tuple(range(3, 84)),
        encoder_contract_digest=_contract().digest,
        members={"tensors.safetensors": f"payload-{sample_id}".encode()},
        metadata={"split": "train"},
    )


def test_tensor_contract_validates_shape_dtype_and_fields() -> None:
    tensors = {
        "latents": np.zeros((16, 21, 60, 108), dtype=np.float32),
        "prompt_embeds": np.zeros((512, 4096), dtype=np.float32),
    }
    validate_encoded_tensors(tensors, _contract())
    tensors["latents"] = np.zeros((16, 20, 60, 108), dtype=np.float32)
    with pytest.raises(DataContractError, match="shape"):
        validate_encoded_tensors(tensors, _contract())


@pytest.mark.parametrize("bad", (np.nan, np.inf, -np.inf))
def test_tensor_contract_rejects_non_finite_values(bad: float) -> None:
    tensors = {
        "latents": np.zeros((16, 21, 60, 108), dtype=np.float32),
        "prompt_embeds": np.zeros((512, 4096), dtype=np.float32),
    }
    tensors["prompt_embeds"][0, 0] = bad
    with pytest.raises(DataContractError, match="non-finite"):
        validate_encoded_tensors(tensors, _contract())


def test_deterministic_tar_and_relative_index(tmp_path: Path) -> None:
    left = write_shard(tmp_path / "left", "shards/part-000.tar", [_payload("a")])
    right = write_shard(tmp_path / "right", "shards/part-000.tar", [_payload("a")])
    assert left.digest == right.digest
    assert left.rows == right.rows
    with tarfile.open(tmp_path / "left/shards/part-000.tar", "r:") as archive:
        provenance = json.load(archive.extractfile(left.rows[0]["provenance_member"]))
    assert provenance["sample_id"] == "a"

    index_digest = write_index(tmp_path / "index/train.jsonl", left.rows)
    row = json.loads((tmp_path / "index/train.jsonl").read_text())
    assert row["shard"] == "shards/part-000.tar"
    assert len(index_digest) == 64


def test_index_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    receipt = write_shard(tmp_path / "payload", "shards/part-000.tar", [_payload("a")])
    with pytest.raises(DataContractError, match="duplicate"):
        write_index(tmp_path / "index.jsonl", [*receipt.rows, *receipt.rows])
