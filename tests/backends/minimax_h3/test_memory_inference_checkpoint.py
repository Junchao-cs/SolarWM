import copy
import hashlib
import json

import pytest

from solar.backends.minimax_h3.memory_contract import MEMORY_SCHEMA
from solar.backends.minimax_h3.memory_inference import (
    CAUSAL_MEMORY_MASK,
    FIX3900_PARENT_SHA256,
    STAGE1_PARENT_SHA256,
    _selected_component,
    validate_memory_checkpoint,
)
from solar.errors import BackendContractError


def mixed_stage1_manifest():
    return {
        "step": 1050,
        "contract": {
            "stage": "stage1-memory",
            "family": "minimax_h3",
            "objective": "anyflow_forward_map",
            "objective_variant": "v1_5",
            "causal_mode": "teacher_forcing",
            "parameterization": "peft-lora-r384-alpha384",
            "camera_translation_transform": "logd4",
            "extras": {
                "memory_schema": MEMORY_SCHEMA,
                "memory_frames_per_chunk": 5,
                "bank_frames": 45,
                "memory_temporal_rope": -1,
                "chunk_memory_mask": CAUSAL_MEMORY_MASK,
                "dual_lora": {
                    "composition": "additive",
                    "frozen": "official-stage1-ema-step3000",
                    "trainable": "memory",
                },
                "base_model": {"frozen_adapter": {"sha256": STAGE1_PARENT_SHA256}},
                "training_profile": {
                    "data_mix": {
                        "enabled": True,
                        "weights": {"memory": 1, "original": 1},
                    }
                },
            },
        },
    }


def test_mixed_stage1_weights_transfer_without_changing_training_parent():
    assert validate_memory_checkpoint(mixed_stage1_manifest()) == (
        "official-stage1-ema-step3000",
        STAGE1_PARENT_SHA256,
        "explicit-inference-parent-transfer",
    )


@pytest.mark.parametrize("change", ["stage", "parent", "bank", "schema", "mask"])
def test_reject_incompatible_memory_checkpoint(change):
    manifest = copy.deepcopy(mixed_stage1_manifest())
    contract = manifest["contract"]
    extras = contract["extras"]
    if change == "stage":
        contract["stage"] = "stage2"
    elif change == "parent":
        extras["base_model"]["frozen_adapter"]["sha256"] = FIX3900_PARENT_SHA256
    elif change == "bank":
        extras["memory_frames_per_chunk"] = 2
    elif change == "schema":
        extras["memory_schema"] = "unknown"
    else:
        extras["chunk_memory_mask"] = "unrestricted"
    with pytest.raises(BackendContractError):
        validate_memory_checkpoint(manifest)


def test_inference_only_checkpoint_requires_matching_step_and_live_file(tmp_path):
    step = 1050
    adapter = tmp_path / "adapter.pt"
    adapter.write_bytes(b"component bytes; tensors are checked when attached")
    manifest = dict(
        mixed_stage1_manifest(),
        schema="solarwm.checkpoint.v2",
        files=[{"path": "adapter.pt", "size": adapter.stat().st_size}],
    )
    payload = json.dumps(manifest).encode()
    (tmp_path / "checkpoint-manifest.json").write_bytes(payload)
    marker = {
        "schema": "solarwm.checkpoint-complete.v2",
        "step": step,
        "manifest_digest": hashlib.blake2s(payload).hexdigest(),
    }
    (tmp_path / "COMPLETE.json").write_text(json.dumps(marker))
    assert _selected_component(tmp_path, step)[0] == adapter
    assert _selected_component(adapter, step)[1] == manifest
    with pytest.raises(BackendContractError, match="step differs"):
        _selected_component(tmp_path, step + 1)
    (tmp_path / "ema.pt").write_bytes(b"not LIVE")
    with pytest.raises(BackendContractError, match="LIVE Memory"):
        _selected_component(tmp_path / "ema.pt", step)
    marker["manifest_digest"] = "invalid"
    (tmp_path / "COMPLETE.json").write_text(json.dumps(marker))
    with pytest.raises(BackendContractError, match="completion marker"):
        _selected_component(tmp_path, step)
