from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from solar.errors import DataContractError
from solar.renderer.data import H3OmniReader
from solar.renderer.render import lora_pairs


def sample(path, *, frames=(0, 1), degree=0.999, bad_observation=False):
    observation = torch.zeros(24, 2, 48, 84)
    if bad_observation:
        observation[0, 0, 0, 0] = float("nan")
    save_file(
        {
            "target_latents": torch.zeros(24, 2, 48, 84),
            "prompt_embeds": torch.zeros(2, 5120),
            "text_token_tags": torch.ones(2, dtype=torch.long),
            "observation_0_latents": observation,
        },
        str(path),
        metadata={"omni": json.dumps({"observations": [{"frames": frames, "degree": degree}]})},
    )


def test_standalone_reader_preserves_observation_alignment(tmp_path):
    path = tmp_path / "sample.safetensors"
    sample(path)
    batch = H3OmniReader(str(tmp_path)).read({"member": path.name, "id": "clip"})
    assert batch.target_latents.shape == (24, 2, 48, 84)
    assert batch.visual_observations[0].frames == (0, 1)
    assert batch.anchor_latents is None
    assert batch.text_prompt_embeds is batch.prompt_embeds


@pytest.mark.parametrize(
    "options", [{"frames": [0, 2]}, {"degree": float("nan")}, {"bad_observation": True}]
)
def test_invalid_observations_fail_before_model_execution(tmp_path, options):
    path = tmp_path / "sample.safetensors"
    sample(path, **options)
    with pytest.raises(DataContractError):
        H3OmniReader(str(tmp_path)).read({"member": path.name, "id": "clip"})


def test_lora_metadata_must_match_tensors(tmp_path):
    path = tmp_path / "adapter.safetensors"
    tensors = {"layer.lora_A.weight": torch.ones(2, 4), "layer.lora_B.weight": torch.ones(3, 2)}
    save_file(tensors, str(path), metadata={"rank": "3"})
    with pytest.raises(ValueError, match="rank metadata"):
        lora_pairs(path)
    save_file(tensors, str(path), metadata={"rank": "2", "alpha": "4"})
    _, rank, alpha = lora_pairs(path)
    assert (rank, alpha) == (2, 4.0)
