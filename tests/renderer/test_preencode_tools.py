from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from solarwm.renderer import preencode
from solarwm.renderer.convert_dmad import convert
from solarwm.renderer.hf_samples import TRAIN, clips
from solarwm.renderer.render import lora_pairs


def test_preencode_commits_only_completed_samples_and_resumes(tmp_path, monkeypatch):
    samples = tmp_path / "samples.jsonl"
    samples.write_text('{"id":"source/a"}\n{"id":"source/b"}\n')
    output = tmp_path / "encoded"
    monkeypatch.setattr(sys, "argv", ["preencode", "--samples", str(samples), "--out", str(output)])
    monkeypatch.setattr(preencode, "SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    loads = []
    monkeypatch.setattr(preencode, "build_pipelines", lambda device: loads.append(device))

    def encode(row, _pipelines, _frames):
        if row["id"] == "source/b":
            raise ValueError("bad media")
        return {"latent": torch.tensor([1.0])}, {
            "latent_frames": 7,
            "canvas": [768, 1344],
            "source": "source",
            "observations": [],
        }

    monkeypatch.setattr(preencode, "encode", encode)
    with pytest.raises(SystemExit, match="1 samples failed"):
        preencode.main()
    index = output / "index.00000.jsonl"
    assert [json.loads(line)["id"] for line in index.read_text().splitlines()] == ["source/a"]
    assert load_file(str(output / "samples/source/a.safetensors"))["latent"].item() == 1
    assert not (output / "samples/source/b.safetensors").exists()

    called = []

    def repaired(row, pipelines, frames):
        called.append(row["id"])
        return encode({"id": "source/a"}, pipelines, frames)

    monkeypatch.setattr(preencode, "encode", repaired)
    preencode.main()
    assert called == ["source/b"]
    assert len(index.read_text().splitlines()) == 2
    preencode.main()
    assert len(loads) == 2
    assert not list(output.rglob(".encoding-*"))


def test_remote_sample_list_cache_includes_full_location(tmp_path, monkeypatch):
    monkeypatch.setenv("OMNI_LIST_CACHE", str(tmp_path))
    monkeypatch.setattr(preencode.storage, "stat", lambda _path: (12, 1.0))
    monkeypatch.setattr(preencode, "copy", lambda src, dst: Path(dst).write_text(src))
    first = preencode.fetch_fresh("memory://one/samples.jsonl", "")
    second = preencode.fetch_fresh("memory://two/samples.jsonl", "")
    assert first != second
    assert Path(first).read_text() == "memory://one/samples.jsonl"
    assert Path(second).read_text() == "memory://two/samples.jsonl"


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../b", "a\\b", "a//b", ""])
def test_sample_names_cannot_escape_the_output_directory(name):
    with pytest.raises(ValueError):
        preencode.sample_name(name)


def test_dmad_conversion_preserves_tensor_values_and_rejects_conflicting_alpha(tmp_path):
    path = tmp_path / "adapter.safetensors"
    state = {
        "diffusion_model.layer.lora.down.weight": torch.ones(2, 4),
        "diffusion_model.layer.lora.up.weight": torch.ones(3, 2),
        "diffusion_model.layer.alpha": torch.tensor(2.0),
    }
    save_file(state, str(path), metadata={"alpha": "2"})
    converted, alpha = convert(str(path))
    assert alpha == 2.0
    torch.testing.assert_close(converted["layer.lora_A.weight"], state[next(iter(state))])
    save_file(state, str(path), metadata={"alpha": "4"})
    with pytest.raises(ValueError, match="inconsistent alpha"):
        convert(str(path))


def test_dataset_tar_cannot_write_outside_output(tmp_path):
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as archive:
        member = tarfile.TarInfo("../escape.render.mp4")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        list(clips(str(path), TRAIN))


def test_export_creates_parent_directory_and_preserves_adapter_metadata(tmp_path, monkeypatch):
    from solarwm.renderer.export_lora import main

    state = {
        "base_model.model.layer.lora_A.weight": torch.ones(2, 4),
        "base_model.model.layer.lora_B.weight": torch.ones(3, 2),
    }
    torch.save({"metadata": {"rank": 2, "alpha": 4}, "state": state}, tmp_path / "adapter.pt")
    out = tmp_path / "weights/export.safetensors"
    monkeypatch.setattr(
        sys, "argv", ["export_lora", "--checkpoint", str(tmp_path), "--out", str(out)]
    )
    main()
    exported, rank, alpha = lora_pairs(out)
    assert (rank, alpha) == (2, 4.0)
    assert set(exported) == {"layer.lora_A.weight", "layer.lora_B.weight"}


def test_prepare_existing_h3_localizes_components_without_extra_model_layout(tmp_path, monkeypatch):
    from solarwm.renderer.prepare_h3 import main

    path = tmp_path / "modular_model_index.json"
    path.write_text(
        json.dumps(
            {
                "transformer": [
                    "diffusers",
                    "MiniMaxH3Transformer3DModel",
                    {
                        "pretrained_model_name_or_path": "MiniMaxAI/MiniMax-H3",
                        "subfolder": "transformer_ref",
                    },
                ]
            }
        )
    )
    monkeypatch.setattr(sys, "argv", ["prepare_h3", "--dir", str(tmp_path), "--skip-download"])
    main()
    metadata = json.loads(path.read_text())["transformer"][2]
    assert metadata == {
        "pretrained_model_name_or_path": str(tmp_path),
        "subfolder": "transformer_ref",
    }
    assert not (tmp_path / "ref2va").exists()
