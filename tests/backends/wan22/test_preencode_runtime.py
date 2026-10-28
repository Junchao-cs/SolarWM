from __future__ import annotations

import gzip
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

import solar.backends.wan22.runtime.preencode as preencode_runtime
from solar.backends.wan22.runtime.preencode import (
    _split_for_sample,
    run_wan_preencode,
)
from solar.backends.wan22.runtime.readiness import probe_runtime
from solar.config import load_config
from solar.data.index import read_index
from solar.errors import BackendContractError
from solar.preencode import EncodedPayload, EncoderContract, TensorSpec

ROOT = Path(__file__).resolve().parents[3]


def _member(archive: tarfile.TarFile, name: str, value: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(value)
    info.mtime = 0
    archive.addfile(info, io.BytesIO(value))


def _raw_tree(root: Path, *, epoch_repeats: int = 1) -> None:
    shard = root / "raw/source.tar"
    shard.parent.mkdir(parents=True)
    rows = []
    with tarfile.open(shard, "w:") as archive:
        for ordinal, split in enumerate(("train", "test")):
            prefix = f"sample-{ordinal}"
            manifest = {
                "video": {"num_frames": 153, "fps": 16.0},
                "prompt": {"text": f"prompt {ordinal}"},
                "metadata": {"scene": "dl3dv-10s"},
            }
            _member(archive, f"{prefix}.mp4", b"not-decoded-by-fake")
            _member(archive, f"{prefix}.npz", b"camera-sidecar")
            _member(
                archive,
                f"{prefix}.json",
                json.dumps(manifest, sort_keys=True).encode(),
            )
            rows.append(
                {
                    "sample_id": prefix,
                    "key": prefix,
                    "shard": "raw/source.tar",
                    "epoch_repeats": epoch_repeats,
                    "video_member": f"{prefix}.mp4",
                    "camera_member": f"{prefix}.npz",
                    "manifest_member": f"{prefix}.json",
                    "num_frames": 153,
                    "fps": 16.0,
                    "start_frame": 0,
                    "split": split,
                    "kept_tier": "high" if split == "train" else "xhigh",
                    "dataset": "dl3dv-10s",
                }
            )
    index = root / "indexes/window-plan.jsonl"
    index.parent.mkdir(parents=True)
    index.write_text("".join(json.dumps(row) + "\n" for row in rows))


class FakeProvider:
    def __init__(
        self,
        *,
        family: str = "wan22_ti2v_5b",
        fail_sample: str = "",
    ) -> None:
        self.family = family
        self.fail_sample = fail_sample
        self.encoded_sample_ids: list[str] = []
        is_5b = family == "wan22_ti2v_5b"
        sequence = 405 if is_5b else 1560
        tensors = [
            TensorSpec(
                "latents",
                (39, 48, 30, 54) if is_5b else (39, 16, 60, 104),
                "bfloat16",
            ),
            TensorSpec("prompt_embeds", (512, 4096), "bfloat16"),
        ]
        if not is_5b:
            tensors.append(TensorSpec("i2v_y", (39, 20, 60, 104), "bfloat16"))
        tensors.extend(
            (
                TensorSpec("camera_viewmats", (39 * sequence, 4, 4), "float32"),
                TensorSpec("camera_K", (39 * sequence, 3, 3), "float32"),
            )
        )
        self.contract = EncoderContract(
            schema="solarwm.encoder.v1",
            family=self.family,
            format_version=(
                "solarwm.wan22-ti2v-5b.online.v1" if is_5b else "solarwm.wan22-i2v-a14b.online.v1"
            ),
            pixel_frames=153,
            latent_frames=39,
            height=480,
            width=864 if is_5b else 832,
            camera_convention="first-frame-relative-w2c-fp32",
            tensors=tuple(tensors),
        )

    def encode(self, sample, config) -> EncodedPayload:
        del config
        self.encoded_sample_ids.append(sample.plan.sample_id)
        if sample.plan.sample_id == self.fail_sample:
            raise BackendContractError("fake codec failure")
        split = str(sample.index_values["split"])
        return EncodedPayload(
            sample_id=sample.plan.sample_id,
            key=sample.plan.key,
            source_sample_id=sample.plan.sample_id,
            start_frame=sample.plan.start_frame,
            source_frame_indices=sample.plan.source_frame_indices,
            encoder_contract_digest=self.contract.digest,
            members={
                "preencoded.safetensors": f"tensor:{split}".encode(),
                "camera.npz": sample.members["camera_member"],
                "manifest.json": b"{}\n",
            },
            metadata={
                "split": split,
                "kept_tier": sample.index_values["kept_tier"],
                "dataset": sample.index_values["dataset"],
                "caption": sample.caption,
                "num_frames": sample.index_values["num_frames"],
                "fps": sample.index_values["fps"],
            },
        )


def _config(
    tmp_path: Path,
    *,
    family: str = "wan22_ti2v_5b",
    epoch_repeats: int = 1,
) -> dict:
    relative = (
        "configs/solar_wm/wan22_ti2v_5b/preencode_153f.yaml"
        if family == "wan22_ti2v_5b"
        else "configs/solar_wm/wan22_i2v_a14b/preencode_153f.yaml"
    )
    config = load_config(ROOT / relative).mutable_copy()
    raw = tmp_path / "raw-root"
    _raw_tree(raw, epoch_repeats=epoch_repeats)
    config["data"]["transport"] = {"kind": "local", "root": str(raw)}
    config["data"].pop("index_root", None)
    config["data"]["index"] = "indexes/window-plan.jsonl"
    config["preencode"]["output_root"] = str(tmp_path / "physical")
    config["preencode"]["logical_output_root"] = str(tmp_path / "logical")
    config["preencode"]["generation_id"] = "test-generation-v1"
    config["preencode"]["shard_max_samples"] = 1
    return config


def _gzip_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_preencode_commits_physical_then_logical_controls(tmp_path: Path) -> None:
    config = _config(tmp_path)
    summary = run_wan_preencode(config, provider=FakeProvider())
    assert summary.samples == 2
    assert summary.shards == 2
    assert summary.train_samples == summary.test_samples == 1
    physical = json.loads((summary.physical_root / "COMPLETE.json").read_text())
    logical = json.loads((summary.logical_root / "COMPLETE.json").read_text())
    assert physical["samples"] == 2
    assert logical["physical_complete_digest"] == summary.physical_complete_digest
    recipe_bytes = (summary.logical_root / "recipe.json").read_bytes()
    recipe = json.loads(recipe_bytes)
    assert "physical_root" not in recipe
    assert str(tmp_path).encode() not in recipe_bytes

    train = _gzip_rows(summary.logical_root / "train-index.jsonl.gz")
    test = _gzip_rows(summary.logical_root / "test-index.jsonl.gz")
    assert len(read_index(summary.logical_root / "train-index.jsonl.gz")) == 1
    assert len(read_index(summary.logical_root / "test-index.jsonl.gz")) == 1
    assert {train[0]["split"], test[0]["split"]} == {"train", "test"}
    for row in (*train, *test):
        assert row["preencoded_member"].endswith("preencoded.safetensors")
        assert row["camera_member"].endswith("camera.npz")
        assert row["manifest_member"].endswith("manifest.json")
        assert row["shard_generation"].startswith("local-digest:")
    assert "/kept-high-" in f"/{train[0]['shard']}"
    assert "/kept-xhigh-" in f"/{test[0]['shard']}"


def test_preencode_encodes_physical_rows_once_and_preserves_repeats(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, epoch_repeats=3)
    provider = FakeProvider()
    summary = run_wan_preencode(config, provider=provider)

    assert provider.encoded_sample_ids == ["sample-0", "sample-1"]
    physical = _gzip_rows(summary.physical_root / "control/physical-index.jsonl.gz")
    train = _gzip_rows(summary.logical_root / "train-index.jsonl.gz")
    test = _gzip_rows(summary.logical_root / "test-index.jsonl.gz")
    assert [row["epoch_repeats"] for row in physical] == [3, 3]
    assert [row["epoch_repeats"] for row in (*train, *test)] == [3, 3]


def test_a14b_preencode_uses_its_distinct_tensor_contract(tmp_path: Path) -> None:
    family = "wan22_i2v_a14b"
    config = _config(tmp_path, family=family)
    summary = run_wan_preencode(config, provider=FakeProvider(family=family))
    contract = json.loads((summary.physical_root / "control/encoder-contract.json").read_text())
    specs = {item["name"]: item["shape"] for item in contract["tensors"]}
    assert specs["latents"] == [39, 16, 60, 104]
    assert specs["i2v_y"] == [39, 20, 60, 104]


def test_codec_failure_never_writes_complete_marker(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(BackendContractError, match="fake codec failure"):
        run_wan_preencode(config, provider=FakeProvider(fail_sample="sample-1"))
    assert not Path(config["preencode"]["output_root"]).exists()
    partials = list(tmp_path.glob(".physical.*.partial"))
    assert len(partials) == 1
    assert not (partials[0] / "COMPLETE.json").exists()
    assert not Path(config["preencode"]["logical_output_root"]).exists()


def test_existing_physical_target_is_never_overwritten(tmp_path: Path) -> None:
    config = _config(tmp_path)
    physical = Path(config["preencode"]["output_root"])
    physical.mkdir()
    sentinel = physical / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(BackendContractError, match="already exists"):
        run_wan_preencode(config, provider=FakeProvider())
    assert sentinel.read_text() == "keep"


def test_physical_commit_race_never_overwrites_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    physical = Path(config["preencode"]["output_root"])
    original = preencode_runtime._assemble_physical

    def race(*args, **kwargs):
        result = original(*args, **kwargs)
        physical.mkdir()
        (physical / "winner.txt").write_text("competitor", encoding="utf-8")
        return result

    monkeypatch.setattr(preencode_runtime, "_assemble_physical", race)
    with pytest.raises(BackendContractError, match="target appeared"):
        run_wan_preencode(config, provider=FakeProvider())
    assert (physical / "winner.txt").read_text(encoding="utf-8") == "competitor"
    assert not Path(config["preencode"]["logical_output_root"]).exists()


@pytest.mark.parametrize(
    ("physical", "logical"),
    (("root", "root/logical"), ("root/physical", "root")),
)
def test_preencode_rejects_nested_transaction_roots(
    tmp_path: Path, physical: str, logical: str
) -> None:
    config = _config(tmp_path)
    config["preencode"]["output_root"] = str(tmp_path / physical)
    config["preencode"]["logical_output_root"] = str(tmp_path / logical)
    with pytest.raises(BackendContractError, match="output_roots_nonoverlapping"):
        run_wan_preencode(config, provider=FakeProvider())
    assert not (tmp_path / physical / "COMPLETE.json").exists()


def test_canonical_recipe_role_is_valid_as_the_split() -> None:
    sample = SimpleNamespace(
        index_values={"recipe_role": "train"},
        manifest={},
    )
    assert _split_for_sample(sample) == "train"


def test_preencode_readiness_does_not_require_transformer_or_flash_attention(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "text.pt").write_bytes(b"fixture")
    (assets / "vae.pt").write_bytes(b"fixture")
    (assets / "tokenizer").mkdir()
    config["model"]["base_path"] = str(assets)
    config["model"]["assets"] = {
        "transformer_config": "builtin",
        "transformer_weights": "missing-transformer",
        "text_encoder": "text.pt",
        "tokenizer": "tokenizer",
        "vae": "vae.pt",
    }
    report = probe_runtime(config, family="wan22_ti2v_5b", require_cuda=False)
    assert report.ready
    assert "diffusers" not in report.dependencies
    assert "flash_attention" not in report.dependencies
    assert report.weight_inventory == {}
