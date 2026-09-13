from __future__ import annotations

import gzip
import json
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from solarwm.cli import _run_backend, main
from solarwm.data.index import read_index
from solarwm.errors import SolarWMError

_WAN_TRAIN_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "examples"
    / "wan22_ti2v_5b"
    / "train_stage0p5_fm_81f.yaml"
)


def test_routes_command(capsys) -> None:
    assert main(["config", "routes"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "wan22_ti2v_5b:stage2:self_gradient_forcing:flow_matching" in payload
    assert payload == sorted(payload)


def test_data_inspect(tmp_path: Path, capsys) -> None:
    index = tmp_path / "index.jsonl.gz"
    with gzip.open(index, "wt", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "sample_id": "sample",
                    "key": "sample-key",
                    "shard": "dataset/shard.tar",
                    "epoch_repeats": 3,
                }
            )
            + "\n"
        )
    assert main(["data", "inspect", str(index)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rows"] == 1
    assert payload["virtual_occurrences"] == 3


def test_cache_warmup_is_an_explicit_optional_data_command(tmp_path, capsys, monkeypatch) -> None:
    import solarwm.data.transport as transport

    index = tmp_path / "index.jsonl"
    index.write_text(
        json.dumps(
            {
                "sample_id": "sample",
                "key": "key",
                "shard": "raw/shard.tar",
                "shard_size": 4,
            }
        )
        + "\n"
    )
    calls = []

    class Resolver:
        def __init__(self, **kwargs):
            self.max_bytes = kwargs["max_bytes"]

        def resolve(self, row):
            calls.append(row.shard)
            return tmp_path / "shard.tar"

    monkeypatch.setattr(transport, "GCSResolver", Resolver)
    assert (
        main(
            [
                "data",
                "warm-cache",
                str(index),
                "--root",
                "gs://bucket/release",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--max-gib",
                "4",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["shards"] == 1
    assert calls == ["raw/shard.tar"]


def test_data_materialize_wan153f_command(tmp_path: Path, capsys) -> None:
    train = tmp_path / "train.jsonl.gz"
    test = tmp_path / "test.jsonl.gz"
    train_windows = tmp_path / "train-windows.jsonl.gz"
    test_windows = tmp_path / "test-windows.jsonl.gz"
    common = {
        "key": "sample-key",
        "shard": "raw/source.tar",
        "dataset": "dl3dv-10s",
        "num_frames": 153,
        "video_member": "sample.video.mp4",
        "camera_member": "sample.camera.npz",
        "manifest_member": "sample.manifest.json",
    }
    for path, window_path, role in (
        (train, train_windows, "train"),
        (test, test_windows, "test"),
    ):
        row = {**common, "sample_id": f"dl3dv-10s/{role}", "recipe_role": role}
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        authority = {
            "sample_id": f"dl3dv-10s/{role}/latent-153f-w00",
            "key": "sample-key__latent153f_w00",
            "shard": "latent/authority.tar",
            "start_frame": 0,
            "source_frame_indices": list(range(153)),
        }
        with gzip.open(window_path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(authority) + "\n")
    output = tmp_path / "indexes" / "preencode-window-index.jsonl.gz"

    assert (
        main(
            [
                "data",
                "materialize-wan153f",
                "--train-index",
                str(train),
                "--test-index",
                str(test),
                "--train-window-index",
                str(train_windows),
                "--test-window-index",
                str(test_windows),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "solarwm.wan22-153f-window-index.v1"
    assert payload["train_windows"] == payload["test_windows"] == 1
    assert len(read_index(output)) == 2


def test_config_resolve_rejects_inline_secret_before_writing(tmp_path: Path, capsys) -> None:
    output = tmp_path / "resolved" / "config.json"
    secret = "do-not-persist-this-value"

    assert (
        main(
            [
                "config",
                "resolve",
                "--config",
                str(_WAN_TRAIN_CONFIG),
                "--set",
                f"metadata.access_token={secret}",
                "--output",
                str(output),
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert "inline credential at metadata.access_token is forbidden" in captured.err
    assert secret not in captured.err
    assert not output.exists()
    assert not output.parent.exists()


def test_train_rejects_inline_secret_before_creating_output(tmp_path: Path, capsys) -> None:
    output_dir = tmp_path / "run"
    secret = "do-not-persist-this-value"

    assert (
        main(
            [
                "train",
                "--config",
                str(_WAN_TRAIN_CONFIG),
                "--set",
                f"metadata.token={secret}",
                "--set",
                f"runtime.output_dir={output_dir}",
            ]
        )
        == 2
    )

    captured = capsys.readouterr()
    assert "inline credential at metadata.token is forbidden" in captured.err
    assert secret not in captured.err
    assert not output_dir.exists()


def test_backend_run_clears_result_from_an_earlier_invocation(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    stale_result = output_dir / "run-result.json"
    stale_result.write_text('{"status":"failed"}\n', encoding="utf-8")

    class Resolved:
        values: ClassVar[dict[str, object]] = {
            "action": "train",
            "runtime": {"output_dir": str(output_dir)},
        }
        path = tmp_path / "config.yaml"
        source_digest = "source"
        resolved_digest = "resolved"

        @staticmethod
        def write_json(path: Path) -> None:
            path.write_text("{}\n", encoding="utf-8")

    class Backend:
        @staticmethod
        def validate_config(config) -> None:
            del config

        @staticmethod
        def train(config) -> int:
            del config
            assert not stale_result.exists()
            return 0

    monkeypatch.setattr("solarwm.cli.load_config", lambda *_args: Resolved())
    monkeypatch.setattr(
        "solarwm.cli.validate_route",
        lambda _config: SimpleNamespace(family="test", key="test:train"),
    )
    monkeypatch.setattr("solarwm.cli.build_launch_manifest", lambda **_kwargs: {})
    monkeypatch.setattr("solarwm.backends.load_backend", lambda _family: Backend())
    monkeypatch.setenv("RANK", "0")

    assert _run_backend(Namespace(config=Resolved.path, set=[], command="train")) == 0
    assert json.loads(stale_result.read_text(encoding="utf-8"))["status"] == "complete"


def test_camera_inference_routes_provenance_to_create_only_run_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    publish_root = tmp_path / "release"
    run_root = publish_root / "runs/camera-run"

    class Resolved:
        values: ClassVar[dict[str, object]] = {
            "action": "infer",
            "name": "camera-default-name",
            "inference": {
                "length": "camera",
                "output_layout": "dataset_triplet_v1",
                "run_id": "camera-run",
            },
            "runtime": {"output_dir": str(publish_root)},
        }
        path = tmp_path / "config.yaml"
        source_digest = "source"
        resolved_digest = "resolved"

        @staticmethod
        def write_json(path: Path) -> None:
            path.write_text("{}\n", encoding="utf-8")

    class Backend:
        @staticmethod
        def validate_config(config) -> None:
            del config

        @staticmethod
        def infer(config) -> int:
            del config
            assert (run_root / "resolved-config.json").is_file()
            publication = run_root / "publication/COMPLETE.json"
            publication.parent.mkdir(parents=True)
            publication.write_text("{}\n", encoding="utf-8")
            return 0

    monkeypatch.setattr("solarwm.cli.load_config", lambda *_args: Resolved())
    monkeypatch.setattr(
        "solarwm.cli.validate_route",
        lambda _config: SimpleNamespace(family="test", key="test:infer"),
    )
    monkeypatch.setattr("solarwm.cli.build_launch_manifest", lambda **_kwargs: {})
    monkeypatch.setattr("solarwm.backends.load_backend", lambda _family: Backend())
    monkeypatch.setenv("RANK", "0")

    assert _run_backend(Namespace(config=Resolved.path, set=[], command="infer")) == 0
    assert (run_root / "launch-manifest.json").is_file()
    assert (run_root / "run-result.json").is_file()
    assert (run_root / "COMPLETE.json").is_file()
    assert not (publish_root / "run-result.json").exists()
    with pytest.raises(SolarWMError, match="run already exists"):
        _run_backend(Namespace(config=Resolved.path, set=[], command="infer"))


def test_backend_run_applies_wan_compile_setting_before_lazy_import(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "run"

    class Resolved:
        values: ClassVar[dict[str, object]] = {
            "action": "train",
            "runtime": {"output_dir": str(output_dir), "compile_flex": True},
        }
        path = tmp_path / "config.yaml"
        source_digest = "source"
        resolved_digest = "resolved"

        @staticmethod
        def write_json(path: Path) -> None:
            path.write_text("{}\n", encoding="utf-8")

    class Backend:
        @staticmethod
        def validate_config(config) -> None:
            del config

        @staticmethod
        def train(config) -> int:
            del config
            return 0

    def load_backend(_family: str) -> Backend:
        assert os.environ["SOLARWM_COMPILE_FLEX"] == "1"
        return Backend()

    monkeypatch.delenv("SOLARWM_COMPILE_FLEX", raising=False)
    monkeypatch.setattr("solarwm.cli.load_config", lambda *_args: Resolved())
    monkeypatch.setattr(
        "solarwm.cli.validate_route",
        lambda _config: SimpleNamespace(family="wan22_ti2v_5b", key="wan:test"),
    )
    monkeypatch.setattr("solarwm.cli.build_launch_manifest", lambda **_kwargs: {})
    monkeypatch.setattr("solarwm.backends.load_backend", load_backend)
    monkeypatch.setenv("RANK", "0")

    assert _run_backend(Namespace(config=Resolved.path, set=[], command="train")) == 0


def test_backend_run_clears_inherited_wan_compile_setting_when_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir = tmp_path / "run"

    class Resolved:
        values: ClassVar[dict[str, object]] = {
            "action": "train",
            "runtime": {"output_dir": str(output_dir), "compile_flex": False},
        }
        path = tmp_path / "config.yaml"
        source_digest = "source"
        resolved_digest = "resolved"

        @staticmethod
        def write_json(path: Path) -> None:
            path.write_text("{}\n", encoding="utf-8")

    class Backend:
        @staticmethod
        def validate_config(config) -> None:
            del config

        @staticmethod
        def train(config) -> int:
            del config
            return 0

    def load_backend(_family: str) -> Backend:
        assert "SOLARWM_COMPILE_FLEX" not in os.environ
        return Backend()

    monkeypatch.setenv("SOLARWM_COMPILE_FLEX", "1")
    monkeypatch.setattr("solarwm.cli.load_config", lambda *_args: Resolved())
    monkeypatch.setattr(
        "solarwm.cli.validate_route",
        lambda _config: SimpleNamespace(family="wan22_ti2v_5b", key="wan:test"),
    )
    monkeypatch.setattr("solarwm.cli.build_launch_manifest", lambda **_kwargs: {})
    monkeypatch.setattr("solarwm.backends.load_backend", load_backend)
    monkeypatch.setenv("RANK", "0")

    assert _run_backend(Namespace(config=Resolved.path, set=[], command="train")) == 0
