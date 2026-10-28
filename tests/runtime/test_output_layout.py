from pathlib import Path

import pytest

from solar.errors import BackendContractError
from solar.runtime.output_layout import (
    CAMERA_DATASET_TRIPLET_LAYOUT,
    camera_inference_output_layout,
    checkpoint_model_dir,
    cleanup_validation_staging,
    invocation_output_dir,
    portable_output_component,
    public_validation_dir,
    validation_staging_root,
)


def test_public_checkpoint_names_match_release_contract(tmp_path: Path) -> None:
    assert checkpoint_model_dir(tmp_path, step=50, width=6) == (
        tmp_path / "checkpoint_model_000050"
    )
    assert checkpoint_model_dir(tmp_path, step=50, width=8) == (
        tmp_path / "checkpoint_model_00000050"
    )


def test_public_validation_name_and_private_staging_are_disjoint(tmp_path: Path) -> None:
    public = public_validation_dir(tmp_path, step=20, pass_name="ema")
    staging = validation_staging_root(tmp_path) / "step-00000020-ema" / "dp-rank-00000"
    assert public == tmp_path / "validation/step_000020_ema"
    assert staging == tmp_path / "validation/.staging/step-00000020-ema/dp-rank-00000"
    assert ".staging" not in public.parts


def test_success_cleanup_removes_only_private_staging(tmp_path: Path) -> None:
    public = public_validation_dir(tmp_path, step=20, pass_name="live")
    (public / "compare").mkdir(parents=True)
    (public / "COMPLETE.json").write_text("{}\n")
    staging = validation_staging_root(tmp_path) / "step-00000020-live"
    (staging / "dp-rank-00000").mkdir(parents=True)
    (staging / "dp-rank-00000/diagnostic.json").write_text("{}\n")

    cleanup_validation_staging(staging, output_dir=tmp_path)

    assert not validation_staging_root(tmp_path).exists()
    assert (public / "COMPLETE.json").is_file()


def test_cleanup_refuses_public_validation_tree(tmp_path: Path) -> None:
    public = public_validation_dir(tmp_path, step=20, pass_name="live")
    public.mkdir(parents=True)
    with pytest.raises(BackendContractError, match="outside private staging"):
        cleanup_validation_staging(public, output_dir=tmp_path)


def test_cleanup_refuses_staging_symlink_without_deleting_target(tmp_path: Path) -> None:
    staging_root = validation_staging_root(tmp_path)
    target = staging_root / "step-00000020-live"
    target.mkdir(parents=True)
    (target / "diagnostic.json").write_text("{}\n")
    alias = staging_root / "step-00000040-live"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(BackendContractError, match="containing a symlink"):
        cleanup_validation_staging(alias, output_dir=tmp_path)

    assert alias.is_symlink()
    assert (target / "diagnostic.json").is_file()


def test_public_layout_rejects_relative_output_root() -> None:
    with pytest.raises(BackendContractError, match="must be absolute"):
        checkpoint_model_dir("relative-output", step=20, width=6)


def test_camera_inference_defaults_to_run_scoped_dataset_triplets(tmp_path: Path) -> None:
    config = {
        "action": "infer",
        "name": "camera-example-run",
        "inference": {"length": "camera"},
        "runtime": {"output_dir": str(tmp_path)},
    }

    layout = camera_inference_output_layout(config)

    assert layout is not None
    assert layout.layout == CAMERA_DATASET_TRIPLET_LAYOUT
    assert layout.publish_root == tmp_path
    assert layout.run_root == tmp_path / "runs/camera-example-run"
    assert invocation_output_dir(config) == layout.run_root


def test_camera_transaction_opt_out_and_fixed_inference_keep_legacy_root(
    tmp_path: Path,
) -> None:
    camera = {
        "action": "infer",
        "name": "camera",
        "inference": {"length": "camera", "output_layout": "transaction_v1"},
        "runtime": {"output_dir": str(tmp_path)},
    }
    fixed = {
        "action": "infer",
        "name": "fixed",
        "inference": {"length": "fixed"},
        "runtime": {"output_dir": str(tmp_path)},
    }

    assert camera_inference_output_layout(camera) is None
    assert camera_inference_output_layout(fixed) is None
    assert invocation_output_dir(camera) == tmp_path
    assert invocation_output_dir(fixed) == tmp_path


@pytest.mark.parametrize("value", ["", "../escape", "a/b", "a\\b", ".hidden", "含中文"])
def test_camera_publication_rejects_nonportable_components(value: str) -> None:
    with pytest.raises(BackendContractError, match="portable ASCII path component"):
        portable_output_component(value, field="test")


@pytest.mark.parametrize("pass_name", [".hidden", "my pass", "直播"])
def test_public_validation_preserves_legacy_pass_names(tmp_path: Path, pass_name: str) -> None:
    assert public_validation_dir(tmp_path, step=20, pass_name=pass_name) == (
        tmp_path / "validation" / f"step_000020_{pass_name}"
    )
