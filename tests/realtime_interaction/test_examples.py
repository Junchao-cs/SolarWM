"""Canonical examples and compatibility with existing deployment manifests."""

import json
import py_compile
from pathlib import Path

import pytest

from deploy.solar_wm.solarwm_web import examples
from deploy.solar_wm.solarwm_web.prepare_scene import bundled_cases
from deploy.solar_wm.solarwm_web.runtime.configuration import file_sha256, load_scenes

LEGACY_IDS = [
    "snow",
    "city_street",
    "indoor_architecture",
    "natural_landscape",
    "industrial_facility",
    "mountain",
    "rail_station",
    "harbor_road",
    "fantasy_world",
    "cinematic_3d_environment",
]


def write_example(root, name, prompt="A test scene."):
    folder = root / name
    folder.mkdir(parents=True)
    (folder / "first.png").write_bytes(b"test image")
    (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
    return folder


def test_checkout_examples_are_the_canonical_image_prompt_pairs():
    root = Path(__file__).resolve().parents[2] / "examples/solar_wm"
    rows = bundled_cases()
    assert list(rows) == ["snow", "fantasy_world"]
    for name, row in rows.items():
        assert row["first"] == str(root / name / "first.png")
        assert row["prompt"] == (root / name / "prompt.txt").read_text().strip()


def test_examples_discover_new_scenes_and_keep_snow_first(tmp_path):
    for name in ("zzz", "garden", "aaa", "snow"):
        write_example(tmp_path, name, "  Prompt for " + name + "\n")
    rows = bundled_cases(tmp_path)
    assert list(rows) == ["snow", "aaa", "garden", "zzz"]
    assert rows["garden"]["prompt"] == "Prompt for garden"


def test_installed_package_uses_packaged_examples(monkeypatch, tmp_path):
    package = tmp_path / "site-packages/solar"
    write_example(package / "examples", "snow")
    monkeypatch.setattr(
        examples, "__file__", str(package.parent / "deploy/solar_wm/solarwm_web/examples.py")
    )
    assert bundled_cases()["snow"]["first"] == str(package / "examples/snow/first.png")


def test_installed_package_ignores_python_bytecode_cache(monkeypatch, tmp_path):
    package = tmp_path / "site-packages/solar"
    example_root = package / "examples"
    write_example(example_root, "snow")
    initializer = example_root / "__init__.py"
    initializer.write_text('"""Packaged scene resources."""\n', encoding="utf-8")
    py_compile.compile(str(initializer), doraise=True)
    assert (example_root / "__pycache__").is_dir()
    monkeypatch.setattr(
        examples, "__file__", str(package.parent / "deploy/solar_wm/solarwm_web/examples.py")
    )
    assert list(bundled_cases()) == ["snow"]


@pytest.mark.parametrize("missing", ["first.png", "prompt.txt"])
def test_examples_require_both_files(tmp_path, missing):
    folder = write_example(tmp_path, "snow")
    (folder / missing).unlink()
    with pytest.raises(FileNotFoundError):
        bundled_cases(tmp_path)


@pytest.mark.parametrize("name", ["invalid.name", "with space"])
def test_examples_require_valid_scene_ids(tmp_path, name):
    write_example(tmp_path, name)
    with pytest.raises(ValueError, match="scene id"):
        bundled_cases(tmp_path)


def test_examples_reject_empty_prompts(tmp_path):
    write_example(tmp_path, "snow", " \n")
    with pytest.raises(ValueError, match="nonempty"):
        bundled_cases(tmp_path)


def test_examples_reject_an_empty_gallery(tmp_path):
    with pytest.raises(ValueError, match="at least one"):
        bundled_cases(tmp_path)


@pytest.mark.parametrize("name", ["first.png", "prompt.txt"])
def test_examples_reject_files_outside_the_directory(tmp_path, name):
    root = tmp_path / "examples"
    folder = write_example(root, "snow")
    external = tmp_path / name
    external.write_bytes((folder / name).read_bytes())
    (folder / name).unlink()
    (folder / name).symlink_to(external)
    with pytest.raises(ValueError, match="within the examples directory"):
        bundled_cases(root)


def test_examples_reject_scene_directory_symlinks_outside_the_directory(tmp_path):
    root = tmp_path / "examples"
    root.mkdir()
    external = write_example(tmp_path / "external", "snow")
    (root / "snow").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="within the examples directory"):
        bundled_cases(root)


@pytest.mark.parametrize("legacy_root", ["old-checkout/src", "site-packages"])
def test_legacy_gallery_adapts_in_memory_and_keeps_user_inputs_and_conditions(
    tmp_path, legacy_root
):
    old = tmp_path / legacy_root / "solarwm/realtime_interaction/cases"
    rows = [
        {"scene_id": name, "first": str(old / name / "first.png"), "prompt": "My " + name}
        for name in LEGACY_IDS
    ]
    condition = tmp_path / "existing-condition.safetensors"
    condition.write_bytes(b"existing condition")
    fantasy = next(row for row in rows if row["scene_id"] == "fantasy_world")
    fantasy.update(path=str(condition), sha256=file_sha256(condition))
    upload = tmp_path / "uploaded.png"
    upload.write_bytes(b"user image")
    custom = {"scene_id": "upload-private", "first": str(upload), "prompt": "User prompt"}
    rows.append(custom)
    manifest = tmp_path / "scenes.json"
    manifest.write_text(json.dumps(rows))
    original = manifest.read_bytes()

    loaded = load_scenes(manifest)

    assert list(loaded) == ["snow", "fantasy_world", "upload-private"]
    canonical = bundled_cases()
    for name in ("snow", "fantasy_world"):
        assert loaded[name]["first"] == canonical[name]["first"]
        assert loaded[name]["prompt"] == "My " + name
    assert loaded["fantasy_world"]["path"] == fantasy["path"]
    assert loaded["fantasy_world"]["sha256"] == fantasy["sha256"]
    assert loaded["upload-private"] == custom
    assert condition.read_bytes() == b"existing condition"
    assert upload.read_bytes() == b"user image"
    assert manifest.read_bytes() == original


@pytest.mark.parametrize(
    ("identity", "image"),
    [
        ("snow", "uploads/missing.png"),
        ("city_street", "other/cases/city_street/first.png"),
        ("snow", "src/solarwm/realtime_interaction/cases/fantasy_world/first.png"),
        ("custom", "src/solarwm/realtime_interaction/cases/custom/first.png"),
    ],
)
def test_missing_custom_images_are_not_silently_replaced_or_discarded(tmp_path, identity, image):
    manifest = tmp_path / "scenes.json"
    manifest.write_text(json.dumps([{"scene_id": identity, "first": image, "prompt": "Custom"}]))
    with pytest.raises(ValueError, match="image and nonempty prompt"):
        load_scenes(manifest)


def test_custom_image_with_a_retired_builtin_id_is_kept(tmp_path):
    image = tmp_path / "custom.png"
    image.write_bytes(b"custom image")
    row = {"scene_id": "city_street", "first": str(image), "prompt": "Custom city"}
    manifest = tmp_path / "scenes.json"
    manifest.write_text(json.dumps([row]))
    assert load_scenes(manifest) == {"city_street": row}
