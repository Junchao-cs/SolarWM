from __future__ import annotations

from pathlib import Path

from solar.config import load_config

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "configs" / "solar_wm"


def test_all_wan_training_examples_declare_validation_camera_guards() -> None:
    expected_guards = {
        ("wan22_ti2v_5b", "train_stage0p5_fm_81f.yaml"): (20.0, 20.0),
        ("wan22_ti2v_5b", "train_stage0p5_fm_153f.yaml"): (20.0, 20.0),
        ("wan22_ti2v_5b", "train_stage1_tf_fm_81f.yaml"): (None, None),
        (
            "wan22_ti2v_5b",
            "train_stage1_tf_anyflow_v1_5_81f.yaml",
        ): (None, None),
        ("wan22_ti2v_5b", "train_stage2_sgf_81f.yaml"): (None, None),
        ("wan22_i2v_a14b", "train_stage0p5_fm_81f.yaml"): (None, None),
        ("wan22_i2v_a14b", "train_stage0p5_fm_153f.yaml"): (None, None),
    }
    for (family, name), guards in expected_guards.items():
        path = EXAMPLES / family / name
        config = load_config(path).values
        validation = config["validation"]
        assert validation["max_rel_translation"] == guards[0], path
        assert validation["max_camera_abs"] == guards[1], path
