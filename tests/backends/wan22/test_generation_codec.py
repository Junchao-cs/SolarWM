from __future__ import annotations

from pathlib import Path

import pytest

from solar.backends.wan22.codec import PREENCODED_PROFILES, profile_for
from solar.backends.wan22.generation import resolve_generation_plan
from solar.config import load_config
from solar.errors import BackendContractError

ROOT = Path(__file__).resolve().parents[3]


def test_dual_153f_tensor_contracts_are_distinct() -> None:
    ti2v = PREENCODED_PROFILES["solarwm.wan22_ti2v_5b.480p.153f.v1"]
    ti2v_720p = PREENCODED_PROFILES["solarwm.wan22_ti2v_5b.720p.153f.v1"]
    a14b = PREENCODED_PROFILES["solarwm.wan22_i2v_a14b.480p.153f.v1"]
    assert ti2v.latent_shape == (39, 48, 30, 54)
    assert ti2v.i2v_y_shape is None
    assert ti2v_720p.latent_shape == (39, 48, 44, 80)
    assert ti2v_720p.i2v_y_shape is None
    assert a14b.latent_shape == (39, 16, 60, 104)
    assert a14b.i2v_y_shape == (39, 20, 60, 104)


def test_schema_cannot_cross_families() -> None:
    with pytest.raises(BackendContractError, match="belongs to"):
        profile_for(
            "solarwm.wan22_ti2v_5b.480p.153f.v1",
            family="wan22_i2v_a14b",
        )


def test_inference_resolves_recipe_test_selection_directly() -> None:
    path = ROOT / "configs/solar_wm/wan22_ti2v_5b/infer_stage1_tf_anyflow_v1_5_81f.yaml"
    config = load_config(path).values
    plan = resolve_generation_plan(config)
    assert plan.index == "recipes/clean-81f/raw-wds/test-index.jsonl.gz"
    assert plan.selection_seed == 42
    assert plan.sample_count == 16
    assert [item.num_inference_steps for item in plan.passes] == [50, 4]
    assert [item.weights for item in plan.passes] == ["ema", "ema"]


def test_validation_sample_count_is_required_and_positive() -> None:
    path = ROOT / "configs/solar_wm/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    del config["validation"]["sample_count"]
    with pytest.raises(BackendContractError, match="sample_count"):
        resolve_generation_plan(config)


@pytest.mark.parametrize(
    ("field", "value"),
    (("selection_seed", -1), ("selection_seed", True), ("noise_seed", "42")),
)
def test_validation_seeds_are_explicit_nonnegative_integers(field: str, value: object) -> None:
    path = ROOT / "configs/solar_wm/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["validation"][field] = value
    with pytest.raises(BackendContractError, match=field):
        resolve_generation_plan(config)


def test_inference_only_sampler_drift_is_rejected() -> None:
    path = ROOT / "configs/solar_wm/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml"
    config = load_config(path).mutable_copy()
    config["inference"]["source"] = "custom"
    with pytest.raises(BackendContractError, match="source must be validation"):
        resolve_generation_plan(config)
