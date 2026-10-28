from __future__ import annotations

from types import SimpleNamespace

import pytest

from solar.backends.ltx25.torch_inference import (
    _finite_generation_metrics,
    inference_cases,
)
from solar.errors import BackendContractError


class _Source:
    def __init__(self, count: int = 1) -> None:
        self.rows = tuple(
            SimpleNamespace(
                sample_id=f"sample-{index}",
                key=f"key-{index}",
                values={"caption": f"caption-{index}"},
            )
            for index in range(count)
        )

    def get(self, sample_id: str) -> SimpleNamespace:
        return SimpleNamespace(start_frame=3, plan_fingerprint=f"plan-{sample_id}")

    def case_fingerprint(self, sample_id: str, batch: object) -> str:
        del batch
        return f"camera-{sample_id}"


def _plan(seed: int = 10) -> SimpleNamespace:
    return SimpleNamespace(spec=SimpleNamespace(seed=seed, num_inference_steps=30))


def test_inference_cases_use_the_configured_seed_and_slot() -> None:
    cases = inference_cases(
        _Source(),
        _plan(27),
        camera_translation_transform="logd4",
    )

    assert cases[0].noise_seed == 27
    assert cases[0].metadata["camera_translation_transform"] == "logd4"
    assert cases[0].metadata["sample_solver"] == "stg-euler"
    assert cases[0].metadata["num_inference_steps"] == 30


def test_inference_cases_select_the_same_unique_recipe_rows_for_one_seed() -> None:
    source = _Source(20)
    first = inference_cases(
        source,
        _plan(),
        camera_translation_transform="linear",
        sample_count=6,
        selection_seed=42,
    )
    repeated = inference_cases(
        source,
        _plan(),
        camera_translation_transform="linear",
        sample_count=6,
        selection_seed=42,
    )
    assert [case.sample_id for case in first] == [case.sample_id for case in repeated]
    assert len({case.sample_id for case in first}) == 6
    assert [case.slot for case in first] == list(range(6))


def test_inference_cases_reject_unknown_camera_translation_transform() -> None:
    with pytest.raises(BackendContractError, match="camera_translation_transform"):
        inference_cases(
            _Source(),
            _plan(),
            camera_translation_transform="unknown",
        )


def test_ltx_finite_generation_metrics_are_explicit() -> None:
    torch = pytest.importorskip("torch")
    metrics = _finite_generation_metrics(
        latent=torch.tensor([0.0, 1.0]),
        decoded=torch.tensor([0.25, 0.75]),
        reference_decoded=torch.tensor([0.0, 1.0]),
    )
    assert metrics["finite_fraction"] == 1.0
    assert metrics["latent_finite_fraction"] == 1.0
    assert metrics["decoded_finite_fraction"] == 1.0
    assert metrics["reference_decoded_finite_fraction"] == 1.0


@pytest.mark.parametrize("field", ("latent", "decoded", "reference_decoded"))
def test_ltx_finite_generation_metrics_reject_nonfinite(field: str) -> None:
    torch = pytest.importorskip("torch")
    values = {
        "latent": torch.tensor([0.0]),
        "decoded": torch.tensor([0.0]),
        "reference_decoded": torch.tensor([0.0]),
    }
    values[field] = torch.tensor([float("nan")])
    with pytest.raises(BackendContractError, match=field):
        _finite_generation_metrics(**values)
