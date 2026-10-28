from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from solar.backends.ltx25.adapter import lora_target_modules
from solar.backends.ltx25.checkpoint import (
    BASE_CHECKPOINT_CONTRACT,
    EMA_CONTRACT,
    LORA_CHECKPOINT_CONTRACT,
    BaseCheckpointContract,
    StrictCodecLoadReceipt,
    TrainingCheckpointManifest,
    fingerprint_digest,
    runtime_fingerprint,
    validate_full_resume,
)
from solar.backends.ltx25.inference import (
    GuidanceSpec,
    ValidationInferenceAdapter,
    build_inference_plan,
    euler_velocity_step,
    guided_clean_prediction,
    native_sigma_schedule,
)
from solar.backends.ltx25.official_codec import codec_receipt
from solar.backends.ltx25.runtime import (
    checkpoint_contract,
    inference_checkpoint_contract,
)
from solar.errors import BackendContractError


def test_native_inference_schedule_and_defaults() -> None:
    schedule = native_sigma_schedule()
    assert schedule.dtype == np.float32
    assert schedule.shape == (31,)
    assert schedule[0] == np.float32(0.9999999403953552)
    assert schedule[-1] == 0.0
    assert np.all(schedule[:-1] > schedule[1:])
    assert hashlib.blake2s(schedule.astype("<f4", copy=False).tobytes()).hexdigest() == (
        "b68c23b5eac261c7e8e37d90778f000a485e022b2cc105f4758c8b7791acac69"
    )
    plan = build_inference_plan()
    assert plan.guidance == GuidanceSpec()
    assert plan.latent_shape == (128, 20, 16, 24)


def test_euler_and_guidance_math() -> None:
    sample = np.asarray([1.0, 2.0], dtype=np.float32)
    velocity = np.asarray([2.0, -2.0], dtype=np.float32)
    assert np.allclose(euler_velocity_step(sample, velocity, 1.0, 0.5), [0.0, 3.0])
    guidance = GuidanceSpec(cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0, stg_blocks=())
    result = guided_clean_prediction(sample, sample * 0, sample * 0, guidance)
    assert np.array_equal(result, sample)


class _Runner:
    def __init__(self) -> None:
        self.plan_ids: list[int] = []

    def run(self, plan, request):
        self.plan_ids.append(id(plan))
        return request


def test_validation_and_inference_delegate_to_same_plan_object() -> None:
    runner = _Runner()
    plan = build_inference_plan()
    adapter = ValidationInferenceAdapter(runner, plan)
    assert adapter.infer("standalone") == "standalone"
    assert adapter.validate("validation") == "validation"
    assert runner.plan_ids == [id(plan), id(plan)]


def test_checkpoint_contract_preserves_strict_split_and_trainable_only_ema() -> None:
    assert BASE_CHECKPOINT_CONTRACT.video_core_tensors == 1362
    assert BASE_CHECKPOINT_CONTRACT.video_connector_tensors == 129
    assert BASE_CHECKPOINT_CONTRACT.fp32_scale_tables == 97
    assert LORA_CHECKPOINT_CONTRACT.target_count == 480
    assert LORA_CHECKPOINT_CONTRACT.trainable_parameters == 1_962_934_272
    assert EMA_CONTRACT.trainable_only is True
    with pytest.raises(BackendContractError, match="base checkpoint"):
        BaseCheckpointContract(video_core_tensors=0)


def test_checkpoint_contracts_keep_semantics_readable() -> None:
    config = {
        "model": {"camera_translation_transform": "linear"},
        "data": {"generation": "dataset-generation-v1"},
        "distributed": {"sequence_parallel_size": 2},
    }
    training = checkpoint_contract(config)
    inference = inference_checkpoint_contract(config)
    assert training.extras["runtime"]["lora"]["target_count"] == 480
    assert training.sp_size == 2
    assert inference.sp_size == 1
    assert inference.extras == {
        "lora": {
            "rank": 384,
            "alpha": 384,
            "dropout": 0.0,
            "target_count": 480,
            "trainable_parameters": 1_962_934_272,
            "base_scale_tables_trainable": False,
        }
    }
    assert "digest" not in json.dumps(training.as_dict(), sort_keys=True)
    assert "digest" not in json.dumps(inference.as_dict(), sort_keys=True)


def test_full_resume_requires_exact_fingerprint_and_all_state() -> None:
    fingerprint = runtime_fingerprint(
        camera_translation_transform="linear",
        data_generation="dataset-generation-v1",
    )
    targets = lora_target_modules()
    manifest = TrainingCheckpointManifest(
        global_step=1000,
        runtime_fingerprint=fingerprint,
        adapter_targets=targets,
        ema_targets=targets,
        optimizer_present=True,
        scheduler_present=True,
    )
    validate_full_resume(manifest, fingerprint)
    assert len(fingerprint_digest(fingerprint)) == 64
    changed = runtime_fingerprint(
        camera_translation_transform="logd4",
        data_generation="dataset-generation-v1",
    )
    with pytest.raises(BackendContractError, match="differing"):
        validate_full_resume(manifest, changed)
    with pytest.raises(BackendContractError, match="optimizer"):
        replace(manifest, optimizer_present=False).validate()


def test_online_training_codec_receipt_binds_decoder_encoder_and_gemma() -> None:
    receipt = StrictCodecLoadReceipt(
        provider_identity="solarwm.ltx25.official.v1",
        video_vae_class="ltx_pipelines.utils.blocks.VideoDecoder",
        diffvae_mode="chunked_eager",
        gemma_feature_extractor_class="ltx_core.text_encoders.FeatureExtractor",
        caption_cache_stage="gemma4_feature_extractor_preconnector",
        video_vae_encoder_class="ltx_core.model.video_vae.VideoEncoder",
    )
    receipt.validate(
        require_gemma=True,
    )
    with pytest.raises(BackendContractError, match="direct VAE encoder"):
        replace(receipt, video_vae_encoder_class="").validate(
            require_gemma=True,
        )


def test_online_codec_receipt_keeps_structural_identity() -> None:
    receipt = codec_receipt(
        provider_identity="solarwm.ltx25.official.v1",
        video_vae_class="ltx_pipelines.utils.blocks.VideoDecoder",
        gemma_feature_extractor_class="ltx_core.text_encoders.FeatureExtractor",
        video_vae_encoder_class="ltx_core.model.video_vae.VideoEncoder",
    )
    receipt.validate(
        require_gemma=True,
    )
    assert receipt.caption_cache_stage == "gemma4_feature_extractor_preconnector"
