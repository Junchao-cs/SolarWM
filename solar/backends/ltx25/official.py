"""Embedded provider for the official LTX-2.5 runtime.

This module is dependency-light by design. Torch and the LTX packages are
imported only after the runtime readiness checks pass.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from solar.checkpoint import VerifiedCheckpoint
from solar.errors import BackendContractError
from solar.runtime.distributed import collective_call, collective_rank_zero_call

from .adapter import lora_target_modules
from .checkpoint import (
    FP32_SCALE_TABLES,
    LORA_CHECKPOINT_CONTRACT,
    VIDEO_CONNECTOR_PARAMETERS,
    VIDEO_CONNECTOR_TENSORS,
    VIDEO_CORE_PARAMETERS,
    VIDEO_CORE_TENSORS,
    BaseCheckpointInspection,
    InferenceAdapterCheckpoint,
    StrictModelLoadReceipt,
)
from .inference import InferencePlan
from .runtime import (
    RUNTIME_PROVIDER_API,
    InferenceRuntimeBundle,
    PreencodeRuntimeBundle,
    ProviderCheck,
    TrainingRuntimeBundle,
)

PROVIDER_IDENTITY = "solarwm.ltx25.official.v1"


def _inference_data_source(config: Mapping[str, Any]) -> Any:
    from .torch_data import IndexedPreencodedSource

    return IndexedPreencodedSource(config)


def _collective_resume_checkpoint(
    path: str | Path,
    config: Mapping[str, Any],
) -> VerifiedCheckpoint:
    """Deep-verify once, then construct the same manifest view on every rank."""

    import torch.distributed as dist

    from solar.checkpoint import verify_checkpoint

    from .runtime import validate_ltx_checkpoint

    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    def deep_verify() -> str:
        verified = verify_checkpoint(path)
        validate_ltx_checkpoint(verified, config)
        return verified.manifest_digest

    identity = str(
        collective_rank_zero_call(
            deep_verify,
            dist=dist,
            rank=rank,
            world_size=world_size,
            label="LTX resume deep verification",
            error_type=BackendContractError,
        )
    )

    def local_verify() -> VerifiedCheckpoint:
        verified = verify_checkpoint(path)
        validate_ltx_checkpoint(verified, config)
        if verified.manifest_digest != identity:
            raise BackendContractError("LTX resume manifest identity differs across ranks")
        return verified

    verified = collective_call(
        local_verify,
        dist=dist,
        rank=rank,
        world_size=world_size,
        label="LTX resume rank-local verification",
        error_type=BackendContractError,
    )
    return verified


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise BackendContractError(f"required LTX package is missing: {name}") from exc


def _initialize_codec_world() -> Any:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    return torch.device("cuda", local_rank)


def _model_receipt(
    inspection: BaseCheckpointInspection,
    *,
    adapter_manifest_digest: str = "",
) -> StrictModelLoadReceipt:
    return StrictModelLoadReceipt(
        provider_identity=PROVIDER_IDENTITY,
        ltx_core_version=_package_version("ltx-core"),
        header_digest=inspection.header_digest,
        retained_layout_digest=inspection.retained_layout_digest,
        strict_state_dict=True,
        missing_keys=(),
        unexpected_keys=(),
        video_core_tensors=VIDEO_CORE_TENSORS,
        video_core_parameters=VIDEO_CORE_PARAMETERS,
        video_connector_tensors=VIDEO_CONNECTOR_TENSORS,
        video_connector_parameters=VIDEO_CONNECTOR_PARAMETERS,
        fp32_scale_tables=FP32_SCALE_TABLES,
        dropped_streams=("audio", "audio_video_cross_attention"),
        adapter_target_count=LORA_CHECKPOINT_CONTRACT.target_count,
        adapter_targets=lora_target_modules(),
        adapter_trainable_parameters=LORA_CHECKPOINT_CONTRACT.trainable_parameters,
        adapter_mode="checkpoint" if adapter_manifest_digest else "initialized",
        adapter_checkpoint_manifest_digest=adapter_manifest_digest,
        fused_prope_parameter_free=True,
    )


def _module_check(name: str) -> ProviderCheck:
    available = importlib.util.find_spec(name) is not None
    return ProviderCheck(
        f"provider.module.{name}",
        "pass" if available else "fail",
        (
            "official runtime module is importable"
            if available
            else "official runtime module is missing"
        ),
        evidence={"module": name},
    )


class OfficialLTX25Provider:
    api_version = RUNTIME_PROVIDER_API
    identity = PROVIDER_IDENTITY

    def readiness(self, config: Mapping[str, Any], action: str) -> Sequence[ProviderCheck]:
        data = config.get("data", {})
        input_mode = str(data.get("input_mode", "")) if isinstance(data, Mapping) else ""
        modules = ["ltx_core", "ltx_pipelines"]
        if action == "train":
            modules.append("peft")
        if action == "preencode" or input_mode == "raw_online":
            modules.extend(("ltx_trainer", "torchvision", "av"))
        checks = [_module_check(name) for name in modules]
        versions: dict[str, str] = {}
        for package in ("ltx-core", "ltx-trainer"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "missing"
        version_ok = versions["ltx-core"] == "1.2.0"
        if action == "preencode" or input_mode == "raw_online":
            version_ok = version_ok and versions["ltx-trainer"] == "1.2.0"
        checks.append(
            ProviderCheck(
                "provider.official_versions",
                "pass" if version_ok else "fail",
                (
                    "official LTX package versions satisfy the runtime contract"
                    if version_ok
                    else "official LTX package versions do not satisfy the runtime contract"
                ),
                evidence={"versions": versions},
            )
        )
        model = config.get("model", {})
        asset_paths: dict[str, str] = {}
        if isinstance(model, Mapping):
            asset_paths["transformer"] = str(model.get("checkpoint_path") or "")
            codec = model.get("codec", {})
            if isinstance(codec, Mapping):
                asset_paths["video_vae"] = str(codec.get("video_vae_path") or "")
                if action == "preencode" or input_mode == "raw_online":
                    asset_paths["gemma4"] = str(codec.get("gemma4_path") or "")
        local_assets = bool(asset_paths) and all(
            value and "://" not in value and Path(value).expanduser().is_absolute()
            for value in asset_paths.values()
        )
        checks.append(
            ProviderCheck(
                "provider.local_model_assets",
                "pass" if local_assets else "fail",
                (
                    "embedded provider accepts absolute local model asset paths"
                    if local_assets
                    else "embedded provider requires absolute local model asset paths"
                ),
                evidence={"paths": asset_paths},
            )
        )
        checks.append(
            ProviderCheck(
                f"provider.route.{action}.{input_mode or 'none'}",
                "pass",
                "embedded official route is implemented",
                evidence={"provider": self.identity},
            )
        )
        return tuple(checks)

    def create_training_runtime(
        self,
        config: Mapping[str, Any],
        inspection: BaseCheckpointInspection,
        validation_plan: InferencePlan,
    ) -> TrainingRuntimeBundle:
        import torch

        from .official_codec import OfficialOnlineCodec, codec_receipt
        from .torch_distributed import initialize
        from .torch_model import load_strict_model
        from .torch_raw import RawOnlineBatchSource
        from .torch_training import LTX25TrainingRuntime
        from .torch_validation import TrainingValidation

        distributed = initialize(dict(config))
        device = torch.device("cuda", distributed.local_rank)
        model = config["model"]
        codec_config = model["codec"]
        input_mode = str(config["data"]["input_mode"])
        checkpoint_config = config["checkpoint"]
        resume_path = str(checkpoint_config.get("resume_from") or "").strip()
        resume_checkpoint = (
            _collective_resume_checkpoint(resume_path, config) if resume_path else None
        )
        loaded = load_strict_model(
            inspection,
            device=device,
            camera_translation_transform=str(model["camera_translation_transform"]),
            attention_backend=str(model["attention_backend"]),
        )
        receipt = _model_receipt(
            inspection,
            adapter_manifest_digest=(
                resume_checkpoint.manifest_digest if resume_checkpoint is not None else ""
            ),
        )
        online_codec = None
        batch_source = None
        validation = None
        try:
            if input_mode == "raw_online":
                online_codec = OfficialOnlineCodec(
                    transformer_path=model["checkpoint_path"],
                    video_vae_path=codec_config["video_vae_path"],
                    gemma4_path=codec_config["gemma4_path"],
                    device=device,
                    identity=f"{self.identity}:direct-vae:gemma4-preconnector",
                )
                batch_source = collective_call(
                    lambda: RawOnlineBatchSource(config, online_codec),
                    dist=torch.distributed,
                    label="LTX raw training reader setup",
                    error_type=BackendContractError,
                )
            validation = collective_call(
                lambda: TrainingValidation(
                    config,
                    validation_plan,
                    device=device,
                    model_receipt=receipt,
                    online_codec=online_codec,
                ),
                dist=torch.distributed,
                label="LTX training validation reader/decoder setup",
                error_type=BackendContractError,
            )
            runtime = LTX25TrainingRuntime(
                config,
                loaded,
                receipt,
                validation_hook=validation,
                resume_checkpoint=resume_checkpoint,
                batch_source=batch_source,
            )
        except Exception:
            if validation is not None:
                validation.close()
            if batch_source is not None:
                batch_source.close()
            if online_codec is not None:
                online_codec.close()
            raise
        if online_codec is None:
            codec_load = codec_receipt(
                provider_identity=self.identity,
                video_vae_class=validation.decoder.implementation_class,
            )
        else:
            codec_load = codec_receipt(
                provider_identity=self.identity,
                video_vae_class=validation.decoder.implementation_class,
                gemma_feature_extractor_class=(online_codec.feature_extractor_class),
                video_vae_encoder_class=online_codec.video_vae_class,
            )

        def close() -> None:
            try:
                runtime.close()
            finally:
                try:
                    validation.close()
                finally:
                    if online_codec is not None:
                        online_codec.close()

        return TrainingRuntimeBundle(runtime, receipt, codec_load, close)

    def create_inference_runtime(
        self,
        config: Mapping[str, Any],
        inspection: BaseCheckpointInspection,
        plan: InferencePlan,
        adapter_checkpoint: InferenceAdapterCheckpoint,
    ) -> InferenceRuntimeBundle:
        import torch

        from .official_codec import OfficialDiffVAEDecoder, codec_receipt
        from .torch_distributed import initialize
        from .torch_inference import (
            LTX25InferenceAdapter,
            LTX25Sampler,
            build_inference_model,
            inference_cases,
            load_negative_caption,
        )
        from .torch_model import load_strict_model

        distributed = initialize(dict(config))
        device = torch.device("cuda", distributed.local_rank)
        model = config["model"]
        codec = model["codec"]
        loaded = load_strict_model(
            inspection,
            device=device,
            camera_translation_transform=str(model["camera_translation_transform"]),
            attention_backend=str(model["attention_backend"]),
        )
        generator, _lora = build_inference_model(
            loaded,
            adapter_checkpoint,
            weights=str(model["adapter_weights"]),
        )
        receipt = _model_receipt(
            inspection,
            adapter_manifest_digest=adapter_checkpoint.manifest_digest,
        )
        source = _inference_data_source(config)
        negative_caption, negative_mask = load_negative_caption(
            str(config["inference"]["negative_caption_cache"]),
            device=device,
        )
        decoder = OfficialDiffVAEDecoder(codec["video_vae_path"], device=device)
        sampler = LTX25Sampler(
            generator,
            source,
            negative_caption,
            negative_mask,
            plan,
            device,
        )
        adapter = LTX25InferenceAdapter(sampler, decoder, model_receipt=receipt)
        cases = inference_cases(
            source,
            plan,
            camera_translation_transform=str(model["camera_translation_transform"]),
            sample_count=int(config["inference"]["sample_count"]),
            selection_seed=int(config["inference"]["selection_seed"]),
        )
        codec_load = codec_receipt(
            provider_identity=self.identity,
            video_vae_class=decoder.implementation_class,
        )

        def non_writer_run(selected: Sequence[Any], _weights_id: str) -> None:
            for case in selected:
                sampler.sample(case)

        def close() -> None:
            try:
                source.close()
            finally:
                decoder.close()
                torch.cuda.empty_cache()

        return InferenceRuntimeBundle(
            adapter=adapter,
            cases=cases,
            model_receipt=receipt,
            codec_receipt=codec_load,
            adapter_checkpoint_manifest_digest=adapter_checkpoint.manifest_digest,
            non_writer_run=non_writer_run,
            close=close,
        )

    def create_preencode_runtime(self, config: Mapping[str, Any]) -> PreencodeRuntimeBundle:
        from .official_codec import OfficialOnlineCodec, codec_receipt
        from .torch_raw import RawIndexedStream

        device = _initialize_codec_world()
        model = config["model"]
        codec_config = model["codec"]
        codec = OfficialOnlineCodec(
            transformer_path=model["checkpoint_path"],
            video_vae_path=codec_config["video_vae_path"],
            gemma4_path=codec_config["gemma4_path"],
            device=device,
            identity=f"{self.identity}:direct-vae:gemma4-preconnector",
        )
        try:
            stream = RawIndexedStream(
                config,
                logical_dp=False,
                physical_once=True,
            )
        except Exception:
            codec.close()
            raise
        receipt = codec_receipt(
            provider_identity=self.identity,
            video_vae_class=codec.video_vae_class,
            gemma_feature_extractor_class=codec.feature_extractor_class,
            video_vae_operation="direct_encode",
            video_vae_encoder_class=codec.video_vae_class,
        )

        def close() -> None:
            try:
                stream.close()
            finally:
                codec.close()

        return PreencodeRuntimeBundle(
            codec=codec,
            samples=stream.iter_epoch_zero(),
            codec_receipt=receipt,
            close=close,
        )


def create_provider() -> OfficialLTX25Provider:
    return OfficialLTX25Provider()


__all__ = [
    "PROVIDER_IDENTITY",
    "OfficialLTX25Provider",
    "create_provider",
]
