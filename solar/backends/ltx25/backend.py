"""Executable LTX-2.5 routes backed by a heavy-runtime provider."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solar.checkpoint import verify_checkpoint
from solar.data import read_index, resolve_index_path
from solar.data.index import IndexRow
from solar.errors import BackendContractError
from solar.inference import InferenceEngine
from solar.preencode import write_shard
from solar.runtime.distributed import collective_call, collective_rank_zero_call
from solar.training import (
    JsonlEventSink,
    StepPolicy,
    TrainingEngine,
)

from .checkpoint import InferenceAdapterCheckpoint, inspect_base_checkpoint
from .codec import RawSample, encode_online
from .config import LTX25RunContract, validate_ltx25_config
from .inference import GuidanceSpec, InferenceSpec, build_inference_plan
from .preencode import encoded_payload, encoder_contract
from .preencode_transaction import (
    create_staging,
    finalize_local_preencode,
    rank_shard_relative,
    write_rank_publication,
)
from .readiness import probe_ltx25_runtime, write_readiness_report
from .runtime import (
    InferenceRuntimeBundle,
    LTX25RuntimeProvider,
    PreencodeRuntimeBundle,
    TrainingRuntimeBundle,
    VerifiedTrainingRuntime,
    require_training_runtime,
    resolve_runtime_provider,
    validate_ltx_checkpoint,
    validate_ltx_inference_checkpoint,
)


def _mapping(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise BackendContractError(f"{key} must be a mapping")
    return value


def _rank() -> int:
    try:
        value = int(os.environ.get("RANK", "0"))
    except ValueError as exc:
        raise BackendContractError("RANK must be an integer") from exc
    if value < 0:
        raise BackendContractError("RANK must be nonnegative")
    return value


def _inference_adapter_checkpoint(
    config: Mapping[str, Any],
) -> InferenceAdapterCheckpoint:
    model = _mapping(config, "model")
    path = Path(str(model["adapter_checkpoint_path"])).expanduser()
    weights = str(model["adapter_weights"]).lower()
    checkpoint_format = str(model.get("adapter_checkpoint_format", "transaction_v1")).lower()
    verified = verify_checkpoint(path)
    if checkpoint_format == "inference_transaction_v1":
        validate_ltx_inference_checkpoint(verified, config, weights=weights)
    else:
        validate_ltx_checkpoint(verified, config)
    component = "adapter" if weights == "live" else "ema"
    tensor_path = verified.path / component / "model.safetensors"
    if not tensor_path.is_file() or tensor_path.is_symlink():
        raise BackendContractError(f"LTX checkpoint lacks {component} safetensors")
    return InferenceAdapterCheckpoint(
        path=verified.path,
        tensor_path=tensor_path,
        manifest_digest=verified.manifest_digest,
        weights=weights,
        checkpoint_format=checkpoint_format,
    )


def _inference_plan(config: Mapping[str, Any], *, validation: bool) -> Any:
    section = _mapping(config, "validation" if validation else "inference")
    if validation:
        value = section.get("inference")
        if not isinstance(value, Mapping):
            raise BackendContractError("validation.inference must be a mapping")
        section = value
    guidance = section["guidance"]
    if not isinstance(guidance, Mapping):
        raise BackendContractError("inference.guidance must be a mapping")
    return build_inference_plan(
        InferenceSpec(
            num_inference_steps=int(section["num_inference_steps"]),
            seed=int(section["seed"]),
            fps=int(section["fps"]),
            decoder_mode=str(section["decoder_mode"]),
        ),
        GuidanceSpec(
            cfg_scale=float(guidance["cfg_scale"]),
            stg_scale=float(guidance["stg_scale"]),
            rescale_scale=float(guidance["rescale_scale"]),
            stg_blocks=tuple(int(item) for item in guidance["stg_blocks"]),
        ),
    )


def _world_size() -> int:
    try:
        value = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError as exc:
        raise BackendContractError("WORLD_SIZE must be an integer") from exc
    rank = _rank()
    if value < 1 or rank >= value:
        raise BackendContractError("RANK must be inside WORLD_SIZE")
    return value


def _preencode_expected_rows(source_index_path: Path) -> tuple[IndexRow, ...]:
    """Apply the same fixed-window policy used by the raw preencode stream."""

    from .torch_raw import normalize_training_window

    return tuple(normalize_training_window(row) for row in read_index(source_index_path))


def _initialize_preencode_collective() -> tuple[Any | None, int, int]:
    """Create the raw-world group before any per-rank preencode preflight."""

    rank = _rank()
    world_size = _world_size()
    if world_size == 1:
        return None, rank, world_size
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        raise BackendContractError("distributed LTX preencoding requires Torch") from exc
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except (KeyError, ValueError) as exc:
        raise BackendContractError(
            "distributed LTX preencoding requires integer LOCAL_RANK"
        ) from exc
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    if (dist.get_rank(), dist.get_world_size()) != (rank, world_size):
        raise BackendContractError(
            "LTX preencode environment differs from the initialized process group"
        )
    return dist, rank, world_size


@dataclass(frozen=True)
class LTX25Backend:
    """Dependency-light public boundary for the independent LTX backend."""

    family: str = "ltx25_video"
    provider: LTX25RuntimeProvider | None = None
    enforce_environment: bool = True

    def validate_config(self, config: Mapping[str, Any]) -> None:
        validate_ltx25_config(config)

    def _prepare(
        self, config: Mapping[str, Any], expected_action: str
    ) -> tuple[LTX25RunContract, LTX25RuntimeProvider, Path]:
        contract = validate_ltx25_config(config)
        if contract.action != expected_action:
            raise BackendContractError(
                f"backend method {expected_action} cannot execute action {contract.action!r}"
            )
        resolution = resolve_runtime_provider(config, injected=self.provider)
        report = probe_ltx25_runtime(
            config,
            resolution=resolution,
            enforce_environment=self.enforce_environment,
        )
        output_dir = Path(str(_mapping(config, "runtime")["output_dir"])).expanduser()
        report_path = output_dir / f"ltx25-readiness.rank-{_rank():05d}.json"
        write_readiness_report(report_path, report)
        report.require_ready(report_path=report_path)
        if resolution.provider is None:
            raise BackendContractError("LTX readiness passed without a runtime provider")
        return contract, resolution.provider, output_dir

    def train(self, config: Mapping[str, Any]) -> int:
        contract, provider, output_dir = self._prepare(config, "train")
        model = _mapping(config, "model")
        inspection = inspect_base_checkpoint(str(model["checkpoint_path"]))
        plan = _inference_plan(config, validation=True)
        checkpoint_config = _mapping(config, "checkpoint")
        resume_path = str(checkpoint_config.get("resume_from") or "").strip()
        resume_manifest_digest = ""
        if resume_path:
            resume_checkpoint = verify_checkpoint(resume_path)
            validate_ltx_checkpoint(resume_checkpoint, config)
            resume_manifest_digest = resume_checkpoint.manifest_digest
        bundle = provider.create_training_runtime(config, inspection, plan)
        if not isinstance(bundle, TrainingRuntimeBundle):
            raise BackendContractError("provider returned an invalid TrainingRuntimeBundle")
        if bundle.model_receipt.provider_identity != provider.identity:
            raise BackendContractError("training model receipt provider identity differs")
        bundle.model_receipt.validate(
            inspection,
            adapter_checkpoint_manifest_digest=resume_manifest_digest,
        )
        if bundle.codec_receipt is None:
            raise BackendContractError("training validation lacks a DiffVAE load receipt")
        if bundle.codec_receipt.provider_identity != provider.identity:
            raise BackendContractError("training codec receipt provider identity differs")
        bundle.codec_receipt.validate(
            require_gemma=contract.input_mode == "raw_online",
        )
        delegate = require_training_runtime(bundle.runtime)
        runtime = VerifiedTrainingRuntime(
            delegate,
            config=config,
            model_receipt=bundle.model_receipt,
            output_dir=output_dir,
        )
        train = _mapping(config, "train")
        checkpoint = _mapping(config, "checkpoint")
        validation = _mapping(config, "validation")
        smoke_step = int(validation.get("smoke_step", 0) or 0)
        policy = StepPolicy(
            max_steps=int(train["max_steps"]),
            grad_accum=int(train["gradient_accumulation_steps"]),
            save_every=int(checkpoint.get("save_every_steps", 0) or 0),
            validate_every=int(validation.get("validate_every_steps", 0) or 0),
            validation_steps=(smoke_step,) if smoke_step > 0 else (),
        )
        sink = JsonlEventSink(output_dir / f"training-events.rank-{_rank():05d}.jsonl")
        try:
            TrainingEngine(runtime, policy, event_sink=sink).run()
        finally:
            if bundle.close is not None:
                bundle.close()
        return 0

    def infer(self, config: Mapping[str, Any]) -> int:
        _contract, provider, output_dir = self._prepare(config, "infer")
        model = _mapping(config, "model")
        inspection = inspect_base_checkpoint(str(model["checkpoint_path"]))
        adapter_checkpoint = _inference_adapter_checkpoint(config)
        plan = _inference_plan(config, validation=False)
        bundle = provider.create_inference_runtime(
            config,
            inspection,
            plan,
            adapter_checkpoint,
        )
        if not isinstance(bundle, InferenceRuntimeBundle):
            raise BackendContractError("provider returned an invalid InferenceRuntimeBundle")
        if (
            bundle.model_receipt.provider_identity != provider.identity
            or bundle.codec_receipt.provider_identity != provider.identity
        ):
            raise BackendContractError("inference receipt provider identity differs")
        if bundle.adapter_checkpoint_manifest_digest != adapter_checkpoint.manifest_digest:
            raise BackendContractError("inference bundle is not bound to the adapter checkpoint")
        bundle.model_receipt.validate(
            inspection,
            adapter_checkpoint_manifest_digest=adapter_checkpoint.manifest_digest,
        )
        bundle.codec_receipt.validate(
            require_gemma=False,
        )
        weights_id = f"{adapter_checkpoint.manifest_digest}:{str(model['adapter_weights']).lower()}"
        cases = tuple(bundle.cases)
        if not cases:
            raise BackendContractError("inference provider returned no cases")
        try:
            if _rank() == 0:
                InferenceEngine(bundle.adapter).run(
                    cases,
                    weights_id=weights_id,
                    output_dir=output_dir / "inference",
                )
            elif bundle.non_writer_run is not None:
                bundle.non_writer_run(cases, weights_id)
            else:
                raise BackendContractError(
                    "distributed inference requires a provider non_writer_run hook"
                )
        finally:
            if bundle.close is not None:
                bundle.close()
        return 0

    def preencode(self, config: Mapping[str, Any]) -> int:
        dist, rank, world_size = _initialize_preencode_collective()
        prepared: list[tuple[LTX25RunContract, LTX25RuntimeProvider, Path]] = []

        def prepare_backend() -> None:
            prepared.append(self._prepare(config, "preencode"))

        collective_call(
            prepare_backend,
            dist=dist,
            rank=rank,
            world_size=world_size,
            label="LTX preencode readiness",
        )
        _contract, provider, _output_dir = prepared[0]

        bundles: list[PreencodeRuntimeBundle] = []

        def create_bundle() -> None:
            value = provider.create_preencode_runtime(config)
            bundles.append(value)

        try:
            collective_call(
                create_bundle,
                dist=dist,
                rank=rank,
                world_size=world_size,
                label="LTX preencode provider setup",
            )
        except Exception:
            close = getattr(bundles[0], "close", None) if bundles else None
            if callable(close):
                close()
            raise
        bundle = bundles[0]
        closed = False

        def close_bundle() -> None:
            nonlocal closed
            close = getattr(bundle, "close", None)
            if not closed and callable(close):
                close()
            closed = True

        settings = _mapping(config, "preencode")
        data = _mapping(config, "data")
        output_root_text = str(settings["output_root"])
        output_root = Path(output_root_text).expanduser().resolve()
        contract = encoder_contract()

        def validate_bundle() -> None:
            if not isinstance(bundle, PreencodeRuntimeBundle):
                raise BackendContractError("provider returned an invalid PreencodeRuntimeBundle")
            if bundle.codec_receipt.provider_identity != provider.identity:
                raise BackendContractError("preencode codec receipt provider identity differs")
            bundle.codec_receipt.validate(
                require_gemma=True,
            )
            if not str(bundle.codec.identity).strip():
                raise BackendContractError("preencode codec identity is empty")
            if "://" in output_root_text:
                raise BackendContractError("preencode.output_root must be a local filesystem path")
            if str(data.get("partition_mode")) != "global_occurrence":
                raise BackendContractError(
                    "LTX preencode requires data.partition_mode=global_occurrence"
                )

        try:
            collective_call(
                validate_bundle,
                dist=dist,
                rank=rank,
                world_size=world_size,
                label="LTX preencode provider contract",
            )
            staging_text = collective_rank_zero_call(
                lambda: str(create_staging(output_root)),
                dist=dist,
                rank=rank,
                world_size=world_size,
                label="LTX preencode staging setup",
            )
            staging = Path(staging_text).resolve()
            shard_size = int(settings["samples_per_shard"])
            index_template = str(settings["index_relative_path"])
            index_relative = index_template.replace("{rank}", f"{rank:05d}")

            def encode_rank() -> None:
                rows: list[Mapping[str, Any]] = []
                shard_receipts = []
                pending = []
                shard_index = 0

                def flush() -> None:
                    nonlocal shard_index
                    if not pending:
                        return
                    receipt = write_shard(
                        staging,
                        rank_shard_relative(rank, shard_index),
                        tuple(pending),
                    )
                    for source in receipt.rows:
                        row = dict(source)
                        row["shard_generation"] = f"local-digest:{receipt.digest}"
                        indices = tuple(int(item) for item in row["source_frame_indices"])
                        row["num_frames"] = indices[-1] + 1
                        metadata = row.get("metadata", {})
                        if not isinstance(metadata, Mapping):
                            raise BackendContractError(
                                "preencoded LTX row metadata must be a mapping"
                            )
                        row["fps"] = float(metadata["source_fps"])
                        rows.append(row)
                    shard_receipts.append(receipt)
                    pending.clear()
                    shard_index += 1

                try:
                    for raw in bundle.samples:
                        if not isinstance(raw, RawSample):
                            raise BackendContractError("provider yielded a non-RawSample value")
                        sample = encode_online(raw, bundle.codec)
                        pending.append(
                            encoded_payload(
                                sample,
                                codec_identity=bundle.codec.identity,
                            )
                        )
                        if len(pending) == shard_size:
                            flush()
                    flush()
                    if not rows:
                        raise BackendContractError(
                            "preencode provider yielded no samples for this rank"
                        )
                    write_rank_publication(
                        staging,
                        rank=rank,
                        world_size=world_size,
                        rows=rows,
                        shards=shard_receipts,
                        index_relative_path=index_relative,
                        provider_identity=provider.identity,
                        codec_identity=bundle.codec.identity,
                        codec_load_receipt_digest=bundle.codec_receipt.digest,
                        encoder_contract_digest=contract.digest,
                    )
                finally:
                    close_bundle()

            collective_call(
                encode_rank,
                dist=dist,
                rank=rank,
                world_size=world_size,
                label="LTX preencode rank production",
            )
            source_index_path = resolve_index_path(data, "index")

            def finalize() -> Mapping[str, Any]:
                expected_rows = _preencode_expected_rows(source_index_path)
                return finalize_local_preencode(
                    staging,
                    output_root,
                    expected_rows=expected_rows,
                    source_index_path=source_index_path,
                    world_size=world_size,
                    provider_identity=provider.identity,
                    codec_identity=bundle.codec.identity,
                    codec_load_receipt_digest=bundle.codec_receipt.digest,
                    encoder_contract=contract,
                )

            collective_rank_zero_call(
                finalize,
                dist=dist,
                rank=rank,
                world_size=world_size,
                label="LTX preencode corpus finalization",
            )
        except Exception:
            close_bundle()
            raise
        return 0


def create_backend(*, family: str = "ltx25_video") -> LTX25Backend:
    if family != "ltx25_video":
        raise BackendContractError(f"LTX-2.5 backend cannot serve family {family!r}")
    return LTX25Backend(family=family)


__all__ = ["LTX25Backend", "create_backend"]
