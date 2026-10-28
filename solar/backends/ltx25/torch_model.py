"""Strict LTX-Core model construction for the embedded provider.

Importing this module requires Torch, LTX-Core, and PEFT. The
dependency-light backend imports it only after readiness succeeds.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.loader import SDOps, SingleGPUModelBuilder
from ltx_core.model.transformer.attention import (
    Attention,
    AttentionFunction,
    MaskedAttentionFunction,
)
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.model import LTXModel, LTXModelType
from ltx_core.model.transformer.model_configurator import LTXVideoOnlyModelConfigurator
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.text_encoders.gemma.embeddings_connector import (
    Embeddings1DConnectorConfigurator,
)

from solar.errors import BackendContractError

from .adapter import lora_target_modules
from .checkpoint import (
    FP32_SCALE_TABLES,
    LORA_TRAINABLE_PARAMETERS_R384,
    STATE_DICT_PREFIX,
    BaseCheckpointInspection,
    _classify_tensor,
    read_safetensors_header,
)
from .geometry import STABLE_GEOMETRY
from .torch_distributed import (
    all_gather_sequence,
    all_to_all_4d,
    register_sequence_length,
    token_bounds,
)
from .torch_distributed import (
    state as distributed_state,
)
from .torch_prope import normalize_translation_transform, prope_qkv

VIDEO_CONNECTOR_PREFIX = "video_connector."


@dataclass(frozen=True)
class TensorLoadFact:
    source: str
    target: str
    dtype: str
    shape: tuple[int, ...]


@dataclass(frozen=True)
class StrictLoadedModel:
    backbone: LTX25VideoBackbone
    core: SolarLTX25VideoModel
    connector: torch.nn.Module
    fp32_scale_tables: tuple[torch.nn.Parameter, ...]


@dataclass(frozen=True)
class LTX25TransformerArgs(TransformerArgs):
    camera_viewmats: torch.Tensor | None = None
    camera_k: torch.Tensor | None = None


def _camera_pair(viewmats: Any, intrinsics: Any) -> bool:
    if (viewmats is None) != (intrinsics is None):
        raise BackendContractError("fused PRoPE requires both camera viewmats and K")
    return viewmats is not None


def _token_camera_rows(
    value: torch.Tensor,
    *,
    batch: int,
    sequence: int,
    matrix_size: int,
    name: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 4 or tuple(value.shape[-2:]) != (matrix_size, matrix_size):
        raise BackendContractError(f"{name} must be [B,N,{matrix_size},{matrix_size}]")
    if value.shape[0] not in {1, batch} or not torch.is_floating_point(value):
        raise BackendContractError(f"{name} has invalid batch or dtype")
    if not bool(torch.isfinite(value).all()):
        raise BackendContractError(f"{name} contains NaN or Inf")
    if value.shape[0] == 1 and batch != 1:
        value = value.expand(batch, -1, -1, -1)
    if value.shape[1] == sequence:
        return value
    if sequence == STABLE_GEOMETRY.video_tokens and value.shape[1] == STABLE_GEOMETRY.latent_frames:
        return value.repeat_interleave(STABLE_GEOMETRY.tokens_per_latent, dim=1)
    raise BackendContractError(f"{name} is not aligned to the LTX token sequence")


def _with_camera(
    arguments: TransformerArgs,
    viewmats: torch.Tensor,
    intrinsics: torch.Tensor,
) -> LTX25TransformerArgs:
    values = {field.name: getattr(arguments, field.name) for field in fields(TransformerArgs)}
    return LTX25TransformerArgs(
        **values,
        camera_viewmats=viewmats,
        camera_k=intrinsics,
    )


class SolarLTX25VideoSelfAttention(Attention):
    """Official video self-attention with one parameter-free PRoPE transform."""

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: Any | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        *,
        camera_viewmats: torch.Tensor | None = None,
        camera_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cameras_active = _camera_pair(camera_viewmats, camera_k)
        runtime = distributed_state()
        if not cameras_active and runtime.sp_size == 1:
            return super().forward(
                x,
                context=context,
                mask=mask,
                pe=pe,
                k_pe=k_pe,
                perturbation_mask=perturbation_mask,
                all_perturbed=all_perturbed,
            )
        if context is not None and context is not x:
            raise BackendContractError("PRoPE/SP may modify only video self-attention")
        context = x
        token_viewmats = None
        token_intrinsics = None
        if cameras_active:
            token_viewmats = _token_camera_rows(
                camera_viewmats,
                batch=x.shape[0],
                sequence=x.shape[1],
                matrix_size=4,
                name="camera_viewmats",
            )
            token_intrinsics = _token_camera_rows(
                camera_k,
                batch=x.shape[0],
                sequence=x.shape[1],
                matrix_size=3,
                name="camera_K",
            )
        value = self.to_v(context)
        if all_perturbed:
            output = value
        else:
            query = self.to_q(x)
            key = self.to_k(context)
            query, key = self.preattention_function(query, key, self, mask, pe, k_pe)
            batch, local_sequence, hidden = query.shape
            if hidden != self.heads * self.dim_head:
                raise BackendContractError("LTX attention head layout drifted")
            query_heads = query.view(batch, local_sequence, self.heads, self.dim_head)
            key_heads = key.view(batch, local_sequence, self.heads, self.dim_head)
            value_heads = value.view(batch, local_sequence, self.heads, self.dim_head)
            output_transform = None
            if cameras_active:
                transformed = prope_qkv(
                    query_heads.transpose(1, 2),
                    key_heads.transpose(1, 2),
                    value_heads.transpose(1, 2),
                    viewmats=token_viewmats.to(device=query.device, dtype=query.dtype),
                    intrinsics=token_intrinsics.to(device=query.device, dtype=query.dtype),
                    camera_translation_transform=self.camera_translation_transform,
                )
                query_heads = transformed[0].transpose(1, 2).contiguous()
                key_heads = transformed[1].transpose(1, 2).contiguous()
                value_heads = transformed[2].transpose(1, 2).contiguous()
                output_transform = transformed[3]
            active_heads = self.heads
            if runtime.sp_size > 1:
                query_heads = all_to_all_4d(
                    query_heads,
                    scatter_dimension=2,
                    gather_dimension=1,
                )
                key_heads = all_to_all_4d(
                    key_heads,
                    scatter_dimension=2,
                    gather_dimension=1,
                )
                value_heads = all_to_all_4d(
                    value_heads,
                    scatter_dimension=2,
                    gather_dimension=1,
                )
                active_heads = int(query_heads.shape[2])
            if mask is not None and not isinstance(mask, torch.Tensor):
                raise BackendContractError("opaque Stage1 masks are not supported")
            output = (
                self.attention_function(
                    query_heads.flatten(-2),
                    key_heads.flatten(-2),
                    value_heads.flatten(-2),
                    active_heads,
                )
                if mask is None
                else self.masked_attention_function(
                    query_heads.flatten(-2),
                    key_heads.flatten(-2),
                    value_heads.flatten(-2),
                    active_heads,
                    mask,
                )
            )
            if runtime.sp_size > 1:
                output_heads = output.view(
                    batch,
                    output.shape[1],
                    active_heads,
                    self.dim_head,
                )
                output_heads = all_to_all_4d(
                    output_heads,
                    scatter_dimension=1,
                    gather_dimension=2,
                )
                output = output_heads.flatten(-2)
            if output_transform is not None:
                output_heads = output.view(
                    batch,
                    local_sequence,
                    self.heads,
                    self.dim_head,
                ).transpose(1, 2)
                output = output_transform(output_heads).transpose(1, 2).contiguous().flatten(-2)
            if perturbation_mask is not None:
                output = output * perturbation_mask + value * (1 - perturbation_mask)
        if self.to_gate_logits is not None:
            output = self.gated_attention_function(x, output, self)
        return self.to_out(output)


class SolarLTX25VideoTransformerBlock(BasicAVTransformerBlock):
    def forward(
        self,
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
    ) -> tuple[TransformerArgs | None, TransformerArgs | None]:
        if audio is not None or video is None:
            raise BackendContractError("LTX blocks are video-only")
        if not isinstance(video, LTX25TransformerArgs):
            return super().forward(video=video, audio=None)
        video_x = video.x
        if not video.enabled or video_x.numel() == 0:
            return video, None
        shift, scale, gate = self.get_ada_values(
            self.scale_shift_table,
            video_x.shape[0],
            video.timesteps,
            slice(0, 3),
        )
        normalized = self.ada_zero_function(video_x, self.norm_eps, scale, shift)
        attention = self.attn1(
            normalized,
            pe=video.positional_embeddings,
            mask=video.self_attention_mask,
            perturbation_mask=video.self_attn_perturbation_mask,
            all_perturbed=video.self_attn_all_perturbed,
            camera_viewmats=video.camera_viewmats,
            camera_k=video.camera_k,
        )
        video_x, normalized = self.post_sa_function(
            video_x,
            attention,
            None,
            self.norm_eps,
            gate,
        )
        video_x = video_x + self._apply_text_cross_attention(
            normalized,
            video.context,
            self.attn2,
            self.scale_shift_table,
            getattr(self, "prompt_scale_shift_table", None),
            video.timesteps,
            video.prompt_timestep,
            video.context_mask,
            cross_attention_adaln=self.cross_attention_adaln,
        )
        shift, scale, gate = self.get_ada_values(
            self.scale_shift_table,
            video_x.shape[0],
            video.timesteps,
            slice(3, 6),
        )
        scaled = self.ada_zero_function(video_x, self.norm_eps, scale, shift)
        return replace(video, x=video_x + self.ff(scaled) * gate), None


class SolarLTX25VideoModel(LTXModel):
    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
        *,
        camera_viewmats: torch.Tensor | None = None,
        camera_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cameras_active = _camera_pair(camera_viewmats, camera_k)
        if not cameras_active:
            return super().forward(video=video, audio=audio, perturbations=perturbations)
        if audio is not None or video is None:
            raise BackendContractError("camera-conditioned LTX is video-only")
        arguments = self.video_args_preprocessor.prepare(video, None)
        viewmats = _token_camera_rows(
            camera_viewmats,
            batch=arguments.x.shape[0],
            sequence=arguments.x.shape[1],
            matrix_size=4,
            name="camera_viewmats",
        )
        intrinsics = _token_camera_rows(
            camera_k,
            batch=arguments.x.shape[0],
            sequence=arguments.x.shape[1],
            matrix_size=3,
            name="camera_K",
        )
        arguments = _with_camera(arguments, viewmats, intrinsics)
        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(
                arguments.x.shape[0],
                self.num_blocks,
                arguments.x.device,
                arguments.x.dtype,
            )
        output, _ = self._process_transformer_blocks(
            video=arguments,
            audio=None,
            perturbations=perturbations,
        )
        velocity = self._process_output(
            self.scale_shift_table,
            self.norm_out,
            self.proj_out,
            output.x,
            output.embedded_timestep,
        )
        return velocity, None


def install_parameter_free_adapter(model: LTXModel) -> SolarLTX25VideoModel:
    if model.model_type is not LTXModelType.VideoOnly:
        raise BackendContractError("official LTX shell is not VideoOnly")
    before = tuple(model.state_dict())
    model.__class__ = SolarLTX25VideoModel
    configured = 0
    for block in model.transformer_blocks:
        if not isinstance(block, BasicAVTransformerBlock):
            raise BackendContractError("official LTX block type drifted")
        block.__class__ = SolarLTX25VideoTransformerBlock
        if not isinstance(block.attn1, Attention):
            raise BackendContractError("official LTX self-attention type drifted")
        block.attn1.__class__ = SolarLTX25VideoSelfAttention
        block.attn1.camera_translation_transform = "linear"
        configured += 1
    if configured != 48 or tuple(model.state_dict()) != before:
        raise BackendContractError("parameter-free PRoPE adapter changed model state")
    return model


class StrictVideoConfigurator:
    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> SolarLTX25VideoModel:
        return _retag_meta(
            install_parameter_free_adapter(LTXVideoOnlyModelConfigurator.from_metadata(metadata)),
            connector=False,
        )


class StrictConnectorConfigurator:
    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> torch.nn.Module:
        container = torch.nn.Module()
        container.add_module(
            "video_connector",
            Embeddings1DConnectorConfigurator.from_metadata(metadata),
        )
        return _retag_meta(container, connector=True)


def _fp32_keys() -> frozenset[str]:
    keys = {"scale_shift_table"}
    for index in range(48):
        keys.add(f"transformer_blocks.{index}.scale_shift_table")
        keys.add(f"transformer_blocks.{index}.prompt_scale_shift_table")
    if len(keys) != FP32_SCALE_TABLES:
        raise AssertionError("internal LTX FP32 inventory drifted")
    return frozenset(keys)


FP32_KEYS = _fp32_keys()


def _retag_meta(module: Any, *, connector: bool) -> Any:
    state = module.state_dict()
    if not state or any(not tensor.is_meta for tensor in state.values()):
        raise BackendContractError("official LTX shell was not constructed on meta")
    typed = {
        key: torch.empty(
            tuple(tensor.shape),
            device="meta",
            dtype=(torch.float32 if not connector and key in FP32_KEYS else torch.bfloat16),
        )
        for key, tensor in state.items()
    }
    result = module.load_state_dict(typed, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise BackendContractError("official LTX shell dtype retag was not strict")
    return module


def _retained_layout(path: Path) -> tuple[dict[str, TensorLoadFact], dict[str, TensorLoadFact]]:
    core: dict[str, TensorLoadFact] = {}
    connector: dict[str, TensorLoadFact] = {}
    for entry in read_safetensors_header(path).tensors:
        category, target = _classify_tensor(entry.name)
        if category not in {"video_core", "video_connector"}:
            continue
        if target is None:
            raise BackendContractError("retained LTX tensor lacks a target key")
        fact = TensorLoadFact(entry.name, target, entry.dtype, entry.shape)
        selected = core if category == "video_core" else connector
        if target in selected:
            raise BackendContractError("duplicate retained LTX target key")
        selected[target] = fact
    return core, connector


def _canonical_dtype(value: Any) -> str:
    return {
        "TORCH.BFLOAT16": "BF16",
        "TORCH.FLOAT32": "F32",
        "TORCH.FLOAT16": "F16",
    }.get(str(value).upper(), str(value).upper())


def _validate_layout(
    module: torch.nn.Module,
    expected: Mapping[str, TensorLoadFact],
    *,
    component: str,
    require_meta: bool,
    device: torch.device | None = None,
) -> None:
    state = module.state_dict()
    if set(state) != set(expected):
        raise BackendContractError(
            f"{component} keys differ: missing={sorted(set(expected) - set(state))[:8]}, "
            f"extra={sorted(set(state) - set(expected))[:8]}"
        )
    for key, fact in expected.items():
        tensor = state[key]
        if tuple(tensor.shape) != fact.shape or _canonical_dtype(tensor.dtype) != fact.dtype:
            raise BackendContractError(f"{component}.{key} layout differs from checkpoint")
        if bool(tensor.is_meta) != require_meta:
            raise BackendContractError(f"{component}.{key} allocation state is invalid")
        if not require_meta and device is not None and tensor.device != device:
            raise BackendContractError(f"{component}.{key} loaded on the wrong device")


def _normalize_metadata(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for key in ("config", "gemma_source_checkpoint"):
        item = result.get(key)
        if isinstance(item, str):
            result[key] = json.loads(item)
    return result


def _validate_builder(
    builder: Any,
    checkpoint_metadata: Mapping[str, Any],
    expected: Mapping[str, TensorLoadFact],
    *,
    component: str,
) -> None:
    metadata = _normalize_metadata(builder.model_metadata())
    source = _normalize_metadata(checkpoint_metadata)
    for key in ("model_version", "config", "gemma_source_checkpoint"):
        if metadata.get(key) != source.get(key):
            raise BackendContractError(f"{component} builder metadata {key} drifted")
    if tuple(builder.module_ops) or tuple(builder.loras):
        raise BackendContractError(f"{component} builder contains unapproved mutations")
    shell = builder.meta_model(builder.model_metadata(), tuple(builder.module_ops))
    _validate_layout(
        shell,
        expected,
        component=f"{component} meta shell",
        require_meta=True,
    )


def _make_builders(
    path: Path,
    core: Mapping[str, TensorLoadFact],
    connector: Mapping[str, TensorLoadFact],
) -> tuple[Any, Any]:
    core_ops = (
        SDOps("SOLARWM_LTX25_VIDEO_CORE_EXACT")
        .with_matching(prefix=STATE_DICT_PREFIX)
        .with_replacement(STATE_DICT_PREFIX, "")
        .with_additional_allowed_keys(frozenset(core))
    )
    connector_source = f"{STATE_DICT_PREFIX}video_embeddings_connector."
    connector_ops = (
        SDOps("SOLARWM_LTX25_VIDEO_CONNECTOR_EXACT")
        .with_matching(prefix=connector_source)
        .with_replacement(connector_source, VIDEO_CONNECTOR_PREFIX)
        .with_additional_allowed_keys(frozenset(connector))
    )
    for entry in read_safetensors_header(path).tensors:
        category, target = _classify_tensor(entry.name)
        wanted_core = target if category == "video_core" else None
        wanted_connector = target if category == "video_connector" else None
        if core_ops.apply_to_key(entry.name) != wanted_core:
            raise BackendContractError("video-core SDOps differs from strict load plan")
        if connector_ops.apply_to_key(entry.name) != wanted_connector:
            raise BackendContractError("video-connector SDOps differs from strict load plan")
    return (
        SingleGPUModelBuilder(StrictVideoConfigurator, str(path), model_sd_ops=core_ops),
        SingleGPUModelBuilder(
            StrictConnectorConfigurator,
            str(path),
            model_sd_ops=connector_ops,
        ),
    )


def configure_camera_transform(backbone: LTX25VideoBackbone, value: object) -> str:
    selected = normalize_translation_transform(value)
    blocks = backbone.transformer.transformer_blocks
    if len(blocks) != 48:
        raise BackendContractError("LTX transformer block count drifted")
    for block in blocks:
        if not isinstance(block.attn1, SolarLTX25VideoSelfAttention):
            raise BackendContractError("LTX attn1 is missing the PRoPE adapter")
        block.attn1.camera_translation_transform = selected
    backbone.camera_translation_transform = selected
    return selected


def configure_attention_backend(backbone: LTX25VideoBackbone, value: object) -> None:
    if str(value).strip().lower() != "cudnn":
        raise BackendContractError("LTX training requires cuDNN SDPA")
    unmasked = AttentionFunction.SDPA_CUDNN.to_callable()
    masked = MaskedAttentionFunction.SDPA_EFFICIENT.to_callable()
    count = 0
    for module in backbone.modules():
        if isinstance(module, Attention):
            module.attention_function = unmasked
            module.masked_attention_function = masked
            count += 1
    if count <= 0:
        raise BackendContractError("official LTX model exposed no attention modules")


def load_strict_model(
    inspection: BaseCheckpointInspection,
    *,
    device: torch.device,
    camera_translation_transform: str,
    attention_backend: str,
) -> StrictLoadedModel:
    path = inspection.path
    before = path.stat()
    core_layout, connector_layout = _retained_layout(path)
    header = read_safetensors_header(path)
    core_builder, connector_builder = _make_builders(path, core_layout, connector_layout)
    _validate_builder(
        core_builder,
        header.metadata,
        core_layout,
        component="video core",
    )
    _validate_builder(
        connector_builder,
        header.metadata,
        connector_layout,
        component="video connector",
    )
    core = core_builder.build(device=device, dtype=None)
    connector_container = connector_builder.build(device=device, dtype=None)
    _validate_layout(
        core,
        core_layout,
        component="video core",
        require_meta=False,
        device=device,
    )
    _validate_layout(
        connector_container,
        connector_layout,
        component="video connector",
        require_meta=False,
        device=device,
    )
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise BackendContractError("base checkpoint changed during strict loading")
    parameters = dict(core.named_parameters())
    if set(FP32_KEYS) - set(parameters):
        raise BackendContractError("official LTX model lacks FP32 scale-table parameters")
    fp32 = tuple(parameters[key] for key in sorted(FP32_KEYS))
    if any(parameter.dtype != torch.float32 for parameter in fp32):
        raise BackendContractError("LTX FP32 scale tables were cast during load")
    connector = getattr(connector_container, "video_connector", None)
    if not isinstance(connector, torch.nn.Module):
        raise BackendContractError("strict connector container lacks video_connector")
    backbone = LTX25VideoBackbone(core, video_connector=connector)
    configure_camera_transform(backbone, camera_translation_transform)
    configure_attention_backend(backbone, attention_backend)
    return StrictLoadedModel(backbone, core, connector, fp32)


class LTX25SequenceParallelModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    @property
    def num_blocks(self) -> int:
        return int(self.model.num_blocks)

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig | None,
        *,
        camera_viewmats: torch.Tensor | None = None,
        camera_k: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if video is None or audio is not None:
            raise BackendContractError("LTX sequence parallelism is video-only")
        if (camera_viewmats is None) != (camera_k is None):
            raise BackendContractError("SP camera tensors must be paired")
        total_tokens = int(video.latent.shape[1])
        start, stop = token_bounds(total_tokens)
        local = replace(
            video,
            latent=video.latent[:, start:stop].contiguous(),
            timesteps=video.timesteps[:, start:stop].contiguous(),
            positions=video.positions[:, :, start:stop].contiguous(),
            keyframes_mask=(
                None
                if video.keyframes_mask is None
                else video.keyframes_mask[:, start:stop].contiguous()
            ),
        )
        register_sequence_length(stop - start)
        local_viewmats = (
            None if camera_viewmats is None else camera_viewmats[:, start:stop].contiguous()
        )
        local_intrinsics = None if camera_k is None else camera_k[:, start:stop].contiguous()
        output, audio_output = self.model(
            local,
            None,
            perturbations,
            camera_viewmats=local_viewmats,
            camera_k=local_intrinsics,
        )
        if output is None or audio_output is not None:
            raise BackendContractError("SP LTX model returned invalid modalities")
        return all_gather_sequence(output, 1), None


class LTX25VideoBackbone(torch.nn.Module):
    """Native patchifier + frozen preconnector + strict video transformer."""

    def __init__(
        self,
        transformer: SolarLTX25VideoModel,
        *,
        video_connector: torch.nn.Module,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.video_connector = video_connector
        self.model_fps = 24.0
        self._freeze_connector()

    def _freeze_connector(self) -> None:
        self.video_connector.requires_grad_(False)
        self.video_connector.eval()

    def train(self, mode: bool = True) -> LTX25VideoBackbone:
        super().train(mode)
        self._freeze_connector()
        return self

    def requires_grad_(self, requires_grad: bool = True) -> LTX25VideoBackbone:
        super().requires_grad_(requires_grad)
        self._freeze_connector()
        return self

    def _caption_context(
        self,
        caption: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(caption.shape[1:]) != (1024, 4096):
            raise BackendContractError("LTX caption cache must be [B,1024,4096]")
        if tuple(mask.shape) != (caption.shape[0], 1024):
            raise BackendContractError("LTX caption mask must be [B,1024]")
        from ltx_core.text_encoders.gemma.embeddings_processor import (
            EmbeddingsProcessor,
            convert_to_additive_mask,
        )

        additive = convert_to_additive_mask(mask, caption.dtype)
        with torch.no_grad():
            context, audio, binary = EmbeddingsProcessor(
                video_connector=self.video_connector
            ).create_embeddings(
                video_features=caption,
                audio_features=None,
                additive_attention_mask=additive,
            )
        if audio is not None:
            raise BackendContractError("video connector returned an audio context")
        return context, binary

    def forward(
        self,
        video_latent: torch.Tensor,
        sigma: torch.Tensor,
        caption_embedding: torch.Tensor,
        first_frame_mask: torch.Tensor,
        camera: Mapping[str, torch.Tensor],
        *,
        caption_mask: torch.Tensor,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> torch.Tensor:
        expected = (
            STABLE_GEOMETRY.latent_channels,
            STABLE_GEOMETRY.latent_frames,
            STABLE_GEOMETRY.latent_height,
            STABLE_GEOMETRY.latent_width,
        )
        if video_latent.ndim != 5 or tuple(video_latent.shape[1:]) != expected:
            raise BackendContractError("LTX latent must be [B,128,20,16,24]")
        batch = video_latent.shape[0]
        sigma = torch.as_tensor(sigma, device=video_latent.device, dtype=torch.float32)
        if tuple(sigma.shape) != (batch,):
            raise BackendContractError("LTX sigma must be [B]")
        condition = torch.as_tensor(
            first_frame_mask,
            device=video_latent.device,
            dtype=torch.bool,
        )
        if tuple(condition.shape) != (batch, STABLE_GEOMETRY.latent_frames):
            raise BackendContractError("LTX first-frame mask must be [B,20]")
        condition = condition.repeat_interleave(STABLE_GEOMETRY.tokens_per_latent, dim=1)
        from ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
        from ltx_core.types import SpatioTemporalScaleFactors, VideoLatentShape

        patchifier = VideoLatentPatchifier(patch_size=1)
        tokens = patchifier.patchify(video_latent)
        shape = VideoLatentShape.from_torch_shape(video_latent.shape)
        positions = patchifier.get_patch_grid_bounds(shape, device=video_latent.device)
        positions = get_pixel_coords(
            positions,
            SpatioTemporalScaleFactors.default(),
            causal_fix=True,
        ).float()
        positions[:, 0] /= self.model_fps
        timesteps = sigma[:, None].expand(-1, STABLE_GEOMETRY.video_tokens).clone()
        timesteps.masked_fill_(condition, 0.0)
        context, context_mask = self._caption_context(caption_embedding, caption_mask)
        modality = Modality(
            latent=tokens,
            sigma=sigma,
            timesteps=timesteps,
            positions=positions,
            context=context,
            context_mask=context_mask,
            attention_mask=None,
            keyframes_mask=condition.unsqueeze(-1).to(video_latent.dtype),
        )
        viewmats = _token_camera_rows(
            camera["viewmats"],
            batch=batch,
            sequence=STABLE_GEOMETRY.video_tokens,
            matrix_size=4,
            name="camera_viewmats",
        )
        intrinsics = _token_camera_rows(
            camera["K"],
            batch=batch,
            sequence=STABLE_GEOMETRY.video_tokens,
            matrix_size=3,
            name="camera_K",
        )
        velocity_tokens, audio = self.transformer(
            video=modality,
            audio=None,
            perturbations=perturbations,
            camera_viewmats=viewmats,
            camera_k=intrinsics,
        )
        if velocity_tokens is None or audio is not None:
            raise BackendContractError("video-only transformer returned invalid modalities")
        return patchifier.unpatchify(velocity_tokens, shape)


class LoRARuntime:
    def __init__(
        self,
        model: torch.nn.Module,
        peft_module: Any,
        parameter_by_key: OrderedDict[str, torch.nn.Parameter],
    ) -> None:
        self.model = model
        self.peft = peft_module
        self.parameter_by_key = parameter_by_key
        self.parameters = tuple(parameter_by_key.values())
        self.keys = tuple(parameter_by_key)

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return OrderedDict(
            (key, parameter.detach()) for key, parameter in self.parameter_by_key.items()
        )

    def load_state_dict(
        self,
        values: Mapping[str, torch.Tensor],
        *,
        broadcast: bool = True,
    ) -> None:
        if set(values) != set(self.parameter_by_key):
            raise BackendContractError("LoRA checkpoint key inventory differs")
        with torch.no_grad():
            for key, parameter in self.parameter_by_key.items():
                value = values[key]
                if value.shape != parameter.shape or value.dtype != parameter.dtype:
                    raise BackendContractError(f"LoRA checkpoint layout differs for {key}")
                parameter.copy_(value.to(device=parameter.device))
        if broadcast:
            self.broadcast()

    def broadcast(self) -> None:
        if torch.distributed.is_initialized():
            for parameter in self.parameters:
                torch.distributed.broadcast(parameter.data, src=0)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": "solarwm.ltx25.peft-lora.v1",
            "peft_version": str(self.peft.__version__),
            "targets": list(lora_target_modules()),
            "state_keys": list(self.keys),
            "state_shapes": {
                key: list(parameter.shape) for key, parameter in self.parameter_by_key.items()
            },
            "state_dtypes": {
                key: str(parameter.dtype).removeprefix("torch.")
                for key, parameter in self.parameter_by_key.items()
            },
        }


def inject_lora(core: torch.nn.Module) -> tuple[torch.nn.Module, LoRARuntime]:
    try:
        import peft
        import torch.distributed.tensor
    except ImportError as exc:
        raise BackendContractError("LTX LoRA requires a compatible PEFT runtime") from exc
    modules = dict(core.named_modules())
    targets = lora_target_modules()
    missing = [
        name
        for name in targets
        if name not in modules or not isinstance(modules[name], torch.nn.Linear)
    ]
    if missing:
        raise BackendContractError(f"LTX LoRA targets are missing/nonlinear: {missing[:8]}")
    configuration = peft.LoraConfig(
        r=384,
        lora_alpha=384,
        lora_dropout=0.0,
        target_modules=list(targets),
        bias="none",
        init_lora_weights=True,
    )
    wrapped = peft.get_peft_model(
        core,
        configuration,
        adapter_name="default",
        autocast_adapter_dtype=False,
    )
    realized = tuple(sorted(str(item) for item in wrapped.base_model.targeted_module_names))
    if set(realized) != set(targets) or len(realized) != len(targets):
        raise BackendContractError("PEFT realized a different LTX LoRA target set")
    wrapped.peft_config["default"].target_modules = set(targets)
    for parameter in wrapped.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    state = peft.get_peft_model_state_dict(
        wrapped,
        state_dict=wrapped.state_dict(keep_vars=True),
        adapter_name="default",
        save_embedding_layers=False,
    )
    parameter_by_key: OrderedDict[str, torch.nn.Parameter] = OrderedDict()
    for key in sorted(state):
        value = state[key]
        if not isinstance(value, torch.nn.Parameter):
            raise BackendContractError("PEFT adapter state did not retain Parameters")
        parameter_by_key[key] = value
    trainable = tuple(parameter for parameter in wrapped.parameters() if parameter.requires_grad)
    if (
        len(parameter_by_key) != 2 * len(targets)
        or {id(item) for item in parameter_by_key.values()} != {id(item) for item in trainable}
        or sum(item.numel() for item in trainable) != LORA_TRAINABLE_PARAMETERS_R384
    ):
        raise BackendContractError("realized LTX LoRA parameter inventory drifted")
    return wrapped, LoRARuntime(wrapped, peft, parameter_by_key)


__all__ = [
    "LTX25SequenceParallelModel",
    "LTX25VideoBackbone",
    "LoRARuntime",
    "SolarLTX25VideoModel",
    "SolarLTX25VideoTransformerBlock",
    "StrictLoadedModel",
    "configure_attention_backend",
    "configure_camera_transform",
    "inject_lora",
    "install_parameter_free_adapter",
    "load_strict_model",
]
