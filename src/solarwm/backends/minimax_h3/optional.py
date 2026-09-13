"""Lazy loaders for the MiniMax-H3 optional runtime."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solarwm.errors import BackendContractError

H3_FP32_PARAMETER_NAMES = frozenset(
    f"{owner}.{slot}"
    for owner in (
        "proj_in",
        "audio_proj_in",
        "time_embedder.linear_1",
        "time_embedder.linear_2",
        "proj_out",
        "audio_proj_out",
    )
    for slot in ("weight", "bias")
)


@dataclass
class H3RuntimeModules:
    transformer: Any = None
    transformer_block_cls: Any = None
    fp32_fsdp_units: tuple[Any, ...] = ()
    text_encoder: Any = None
    processor: Any = None
    tokenizer: Any = None
    video_vae: Any = None
    audio_vae: Any = None
    video_scheduler: Any = None
    audio_scheduler: Any = None


def require_h3_runtime() -> tuple[Any, Any, Any]:
    """Resolve heavy packages with an image-specific actionable error."""

    try:
        import diffusers
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - optional local dependency
        raise BackendContractError(
            "MiniMax-H3 execution requires torch, an H3-compatible Diffusers "
            "build, Transformers, PEFT, "
            "FlashAttention, and safetensors"
        ) from exc
    if not torch.cuda.is_available():
        raise BackendContractError("MiniMax-H3 heavy execution requires CUDA")
    return torch, diffusers, transformers


def torch_dtype(name: str) -> Any:
    torch, _diffusers, _transformers = require_h3_runtime()
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return aliases[str(name).lower()]
    except KeyError as exc:
        raise BackendContractError(f"unsupported H3 torch dtype {name!r}") from exc


def validate_parameter_dtypes(model: Any) -> None:
    """Require the official BF16 block / six-FP32-owner split."""

    torch, _diffusers, _transformers = require_h3_runtime()
    parameters = dict(model.named_parameters())
    missing = sorted(H3_FP32_PARAMETER_NAMES - set(parameters))
    if missing:
        raise BackendContractError(f"H3 checkpoint lacks FP32 owners: {missing}")
    mismatches = []
    for name, parameter in parameters.items():
        expected = torch.float32 if name in H3_FP32_PARAMETER_NAMES else torch.bfloat16
        if parameter.dtype != expected:
            mismatches.append(f"{name}:{parameter.dtype}->{expected}")
            if len(mismatches) == 8:
                break
    if mismatches:
        raise BackendContractError(
            f"H3 checkpoint parameter dtype contract differs; first mismatches={mismatches}"
        )


def _resolve_owner(model: Any, dotted: str) -> Any:
    current = model
    for part in dotted.split("."):
        current = getattr(current, part)
    return current


def load_transformer(model_cfg: Mapping[str, Any], *, device: Any) -> H3RuntimeModules:
    """Strict-load the official 33B transformer through the camera adapter."""

    _torch, _diffusers, _transformers = require_h3_runtime()
    from .model import MiniMaxH3TransformerBlock, SolarMiniMaxH3Transformer3DModel

    path = str(model_cfg["checkpoint_path"])
    kwargs = {
        "subfolder": str(model_cfg.get("transformer_subfolder", "transformer")),
        "torch_dtype": torch_dtype(str(model_cfg.get("torch_dtype", "bfloat16"))),
        "device_map": model_cfg.get("transformer_device_map"),
        "low_cpu_mem_usage": bool(model_cfg.get("low_cpu_mem_usage", True)),
        "local_files_only": bool(model_cfg.get("local_files_only", False)),
    }
    if model_cfg.get("revision") is not None:
        kwargs["revision"] = model_cfg["revision"]
    transformer = SolarMiniMaxH3Transformer3DModel.strict_from_pretrained(path, **kwargs)
    validate_parameter_dtypes(transformer)
    # W6 uses our explicit compiled BlockMask path. Native unmasked attention
    # (including Qwen token refinement and SGF score models) remains Flash.
    requested_backend = str(model_cfg.get("attention_backend", "flash"))
    transformer.set_attention_backend("flash" if requested_backend == "flex" else requested_backend)
    if model_cfg.get("transformer_device_map") is None:
        transformer = transformer.to(device)
    fp32_units = tuple(
        _resolve_owner(transformer, owner)
        for owner in (
            "proj_in",
            "audio_proj_in",
            "time_embedder.linear_1",
            "time_embedder.linear_2",
            "proj_out",
            "audio_proj_out",
        )
    )
    return H3RuntimeModules(
        transformer=transformer,
        transformer_block_cls=MiniMaxH3TransformerBlock,
        fp32_fsdp_units=fp32_units,
    )


def load_conditioners(
    model_cfg: Mapping[str, Any],
    *,
    device: Any,
    qwen: bool = True,
    video_vae: bool = True,
    audio_vae: bool = True,
    schedulers: bool = True,
) -> H3RuntimeModules:
    """Load official Qwen, VisualVAE, AudioVAE, and scheduler components."""

    torch, _diffusers, _transformers = require_h3_runtime()
    from diffusers import (
        AutoencoderKLMiniMaxH3,
        AutoencoderKLMiniMaxH3Audio,
        MiniMaxH3Scheduler,
    )
    from transformers import (
        Qwen2TokenizerFast,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
    )

    root = Path(str(model_cfg["checkpoint_path"]))
    dtype = torch_dtype(str(model_cfg.get("torch_dtype", "bfloat16")))
    result = H3RuntimeModules()
    if qwen:
        result.text_encoder = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                str(root), subfolder="text_encoder", torch_dtype=dtype
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )
        result.tokenizer = Qwen2TokenizerFast.from_pretrained(str(root / "tokenizer"))
        result.processor = Qwen3VLProcessor.from_pretrained(str(root / "processor"))
    if video_vae:
        # The offline encoder uses the configured BF16 component,
        # while the official validation decoder is loaded in FP32 and runs
        # its convolutional kernels under FP16 autocast.
        video_dtype = torch.float32 if not qwen else dtype
        result.video_vae = (
            AutoencoderKLMiniMaxH3.from_pretrained(
                str(root), subfolder="vae", torch_dtype=video_dtype
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )
    if audio_vae:
        result.audio_vae = (
            AutoencoderKLMiniMaxH3Audio.from_pretrained(
                str(root), subfolder="audio_vae", torch_dtype=dtype
            )
            .eval()
            .requires_grad_(False)
            .to(device)
        )
    if schedulers:
        result.video_scheduler = MiniMaxH3Scheduler.from_pretrained(
            str(root), subfolder=str(model_cfg.get("scheduler_subfolder", "scheduler"))
        )
        # The model package may provide a distinct audio scheduler; fall
        # back to a separately configured copy with the same class/config.
        audio_subfolder = str(model_cfg.get("audio_scheduler_subfolder", "scheduler"))
        result.audio_scheduler = MiniMaxH3Scheduler.from_pretrained(
            str(root), subfolder=audio_subfolder
        )
    return result


__all__ = [
    "H3_FP32_PARAMETER_NAMES",
    "H3RuntimeModules",
    "load_conditioners",
    "load_transformer",
    "require_h3_runtime",
    "torch_dtype",
    "validate_parameter_dtypes",
]
