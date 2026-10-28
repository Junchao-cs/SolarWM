"""Configuration contracts for SolarRenderer training."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from solar.errors import ConfigurationError


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject unsupported recipes before allocating models or starting collectives."""
    if config.get("action") != "train":
        raise ConfigurationError("SolarRenderer configurations must use action: train")
    model = config["model"]
    data = config["data"]
    train = config.get("train", {})
    distributed = config.get("distributed", {})
    checkpoint = config.get("checkpoint", {})
    if model.get("family") != "minimax_h3_renderer":
        raise ConfigurationError("SolarRenderer requires model.family: minimax_h3_renderer")
    if model.get("transformer_subfolder") != "transformer_ref":
        raise ConfigurationError("SolarRenderer requires the Ref2VA transformer_ref weights")
    adapter = model.get("adapter", {})
    if (adapter.get("rank"), adapter.get("alpha"), adapter.get("dropout")) != (384, 384, 0.0):
        raise ConfigurationError("SolarRenderer requires rank-384, alpha-384 LoRA without dropout")
    if train.get("stage") != "stage0p5" or train.get("objective") not in {"flow_matching", "dmd"}:
        raise ConfigurationError("SolarRenderer supports control flow matching and four-step DMD")
    for name in ("omni_root", "omni_index"):
        if not isinstance(data.get(name), str) or not data[name].strip():
            raise ConfigurationError(f"data.{name} is required")
    if (data.get("pixel_frames"), data.get("encoded_latents")) != (158, 47):
        raise ConfigurationError("SolarRenderer uses the 158-frame / 47-latent encoder profile")

    def positive(section: Mapping[str, Any], name: str) -> int:
        value = section.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ConfigurationError(f"{name} must be a positive integer")
        return value

    world = positive(distributed, "world_size")
    sp = positive(distributed, "sequence_parallel_size")
    if distributed.get("context_parallel_size", 1) != 1:
        raise ConfigurationError("SolarRenderer requires context_parallel_size: 1")
    if world % sp:
        raise ConfigurationError("world_size must be divisible by sequence_parallel_size")
    micro = positive(train, "micro_batch_size")
    accum = positive(train, "gradient_accumulation_steps")
    if micro != 1 or positive(train, "global_batch_size") != world // sp * accum:
        raise ConfigurationError("global_batch_size must equal (world_size / SP) * accumulation")
    positive(train, "max_steps")

    def nonnegative(section: Mapping[str, Any], name: str, default: int = 0) -> int:
        value = section.get(name, default)
        if type(value) is not int or value < 0:
            raise ConfigurationError(f"{name} must be a non-negative integer")
        return value

    nonnegative(checkpoint, "save_every_steps")
    if any(type(step) is not int or step < 1 for step in checkpoint.get("save_steps", ())):
        raise ConfigurationError("checkpoint.save_steps must contain positive integers")

    def number(value: Any, name: str, *, minimum: float = 0.0, strict: bool = True) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value <= minimum if strict else value < minimum)
        ):
            raise ConfigurationError(f"{name} has an invalid numeric value")
        return float(value)

    def optimizer(values: Mapping[str, Any]) -> None:
        if values.get("name") != "fp32_master_adamw":
            raise ConfigurationError("SolarRenderer requires fp32_master_adamw")
        for key in ("learning_rate", "epsilon", "gradient_clip"):
            number(values.get(key), key)
        number(values.get("weight_decay", 0.0), "weight_decay", strict=False)
        betas = values.get("betas", ())
        if len(betas) != 2 or any(
            not 0.0 <= number(beta, "betas", strict=False) < 1.0 for beta in betas
        ):
            raise ConfigurationError("optimizer.betas must contain two values in [0, 1)")
        nonnegative(values, "warmup_steps")

    optimizer(train.get("optimizer", {}))
    initial = (checkpoint.get("initialization") or {}).get("student")
    if initial and initial.get("weight_source", "live") != "live":
        raise ConfigurationError("renderer initialization requires live adapter weights")
    saved_attention = train.get("fsdp", {}).get("saved_attention_blocks", 0)
    if type(saved_attention) is not int or not 0 <= saved_attention <= 50:
        raise ConfigurationError("saved_attention_blocks must be between 0 and 50")
    observations = train.get("observations", {})
    if observations.get("patterns") != {"shared": 1.0}:
        raise ConfigurationError("SolarRenderer requires the shared observation pattern")
    if observations.get("dropout", {}).get("observation", 0.0) != 0.0:
        raise ConfigurationError("the released SolarRenderer recipe uses zero observation dropout")
    if not str(config["runtime"].get("output_dir", "")).strip():
        raise ConfigurationError("runtime.output_dir is required")
    if "://" in str(config["runtime"]["output_dir"]):
        raise ConfigurationError("runtime.output_dir must be a local or mounted filesystem path")
    if not Path(config["runtime"]["output_dir"]).expanduser().is_absolute():
        raise ConfigurationError("runtime.output_dir must be an absolute directory")
    validation = config.get("validation", {})
    if validation.get("validate_every_steps", 0) or validation.get("smoke_step", 0):
        raise ConfigurationError("use the renderer inference command for checkpoint evaluation")
    causal = train.get("causal_omni")
    if causal is not None and causal != {
        "mode": "teacher",
        "chunk": 5,
        "window": 6,
        "corrupt": 0.0,
    }:
        raise ConfigurationError("causal control training requires teacher forcing with W6/C5")
    if train["objective"] == "flow_matching":
        if train.get("dmd"):
            raise ConfigurationError("flow_matching cannot also configure DMD")
        if not checkpoint.get("ema", {}).get("enabled", False):
            raise ConfigurationError("control training requires EMA")
        return
    if causal is not None or accum != 1:
        raise ConfigurationError("DMD uses causal_stream and gradient_accumulation_steps: 1")
    if checkpoint.get("ema", {}).get("enabled", True):
        raise ConfigurationError("the SolarRenderer DMD recipe uses live weights without EMA")
    dmd = train.get("dmd", {})
    mode = dmd.get("mode")
    allowed = {
        "mode",
        "init_lora",
        "init_lora_rank",
        "init_lora_alpha",
        "forwards",
        "critic_updates_per_student",
        "score_shift",
        "control",
        "structure_guidance",
        "decoupled",
        "student_init",
        "accel_init",
        "causal_stream",
        "critic_rank",
    }
    if set(dmd) - allowed:
        raise ConfigurationError(f"unsupported DMD settings: {sorted(set(dmd) - allowed)}")
    if mode not in {"adapt_plug", "causal"}:
        raise ConfigurationError("DMD mode must be adapt_plug or causal")
    if mode == "causal":
        shard = train.get("fsdp", {}).get("frozen_base_shard_size")
        if shard is not None and (type(shard) is not int or shard < 1 or world % shard):
            raise ConfigurationError(
                "frozen_base_shard_size must be a positive divisor of world_size"
            )
        if (shard is None and world != sp) or (shard is not None and sp % shard):
            raise ConfigurationError(
                "causal DMD requires each FSDP shard to contain one logical sample: "
                "sequence_parallel_size must be divisible by frozen_base_shard_size, "
                "or equal world_size when frozen_base_shard_size is unset"
            )
    if dmd.get("forwards") != 4 or dmd.get("score_shift") != 12.0:
        raise ConfigurationError("SolarRenderer DMD uses four evaluations with video shift 12")
    positive(dmd, "critic_updates_per_student")
    if "critic_rank" in dmd:
        positive(dmd, "critic_rank")
    if "init_lora_rank" in dmd:
        positive(dmd, "init_lora_rank")
    if "init_lora_alpha" in dmd:
        number(dmd["init_lora_alpha"], "init_lora_alpha")
    guidance = dmd.get("structure_guidance", 1.0)
    if isinstance(guidance, bool) or not isinstance(guidance, (int, float)) or guidance < 1.0:
        raise ConfigurationError("structure_guidance must be at least one")
    required = (
        ("control", "init_lora")
        if mode == "adapt_plug"
        else ("control", "student_init", "accel_init")
    )
    if any(not isinstance(dmd.get(key), str) or not dmd[key].strip() for key in required):
        raise ConfigurationError(f"DMD {mode} requires paths for {', '.join(required)}")
    if mode == "causal" and (
        dmd.get("decoupled") is not True or dmd.get("causal_stream") != {"chunk": 5, "window": 6}
    ):
        raise ConfigurationError("causal DMD requires decoupled LoRAs and W6/C5 streaming")
    if train.get("critic_optimizer", {}).get("name") != "fp32_master_adamw":
        raise ConfigurationError("the DMD critic requires fp32_master_adamw")
    optimizer(train["critic_optimizer"])
