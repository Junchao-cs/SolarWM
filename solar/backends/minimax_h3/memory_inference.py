"""Optional Memory LoRA loading for H3 Stage2 inference."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from solar.errors import BackendContractError, ConfigurationError

from .memory_contract import RETRIEVAL_REVISION, matches_identity

STAGE1_PARENT_SHA256 = "0fd068ac43e95d799b12e1f1cd24efe24c8b65ba419b46289756b5120f8aafab"
FIX3900_PARENT_SHA256 = "2f8058b1f505e99a381759305e991a65a628517bb528aa47e3539a34b8a75b24"
CAUSAL_MEMORY_MASK = "bank-free-shared;own-group-clean-and-noisy;memory-reads-strict-past-w5-v1"


def validate_memory_inference(config):
    memory = config.get("inference", {}).get("memory")
    if memory is None:
        return
    if not isinstance(memory, Mapping):
        raise ConfigurationError("inference.memory must be a mapping")
    if not isinstance(memory.get("enabled", False), bool):
        raise ConfigurationError("inference.memory.enabled must be boolean")
    if not memory.get("enabled", False):
        return
    if (
        config.get("action") != "infer"
        or config.get("train", {}).get("stage") != "stage2"
        or config["inference"].get("length_policy") != "source"
    ):
        raise ConfigurationError("Memory inference requires H3 Stage2 source-length inference")
    if (
        config["checkpoint"].get("weight_source") != "ema"
        or config["inference"].get("expected_step") != 3900
    ):
        raise ConfigurationError("Memory transfer requires official Stage2 FIX3900 EMA")
    if not isinstance(memory.get("path"), str) or not memory["path"].strip():
        raise ConfigurationError("Specify --memory-lora /path/to/memory-lora")
    step = memory.get("expected_step", 1050)
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ConfigurationError("inference.memory.expected_step must be a positive integer")
    if memory.get("weight_source", "live") != "live":
        raise ConfigurationError("This transfer profile uses LIVE Memory adapter.pt, not EMA")
    if not isinstance(memory.get("save_bank", True), bool):
        raise ConfigurationError("inference.memory.save_bank must be boolean")
    for field, expected in (("top_k", 5), ("recent_chunks", 5)):
        if memory.get(field, expected) != expected:
            raise ConfigurationError(f"inference.memory.{field} must be {expected}")


def _selected_component(path, step):
    """Verify the adapter file and checkpoint receipts."""
    root = Path(path)
    if root.is_file():
        if root.name != "adapter.pt":
            raise BackendContractError(
                "LIVE Memory must select adapter.pt or its checkpoint directory"
            )
        root = root.parent
    manifest_bytes = (root / "checkpoint-manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    complete = json.loads((root / "COMPLETE.json").read_text())
    if (
        manifest.get("schema") != "solarwm.checkpoint.v2"
        or complete.get("schema") != "solarwm.checkpoint-complete.v2"
        or manifest.get("step") != step
        or complete.get("step") != step
        or hashlib.blake2s(manifest_bytes).hexdigest() != complete.get("manifest_digest")
    ):
        raise BackendContractError("Memory checkpoint completion marker, manifest or step differs")
    matches = [item for item in manifest["files"] if item["path"] == "adapter.pt"]
    component = root / "adapter.pt"
    if len(matches) != 1 or component.stat().st_size != matches[0]["size"]:
        raise BackendContractError("Memory LIVE component differs from its manifest")
    return component, manifest, complete["manifest_digest"]


def validate_memory_checkpoint(manifest):
    """Validate Memory checkpoint compatibility."""
    contract = manifest.get("contract", {})
    extras = contract.get("extras", {})
    dual = extras.get("dual_lora", {})
    trained_parent = extras.get("base_model", {}).get("frozen_adapter", {}).get("sha256")
    required = dict(
        family="minimax_h3",
        objective="anyflow_forward_map",
        objective_variant="v1_5",
        causal_mode="teacher_forcing",
        parameterization="peft-lora-r384-alpha384",
        camera_translation_transform="logd4",
    )
    if contract.get("stage") != "stage1-memory" or any(
        contract.get(k) != v for k, v in required.items()
    ):
        raise BackendContractError("Expected a Stage1 AnyFlow Memory checkpoint")
    required_extras = dict(
        memory_frames_per_chunk=5,
        bank_frames=45,
        memory_temporal_rope=-1,
        chunk_memory_mask=CAUSAL_MEMORY_MASK,
    )
    if not matches_identity(extras.get("memory_schema"), "schema") or any(
        extras.get(k) != v for k, v in required_extras.items()
    ):
        raise BackendContractError(
            "Memory checkpoint must use causal-history attention and top-five retrieval"
        )
    allowed_parents = {
        "official-stage1-ema-step3000": STAGE1_PARENT_SHA256,
    }
    if (
        dual.get("composition") != "additive"
        or dual.get("trainable") != "memory"
        or dual.get("frozen") not in allowed_parents
        or trained_parent != allowed_parents[dual["frozen"]]
    ):
        raise BackendContractError(
            "Memory frozen-parent identity differs from the supported recipe"
        )
    return dual["frozen"], trained_parent, "explicit-inference-parent-transfer"


def attach_memory(model, frozen_lora, profile):
    """Freeze the official parent and load only the additive LIVE Memory weights."""
    import torch

    from .memory_adapter import attach_memory_adapter
    from .source_length_cache import file_sha256
    from .weights import canonical_lora_key

    step = int(profile.get("expected_step", 1050))
    component, manifest, manifest_digest = _selected_component(profile["path"], step)
    parent_name, trained_parent, mode = validate_memory_checkpoint(manifest)

    model, memory, _ = attach_memory_adapter(model, frozen_lora)
    actual_parent = memory.base_identity["frozen_adapter"]["sha256"]
    if actual_parent != FIX3900_PARENT_SHA256:
        raise BackendContractError("Loaded Stage2 LoRA is not the official FIX3900 EMA")
    payload = torch.load(component, map_location="cpu", mmap=True, weights_only=True)
    values = {
        canonical_lora_key(k)
        .replace(".lora_A.memory.weight", ".lora_A.weight")
        .replace(".lora_B.memory.weight", ".lora_B.weight"): v
        for k, v in payload["state"].items()
    }
    if len(values) != len(payload["state"]) or set(values) != set(memory.parameter_by_key):
        raise BackendContractError("Memory LIVE LoRA tensor keys differ")
    with torch.no_grad():
        for key, parameter in memory.parameter_by_key.items():
            value = values[key]
            if value.shape != parameter.shape or value.dtype != torch.bfloat16:
                raise BackendContractError(f"Memory LIVE tensor shape/dtype differs: {key}")
            if not bool(torch.isfinite(value).all()):
                raise BackendContractError(f"Non-finite Memory LIVE tensor: {key}")
            parameter.copy_(value)
    model.eval().requires_grad_(False)
    identity = dict(
        schema="solarwm.h3-stage2-memory-transfer.v1",
        enabled=True,
        mode=mode,
        step=step,
        weight_source="live",
        adapter_sha256=file_sha256(component),
        manifest_digest=manifest_digest,
        trained_parent=parent_name,
        trained_parent_adapter_sha256=trained_parent,
        actual_parent="official-stage2-fix-ema-step3900",
        actual_parent_adapter_sha256=actual_parent,
        composition="H3 + official Stage2 LoRA + Memory LoRA; unit scales",
        anyflow_delta_embedding_enabled=False,
        attention_mask=CAUSAL_MEMORY_MASK,
        retrieval_revision=RETRIEVAL_REVISION,
        top_k=5,
        recent_chunks=5,
        memory_temporal_rope=-1,
        save_bank=profile.get("save_bank", True),
    )
    return model, identity


def save_memory_trace(dest, key, forward, identity):
    """Save retrieval metadata and the optional latent bank."""
    import torch
    from safetensors.torch import save_file

    from .source_length_cache import atomic_json, file_sha256

    stem = Path(dest) / key
    bank_sha = None
    if identity["save_bank"]:
        # Concatenate on CPU to avoid duplicating the bank on GPU.
        latents = torch.cat([chunk.cpu() for chunk in forward.bank.chunks], dim=2).contiguous()
        path = stem.with_name(stem.name + ".memory-bank.safetensors")
        part = path.with_name("." + path.name + ".tmp")
        try:
            save_file(
                {"generated_latents": latents},
                str(part),
                metadata={"schema": "solarwm.h3-generated-latent-bank.v1", "dtype": "bfloat16"},
            )
            part.replace(path)
        finally:
            part.unlink(missing_ok=True)
        bank_sha = file_sha256(path)
    trace = stem.with_name(stem.name + ".memory.json")
    atomic_json(
        trace,
        dict(
            schema="solarwm.h3-online-memory-trace.v1",
            weights=identity,
            records=forward.records,
            memory_source="completed generated latent chunks only; no GT or future video",
            camera_c2w=forward.c2w.tolist(),
            camera_K=forward.K.tolist(),
            kv_cache="recent five chunks, raw before RoPE/PRoPE; no persistent Memory KV",
            latent_bank_sha256=bank_sha,
        ),
    )
    return dict(
        memory=identity, memory_trace_sha256=file_sha256(trace), memory_bank_sha256=bank_sha
    )
