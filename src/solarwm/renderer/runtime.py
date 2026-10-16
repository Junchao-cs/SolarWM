"""SolarRenderer training on SolarWM's distributed optimizer and checkpoint runtime.

Adapted from SolarRenderer, Copyright 2026 SolarRenderer authors, Apache-2.0.
Modified to keep renderer checkpoints and data independent of camera-model routes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from solarwm.backends.minimax_h3.runtime import H3TrainingRuntime
from solarwm.checkpoint import CheckpointContract, verify_checkpoint
from solarwm.errors import BackendContractError


def weight_identity(path: str) -> str:
    """Bind an initialization to file contents, including standalone upstream adapters."""
    source = Path(path)
    if source.is_dir():
        if (source / "checkpoint-manifest.json").is_file():
            return verify_checkpoint(source).manifest_digest
        source = source / "adapter.pt"
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_adapter(path: str) -> dict[str, Any]:
    """Load a control checkpoint or a released safetensors LoRA with canonical keys."""
    import torch

    from solarwm.backends.minimax_h3.weights import canonical_lora_key

    root = Path(path)
    if root.suffix == ".safetensors":
        from safetensors.torch import load_file

        values = load_file(str(root), device="cpu")
        values = {
            key if key.startswith("base_model.model.") else f"base_model.model.{key}": value
            for key, value in values.items()
        }
    else:
        if root.is_dir() and (root / "checkpoint-manifest.json").is_file():
            verify_checkpoint(root)
        values = torch.load(
            root / "adapter.pt" if root.is_dir() else root,
            map_location="cpu",
            weights_only=True,
        )["state"]
    translated = {canonical_lora_key(key): value for key, value in values.items()}
    if len(translated) != len(values):
        raise BackendContractError("renderer adapter contains duplicate canonical keys")
    return translated


class RendererTrainingRuntime(H3TrainingRuntime):
    def _check_finite(self, finite: bool, message: str) -> None:
        if self.dist.is_initialized():
            flag = self.torch.tensor(int(finite), dtype=self.torch.int32, device=self.device)
            self.dist.all_reduce(flag, op=self.dist.ReduceOp.MIN)
            finite = bool(flag.item())
        if not finite:
            raise FloatingPointError(message)

    def _next_batch(self, stream: Any | None = None) -> Any:
        from solarwm.backends.minimax_h3.runtime import _collective_call

        source = self.reader if stream is None else stream
        return _collective_call(
            source.next, dist=self.dist, topology=self.topology, label="renderer data read"
        )

    def _make_reader(self) -> Any:
        from .data import H3OmniStream

        data = self.config["data"]
        return H3OmniStream(
            root=str(data["omni_root"]),
            index=str(data["omni_index"]),
            topology=self.topology,
            seed=int(data.get("seed", 42)),
            encoder_contract_path=str(data["encoder_contract_path"]),
            length_buckets=bool(data.get("omni_length_buckets", True)),
        )

    def _initialize_adapter(self) -> None:
        spec = (self.checkpoint_cfg.get("initialization") or {}).get("student")
        if spec:
            if spec.get("weight_source", "live") != "live":
                raise BackendContractError("renderer initialization requires live adapter weights")
            self.lora.load_state_dict(read_adapter(str(spec["path"])), broadcast=False)
            self._weights_id = weight_identity(str(spec["path"]))

    def _make_core(self) -> Any:
        from .core import H3Stage0p5Core
        from .observations import observation_profile_from_train_config

        options = {"observations": observation_profile_from_train_config(self.train_cfg)}
        if self.train_cfg.get("causal_omni"):
            from .causal import H3OmniCausalCore

            return H3OmniCausalCore(
                self.model,
                self.silence,
                self.device,
                dict(self.train_cfg["causal_omni"]),
                **options,
            )
        return H3Stage0p5Core(self.model, self.silence, self.device, **options)

    def _make_ema(self) -> Any:
        if self.train_cfg["objective"] == "dmd":
            return None
        return super()._make_ema()

    def _restore_ema(self, state: Any) -> None:
        if self.train_cfg["objective"] == "dmd":
            if state is not None:
                raise BackendContractError("renderer DMD checkpoint unexpectedly contains EMA")
            self.ema = None
            return
        super()._restore_ema(state)

    def _make_contract(
        self,
        *,
        encoder_profile: Mapping[str, Any],
        silence_profile: Mapping[str, Any],
        base_model: Mapping[str, Any],
        config: Mapping[str, Any],
    ) -> CheckpointContract:
        from .observations import observation_profile_from_train_config

        train = config["train"]
        dmd = train.get("dmd", {})
        causal = bool(train.get("causal_omni") or dmd.get("mode") == "causal")
        parents = {
            name: weight_identity(str(dmd[name]))
            for name in ("control", "init_lora", "student_init", "accel_init")
            if dmd.get(name)
        }
        initialization = (config["checkpoint"].get("initialization") or {}).get("student")
        if initialization:
            parents["control_initialization"] = weight_identity(str(initialization["path"]))
        return CheckpointContract(
            family="minimax_h3_renderer",
            stage="fewstep" if dmd else "control",
            causal_mode="block_causal" if causal else "bidirectional",
            objective=str(train["objective"]),
            objective_variant="structure_guided" if dmd else "data_ward_velocity",
            camera_translation_transform="logd4",
            parameterization="peft-lora-r384-alpha384",
            sp_size=int(config["distributed"]["sequence_parallel_size"]),
            data_generation="solarrenderer.ref2va.v1",
            extras=deepcopy(
                {
                    "encoder_profile": dict(encoder_profile),
                    "silence_profile": dict(silence_profile),
                    "base_model": dict(base_model),
                    "observations": observation_profile_from_train_config(train).fingerprint(),
                    "training_profile": {
                        key: value for key, value in train.items() if key != "max_steps"
                    },
                    "data_plan": self.reader.current_plan_fingerprint,
                    "parents": parents,
                    "ema": config["checkpoint"]["ema"],
                }
            ),
        )

    def save_checkpoint(self, step: int) -> str:
        self.torch.cuda.empty_cache()
        try:
            return super().save_checkpoint(step)
        finally:
            self.torch.cuda.empty_cache()

    def validate(self, step: int) -> Mapping[str, Any]:
        del step
        raise BackendContractError("evaluate renderer checkpoints with the renderer inference CLI")
