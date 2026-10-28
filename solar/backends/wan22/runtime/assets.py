"""Resolved, explicit paths for the Wan runtime asset closure."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solar.errors import BackendContractError


@dataclass(frozen=True)
class WanAssetLayout:
    """All files needed to construct a Wan train or inference process."""

    base: Path
    transformer_config: Path
    transformer_weights: Path
    text_encoder: Path
    tokenizer: Path
    vae: Path
    anyflow_negative_embedding: Path | None

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> WanAssetLayout:
        model = config.get("model", {})
        if not isinstance(model, Mapping):
            raise BackendContractError("model must be a mapping")
        raw_base = str(model.get("base_path", "")).strip()
        if not raw_base or not raw_base.startswith("/"):
            raise BackendContractError("model.base_path must be an absolute path")
        base = Path(raw_base)
        family = str(model.get("family", ""))
        raw_assets = model.get("assets", {})
        if raw_assets is None:
            raw_assets = {}
        if not isinstance(raw_assets, Mapping):
            raise BackendContractError("model.assets must be a mapping")

        def resolve(name: str, default: Path) -> Path:
            raw = str(raw_assets.get(name, "")).strip()
            if not raw:
                return default
            path = Path(raw)
            return path if path.is_absolute() else base / path

        raw_transformer_config = str(raw_assets.get("transformer_config", "")).strip()
        if raw_transformer_config == "builtin":
            filename = {
                "wan22_ti2v_5b": "ti2v_5b.json",
                "wan22_i2v_a14b": "i2v_a14b.json",
            }.get(family)
            if filename is None:
                raise BackendContractError(f"no built-in Wan architecture config for {family!r}")
            transformer_config = Path(__file__).parent / "architectures" / filename
        else:
            transformer_config = resolve("transformer_config", base / "config.json")

        train = config.get("train", {})
        if not isinstance(train, Mapping):
            raise BackendContractError("train must be a mapping")
        raw_negative = str(train.get("anyflow_negative_embedding", "")).strip()
        negative_path = Path(raw_negative)
        if raw_negative and negative_path.is_absolute():
            raise BackendContractError(
                "AnyFlow negative embedding must resolve beneath model.base_path"
            )
        negative = base / negative_path if raw_negative else None

        return cls(
            base=base,
            transformer_config=transformer_config,
            transformer_weights=resolve("transformer_weights", base),
            text_encoder=resolve("text_encoder", base / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer=resolve("tokenizer", base / "google" / "umt5-xxl"),
            vae=resolve("vae", base / "Wan2.2_VAE.pth"),
            anyflow_negative_embedding=negative,
        )

    def as_dict(self) -> dict[str, str]:
        result = {
            "base": str(self.base),
            "transformer_config": str(self.transformer_config),
            "transformer_weights": str(self.transformer_weights),
            "text_encoder": str(self.text_encoder),
            "tokenizer": str(self.tokenizer),
            "vae": str(self.vae),
        }
        if self.anyflow_negative_embedding is not None:
            result["anyflow_negative_embedding"] = str(self.anyflow_negative_embedding)
        return result


def is_placeholder_path(path: Path) -> bool:
    """Recognize documentation placeholders that must never reach a job."""

    return tuple(path.parts[:3]) == ("/", "path", "to") or str(path).startswith("/path/to/")


__all__ = ["WanAssetLayout", "is_placeholder_path"]
