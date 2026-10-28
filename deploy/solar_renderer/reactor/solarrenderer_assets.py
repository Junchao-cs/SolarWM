"""Resolve deployment paths once, before starting the model worker."""

import hashlib
import json
from pathlib import Path

import yaml
from solarrenderer_model import SolarSettings


def read_settings(config_path: Path, weights_root: Path) -> SolarSettings:
    config = yaml.safe_load(config_path.read_text())
    source = Path(config["source"])
    if not source.is_absolute():
        source = (config_path.parent / source).resolve()
    if not (source / "solar/renderer/render.py").is_file():
        raise FileNotFoundError(
            f"Integrated SolarWM renderer missing under {source}; run python stage_source.py."
        )
    provenance_path = source / "provenance.json"
    if source.name == ".solarwm-source" and not provenance_path.is_file():
        raise FileNotFoundError("Source provenance missing; run python stage_source.py again.")
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
        for name, expected in provenance["files"].items():
            path = source / name
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(
                    f"Staged source changed: {name}; run python stage_source.py again."
                )
    root = weights_root.resolve()
    scratch = root / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return SolarSettings(
        source=source,
        h3=root / "MiniMax-H3",
        control=root / "solarrenderer_causal_control_lora.safetensors",
        fewstep=root / "solarrenderer_causal_4step_lora.safetensors",
        scratch=scratch,
    )
