"""Train bidirectional or causal SolarRenderer LoRAs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from solarwm.config import load_config
from solarwm.runtime.provenance import reject_inline_secrets

from .config import validate_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--resolve", action="store_true", help="validate and print the configuration"
    )
    args = parser.parse_args(argv)
    resolved = load_config(args.config, args.set, route_validator=validate_config)
    reject_inline_secrets(resolved.values)
    config = resolved.mutable_copy()
    config["runtime"]["output_dir"] = str(
        Path(config["runtime"]["output_dir"]).expanduser().resolve()
    )
    assets = Path(__file__).resolve().parent / "assets"
    for key, filename in (
        ("silence_latents_path", "h3_silence_153_158_170.safetensors"),
        ("encoder_contract_path", "encoder_contract.json"),
    ):
        if not config["data"].get(key):
            config["data"][key] = str(assets / filename)
    if args.resolve:
        print(json.dumps(config, indent=2, sort_keys=True))
        return 0
    output = Path(config["runtime"]["output_dir"])
    if int(os.environ.get("RANK", "0")) == 0:
        output.mkdir(parents=True, exist_ok=True)
        resolved.write_json(output / "resolved-config.json")
        (output / "runtime-config.json").write_text(json.dumps(config, indent=2) + "\n")
    if config["train"]["objective"] == "dmd":
        from .dmd_runtime import run_omni_dmd_training

        return run_omni_dmd_training(config)
    return run_control_training(config)


def run_control_training(config: dict[str, Any]) -> int:
    from solarwm.training import JsonlEventSink, StepPolicy, TrainingEngine

    from .runtime import RendererTrainingRuntime

    runtime = RendererTrainingRuntime(config)
    policy = StepPolicy(
        max_steps=int(config["train"]["max_steps"]),
        grad_accum=int(config["train"]["gradient_accumulation_steps"]),
        save_every=int(config["checkpoint"]["save_every_steps"]),
        save_steps=tuple(config["checkpoint"].get("save_steps", ())),
    )
    sink = (
        JsonlEventSink(Path(config["runtime"]["output_dir"]) / "training-events.jsonl")
        if runtime.is_main
        else None
    )
    try:
        TrainingEngine(runtime, policy, event_sink=sink).run()
    finally:
        runtime.reader.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
