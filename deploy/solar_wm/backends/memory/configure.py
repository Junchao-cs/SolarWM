"""Create a deployment; the webpage prepares image/prompt scenes on demand."""
import argparse
import json
from pathlib import Path

from deploy.solar_wm.shared.examples import bundled_cases


def configure(weights, output, memory_lora=None):
    demo = Path(__file__).resolve().parent
    repository = demo.parents[3]
    weights = Path(weights).expanduser().resolve(strict=True)
    output = Path(output).expanduser().resolve()
    if not weights.is_dir():
        raise ValueError("--weights must be a directory")
    if output == repository or repository in output.parents:
        raise ValueError("choose an --output directory outside the source checkout")
    if output == weights or weights in output.parents:
        raise ValueError("choose an --output directory outside the weights directory")
    if output.exists():
        raise FileExistsError("output already exists; use its deployment.json or choose a new directory")

    base = weights / "SolarWM-h3-33B-base"
    checkpoint = weights / "SolarWM-h3-33B-sgf-stage2-158f-fix"
    silence = weights / "h3_silence_153_158_170.safetensors"
    renderer = weights / "causal_vit24_step6000.safetensors"
    memory = Path(memory_lora).expanduser().resolve(strict=True) if memory_lora else weights / "SolarWM-h3-33B-memory-stage1-158f"
    if memory.is_file():
        if memory.name != "adapter.pt":
            raise ValueError("Memory must select adapter.pt or its checkpoint directory")
        memory = memory.parent
    for name in ("adapter.pt", "checkpoint-manifest.json", "COMPLETE.json"):
        if not (memory / name).is_file():
            raise FileNotFoundError("missing Memory checkpoint member: " + str(memory / name))
    for path in (base, checkpoint):
        if not path.is_dir():
            raise FileNotFoundError("missing model directory: " + str(path))
    if not silence.is_file():
        raise FileNotFoundError("missing silence weights: " + str(silence))
    if renderer.exists() and not renderer.is_file():
        raise ValueError("StreamRender weights must be a file: " + str(renderer))

    entries = bundled_cases()

    model = json.loads((demo / "model.example.json").read_text())
    model["model"]["checkpoint_path"] = str(base)
    model["checkpoint"]["resume_from"] = str(checkpoint)
    model["inference"]["memory"]["path"] = str(memory)
    deployment = json.loads((demo / "deployment.example.json").read_text())
    deployment.update(model_config=str(output / "model-config.json"),
                      silence=str(silence), scenes=str(output / "scenes.json"),
                      state_directory=str(output / "runtime"),
                      cache_directory=str(output / "compiler-cache"))
    deployment["memory_smoke_scenes"] = list(entries)[:3]
    if renderer.is_file():
        deployment.update(streamrender_weights=str(renderer), vae_backend="streamrender")
    else:
        deployment.pop("streamrender_weights", None)
        deployment["vae_backend"] = "current"

    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("model-config.json", model), ("deployment.json", deployment),
                        ("scenes.json", list(entries.values()))):
        with (output / name).open("x") as file:
            json.dump(value, file, indent=2)
            file.write("\n")
    return output / "deployment.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="directory containing the downloaded weights")
    parser.add_argument("--output", required=True, help="new directory for deployment configuration and runtime data")
    parser.add_argument("--memory-lora", help="Memory checkpoint directory, or adapter.pt with both receipts alongside")
    args = parser.parse_args()
    try:
        config = configure(args.weights, args.output, args.memory_lora)
    except (OSError, ValueError) as error:
        parser.exit(2, str(error) + "\n")
    print("Configuration created: " + str(config))
    print("Next: run deploy/solar_wm/launch.sh --backend memory serve --config " + str(config))


if __name__ == "__main__":
    main()
