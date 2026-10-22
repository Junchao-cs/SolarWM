"""Create a deployment; the webpage prepares image/prompt scenes on demand."""
import argparse
import json
from pathlib import Path

from .prepare_scene import bundled_cases


def configure(weights, output):
    demo = Path(__file__).resolve().parent.parent
    repository = demo.parents[2]
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
    deployment = json.loads((demo / "deployment.example.json").read_text())
    deployment.update(model_config=str(output / "model-config.json"),
                      silence=str(silence), scenes=str(output / "scenes.json"),
                      state_directory=str(output / "runtime"),
                      cache_directory=str(output / "compiler-cache"))
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
    args = parser.parse_args()
    try:
        config = configure(args.weights, args.output)
    except (OSError, ValueError) as error:
        parser.exit(2, str(error) + "\n")
    print("Configuration created: " + str(config))
    print("Next: run launch.sh serve --config " + str(config))


if __name__ == "__main__":
    main()
