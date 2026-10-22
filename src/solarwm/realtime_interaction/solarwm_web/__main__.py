"""Run from the public checkout; no experiment code on PYTHONPATH."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(prog="solarwm-web")
    parser.add_argument("command", choices=("check", "serve", "smoke"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--scene")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunks", type=int, default=8)
    parser.add_argument("--serve-after-smoke", action="store_true",
                        help="keep verified smoke models resident and then open the web service")
    args = parser.parse_args()
    if args.serve_after_smoke and args.command not in ("smoke", "check"):
        parser.error("--serve-after-smoke applies to smoke only")
    if args.chunks < 6:
        parser.error("smoke must cover at least six chunks including steady refresh")
    from .runtime.configuration import load_config, load_scenes
    config = load_config(args.config)
    from .runtime.dependencies import install
    install(config)
    scenes = load_scenes(config["scenes"])
    if args.scene and args.scene not in scenes:
        parser.error("unknown scene")
    if args.command == "check":
        import importlib.metadata
        import shutil
        for file in (config["model_config"], config["silence"]):
            if not Path(file).is_file():
                raise FileNotFoundError(file)
        if not shutil.which(config.get("ffmpeg", "ffmpeg")):
            raise RuntimeError("FFmpeg is not available")
        versions = {n: importlib.metadata.version(n) for n in
                    ("torch", "triton", "diffusers", "transformers", "peft", "safetensors")}
        if config.get("streamrender_weights"):
            # gemm_act loads this lazily on its first real FFN. Resolve it
            # before paying for eight-rank model loading / DiT preparation.
            import quack.epilogue.library
            from .vae.streamrender_tp import TensorParallelCausalViT
            versions.update({n: importlib.metadata.version(n) for n in
                             ("quack-kernels", "nvidia-cutlass-dsl")})
        print(json.dumps(dict(configuration_valid=True, scene_count=len(scenes),
            versions=versions, gpu_or_model_executed=False), indent=2))
        return
    from .worker import run
    run(args)


if __name__ == "__main__":
    main()
