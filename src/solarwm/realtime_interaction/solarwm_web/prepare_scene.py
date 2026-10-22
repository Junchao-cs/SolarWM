"""Encode a new image/prompt using the public repository's official codec.

The web worker runs this in a disposable encoder process on first selection.
Does not pad prompts or change the inference model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from .examples import bundled_cases as bundled_cases
from .runtime.configuration import file_sha256, load_config, load_scenes


def encode_scene(model_config, image_path, prompt_text, scene_id, output, device="cuda:0"):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", scene_id):
        raise ValueError("invalid scene-id")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("use a fresh scene directory")
    image_path = Path(image_path).resolve(strict=True)
    prompt_text = prompt_text.strip()
    if not prompt_text:
        raise ValueError("nonempty prompt required")
    config = json.loads(Path(model_config).read_text())
    import numpy as np
    import torch
    from PIL import Image, ImageOps
    from safetensors.torch import save_file
    from solarwm.backends.minimax_h3.optional import load_conditioners
    from solarwm.backends.minimax_h3.official_codec import encode_joint_prompt_condition
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    device = torch.device(device)
    torch.cuda.set_device(device)
    started = time.perf_counter()
    components = load_conditioners(config["model"], device=device, audio_vae=False, schedulers=False)
    components.video_vae.set_attention_backend("native")
    with Image.open(image_path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB").resize((1344, 768), Image.Resampling.LANCZOS)
    with torch.random.fork_rng(devices=[device.index]), torch.inference_mode():
        prompt, tags = encode_joint_prompt_condition(image, prompt_text,
            processor=components.processor, tokenizer=components.tokenizer,
            text_encoder=components.text_encoder, device=device)
        if not 1000 <= prompt.shape[-2] <= 1300:
            raise ValueError("actual prompt length outside current dynamic compiler domain; no padding applied")
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None].to(device)
        anchor = encode_vae_condition(components.video_vae, pixels, (.485, .456, .406),
                                      (.229, .224, .225), encode_seed=42)
    values = dict(prompt_embeds=prompt.cpu().contiguous(), text_token_tags=tags.cpu().contiguous(),
        anchor_latents=anchor.cpu().contiguous(),
        K=torch.tensor([[1., 0., .5], [0., 1., .5], [0., 0., 1.]]).expand(290, -1, -1).contiguous())
    if not all(bool(torch.isfinite(t).all()) for t in values.values()):
        raise FloatingPointError("nonfinite native conditions")
    output.mkdir(parents=True, exist_ok=False)
    image.save(output / "first.png")
    save_file(values, str(output / "condition.safetensors"))
    row = dict(scene_id=scene_id, title=scene_id, prompt=prompt_text,
        first=str(output / "first.png"), path=str(output / "condition.safetensors"),
        sha256=file_sha256(output / "condition.safetensors"))
    with (output / "scene.json").open("x") as f:
        json.dump(row, f, indent=2, ensure_ascii=False)
    receipt = dict(image_sha256=file_sha256(image_path),
        prompt_sha256=hashlib.sha256(prompt_text.encode()).hexdigest(),
        model_config_sha256=file_sha256(model_config), condition_sha256=row["sha256"],
        prompt_padding=False, shapes={k: list(v.shape) for k, v in values.items()},
        elapsed_s=time.perf_counter() - started)
    with (output / "encoding.json").open("x") as f:
        json.dump(receipt, f, indent=2)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job", help=argparse.SUPPRESS)
    p.add_argument("--config", help="deployment.json created by launch.sh configure")
    p.add_argument("--scene", help="one bundled case, e.g. snow")
    for flag in ("--model-config", "--image", "--prompt-file", "--scene-id", "--output"):
        p.add_argument(flag)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    if args.job:
        from .scene_inputs import encode_job
        encode_job(args.job)
        return
    custom = (args.model_config, args.image, args.prompt_file, args.scene_id, args.output)
    if args.config or args.scene:
        if not (args.config and args.scene) or any(custom):
            p.error("use --config and --scene together, without custom-image options")
        from .scene_inputs import encode_cached
        config = load_config(args.config)
        from .runtime.dependencies import install
        install(config)
        row = encode_cached(config, load_scenes(config["scenes"])[args.scene], args.device)
    else:
        if not all(custom):
            p.error("use --config and --scene, or supply all custom-image options")
        row = encode_scene(args.model_config, args.image, Path(args.prompt_file).read_text(),
                           args.scene_id, args.output, args.device)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
