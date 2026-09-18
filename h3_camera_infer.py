#!/usr/bin/env python3
"""Real camera-conditioned MiniMax-H3 Stage0.5 generation on a custom image.

UNTESTED -- written by tracing SolarWM's actual source (backends/minimax_h3/
{optional,lora,runtime,official_codec,artifacts,camera,stage0p5,inference}.py)
with no local GPU/Python available to run it. Expect to debug real errors on
first run; report them back with the actual traceback.

WHY THIS EXISTS: SolarWM's real dataset-conditioned H3 path (`solarwm infer`,
runtime.py's run_inference()) needs the still-unpublished
minimax-h3-158f-768p-nomind-v1 latent dataset and is built entirely around
torchrun/FSDP/distributed collectives -- not usable for a single custom
image. h3_infer.py (the other script here) works around that by driving the
UNCONDITIONED base diffusers pipeline instead, which has no camera input at
all (confirmed: a "goes straight" prompt produced a left turn).

This script is a third path: it reuses the REAL trained Stage0.5 LoRA
adapter (SolarWM-h3-33B-bid-stage0p5-158f) and REAL camera-conditioned
H3Stage0p5Core.generate(), but builds the H3ArtifactBatch by hand from:
  - your own first-frame image (encoded via the real VisualVAE -> anchor_latents)
  - your own prompt text (encoded jointly with the image via the real Qwen
    text encoder -> prompt_embeds/text_token_tags)
  - a SYNTHETIC, hand-authored camera_c2w trajectory (see build_camera_c2w)
      -- this is genuine per-frame conditioning, not a text hint
Target_latents is NOT required: H3Stage0p5Core.generate() never reads it
(confirmed by reading stage0p5.py -- it's training-loss/comparison-video
only), so a real full video clip is not needed, just the one image.

CAVEAT, partially resolved: forward-translation sign was confirmed backwards
on a real generation (straight case moved in reverse) and is now fixed
(_compose_c2w uses +forward_step). Rotation AXIS/SIGN convention (does
positive yaw here actually correspond to "turn left" on screen?) is still
UNVERIFIED against a real example -- expect to flip --yaw-deg's sign if
left/right come out backwards too.

LOAD-ONCE: --mind-batch <manifest.json> loads the model/codec ONCE and loops
every entry (same idea as h3_infer.py's --mind-batch). Each entry needs
image, prompt, out, and either "direction" (straight/left/right, uses
build_camera_c2w) or "camera_c2w" (path to a real [47,4,4] .npy, e.g. from
MIND action data -- see MIND/src/drive_solarwm.py's mind_actions_to_camera_c2w).

Run (single GPU, no torchrun needed):
    .venv-h3/bin/python h3_camera_infer.py \\
        --base-model /path/to/SolarWM-models/SolarWM-h3-33B-base \\
        --adapter /path/to/SolarWM-models/SolarWM-h3-33B-bid-stage0p5-158f \\
        --image examples/racer/Screenshot.png \\
        --prompt-file examples/racer/prompt.txt \\
        --direction left \\
        --out outputs/h3-camera/left.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Qwen's joint image+text encoder always needs SOME prompt string (there is
# no code path to skip it), but it does not need to describe direction --
# camera_c2w is the real per-frame conditioning now, so a fixed neutral
# prompt is enough for straight/left/right alike. Matches MIND's
# drive_solarwm.py --engine camera BASE_PROMPT.
GENERIC_PROMPT = "Third-person view moving through the scene, consistent environment, realistic camera motion."


def build_camera_c2w(direction: str, *, num_latents: int, yaw_deg: float, forward_step: float):
    """Author a synthetic [num_latents,4,4] absolute C2W trajectory.

    Frame 0 is identity (matches how align_h3_camera() re-relativizes to
    frame 0 anyway, so starting there is not a loss of generality). Each
    subsequent frame composes a small per-step yaw (about Y) + forward
    translation (local -Z) onto the previous frame -- an orbit/pan, not a
    real recorded trajectory. See module docstring's axis-convention caveat.
    """
    import numpy as np

    yaw_per_frame = {"straight": 0.0, "left": -abs(yaw_deg) / num_latents, "right": abs(yaw_deg) / num_latents}[direction]
    return _compose_c2w(num_latents, yaw_deg_per_frame=yaw_per_frame, forward_step=forward_step)


def _compose_c2w(num_latents: int, *, yaw_deg_per_frame: float, forward_step: float):
    import numpy as np

    theta = np.deg2rad(yaw_deg_per_frame)
    rot = np.array(
        [[np.cos(theta), 0.0, np.sin(theta)],
         [0.0, 1.0, 0.0],
         [-np.sin(theta), 0.0, np.cos(theta)]],
        dtype=np.float32,
    )
    delta = np.eye(4, dtype=np.float32)
    delta[:3, :3] = rot
    # Confirmed backwards on a real generation (racer_overlay.mp4, straight
    # case): -forward_step moved the camera in reverse. Flipped to
    # +forward_step so straight now advances into the scene.
    delta[:3, 3] = [0.0, 0.0, forward_step]

    c2w = np.zeros((num_latents, 4, 4), dtype=np.float32)
    current = np.eye(4, dtype=np.float32)
    for i in range(num_latents):
        c2w[i] = current
        current = current @ delta
    return c2w


def load_runtime(base_model: str, adapter: str, *, attention_backend: str = "native"):
    """Load everything needed for generation ONCE: base model + LoRA + codec."""
    import torch

    from solarwm.backends.minimax_h3.lora import inject_h3_lora
    from solarwm.backends.minimax_h3.official_codec import OfficialH3Codec
    from solarwm.backends.minimax_h3.optional import load_conditioners, load_transformer
    from solarwm.backends.minimax_h3.runtime import _load_lora_checkpoint

    if not torch.cuda.is_available():
        raise SystemExit("ERROR: no CUDA device visible")
    device = torch.device("cuda", 0)

    model_cfg: dict[str, Any] = {
        "checkpoint_path": base_model,
        "transformer_subfolder": "transformer",
        "torch_dtype": "bfloat16",
        "attention_backend": attention_backend,
        "low_cpu_mem_usage": True,
        "adapter": {
            "rank": 384, "alpha": 384, "dropout": 0.0,
            "expected_trainable_parameters": 2_075_394_048,
        },
    }

    print("[h3-camera] loading base 33B transformer (slow)...", flush=True)
    modules = load_transformer(model_cfg, device=device)
    modules.transformer.eval().requires_grad_(False)

    print("[h3-camera] injecting LoRA-384 topology...", flush=True)
    model, lora = inject_h3_lora(
        modules.transformer, model_cfg["adapter"],
        base_identity={"note": "standalone script, not checkpoint-contract-verified"},
    )

    print(f"[h3-camera] loading trained adapter weights from {adapter}...", flush=True)
    weights_id = _load_lora_checkpoint(adapter, lora, weight_source="live", broadcast=False)
    print(f"[h3-camera] adapter loaded: {weights_id}", flush=True)

    print("[h3-camera] loading Qwen text/vision encoder + VisualVAE + AudioVAE...", flush=True)
    conditioners = load_conditioners(model_cfg, device=device, qwen=True, video_vae=True, audio_vae=True, schedulers=False)
    codec = OfficialH3Codec(
        text_encoder=conditioners.text_encoder,
        tokenizer=conditioners.tokenizer,
        processor=conditioners.processor,
        video_vae=conditioners.video_vae,
        audio_vae=conditioners.audio_vae,
        device=device,
        encoder_identity="standalone-h3-camera-infer",
    )
    return {
        "device": device, "model": model, "codec": codec,
        "conditioners": conditioners, "weights_id": weights_id,
    }


def generate_one(
    runtime: dict,
    *,
    image_path: Path,
    prompt: str,
    c2w,
    steps: int,
    seed: int,
    out: Path,
) -> None:
    """Run one generation on an already-loaded runtime. c2w is [47,4,4] float32."""
    import numpy as np
    import torch
    from PIL import Image

    from solarwm.backends.minimax_h3.artifacts import H3ArtifactBatch, align_h3_camera
    from solarwm.backends.minimax_h3.camera import WAN_FIXED_CX, WAN_FIXED_CY, WAN_FIXED_FX, WAN_FIXED_FY
    from solarwm.backends.minimax_h3.inference import package_generated
    from solarwm.backends.minimax_h3.official_codec import _pixels
    from solarwm.backends.minimax_h3.stage0p5 import H3Stage0p5Core

    device = runtime["device"]
    codec = runtime["codec"]
    conditioners = runtime["conditioners"]

    print(f"[h3-camera] encoding {image_path} + prompt...", flush=True)
    img = Image.open(image_path).convert("RGB").resize((1344, 768), Image.LANCZOS)
    frame = np.asarray(img, dtype=np.uint8)
    # _pixels() requires the full [1,3,158,768,1344] shape even though only
    # frame 0 is ever actually read below -- duplicate the single image to
    # satisfy that shape check cheaply (no extra VAE compute, just a stacked
    # array before slicing back to 1 frame).
    frames_158 = np.stack([frame] * 158, axis=0)
    prepared = _pixels(frames_158, device=device)
    with torch.inference_mode():
        anchor_latents = codec._video_latents(prepared[:, :, :1], seed=42)[0]
        prompt_embeds, text_token_tags = codec._joint_prompt(frame, prompt)

    K = np.array(
        [[WAN_FIXED_FX, 0.0, WAN_FIXED_CX], [0.0, WAN_FIXED_FY, WAN_FIXED_CY], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    camera_viewmats, camera_K = align_h3_camera({
        "camera_c2w": torch.from_numpy(np.asarray(c2w, dtype=np.float32)),
        "camera_K": torch.from_numpy(K),
    })

    batch = H3ArtifactBatch(
        sample_id=f"standalone/{out.stem}",
        start_frame=0,
        plan_fingerprint="standalone-script",
        # Never read by generate() -- training-loss/comparison-video only.
        target_latents=torch.zeros((24, 47, 48, 84), dtype=torch.bfloat16),
        anchor_latents=anchor_latents,
        prompt_embeds=prompt_embeds,
        text_token_tags=text_token_tags,
        source_frame_indices=torch.arange(158, dtype=torch.int64),
        camera_viewmats=camera_viewmats,
        camera_K=camera_K,
        source_fps=None,
    )

    print(f"[h3-camera] generating {out.name} ({steps} steps)...", flush=True)
    core = H3Stage0p5Core(runtime["model"], codec._silence_latents(), device)
    generated_latents = core.generate(batch, noise_seed=seed, num_inference_steps=steps)

    print("[h3-camera] decoding...", flush=True)
    packaged = package_generated(
        generated_latents,
        video_vae=conditioners.video_vae,
        device=device,
        weights_id=runtime["weights_id"],
        num_inference_steps=steps,
        reference_latents=None,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(packaged.artifacts["generated.mp4"])
    # Save the REAL [47,4,4] c2w that actually drove this generation --
    # lets a downstream overlay draw the genuine per-frame conditioning
    # signal instead of a fake/guessed label.
    np.save(out.with_suffix(".camera_c2w.npy"), np.asarray(c2w, dtype=np.float32))
    print(f"[h3-camera] wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", required=True, help="SolarWM-h3-33B-base directory")
    ap.add_argument("--adapter", required=True, help="SolarWM-h3-33B-bid-stage0p5-158f directory")
    ap.add_argument("--attention-backend", default="native",
                    help="diffusers attention backend (default 'native', the only backend "
                         "confirmed working on this box so far. '_flash_3' worked once with "
                         "flash-attn-4 installed but regressed to 'not usable' afterward -- "
                         "root cause not yet diagnosed, suspected sageattention install side "
                         "effect. Plain 'flash' needs classic FA2 (flash-attn), which failed to "
                         "compile on this box)")
    ap.add_argument("--image", type=Path, help="first frame")
    ap.add_argument("--prompt", help=f"default: GENERIC_PROMPT ({GENERIC_PROMPT!r})")
    ap.add_argument("--prompt-file", type=Path)
    ap.add_argument("--direction", choices=("straight", "left", "right"), default="straight")
    ap.add_argument("--yaw-deg", type=float, default=25.0, help="total yaw over the clip for left/right")
    ap.add_argument("--forward-step", type=float, default=0.05, help="per-latent-frame forward translation")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("outputs/h3-camera/generated.mp4"))
    ap.add_argument(
        "--mind-batch", type=Path, default=None,
        help="JSON manifest (list of {image, prompt, out, steps?, seed?, direction? | camera_c2w?}) -- "
             "loads the model ONCE and loops every entry. Overrides single-run args.",
    )
    args = ap.parse_args()

    import numpy as np

    if args.mind_batch:
        entries = json.loads(args.mind_batch.read_text(encoding="utf-8"))
        if not entries:
            print("[h3-camera-batch] empty manifest, nothing to do")
            return 0
        runtime = load_runtime(args.base_model, args.adapter, attention_backend=args.attention_backend)
        for i, e in enumerate(entries):
            print(f"[h3-camera-batch] {i + 1}/{len(entries)}: {e.get('out', 'sample')}", flush=True)
            if e.get("camera_c2w"):
                c2w = np.load(e["camera_c2w"])
            else:
                c2w = build_camera_c2w(
                    e.get("direction", "straight"), num_latents=47,
                    yaw_deg=float(e.get("yaw_deg", args.yaw_deg)),
                    forward_step=float(e.get("forward_step", args.forward_step)),
                )
            generate_one(
                runtime,
                image_path=Path(e["image"]),
                prompt=e.get("prompt") or GENERIC_PROMPT,
                c2w=c2w,
                steps=int(e.get("steps", args.steps)),
                seed=int(e.get("seed", args.seed)),
                out=Path(e["out"]),
            )
        print(f"[h3-camera-batch] done: {len(entries)} generated")
        return 0

    if not args.image:
        ap.error("pass --image (or use --mind-batch)")
    prompt = args.prompt
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        prompt = GENERIC_PROMPT

    runtime = load_runtime(args.base_model, args.adapter, attention_backend=args.attention_backend)
    c2w = build_camera_c2w(args.direction, num_latents=47, yaw_deg=args.yaw_deg, forward_step=args.forward_step)
    generate_one(runtime, image_path=args.image, prompt=prompt, c2w=c2w, steps=args.steps, seed=args.seed, out=args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
