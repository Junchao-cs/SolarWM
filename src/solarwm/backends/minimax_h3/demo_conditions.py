"""First-image and supplied full camera tracks for the public demo collection."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from .source_length_cache import atomic_json, file_sha256, preparation_identity


def source_digest(case):
    return hashlib.sha256(
        json.dumps(
            {
                name: file_sha256(case / name)
                for name in ("source.image.png", "source.camera.npy", "source.prompt.txt")
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def stage_source(row, args):
    root = Path(args.dataset_root).resolve()
    case = Path(args.work_dir) / "cases" / row["key"]
    case.mkdir(parents=True, exist_ok=True)
    for field, name in [
        ("image", "source.image.png"),
        ("camera", "source.camera.npy"),
        ("prompt", "source.prompt.txt"),
    ]:
        source = (root / row[field]).resolve()
        if not source.is_relative_to(root) or file_sha256(source) != row[field + "_sha256"]:
            raise ValueError("Demo source identity changed")
        target = case / name
        if target.exists():
            if file_sha256(target) != row[field + "_sha256"]:
                raise ValueError("Demo cache contains another input")
        else:
            shutil.copyfile(source, target)
    return case


def prepare_demo(row, args, components, device):
    import time
    from dataclasses import asdict

    import numpy as np
    import torch
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from PIL import Image
    from safetensors.torch import save_file

    from .full_length import full_length_geometry
    from .official_codec import encode_joint_prompt_condition
    from .torch_prope import _invert_se3

    started = time.perf_counter()
    if args.config["model"]["camera_intrinsics_mode"] != "wan_fixed":
        raise ValueError("Image/camera demo without intrinsic sidecars requires wan_fixed")
    case = stage_source(row, args)
    c2w = np.load(case / "source.camera.npy", allow_pickle=False)
    frames = int(row["num_frames"])
    if c2w.dtype != np.float64 or c2w.shape != (frames, 4, 4) or not np.isfinite(c2w).all():
        raise ValueError("Demo needs its complete authoritative float64 C2W trajectory")
    np.testing.assert_allclose(c2w[:, 3], np.broadcast_to([0, 0, 0, 1], (frames, 4)), atol=1e-5)
    image = (
        Image.open(case / "source.image.png")
        .convert("RGB")
        .resize((1344, 768), Image.Resampling.LANCZOS)
    )
    image.save(case / "first.png")
    poses = torch.from_numpy(c2w).float()
    relative = _invert_se3(poses[:1]) @ poses
    relative[0] = torch.eye(4)
    views = _invert_se3(relative).contiguous()
    geometry = full_length_geometry(frames)
    idx = torch.tensor(geometry.camera_indices)
    # PRoPE replaces this placeholder with the trained fixed Wan intrinsics.
    K = (
        torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]])
        .expand(len(idx), -1, -1)
        .contiguous()
    )
    caption = (case / "source.prompt.txt").read_text().strip()
    if caption != row["caption"].strip() or not caption:
        raise ValueError("Demo caption changed")
    with torch.inference_mode():
        prompt, tags = encode_joint_prompt_condition(
            image,
            caption,
            processor=components.processor,
            tokenizer=components.tokenizer,
            text_encoder=components.text_encoder,
            device=device,
        )
        pixels = (
            torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None, :, None].to(device)
        )
        anchor = encode_vae_condition(
            components.video_vae,
            pixels,
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
            encode_seed=42,
        )
    values = dict(
        prompt_embeds=prompt.cpu().contiguous(),
        text_token_tags=tags.cpu().contiguous(),
        anchor_latents=anchor.cpu().contiguous(),
        viewmats=views[idx].contiguous(),
        K=K,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values.values()):
        raise ValueError("Non-finite demo condition")
    target = case / "condition.safetensors"
    save_file(values, str(target.with_suffix(".tmp")))
    target.with_suffix(".tmp").replace(target)
    torch.cuda.synchronize()
    atomic_json(
        case / "READY.json",
        dict(
            schema="solarwm.h3-full-condition.v1",
            row=row,
            preparation_identity=preparation_identity(row, args),
            geometry=asdict(geometry),
            condition_sha256=file_sha256(target),
            source_sha256=source_digest(case),
            preparation_seconds=time.perf_counter() - started,
            source_fps=float(row["fps"]),
            image_resize="PIL_LANCZOS_stretch_1344x768",
            model_fps=24,
            camera_convention="relative_w2c+wan_fixed_K",
            source_frame_policy="all_supplied_camera_rows_no_resampling",
            ground_truth_video_required=False,
        ),
    )
    print(f"DEMO_PREPARED key={row['key']} frames={frames}", flush=True)
