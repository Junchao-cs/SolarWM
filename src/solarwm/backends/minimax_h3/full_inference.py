"""Bucket-native, source-length H3 Stage2 SGF evaluation with SP8.

Preparation encodes only the caption/first-image condition. Generation uses
the existing student raw-KV W6 sampler and the official H3 temporal VAE.
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from .full_length import exported_fps, full_length_geometry
from .source_length_cache import (
    atomic_json,
    preparation_identity,
    prepared_case_receipt,
    result_identity,
    source_files,
    verify_completed_result,
)
from .source_length_cache import (
    file_sha256 as sha,
)


def selected_rows(args):
    from solarwm.data.index import validate_relative_key

    rows = json.loads(Path(args.plan).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Full-length inference plan must be a nonempty JSON row array")
    keys = set()
    for row in rows:
        for field in ("key", "dataset"):
            value = validate_relative_key(row[field])
            if "/" in value:
                raise ValueError("Case key and dataset must each be one path component")
        if row["key"] in keys:
            raise ValueError("Duplicate case key in inference plan")
        keys.add(row["key"])
        if row.get("input_kind", "wds") not in {"wds", "demo"}:
            raise ValueError("Unsupported source-length input_kind")
        if int(row["num_frames"]) > 960 and not args.config["inference"].get(
            "stream_decode", False
        ):
            raise ValueError("Videos longer than 960 frames require inference.stream_decode=true")
        full_length_geometry(int(row["num_frames"]))
        exported_fps(float(row["fps"]), args.fps_policy)
    return rows


def prepare(args):
    import numpy as np
    import torch
    from decord import VideoReader, cpu
    from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
    from PIL import Image
    from safetensors.torch import save_file

    from .official_codec import encode_joint_prompt_condition
    from .optional import load_conditioners
    from .raw_data import _normalise_K
    from .torch_prope import _invert_se3

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    torch.set_num_threads(2)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    pending = []
    for row in selected_rows(args)[rank::world]:
        case = Path(args.work_dir) / "cases" / row["key"]
        if (case / "READY.json").is_file():
            prepared_case_receipt(case, row, args)
            continue
        pending.append(row)
    if not pending:
        return
    components = load_conditioners(
        args.config["model"], device=device, audio_vae=False, schedulers=False
    )
    print(f"PREPARE_READY rank={rank} samples={len(pending)}", flush=True)
    for row in pending:
        if row.get("input_kind") == "demo":
            from .demo_conditions import prepare_demo

            prepare_demo(row, args, components, device)
            gc.collect()
            torch.cuda.empty_cache()
            continue
        started = time.perf_counter()
        case = source_files(row, args)
        reader = VideoReader(str(case / "source.mp4"), ctx=cpu(), num_threads=2)
        frames = len(reader)
        if frames != int(row["num_frames"]):
            raise ValueError(f"Source frame count mismatch: {frames} != {row['num_frames']}")
        fps = float(reader.get_avg_fps())
        if abs(fps - float(row["fps"])) > 0.02:
            raise ValueError(f"Source FPS mismatch: {fps} != {row['fps']}")
        first = reader[0].asnumpy()
        height, width = first.shape[:2]
        if (height, width) != (row["height"], row["width"]):
            raise ValueError("Source dimensions differ from index")
        image = Image.fromarray(first).resize((1344, 768), Image.Resampling.LANCZOS)
        image.save(case / "first.png")
        del reader, first
        geometry = full_length_geometry(frames)
        with np.load(case / "source.camera.npz", allow_pickle=False) as data:
            if "c2w" not in data:
                raise ValueError("Standalone test source must contain authoritative c2w")
            c2w = np.asarray(data["c2w"]).copy()
        if c2w.shape != (frames, 4, 4) or not np.isfinite(c2w).all():
            raise ValueError("Invalid full camera trajectory")
        np.testing.assert_allclose(c2w[:, 3], np.broadcast_to([0, 0, 0, 1], (frames, 4)), atol=1e-5)
        transform = dict(
            source_w=width,
            source_h=height,
            resized_w=1344,
            resized_h=768,
            target_w=1344,
            target_h=768,
            crop_left=0,
            crop_top=0,
        )
        source_K = np.load(case / "source.intrinsics.npy", allow_pickle=False)
        # Full test-set evaluation preserves every source sample.  With the
        # checkpoint's wan_fixed contract, attention replaces source focal
        # values before PRoPE; training's source focal-range filter is unused.
        # Still validate source shape, temporal alignment and finite values.
        K = _normalise_K(
            source_K,
            tuple(range(frames)),
            transform,
            enforce_focal_guard=args.config["model"].get("camera_intrinsics_mode") != "wan_fixed",
        )
        poses = torch.from_numpy(c2w).float()
        relative = torch.matmul(_invert_se3(poses[:1]), poses)
        relative[0] = torch.eye(4)
        views = _invert_se3(relative).contiguous()
        idx = torch.tensor(geometry.camera_indices)
        with torch.inference_mode():
            prompt, tags = encode_joint_prompt_condition(
                image,
                row["caption"],
                processor=components.processor,
                tokenizer=components.tokenizer,
                text_encoder=components.text_encoder,
                device=device,
            )
            pixels = (
                torch.from_numpy(np.asarray(image).copy())
                .permute(2, 0, 1)[None, :, None]
                .to(device)
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
            anchor_latents=anchor.detach().cpu().contiguous(),
            viewmats=views[idx].contiguous(),
            K=torch.from_numpy(K)[idx].contiguous(),
        )
        if not all(bool(torch.isfinite(t).all()) for t in values.values()):
            raise FloatingPointError("Non-finite encoded condition")
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
                condition_sha256=sha(target),
                source_sha256=sha(case / "source.mp4"),
                preparation_seconds=time.perf_counter() - started,
                source_fps=fps,
                full_camera_max_relative_translation=float(views[:, :3, 3].norm(dim=-1).max()),
                image_resize="PIL_LANCZOS_stretch_1344x768",
                model_fps=24,
                camera_convention="relative_w2c+normalized_K",
                source_frame_policy="all_consecutive_no_resampling",
            ),
        )
        print(
            f"PREPARED rank={rank} key={row['key']} frames={frames}"
            f" seconds={time.perf_counter() - started:.3f}",
            flush=True,
        )
        del values, anchor, pixels, prompt, tags
        gc.collect()
        torch.cuda.empty_cache()


def decode_write(vae, generated, geometry, output, fps, device, *, streaming=False, progress=None):
    """Use official temporal overlap blending; postprocess one CPU frame at a time."""
    import imageio.v2 as imageio
    import torch

    if streaming:
        from .stream_decode import decoded_chunks

        chunks = decoded_chunks(vae, generated[:, :, : geometry.decode_latents], device=device)
    else:
        z = generated[:, :, : geometry.decode_latents].float()
        mean = torch.tensor(vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            pixels = vae.decode(z * std + mean, return_dict=False)[0]
        if pixels.shape[2] != geometry.decoded_frames:
            raise ValueError("VAE output frame count differs")
        chunks = (pixels.cpu(),)

    def frames():
        count = 0
        for chunk in chunks:
            for frame in chunk[0].unbind(1):
                if count < geometry.source_frames:
                    yield count, frame
                count += 1
            if progress is not None:
                progress(count)
        if count != geometry.decoded_frames:
            raise ValueError(f"VAE returned {count} frames, expected {geometry.decoded_frames}")

    part = output.with_name("." + output.stem + ".tmp.mp4")
    pixel_mean = torch.tensor((0.485, 0.456, 0.406)).view(3, 1, 1)
    pixel_std = torch.tensor((0.229, 0.224, 0.225)).view(3, 1, 1)
    story_indices = sorted(
        set(round((geometry.source_frames - 1) * p) for p in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1))
    )
    story = []
    luminance = []
    with imageio.get_writer(
        str(part),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        ffmpeg_params=["-threads", "2", "-pix_fmt", "yuv420p"],
    ) as writer:
        for i, decoded in frames():
            frame = (decoded.float() * pixel_std + pixel_mean).clamp(0, 1)
            if not bool(torch.isfinite(frame).all()):
                raise FloatingPointError("Non-finite decoded frame")
            array = frame.mul(255).round().byte().permute(1, 2, 0).numpy()
            writer.append_data(array)
            if i in story_indices:
                from PIL import Image

                story.append(Image.fromarray(array).resize((448, 256)))
                luminance.append(float(array.mean()))
    part.replace(output)
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (448 * len(story), 280), "#202020")
    draw = ImageDraw.Draw(sheet)
    for j, (im, index) in enumerate(zip(story, story_indices, strict=True)):
        sheet.paste(im, (448 * j, 24))
        draw.text((448 * j + 6, 5), f"{index / fps:.2f}s / frame {index}", fill="white")
    sheet.save(output.with_suffix(".storyboard.jpg"), quality=90)
    return dict(sampled_frames=story_indices, mean_pixel_value=luminance)


def generate(args):
    import torch
    import torch.distributed as dist
    from safetensors.torch import load_file

    from .artifacts import load_silence_latents
    from .distributed import broadcast_sp_tensor as broadcast_sequence_parallel_tensor
    from .layout import patchify_video
    from .lora import inject_h3_lora
    from .optional import load_conditioners, load_transformer
    from .runtime import _base_model_load_receipt
    from .sgf_rollout import H3SGFInputs, h3_sgf_rollout
    from .torch_flow import sample_shifted_timestep, scale_noise
    from .weights import load_initial_weights

    torch.set_num_threads(2)
    local = int(os.environ["LOCAL_RANK"])
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 8:
        raise ValueError("Full test-set worker requires exactly eight GPUs, SP8")
    device = torch.device("cuda", local)
    cfg = args.config
    torch.manual_seed(42)
    modules = load_transformer(cfg["model"], device=device)
    base = _base_model_load_receipt(cfg["model"], modules.transformer)
    model, lora = inject_h3_lora(modules.transformer, cfg["model"]["adapter"], base_identity=base)
    weights_id = load_initial_weights(
        dict(path=args.checkpoint, stage="stage2", weight_source=args.weight_source), lora
    )
    step = int(weights_id.rsplit("step=", 1)[1])
    if step != args.expected_step:
        raise ValueError(f"Expected checkpoint step {args.expected_step}, found {step}")
    model.eval().requires_grad_(False)
    silence, _ = load_silence_latents(args.silence, pixel_frames=158)
    silence = (
        silence.to(device).float().unsqueeze(0).permute(0, 1, 3, 2).reshape(1, -1, 32).contiguous()
    )
    vae = (
        load_conditioners(
            cfg["model"], device=device, qwen=False, audio_vae=False, schedulers=False
        ).video_vae
        if rank == 0
        else None
    )
    if vae is not None:
        vae.set_attention_backend("native")
    print(
        f"MODEL_READY rank={rank} weight_source={args.weight_source} step={step} sp=8"
        f" memory_gib={torch.cuda.memory_allocated() / 2**30:.2f}",
        flush=True,
    )
    checkpoint_receipt_sha256 = sha(Path(args.checkpoint) / "COMPLETE.json")
    silence_sha256 = sha(Path(args.silence))
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        for row in selected_rows(args):
            dest = root / row["dataset"]
            dest.mkdir(parents=True, exist_ok=True)
            manifest = dest / (row["key"] + ".json")
            noise_seed = int(
                row.get("validation_noise_seed", row.get("validation_seed", args.noise_seed))
            )
            started = time.perf_counter()
            case = Path(args.work_dir) / "cases" / row["key"]
            condition_receipt = prepared_case_receipt(case, row, args)
            identity = result_identity(
                args,
                condition_receipt,
                checkpoint_sha256=checkpoint_receipt_sha256,
                silence_sha256=silence_sha256,
                seed=noise_seed,
            )
            if manifest.is_file():
                verify_completed_result(
                    manifest,
                    expected_identity=identity,
                    save_latents=bool(cfg["inference"].get("save_latents", False)),
                )
                continue
            values = load_file(str(case / "condition.safetensors"), device="cpu")
            from types import SimpleNamespace

            prompt, tags, anchor = (
                values["prompt_embeds"],
                values["text_token_tags"],
                values["anchor_latents"],
            )
            if prompt.ndim == 2:
                prompt = prompt.unsqueeze(0)
            if anchor.ndim == 4:
                anchor = anchor.unsqueeze(0)
            if (
                prompt.ndim != 3
                or prompt.shape[0] != 1
                or prompt.shape[-1] != 5120
                or tuple(tags.shape) != (prompt.shape[1],)
                or tuple(anchor.shape) != (1, 24, 1, 48, 84)
            ):
                raise ValueError("Invalid H3 prompt, tags or anchor shape")
            if not set(tags.tolist()) <= {0, 1} or not all(
                bool(torch.isfinite(t).all()) for t in values.values()
            ):
                raise ValueError("Invalid H3 condition values")
            condition = SimpleNamespace(
                prompt_embeds=prompt, text_token_tags=tags, anchor_latents=anchor
            )
            geometry = full_length_geometry(int(row["num_frames"]))
            rng = torch.Generator(device=device).manual_seed(noise_seed)

            def random_like(value, *, rng=rng):
                out = torch.randn(value.shape, device=device, dtype=torch.float32, generator=rng)
                broadcast_sequence_parallel_tensor(out)
                return out

            anchor = condition.anchor_latents.to(device).float()
            anchor = scale_noise(anchor, random_like(anchor), 0.999)
            audio_t = sample_shifted_timestep(1, 3.0, device, torch.float32, generator=rng).reshape(
                ()
            )
            broadcast_sequence_parallel_tensor(audio_t)
            audio = scale_noise(silence, random_like(silence), audio_t)
            views, intrinsics = (
                values["viewmats"].unsqueeze(0).to(device),
                values["K"].unsqueeze(0).to(device),
            )
            views = torch.cat((views[:, :1], views), 1)
            intrinsics = torch.cat((intrinsics[:, :1], intrinsics), 1)
            inputs = H3SGFInputs(
                condition.prompt_embeds.to(device).bfloat16(),
                condition.text_token_tags.to(device),
                patchify_video(anchor),
                audio,
                audio_t,
                views,
                intrinsics,
                48,
                84,
                0.999,
                True,
            )
            noise = random_like(torch.empty(1, 24, geometry.rollout_latents, 48, 84, device=device))
            dist.barrier()
            torch.cuda.synchronize()
            generation_start = time.perf_counter()

            def progress(chunk, *, row=row, geometry=geometry, generation_start=generation_start):
                if rank == 0:
                    torch.cuda.synchronize()
                    print(
                        f"CHUNK key={row['key']} chunk={chunk}/{geometry.rollout_latents // 5}"
                        f" seconds={time.perf_counter() - generation_start:.3f}",
                        flush=True,
                    )

            rollout = h3_sgf_rollout(
                student=model,
                inputs=inputs,
                noise=noise,
                exit_index=3,
                generator=rng,
                video_shift=12.0,
                progress=progress,
                inference_full_length=True,
                output_device="cpu" if cfg["inference"].get("stream_decode", False) else None,
            )
            torch.cuda.synchronize()
            dist.barrier()
            generation_seconds = time.perf_counter() - generation_start
            streaming = bool(cfg["inference"].get("stream_decode", False))

            def decode_progress(_count):
                heartbeat = torch.ones((), device=device, dtype=torch.int32)
                dist.broadcast(heartbeat, src=0)

            if rank != 0 and streaming:
                # Hour-long VAE/encoding can exceed the process-group timeout.
                # Participate in each bounded temporal chunk's progress event.
                while True:
                    event = torch.empty((), device=device, dtype=torch.int32)
                    dist.broadcast(event, src=0)
                    if int(event) == 0:
                        break
                    if int(event) < 0:
                        raise RuntimeError("Rank zero failed during streaming video publication")
            if rank == 0:
                try:
                    if cfg["inference"].get("save_latents", False):
                        from safetensors.torch import save_file

                        save_file(
                            {
                                "generated_latents": rollout.cache_target[
                                    :, :, : geometry.decode_latents
                                ]
                                .float()
                                .cpu()
                                .contiguous()
                            },
                            str(dest / (row["key"] + ".latents.safetensors")),
                        )
                    fps = exported_fps(float(row["fps"]), args.fps_policy)
                    output = dest / (row["key"] + ".mp4")
                    decode_start = time.perf_counter()
                    visual = decode_write(
                        vae,
                        rollout.cache_target,
                        geometry,
                        output,
                        fps,
                        device,
                        streaming=streaming,
                        progress=decode_progress if streaming else None,
                    )
                    decode_seconds = time.perf_counter() - decode_start
                    import imageio_ffmpeg

                    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
                    compare = output.with_name(output.stem + ".compare.mp4")
                    part = compare.with_name("." + compare.stem + ".tmp.mp4")
                    if row.get("input_kind") != "demo":
                        subprocess.run(
                            [
                                ffmpeg,
                                "-nostdin",
                                "-v",
                                "error",
                                "-y",
                                "-i",
                                str(case / "source.mp4"),
                                "-i",
                                str(output),
                                "-filter_complex",
                                f"[0:v]setpts=N/({fps}*TB),scale=1344:768:flags=lanczos,setsar=1[a];[1:v]setsar=1[b];[a][b]hstack",
                                "-frames:v",
                                str(geometry.source_frames),
                                "-r",
                                str(fps),
                                "-an",
                                "-c:v",
                                "libx264",
                                "-crf",
                                "20",
                                "-preset",
                                "fast",
                                "-threads",
                                "2",
                                str(part),
                            ],
                            check=True,
                        )
                        part.replace(compare)
                    from decord import VideoReader, cpu

                    for path in (output,) if row.get("input_kind") == "demo" else (output, compare):
                        actual = VideoReader(str(path), ctx=cpu(), num_threads=2)
                        if (
                            len(actual) != geometry.source_frames
                            or abs(actual.get_avg_fps() - fps) > 0.02
                        ):
                            raise ValueError("Written output frame count or FPS differs")
                        actual[len(actual) - 1]
                        del actual
                    elapsed = time.perf_counter() - started
                    atomic_json(
                        manifest,
                        dict(
                            schema="solarwm.h3-full-test-result.v1",
                            source_row=row,
                            result_identity=identity,
                            step=step,
                            weights_id=weights_id,
                            noise_seed=noise_seed,
                            checkpoint_receipt_sha256=checkpoint_receipt_sha256,
                            condition_sha256=condition_receipt["condition_sha256"],
                            weights_source=args.weight_source,
                            sp_size=8,
                            world_size=8,
                            geometry=asdict(geometry),
                            fps=fps,
                            model_fps=24,
                            fps_policy=args.fps_policy,
                            generation_seconds=generation_seconds,
                            generation_fps=geometry.source_frames / generation_seconds,
                            decode_write_seconds=decode_seconds,
                            inference_end_to_end_seconds=elapsed,
                            preparation_seconds=condition_receipt["preparation_seconds"],
                            end_to_end_seconds=elapsed + condition_receipt["preparation_seconds"],
                            end_to_end_fps=geometry.source_frames
                            / (elapsed + condition_receipt["preparation_seconds"]),
                            timing_excludes="model loading",
                            nfe=4,
                            forwards_per_chunk=5,
                            window_chunks=6,
                            chunk_latents=5,
                            context_noise=0,
                            camera_translation_transform="logd4",
                            camera_intrinsics_mode="wan_fixed",
                            student_rope_mode="sliding_local",
                            kv_cache="raw_before_rope_and_prope",
                            audio_condition_policy="fixed_noised_encoded_158f_silence_per_rollout",
                            tail_camera_policy="repeat_last_available_latent_only_for_codec_tail",
                            output_sha256=sha(output),
                            compare_sha256=sha(compare)
                            if row.get("input_kind") != "demo"
                            else None,
                            stream_decode=bool(cfg["inference"].get("stream_decode", False)),
                            storyboard_sha256=sha(output.with_suffix(".storyboard.jpg")),
                            latents_sha256=(
                                sha(dest / (row["key"] + ".latents.safetensors"))
                                if cfg["inference"].get("save_latents", False)
                                else None
                            ),
                            visual_samples=visual,
                            gpu_peak_memory_gib=torch.cuda.max_memory_allocated() / 2**30,
                        ),
                    )
                    print("RESULT " + json.dumps(json.loads(manifest.read_text())), flush=True)
                finally:
                    if streaming:
                        import sys

                        event = torch.tensor(
                            0 if sys.exc_info()[0] is None else -1, device=device, dtype=torch.int32
                        )
                        dist.broadcast(event, src=0)
            dist.barrier()
            del rollout, inputs, noise, values, anchor, audio, condition
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        dist.destroy_process_group()


def run_source_length_inference(config):
    """Public ``solarwm infer`` route: DP conditioning, then one SP8 rollout group."""
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    from .fsdp import initialize_distributed

    inference = config["inference"]
    args = SimpleNamespace(
        config=config,
        plan=inference["plan"],
        work_dir=inference["work_dir"],
        dataset_root=inference["dataset_root"],
        model_dir=config["model"]["checkpoint_path"],
        checkpoint=config["checkpoint"]["resume_from"],
        weight_source=config["checkpoint"]["weight_source"],
        silence=config["data"]["silence_latents_path"],
        output_dir=config["runtime"]["output_dir"],
        expected_step=int(inference["expected_step"]),
        fps_policy=inference["fps_policy"],
        noise_seed=int(config["validation"]["noise_seed"]),
    )
    if int(os.environ.get("RANK", "0")) == 0:
        (Path(args.output_dir) / "WORKER_COMPLETE.json").unlink(missing_ok=True)
    rows = selected_rows(args)
    if int(os.environ.get("WORLD_SIZE", 0)) != 8 or int(os.environ.get("LOCAL_WORLD_SIZE", 0)) != 8:
        raise ValueError(
            "Source-length inference requires a single node with eight torchrun workers"
        )
    local = int(os.environ["LOCAL_RANK"])
    initialize_distributed(sp_size=8, local_rank=local)
    phase = inference.get("phase", "all")
    if phase in {"prepare", "all"}:
        prepare(args)
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    if phase == "prepare":
        dist.destroy_process_group()
        return 0
    generate(args)
    if local == 0:
        atomic_json(
            Path(args.output_dir) / "WORKER_COMPLETE.json",
            {
                "schema": "solarwm.h3-source-length-worker.v1",
                "case_keys": [row["key"] for row in rows],
                "step": args.expected_step,
                "weight_source": args.weight_source,
                "entrypoint": "python -m solarwm infer",
                "config": config,
            },
        )
    return 0
