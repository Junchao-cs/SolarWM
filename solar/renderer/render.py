"""Render a structure video with SolarRenderer's control and four-step LoRAs."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import tempfile
import time
from pathlib import Path

FRAMES = 158
FPS = 24
RANK = ALPHA = 384
SILENCE = Path(__file__).parent / "assets" / "h3_silence_153_158_170.safetensors"
CAUSAL = {"mode": "teacher", "chunk": 5, "window": 6, "corrupt": 0.0}


def parse(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--h3", required=True, type=Path, help="local MiniMaxAI/MiniMax-H3 directory"
    )
    parser.add_argument("--control", required=True, type=Path, help="control LoRA safetensors")
    parser.add_argument("--fewstep", required=True, type=Path, help="four-step LoRA safetensors")
    parser.add_argument("--structure", required=True, type=Path, help="structure video")
    parser.add_argument("--first", type=Path, help="optional first-frame image")
    parser.add_argument("--prompt", required=True, type=Path, help="UTF-8 prompt text file")
    parser.add_argument("--out", required=True, type=Path, help="output MP4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--forwards", type=int, default=4, help="model evaluations per clip or causal block"
    )
    parser.add_argument("--causal", action="store_true", help="use the causal LoRA pair")
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="output duration, capped by the structure video (default: 158 frames, about 6.6 s)",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.forwards < 1:
        parser.error("--forwards must be positive")
    if not math.isfinite(args.seconds) or args.seconds < 0:
        parser.error("--seconds must be finite and nonnegative")
    if args.seconds and args.seconds < 1 / FPS:
        parser.error("--seconds must cover at least one frame at 24 fps")
    if not 0 <= args.seed < 2**63:
        parser.error("--seed must lie in [0, 2**63)")
    if args.out.suffix.lower() != ".mp4":
        parser.error("--out must end in .mp4")
    for name in ("control", "fewstep", "structure", "first", "prompt"):
        value = getattr(args, name)
        if value is not None and not value.is_file():
            parser.error(f"--{name} is not a file: {value}")
        if value is not None and value.resolve() == args.out.resolve():
            parser.error(f"--out must differ from --{name}")
    if not (args.h3 / "transformer_ref").is_dir():
        parser.error("--h3 must contain transformer_ref; run solar.renderer.prepare_h3 first")
    return args


def load_encoder(h3: str | Path, device: str, seconds: float = 0.0):
    from . import preencode

    preencode.ROOT = str(Path(h3).resolve())
    return preencode, preencode.build_pipelines(
        device, max_seconds=max(seconds + 1, 15) if seconds else None
    )


def sample_of(name: str, prompt: str, structure: str, first: str | None = None) -> dict:
    observations = [{"media": structure, "frames": "all"}]
    if first:
        observations.append({"media": first, "frames": [0]})
    return {"id": name, "target": structure, "text": prompt, "observations": observations}


def encode_sample(encoder, sample: dict, member: str, frames: int = FRAMES) -> None:
    from safetensors.torch import save_file

    preencode, pipelines = encoder
    tensors, metadata = preencode.encode(sample, pipelines, frames)
    save_file(tensors, member, metadata={"omni": json.dumps(metadata)})


def clip_frames(seconds: float, held: int) -> tuple[int, int]:
    """Return the H3-aligned encoding length and the requested output frame count."""
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("seconds must be finite and nonnegative")
    if held < FRAMES:
        raise ValueError(f"the structure video must contain at least {FRAMES} frames at {FPS} fps")
    if seconds and seconds < 1 / FPS:
        raise ValueError("seconds must cover at least one frame at 24 fps")
    keep = min(round(min(seconds, held / FPS) * FPS), held) if seconds else FRAMES
    encoded = max(min(17 * ((keep - 5 + 16) // 17) + 5, 17 * ((held - 5) // 17) + 5), FRAMES)
    return encoded, min(keep, encoded)


def free_cuda() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def lora_pairs(path: str | Path) -> tuple[dict, int, float]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    from solar.backends.minimax_h3.lora import _released_pairs

    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata() or {}
    state = load_file(str(path))
    pairs = _released_pairs(state)
    ranks = {int(pair["A"].shape[0]) for pair in pairs.values()}
    if len(ranks) != 1:
        raise ValueError("SolarRenderer requires a single positive LoRA rank")
    rank = ranks.pop()
    if rank <= 0 or int(metadata.get("rank", rank)) != rank:
        raise ValueError("LoRA rank metadata does not match its tensors")
    alpha = float(metadata.get("alpha", rank))
    if not 0.0 < alpha < float("inf"):
        raise ValueError("LoRA alpha must be finite and positive")
    return state, rank, alpha


def stack_lora_pairs(*adapters: tuple[dict, int, float]) -> dict:
    """Stack the released adapters so their scaled deltas are merged in one FP32 operation."""
    import torch

    stacked = {}
    for state, rank, alpha in adapters:
        for key, value in state.items():
            is_down = key.endswith("lora_A.weight")
            value = value.float() if is_down else value.float() * (alpha / rank)
            stacked[key] = (
                torch.cat((stacked[key], value), dim=0 if is_down else 1)
                if key in stacked
                else value
            )
    return stacked


def load_renderer(
    h3: str | Path,
    control: str | Path,
    fewstep: str | Path,
    device: str,
    causal: bool = False,
    frames: int = FRAMES,
):
    """Load Ref2VA with both LoRAs merged into its weights, ready for up to ``frames``."""
    import torch

    from solar.backends.minimax_h3.artifacts import load_silence_latents
    from solar.backends.minimax_h3.lora import merge_lora_delta
    from solar.backends.minimax_h3.optional import load_conditioners, load_transformer

    from .core import H3Stage0p5Core
    from .observations import observation_profile_from_train_config
    from .silence import extend_silence

    if frames < FRAMES or (frames - 5) % 17:
        raise ValueError("renderer frames must be 17n + 5 and at least 158")
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("SolarRenderer inference requires a CUDA device")
    torch.cuda.set_device(device)
    control_state, control_rank, control_alpha = lora_pairs(control)
    fewstep_state, rank, alpha = lora_pairs(fewstep)
    if control_rank != RANK:
        raise ValueError(f"the control LoRA must have rank {RANK}")
    stacked = stack_lora_pairs(
        (fewstep_state, rank, alpha), (control_state, control_rank, control_alpha)
    )
    del fewstep_state, control_state
    model_cfg = {
        "checkpoint_path": str(Path(h3).resolve()),
        "transformer_subfolder": "transformer_ref",
        "torch_dtype": "bfloat16",
        "attention_backend": "flash",
        "adapter": {"rank": RANK, "alpha": ALPHA, "dropout": 0.0},
    }
    modules = load_transformer(model_cfg, device=device)
    merge_lora_delta(modules.transformer, stacked, alpha=None, rank=None)
    del stacked
    wrapped = modules.transformer
    wrapped.eval().requires_grad_(False)
    conditioners = load_conditioners(
        model_cfg, device=device, qwen=False, video_vae=True, audio_vae=False, schedulers=False
    )
    silence, _ = load_silence_latents(str(SILENCE), pixel_frames=FRAMES)
    if frames > FRAMES:
        silence = extend_silence(
            load_silence_latents(str(SILENCE), pixel_frames=170)[0],
            H3Stage0p5Core.audio_latents(5 * ((frames - 5) // 17) + 2),
        )
    options = {"observations": observation_profile_from_train_config({})}
    if causal:
        from .causal import H3OmniCausalCore

        core = H3OmniCausalCore(wrapped, silence, device, dict(CAUSAL), **options)
    else:
        core = H3Stage0p5Core(wrapped, silence, device, **options)
    return core, conditioners, device


def render_member(
    renderer, member: str, out: str | Path, seed: int = 0, forwards: int = 4, keep: int = 0
) -> float:
    import torch

    from solar.backends.minimax_h3.inference import _mp4_bytes, decode_h3_latents

    from .data import H3OmniReader

    if forwards < 1:
        raise ValueError("forwards must be positive")
    if keep < 0:
        raise ValueError("keep must be nonnegative")
    core, conditioners, device = renderer
    output = Path(out)
    batch = H3OmniReader("/").read({"id": output.stem, "member": str(Path(member).resolve())})
    began = time.perf_counter()
    with torch.no_grad():
        if hasattr(core, "stream"):
            from solar.backends.minimax_h3.flow import make_shifted_schedule

            latents, *_ = core.stream(
                batch,
                generator=torch.Generator(device=device).manual_seed(seed),
                schedule=make_shifted_schedule(forwards + 1, shift=12.0),
                exit_index=forwards - 1,
                cache_structure=False,
            )
        else:
            latents = core.generate(
                batch, noise_seed=seed, num_inference_steps=forwards + 1, renoise=True
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - began
    decoded = decode_h3_latents(conditioners.video_vae, latents, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(_mp4_bytes(decoded[:, :, : keep or None]))
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return elapsed


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    from . import preencode

    prompt = args.prompt.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("the prompt file is empty")
    held = int(
        preencode.frame_count(str(args.structure)) * FPS / preencode.frame_rate(str(args.structure))
    )
    frames, keep = clip_frames(args.seconds, held)
    with tempfile.TemporaryDirectory() as work:
        began = time.perf_counter()
        encoder = load_encoder(args.h3, args.device, seconds=args.seconds)
        member = str(Path(work) / "sample.safetensors")
        sample = sample_of(
            args.out.stem, prompt, str(args.structure), str(args.first) if args.first else None
        )
        encode_sample(encoder, sample, member, frames=frames)
        del encoder
        free_cuda()
        print(f"Encoded {frames} frames in {time.perf_counter() - began:.0f} s", flush=True)
        renderer = load_renderer(
            args.h3, args.control, args.fewstep, args.device, causal=args.causal, frames=frames
        )
        elapsed = render_member(renderer, member, args.out, args.seed, args.forwards, keep=keep)
    print(
        f"Sampled {args.forwards} steps in {elapsed:.0f} s; "
        f"wrote {keep} frames ({keep / FPS:.1f} s) to {args.out}",
        flush=True,
    )


if __name__ == "__main__":
    main()
