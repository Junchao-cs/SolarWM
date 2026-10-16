"""Render a structure video with SolarRenderer's control and four-step LoRAs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import tempfile
import time
from pathlib import Path

FRAMES = 158
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
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.forwards < 1:
        parser.error("--forwards must be positive")
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
        parser.error("--h3 must contain transformer_ref; run solarwm.renderer.prepare_h3 first")
    return args


def load_encoder(h3: str | Path, device: str):
    from . import preencode

    preencode.ROOT = str(Path(h3).resolve())
    return preencode, preencode.build_pipelines(device)


def sample_of(name: str, prompt: str, structure: str, first: str | None = None) -> dict:
    observations = [{"media": structure, "frames": "all"}]
    if first:
        observations.append({"media": first, "frames": [0]})
    return {"id": name, "target": structure, "text": prompt, "observations": observations}


def encode_sample(encoder, sample: dict, member: str) -> None:
    from safetensors.torch import save_file

    preencode, pipelines = encoder
    tensors, metadata = preencode.encode(sample, pipelines, FRAMES)
    save_file(tensors, member, metadata={"omni": json.dumps(metadata)})


def free_cuda() -> None:
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def lora_pairs(path: str | Path) -> tuple[dict, int, float]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    from solarwm.backends.minimax_h3.lora import _released_pairs

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


def load_renderer(
    h3: str | Path, control: str | Path, fewstep: str | Path, device: str, causal: bool = False
):
    import torch

    from solarwm.backends.minimax_h3.artifacts import load_silence_latents
    from solarwm.backends.minimax_h3.lora import inject_h3_lora, merge_lora_delta
    from solarwm.backends.minimax_h3.optional import load_conditioners, load_transformer

    from .core import H3Stage0p5Core
    from .observations import observation_profile_from_train_config

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("SolarRenderer inference requires a CUDA device")
    torch.cuda.set_device(device)
    control_state, control_rank, control_alpha = lora_pairs(control)
    fewstep_state, rank, alpha = lora_pairs(fewstep)
    if control_rank != RANK:
        raise ValueError(f"the control LoRA must have rank {RANK}")
    model_cfg = {
        "checkpoint_path": str(Path(h3).resolve()),
        "transformer_subfolder": "transformer_ref",
        "torch_dtype": "bfloat16",
        "attention_backend": "flash",
        "adapter": {"rank": RANK, "alpha": ALPHA, "dropout": 0.0},
    }
    modules = load_transformer(model_cfg, device=device)
    merge_lora_delta(modules.transformer, fewstep_state, alpha=alpha, rank=rank)
    del fewstep_state
    wrapped, lora = inject_h3_lora(modules.transformer, model_cfg["adapter"], base_identity={})
    control_state = {
        f"base_model.model.{key}": (value.float() * (control_alpha / ALPHA)).to(torch.bfloat16)
        if key.endswith("lora_B.weight")
        else value.to(torch.bfloat16)
        for key, value in control_state.items()
    }
    lora.load_state_dict(control_state, broadcast=False)
    wrapped.eval().requires_grad_(False)
    conditioners = load_conditioners(
        model_cfg, device=device, qwen=False, video_vae=True, audio_vae=False, schedulers=False
    )
    silence, _ = load_silence_latents(str(SILENCE), pixel_frames=FRAMES)
    options = {"observations": observation_profile_from_train_config({})}
    if causal:
        from .causal import H3OmniCausalCore

        core = H3OmniCausalCore(wrapped, silence, device, dict(CAUSAL), **options)
    else:
        core = H3Stage0p5Core(wrapped, silence, device, **options)
    return core, conditioners, device


def render_member(
    renderer, member: str, out: str | Path, seed: int = 0, forwards: int = 4
) -> float:
    import torch

    from solarwm.backends.minimax_h3.inference import _mp4_bytes, decode_h3_latents

    from .data import H3OmniReader

    if forwards < 1:
        raise ValueError("forwards must be positive")
    core, conditioners, device = renderer
    output = Path(out)
    batch = H3OmniReader("/").read({"id": output.stem, "member": str(Path(member).resolve())})
    began = time.perf_counter()
    with torch.no_grad():
        if hasattr(core, "stream"):
            from solarwm.backends.minimax_h3.flow import make_shifted_schedule

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
            stream.write(_mp4_bytes(decoded))
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return elapsed


def main(argv: list[str] | None = None) -> None:
    args = parse(argv)
    prompt = args.prompt.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("the prompt file is empty")
    with tempfile.TemporaryDirectory() as work:
        began = time.perf_counter()
        encoder = load_encoder(args.h3, args.device)
        member = str(Path(work) / "sample.safetensors")
        sample = sample_of(
            args.out.stem, prompt, str(args.structure), str(args.first) if args.first else None
        )
        encode_sample(encoder, sample, member)
        del encoder
        free_cuda()
        print(f"Encoded in {time.perf_counter() - began:.0f} s", flush=True)
        renderer = load_renderer(
            args.h3, args.control, args.fewstep, args.device, causal=args.causal
        )
        elapsed = render_member(renderer, member, args.out, args.seed, args.forwards)
    print(f"Sampled {args.forwards} steps in {elapsed:.0f} s; wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
