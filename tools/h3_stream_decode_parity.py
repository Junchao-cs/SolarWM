"""Compare streaming with the real official VAE, including temporal boundaries."""

import argparse
import json

import torch
from diffusers import AutoencoderKLMiniMaxH3

from solarwm.backends.minimax_h3.stream_decode import decoded_chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(2)
    vae = AutoencoderKLMiniMaxH3.from_pretrained(
        args.model_dir, subfolder="vae", torch_dtype=torch.float32
    ).to(args.device)
    vae.set_attention_backend("native")
    mean = torch.tensor(vae.config.latents_mean, device=args.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=args.device).view(1, -1, 1, 1, 1)
    results = []
    # Whole and partial temporal chunks, then a real 768p/158f decode.
    for frames, height, width in [
        (7, 16, 16),
        (8, 16, 16),
        (9, 16, 16),
        (10, 16, 16),
        (11, 16, 16),
        (47, 48, 84),
    ]:
        z = torch.randn(
            1, 24, frames, height, width, generator=torch.Generator().manual_seed(42)
        ).to(args.device)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            reference = vae.decode(z * std + mean, return_dict=False)[0].cpu()
        actual = torch.cat(list(decoded_chunks(vae, z.cpu(), device=args.device)), dim=2)
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
        results.append(
            dict(
                latents=frames,
                height=height,
                width=width,
                frames=actual.shape[2],
                bitwise_equal=True,
            )
        )
        print("H3_STREAM_DECODE_CASE_PASS", json.dumps(results[-1]), flush=True)
        del actual, reference, z
        torch.cuda.empty_cache()
    from pathlib import Path

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2))
    print("H3_STREAM_DECODE_BITWISE_PASS", json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
