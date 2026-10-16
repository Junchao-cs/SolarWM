"""Export a training checkpoint adapter to SolarRenderer safetensors."""

import argparse
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--checkpoint", required=True, help="a checkpoint directory holding adapter.pt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from safetensors.torch import save_file

    from solarwm.backends.minimax_h3.lora import _released_pairs

    saved = torch.load(
        f"{a.checkpoint.rstrip('/')}/adapter.pt", map_location="cpu", weights_only=True
    )
    meta = saved["metadata"]
    state = {
        k.removeprefix("base_model.model."): v.to(torch.bfloat16).contiguous()
        for k, v in saved["state"].items()
    }
    pairs = _released_pairs(state)
    if any(pair["A"].shape[0] != int(meta["rank"]) for pair in pairs.values()):
        raise ValueError("checkpoint rank metadata differs from its LoRA tensors")
    if not 0 < float(meta["alpha"]) < float("inf"):
        raise ValueError("checkpoint LoRA alpha must be finite and positive")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(
        state,
        a.out,
        metadata={"format": "pt", "rank": str(meta["rank"]), "alpha": str(meta["alpha"])},
    )
    print(f"{len(state) // 2} LoRA pairs, rank {meta['rank']}, alpha {meta['alpha']} -> {a.out}")


if __name__ == "__main__":
    main()
