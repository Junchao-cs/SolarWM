"""Convert the public DMAD MiniMax-H3 LoRA to validated SolarRenderer tensor keys."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def base_shapes(h3: Path) -> dict[str, tuple[int, ...]]:
    from safetensors import safe_open

    root = h3 / "transformer_ref"
    shards = sorted(
        set(
            json.loads((root / "diffusion_pytorch_model.safetensors.index.json").read_text())[
                "weight_map"
            ].values()
        )
    )
    shapes = {}
    for shard in shards:
        with safe_open(str(root / shard), framework="pt") as handle:
            keys = handle.keys()
            for key in keys:
                if key.endswith(".weight"):
                    shapes[key[: -len(".weight")]] = tuple(handle.get_slice(key).get_shape())
    return shapes


def convert(src: str) -> tuple[dict, float | None]:
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(src, framework="pt") as handle:
        meta = handle.metadata() or {}
    out, alphas = {}, set()
    for key, value in load_file(src).items():
        key = key.removeprefix("diffusion_model.").removeprefix("transformer.")
        if key.endswith(".alpha"):
            alphas.add(float(value))
            continue
        key = (
            key.replace(".default.weight", ".weight")
            .replace(".lora.down.weight", ".lora_A.weight")
            .replace(".lora.up.weight", ".lora_B.weight")
        )
        if not key.endswith((".lora_A.weight", ".lora_B.weight")):
            raise ValueError(f"unexpected key {key}")
        if key in out:
            raise ValueError(f"multiple source tensors normalize to {key}")
        out[key] = value.contiguous()
    named = meta.get("alpha", meta.get("lora_alpha"))
    if named is not None:
        alphas.add(float(named))
    if len(alphas) > 1:
        raise ValueError("DMAD LoRA has inconsistent alpha values")
    alpha = alphas.pop() if alphas else None
    if alpha is not None and not 0 < alpha < float("inf"):
        raise ValueError("DMAD LoRA alpha must be finite and positive")
    from solarwm.backends.minimax_h3.lora import _released_pairs

    _released_pairs(out)
    return out, alpha


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--h3", required=True, type=Path)
    parser.add_argument("--src", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    from safetensors.torch import save_file

    state, alpha = convert(args.src)
    shapes = base_shapes(args.h3.resolve())
    modules = {key.rsplit(".lora_", 1)[0] for key in state}
    for module in sorted(modules):
        a, b = state[f"{module}.lora_A.weight"], state[f"{module}.lora_B.weight"]
        if shapes.get(module) != (b.shape[0], a.shape[1]) or a.shape[0] != b.shape[1]:
            raise ValueError(
                f"{module}: LoRA {tuple(b.shape)} x {tuple(a.shape)} "
                f"does not fit {shapes.get(module)}"
            )
    metadata = {
        "rank": str(min(int(v.shape[0]) for k, v in state.items() if k.endswith("lora_A.weight")))
    }
    if alpha is not None:
        metadata["alpha"] = str(alpha)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    save_file(state, args.out, metadata=metadata)
    print(
        f"{len(modules)} modules, rank {metadata['rank']}, "
        f"alpha {metadata.get('alpha', 'rank')} -> {args.out}"
    )


if __name__ == "__main__":
    main()
