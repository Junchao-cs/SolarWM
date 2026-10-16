"""Download MiniMax-H3 and prepare its local Ref2VA component layout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = "MiniMaxAI/MiniMax-H3"
NEEDED = [
    "transformer_ref/*",
    "text_encoder/*",
    "tokenizer/*",
    "processor/*",
    "vae/*",
    "audio_vae/*",
    "scheduler/*",
    "audio_scheduler/*",
    "model_index.json",
    "modular_model_index.json",
    "LICENSE",
]


def localize(directory: Path) -> None:
    index = directory / "modular_model_index.json"
    spec = json.loads(index.read_text())
    for value in spec.values():
        if isinstance(value, list) and len(value) == 3 and isinstance(value[2], dict):
            value[2]["pretrained_model_name_or_path"] = str(directory)
    index.write_text(json.dumps(spec, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument(
        "--skip-download", action="store_true", help="only rewrite the index of an existing copy"
    )
    args = parser.parse_args()
    directory = args.dir.resolve()
    if not args.skip_download:
        from huggingface_hub import snapshot_download

        snapshot_download(REPO, local_dir=str(directory), allow_patterns=NEEDED)
    localize(directory)
    print(f"MiniMax-H3 ready at {directory}")


if __name__ == "__main__":
    main()
