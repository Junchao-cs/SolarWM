"""Prepare a reference frame, image prompt, and fitted canvas for SolarRenderer.

Adapted from SolarRenderer, Copyright 2026 SolarRenderer authors, Apache-2.0.
Modified to isolate the opening shot and validate command-line arguments.
"""

import argparse
import re
import subprocess
from pathlib import Path

KEEP = (
    "Re-render the reference image with IDENTICAL camera, framing, perspective and "
    "composition: every object, "
    "building, prop, terrain shape and character stays exactly where it is, at the same "
    "size and with the same "
    "outline and pose, pixel-aligned with the reference. Every character keeps the same "
    "facing direction as in "
    "the reference (if we see their back, we still see their back), the same pose and the"
    " same size. Only replace "
    "the untextured grey or flat-coloured surfaces with final materials, lighting and detail. "
)
QUALITY = (
    " Masterful cinematic frame from a next-generation AAA game or film: physically based"
    " materials, rich "
    "micro-detail, volumetric light and atmosphere, deliberate colour grading, sharp "
    "focus on the subject. "
    "No text, no logos, no UI, no watermark, no borders."
)
# Keep stylised looks from being pulled toward photorealism by the quality prompt.
QUALITY_STYLISED = (
    " A polished, faithful frame rendered strictly in the art style described above - do "
    "not make it "
    "photorealistic, keep every hallmark of that style (its geometry, texture resolution, palette, "
    "shading and rendering artefacts). No text, no logos, no UI, no watermark, no borders."
)


def image_prompt(prompt: str, stylised: bool = False) -> str:
    """Combine the layout constraint, style, and first three sentences of the opening shot."""

    detail = prompt.split("detailed_description:", 1)[-1].split("overall_soundscape:", 1)[0]
    style, _, shot = detail.partition("[Shot 1]")
    shot = re.split(r"\[Shot \d+\]", shot, maxsplit=1)[0]
    shot = re.sub(r"^\s*The shot begins from <Video \d>\.\s*", "", shot.strip())
    opening = " ".join(re.split(r"(?<=[.!?])\s+", shot)[:3])
    return (
        KEEP
        + style.strip()
        + " The first moment of the shot: "
        + opening
        + (QUALITY_STYLISED if stylised else QUALITY)
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Prepare a SolarRenderer first frame.")
    commands = parser.add_subparsers(dest="command", required=True)
    frame = commands.add_parser("frame0", help="extract the structure clip's first frame")
    frame.add_argument("source", type=Path)
    frame.add_argument("output", type=Path)
    prompt = commands.add_parser("prompt", help="print an image prompt from a video prompt")
    prompt.add_argument("source", type=Path)
    prompt.add_argument("--stylised", action="store_true")
    fit = commands.add_parser("fit", help="crop and scale an image to the renderer canvas")
    fit.add_argument("source", type=Path)
    fit.add_argument("output", type=Path)
    fit.add_argument("canvas", nargs="?", choices=("1344x768", "768x1344"), default="1344x768")
    args = parser.parse_args(argv)
    if args.command == "frame0":
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(args.source.resolve()),
                "-frames:v",
                "1",
                str(args.output.resolve()),
            ],
            check=True,
        )
    elif args.command == "prompt":
        print(image_prompt(args.source.read_text(encoding="utf-8"), stylised=args.stylised))
    elif args.command == "fit":
        w, h = args.canvas.split("x")
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(args.source.resolve()),
                "-vf",
                f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
                "-frames:v",
                "1",
                str(args.output.resolve()),
            ],
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
