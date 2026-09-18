#!/usr/bin/env python3
"""Burn a WASDIJKL key-press overlay onto a generated video, synced
frame-for-frame against the action matrix that drove the generation.

Ported from H3-World's examples/overlay_keys.py, which imports its key
layout from H3-World's code/abot/abot_action.py -- that module doesn't
exist in SolarWM, so the KEY_COLS/ACTIVE_KEY_COLS/ACTION_DIM constants are
inlined here instead (same values, same column order: abot_action.py's
KEY_COLS = ["W","A","S","D","Q","E","I","J","K","L","Space"]).

SolarWM's own h3_infer.py has NO per-frame action-conditioning input at
all (prompt + first frame only, see h3_infer.py's docstring) -- there is no
real per-frame action signal to overlay for a SolarWM-generated clip. To
still use this script (rather than not overlaying anything), pass a
SYNTHETIC action matrix built by examples/racer/build_direction_actions.py:
one key (W/A/D) held for every frame, labeling which prompt variant
(straight/left/right) produced the clip -- a label, not a measured control
signal. Real per-frame overlays (e.g. for H3-World's action-conditioned
racer clip) still work exactly as before if such a matrix is provided.

Draws all 8 ACTIVE_KEY_COLS (W, A, S, D, I, J, K, L) as a small on-screen
keyboard, highlighting whichever are set (> 0) on that frame.

Run:
    python3 examples/overlay_keys.py \\
        --video outputs/racer/straight.mp4 \\
        --actions outputs/racer/straight_actions.npy \\
        --out outputs/racer/straight_overlay.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

KEY_COLS = ["W", "A", "S", "D", "Q", "E", "I", "J", "K", "L", "Space"]
ACTIVE_KEY_COLS = ["W", "A", "S", "D", "I", "J", "K", "L"]
ACTION_DIM = len(KEY_COLS) + 6  # + 3 rotation + 3 translation columns, per abot_action.py

# 3x3 grid layout: WASD on the left (movement), IJKL on the right (camera) --
# matches the physical keyboard layout these correspond to.
KEY_LAYOUT = {
    "W": (1, 0), "A": (0, 1), "S": (1, 1), "D": (2, 1),
    "I": (5, 0), "J": (4, 1), "K": (5, 1), "L": (6, 1),
}
KEY_SIZE = 28
KEY_GAP = 4
ORIGIN = (16, 16)
COLOR_OFF = (60, 60, 60, 180)
COLOR_ON = (255, 210, 0, 220)
COLOR_TEXT_OFF = (200, 200, 200, 255)
COLOR_TEXT_ON = (20, 20, 20, 255)


def draw_keys_frame(frame: np.ndarray, active_keys: set[str]) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for key, (col, row) in KEY_LAYOUT.items():
        x = ORIGIN[0] + col * (KEY_SIZE + KEY_GAP)
        y = ORIGIN[1] + row * (KEY_SIZE + KEY_GAP)
        on = key in active_keys
        fill = COLOR_ON if on else COLOR_OFF
        text_color = COLOR_TEXT_ON if on else COLOR_TEXT_OFF
        draw.rectangle([x, y, x + KEY_SIZE, y + KEY_SIZE], fill=fill, outline=(0, 0, 0, 255))
        # Centered text without needing a specific font file -- default PIL
        # bitmap font is small but always available, no extra dependency.
        bbox = draw.textbbox((0, 0), key)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text((x + (KEY_SIZE - tw) / 2, y + (KEY_SIZE - th) / 2 - bbox[1]), key, fill=text_color)
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    return np.asarray(composited)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--actions", required=True, type=Path,
                     help="[num_frames, 17] action matrix: cols 0-10 are KEY_COLS "
                          "(W,A,S,D,Q,E,I,J,K,L,Space), cols 11-16 are 3 rotation + "
                          "3 translation values. Only the 8 ACTIVE_KEY_COLS "
                          "(W,A,S,D,I,J,K,L) are drawn/highlighted -- Q, E, Space, "
                          "and the rotation/translation columns are present in the "
                          "matrix but not rendered. e.g. from "
                          "examples/racer/build_direction_actions.py")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    mat = np.load(args.actions)
    reader = iio.get_reader(str(args.video))
    fps = reader.get_meta_data().get("fps", 24)
    frames = [frame for frame in reader]
    reader.close()

    if len(frames) != mat.shape[0]:
        print(f"WARNING: video has {len(frames)} frames but actions matrix has {mat.shape[0]} rows -- "
              f"using min({len(frames)}, {mat.shape[0]}); this shouldn't happen for a video/actions pair "
              f"that came from the same infer.py run.", file=sys.stderr)
    n = min(len(frames), mat.shape[0])

    out_frames = []
    for i in range(n):
        active = {key for key in ACTIVE_KEY_COLS if mat[i, KEY_COLS.index(key)] > 0}
        out_frames.append(draw_keys_frame(frames[i], active))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(str(args.out), out_frames, fps=fps, quality=8)
    print(f"Wrote {args.out}: {n} frames @ {fps}fps")


if __name__ == "__main__":
    main()
