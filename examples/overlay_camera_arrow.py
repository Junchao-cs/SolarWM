#!/usr/bin/env python3
"""Burn a heading arrow onto a generated H3 video, driven by the REAL
camera_c2w trajectory that drove generation (h3_camera_infer.py saves this
as <out>.camera_c2w.npy next to every clip it writes) -- not a fake/guessed
direction label like the old overlay_keys.py flow used before real camera
conditioning existed.

The arrow shows CUMULATIVE yaw relative to frame 0: extracted from each
[4,4] c2w's rotation block via atan2(R[0,2], R[2,2]), which recovers the
same yaw angle build_camera_c2w()/_compose_c2w() composed it from (a
rotation about Y). This assumes the trajectory's rotation is primarily
yaw (steering-style turns) -- true for this project's straight/left/right
paths and for MIND's mind_actions_to_camera_c2w() output, not a general
claim for arbitrary 6DoF camera motion.

camera_c2w has 47 rows (H3's real latent-frame count); the decoded video
may have a different (larger) frame count, so each video frame index is
mapped proportionally into the 47-row trajectory rather than assuming a
fixed ratio.

Run:
    python3 examples/overlay_camera_arrow.py \\
        --video outputs/h3/racer_right.mp4 \\
        --camera-c2w outputs/h3/racer_right.camera_c2w.npy \\
        --out outputs/h3/racer_right_overlay.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

ARROW_CENTER = (48, 48)
ARROW_LEN = 32
ARROW_COLOR = (255, 210, 0, 230)
DIAL_COLOR = (40, 40, 40, 160)
DIAL_RADIUS = 40


def yaw_from_c2w(c2w: np.ndarray) -> np.ndarray:
    """[N,4,4] -> [N] cumulative yaw (radians) relative to frame 0."""
    r02 = c2w[:, 0, 2]
    r22 = c2w[:, 2, 2]
    return np.arctan2(r02, r22)


def draw_arrow_frame(frame: np.ndarray, yaw_rad: float) -> np.ndarray:
    img = Image.fromarray(frame).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = ARROW_CENTER
    draw.ellipse(
        [cx - DIAL_RADIUS, cy - DIAL_RADIUS, cx + DIAL_RADIUS, cy + DIAL_RADIUS],
        fill=DIAL_COLOR, outline=(0, 0, 0, 255),
    )
    # Screen convention: 0 yaw points "up" (forward), positive yaw sweeps
    # toward +x (right side of screen) -- matches how a top-down heading
    # dial is normally read, independent of the world-space sign convention
    # question flagged in h3_camera_infer.py's docstring.
    tip = (cx + ARROW_LEN * np.sin(yaw_rad), cy - ARROW_LEN * np.cos(yaw_rad))
    left = (cx + (ARROW_LEN * 0.35) * np.sin(yaw_rad + 2.6), cy - (ARROW_LEN * 0.35) * np.cos(yaw_rad + 2.6))
    right = (cx + (ARROW_LEN * 0.35) * np.sin(yaw_rad - 2.6), cy - (ARROW_LEN * 0.35) * np.cos(yaw_rad - 2.6))
    draw.polygon([tip, left, right], fill=ARROW_COLOR, outline=(0, 0, 0, 255))
    deg = np.degrees(yaw_rad)
    draw.text((cx - 14, cy + DIAL_RADIUS + 4), f"{deg:+.1f}deg", fill=(255, 255, 255, 255))
    composited = Image.alpha_composite(img, overlay).convert("RGB")
    return np.asarray(composited)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--camera-c2w", required=True, type=Path, help="[47,4,4] real trajectory .npy")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    c2w = np.load(args.camera_c2w)
    yaw = yaw_from_c2w(c2w)

    reader = iio.get_reader(str(args.video))
    fps = reader.get_meta_data().get("fps", 24)
    frames = [frame for frame in reader]
    reader.close()

    n = len(frames)
    # Proportional index mapping: video frame i -> trajectory row
    # round(i * (len(c2w)-1) / (n-1)), so it's correct regardless of the
    # VAE's temporal upsampling ratio between latent frames and decoded
    # video frames.
    idx = np.round(np.linspace(0, len(yaw) - 1, n)).astype(int) if n > 1 else np.array([0])

    out_frames = [draw_arrow_frame(frames[i], float(yaw[idx[i]])) for i in range(n)]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    iio.mimwrite(str(args.out), out_frames, fps=fps, quality=8)
    print(f"Wrote {args.out}: {n} frames @ {fps}fps")


if __name__ == "__main__":
    main()
