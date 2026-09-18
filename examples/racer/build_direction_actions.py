#!/usr/bin/env python3
"""Build a synthetic [num_frames, 17] action matrix for one SolarWM racer
clip, so examples/overlay_keys.py has something real to draw from.

SolarWM's h3_infer.py has no per-frame action-conditioning input at all
(prompt + first frame only) -- there is no genuine per-frame steering
signal to overlay. This just holds ONE key (W for straight, A for left,
D for right) for every frame, matching the direction that prompt.txt /
prompt_left.txt / prompt_right.txt asked for. It's a label, not a measured
control signal -- same column layout as H3-World's abot_action.py
(KEY_COLS = W,A,S,D,Q,E,I,J,K,L,Space + 3 rotation + 3 translation) so the
same overlay_keys.py can draw it.

Run:
    python3 examples/racer/build_direction_actions.py \\
        --direction left --num-frames 158 \\
        --out outputs/racer/left_actions.npy
"""
import argparse
from pathlib import Path

import numpy as np

KEY_COLS = ["W", "A", "S", "D", "Q", "E", "I", "J", "K", "L", "Space"]
ACTION_DIM = len(KEY_COLS) + 6
DIRECTION_KEY = {"straight": "W", "left": "A", "right": "D"}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--direction", required=True, choices=tuple(DIRECTION_KEY))
ap.add_argument("--num-frames", type=int, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

mat = np.zeros((args.num_frames, ACTION_DIM), dtype=np.float32)
mat[:, KEY_COLS.index(DIRECTION_KEY[args.direction])] = 1.0

args.out.parent.mkdir(parents=True, exist_ok=True)
np.save(args.out, mat)
print(f"Wrote {args.out}: shape {mat.shape} (direction={args.direction}, key={DIRECTION_KEY[args.direction]!r})")
