#!/usr/bin/env python
"""Tables from eval/solar_renderer/metrics.py results: the mean of every reading per method, and the automatic score.

    python eval/solar_renderer/summary.py --scores ./scores --methods mine,other [--split hard|heldout] [--tag "unseen condition"]

Readings (lower is better for flicker and motion error), in four groups:
  quality      DOVER overall, LAION aesthetic, VideoScore2 visual quality (expected)
  stability    low-pass warp residual, DINOv2 adjacent-frame cosine
  structure    CoTracker3 motion error, edge F1
  first frame  DINOv2 cosine to the first frame over the last second
Held-out split: also DISTS and LPIPS against the real video (eval/solar_renderer/gt_metrics.py), when present.

Automatic score (0-100), the score of the paper: each reading is mapped linearly from a fixed range - DOVER, edge F1,
DINOv2 adjacent-frame and first-frame similarity from [0, 1], the aesthetic score from [1, 10], VideoScore2 from
[1, 5], flicker from [0, 10] and motion error from [0, 0.5], both reversed and clipped at 0 - the readings of each
group are averaged, and the score is the mean of the four groups. It uses each method's means over the cases, so it
does not depend on the other methods. Methods are compared on the cases every listed method has.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

READ = {  # name: (metric directory, getter, group)
    "dover_overall": ("dover", lambda j: j["dover"]["overall"], "quality"),
    "laion_aes": ("laion_aes", lambda j: j["laion_aes"], "quality"),
    "vs2_visual": ("videoscore2", lambda j: j.get("visual_quality_expected"), "quality"),
    "warp_l1_lp": ("flicker", lambda j: j["warp_l1_lp"], "stability"),
    "dino_adjacent": ("dino", lambda j: j["adjacent_cos"], "stability"),
    "motion_err": ("motion", lambda j: j["err_mean"], "structure"),
    "edge_f1": ("edges", lambda j: j["F1"], "structure"),
    "first_last1s": ("firstframe", lambda j: j["last1s_cos"], "first_frame"),
    "dists_real": ("gt", lambda j: j["dists_real"], "ground_truth"),
    "lpips_real": ("gt", lambda j: j["lpips_real"], "ground_truth"),
}
SCALE = {  # reading -> 0-100
    "dover_overall": lambda x: 100 * x,
    "laion_aes": lambda x: 100 * (x - 1) / 9,
    "vs2_visual": lambda x: 100 * (x - 1) / 4,
    "warp_l1_lp": lambda x: 100 * max(0.0, 1 - x / 10),
    "dino_adjacent": lambda x: 100 * x,
    "motion_err": lambda x: 100 * max(0.0, 1 - x / 0.5),
    "edge_f1": lambda x: 100 * x,
    "first_last1s": lambda x: 100 * x,
}
GROUPS = ("quality", "stability", "structure", "first_frame")


def score(means: dict[str, float]) -> tuple[float, ...]:
    """(score, quality, stability, structure, first frame) from a method's mean readings."""
    groups = [[SCALE[n](means[n]) for n in SCALE if READ[n][2] == g] for g in GROUPS]
    groups = [sum(v) / len(v) for v in groups]
    return (sum(groups) / len(groups), *groups)


def load(
    scores: Path, method: str, split: str, tag: str | None = None
) -> dict[str, dict[str, float]]:
    """case id -> reading -> value"""
    out: dict[str, dict[str, float]] = {}
    for name, (metric, get, _) in READ.items():
        root = scores / metric / method / split
        for p in root.rglob("*.json") if root.exists() else []:
            try:
                j = json.loads(p.read_text())
                v = get(j)
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
            if tag and metric != "gt" and tag not in (j.get("tags") or []):
                continue
            if v is not None:
                out.setdefault(str(p.relative_to(root))[:-5], {})[name] = float(v)
    if tag:  # the gt readings carry no tags; keep the cases the tagged readings selected
        out = {k: v for k, v in out.items() if set(v) - {"dists_real", "lpips_real"}}
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--scores", required=True, type=Path)
    ap.add_argument("--methods", required=True, help="comma list")
    ap.add_argument("--split", default="hard", choices=("hard", "heldout"))
    ap.add_argument(
        "--tag", help='hard split only: the cases with this tag, e.g. "unseen condition"'
    )
    a = ap.parse_args()
    methods = [m for m in a.methods.split(",") if m]
    data = {m: load(a.scores, m, a.split, a.tag) for m in methods}
    need = [n for n, (_, _, g) in READ.items() if g != "ground_truth" or a.split == "heldout"]
    common = set.intersection(
        *({k for k, v in d.items() if all(n in v for n in SCALE)} for d in data.values())
    )
    if not common:
        raise SystemExit("no case scored for every method")
    print(f"{a.split}{', ' + a.tag if a.tag else ''}: {len(common)} cases every method has\n")
    print(
        f"{'method':24s} {'score':>6s} {'qual':>6s} {'stab':>6s} {'struct':>6s} {'first':>6s}  "
        + " ".join(f"{n:>13s}" for n in need)
    )
    rows = []
    for m, d in data.items():
        means = {}
        for n in need:
            v = [d[k][n] for k in common if n in d[k]]
            if v:
                means[n] = sum(v) / len(v)
        rows.append((score(means), m, means))
    for sc, m, means in sorted(rows, key=lambda r: -r[0][0]):
        print(
            f"{m:24s} "
            + " ".join(f"{x:6.1f}" for x in sc)
            + "  "
            + " ".join(f"{means[n]:13.4f}" if n in means else f"{'-':>13s}" for n in need)
        )


if __name__ == "__main__":
    main()
