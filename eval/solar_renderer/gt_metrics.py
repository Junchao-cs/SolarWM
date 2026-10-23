#!/usr/bin/env python
"""Pixel metrics against the ground truth, for the held-out split.

    python eval/solar_renderer/gt_metrics.py --bench ./SolarRenderer-Data --renders ./renders/mymethod --method mymethod --out ./scores

Writes <out>/gt/<method>/heldout/<clip_id>.json. Render and ground truth are decoded at the ground truth's size
and cut to the shorter. Readings (lower is closer for DISTS / LPIPS, higher for PSNR / SSIM):
  dists_real / lpips_real   DISTS and LPIPS (AlexNet), 4 fps, 672 px wide, mean over frames
  dists_hr / lpips_hr       the same on the four full-size quarters of every frame (2x2 tiles of 672x384)
  psnr                      per-frame 10 log10(255^2 / MSE) on full-size RGB, all frames, averaged
  ssim                      per-pixel greyscale SSIM (11x11 Gaussian, sigma 1.5), all frames, averaged
  sharp_log / sharp_log_hr  log of (mean Laplacian variance of the render / of the ground truth), grey, 8 fps,
                            672 px and full size: 0 is the ground truth's sharpness
Each reading is also given without frame 0 (suffix _nof0): frame 0 is the given first frame, so it only measures how
faithfully that frame is copied.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

W = 672


def size_of(path):
    probe = (
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "csv=p=0",
                path,
            ],
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split(",")
    )
    return int(probe[0]), int(probe[1])


def decode(path, fps, gray, size):
    """Frames at fps over the whole clip at size (w, h): uint8 [T,H,W] (grey) or [T,H,W,3]."""
    w, h = size
    chain = f"fps={fps},scale={w}:{h}:flags=area,format={'gray' if gray else 'rgb24'}"
    raw = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            path,
            "-vf",
            chain,
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    c = 1 if gray else 3
    a = np.frombuffer(raw, np.uint8)
    n = len(a) // (w * h * c)
    return a[: n * w * h * c].reshape(n, h, w) if gray else a[: n * w * h * c].reshape(n, h, w, 3)


def sizes(real):
    w, h = size_of(real)
    return (W, int(round(h * W / w / 2) * 2)), (w, h)


def sharpness(gray):
    f = gray.astype(np.float32)
    lap = (
        -4 * f[:, 1:-1, 1:-1] + f[:, :-2, 1:-1] + f[:, 2:, 1:-1] + f[:, 1:-1, :-2] + f[:, 1:-1, 2:]
    )
    return float(lap.var(axis=(1, 2)).mean())


def to_tensor(rgb, device):
    return torch.from_numpy(rgb).to(device).permute(0, 3, 1, 2).float().div(255.0)


@torch.no_grad()
def distances(a, b, metrics, device, tiles=False):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if tiles:  # the four quarters of every frame, each as its own picture
        h, w = a.shape[1] // 2, a.shape[2] // 2

        def cut(v):
            return np.concatenate(
                [v[:, i * h : (i + 1) * h, j * w : (j + 1) * w] for i in range(2) for j in range(2)]
            )

        a, b = cut(a), cut(b)
    out = {}
    for name, m in metrics.items():
        vals = []
        for i in range(0, len(a), 8):
            x, y = (
                to_tensor(np.ascontiguousarray(a[i : i + 8]), device),
                to_tensor(np.ascontiguousarray(b[i : i + 8]), device),
            )
            vals.append(m(x, y).flatten().float().cpu())
        out[name] = float(torch.cat(vals).mean())
    return out


def real_readings(real):
    lo, hi = sizes(real)
    return {
        "lo": lo,
        "hi": hi,
        "sharp": sharpness(decode(real, 8, True, lo)),
        "sharp_hr": sharpness(decode(real, 8, True, hi)),
        "rgb": decode(real, 4, False, lo),
        "rgb_hr": decode(real, 4, False, hi),
    }


def _gaussian(window: int = 11, sigma: float = 1.5) -> torch.Tensor:
    axis = torch.arange(window, dtype=torch.float32) - (window - 1) / 2
    kernel = torch.exp(-(axis**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return (kernel[:, None] @ kernel[None, :])[None, None]


def ssim(made: torch.Tensor, want: torch.Tensor) -> torch.Tensor:
    window = _gaussian().to(made.device, made.dtype)
    pad = window.shape[-1] // 2
    blur = lambda x: torch.nn.functional.conv2d(x, window, padding=pad)  # noqa: E731
    mu_a, mu_b = blur(made), blur(want)
    var_a = blur(made * made) - mu_a * mu_a
    var_b = blur(want * want) - mu_b * mu_b
    covariance = blur(made * want) - mu_a * mu_b
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    return ((2 * mu_a * mu_b + c1) * (2 * covariance + c2)) / (
        (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
    )


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * np.log10(255.0**2 / max(mse, 1e-6))


def pixel(made, real, device):
    hi = size_of(real)
    a, b = decode(made, 24, False, hi), decode(real, 24, False, hi)
    n = min(len(a), len(b))
    ps, ss = [], []
    w = torch.tensor([0.299, 0.587, 0.114], device=device)
    for k in range(n):
        ps.append(float(psnr(a[k].astype(np.float32), b[k].astype(np.float32))))
        x = torch.from_numpy(a[k]).to(device).float() @ w
        y = torch.from_numpy(b[k]).to(device).float() @ w
        ss.append(float(ssim(x[None, None], y[None, None]).mean()))
    return {
        "psnr": float(np.mean(ps)),
        "psnr_nof0": float(np.mean(ps[1:])),
        "psnr_frame0": ps[0],
        "ssim": float(np.mean(ss)),
        "ssim_nof0": float(np.mean(ss[1:])),
        "ssim_frame0": ss[0],
        "frames": n,
    }


def quality(made, R, metrics, device):
    lo, hi = decode(made, 4, False, R["lo"]), decode(made, 4, False, R["hi"])
    d = distances(lo, R["rgb"], metrics, device)
    dh = distances(hi, R["rgb_hr"], metrics, device, tiles=True)
    d0 = distances(lo[1:], R["rgb"][1:], metrics, device)
    dh0 = distances(hi[1:], R["rgb_hr"][1:], metrics, device, tiles=True)
    return {
        "dists_real": d["dists"],
        "dists_hr": dh["dists"],
        "dists_real_nof0": d0["dists"],
        "dists_hr_nof0": dh0["dists"],
        "lpips_real": d["lpips"],
        "lpips_hr": dh["lpips"],
        "lpips_real_nof0": d0["lpips"],
        "lpips_hr_nof0": dh0["lpips"],
        "sharp_log": math.log(sharpness(decode(made, 8, True, R["lo"])) / R["sharp"]),
        "sharp_log_hr": math.log(sharpness(decode(made, 8, True, R["hi"])) / R["sharp_hr"]),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bench", required=True)
    ap.add_argument("--renders", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    import pyiqa

    device = torch.device(a.device)
    metrics = {
        "dists": pyiqa.create_metric("dists", device=device, as_loss=False),
        "lpips": pyiqa.create_metric("lpips", device=device, as_loss=False),
    }
    for c in bench.load(a.bench, "heldout")[a.shard :: a.shards]:
        out = Path(a.out) / "gt" / a.method / "heldout" / f"{c.key}.json"
        made = c.render(a.renders)
        if out.exists() or not made.exists():
            continue
        try:
            res = {
                **quality(str(made), real_readings(c.ground_truth), metrics, device),
                **pixel(str(made), c.ground_truth, device),
            }
        except Exception:
            print(f"SKIP {c.key}\n{traceback.format_exc()[-600:]}", flush=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"id": c.id, "method": a.method, **res}))
        print(f"{c.key} dists_hr={res['dists_hr']:.4f} psnr={res['psnr']:.2f}", flush=True)


if __name__ == "__main__":
    main()
