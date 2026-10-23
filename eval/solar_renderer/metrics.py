#!/usr/bin/env python
"""Automatic metrics of the SolarRenderer benchmark, the same on both splits.

    python eval/solar_renderer/metrics.py --bench ./SolarRenderer-Data --renders ./renders/mymethod --method mymethod \
        --out ./scores [--split hard|heldout|all] [--metrics dover,laion_aes,...] [--shard k --shards n]

For every case with a render, computes the missing metrics and writes <out>/<metric>/<method>/<case key>.json, where
the case key is hard/<case_id> or heldout/<clip_id>. Run one process per GPU with --shard / --shards.
VideoScore2 needs its own environment (transformers 4.53); run it as a separate pass with --metrics videoscore2.
Models are set up by eval/solar_renderer/setup.sh.

Readings, grouped into the four dimensions of eval/solar_renderer/summary.py:
  quality     dover        DOVER (ICCV 2023), technical / aesthetic / overall; the repo's preprocessing (val-l1080p
                           views), numpy seeded with 0 before every video, score-level fusion (see dover_score.py)
              laion_aes    LAION aesthetic predictor v2 (sac+logos+ava1-l14-linearMSE) on CLIP ViT-L/14 image
                           embeddings, 2 frames per second, mean
              videoscore2  VideoScore2 visual quality: greedy decoding at 2 fps, the expectation over levels 1-5
  stability   flicker      RAFT-large flow t->t+1 and t+1->t at 672x384, frame t+1 warped back to t; mean absolute
                           residual (0-255) over forward-backward consistent pixels after a Gaussian low-pass
                           (sigma 2 px), warp_l1_lp - lower is better. The unfiltered residual (warp_l1) rises with
                           sharpness and is reported only for reference.
              dino         DINOv2 ViT-S/14 CLS embeddings of all frames at 224x392: mean adjacent-frame cosine
  control     motion       CoTracker3 offline on the structure video and on the render, a 20x12 grid queried at
                           frames 0/40/80/120; error = |render track - structure track| / frame diagonal, over the
                           (point, frame) pairs where the structure track is reliable (visibility x confidence > 0.6)
              edges        Canny (5x5 blur, sigma 1.2, 50/150) on both, every 4th frame, edge F1 with a 2 px tolerance
  first frame firstframe   the DINOv2 embeddings against the first frame's: mean cosine over the last second
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench  # noqa: E402

MODELS = Path(
    os.environ.get("SOLAR_EVAL_MODELS", HERE.parents[1] / "eval_models" / "solar_renderer")
)
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
ALL = ["dover", "laion_aes", "flicker", "dino", "firstframe", "motion", "edges"]
VERSION = "1"


# ---------------------------------------------------------------------------------------------- io


def decode(path: str, size: tuple[int, int] | None = None, fps: float | None = None) -> np.ndarray:
    """uint8 [T,H,W,3] RGB; size = (w, h), area-resampled."""
    vf = []
    if fps:
        vf.append(f"fps={fps}")
    if size:
        vf.append(f"scale={size[0]}:{size[1]}:flags=area")
    vf.append("format=rgb24")
    if size is None:
        w, h = probe_size(path)
    else:
        w, h = size
    raw = subprocess.run(
        [
            FFMPEG,
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            path,
            "-vf",
            ",".join(vf),
            "-f",
            "rawvideo",
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    a = np.frombuffer(raw, np.uint8)
    n = len(a) // (w * h * 3)
    return a[: n * w * h * 3].reshape(n, h, w, 3)


def probe_size(path: str) -> tuple[int, int]:
    out = (
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
    return int(out[0]), int(out[1])


def load_png(path: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


# ---------------------------------------------------------------------------------------------- models


class Models:
    """Lazily built, one per worker; weights from eval/solar_renderer/setup.sh under SOLAR_EVAL_MODELS (default <repo>/eval_models/solar_renderer)."""

    def __init__(self, device):
        self.device = device
        self._m = {}

    def get(self, name):
        if name not in self._m:
            self._m[name] = getattr(self, f"_build_{name}")()
        return self._m[name]

    def _build_dover(self):
        import dover_score

        model, stypes, samplers = dover_score.build(self.device)
        return dover_score, model, stypes, samplers

    def _build_clip(self):
        from transformers import CLIPModel

        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.float32)
        clip = clip.to(self.device).eval()
        head = torch.nn.Sequential(  # improved-aesthetic-predictor MLP (simple_inference.py), dropouts are no-ops
            torch.nn.Linear(768, 1024),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(1024, 128),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 16),
            torch.nn.Linear(16, 1),
        )
        sd = torch.load(MODELS / "sac+logos+ava1-l14-linearMSE.pth", map_location="cpu")
        sd = {k.replace("layers.", ""): v for k, v in sd.items()}
        head.load_state_dict(sd, strict=True)
        return clip, head.to(self.device).eval()

    def _build_dino(self):
        import timm

        m = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m",
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True,
        )
        return m.to(self.device).eval()

    def _build_raft(self):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        return (
            raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2, progress=False)
            .to(self.device)
            .eval()
        )

    def _build_cotracker(self):
        sys.path.insert(0, str(MODELS / "co-tracker"))
        from cotracker.models.build_cotracker import build_cotracker

        model = build_cotracker(None, offline=True, window_len=60, v2=False)
        sd = torch.load(MODELS / "cotracker3" / "baseline_offline.pth", map_location="cpu")
        if "model" in sd:
            sd = sd["model"]
        model.load_state_dict(sd, strict=True)
        return model.to(self.device).eval()

    def _build_vs2(self):
        import videoscore2

        model, processor, tokenizer = videoscore2.build()
        return videoscore2, model, processor, tokenizer


# ---------------------------------------------------------------------------------------------- metrics


@torch.no_grad()
def m_dover(path, M):
    ds, model, stypes, samplers = M.get("dover")
    raw = ds.score(path, model, stypes, samplers, M.device, 0)
    return {"dover": ds.fuse(raw), "dover_raw": raw}


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def clip_pre(frames: np.ndarray) -> torch.Tensor:
    """openai CLIP preprocessing: bicubic resize of the short side to 224 (PIL), centre crop 224."""
    from PIL import Image

    out = []
    for f in frames:
        im = Image.fromarray(f)
        w, h = im.size
        s = 224 / min(w, h)
        im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - 224) // 2, (h - 224) // 2
        out.append(np.asarray(im.crop((l, t, l + 224, t + 224))))
    x = torch.from_numpy(np.stack(out)).permute(0, 3, 1, 2).float() / 255.0
    return (x - CLIP_MEAN) / CLIP_STD


@torch.no_grad()
def m_laion(frames24, M):
    clip, head = M.get("clip")
    sel = frames24[::12]  # 2 per second at 24 fps
    x = clip_pre(sel).to(M.device)
    emb = clip.visual_projection(clip.vision_model(pixel_values=x).pooler_output)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    s = head(emb).squeeze(-1).float().cpu().numpy()
    return {"laion_aes": float(s.mean()), "laion_aes_min": float(s.min()), "frames": len(sel)}


IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DINO_HW = (224, 392)  # 16 x 28 patches of 14, the 1344x768 aspect (1.75)


@torch.no_grad()
def dino_embed(frames: np.ndarray, M) -> torch.Tensor:
    m = M.get("dino")
    out = []
    for i in range(0, len(frames), 64):
        x = (
            torch.from_numpy(np.ascontiguousarray(frames[i : i + 64]))
            .to(M.device)
            .permute(0, 3, 1, 2)
            .float()
            / 255
        )
        x = F.interpolate(
            x, size=DINO_HW, mode="bicubic", align_corners=False, antialias=True
        ).clamp(0, 1)
        x = (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)
        out.append(F.normalize(m(x).float(), dim=-1))
    return torch.cat(out)


def m_dino(emb):
    adj = (emb[1:] * emb[:-1]).sum(-1)
    return {
        "adjacent_cos": float(adj.mean()),
        "adjacent_cos_min": float(adj.min()),
        "first_last_cos": float((emb[0] * emb[-1]).sum()),
    }


def m_firstframe(emb, first_emb, fps=24.0):
    cos = (emb @ first_emb).cpu().numpy().astype(np.float64)
    t = np.arange(len(cos)) / fps
    slope = float(np.polyfit(t, cos, 1)[0])
    return {
        "last1s_cos": float(cos[-int(fps) :].mean()),
        "slope_per_s": slope,
        "frame0_cos": float(cos[0]),
        "mean_cos": float(cos.mean()),
        "curve_1fps": [round(float(c), 4) for c in cos[:: int(fps)]],
    }


FLOW_WH = (672, 384)


@torch.no_grad()
def flows(frames: np.ndarray, M, batch=12):
    """Forward (t->t+1) and backward (t+1->t) RAFT flows at FLOW_WH. frames uint8 [T,h,w,3] already at FLOW_WH."""
    raft = M.get("raft")
    x = (
        torch.from_numpy(np.ascontiguousarray(frames)).to(M.device).permute(0, 3, 1, 2).float()
        / 127.5
        - 1.0
    )
    fw, bw = [], []
    for i in range(0, len(x) - 1, batch):
        a = x[i : min(i + batch, len(x) - 1)]
        b = x[i + 1 : min(i + batch, len(x) - 1) + 1]
        f = raft(torch.cat([a, b]), torch.cat([b, a]))[-1]
        fw.append(f[: len(a)])
        bw.append(f[len(a) :])
    return x, torch.cat(fw), torch.cat(bw)


def warp(img, flow):
    """Sample img at x + flow(x). img [N,C,H,W], flow [N,2,H,W] (pixels). Returns warped and in-bounds mask."""
    _n, _, h, w = img.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=img.device), torch.arange(w, device=img.device), indexing="ij"
    )
    gx = xx[None] + flow[:, 0]
    gy = yy[None] + flow[:, 1]
    grid = torch.stack([2 * gx / (w - 1) - 1, 2 * gy / (h - 1) - 1], -1)
    out = F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)
    inb = (gx >= 0) & (gx <= w - 1) & (gy >= 0) & (gy <= h - 1)
    return out, inb


@torch.no_grad()
def m_flicker(frames_lo: np.ndarray, M):
    x, fw, bw = flows(frames_lo, M)
    img_t, img_t1 = x[:-1], x[1:]
    warped, inb = warp(img_t1, fw)
    bw_at = warp(bw, fw)[0]  # backward flow sampled where the forward flow lands
    d = ((fw + bw_at) ** 2).sum(1)
    ok = inb & (
        d < 0.01 * ((fw**2).sum(1) + (bw_at**2).sum(1)) + 0.5
    )  # Sundaram et al. 2010 consistency check
    err = (img_t - warped).abs().mean(1) * 127.5  # [-1,1] -> 0-255 units
    okf = ok.float()
    per_pair = (err * okf).sum((1, 2)) / okf.sum((1, 2)).clamp(min=1)
    pp = per_pair.cpu().numpy()
    # low-pass residual: the warp residual blurred (Gaussian sigma 2 px at 672 wide, normalised convolution over the
    # consistent pixels) before |.|. Sub-pixel misalignment of fine texture is high-frequency and cancels; brightness /
    # colour flicker is low-frequency and stays. (The raw reading rises with sharpness.)
    k = torch.exp(-0.5 * (torch.arange(-6, 7, device=x.device).float() / 2.0) ** 2)
    k = k / k.sum()

    def blur(z):  # [N,C,H,W], separable
        c = z.shape[1]
        z = F.conv2d(
            F.pad(z, (6, 6, 0, 0), mode="replicate"),
            k.view(1, 1, 1, -1).repeat(c, 1, 1, 1),
            groups=c,
        )
        return F.conv2d(
            F.pad(z, (0, 0, 6, 6), mode="replicate"),
            k.view(1, 1, -1, 1).repeat(c, 1, 1, 1),
            groups=c,
        )

    lp_pp = []
    for i in range(0, len(img_t), 16):
        m = okf[i : i + 16, None]
        r = (img_t[i : i + 16] - warped[i : i + 16]) * m
        bm = blur(m)
        lp = (blur(r) / bm.clamp(min=1e-3)).abs().mean(1) * 127.5
        v = m[:, 0] * (bm[:, 0] > 0.5)
        lp_pp.append((lp * v).sum((1, 2)) / v.sum((1, 2)).clamp(min=1))
    lp_pp = torch.cat(lp_pp).cpu().numpy()
    g = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]) * 127.5  # grey, 0-255 offsets
    lap = (
        -4 * g[:, 1:-1, 1:-1] + g[:, :-2, 1:-1] + g[:, 2:, 1:-1] + g[:, 1:-1, :-2] + g[:, 1:-1, 2:]
    )
    return {
        "warp_l1": float(pp.mean()),
        "warp_l1_p90": float(np.percentile(pp, 90)),
        "warp_l1_lp": float(lp_pp.mean()),
        "warp_l1_lp_p90": float(np.percentile(lp_pp, 90)),
        "sharpness_lapvar": float(lap.flatten(1).var(1).mean()),
        "warp_l1_max": float(pp.max()),
        "consistent_frac": float(okf.mean()),
        "flow_mag": float(fw.norm(dim=1).mean()),
        "pairs": len(pp),
    }


GRID = (20, 12)  # x, y
QUERY_FRAMES = (
    0,
    40,
    80,
    120,
)  # a fresh grid every 40 frames: under fast camera moves a frame-0 grid has left the
# picture by frame 60 (0-22% of frame-0 points were still in frame at 100)
CT_HW = (384, 512)
REL_THR = 0.6


@torch.no_grad()
def track(frames: np.ndarray, M, query_frames=QUERY_FRAMES):
    """CoTracker3 offline; a GRID at each query frame. Returns tracks [T,N,2] (input pixels), vis, conf [T,N],
    query xy [N,2], query frame [N]."""
    model = M.get("cotracker")
    T, H, W, _ = frames.shape
    v = torch.from_numpy(frames.copy()).to(M.device).permute(0, 3, 1, 2).float()
    v = F.interpolate(v, CT_HW, mode="bilinear", align_corners=True)[
        None
    ]  # as CoTrackerPredictor does
    gx = (np.arange(GRID[0]) + 0.5) / GRID[0] * W
    gy = (np.arange(GRID[1]) + 0.5) / GRID[1] * H
    grid = np.array([(x, y) for y in gy for x in gx], np.float32)
    qf = [t for t in query_frames if t < T - 1]
    pts = np.concatenate([grid] * len(qf))
    qt = np.repeat(np.array(qf), len(grid))
    q = torch.zeros(1, len(pts), 3, device=M.device)
    q[0, :, 0] = torch.from_numpy(qt).to(M.device).float()
    q[0, :, 1] = torch.from_numpy(pts[:, 0]).to(M.device) * (CT_HW[1] - 1) / (W - 1)
    q[0, :, 2] = torch.from_numpy(pts[:, 1]).to(M.device) * (CT_HW[0] - 1) / (H - 1)
    tr, vis, conf, _ = model(video=v, queries=q, iters=6)
    tr = tr[0] * tr.new_tensor([(W - 1) / (CT_HW[1] - 1), (H - 1) / (CT_HW[0] - 1)])
    tr[torch.from_numpy(qt).long(), torch.arange(len(pts))] = torch.from_numpy(pts).to(
        tr.device
    )  # query = query
    return (
        tr.float().cpu().numpy(),
        vis[0].float().cpu().numpy(),
        conf[0].float().cpu().numpy(),
        pts,
        qt,
    )


def motion_compare(tc, vc, cc, tm, vm, cm, pts, qt, W, H):
    """Per (point, frame after its query frame) where the cond track is reliable."""
    diag = math.hypot(W, H)
    n = min(len(tc), len(tm))
    t = np.arange(n)[:, None]
    after = t > qt[None]
    rel = ((vc[:n] * cc[:n]) > REL_THR) & after
    err = np.linalg.norm(tm[:n] - tc[:n], axis=-1) / diag
    still = (
        np.linalg.norm(tc[:n] - pts[None], axis=-1) / diag
    )  # a frozen render: every track stays at its query
    out_vis = (vm[:n] * cm[:n]) > REL_THR
    # velocity step k -> k+1, valid when both ends are reliable (the query frame end is exact)
    start_ok = ((vc[: n - 1] * cc[: n - 1]) > REL_THR) & (t[:-1] > qt[None]) | (t[:-1] == qt[None])
    vrel = rel[1:] & start_ok
    verr = np.linalg.norm((tm[1:n] - tm[: n - 1]) - (tc[1:n] - tc[: n - 1]), axis=-1) / diag
    if rel.sum() == 0:
        return {"kept_frac": 0.0, "err_mean": None}
    return {
        "err_mean": float(err[rel].mean()),
        "frac_lt_2pct": float((err[rel] < 0.02).mean()),
        "err_median": float(np.median(err[rel])),
        "vel_err_mean": float(verr[vrel].mean()) if vrel.any() else None,
        "frozen_err_mean": float(still[rel].mean()),
        "frozen_frac_lt_2pct": float((still[rel] < 0.02).mean()),
        "kept_frac": float(rel.sum() / after.sum()),
        "kept_pairs": int(rel.sum()),
        "kept_frac_by_query": [
            round(float(rel[:, qt == q].sum() / max(after[:, qt == q].sum(), 1)), 3)
            for q in np.unique(qt)
        ],
        "render_reliable_frac": float(out_vis[rel].mean()),
        "frames": int(n),
    }


def canny(frames: np.ndarray) -> np.ndarray:
    import cv2

    return np.stack(
        [
            cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (5, 5), 1.2), 50, 150)
            for f in frames
        ]
    )


def edge_errors(made: np.ndarray, want: np.ndarray, tolerance: int = 2, level: int = 64) -> dict:
    """Edge F1: an edge counts as found when the other side has one within ``tolerance`` pixels."""

    frames = min(len(made), len(want))
    found = torch.from_numpy(np.ascontiguousarray(made[:frames]) >= level).float()[:, None]
    given = torch.from_numpy(np.ascontiguousarray(want[:frames]) >= level).float()[:, None]
    window = 2 * tolerance + 1

    def near(edges: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.max_pool2d(edges, window, stride=1, padding=tolerance)

    hits_found = float((found * near(given)).sum())
    hits_given = float((given * near(found)).sum())
    precision = hits_found / max(float(found.sum()), 1.0)
    recall = hits_given / max(float(given.sum()), 1.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"F1": f1, "precision": precision, "recall": recall, "frames": frames}


def m_edges(render_full, cond_full):
    r, c = render_full[::4], cond_full[::4]
    n = min(len(r), len(c))
    out = edge_errors(canny(r[:n]), canny(c[:n]))
    out["cond_edge_density"] = float((canny(c[:n]) > 0).mean())
    return out


@torch.no_grad()
def m_vs2(path, prompt, M):
    vs, model, processor, tokenizer = M.get("vs2")
    r = vs.judge(model, processor, tokenizer, path, prompt)
    return {
        "visual_quality": r["visual_quality"],
        "visual_quality_expected": r["visual_quality_expected"],
        "alignment": r["alignment"],
        "physics": r["physics"],
        "reason": r["reason"][-400:],
    }


# ---------------------------------------------------------------------------------------------- driver


def compute(
    need: list[str], render: str, cond: str | None, first: str | None, M, prompt: str = ""
) -> dict:
    """All metrics in need on local files; returns {metric: payload with 'sec'}."""
    res = {}

    def timed(name, fn):
        torch.cuda.synchronize(M.device)
        t0 = time.time()
        try:
            out = fn()
        except Exception:
            traceback.print_exc()
            out = {"error": traceback.format_exc()[-400:]}
        torch.cuda.synchronize(M.device)
        out["sec"] = round(time.time() - t0, 2)
        res[name] = out

    full = (
        decode(render)
        if set(need) & {"laion_aes", "dino", "firstframe", "motion", "edges"}
        else None
    )
    if full is not None and len(full) == 0:
        raise RuntimeError("render decodes to 0 frames")
    cv = None
    if set(need) & {"motion", "edges"}:
        cv = decode(cond)
        if cv.shape[1:3] != full.shape[1:3]:
            cv = decode(cond, size=(full.shape[2], full.shape[1]))
    if "dover" in need:
        timed("dover", lambda: m_dover(render, M))
    if "laion_aes" in need:
        timed("laion_aes", lambda: m_laion(full, M))
    if "flicker" in need:
        timed("flicker", lambda: m_flicker(decode(render, size=FLOW_WH), M))
    if "dino" in need or "firstframe" in need:
        t0 = time.time()
        emb = dino_embed(full, M)
        te = time.time() - t0
        if "dino" in need:
            timed("dino", lambda: m_dino(emb))
            res["dino"]["sec"] = round(res["dino"]["sec"] + te, 2)
        if "firstframe" in need:
            timed("firstframe", lambda: m_firstframe(emb, dino_embed(load_png(first)[None], M)[0]))
            res["firstframe"]["sec"] = round(res["firstframe"]["sec"] + te, 2)
    if "motion" in need:

        def mo():
            # the query frames both videos have: a render shorter than its structure (Wan, 121 frames) would
            # otherwise track a different number of points than the structure
            qf = tuple(t for t in QUERY_FRAMES if t < min(len(cv), len(full)) - 1)
            tc, vc, cc, pts, qt = track(cv, M, qf)
            tm, vm, cm, _, _ = track(full, M, qf)
            return motion_compare(tc, vc, cc, tm, vm, cm, pts, qt, full.shape[2], full.shape[1])

        timed("motion", mo)
    if "edges" in need:
        timed("edges", lambda: m_edges(full, cv))
    if "videoscore2" in need:
        timed("videoscore2", lambda: m_vs2(render, prompt, M))
    return res


def result_path(out: str, metric: str, method: str, case: bench.Case) -> Path:
    """<out>/<metric>/<method>/hard/<case_id>.json or <out>/<metric>/<method>/heldout/<clip_id>.json"""
    return Path(out) / metric / method / case.split / f"{case.key}.json"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--bench", required=True, help="local copy of the SolarRenderer-Data dataset")
    ap.add_argument(
        "--renders",
        required=True,
        help="<renders>/hard/<case_id>.mp4, <renders>/heldout/<clip_id>.mp4",
    )
    ap.add_argument("--method", required=True, help="the name results are filed under")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="all", choices=("hard", "heldout", "all"))
    ap.add_argument(
        "--metrics", default=",".join(ALL), help=f"comma list of {[*ALL, 'videoscore2']}"
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    metrics = [m for m in a.metrics.split(",") if m]
    M = Models(torch.device(a.device))
    torch.cuda.set_device(M.device)
    cases = bench.load(a.bench, a.split)[a.shard :: a.shards]
    done = skipped = 0
    for c in cases:
        render = c.render(a.renders)
        need = [m for m in metrics if not result_path(a.out, m, a.method, c).exists()]
        if not need:
            continue
        if not render.exists():
            skipped += 1
            continue
        try:
            res = compute(need, str(render), c.condition, c.first_frame, M, c.prompt)
        except Exception:
            traceback.print_exc()
            print(f"SKIP {c.key}", flush=True)
            continue
        for m, payload in res.items():
            if "error" in payload:  # not written: the next run retries it
                print(f"{c.key} {m} ERROR {payload['error'][-200:]}", flush=True)
                continue
            p = result_path(a.out, m, a.method, c)
            p.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "id": c.id,
                "split": c.split,
                "method": a.method,
                "metric": m,
                "kind": c.kind,
                "tags": c.tags,
                "version": VERSION,
                **payload,
            }
            p.write_text(json.dumps(rec, ensure_ascii=False))
        done += 1
        brief = {
            m: {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in p.items()
                if not isinstance(v, (list, dict, str))
            }
            for m, p in res.items()
        }
        print(f"{c.key} {json.dumps(brief)}", flush=True)
    print(f"done: {done} cases scored, {skipped} without a render", flush=True)


if __name__ == "__main__":
    main()
