"""DOVER video quality (Wu et al., ICCV 2023), as eval/solar_renderer/metrics.py reads it.

Code: github.com/VQAssessment/DOVER at f1ddc96 (eval/solar_renderer/setup.sh clones it to $SOLAR_EVAL_MODELS/DOVER), weights: the
v0.1.0 release DOVER.pth. Preprocessing is the repo's own: the "val-l1080p" sample_types of dover.yml (technical: 7x7
fragments of 32x32, 3 clips x 32 frames, interval 2; aesthetic: 224x224 resize, 32 frames over the clip), the
UnifiedFrameSampler set-up and per-view normalisation of evaluate_one_video.py. Each branch's raw score is the mean of
its head output. numpy (the temporal offsets) and torch (the fragment positions) are seeded with 0 before every video,
so a video always gets the same score. The published numbers seeded numpy only, so their fragment positions came from
torch's running state: a clip's published DOVER can differ from this one by about 0.01-0.02, which averages out over a
split.

Scores (all in (0, 1), higher is better), the repo's score-level fusion with each (mean, std) pair applied to the
branch it was computed from:
    t = (raw_technical + 0.08285) / 0.03774      technical = sigmoid(t)
    a = (raw_aesthetic - 0.1107)  / 0.07355      aesthetic = sigmoid(a)
    overall = sigmoid(0.6104 * t + 0.3896 * a)
At f1ddc96 both repo scripts normalise the technical output with (0.1107, 0.07355) and the aesthetic output with
(-0.08285, 0.03774), but the repo's own LIVE-VQC predictions give 0.1105/0.0735 for the aesthetic head and
-0.0827/0.0377 for the technical head: the pairs are swapped there. Verbatim, technical sits at ~0.05-0.09 for every
video and aesthetic saturates near 1.
"""

import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

MODELS = Path(
    os.environ.get(
        "SOLAR_EVAL_MODELS", Path(__file__).resolve().parents[2] / "eval_models" / "solar_renderer"
    )
)
DOVER_REPO = os.environ.get("DOVER_REPO", str(MODELS / "DOVER"))
DOVER_CKPT = os.environ.get(
    "DOVER_CKPT", str(MODELS / "DOVER" / "pretrained_weights" / "DOVER.pth")
)
sys.path.insert(0, DOVER_REPO)
os.environ.setdefault(
    "TORCH_HOME", str(MODELS / "torchhub")
)  # the aesthetic branch's ImageNet init, before DOVER.pth

MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
STD = torch.FloatTensor([58.395, 57.12, 57.375])


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def fuse(raw):
    t = (raw["technical"] + 0.08285) / 0.03774
    a = (raw["aesthetic"] - 0.1107) / 0.07355
    return {
        "aesthetic": sigmoid(a),
        "technical": sigmoid(t),
        "overall": sigmoid(0.6104 * t + 0.3896 * a),
    }


def build(device):
    from dover.datasets import UnifiedFrameSampler
    from dover.models import DOVER

    with open(os.path.join(DOVER_REPO, "dover.yml")) as f:
        opt = yaml.safe_load(f)
    model = DOVER(**opt["model"]["args"]).to(device)
    model.load_state_dict(torch.load(DOVER_CKPT, map_location=device))
    model.eval()
    stypes = opt["data"]["val-l1080p"]["args"]["sample_types"]
    samplers = {}
    for k, s in stypes.items():  # same as evaluate_one_video.py
        if "t_frag" not in s:
            samplers[k] = UnifiedFrameSampler(s["clip_len"], s["num_clips"], s["frame_interval"])
        else:
            samplers[k] = UnifiedFrameSampler(
                s["clip_len"] // s["t_frag"], s["t_frag"], s["frame_interval"], s["num_clips"]
            )
    return model, stypes, samplers


@torch.no_grad()
def score(path, model, stypes, samplers, device, seed):
    from dover.datasets import spatial_temporal_view_decomposition

    np.random.seed(seed)
    torch.manual_seed(seed)
    views, _ = spatial_temporal_view_decomposition(path, stypes, samplers)
    keys = list(views)
    for k in keys:
        v = views[k]
        nc = stypes[k].get("num_clips", 1)
        views[k] = (
            ((v.permute(1, 2, 3, 0) - MEAN) / STD)
            .permute(3, 0, 1, 2)
            .reshape(v.shape[0], nc, -1, *v.shape[2:])
            .transpose(0, 1)
            .to(device)
        )
    outs = model(views)  # one score map per view, in the order of `views`
    return {k: o.mean().item() for k, o in zip(keys, outs, strict=False)}
