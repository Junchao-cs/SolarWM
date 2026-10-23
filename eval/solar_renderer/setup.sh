#!/usr/bin/env bash
# Set up the models eval/solar_renderer/metrics.py reads, under SOLAR_EVAL_MODELS (default <repo>/eval_models/solar_renderer), at the versions the
# published numbers used. Run once from the repository root:  bash eval/solar_renderer/setup.sh
set -euo pipefail
SOLAR_EVAL_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
M=${SOLAR_EVAL_MODELS:-"$SOLAR_EVAL_ROOT/eval_models/solar_renderer"}
mkdir -p "$M"
python -m pip install "timm>=1.0" "pyiqa>=0.1.16" opencv-python-headless decord scikit-video

check() {  # check <file> <sha256>
  echo "$2  $1" | sha256sum -c --quiet - || { echo "checksum mismatch: $1"; exit 1; }
}

# DOVER at f1ddc96 and its v0.1.0 weights
if [ ! -d "$M/DOVER" ]; then
  git clone https://github.com/VQAssessment/DOVER "$M/DOVER"
  git -C "$M/DOVER" checkout f1ddc96
fi
mkdir -p "$M/DOVER/pretrained_weights"
W="$M/DOVER/pretrained_weights/DOVER.pth"
[ -s "$W" ] || curl -L -o "$W" https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth
check "$W" f4a42c0bbc94c94dd7409e7f40887d44c5c30314d1d09e7edf03cc35813b4838

# LAION aesthetic predictor v2 head (CLIP ViT-L/14 comes from the Hub: openai/clip-vit-large-patch14)
W="$M/sac+logos+ava1-l14-linearMSE.pth"
[ -s "$W" ] || curl -L -o "$W" "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac+logos+ava1-l14-linearMSE.pth"
check "$W" 21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0

# CoTracker3 at 82e02e8, offline baseline weights
if [ ! -d "$M/co-tracker" ]; then
  git clone https://github.com/facebookresearch/co-tracker "$M/co-tracker"
  git -C "$M/co-tracker" checkout 82e02e8
fi
hf download facebook/cotracker3 baseline_offline.pth --local-dir "$M/cotracker3"
check "$M/cotracker3/baseline_offline.pth" da09bbac871f7398e5b29c4de5213652658949737bc158840b101678ba8ad1df

# DINOv2 ViT-S/14 (timm), RAFT-large (torchvision) and DISTS / LPIPS (pyiqa) download themselves on first use.
echo "eval models ready in $M"
