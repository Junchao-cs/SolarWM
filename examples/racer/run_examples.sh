#!/bin/bash
# Generate straight/left/right racer examples with REAL camera-conditioned
# generation via ../../h3_camera_infer.py -- the trained Stage0.5 LoRA
# adapter + a hand-authored per-frame camera trajectory, not a prompt-text
# hint. See h3_camera_infer.py's docstring for the full explanation and its
# real, unverified caveats (camera yaw sign/axis convention untested).
#
# UNTESTED end to end -- no local GPU/Python available to run this tonight.
# Expect to debug real errors on first run; report the traceback back.
#
# SLOW: h3_camera_infer.py has no batch mode (unlike h3_infer.py's
# --mind-batch) -- it reloads the 33B model + LoRA + Qwen/VisualVAE/AudioVAE
# on every invocation, so this script pays that cost 3x, once per direction.
#
#   bash examples/racer/run_examples.sh [--steps N]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

PY="$ROOT/.venv-h3/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found -- run 'bash setup_env_h3.sh' first." >&2
  exit 1
fi

BASE_MODEL="${SOLAR_H3_BASE:-$ROOT/../SolarWM-models/SolarWM-h3-33B-base}"
ADAPTER="${SOLAR_H3_ADAPTER:-$ROOT/../SolarWM-models/SolarWM-h3-33B-bid-stage0p5-158f}"
for path in "$BASE_MODEL" "$ADAPTER"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: expected path not found: $path" >&2
    echo "  Override with SOLAR_H3_BASE= / SOLAR_H3_ADAPTER= if models live elsewhere," >&2
    echo "  or run 'bash download_h3_models.sh' first." >&2
    exit 1
  fi
done

STEPS=30
while [ $# -gt 0 ]; do
  case "$1" in
    --steps) STEPS="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

OUT_DIR="$ROOT/outputs/racer"
mkdir -p "$OUT_DIR"
IMAGE="$HERE/Screenshot.png"

for d in straight left right; do
  case "$d" in
    straight) prompt_file="$HERE/prompt.txt" ;;
    left)     prompt_file="$HERE/prompt_left.txt" ;;
    right)    prompt_file="$HERE/prompt_right.txt" ;;
  esac
  echo "============================================================"
  echo "[racer] $d (real camera-conditioned Stage0.5 generation)"
  echo "============================================================"
  "$PY" "$ROOT/h3_camera_infer.py" \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --image "$IMAGE" \
    --prompt-file "$prompt_file" \
    --direction "$d" \
    --steps "$STEPS" \
    --out "$OUT_DIR/$d.mp4"
done

echo
echo "Done. Outputs:"
echo "  $OUT_DIR/straight.mp4"
echo "  $OUT_DIR/left.mp4"
echo "  $OUT_DIR/right.mp4"
