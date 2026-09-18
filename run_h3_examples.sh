#!/bin/bash
# Run the bundled MiniMax-H3 racer examples end to end, with REAL
# camera-conditioned generation via h3_camera_infer.py -- the trained
# Stage0.5 LoRA adapter + a hand-authored per-frame camera trajectory, not a
# prompt-text hint. See h3_camera_infer.py's docstring for the full
# explanation and its real, unverified caveats (camera yaw sign/axis
# convention untested).
#
# UNTESTED end to end -- no local GPU/Python available to run this tonight.
# Expect to debug real errors on first run; report the traceback back.
#
# SLOW: h3_camera_infer.py has no batch mode in single-run mode -- each of
# racer/left/right reloads the 33B model + LoRA + Qwen/VisualVAE/AudioVAE.
#
# Every case here uses the same source image (examples/racer/Screenshot.png)
# with a direction-specific synthetic camera path (straight/left/right).
#
# Usage:
#   bash run_h3_examples.sh                    default: racer + left + right
#   bash run_h3_examples.sh racer              just the straight case
#   bash run_h3_examples.sh left               just the left case
#   bash run_h3_examples.sh right              just the right case
#   bash run_h3_examples.sh racer --steps 50   extra flags pass through to h3_camera_infer.py
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

WHICH="steer"
case "${1:-}" in
  racer|left|right|steer) WHICH="$1"; shift ;;
esac

STEPS=10
IMAGE="$HERE/examples/racer/Screenshot.png"

# No prompt files anymore -- camera_c2w (via --direction) is the real
# per-frame conditioning now, so h3_camera_infer.py's GENERIC_PROMPT default
# (same neutral text for straight/left/right) is enough; direction-specific
# prompt text was only ever a text hint, not real control.
if [ ! -e "$IMAGE" ]; then
  echo "ERROR: example asset missing: $IMAGE" >&2
  exit 1
fi

VENV="$HERE/.venv-h3"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found -- run 'bash setup_env_h3.sh' first." >&2
  exit 1
fi

BASE_MODEL="${SOLAR_H3_BASE:-$HERE/../SolarWM-models/SolarWM-h3-33B-base}"
ADAPTER="${SOLAR_H3_ADAPTER:-$HERE/../SolarWM-models/SolarWM-h3-33B-bid-stage0p5-158f}"
for path in "$BASE_MODEL" "$ADAPTER"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: expected path not found: $path" >&2
    echo "  Override with SOLAR_H3_BASE= / SOLAR_H3_ADAPTER= if models live elsewhere," >&2
    echo "  or run 'bash download_h3_models.sh' first." >&2
    exit 1
  fi
done

run_case() {
  local name="$1" direction="$2"
  shift 2
  echo "============================================================"
  echo "Example: $name (real camera-conditioned Stage0.5 generation, $direction)"
  echo "============================================================"
  "$PY" "$HERE/h3_camera_infer.py" \
    --base-model "$BASE_MODEL" \
    --adapter "$ADAPTER" \
    --image "$IMAGE" \
    --direction "$direction" \
    --steps "$STEPS" \
    --out "$HERE/outputs/h3/$name.mp4" \
    "$@"
  "$PY" "$HERE/examples/overlay_camera_arrow.py" \
    --video "$HERE/outputs/h3/$name.mp4" \
    --camera-c2w "$HERE/outputs/h3/$name.camera_c2w.npy" \
    --out "$HERE/outputs/h3/${name}_overlay.mp4"
}

case "$WHICH" in
  racer) run_case racer straight "$@" ;;
  left)  run_case racer_left left "$@" ;;
  right) run_case racer_right right "$@" ;;
  steer)
    run_case racer straight "$@"
    echo
    run_case racer_left left "$@"
    echo
    run_case racer_right right "$@"
    ;;
esac

echo
echo "Outputs under: $HERE/outputs/h3/"
ls -lh "$HERE/outputs/h3/" 2>/dev/null || true
