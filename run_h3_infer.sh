#!/bin/bash
# Standalone MiniMax-H3 inference: prompt (+ optional first frame) -> video WITH audio.
#
# This does NOT use `solarwm infer`. That path replays samples from a WDS validation
# index and needs the gated SolarWM-Data dataset. The H3 base checkpoint is a complete
# diffusers modular pipeline, so h3_infer.py drives it directly -- no dataset needed.
#
# Usage:
#   bash run_h3_infer.sh --prompt "a fox running through snow"
#   bash run_h3_infer.sh --prompt "..." --image first.png
#   bash run_h3_infer.sh --prompt "..." --steps 30 --num-frames 61
#   (all flags pass through to h3_infer.py)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

MODEL_PATH="${MODEL_PATH:-$HERE/../SolarWM-models/SolarWM-h3-33B-base}"

# Same global cache the download script uses (~/.cache/huggingface). The checkpoint's
# modular index names component sources, so loading can still consult the Hub -- point
# it at the cache that already holds the weights rather than a fresh empty one.
SHARED_HF_ROOT="${SHARED_HF_ROOT:-$HOME/.cache/huggingface}"
mkdir -p "$SHARED_HF_ROOT/hub"
export HF_HOME="$SHARED_HF_ROOT"
export HF_HUB_CACHE="$SHARED_HF_ROOT/hub"

VENV="$HERE/.venv-h3"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found -- run 'bash setup_env_h3.sh' first." >&2
  exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model not found: $MODEL_PATH" >&2
  echo "  Run 'bash download_h3_models.sh' first." >&2
  exit 1
fi

# No args at all -> run the bundled example rather than failing on a missing prompt.
if [ "$#" -eq 0 ]; then
  echo "[h3-infer] no arguments given -- running the bundled racer example."
  echo "[h3-infer] (pass --prompt/--prompt-file for your own, or use run_h3_examples.sh)"
  set -- --image "$HERE/examples/first_frame.png" \
         --prompt-file "$HERE/examples/racer/prompt.txt" \
         --width 832 --height 480 --steps 30 --num-frames 61 --name racer
fi

echo "[h3-infer] model : $MODEL_PATH"
echo "[h3-infer] cache : $SHARED_HF_ROOT"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader || true
echo

exec "$PY" "$HERE/h3_infer.py" --model-path "$MODEL_PATH" "$@"
