#!/bin/bash
# Run MiniMax-H3 inference on SolarWM's example config
# (configs/examples/minimax_h3/infer-158f-lora384-sp2.yaml), per
# docs/backends/minimax-h3.md's documented command.
#
# REQUIRES 8 GPUs (--nproc-per-node=8, hardcoded in the doc's own command --
# not something this script invented). Check `nvidia-smi` before running.
#
# REQUIRES the minimax-h3-158f-768p-nomind-v1 latent data under
# SOLAR_DATA_ROOT, which SolarWM's own docs flag as "upload coming soon" as
# of this writing (see download_h3_models.sh's note) -- this script checks
# that the expected support files exist before launching and fails with a
# clear message instead of a confusing deep-in-torchrun error if they don't.
#
#   SOLAR_MODEL_ROOT=/path SOLAR_DATA_ROOT=/path SOLAR_OUTPUT_ROOT=/path \
#     bash run_h3_example.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SOLAR_MODEL_ROOT="${SOLAR_MODEL_ROOT:-$HERE/../SolarWM-models}"
SOLAR_DATA_ROOT="${SOLAR_DATA_ROOT:-$HERE/../SolarWM-Data/releases-v1}"
SOLAR_OUTPUT_ROOT="${SOLAR_OUTPUT_ROOT:-$HERE/../SolarWM-outputs}"

VENV="$HERE/.venv-h3"
PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
  echo "ERROR: $PY not found -- run 'bash setup_env_h3.sh' first." >&2
  exit 1
fi
source "$VENV/bin/activate"

H3_BASE="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-base"
H3_RESUME="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-bid-stage0p5-158f"
H3_SUPPORT="$SOLAR_DATA_ROOT/latent-wds/minimax-h3-158f-768p-nomind-v1/support"

for path in "$H3_BASE" "$H3_RESUME" "$H3_SUPPORT/h3_silence_153_158_170.safetensors" "$H3_SUPPORT/encoder_contract.json"; do
  if [ ! -e "$path" ]; then
    echo "ERROR: expected path not found: $path" >&2
    echo "  Either 'bash download_h3_models.sh' hasn't been run, or (more likely" >&2
    echo "  for the SOLAR_DATA_ROOT/support files) the latent data isn't published" >&2
    echo "  yet -- SolarWM's own docs flag minimax-h3-158f-768p-nomind-v1 as" >&2
    echo "  'upload coming soon' as of this writing. Check docs/latent-wds.md." >&2
    exit 1
  fi
done

echo "[solarwm-h3-run] all expected paths present, launching inference..."
GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
echo "[solarwm-h3-run] nvidia-smi reports $GPU_COUNT GPU(s); this command needs 8 (--nproc-per-node=8)."
if [ "$GPU_COUNT" -lt 8 ]; then
  echo "WARNING: fewer than 8 GPUs visible -- the documented command hardcodes" >&2
  echo "  --nproc-per-node=8. Continuing anyway; torchrun will fail its own way" >&2
  echo "  if there really aren't 8 available." >&2
fi

mkdir -p "$SOLAR_OUTPUT_ROOT"
torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-158f-lora384-sp2.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$H3_RESUME" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage0p5-158f-infer"

echo
echo "Done. Output: $SOLAR_OUTPUT_ROOT/h3-stage0p5-158f-infer"
