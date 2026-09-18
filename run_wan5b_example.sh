#!/bin/bash
# Run SolarWM Wan2.2-TI2V-5B stage2 (self-gradient-forcing) inference on ONE GPU.
#
# Why this config and not the H3 one: run_h3_example.sh uses
# configs/examples/minimax_h3/infer-158f-lora384-sp2.yaml -- a 33B model with
# sequence_parallel_size: 2 and a hardcoded --nproc-per-node=8. These two configs
# are the only ones in the repo declaring BOTH distributed.world_size: 1 and
# sequence_parallel_size: 1, so they are the only single-GPU inference paths
# SolarWM ships:
#
#   infer_stage2_sgf_81f.yaml            (default here) 81 pixel frames, 480x864
#   infer_stage2_sgf_camera_length.yaml  camera-length variant
#
# The configs ship with /path/to/... placeholders, so every root is injected with
# --set below, the same way run_h3_example.sh does it.
#
#   SOLAR_MODEL_ROOT=/path SOLAR_DATA_ROOT=/path SOLAR_OUTPUT_ROOT=/path \
#     bash run_wan5b_example.sh
#
# Usage:
#   bash run_wan5b_example.sh                    stage2 sgf 81f
#   bash run_wan5b_example.sh camera             the camera_length variant
#   bash run_wan5b_example.sh --dry-run-paths    check paths, don't launch
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

SOLAR_MODEL_ROOT="${SOLAR_MODEL_ROOT:-$HERE/../SolarWM-models}"
SOLAR_DATA_ROOT="${SOLAR_DATA_ROOT:-$HERE/../SolarWM-Data/releases-v1}"
SOLAR_OUTPUT_ROOT="${SOLAR_OUTPUT_ROOT:-$HERE/../SolarWM-outputs}"
# The config ships num_workers: 8; override here if the box wants fewer.
SOLAR_NUM_WORKERS="${SOLAR_NUM_WORKERS:-8}"

CONFIG="configs/examples/wan22_ti2v_5b/infer_stage2_sgf_81f.yaml"
RUN_TAG="wan22-ti2v-5b-stage2-sgf-81f"
if [ "${1:-}" = "camera" ]; then
  CONFIG="configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml"
  RUN_TAG="wan22-ti2v-5b-stage2-sgf-camera-length"
  shift
fi

DRY_RUN=0
if [ "${1:-}" = "--dry-run-paths" ]; then
  DRY_RUN=1
  shift
fi

VENV="$HERE/.venv-wan5b"
PY="$VENV/bin/python"
WAN_BASE="$SOLAR_MODEL_ROOT/Wan2.2-TI2V-5B"
SGF_CKPT="$SOLAR_MODEL_ROOT/SolarWM-5B-sgf-stage2-81f"
OUT_DIR="$SOLAR_OUTPUT_ROOT/$RUN_TAG"

echo "============================================================"
echo "SolarWM Wan2.2-TI2V-5B stage2 inference (single GPU)"
echo "============================================================"
echo "  config     : $CONFIG"
echo "  wan base   : $WAN_BASE"
echo "  checkpoint : $SGF_CKPT"
echo "  data root  : $SOLAR_DATA_ROOT"
echo "  output     : $OUT_DIR"
echo "  workers    : $SOLAR_NUM_WORKERS"
echo "============================================================"
echo

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)"
echo "[solarwm-wan5b-run] nvidia-smi reports $GPU_COUNT GPU(s); this config needs 1."
if [ "$GPU_COUNT" -lt 1 ]; then
  echo "ERROR: no usable GPU visible." >&2
  exit 2
fi
echo

# Fail on missing inputs here rather than deep inside torchrun.
MISSING=0
check() {
  if [ ! -e "$2" ]; then
    echo "MISSING: $1  $2" >&2
    MISSING=1
  fi
}
check "venv python" "$PY"
check "config    " "$HERE/$CONFIG"
check "Wan base  " "$WAN_BASE"
check "checkpoint" "$SGF_CKPT"
check "data root " "$SOLAR_DATA_ROOT"

if [ "$MISSING" -ne 0 ]; then
  echo >&2
  echo "One or more required inputs are missing:" >&2
  echo "  - venv        : bash setup_env_wan5b.sh (or reuse an existing SolarWM venv)" >&2
  echo "  - model roots : Wan2.2-TI2V-5B base weights + the SolarWM stage2 checkpoint" >&2
  echo "  - data root   : SolarWM-Data releases-v1, providing" >&2
  echo "                  recipes/clean-81f/raw-wds/test-index.jsonl.gz" >&2
  exit 1
fi

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[solarwm-wan5b-run] all expected paths present. Not launching (--dry-run-paths)."
  exit 0
fi

source "$VENV/bin/activate"
mkdir -p "$OUT_DIR"

echo "[solarwm-wan5b-run] all expected paths present, launching inference..."
# world_size is 1 in this config, but SolarWM still initialises a process group.
torchrun --standalone --nproc-per-node=1 -m solarwm infer \
  --config "$CONFIG" \
  --set model.base_path="$WAN_BASE" \
  --set checkpoint.path="$SGF_CKPT" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.num_workers="$SOLAR_NUM_WORKERS" \
  --set runtime.output_dir="$OUT_DIR" \
  "$@"

echo
echo "Done. Output: $OUT_DIR"
