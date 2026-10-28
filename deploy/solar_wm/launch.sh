#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd -- "${DEMO_DIR}/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${DEMO_DIR}:${REPOSITORY}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${SOLARWM_PYTHON:-python}"
MODE="${1:-}"
if [[ "${MODE}" == configure ]]; then
  shift
  exec "${PYTHON_BIN}" -B -m solarwm_web.configure "$@"
elif [[ "${MODE}" == prepare ]]; then
  shift
  exec "${PYTHON_BIN}" -B -m solarwm_web.prepare_scene "$@"
elif [[ "${MODE}" == check ]]; then
  exec "${PYTHON_BIN}" -B -m solarwm_web "$@"
elif [[ "${MODE}" == serve || "${MODE}" == smoke ]]; then
  shift
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -B -m solarwm_web check "$@"
  exec "${PYTHON_BIN}" -B -m torch.distributed.run --standalone --max-restarts=0 --nproc-per-node=8 \
    --module solarwm_web "${MODE}" "$@"
else
  echo "Usage: bash launch.sh configure --weights /path/to/models --output /path/to/web-data" >&2
  echo "       bash launch.sh prepare --config /path/to/web-data/deployment.json --scene snow" >&2
  echo "       bash launch.sh {check|smoke|serve} --config /path/to/web-data/deployment.json" >&2
  exit 2
fi
