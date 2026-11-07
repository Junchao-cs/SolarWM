#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY="$(cd -- "${DEMO_DIR}/../.." && pwd)"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${DEMO_DIR}:${REPOSITORY}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN="${SOLARWM_PYTHON:-python}"
BACKEND="standard"
if [[ "${1:-}" == --backend ]]; then
  BACKEND="${2:-}"
  if [[ -z "${BACKEND}" ]]; then
    echo "--backend requires standard or memory" >&2
    exit 2
  fi
  shift 2
fi
case "${BACKEND}" in
  standard) MODULE="deploy.solar_wm.backends.standard" ;;
  memory) MODULE="deploy.solar_wm.backends.memory" ;;
  *) echo "Unknown backend: ${BACKEND}; use standard or memory" >&2; exit 2 ;;
esac
MODE="${1:-}"
if [[ "${MODE}" == configure ]]; then
  shift
  exec "${PYTHON_BIN}" -B -m "${MODULE}.configure" "$@"
elif [[ "${MODE}" == prepare ]]; then
  shift
  exec "${PYTHON_BIN}" -B -m "${MODULE}.prepare_scene" "$@"
elif [[ "${MODE}" == check ]]; then
  exec "${PYTHON_BIN}" -B -m "${MODULE}" "$@"
elif [[ "${MODE}" == serve || "${MODE}" == smoke ]]; then
  shift
  CUDA_VISIBLE_DEVICES="" "${PYTHON_BIN}" -B -m "${MODULE}" check "$@"
  exec "${PYTHON_BIN}" -B -m torch.distributed.run --standalone --max-restarts=0 --nproc-per-node=8 \
    --module "${MODULE}" "${MODE}" "$@"
else
  echo "Usage: bash launch.sh [--backend standard|memory] configure --weights /path/to/models --output /path/to/web-data" >&2
  echo "       bash launch.sh [--backend standard|memory] prepare --config /path/to/web-data/deployment.json --scene snow" >&2
  echo "       bash launch.sh [--backend standard|memory] {check|smoke|serve} --config /path/to/web-data/deployment.json" >&2
  echo "Memory configure additionally accepts --memory-lora /path/to/memory-checkpoint" >&2
  exit 2
fi
