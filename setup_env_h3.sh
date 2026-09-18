#!/bin/bash
# Build SolarWM's MiniMax-H3 backend venv (.venv-h3), per
# environments/README.md's H3-specific instructions -- adapted for this
# GB300 box (aarch64, CUDA 13.2) the same way as every other project set up
# tonight:
#   1. No hardcoded python3.10 -- uses whatever python3 is on PATH.
#   2. torch/torchvision unpinned from cu132 (not the doc's pinned
#      torch==2.6.0 from cu124) -- a pinned version can silently not exist
#      on a different CUDA index and pip falls back to something wrong
#      instead of erroring (confirmed the hard way on H3-World tonight).
#   3. flash-attn unpinned (not the doc's pinned ==2.8.3) -- that exact
#      pinned version already failed to build from source on THIS box
#      tonight for a different project (zing-world-model), so pinning to
#      it again here would very likely just repeat the same failure.
#   4. Explicitly NOT using uv, even though SolarWM's own pyproject.toml
#      has uv-specific build config ([tool.uv.extra-build-dependencies])
#      that looks like the intended tool -- staying consistent with every
#      other script written tonight, per explicit instruction.
#
# SolarWM's docs also warn: do not install Wan/LTX/H3 into the same venv.
# This script only builds the H3 one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH." >&2
  exit 1
fi

VENV="$HERE/.venv-h3"
PY="$VENV/bin/python"

echo "[solarwm-h3-setup] creating venv ($(python3 --version 2>&1))..."
python3 -m venv "$VENV"
source "$VENV/bin/activate"
"$PY" -m pip install --upgrade pip setuptools wheel

echo "[solarwm-h3-setup] torch + torchvision (cu132, unpinned -- see script header)..."
"$PY" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

echo "[solarwm-h3-setup] H3-specific deps (diffusers, transformers, peft, imageio)..."
"$PY" -m pip install \
  diffusers==0.40.0 \
  transformers==5.12.1 peft==0.20.0 \
  "imageio[pyav]==2.37.4" imageio-ffmpeg==0.6.0

echo "[solarwm-h3-setup] installing SolarWM itself (train extras)..."
"$PY" -m pip install -e ".[train]"

echo "[solarwm-h3-setup] flash-attn (--no-build-isolation, unpinned, source build -- see script header)..."
# Confirmed the hard way on this GB300/aarch64 box: a plain `pip install
# --no-build-isolation flash-attn` here produced a flash_attn_2_cuda.so that
# crashed with `Bus error (core dumped)` on `import flash_attn` -- not a
# normal ImportError, a SIGBUS, which meant a corrupted/truncated .so (very
# possibly from an interrupted build -- e.g. Ctrl-C mid-compile). After a
# clean uninstall + rebuild, the failure mode changed to a normal
# `ImportError: undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...`
# -- i.e. flash-attn had compiled against a DIFFERENT installed torch build
# than what was present at import time (ABI mismatch on c10's CUDA symbols).
# Fix used: fully purge (pip uninstall + delete leftover flash_attn* dirs
# under site-packages + clear ~/.cache/pip) and rebuild from source with
# --no-cache-dir --no-binary flash-attn, capped at MAX_JOBS=8 to avoid an
# unbounded parallel compile thrashing/OOMing on this shared box (also a
# plausible cause of a corrupted .so). Let the build run to full completion
# -- do NOT Ctrl-C an in-progress flash-attn compile, that is exactly how an
# earlier attempt produced the SIGBUS in the first place.
"$PY" -m pip uninstall -y flash-attn 2>/dev/null || true
find "$VENV" -iname "flash_attn*" -exec rm -rf {} + 2>/dev/null || true
# UPDATE: classic flash-attn (FA2, package "flash-attn") also failed to
# compile on this box with a real g++ error building csrc/flash_attn/
# flash_api.cpp -- root cause not fully isolated (see run history), but
# strongly suspected to be an ABI mismatch against unpinned torch==2.14.0
# (too new for FA2's pinned release). This box is GB300 (Blackwell,
# sm100/sm103), for which Dao-AILab now ships a SEPARATE package, FA4
# ("flash-attn-4", CuTeDSL-based, built specifically for Hopper/Blackwell)
# -- more correct for this hardware than FA2 anyway, and less likely to hit
# the same C++/ABI compile breakage since it's not a classic setup.py
# CUDA/C++ source build. Its extras use short CUDA major.minor tags
# (cu12/cu13), NOT the full patch version (cu132 is wrong, use cu13).
# NOTE: FA4 exposes itself as `flash_attn_interface`, not `flash_attn` --
# code checking for flash-attn availability (h3_camera_infer.py's
# --attention-backend, diffusers' attention-backend dispatch) may need to
# be taught about this if FA4 doesn't register as a drop-in.
"$PY" -m pip install --pre "flash-attn-4[cu13]"

echo
echo "[solarwm-h3-setup] probing environment..."
"$PY" -m solarwm environment probe

echo
echo "Done. Activate with: source .venv-h3/bin/activate"
echo "Next: bash download_h3_models.sh"
