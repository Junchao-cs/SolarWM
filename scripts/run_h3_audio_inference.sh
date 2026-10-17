#!/usr/bin/env bash
set -euo pipefail

# Fixed-length MiniMax-H3 inference with native stereo audio enabled.
# Set the paths below (or export them) before running this script.
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
PYTHON_BIN=${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}
H3_BASE=${H3_BASE:?Set H3_BASE to the MiniMax-H3 base checkpoint}
H3_ADAPTER=${H3_ADAPTER:?Set H3_ADAPTER to the SolarWM LoRA checkpoint}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to a prepared 158-frame H3 preencoded release}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_ROOT/outputs/h3-audio-inference}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 2
fi
for path in "$H3_BASE" "$H3_ADAPTER" "$DATA_ROOT"; do
  [[ -e "$path" ]] || { echo "Path not found: $path" >&2; exit 2; }
done

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node="$NPROC_PER_NODE" \
  -m solarwm infer \
  --config configs/examples/minimax_h3/infer-158f-lora384-sp2.yaml \
  --set distributed.world_size="$NPROC_PER_NODE" \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$H3_ADAPTER" \
  --set data.index_root="$DATA_ROOT" \
  --set data.transport.root="$DATA_ROOT" \
  --set data.train_index=index.jsonl \
  --set data.test_index=index.jsonl \
  --set data.silence_latents_path="$DATA_ROOT/silence/h3-silence-158f.safetensors" \
  --set data.encoder_contract_path="$DATA_ROOT/encoder-contract.json" \
  --set validation.generate_audio=true \
  --set runtime.output_dir="$OUTPUT_DIR"

echo "Audio inference outputs: $OUTPUT_DIR"
