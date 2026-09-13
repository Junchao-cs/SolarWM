# MiniMax-H3

MiniMax-H3 supports Stage0.5 FM, Stage1 TF-AnyFlow, and Stage2 SGF.
Training and fixed 158-frame inference use [preencoded data](../latent-wds.md).
Stage2 also supports full-length inference from a first image and camera trajectory.

## Setup

Activate the H3 environment from
[Runtime environments](../../environments/README.md), then set:

```bash
export SOLAR_REPO=/path/to/SolarWM
export SOLAR_MODEL_ROOT=/path/to/SolarWM-models
export SOLAR_DATA_ROOT=/path/to/SolarWM-Data/releases-v1
export SOLAR_OUTPUT_ROOT=/path/to/outputs
cd "$SOLAR_REPO"
```

Accept the model repository's access terms on Hugging Face, then download the
base model if it is not already installed:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf download junchaoh-cs/SolarWM-H3-33B \
  --include "SolarWM-h3-33B-base/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

For inference only, download `SolarWM-h3-33B-sgf-stage2-158f`; Stage0.5 and
Stage1 checkpoints are not needed. The base model and the input dependencies
used by your inference command are still required.

```bash
hf download junchaoh-cs/SolarWM-H3-33B \
  --include "SolarWM-h3-33B-sgf-stage2-158f/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

The latent package supplies the files in `H3_SUPPORT`. Set the paths and
rendezvous address below; set `NODE_RANK` separately on each node.

```bash
export H3_BASE="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-base"
export H3_STAGE2_CHECKPOINT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-sgf-stage2-158f"
export H3_SUPPORT="$SOLAR_DATA_ROOT/latent-wds/minimax-h3-158f-768p-nomind-v1/support"
export NNODES=32
export NODE_RANK=0
export MASTER_ADDR=hostname-or-ip-of-node-0
export MASTER_PORT=29500
```

## Stage0.5 training

```bash
torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage0p5-158f-lora384-sp2.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage0p5-158f"
```

For a two-node run, set `NNODES=2` and add
`--set distributed.world_size=16 --set train.global_batch_size=8` to Stage0.5
or Stage1 training, or `--set train.global_batch_size=4` with the same world-size
override for Stage2. These smaller runs change the global batch size; the
unmodified configs retain the released recipes.

## Stage1 / Stage2 setup

For training across stages, download the three EMA packages from the
[SolarWM-H3 weight repository](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B).
Keep each directory intact under `SOLAR_MODEL_ROOT`.

```bash
hf download junchaoh-cs/SolarWM-H3-33B \
  --include "SolarWM-h3-33B-bid-stage0p5-158f/**" \
            "SolarWM-h3-33B-tf-stage1-158f/**" \
            "SolarWM-h3-33B-sgf-stage2-158f/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

| Package | EMA checkpoint |
|---|---|
| `SolarWM-h3-33B-bid-stage0p5-158f` | Stage0.5 step 10500 |
| `SolarWM-h3-33B-tf-stage1-158f` | Stage1 step 3000 |
| `SolarWM-h3-33B-sgf-stage2-158f` | Stage2 step 1200 |

Validation selects fixed cases from the test index automatically. Stage1 also
reads complete camera trajectories from the raw test data.

```bash
export H3_STAGE0P5_INIT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-bid-stage0p5-158f"
export H3_STAGE1_CHECKPOINT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-tf-stage1-158f"
```

### Stage1 TF-AnyFlow

```bash
torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage1-158f-lora384-w6-sp2.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set checkpoint.initialization.student.path="$H3_STAGE0P5_INIT" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage1-anyflow-158f"
```

### Stage2 SGF

```bash
torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage2-158f-lora384-w6-sp4.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set checkpoint.initialization.student.path="$H3_STAGE1_CHECKPOINT" \
  --set checkpoint.initialization.teacher.path="$H3_STAGE0P5_INIT" \
  --set checkpoint.initialization.critic.path="$H3_STAGE0P5_INIT" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage2-sgf-158f"
```

## Resume training

Use the original resolved configuration and a complete training checkpoint:

```bash
export H3_PREVIOUS_RUN="$SOLAR_OUTPUT_ROOT/h3-stage2-sgf-158f"

torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config "$H3_PREVIOUS_RUN/resolved-config.json" \
  --set checkpoint.resume_from="$H3_PREVIOUS_RUN/checkpoint_model_000200" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage2-sgf-resumed"
```

## Inference

### Stage0.5

The released stage packages contain EMA weights.

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-158f-lora384-sp2.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-bid-stage0p5-158f" \
  --set checkpoint.weight_source=ema \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage0p5-158f-infer"
```

### Stage2 SGF

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-stage2-158f-sp4.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$H3_STAGE2_CHECKPOINT" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage2-sgf-infer"
```

For a checkpoint without EMA weights, add `--set checkpoint.weight_source=live`.

### Full-length Stage2 inference

Generate each test video at its original length and frame rate using 8 GPUs
(SP8, NFE4, W6). Only the first image is encoded; the supplied camera trajectory
controls the generated video.

Download the [standalone test set](../data-access.md#standalone-test-set) first.
Create a one-case plan from its index; select more complete rows for a larger run:

```bash
export H3_TEST_PLAN="$SOLAR_OUTPUT_ROOT/h3-test-plan.json"
python - <<'PY'
import gzip
import json
import os
from pathlib import Path

index = Path(os.environ["SOLAR_TEST_ROOT"]) / "indexes/all.jsonl.gz"
with gzip.open(index, "rt") as handle:
    rows = [next(json.loads(line) for line in handle if line.strip())]
plan = Path(os.environ["H3_TEST_PLAN"])
plan.parent.mkdir(parents=True, exist_ok=True)
plan.write_text(json.dumps(rows, indent=2) + "\n")
PY
```

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-stage2-source-length-sp8.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$H3_STAGE2_CHECKPOINT" \
  --set checkpoint.weight_source=ema \
  --set inference.expected_step=1200 \
  --set inference.plan="$H3_TEST_PLAN" \
  --set inference.dataset_root="$SOLAR_TEST_ROOT" \
  --set inference.work_dir="$SOLAR_OUTPUT_ROOT/h3-condition-cache" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage2-full-length"
```

`inference.plan` is a JSON list of selected raw test-index rows; the dataset root
can be a local path or `gs://` URI. Set `inference.expected_step` to the checkpoint
step and use `checkpoint.weight_source=live` for LIVE weights.

Keep every selected row intact. The reader uses these index fields:

| Fields | Meaning |
|---|---|
| `sample_id`, `key`, `dataset` | Sample identity; each `key` must be unique in the plan |
| `num_frames`, `fps`, `height`, `width`, `caption` | Source geometry, playback rate, and prompt |
| `shard`, `shard_size` | Archive path relative to the dataset root and its byte size |
| `video_member`, `camera_member`, `intrinsics_member`, `manifest_member` | Member paths within the archive |
| `shard_generation` | Also required for GCS inputs |

The camera member contains absolute `c2w` matrices, one per source frame;
intrinsics use the source image coordinates. Results include videos, comparison
previews and FPS measurements. For videos longer than 960 frames, add
`--set inference.stream_decode=true`.

## Optional preencoding

To prepare a new latent generation from [raw-WDS](../data-access.md):

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm preencode \
  --config configs/examples/minimax_h3/preencode-158f.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set preencode.output_root="$SOLAR_OUTPUT_ROOT/preencoded/minimax-h3-158f-768p-nomind-v1" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-preencode-158f"
```

## License

The MiniMax-H3 base model and released adapters remain subject to the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
included with the model packages. Review its terms, including territorial
restrictions, before download, use, or redistribution. SolarWM's code license
does not replace it.
