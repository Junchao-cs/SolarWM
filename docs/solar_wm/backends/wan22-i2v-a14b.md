# Wan2.2 I2V-A14B

Wan2.2 I2V-A14B uses the Wan image-to-video architecture with a separate
first-image condition.

## Availability and data

| Route | Training data | Status |
|---|---|---|
| Stage0.5 81f | raw-WDS | Code and weights available |
| Stage0.5 153f | `wan22-i2v-a14b-153f-480p-v1` latent | Code available; latent upload coming soon |
| Stage1 | — | Coming soon |
| Stage2 | — | Coming soon |
| Inference | raw-WDS test payload | Available for Stage0.5 81f |

Use [Download and access](../data-access.md) for raw-WDS. The
[latent-WDS release list](../latent-wds.md) tracks the 153f latent upload.

## Setup

Activate the Wan environment from
[Runtime environments](../environments.md), then set:

```bash
export SOLAR_REPO=/path/to/SolarWM
export SOLAR_MODEL_ROOT=/path/to/SolarWM-models
export SOLAR_DATA_ROOT=/path/to/SolarWM-Data/releases-v1
export SOLAR_OUTPUT_ROOT=/path/to/outputs
cd "$SOLAR_REPO"
```

To download the A14B base model and released checkpoint, first accept
the [model repository's access terms](https://huggingface.co/junchaoh-cs/SolarWM)
on Hugging Face and authenticate with the same account:

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
hf download junchaoh-cs/SolarWM \
  --include "SolarWM-14B-*/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

## Stage0.5 training

The following 81f command uses two eight-GPU nodes. Set `NODE_RANK=0` on the
first node and `NODE_RANK=1` on the second.

```bash
export NODE_RANK=0
export MASTER_ADDR=hostname-or-ip-of-node-0
export MASTER_PORT=29500

torchrun --nnodes=2 --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solar train \
  --config configs/solar_wm/wan22_i2v_a14b/train_stage0p5_fm_81f.yaml \
  --set distributed.world_size=16 \
  --set train.global_batch_size=8 \
  --set validation.sample_count=16 \
  --set model.base_path="$SOLAR_MODEL_ROOT/SolarWM-14B-base-high" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/wan14-stage0p5-81f"
```

The 153f config uses preencoded data. Once its latent generation is available,
use the same launcher with:

```bash
--config configs/solar_wm/wan22_i2v_a14b/train_stage0p5_fm_153f.yaml \
--set checkpoint.path="$SOLAR_MODEL_ROOT/SolarWM-14B-bid-stage0p5-81f/model.pt" \
--set runtime.validate_every=0 \
--set runtime.output_dir="$SOLAR_OUTPUT_ROOT/wan14-stage0p5-153f"
```

`runtime.validate_every=0` keeps the 153f training route latent-only. Remove
that override only after preparing the raw-WDS test payload used by Wan
validation.

Run `solar config resolve` with the selected config and the same `--set`
arguments before training if you want to inspect the resolved configuration.

## Inference

```bash
torchrun --standalone --nproc-per-node=8 -m solar infer \
  --config configs/solar_wm/wan22_i2v_a14b/infer_stage0p5_fm_81f.yaml \
  --set model.base_path="$SOLAR_MODEL_ROOT/SolarWM-14B-base-high" \
  --set checkpoint.path="$SOLAR_MODEL_ROOT/SolarWM-14B-bid-stage0p5-81f" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/wan14-stage0p5-81f-infer"
```

Inference reads the raw test payload selected by the config and writes videos
below `runtime.output_dir/generation`.

## Model and camera conventions

Raw records store camera poses as C2W. The A14B reader converts them to
first-frame-relative W2C and keeps the first-image condition separate from the
video latents. Released A14B checkpoints use the `linear` camera-translation
transform.
