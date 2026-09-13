# Wan2.2 TI2V-5B

Wan2.2 TI2V-5B is the complete SolarWM training route:

```text
Stage0.5 bidirectional FM -> Stage1 TF-AnyFlow -> Stage2 DMD via SGF
```

## Availability and data

| Route | Training data | Released weights |
|---|---|---|
| Stage0.5 153f | `wan22-ti2v5b-153f-480p-v1` latent | Available |
| Stage0.5 81f | raw-WDS | Available |
| Stage1 TF-AnyFlow 81f | raw-WDS | Available |
| Stage2 DMD via SGF 81f | raw-WDS | Available |
| Inference | raw-WDS test payload | Available |

The recommended starting point is the 153f latent-only recipe below. Its
latent generation is available on
[ModelScope](https://modelscope.ai/datasets/Junchao-cs/SolarWM-Data_Latent-WDS_wan22-ti2v5b-153f-480p-v1).
Use [Download and access](../data-access.md) only if you need raw-WDS for the
81f, Stage1, Stage2, or inference routes.

## Setup

Activate the Wan environment from
[Runtime environments](../../environments/README.md), then set:

```bash
export SOLAR_REPO=/path/to/SolarWM
export SOLAR_MODEL_ROOT=/path/to/SolarWM-models
export SOLAR_DATA_ROOT=/path/to/SolarWM-Data/releases-v1
export SOLAR_OUTPUT_ROOT=/path/to/outputs
cd "$SOLAR_REPO"
```

Download the Wan2.2-5B base model and released stage checkpoints:

```bash
python -m pip install --upgrade huggingface_hub
hf download junchaoh-cs/SolarWM \
  --include "SolarWM-5B-*/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

Keep each downloaded checkpoint directory intact. Commands below refer to the
`model.pt` file when initializing training and to the directory when running
standalone inference.

## Full-resume compatibility

To resume training, use the complete `checkpoint_model_*` directory with the
same stage and objective. A standalone `model.pt` is only for weights
initialization; it does not restore the optimizer, data position, or RNG state.

Stage0.5/Stage1 full resume requires `rank_runtime_state` in `model.pt`, with
one reader and RNG state per rank. Stage2 stores these under `rng_state_by_rank`
in `model.pt` and also requires the matching `critic.pt`. Checkpoints missing
the reader/RNG state are rejected for full resume; they remain usable for
weights-only initialization where the stage contract permits it.

## Recommended: Stage0.5 153f with released latents

This single-node example uses the released latent generation and does not
require raw-WDS. Periodic video validation is disabled because the Wan
validation recipe reads raw video.

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm train \
  --config configs/examples/wan22_ti2v_5b/train_stage0p5_fm_153f.yaml \
  --set distributed.world_size=8 \
  --set train.global_batch_size=8 \
  --set model.base_path="$SOLAR_MODEL_ROOT/SolarWM-5B-base" \
  --set checkpoint.path="$SOLAR_MODEL_ROOT/SolarWM-5B-bid-stage0p5-81f/model.pt" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set runtime.validate_every=0 \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/wan5-stage0p5-153f"
```

Run `solarwm config resolve` with the same config and `--set` arguments before
training if you want to inspect the resolved configuration.

## Full raw-data training route

The remaining stages use raw-WDS. The commands below show one eight-GPU node;
change the world size and global batch together when scaling out.

### Stage0.5 81f

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm train \
  --config configs/examples/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml \
  --set distributed.world_size=8 \
  --set train.global_batch_size=8 \
  --set validation.sample_count=8 \
  --set model.base_path="$SOLAR_MODEL_ROOT/SolarWM-5B-base" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/wan5-stage0p5-81f"
```

### Stage1 TF-AnyFlow 81f

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm train \
  --config configs/examples/wan22_ti2v_5b/train_stage1_tf_anyflow_v1_5_81f.yaml \
  --set distributed.world_size=8 \
  --set train.global_batch_size=8 \
  --set validation.sample_count=8 \
  --set model.base_path="$SOLAR_MODEL_ROOT/SolarWM-5B-base" \
  --set checkpoint.path="$SOLAR_MODEL_ROOT/SolarWM-5B-bid-stage0p5-153f/model.pt" \
  --set checkpoint.weights=ema \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/wan5-stage1-anyflow-81f"
```

The released `SolarWM-5B-tf-stage1-81f` weights use TF-AnyFlow v1.5. The
optional TF-FM baseline is available as
`train_stage1_tf_fm_81f.yaml`, but it is not the released Stage2 initializer.

### Stage2 DMD via SGF 81f

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm train \
  --config configs/examples/wan22_ti2v_5b/train_stage2_sgf_81f.yaml \
  --set distributed.world_size=8 \
  --set train.global_batch_size=8 \
  --set validation.sample_count=8 \
  --set model.base_path="$SOLAR_MODEL_ROOT/SolarWM-5B-base" \
  --set checkpoint.roles.student.path="$SOLAR_MODEL_ROOT/SolarWM-5B-tf-stage1-81f/model.pt" \
  --set checkpoint.roles.teacher.path="$SOLAR_MODEL_ROOT/SolarWM-5B-bid-stage0p5-81f/model.pt" \
  --set checkpoint.roles.critic.path="$SOLAR_MODEL_ROOT/SolarWM-5B-bid-stage0p5-81f/model.pt" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/wan5-stage2-sgf-81f"
```

## Inference

| Route | Config | GPUs | Checkpoint directory |
|---|---|---:|---|
| Stage0.5 81f | `infer_stage0p5_fm_81f.yaml` | 8 | `SolarWM-5B-bid-stage0p5-81f` |
| Stage0.5 153f | `infer_stage0p5_fm_153f.yaml` | 8 | `SolarWM-5B-bid-stage0p5-153f` |
| Stage1 TF-AnyFlow | `infer_stage1_tf_anyflow_v1_5_81f.yaml` | 8 | `SolarWM-5B-tf-stage1-81f` |
| Stage2 DMD via SGF | `infer_stage2_sgf_camera_length.yaml` | 1 | `SolarWM-5B-sgf-stage2-81f` |

For example, run the final Stage2 checkpoint with:

```bash
torchrun --standalone --nproc-per-node=1 -m solarwm infer \
  --config configs/examples/wan22_ti2v_5b/infer_stage2_sgf_camera_length.yaml \
  --set model.base_path="$SOLAR_MODEL_ROOT/SolarWM-5B-base" \
  --set checkpoint.path="$SOLAR_MODEL_ROOT/SolarWM-5B-sgf-stage2-81f" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set inference.run_id=my-camera-run \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/wan5-stage2-sgf-infer"
```

The Stage2 command loads the checkpoint directory as one model, resolves its
weight role from `release-manifest.json` (`ema` for the published checkpoint),
and uses the longest camera-backed horizon available for each selected test
sample. The horizon is rounded down to complete three-latent chunks; at the
configured 16 fps, a 960-frame camera track produces 240 latents and 957 model
frames. Publication repeats the final generated frame three times so the
generated and comparison MP4s, and their camera trajectory, all contain 960
frames. Outputs longer than 60 latents are VAE-decoded as consecutive 60-latent
tiles with one continuous temporal cache. Hour-scale outputs use smaller
decode tiles and stream video to disk to bound memory use.

Camera-length inference defaults to `inference.output_layout=dataset_triplet_v1`.
`runtime.output_dir` is the shared publication root, while `inference.run_id`
selects a create-only provenance transaction beneath `runs/`. Every selected
index row must provide `clip_id`. The physical dataset is `physical_generation`
when that field is non-empty, and otherwise `dataset`; outputs are:

```text
runtime.output_dir/
  generate/<physical_dataset>/<clip_id>.mp4
  compare/<physical_dataset>/<clip_id>.mp4
  camera/<physical_dataset>/<clip_id>.npy
  runs/<run_id>/
    generation/...
    publication/...
    resolved-config.json
    launch-manifest.json
    run-result.json
    COMPLETE.json
```

The camera file is the source authoritative absolute C2W trajectory selected
at the published frame timestamps and cast directly to contiguous float64 for
compatibility. It is never the rebased or translation-transformed model camera.
All output artifacts are create-only, and consumers must require the run-level
`COMPLETE.json`. A run ID cannot be reused. The dataset-triplet layout currently
requires a single shared-filesystem node; multi-node node-local output is
rejected. Set `inference.output_layout=transaction_v1` only when the original
`runtime.output_dir/generation` layout is required.

Override `data.test_index` to select a different raw-WDS test index.

The other released checkpoints use the same command with the config, GPU
count, checkpoint directory, and output directory shown above. The fixed
237-frame Stage2 validation-parity route remains available as
`infer_stage2_sgf_81f.yaml`.
