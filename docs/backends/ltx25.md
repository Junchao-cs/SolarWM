# LTX-2.5

The LTX-2.5 backend supports Stage0.5 native rectified-flow LoRA training,
inference, and preencoding for 153-frame 512×768 video.

## Availability and data

| Route | Data | Status |
|---|---|---|
| Stage0.5 preencoded | `ltx-153f-h512-w768` latent | Code and weights available; latent upload coming soon |
| Stage0.5 online | raw-WDS | Available |
| Inference | `ltx-153f-h512-w768` latent | Code and weights available; latent upload coming soon |
| Preencoding | raw-WDS input | Available |

Preencoded training is the recommended route because it avoids running the VAE
and text encoder during every training step. Track the payload in the
[latent-WDS release list](../latent-wds.md), or use
[Download and access](../data-access.md) for raw-WDS.

## Setup

Activate the LTX environment from
[Runtime environments](../../environments/README.md), then set:

```bash
export SOLAR_REPO=/path/to/SolarWM
export SOLAR_MODEL_ROOT=/path/to/SolarWM-models
export SOLAR_DATA_ROOT=/path/to/SolarWM-Data/releases-v1
export SOLAR_OUTPUT_ROOT=/path/to/outputs
cd "$SOLAR_REPO"
```

Download the LTX base model and released adapter:

```bash
python -m pip install --upgrade huggingface_hub
hf download junchaoh-cs/SolarWM \
  --include "SolarWM-ltx-22B-*/**" \
  --local-dir "$SOLAR_MODEL_ROOT"
```

Set the model paths used below:

```bash
export LTX_BASE="$SOLAR_MODEL_ROOT/SolarWM-ltx-22B-base"
export LTX_TRANSFORMER="$LTX_BASE/diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
export LTX_VAE="$LTX_BASE/vae/ltx-2.5-video-vae-bf16.safetensors"
export LTX_GEMMA="$LTX_BASE/text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
export LTX_NEGATIVE="$LTX_BASE/conditioning/ltx25-negative-caption.safetensors"
```

## Preencoded training

The released training profile uses two eight-GPU nodes. Set `NODE_RANK=0` on
the first node and `NODE_RANK=1` on the second.

```bash
export NODE_RANK=0
export MASTER_ADDR=hostname-or-ip-of-node-0
export MASTER_PORT=29500

torchrun --nnodes=2 --node-rank="$NODE_RANK" --nproc-per-node=8 \
  --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
  -m solarwm train \
  --config configs/examples/ltx25/stage0p5-train-153f-lora384-sp2.yaml \
  --set distributed.world_size=16 \
  --set distributed.local_world_size=8 \
  --set train.global_batch_size=16 \
  --set validation.sample_count=16 \
  --set model.checkpoint_path="$LTX_TRANSFORMER" \
  --set model.codec.video_vae_path="$LTX_VAE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set validation.inference.negative_caption_cache="$LTX_NEGATIVE" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/ltx-stage0p5-153f"
```

Run `solarwm config resolve` with the same config and `--set` arguments before
training if you want to inspect the resolved configuration.

New LTX checkpoints restore all reader and RNG state. Older checkpoints remain
loadable, but lack the Python and NumPy RNG state needed for an exact resume.

## Inference

Inference uses two GPUs and the released adapter:

```bash
torchrun --standalone --nproc-per-node=2 -m solarwm infer \
  --config configs/examples/ltx25/stage0p5-infer-153f.yaml \
  --set model.checkpoint_path="$LTX_TRANSFORMER" \
  --set model.codec.video_vae_path="$LTX_VAE" \
  --set model.adapter_checkpoint_path="$SOLAR_MODEL_ROOT/SolarWM-ltx-22B-bid-stage0p5-153f" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set inference.negative_caption_cache="$LTX_NEGATIVE" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/ltx-stage0p5-153f-infer"
```

## Optional raw-WDS workflows

Online training uses
`configs/examples/ltx25/stage0p5-train-online-153f-unpaired.yaml` with the same
training command. Add
`--set model.codec.gemma4_path="$LTX_GEMMA"` and choose a new output directory.

To create a new LTX latent generation from raw-WDS:

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm preencode \
  --config configs/examples/ltx25/preencode-153f.yaml \
  --set model.checkpoint_path="$LTX_TRANSFORMER" \
  --set model.codec.video_vae_path="$LTX_VAE" \
  --set model.codec.gemma4_path="$LTX_GEMMA" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set preencode.output_root="$SOLAR_OUTPUT_ROOT/preencoded/ltx-153f-h512-w768" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/ltx-preencode-153f"
```

## Model and camera conventions

Raw records store camera poses as C2W. LTX preencoding converts them to
first-frame-relative W2C with normalized intrinsics, which is also the camera
format stored in the released LTX latent generation. Released LTX configs use
the `linear` camera-translation transform.

## License

The LTX base model and released adapter remain subject to the LTX Community
License included with the model packages. Review that license before use or
redistribution; the SolarWM code license does not replace it.
