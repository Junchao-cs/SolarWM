# Quickstart

This guide runs MiniMax-H3 Stage0.5 training and Stage2 inference with released
158-frame preencoded latents. Neither requires raw-WDS. For inference only,
complete setup and downloads, then skip to [Stage2 inference](#5-run-stage2-inference).

## 1. Set up the H3 environment

Wan, LTX, and MiniMax-H3 use separate environments. Follow the H3 setup in
[Runtime environments](../environments/README.md), activate it, and install
SolarWM:

```bash
export SOLAR_REPO=/path/to/SolarWM
cd "$SOLAR_REPO"
python -m pip install -e .
solarwm environment probe
```

## 2. Download weights and data

Choose local directories for the model, data, and outputs:

```bash
export SOLAR_MODEL_ROOT=/path/to/SolarWM-models
export SOLAR_DATA_HOME=/path/to/SolarWM-Data
export SOLAR_DATA_ROOT="$SOLAR_DATA_HOME/releases-v1"
export SOLAR_OUTPUT_ROOT=/path/to/outputs
mkdir -p "$SOLAR_MODEL_ROOT" "$SOLAR_DATA_HOME" "$SOLAR_OUTPUT_ROOT"
```

Accept the [H3 model repository's](https://huggingface.co/junchaoh-cs/SolarWM-H3-33B)
access terms, then download the base model, the Stage2 EMA checkpoint, and the
data indexes. The Stage2 checkpoint is only needed for inference.

```bash
python -m pip install --upgrade huggingface_hub
hf auth login

hf download junchaoh-cs/SolarWM-H3-33B \
  --include "SolarWM-h3-33B-base/**" "SolarWM-h3-33B-sgf-stage2-158f/**" \
  --local-dir "$SOLAR_MODEL_ROOT"

hf download junchaoh-cs/SolarWM-Data \
  --repo-type dataset \
  --exclude "SolarWM-Data-Annotation/**" \
  --local-dir "$SOLAR_DATA_HOME"
```

Download `minimax-h3-158f-768p-nomind-v1` from
[ModelScope International](https://modelscope.ai/datasets/Junchao-cs/SolarWM-Data_Latent-WDS_minimax-h3-158f-768p-nomind-v1)
or [ModelScope China](https://modelscope.cn/datasets/junchao2003/SolarWM-Data_Latent-WDS_minimax-h3-158f-768p-nomind-v1)
and place it at:

```text
$SOLAR_DATA_ROOT/latent-wds/minimax-h3-158f-768p-nomind-v1/
```

The main data repository supplies the matching recipe indexes. Keep the latent
generation's `support/` directory alongside its shards, then set:

```bash
export H3_BASE="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-base"
export H3_STAGE2_CHECKPOINT="$SOLAR_MODEL_ROOT/SolarWM-h3-33B-sgf-stage2-158f"
export H3_SUPPORT="$SOLAR_DATA_ROOT/latent-wds/minimax-h3-158f-768p-nomind-v1/support"
```

## 3. Check the configuration

Resolve the example with your local paths before starting training:

```bash
solarwm config resolve \
  --config configs/examples/minimax_h3/stage0p5-158f-lora384-sp2.yaml \
  --set distributed.world_size=8 \
  --set train.global_batch_size=4 \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage0p5-158f"
```

## 4. Launch training

The following command runs on one eight-GPU node with a smaller global batch
than the default training config:

```bash
torchrun --standalone --nproc-per-node=8 \
  -m solarwm train \
  --config configs/examples/minimax_h3/stage0p5-158f-lora384-sp2.yaml \
  --set distributed.world_size=8 \
  --set train.global_batch_size=4 \
  --set model.checkpoint_path="$H3_BASE" \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage0p5-158f"
```

Stage0.5 validation also uses the preencoded data and remains enabled. The
output directory contains the resolved configuration, launch manifest,
checkpoints, and validation results.

## 5. Run Stage2 inference

Generate 158-frame videos with the released Stage2 EMA checkpoint:

```bash
torchrun --standalone --nproc-per-node=8 -m solarwm infer \
  --config configs/examples/minimax_h3/infer-stage2-158f-sp4.yaml \
  --set model.checkpoint_path="$H3_BASE" \
  --set checkpoint.resume_from="$H3_STAGE2_CHECKPOINT" \
  --set checkpoint.weight_source=ema \
  --set data.index_root="$SOLAR_DATA_ROOT" \
  --set data.transport.root="$SOLAR_DATA_ROOT" \
  --set data.silence_latents_path="$H3_SUPPORT/h3_silence_153_158_170.safetensors" \
  --set data.encoder_contract_path="$H3_SUPPORT/encoder_contract.json" \
  --set runtime.output_dir="$SOLAR_OUTPUT_ROOT/h3-stage2-infer"
```

## Next steps

- [MiniMax-H3](backends/minimax-h3.md): Stage1/Stage2 training, checkpoint
  resume, full-length inference, and preencoding.
- [Wan2.2 TI2V-5B](backends/wan22-ti2v-5b.md)
- [Wan2.2 I2V-A14B](backends/wan22-i2v-a14b.md)
- [LTX-2.5](backends/ltx25.md)
- [Download and access](data-access.md): raw-WDS and preencoded latent options.
