# SolarRenderer

[SolarRenderer](https://github.com/Enderfga/SolarRenderer) renders a video from
frame-aligned structure, an optional first frame, and a text prompt. It uses
MiniMax-H3's Ref2VA transformer with a control LoRA and a four-step LoRA. This
guide covers bidirectional and block-causal inference, data encoding, and the
four training stages in SolarWM.

The [paper](https://enderfga.cn/data/solarrenderer.pdf),
[project page](https://enderfga.cn/SolarRenderer),
[weights](https://huggingface.co/Enderfga/SolarRenderer), and
[dataset](https://huggingface.co/datasets/Enderfga/SolarRenderer-Data) are
maintained by the SolarRenderer authors. Please cite both projects when using
this integration; see [Citation](../README.md#citation).

## Install

Use a separate environment for SolarRenderer. Its PyTorch and Transformers
versions differ from the SolarWM camera-model environments; do not combine the
`renderer` extra with `wan` or `dev` in the same environment. The commands below
use Python 3.10 and CUDA 12.8 PyTorch wheels. Install `ffmpeg` and `ffprobe` so
both are available on your path.

Run all commands from the SolarWM repository root:

```bash
python3.10 -m venv .venv-renderer
source .venv-renderer/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e ".[renderer]"
python -m pip install --no-build-isolation flash-attn==2.7.4.post1
python -m solarwm.renderer --help
```

Inference uses one CUDA GPU. The text encoder is released before the video
transformer is loaded. Training uses one node with eight GPUs; the
[original training guide](https://github.com/Enderfga/SolarRenderer/blob/main/docs/training.md)
reports approximately 140 GB of memory per GPU for its configurations.

## Download the models

MiniMax-H3 and the SolarRenderer adapters retain their own model licenses.
Review the [MiniMax-H3 license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE)
and [SolarRenderer weight license](https://huggingface.co/Enderfga/SolarRenderer/blob/main/LICENSE)
before downloading or using them.

Choose a directory with enough space for the base model, adapters, and outputs:

```bash
export SOLAR_RENDERER_ROOT=/path/to/solar-renderer
export H3_ROOT="$SOLAR_RENDERER_ROOT/MiniMax-H3"
export RENDERER_WEIGHTS="$SOLAR_RENDERER_ROOT/weights"
mkdir -p "$RENDERER_WEIGHTS"

python -m solarwm.renderer.prepare_h3 --dir "$H3_ROOT"
hf download Enderfga/SolarRenderer \
  solarrenderer_control_lora.safetensors \
  solarrenderer_4step_lora.safetensors \
  --local-dir "$RENDERER_WEIGHTS"
```

`prepare_h3` downloads the Ref2VA components and points the modular pipeline at the
local files. For an existing complete MiniMax-H3 download, use
`--skip-download` to configure that local copy.
SolarRenderer needs the `transformer_ref` weights from MiniMax-H3; SolarWM's
camera-model LoRAs are separate checkpoints.

## Render the included example

The [synthwave highway example](../examples/solar_renderer/sem-highway_synthwave)
contains a semantic structure video, a first frame, and a six-part prompt:

```bash
python -m solarwm.renderer \
  --h3 "$H3_ROOT" \
  --control "$RENDERER_WEIGHTS/solarrenderer_control_lora.safetensors" \
  --fewstep "$RENDERER_WEIGHTS/solarrenderer_4step_lora.safetensors" \
  --structure examples/solar_renderer/sem-highway_synthwave/structure.mp4 \
  --first examples/solar_renderer/sem-highway_synthwave/first.png \
  --prompt examples/solar_renderer/sem-highway_synthwave/prompt.txt \
  --out "$SOLAR_RENDERER_ROOT/render.mp4" \
  --seed 0
```

The output is 158 frames at 24 fps, with a landscape canvas of 1344×768 or a
portrait canvas of 768×1344. The default is four model evaluations. Keep that
default when using the released four-step adapters.

For your own inputs:

- Supply a structure video covering at least 158 frames at 24 fps. Other frame
  rates are resampled to 24 fps; longer videos use the beginning of the clip.
  Inputs are fitted to the landscape or portrait canvas.
- Use a first frame whose layout and camera match the beginning of the
  structure video. Omit `--first` for structure-and-text conditioning.
- Pass a UTF-8 text file to `--prompt`. Describe what the structure represents
  and which appearance, materials, lighting, and motion the output should use.

The prompt follows the released six-part format:

```text
subject_definitions:
summary:
retention_analysis:
detailed_description:
overall_soundscape:
non_diegetic_music:
```

Start with the included `prompt.txt` and follow the
[upstream prompt guide](https://github.com/Enderfga/SolarRenderer/blob/main/docs/prompts.md)
for the fixed input descriptions and retention rules. If you omit the first
frame, update the prompt to describe only the inputs you supply. The output of
this command is video only; the soundscape fields remain part of the model's
prompt format.

## Block-causal rendering

Download the causal adapter pair and use both files with `--causal`:

```bash
hf download Enderfga/SolarRenderer \
  solarrenderer_causal_control_lora.safetensors \
  solarrenderer_causal_4step_lora.safetensors \
  --local-dir "$RENDERER_WEIGHTS"

python -m solarwm.renderer --causal \
  --h3 "$H3_ROOT" \
  --control "$RENDERER_WEIGHTS/solarrenderer_causal_control_lora.safetensors" \
  --fewstep "$RENDERER_WEIGHTS/solarrenderer_causal_4step_lora.safetensors" \
  --structure examples/solar_renderer/sem-highway_synthwave/structure.mp4 \
  --first examples/solar_renderer/sem-highway_synthwave/first.png \
  --prompt examples/solar_renderer/sem-highway_synthwave/prompt.txt \
  --out "$SOLAR_RENDERER_ROOT/render-causal.mp4" \
  --seed 0
```

Each block contains at most five video latents. It attends to the prompt, the
first frame, structure within a six-block window, and up to five preceding
blocks of generated video. The command generates the same 158-frame clip and
writes the completed MP4 after sampling and decoding.

## Prepare training data

Download the released training shards and extract the samples:

```bash
export RENDERER_DATA="$SOLAR_RENDERER_ROOT/data"
export RENDERER_RUNS="$SOLAR_RENDERER_ROOT/runs"

hf download Enderfga/SolarRenderer-Data --repo-type dataset \
  --include "train/*" --local-dir "$RENDERER_DATA/SolarRenderer-Data"
python -m solarwm.renderer.hf_samples \
  --data "$RENDERER_DATA/SolarRenderer-Data" \
  --out "$RENDERER_DATA/samples"
```

`hf_samples` preserves each released sample's prompt and conditioning inputs.
Use `--per-source N` to extract a small subset. To supply your own data, write
one JSON object per line with a target video, prompt, and observations:

```json
{"id":"clip_0001","target":"/path/to/target.mp4","text":"subject_definitions:\n...","observations":[{"media":"/path/to/structure.mp4","frames":"all"},{"media":"/path/to/first.png","frames":[0]}]}
```

`"all"` means frame-aligned structure, `[0]` aligns a first frame to target
latent zero, and `null` denotes an unaligned reference. Encode the samples with
one worker per GPU:

```bash
H3_REF_ROOT="$H3_ROOT" torchrun --standalone --nproc-per-node=8 \
  -m solarwm.renderer.preencode \
  --samples "$RENDERER_DATA/samples/samples.jsonl" \
  --out "$RENDERER_DATA/encoded"
cat "$RENDERER_DATA"/encoded/index.*.jsonl \
  > "$RENDERER_DATA/encoded/train.jsonl"
```

Merge the worker indexes after all workers finish. To resume an interrupted
encoding run, use the same sample list, worker count, and output directory.
Use a new output directory when changing the inputs, prompts, or worker count.
Training reads the encoded samples directly. Install the corresponding
`fsspec` backend if you use remote object storage.

## Train the adapters

The four configurations in
[`configs/solar_renderer/`](../configs/solar_renderer)
use the following schedules:

| Stage | Configuration | Steps | Global batch |
|---|---|---:|---:|
| Bidirectional control | `control_lora.yaml` | 38,400 | 4 |
| Bidirectional four-step | `structure_guided_4step.yaml` | 800 | 4 |
| Causal control | `causal_control_lora.yaml` | 19,200 | 2 |
| Causal four-step | `causal_structure_guided_4step.yaml` | 1,600 | 1 |

The causal four-step recipe uses all eight GPUs for sequence parallelism. Each
frozen-base shard must stay inside one sequence-parallel group so that all
members process the same clip. Configurations that split a shard across samples
are rejected before loading the model.

Inspect a recipe without starting training:

```bash
python -m solarwm.renderer.training \
  --config configs/solar_renderer/control_lora.yaml --resolve
```

Use the SolarRenderer training entry point. The following Bash helper supplies
the shared model and data paths for each stage:

```bash
renderer_train() {
  torchrun --standalone --nproc-per-node=8 \
    -m solarwm.renderer.training \
    --set model.checkpoint_path="$H3_ROOT" \
    --set data.omni_root="$RENDERER_DATA/encoded" \
    --set data.omni_index="$RENDERER_DATA/encoded/train.jsonl" \
    "$@"
}
```

### 1. Bidirectional control

Train the structure-conditioned control adapter with flow matching:

```bash
renderer_train \
  --config configs/solar_renderer/control_lora.yaml \
  --set runtime.output_dir="$RENDERER_RUNS/control_lora"

export CONTROL_CHECKPOINT="$RENDERER_RUNS/control_lora/checkpoint_model_038400"
python -m solarwm.renderer.export_lora \
  --checkpoint "$CONTROL_CHECKPOINT" \
  --out "$RENDERER_WEIGHTS/my_control_lora.safetensors"
```

### 2. Bidirectional four-step

Download and convert the DMAD initializer, then train the acceleration adapter
with structure guidance. The control checkpoint stays frozen:

```bash
hf download ZhengmingYu/DMAD \
  minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors \
  --local-dir "$SOLAR_RENDERER_ROOT/dmad"
python -m solarwm.renderer.convert_dmad \
  --h3 "$H3_ROOT" \
  --src "$SOLAR_RENDERER_ROOT/dmad/minimax_h3/dmad_minimax_h3_4step_lora_critic.safetensors" \
  --out "$RENDERER_WEIGHTS/dmad_4step_lora.safetensors"

renderer_train \
  --config configs/solar_renderer/structure_guided_4step.yaml \
  --set train.dmd.init_lora="$RENDERER_WEIGHTS/dmad_4step_lora.safetensors" \
  --set train.dmd.control="$CONTROL_CHECKPOINT" \
  --set runtime.output_dir="$RENDERER_RUNS/structure_guided_4step"

python -m solarwm.renderer.export_lora \
  --checkpoint "$RENDERER_RUNS/structure_guided_4step/checkpoint_model_000800" \
  --out "$RENDERER_WEIGHTS/my_4step_lora.safetensors"
```

`train.dmd.structure_guidance` is 1.5 in the released recipe. Render with
`my_control_lora.safetensors` and `my_4step_lora.safetensors` using the
bidirectional command above.

### 3. Causal control

Initialize from the bidirectional control checkpoint and train with
block-causal attention and teacher forcing:

```bash
renderer_train \
  --config configs/solar_renderer/causal_control_lora.yaml \
  --set checkpoint.initialization.student.path="$CONTROL_CHECKPOINT" \
  --set runtime.output_dir="$RENDERER_RUNS/causal_control_lora"

export CAUSAL_CONTROL_CHECKPOINT="$RENDERER_RUNS/causal_control_lora/checkpoint_model_019200"
python -m solarwm.renderer.export_lora \
  --checkpoint "$CAUSAL_CONTROL_CHECKPOINT" \
  --out "$RENDERER_WEIGHTS/my_causal_control_lora.safetensors"
```

### 4. Causal four-step

The causal control checkpoint is frozen in the student. The acceleration
adapter starts from the bidirectional four-step adapter and learns from its
own block-causal rollouts, with the bidirectional control checkpoint as teacher:

```bash
renderer_train \
  --config configs/solar_renderer/causal_structure_guided_4step.yaml \
  --set train.dmd.student_init="$CAUSAL_CONTROL_CHECKPOINT" \
  --set train.dmd.accel_init="$RENDERER_WEIGHTS/my_4step_lora.safetensors" \
  --set train.dmd.control="$CONTROL_CHECKPOINT" \
  --set runtime.output_dir="$RENDERER_RUNS/causal_structure_guided_4step"

python -m solarwm.renderer.export_lora \
  --checkpoint "$RENDERER_RUNS/causal_structure_guided_4step/checkpoint_model_001600" \
  --out "$RENDERER_WEIGHTS/my_causal_4step_lora.safetensors"
```

Render with `--causal`, `my_causal_control_lora.safetensors`, and
`my_causal_4step_lora.safetensors`. The export command reads the live adapter
from `adapter.pt`. If you change the training schedule, use the checkpoint
corresponding to your completed run.

## Further resources

- [Data processing and captioning pipeline](https://github.com/Enderfga/SolarRenderer/tree/data)
- [Benchmark and evaluation instructions](https://github.com/Enderfga/SolarRenderer/blob/main/eval/README.md)
- [SolarWM MiniMax-H3 camera-model guide](backends/minimax-h3.md)
