# SolarRenderer training

Adapted from the upstream [training guide at revision
`e018b0f`](https://github.com/Enderfga/SolarRenderer/blob/e018b0f673de865711fe99844d3976865722b8d0/docs/training.md),
using the integrated `solarwm.renderer` APIs and configurations. The
[integrated tutorial](README.md) remains the end-to-end reference.

## Prerequisites and resources

Complete [installation](README.md#install) and
[model preparation](README.md#download-the-models) first. Those steps define
`SOLAR_RENDERER_ROOT`, `H3_ROOT`, and `RENDERER_WEIGHTS`. Run all commands from
the repository root in the renderer environment.

The four recipes target one eight-GPU node with FSDP and sequence parallelism.
The upstream guide reports about 140 GB of memory per GPU, about 84 seconds
per control step, and three to eight minutes per post-training step on its
setup. Its full schedules therefore take weeks or days. These are upstream
reference figures; this integration has not re-benchmarked them. The released
weights used the same recipe and sample counts at larger global batches.

Use local paths for the prepared H3 model and run/checkpoint directories.
Training data and encoded samples can be local paths or object-storage URLs;
install the matching `fsspec` backend for remote data. The commands below use
local paths, including the worker-index merge. Replace the `/path/to/...`
values before launching.

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
[`configs/solar_renderer/`](../../configs/solar_renderer)
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
[bidirectional rendering command](inference.md#render-a-clip).

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


## Related workflows

- [Inference](inference.md): use exported adapters for bidirectional or causal rendering.
- [Prompts and first frames](prompts.md): prepare the conditioning text and appearance reference.
- [Data processing and captioning pipeline](https://github.com/Enderfga/SolarRenderer/tree/data).
- [Benchmark and evaluation](../../eval/solar_renderer/README.md).
