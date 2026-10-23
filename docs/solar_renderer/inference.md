# SolarRenderer inference

Adapted from the upstream [inference guide at revision
`e018b0f`](https://github.com/Enderfga/SolarRenderer/blob/e018b0f673de865711fe99844d3976865722b8d0/docs/inference.md).
The [integrated tutorial](README.md) is the end-to-end reference for SolarWM's
renderer commands. Run the commands below from the repository root.

## Install and download

Follow [Install](README.md#install) to create the separate renderer environment,
then [Download the models](README.md#download-the-models) to prepare MiniMax-H3
Ref2VA and the bidirectional adapter pair. Those steps define `H3_ROOT`,
`RENDERER_WEIGHTS`, and `SOLAR_RENDERER_ROOT`, which are used below.

Inference requires one CUDA GPU, FlashAttention, and `ffmpeg` / `ffprobe` on
the path. The text encoder is released before loading the video transformer.
Use `python -m solarwm.renderer --help` to inspect the integrated CLI.

## Prepare inputs

- `structure.mp4`: frame-aligned structure covering at least 158 frames at
  24 fps, about 6.6 seconds. Other frame rates are resampled to 24 fps; longer
  clips use their beginning. The input is fitted to a 1344×768 landscape canvas
  or a 768×1344 portrait canvas.
- `first.png`, optional: a first frame in the desired appearance, with the
  same camera, subjects, and layout as the start of the structure video.
- `prompt.txt`: a UTF-8 prompt in the six-part training format. Name the
  structure representation and describe the intended materials, lighting,
  motion, and appearance. See [Prompts and first frames](prompts.md).

The included [synthwave highway example](../../examples/solar_renderer/sem-highway_synthwave)
contains all three inputs. The renderer requires a structure video; when
omitting `--first`, use a prompt describing structure-and-text conditioning.

## Render a clip

```bash
python -m solarwm.renderer \
  --h3 "$H3_ROOT" \
  --control "$RENDERER_WEIGHTS/solarrenderer_control_lora.safetensors" \
  --fewstep "$RENDERER_WEIGHTS/solarrenderer_4step_lora.safetensors" \
  --structure structure.mp4 --first first.png --prompt prompt.txt \
  --out "$SOLAR_RENDERER_ROOT/render.mp4" --seed 0
```

The output is a video-only MP4 with 158 frames at 24 fps. The default is four
model evaluations; retain it for the released four-step adapters. Omit
`--first first.png` for structure-and-text conditioning. `--device cuda:0`
selects the CUDA device; replace `0` when choosing another visible GPU.

The upstream guide reports roughly four minutes for encoding and three and a
half minutes for sampling on its single-GPU setup. These are upstream reference
figures, not a benchmark of this integration; hardware and environment affect
the runtime.

## Block-causal rendering

Download the causal pair and use both causal adapters with `--causal`:

```bash
hf download Enderfga/SolarRenderer \
  solarrenderer_causal_control_lora.safetensors \
  solarrenderer_causal_4step_lora.safetensors \
  --local-dir "$RENDERER_WEIGHTS"

python -m solarwm.renderer --causal \
  --h3 "$H3_ROOT" \
  --control "$RENDERER_WEIGHTS/solarrenderer_causal_control_lora.safetensors" \
  --fewstep "$RENDERER_WEIGHTS/solarrenderer_causal_4step_lora.safetensors" \
  --structure structure.mp4 --first first.png --prompt prompt.txt \
  --out "$SOLAR_RENDERER_ROOT/render-causal.mp4" --seed 0
```

Each block has at most five video latents, roughly 20 frames. It attends to the
prompt, the first frame, a six-block window of structure, and up to five
preceding generated-video blocks. Four sampling steps produce a block before
it is committed to the key/value cache. No block sees future structure or
future generated video.

The integrated command writes the completed 158-frame MP4 after sampling and
decoding. It does not expose a live frame stream. The [Reactor integration](../../reactor/solar_renderer/README.md)
documents its own interactive interface.

## Related workflows

- [Benchmark and evaluation](../../eval/solar_renderer/README.md): render and evaluate benchmark cases.
- [ComfyUI integration](../../comfyui/solar_renderer/README.md): use the renderer through nodes.
- [Training](training.md): create and export your own adapter pair.
- [Prompts and first frames](prompts.md): caption and first-frame preparation.
