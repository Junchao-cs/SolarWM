# SolarRenderer inference

Adapted from the upstream [inference guide at revision
`41e136e`](https://github.com/Enderfga/SolarRenderer/blob/41e136e64b30d4aa0095070ce055ff00ae90fe3b/docs/inference.md).
The [integrated tutorial](README.md) is the end-to-end reference for SolarWM's
renderer commands. Run the commands below from the repository root.

## Install and download

Follow [Install](README.md#install) to create the separate renderer environment,
then [Download the models](README.md#download-the-models) to prepare MiniMax-H3
Ref2VA and the bidirectional adapter pair. Those steps define `H3_ROOT`,
`RENDERER_WEIGHTS`, and `SOLAR_RENDERER_ROOT`, which are used below.

Inference requires one CUDA GPU, FlashAttention, and `ffmpeg` / `ffprobe` on
the path. The text encoder is released before loading the video transformer.
Use `python -m solar.renderer --help` to inspect the integrated CLI.

## Prepare inputs

- `structure.mp4`: frame-aligned structure covering at least 158 frames at
  24 fps, about 6.6 seconds, and enough frames for the duration requested with
  `--seconds`. Other frame rates are resampled to 24 fps; rendering starts at
  the beginning. The input is fitted to a 1344×768 landscape canvas or a
  768×1344 portrait canvas.
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
python -m solar.renderer \
  --h3 "$H3_ROOT" \
  --control "$RENDERER_WEIGHTS/solarrenderer_control_lora.safetensors" \
  --fewstep "$RENDERER_WEIGHTS/solarrenderer_4step_lora.safetensors" \
  --structure structure.mp4 --first first.png --prompt prompt.txt \
  --out "$SOLAR_RENDERER_ROOT/render.mp4" --seed 0
```

The default output is a video-only MP4 with 158 frames at 24 fps. The default
is four model evaluations; retain it for the released four-step adapters. Omit
`--first first.png` for structure-and-text conditioning. `--device cuda:0`
selects the CUDA device; replace `0` when choosing another visible GPU.

The control and four-step LoRA deltas are combined in float32, then merged
into the base weights with a single conversion back to bfloat16. This applies
to both adapter pairs and avoids rounding each adapter into the base separately.

The upstream guide reports roughly four minutes for encoding and three and a
half minutes for sampling on its single-GPU setup. These are upstream reference
figures; runtime depends on your hardware, environment and input size.

## Block-causal rendering

Download the causal pair and use both causal adapters with `--causal`:

```bash
hf download Enderfga/SolarRenderer \
  solarrenderer_causal_control_lora.safetensors \
  solarrenderer_causal_4step_lora.safetensors \
  --local-dir "$RENDERER_WEIGHTS"

python -m solar.renderer --causal \
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

The integrated command writes the completed MP4 after sampling and decoding;
its default duration is 158 frames. It does not expose a live frame stream.
The [Reactor integration](../../deploy/solar_renderer/reactor/README.md)
documents its own interactive interface.

## Render a longer clip

Use `--seconds` with either adapter pair to request a longer output. For
example, with a structure video covering at least a minute:

```bash
python -m solar.renderer --causal \
  --h3 "$H3_ROOT" \
  --control "$RENDERER_WEIGHTS/solarrenderer_causal_control_lora.safetensors" \
  --fewstep "$RENDERER_WEIGHTS/solarrenderer_causal_4step_lora.safetensors" \
  --structure structure.mp4 --first first.png --prompt prompt.txt \
  --out "$SOLAR_RENDERER_ROOT/render-causal-60s.mp4" --seconds 60 --seed 0
```

Write the prompt for the whole clip. Omitting `--seconds`, or setting it to
`0`, retains the trained 158-frame default. A positive duration is rounded to
an output frame count at 24 fps. Encoding uses H3's `17n + 5` frame lengths
with a minimum of 158 frames, then trims the decoded result. The output is
limited by the structure video's available aligned frames, so it can be
slightly shorter than requested near the end of the source. The structure
video must still cover at least 158 frames for a shorter requested output.

The bidirectional pair samples all frames together, so its attention memory
and runtime increase with clip length. The causal pair uses a bounded
attention window and advances block by block. For blocks beyond the training
clip, the prompt, first-frame, and audio condition positions shift by whole
five-latent steps to retain the trained relative-position range. Whole-clip
inputs, generated latents, and final decoding still consume memory as the
clip grows; available memory can limit the duration in either mode.

For long inputs, encoding handles VAE clips contiguously and normalizes
visual conditions clip by clip above 1,500 frames, reducing large temporary
GPU allocations. The silent audio conditioning is extended from the packaged
encoded-silence artifact; the exported MP4 remains video-only. These changes
do not provide live playback from this command.

## Related workflows

- [Benchmark and evaluation](../../eval/solar_renderer/README.md): render and evaluate benchmark cases.
- [ComfyUI integration](../../deploy/solar_renderer/comfyui/README.md): use the renderer through nodes.
- [Training](training.md): create and export your own adapter pair.
- [Prompts and first frames](prompts.md): caption and first-frame preparation.

See [NOTICE](../../NOTICE) for upstream credits and source revisions.
