# SolarRenderer in ComfyUI

SolarRenderer runs on ComfyUI's own MiniMax-H3 with a few extra nodes. Load a structure video (a grey box, a
segmentation, skeletons, depth, a game's frames, a sketch), add a first frame and a prompt, and render it in four steps.

## Install

1. Update ComfyUI to a version with MiniMax-H3 (v0.30.0 or later, the same minimum as upstream).
   The nodes use ComfyUI's MiniMax-H3 implementation, loaders and sampler APIs, so an older ComfyUI
   installation cannot load this pack.
2. Keep a checkout of SolarWM and link **only this node pack** into `ComfyUI/custom_nodes`:

   ```bash
   cd /path/to/SolarWM
   ln -s "$(pwd)/deploy/solar_renderer/comfyui" /path/to/ComfyUI/custom_nodes/solar_renderer
   ```

   Keep the checkout at that location. The link resolves the shared silence tensor from
   `solar/renderer/assets`, examples from `examples/solar_renderer`, and prompt helpers from
   `skills/solarrenderer-caption/scripts`. Copying this folder alone omits those resources.
   ComfyUI loads `deploy/solar_renderer/comfyui/__init__.py`; the repository root is not a node pack.

   Use ComfyUI's existing Python environment and dependencies (`torch`, `numpy`, `safetensors`,
   `aiohttp`, `av` and Pillow); installing SolarWM's training or renderer extras is unnecessary for
   these nodes. Web Capture additionally needs `playwright` and its Chromium browser, installed
   in that same environment. FlashAttention-4 is optional; the nodes fall back to PyTorch attention.

3. Download the models into `ComfyUI/models` (pick one row per folder; the workflows below say which):

   | Folder | Full quality | Low memory | From |
   |---|---|---|---|
   | `diffusion_models` | `minimax_h3_ref2va_bf16.safetensors` | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | [Comfy-Org/MiniMax-H3](https://huggingface.co/Comfy-Org/MiniMax-H3) |
   | `text_encoders` | `qwen3vl_32b_minimax_h3_bf16.safetensors` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` | Comfy-Org/MiniMax-H3 |
   | `vae` | `minimax_h3_video_vae_fp16.safetensors` | same | Comfy-Org/MiniMax-H3 |
   | `loras` | `solarrenderer_control_lora.safetensors`, `solarrenderer_4step_lora.safetensors` | same | [Enderfga/SolarRenderer](https://huggingface.co/Enderfga/SolarRenderer) |
   | `loras` (long clips) | `solarrenderer_causal_control_lora.safetensors`, `solarrenderer_causal_4step_lora.safetensors` | same | Enderfga/SolarRenderer |
   | `vae_approx` (optional, previews) | `taeh3.safetensors` | same | [madebyollin/taehv](https://github.com/madebyollin/taehv/tree/main/safetensors) |

   The SolarRenderer LoRAs are used as released; the LoRA loader below maps them onto ComfyUI's weights.

## The SolarRenderer page

Start ComfyUI and open `http://127.0.0.1:8188/solarrenderer`: pick an example or drop in your own structure video and
first frame, write the prompt, choose the models and render: **6.6 s** renders the first 158 frames in one pass,
**Whole video** the whole structure video chunk by chunk (see [Long clips](#long-clips)). The page shows the sampler's
previews as they arrive and plays the render beside its structure when it is done. It runs on the same ComfyUI, so
everything below (models, memory, the queue) applies to it too.

## Workflows

For the node graph itself, drag a file from [`workflows/`](workflows) into ComfyUI. Both render the
[synthwave example](../../../examples/solar_renderer/sem-highway_synthwave) (copy its `structure.mp4` and `first.png` into
`ComfyUI/input`).

| Workflow | Models | GPU memory |
|---|---|---|
| `solarrenderer_full.json` | full quality | a large GPU; less memory works, more slowly |
| `solarrenderer_low_memory.json` | low memory (8-bit) | **16 GB**, e.g. RTX 4080 or RTX 4090 |

The low-memory models render close to the full ones. ComfyUI keeps on the GPU only what fits and moves the rest in
from system memory as it is needed: the low-memory workflow completed with ComfyUI held to 14 GB of GPU memory, peaking
at 14.5 GB including the CUDA context. Give it 64 GB of system memory. On a GPU that holds every model, the low-memory
workflow takes about twice as long as the full one; on a smaller card, moving the weights in adds to that.

## Long clips

`solarrenderer_long.json` renders a structure video of any length (up to 150 s) with the causal LoRA pair: the
**SolarRenderer Causal Sampler** generates five latent frames (about 17 video frames) at a time, each chunk seeing the
prompt, the first frame and the structure and render of the five chunks before it, so the memory stays that of a
six-chunk window however long the clip. Its `length` is set to the maximum, so the whole structure video is rendered.

The sampler keeps the window's keys and values in a cache of up to about 80 GB at 1344×768. `cache: auto` keeps it on
the GPU when it fits and in system memory otherwise (or choose `system memory`), which lets a smaller GPU render long
clips with the low-memory models, more slowly.

**Live previews**: put `taeh3.safetensors` in `ComfyUI/models/vae_approx` and start ComfyUI with
`--preview-method taesd` to watch each step (the page asks for previews by itself). The same file, chosen in a
`Load VAE` node, decodes in a fraction of the time at a visible cost in detail.

## Nodes

| Node | What it does |
|---|---|
| **SolarRenderer LoRA Loader** | Loads a released SolarRenderer LoRA. Chain two: the four-step LoRA, then the control LoRA. |
| **SolarRenderer Structure to Video** | The structure video, the first frame and the prompt in; the conditioning and an empty latent out. `length` is in frames at 24 fps (trained at 158). Connect `taeh3` to `structure_vae` to encode the structure faster. |
| **SolarRenderer Model (silent soundtrack)** | Holds the soundtrack at silence while sampling, as the renderer was trained. |
| **SolarRenderer Four-Step Sigmas** | The four-step schedule. Sample with `SamplerCustomAdvanced`, the `lcm` sampler and a `BasicGuider`. |
| **SolarRenderer Causal Sampler (long clips)** | Renders the conditioning chunk by chunk with the causal LoRA pair, at any length; split the result with `Separate AV Latent` and decode it with `VAE Decode`. |
| **SolarRenderer Fast Attention** | Runs attention on another kernel, which on many GPUs samples faster than ComfyUI's default; same output. Bypass it if it is slower on yours. |
| **SolarRenderer Prompt** | Builds the six-part prompt from its free parts ([prompt guide](../../../docs/solar_renderer/prompts.md)). |
| **SolarRenderer Web Capture** | Records a web page's 3D canvas (three.js or any WebGL) as a structure video, with its sound: a URL in, frames and audio out. An optional key script plays it. Needs `pip install playwright && python -m playwright install chromium`. |

Split the sampled latent with `Separate AV Latent` and decode the video with `VAE Decode`.
