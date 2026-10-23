"""SolarRenderer nodes for ComfyUI.

They run the released renderer (solarwm.renderer.render) on ComfyUI's own MiniMax-H3 (Comfy-Org/MiniMax-H3, the Ref2VA
transformer):

* SolarRendererLoraLoader - a released SolarRenderer LoRA as it is on the Hub, each module patched onto its part of
  ComfyUI's fused weights (q, k and v onto their thirds of attn.qkv_proj, the two halves of the feed-forward input
  onto theirs of mlp.fc1);
* SolarRendererConditioning - the clip's inputs as training encoded them: the structure video laid over every target
  frame and the first frame over frame 0, both as frame-aligned keyframes, and the prompt read by H3's Qwen3-VL with
  the two videos shown to it as <Video 1> and <Video 2> at 2 fps;
* SolarRendererModel - the soundtrack held at H3's encoded silence through sampling, as the renderer holds it (it
  never generates sound, and a generated one would sit in every attention);
* SolarRendererSigmas - the four-step grid (shift 12) the four-step LoRA was trained on; sample it with the "lcm"
  sampler, which takes the clean estimate at every step and noises it afresh, as the renderer does;
* SolarRendererPrompt - the six-part prompt assembled from its free parts with the fixed sentences training used
  (skills/solarrenderer-caption);
* SolarRendererWebCapture - a web page's 3D canvas (three.js, any WebGL) recorded as a structure video, with its
  sound, optionally played by a key script and drawn as a rough early build (web_capture.py).
"""  # noqa: E501

import importlib.util
import json
import logging
import os
import re

import comfy.lora
import comfy.patcher_extension
import comfy.utils
import folder_paths
import torch
from comfy.ldm.minimax.model import time_shift_sigma
from comfy_extras.nodes_minimax_h3 import _empty_av_latent

from ._paths import CAPTION_HELPERS, SILENCE

FPS = 24
LONG, SHOWN_LONG = 362, 158  # past LONG frames, the text encoder sees the first SHOWN_LONG
STILL_FRAMES = (
    13  # H3 takes no reference video shorter than this; a first frame is shown to it held that long
)
SHIFT_VIDEO, SHIFT_AUDIO = 12.0, 3.0
CATEGORY = "SolarRenderer"


# LoRA

_BLOCK = re.compile(r"^(transformer_blocks|token_refiner\.refiner_blocks)\.(\d+)\.(.+)$")


def lora_patches(lora: dict, inner: int, ffn: int) -> tuple[dict, dict]:
    """A SolarRenderer LoRA (diffusers PEFT names) as ComfyUI LoRA entries and the weights they patch.

    Returns (state dict, key map) for comfy.lora.load_lora. A key map entry may name a slice of a fused weight,
    (key, (dim, offset, size)), so nothing is stacked or padded: q, k, v land on rows [0, inner), [inner, 2 inner),
    [2 inner, 3 inner) of qkv_proj, and the feed-forward input, [value; gate] in diffusers, on rows [ffn, 2 ffn) and
    [0, ffn) of fc1, which ComfyUI holds as [gate; value]."""  # noqa: E501
    entries, key_map = {}, {}
    for key in lora:
        if not key.endswith(".lora_A.weight"):
            continue
        base = key[: -len(".lora_A.weight")]
        match = _BLOCK.match(base)
        if match is None:
            raise ValueError(f"{key}: not a MiniMax-H3 transformer block or token refiner module")
        prefix, index, module = match.groups()
        root = (
            "diffusion_model."
            + ("blocks" if prefix == "transformer_blocks" else "token_refiner.blocks")
            + f".{index}"
        )
        a, b = lora[key], lora[base + ".lora_B.weight"]
        if module in ("attn.to_q", "attn.to_k", "attn.to_v"):
            third = "qkv".index(module[-1])
            entries[base + ".lora_A.weight"], entries[base + ".lora_B.weight"] = a, b
            key_map[base] = (f"{root}.attn.qkv_proj.weight", (0, third * inner, inner))
        elif module == "ff.net.0.proj":
            half = b.shape[0] // 2
            for part, rows, start in (("value", b[:half], ffn), ("gate", b[half:], 0)):
                entries[f"{base}.{part}.lora_A.weight"], entries[f"{base}.{part}.lora_B.weight"] = (
                    a,
                    rows,
                )
                key_map[f"{base}.{part}"] = (f"{root}.mlp.fc1.weight", (0, start, ffn))
        elif module in ("attn.to_out.0", "ff.net.2"):
            entries[base + ".lora_A.weight"], entries[base + ".lora_B.weight"] = a, b
            key_map[base] = f"{root}." + (
                "attn.out_proj.weight" if module == "attn.to_out.0" else "mlp.fc2.weight"
            )
        else:
            raise ValueError(f"{key}: no ComfyUI counterpart for {module}")
    return entries, key_map


class SolarRendererLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (
                    folder_paths.get_filename_list("loras"),
                    {
                        "tooltip": "A released SolarRenderer LoRA, as downloaded from the Hub "
                        "(solarrenderer_4step_lora / solarrenderer_control_lora)."
                    },
                ),
                "strength": ("FLOAT", {"default": 1.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model, lora_name, strength):
        lora = comfy.utils.load_torch_file(
            folder_paths.get_full_path_or_raise("loras", lora_name), safe_load=True
        )
        if not any(
            k.startswith("transformer_blocks.") for k in lora
        ):  # already in ComfyUI's own names
            key_map = comfy.lora.model_lora_keys_unet(model.model, {})
            patches = comfy.lora.load_lora(lora, key_map)
        else:
            dm = model.model.diffusion_model
            inner = dm.blocks[0].attn.out_proj.weight.shape[1]
            ffn = dm.blocks[0].mlp.fc2.weight.shape[1]
            entries, key_map = lora_patches(lora, inner, ffn)
            patches = comfy.lora.load_lora(entries, key_map)
        model = model.clone()
        missed = set(patches) - set(model.add_patches(patches, strength))
        if missed:
            raise ValueError(f"{lora_name}: {len(missed)} modules found no weight in this model")
        logging.info(f"SolarRenderer: {lora_name} patched {len(patches)} weights")
        return (model,)


# conditioning


def _fit(frames, width, height):
    """[T, H, W, C] -> [T, height, width, 3]: cut to the canvas's aspect at the centre and LANCZOS-resized, as the
    renderer lays a structure video or a first frame over the target's canvas."""  # noqa: E501
    samples = frames[..., :3].movedim(-1, 1)
    if tuple(samples.shape[-2:]) != (height, width):
        samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "center")
    return samples.movedim(1, -1).clamp(0.0, 1.0)


def _shown(frames):
    """A reference as H3's Qwen3-VL reads it: every 12th of the 24 fps frames, timestamped at 2 fps."""  # noqa: E501
    index = list(range(0, frames.shape[0], FPS // 2))
    return {
        "type": "video",
        "data": frames[index],
        "timestamps": [i / 2.0 for i in range(len(index))],
    }


class SolarRendererConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "structure": (
                    "IMAGE",
                    {
                        "tooltip": "The structure video at 24 fps: a grey box, a segmentation, "
                        "skeletons, depth, a game's frames, a sketch. The render follows "
                        "it frame for frame."
                    },
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamic_prompts": True,
                        "tooltip": "The six-part prompt (SolarRenderer Prompt builds it): <Video 1> is "  # noqa: E501
                        "the structure video, <Video 2> the first frame.",
                    },
                ),
                "width": ("INT", {"default": 1344, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
                "length": (
                    "INT",
                    {
                        "default": 158,
                        "min": 22,
                        "max": 3609,
                        "step": 1,
                        "tooltip": "Frames to render at 24 fps, at most the structure video's; snapped down "  # noqa: E501
                        "to the 17n + 5 frames the video VAE takes. Trained at 158 (6.6 s); "
                        "past 362, render with SolarRenderer Causal Sampler.",
                    },
                ),
            },
            "optional": {
                "first_frame": (
                    "IMAGE",
                    {"tooltip": "The look of the shot: its first frame, already in that look."},
                ),
                "structure_vae": (
                    "VAE",
                    {
                        "tooltip": "Encode the structure video with this one instead (taeh3: ten "
                        "times faster, a little less exact); the first frame keeps vae."
                    },
                ),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = CATEGORY

    def encode(
        self,
        clip,
        vae,
        structure,
        prompt,
        width,
        height,
        length,
        first_frame=None,
        structure_vae=None,
    ):
        frames = min(int(length), int(structure.shape[0]))
        if frames < 22:
            raise ValueError(
                f"SolarRenderer needs at least 22 frames of structure video, got {frames}"
            )
        while frames % 17 != 5:
            frames -= 1
        video = _fit(structure[:frames], width, height)
        # a long clip (the causal sampler's) is described to the text encoder by its opening, as the
        # renderer streams it
        items = [_shown(video if frames <= LONG else video[:SHOWN_LONG])]
        still = None
        if first_frame is not None:
            still = _fit(first_frame[:1], width, height)
            items.append(_shown(still.repeat(STILL_FRAMES, 1, 1, 1)))
        positive = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, minimax_ref_items=items))
        # both laid over the target from its frame 0: the structure over every frame, the first
        # frame over the first
        keyframes = [{"resolved_frame_index": 0, "latent": (structure_vae or vae).encode(video)}]
        if still is not None:
            keyframes.append({"resolved_frame_index": 0, "latent": vae.encode(still)})
        positive = [[c[0], {**c[1], "minimax_keyframes": keyframes}] for c in positive]
        latent, _ = _empty_av_latent(width, height, frames)
        return (positive, latent)


# sampling


class _SilentSoundtrack:
    """Holds the audio stream at H3's encoded silence, noised to the step's audio time with one fixed draw, before
    every evaluation: the renderer's soundtrack, which it never samples."""  # noqa: E501

    def __init__(self, silence, seed):
        self.silence, self.seed, self.noise = silence, seed, None

    def __call__(self, executor, x, timestep, context, transformer_options=None, **kwargs):
        if transformer_options is None:
            transformer_options = {}
        video, audio = x[0], x[1]
        t = audio.shape[-1]
        if self.silence.shape[-1] < t:
            raise ValueError(
                f"the held silence covers 170 frames ({self.silence.shape[-1]} audio latents); "
                f"this clip needs {t}"
            )
        clean = self.silence[..., :t].to(audio.device, torch.float32)
        if self.noise is None or self.noise.shape != clean.shape:
            generator = torch.Generator("cpu").manual_seed(self.seed)
            self.noise = torch.randn(clean.shape, generator=generator, dtype=torch.float32)
        sigma_v = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        shift_v = float(transformer_options.get("minimax_h3_sigma_shift_video", SHIFT_VIDEO))
        shift_a = float(transformer_options.get("minimax_h3_sigma_shift_audio", SHIFT_AUDIO))
        sigma_a = float(time_shift_sigma(sigma_v, shift_v, shift_a))
        held = ((1.0 - sigma_a) * clean + sigma_a * self.noise.to(clean.device)).to(audio.dtype)
        return executor([video, held], timestep, context, transformer_options, **kwargs)


class SolarRendererModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "The draw that noises the held silence.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, seed):
        silence = comfy.utils.load_torch_file(SILENCE)[
            "silence_170f"
        ].float()  # [channels 2, 32, T]
        silence = silence.permute(1, 0, 2).unsqueeze(
            0
        )  # ComfyUI's audio latent is [1, 32, channels, T]
        model = model.clone()
        model.add_wrapper(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, _SilentSoundtrack(silence, seed)
        )
        return (model,)


def shifted_sigmas(steps: int, shift: float = SHIFT_VIDEO):
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
    return shift * base / (1.0 + (shift - 1.0) * base)


class SolarRendererSigmas:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 50,
                        "tooltip": "Model evaluations; the four-step LoRA was trained for 4.",
                    },
                )
            }
        }

    RETURN_TYPES = ("SIGMAS",)
    FUNCTION = "sigmas"
    CATEGORY = CATEGORY

    def sigmas(self, steps):
        return (shifted_sigmas(steps),)


# speed


def _fa4():
    try:
        from flash_attn.cute import flash_attn_func  # FlashAttention-4 (pip install flash-attn-4)

        return flash_attn_func
    except Exception:
        return None


class _FastAttention:
    """The DiT's attention on another kernel: FlashAttention-4 when it is installed, otherwise PyTorch's cuDNN
    attention. Which is fastest depends on the GPU and the software on it; the output is the same."""  # noqa: E501

    def __init__(self, backend):
        self.backend, self.fa4 = backend, _fa4() if backend in ("auto", "flash_attn_4") else None

    def __call__(self, func, q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
        if (
            mask is not None
            or not skip_reshape
            or q.dtype not in (torch.bfloat16, torch.float16)
            or not q.is_cuda
        ):
            return func(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)
        b, h, s, d = q.shape  # [B, heads, tokens, head_dim]
        if self.fa4 is not None:
            out = self.fa4(
                *(x.transpose(1, 2) for x in (q, k, v))
            )  # FA4 takes [B, tokens, heads, head_dim]
            out = out[0] if isinstance(out, tuple) else out
            return out.reshape(b, s, h * d)
        from torch.nn.attention import SDPBackend, sdpa_kernel

        q, k, v = (
            x.contiguous() for x in (q, k, v)
        )  # cuDNN takes no strided views of the fused qkv
        order = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        with sdpa_kernel(
            order, set_priority=True
        ):  # ComfyUI puts flash first; cuDNN first is the point here
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        return out.transpose(1, 2).reshape(b, s, h * d)


class SolarRendererFastAttention:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "backend": (
                    ["auto", "flash_attn_4", "cudnn"],
                    {
                        "tooltip": "auto: FlashAttention-4 if installed (pip install flash-attn-4), "  # noqa: E501
                        "else PyTorch's cuDNN attention."
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CATEGORY

    def patch(self, model, backend):
        attention = _FastAttention(backend)
        if backend == "flash_attn_4" and attention.fa4 is None:
            raise RuntimeError("FlashAttention-4 is not installed: pip install flash-attn-4")
        logging.info(
            f"SolarRenderer: attention on {'FlashAttention-4' if attention.fa4 else 'cuDNN'}"
        )
        model = model.clone()
        model.model_options.setdefault("transformer_options", {})[
            "optimized_attention_override"
        ] = attention
        return (model,)


# prompt


def _caption_helpers():
    spec = importlib.util.spec_from_file_location("solarrenderer_conditions", CAPTION_HELPERS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SolarRendererPrompt:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "structure_type": (
                    [
                        "scene_graybox",
                        "semantic",
                        "scene_whitebox",
                        "mask_skeleton",
                        "smpl_mesh",
                        "near_depth",
                        "custom",
                    ],
                    {
                        "tooltip": "What the structure video is. The six trained types use the sentences "  # noqa: E501
                        "training used; anything else is 'custom'."
                    },
                ),
                "with_first_frame": ("BOOLEAN", {"default": True}),
                "summary": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "One paragraph: who does what, and how the camera moves.",
                    },
                ),
                "detailed_description": (
                    "STRING",
                    {
                        "multiline": True,
                        "tooltip": "350-500 words: the look first (style, light, weather), "
                        "then [Shot 1] and what is seen, in order.",
                    },
                ),
                "soundscape": ("STRING", {"multiline": True, "default": "N/A"}),
            },
            "optional": {
                "custom_looks_like": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "For 'custom': what the structure video "
                        "looks like, e.g. 'a black pencil line sketch of the scene on white paper'.",  # noqa: E501
                    },
                ),
                "custom_not_shown": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "For 'custom': what of its look must not "
                        "appear, e.g. 'the pencil lines and white paper'.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(
        self,
        structure_type,
        with_first_frame,
        summary,
        detailed_description,
        soundscape,
        custom_looks_like="",
        custom_not_shown="",
    ):
        helpers = _caption_helpers()
        if structure_type == "custom":
            if not custom_looks_like.strip() or not custom_not_shown.strip():
                raise ValueError("a custom structure needs custom_looks_like and custom_not_shown")
            condition = helpers.custom_condition(
                custom_looks_like.strip(), custom_not_shown.strip()
            )
        else:
            condition = structure_type
        free = {
            "summary": summary,
            "detailed_description": detailed_description,
            "overall_soundscape": soundscape,
        }
        words = len(detailed_description.split())
        if not 350 <= words <= 500:
            logging.warning(
                f"SolarRenderer Prompt: detailed_description has {words} words; it renders best with "  # noqa: E501
                f"350-500"
            )
        return (helpers.assemble(free, "full" if with_first_frame else "structure", condition),)


# web input


def _decode_sound(path):
    """A WebAudio recording (.webm) as ComfyUI audio: {"waveform": [1, channels, samples], "sample_rate": rate}."""  # noqa: E501
    import av
    import numpy as np

    chunks, rate = [], 48000
    with av.open(path) as box:
        stream = box.streams.audio[0]
        rate = stream.rate or rate
        for frame in box.decode(stream):
            array = frame.to_ndarray()
            if array.dtype.kind in "iu":
                array = array.astype(np.float32) / float(np.iinfo(array.dtype).max)
            if frame.format.is_packed:
                array = array.reshape(-1, len(frame.layout.channels)).T
            chunks.append(array.astype(np.float32))
    waveform = torch.from_numpy(np.concatenate(chunks, axis=1)) if chunks else torch.zeros(2, 1)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": int(rate)}


class SolarRendererWebCapture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "A page drawing a 3D canvas: three.js, Babylon, any WebGL "
                        "(http(s):// or file://).",
                    },
                ),
                "frames": (
                    "INT",
                    {
                        "default": 158,
                        "min": 22,
                        "max": 2400,
                        "tooltip": "Frames to record at 24 fps.",
                    },
                ),
                "width": ("INT", {"default": 1344, "min": 256, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 32}),
            },
            "optional": {
                "keys": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "What to press, one action per line: '1.5:KeyW:2' (W held 2 s), "
                        "'3:down:ShiftLeft' / '4:up:ShiftLeft', '5:mouse:200:0' (look right), "
                        "'6:click'.",
                    },
                ),
                "click": (
                    "STRING",
                    {"default": "", "tooltip": "The selector of a start button to press first."},
                ),
                "setup": (
                    "STRING",
                    {"default": "", "tooltip": "JavaScript run once the page has started."},
                ),
                "warm": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 30.0,
                        "step": 0.5,
                        "tooltip": "Seconds the page runs before the first frame.",
                    },
                ),
                "frame_exact": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Run the page on a virtual clock, 1/24 s per "
                        "frame, so a slow machine still records exactly "
                        "24 fps. Off records in real time, with the "
                        "page's sound.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("frames", "sound")
    FUNCTION = "capture"
    CATEGORY = CATEGORY

    def capture(
        self, url, frames, width, height, keys="", click="", setup="", warm=1.0, frame_exact=True
    ):
        import subprocess
        import sys
        import tempfile

        import numpy as np

        if not url.strip():
            raise ValueError("SolarRenderer Web Capture needs a url")
        try:
            import playwright  # noqa: F401
        except ImportError as error:
            raise RuntimeError(
                "SolarRenderer Web Capture records the page in headless Chromium: "
                "pip install playwright && python -m playwright install chromium"
            ) from error
        with tempfile.TemporaryDirectory() as work:
            out = os.path.join(work, "capture")
            command = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "web_capture.py"),
                "--url",
                url.strip(),
                "--out",
                out,
                "--frames",
                str(frames),
                "--width",
                str(width),
                "--height",
                str(height),
                "--keys",
                keys,
                "--click",
                click,
                "--setup",
                setup,
                "--warm",
                str(warm),
            ] + (["--exact"] if frame_exact else [])
            done = subprocess.run(command, capture_output=True, text=True)
            if done.returncode != 0:
                raise RuntimeError(f"web capture failed: {done.stderr.strip()[-2000:]}")
            report = json.loads(done.stdout.strip().splitlines()[-1])
            if not frame_exact and report["real_time"] < 0.9:
                logging.warning(
                    f"SolarRenderer Web Capture grabbed at {report['real_time']}x real time: the page ran "  # noqa: E501
                    f"ahead of its frames, so the clip plays fast. Turn frame_exact on, or record smaller."  # noqa: E501
                )
            images = torch.from_numpy(np.load(f"{out}.npy")).float() / 255.0
            sound = (
                _decode_sound(f"{out}.webm")
                if os.path.exists(f"{out}.webm")
                else {"waveform": torch.zeros(1, 2, 1), "sample_rate": 48000}
            )
        return (images, sound)


NODE_CLASS_MAPPINGS = {
    "SolarRendererLoraLoader": SolarRendererLoraLoader,
    "SolarRendererConditioning": SolarRendererConditioning,
    "SolarRendererModel": SolarRendererModel,
    "SolarRendererSigmas": SolarRendererSigmas,
    "SolarRendererFastAttention": SolarRendererFastAttention,
    "SolarRendererPrompt": SolarRendererPrompt,
    "SolarRendererWebCapture": SolarRendererWebCapture,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SolarRendererLoraLoader": "SolarRenderer LoRA Loader",
    "SolarRendererConditioning": "SolarRenderer Structure to Video",
    "SolarRendererModel": "SolarRenderer Model (silent soundtrack)",
    "SolarRendererSigmas": "SolarRenderer Four-Step Sigmas",
    "SolarRendererFastAttention": "SolarRenderer Fast Attention",
    "SolarRendererPrompt": "SolarRenderer Prompt",
    "SolarRendererWebCapture": "SolarRenderer Web Capture (three.js / WebGL page)",
}
