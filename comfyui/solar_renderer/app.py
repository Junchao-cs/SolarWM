"""The SolarRenderer page: a browser front end served by this ComfyUI, at http://<comfyui>/solarrenderer.

Pick an example or upload a structure video (and a first frame), write the prompt, choose full quality or low memory,
and render. The page builds the same graph as the workflows in workflows/, queues it on this server, shows the
sampler's previews as they arrive and plays the render beside its structure when it is done. Nothing runs outside
ComfyUI: the models, the memory management and the queue are ComfyUI's own.
"""  # noqa: E501

import logging
import os
import re
import shutil

import folder_paths
from aiohttp import web
from server import PromptServer

from ._paths import EXAMPLES, WEB

INPUT_SUBFOLDER = "solarrenderer"

QUALITIES = {
    "full": {
        "label": "Full quality",
        "note": "bf16 weights, for a large GPU",
        "unet": "minimax_h3_ref2va_bf16.safetensors",
        "clip": "qwen3vl_32b_minimax_h3_bf16.safetensors",
    },
    "low_memory": {
        "label": "Low memory",
        "note": "8-bit weights; runs on a 16 GB GPU",
        "unet": "minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        "clip": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    },
}
VAE = "minimax_h3_video_vae_fp16.safetensors"
FEWSTEP_LORA = "solarrenderer_4step_lora.safetensors"
CONTROL_LORA = "solarrenderer_control_lora.safetensors"
CAUSAL_FEWSTEP_LORA = "solarrenderer_causal_4step_lora.safetensors"
CAUSAL_CONTROL_LORA = "solarrenderer_causal_control_lora.safetensors"


def _missing(quality: dict, causal: bool = False) -> list:
    """The model files this quality (and the causal pair, for long clips) needs that ComfyUI cannot find."""  # noqa: E501
    loras = (CAUSAL_FEWSTEP_LORA, CAUSAL_CONTROL_LORA) if causal else (FEWSTEP_LORA, CONTROL_LORA)
    need = [
        ("diffusion_models", quality["unet"]),
        ("text_encoders", quality["clip"]),
        ("vae", VAE),
        *(("loras", name) for name in loras),
    ]
    return [
        f"{folder}/{name}"
        for folder, name in need
        if name not in folder_paths.get_filename_list(folder)
    ]


def graph(
    quality: str,
    structure: str,
    first: str | None,
    prompt: str,
    seed: int,
    width: int,
    height: int,
    length: int,
    causal: bool = False,
) -> dict:
    """The render as an API-format prompt: the nodes and settings of workflows/solarrenderer_*.json; ``causal`` renders
    a long clip chunk by chunk with the causal LoRA pair."""  # noqa: E501
    q = QUALITIES[quality]
    fewstep, control = (
        (CAUSAL_FEWSTEP_LORA, CAUSAL_CONTROL_LORA) if causal else (FEWSTEP_LORA, CONTROL_LORA)
    )
    nodes = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": q["unet"], "weight_dtype": "default"},
        },
        "2": {
            "class_type": "SolarRendererLoraLoader",
            "inputs": {"model": ["1", 0], "lora_name": fewstep, "strength": 1.0},
        },
        "3": {
            "class_type": "SolarRendererLoraLoader",
            "inputs": {"model": ["2", 0], "lora_name": control, "strength": 1.0},
        },
        "4": {
            "class_type": "SolarRendererFastAttention",
            "inputs": {"model": ["3", 0], "backend": "auto"},
        },
        "5": {"class_type": "SolarRendererModel", "inputs": {"model": ["4", 0], "seed": seed}},
        "6": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": q["clip"], "type": "minimax", "device": "default"},
        },
        "7": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "8": {"class_type": "LoadVideo", "inputs": {"file": structure}},
        "9": {"class_type": "GetVideoComponents", "inputs": {"video": ["8", 0]}},
        "11": {
            "class_type": "SolarRendererConditioning",
            "inputs": {
                "clip": ["6", 0],
                "vae": ["7", 0],
                "structure": ["9", 0],
                "prompt": prompt,
                "width": width,
                "height": height,
                "length": length,
            },
        },
        "12": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "lcm"}},
        "13": {"class_type": "SolarRendererSigmas", "inputs": {"steps": 4}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "15": {
            "class_type": "BasicGuider",
            "inputs": {"model": ["5", 0], "conditioning": ["11", 0]},
        },
        "16": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["14", 0],
                "guider": ["15", 0],
                "sampler": ["12", 0],
                "sigmas": ["13", 0],
                "latent_image": ["11", 1],
            },
        },
        "17": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["16", 0]}},
        "18": {"class_type": "VAEDecode", "inputs": {"samples": ["17", 0], "vae": ["7", 0]}},
        "19": {"class_type": "CreateVideo", "inputs": {"images": ["18", 0], "fps": 24.0}},
        "20": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["19", 0],
                "filename_prefix": "SolarRenderer/render",
                "format": "mp4",
                "codec": "h264",
            },
        },
    }
    if causal:  # one node samples the whole clip, chunk by chunk
        for key in ("4", "5", "12", "13", "14", "15"):
            nodes.pop(key)
        nodes["16"] = {
            "class_type": "SolarRendererCausalSampler",
            "inputs": {
                "model": ["3", 0],
                "positive": ["11", 0],
                "latent": ["11", 1],
                "seed": seed,
                "cache": "auto",
            },
        }
    if first:
        nodes["10"] = {"class_type": "LoadImage", "inputs": {"image": first}}
        nodes["11"]["inputs"]["first_frame"] = ["10", 0]
    return nodes


def _examples() -> list:
    found = []
    for name in sorted(os.listdir(EXAMPLES)) if os.path.isdir(EXAMPLES) else []:
        root = os.path.join(EXAMPLES, name)
        if not os.path.isfile(os.path.join(root, "structure.mp4")):
            continue
        title = name
        readme = os.path.join(root, "README.md")
        if os.path.isfile(readme):
            with open(readme, encoding="utf-8") as handle:
                heading = handle.readline().lstrip("# ").strip()
            title = re.sub(r"^Example:\s*", "", heading) or name
        found.append(
            {
                "id": name,
                "title": title[:1].upper() + title[1:],
                "first": os.path.isfile(os.path.join(root, "first.png")),
            }
        )
    return found


def _example_dir(request) -> str:
    name = request.match_info["name"]
    if name not in {e["id"] for e in _examples()}:
        raise web.HTTPNotFound()
    return os.path.join(EXAMPLES, name)


routes = PromptServer.instance.routes


@routes.get("/solarrenderer")
async def page(request):
    return web.FileResponse(os.path.join(WEB, "index.html"))


@routes.get("/solarrenderer/static/{file}")
async def static(request):
    path = os.path.join(WEB, os.path.basename(request.match_info["file"]))
    if not os.path.isfile(path):
        raise web.HTTPNotFound()
    return web.FileResponse(path)


@routes.get("/solarrenderer/api/info")
async def info(request):
    qualities = [
        {
            "id": k,
            "label": q["label"],
            "note": q["note"],
            "missing": _missing(q),
            "missing_causal": _missing(q, causal=True),
        }
        for k, q in QUALITIES.items()
    ]
    return web.json_response({"qualities": qualities, "examples": _examples()})


@routes.get("/solarrenderer/api/example/{name}/{file}")
async def example_file(request):
    file = request.match_info["file"]
    if file not in ("structure.mp4", "first.png", "preview.gif", "prompt.txt"):
        raise web.HTTPNotFound()
    path = os.path.join(_example_dir(request), file)
    if not os.path.isfile(path):
        raise web.HTTPNotFound()
    return web.FileResponse(path)


@routes.post("/solarrenderer/api/example/{name}")
async def use_example(request):
    """Copy an example's files into ComfyUI's input folder, where the loaders read them."""
    root, name = _example_dir(request), request.match_info["name"]
    target = os.path.join(folder_paths.get_input_directory(), INPUT_SUBFOLDER)
    os.makedirs(target, exist_ok=True)
    chosen = {}
    for file, key in (("structure.mp4", "structure"), ("first.png", "first")):
        if os.path.isfile(os.path.join(root, file)):
            shutil.copyfile(os.path.join(root, file), os.path.join(target, f"{name}_{file}"))
            chosen[key] = f"{INPUT_SUBFOLDER}/{name}_{file}"
    prompt = os.path.join(root, "prompt.txt")
    chosen["prompt"] = ""
    if os.path.isfile(prompt):
        with open(prompt, encoding="utf-8") as handle:
            chosen["prompt"] = handle.read()
    return web.json_response(chosen)


@routes.post("/solarrenderer/api/graph")
async def make_graph(request):
    body = await request.json()
    quality = body.get("quality", "low_memory")
    if quality not in QUALITIES:
        return web.json_response({"error": f"unknown quality {quality}"}, status=400)
    causal = bool(body.get("causal"))
    missing = _missing(QUALITIES[quality], causal)
    if missing:
        return web.json_response({"error": "missing models: " + ", ".join(missing)}, status=400)
    if not body.get("structure") or not str(body.get("prompt", "")).strip():
        return web.json_response({"error": "a structure video and a prompt are needed"}, status=400)
    landscape = body.get("orientation", "landscape") != "portrait"
    return web.json_response(
        {
            "prompt": graph(
                quality,
                body["structure"],
                body.get("first") or None,
                body["prompt"],
                int(body.get("seed", 0)),
                1344 if landscape else 768,
                768 if landscape else 1344,
                int(body.get("length", 158)),
                causal,
            )
        }
    )


logging.info("SolarRenderer: the page is at /solarrenderer")
