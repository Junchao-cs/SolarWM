"""Preencode unified omni samples with the released MiniMax-H3 ``ref2va`` front end.

Every sample of the input JSONL is::

    {"id": "...", "target": "clip.mp4", "text": "...",
     "observations": [{"media": "x.mp4|x.png", "frames": "all" | [0, 5], "degree": 0.999}, ...],
     "known_mask": "mask.mp4" | null, "source": "dataset-name"}

``frames`` says how an observation relates to the clip: ``"all"`` or a list of target latent frames
for an aligned one, ``null`` for a free one. No field names a task or a condition type.

The prompt embeddings, their token tags and the observation latents come from the released modular
blocks (setup -> Qwen3-VL presentation -> reference VAE encode), so a sample is conditioned exactly
as the released checkpoint expects. The target goes through the same reference encoder, on its own,
to get latents in the same convention. One worker per GPU: ``--shard i --shards n``.

Media and samples may be local paths or storage URLs (``<scheme>://...``, read and written through
solarwm.renderer.storage); with URLs, node-local scratch only ever holds what a worker is busy with.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

from solarwm.renderer import storage

ROOT = os.environ.get("H3_REF_ROOT", "MiniMax-H3")

FPS = 24.0

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")

STILL_FRAMES = 13

SCRATCH = os.environ.get("OMNI_SCRATCH", "/tmp/omni_pre")


class WriteRefused(RuntimeError):
    """The storage would not take a member this worker had already encoded."""


class FetchFailed(RuntimeError):
    """A row's media could not be fetched - most often because it is not there yet."""


def copy(source: str, destination: str) -> None:
    """Copy one file to or from storage (solarwm.renderer.storage retries a busy store a few times).

    Which way it went decides whose failure it is: a write the storage refused says nothing about
    the row, a fetch that failed says the row's media is missing - and that row is set aside."""

    try:
        if storage.is_remote(destination):
            storage.put(source, destination)
        else:
            storage.get(source, destination)
    except Exception as exc:
        error = WriteRefused if storage.is_remote(destination) else FetchFailed
        raise error(f"storage copy {source} -> {destination}: {str(exc)[-300:]}") from exc


class Sink:
    """Where encoded samples go: a remote object-storage prefix, or a local directory."""

    def __init__(self, out: str, shard: int) -> None:
        self.out, self.remote = out.rstrip("/"), "://" in out
        # Scratch is per set, not just per shard. A node encodes one set after another, and an
        # index named only by its shard number would be found by the next set, appended to, and
        # published under that set's prefix - carrying the previous set's rows into it.
        identity = hashlib.sha256(self.out.encode()).hexdigest()[:24]
        self.stage = f"{SCRATCH}/out/{identity}"
        self.index_name = f"index.{shard:05d}.jsonl"
        self.index_path = f"{self.stage}/{self.index_name}"
        os.makedirs(f"{self.stage}/samples", exist_ok=True)
        if not self.remote:
            os.makedirs(f"{self.out}/samples", exist_ok=True)
            self.index_path = f"{self.out}/{self.index_name}"
        elif not os.path.exists(self.index_path):
            # A restarted shard picks up its own index and skips what it already wrote.
            if storage.stat(f"{self.out}/{self.index_name}") is not None:
                copy(f"{self.out}/{self.index_name}", self.index_path)

    def member(self, name: str) -> str:
        """Local path to write a member to; ``publish`` moves it where it belongs."""

        return f"{self.stage if self.remote else self.out}/{name}"

    def publish(self, name: str) -> None:
        if self.remote:
            # Copied, then dropped here - not moved. A move that is tried again after it failed
            # has nothing left to move, because the first try took the file whether or not it
            # reported success, and the worker then throws away a member the storage already holds.
            copy(f"{self.stage}/{name}", f"{self.out}/{name}")
            with contextlib.suppress(OSError):
                os.remove(f"{self.stage}/{name}")

    def publish_index(self) -> None:
        if self.remote:
            copy(self.index_path, f"{self.out}/{self.index_name}")


def fetch_fresh(path: str, destination: str) -> str:
    """The current version of a sample list, one copy per node.

    A list is rewritten under its own name, so a copy is kept per version - its size and time in
    the storage - and a new version is fetched as soon as there is one. The biggest lists run to
    gigabytes; every worker of every node fetching its own each pass would read terabytes
    for nothing. The first worker to get the lock fetches, the others wait and share its copy.
    ``destination`` is kept for the callers' sake; the copy lives in the node's list cache.
    """

    import fcntl

    if "://" not in path:
        return path
    found = storage.stat(path)
    version = f"{found[0]}_{int(found[1] or 0)}" if found else "unknown"
    cache = os.environ.get("OMNI_LIST_CACHE", "/tmp/omni_lists")
    os.makedirs(cache, exist_ok=True)
    identity = hashlib.sha256(path.encode()).hexdigest()[:24]
    local = f"{cache}/{identity}.{version}"
    if os.path.exists(local):
        return local
    with open(f"{local}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(local):
            copy(path, f"{local}.part")
            os.replace(f"{local}.part", local)
            # Older versions of the same list are no longer anyone's.
            for name in os.listdir(cache):
                if name.startswith(identity + ".") and not name.startswith(os.path.basename(local)):
                    with contextlib.suppress(OSError):
                        os.remove(f"{cache}/{name}")
    return local


def localize(path: str) -> str:
    """A remote path is fetched to local scratch once; anything else is used as is."""

    if "://" not in path:
        return path
    import hashlib

    local = f"{SCRATCH}/in/{hashlib.sha1(path.encode()).hexdigest()}{os.path.splitext(path)[1]}"
    if not os.path.exists(local):
        os.makedirs(f"{SCRATCH}/in", exist_ok=True)
        part = f"{local}.{os.getpid()}.part"  # the workers of a node share this cache
        copy(path, part)
        os.replace(part, local)
    return local


def open_media(path: str):
    """A PyAV container that tolerates undecodable metadata; newer PyAV releases dropped the
    option."""

    import av

    try:
        return av.open(path, metadata_errors="ignore")
    except TypeError:
        return av.open(path)


def read_video(path: str, seconds: float = 0.0) -> tuple[np.ndarray, float]:
    """Decode a clip - only its first ``seconds`` when given.

    H3 keeps 6.6 s of any clip, and OmniHuman alone holds 4K clips a thousand frames long: decoding
    all of one to use its start kept one worker busy for most of a pass while its seven siblings,
    finished, held idle cards waiting for it.
    """

    path = localize(path)
    container = open_media(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"  # frame threads: otherwise a worker decodes on a single core
    fps = float(stream.average_rate)
    limit = math.ceil(seconds * fps) + 2 if seconds else 0
    frames = []
    for frame in container.decode(stream):
        frames.append(frame.to_ndarray(format="rgb24"))
        if limit and len(frames) >= limit:
            break
    container.close()
    return np.stack(frames), fps


def prepared(frames: np.ndarray, fps: float, num_frames: int) -> np.ndarray:
    """A clip exactly as H3 would make it - on 24 fps, cut to ``num_frames``, LANCZOS-resized to the
    canvas its own aspect resolves to - with the resize spread over threads.

    H3 does this itself, one frame after another on one core, while the card waits: a quarter of
    every card's time went to it. A clip handed over already at 24 fps and at that size goes
    through H3 untouched, so doing the same steps here, in parallel, changes no pixel.
    """

    from concurrent.futures import ThreadPoolExecutor

    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import resolve_canvas_size
    from PIL import Image

    frames = on_grid(frames, fps)[:num_frames]
    # The released checkpoint's canvas rule: axes on multiples of 32, short edge 768, area 768x1344.
    height, width = resolve_canvas_size(frames.shape[2], frames.shape[1], 32, 768, 768 * 1344)
    if frames.shape[1:3] == (height, width):
        return np.ascontiguousarray(frames)

    def one(frame: np.ndarray) -> np.ndarray:
        return np.asarray(Image.fromarray(frame).resize((width, height), Image.Resampling.LANCZOS))

    with ThreadPoolExecutor(16) as pool:
        return np.stack(list(pool.map(one, frames)))


def frame_rate(path: str) -> float:
    """A clip's frame rate from its container."""

    container = open_media(localize(path))
    rate = float(container.streams.video[0].average_rate)
    container.close()
    return rate


def label_mask(path: str, label: int, fps: float, start: float, span: float):
    """One object's region out of a per-frame label map (VPData's all_masks.npz: every frame of the
    clip, each pixel the number of the object on it), cut to the stretch that is needed, as frames
    the rest of the mask handling reads: white where the object is. It runs on the clip's clock -
    its frame i is the clip's frame i - so the clip's own rate is its rate."""

    labels = np.load(localize(path))["arr_0"]
    first = round(start * fps)
    last = math.ceil((start + span) * fps) + 2
    # Already cut at the start: the caller must not cut it again.
    return (labels[first:last] == label).astype(np.uint8)[..., None] * 255, fps


def frame_count(path: str) -> int:
    """How many frames a clip holds, from its container rather than by decoding it."""

    container = open_media(localize(path))
    stream = container.streams.video[0]
    count = int(stream.frames or 0)
    if not count:
        seconds = (
            float(stream.duration * stream.time_base)
            if stream.duration
            else (container.duration or 0) / 1e6
        )
        count = round(seconds * float(stream.average_rate))
    container.close()
    return count


def on_grid(frames: np.ndarray, fps: float) -> np.ndarray:
    """A clip put on 24 fps the way H3 puts a reference there: whole frames dropped or held."""

    if fps == FPS:
        return frames
    scale = FPS / fps
    slots = np.floor(np.arange(frames.shape[0]) * scale + 0.5).astype(np.int64)
    end = math.floor(frames.shape[0] * scale + 0.5)
    return np.repeat(frames, np.diff(slots, append=end), axis=0)


def crop_scale(size: tuple[int, int], canvas_shape: tuple[int, int]) -> tuple[float, float]:
    """What fraction of the height and width a centre-crop to the canvas aspect keeps."""

    height, width = size
    wanted = canvas_shape[1] / canvas_shape[0]
    if abs(width / height - wanted) < 1e-3:
        return 1.0, 1.0
    if width / height > wanted:
        return 1.0, round(height * wanted) / width
    return round(width / wanted) / height, 1.0


def fit(frames: np.ndarray, canvas_shape: tuple[int, int]) -> np.ndarray:
    """Centre-crop to the canvas's own aspect, because the encoder will not stretch to it.

    The reference blocks fit a frame inside the canvas and keep its aspect, so a clip that is not
    1344:768 comes out on a smaller grid than the model's - 48x64 for four-thirds, 48x48 for a
    square render - and the reader will not admit it. Cropping keeps every pixel it keeps real;
    padding would teach the model to draw the bars.
    """

    height, width = frames.shape[1:3]
    wanted = canvas_shape[1] / canvas_shape[0]
    if abs(width / height - wanted) < 1e-3:
        return frames
    if width / height > wanted:  # too wide: take a full-height slice
        keep = round(height * wanted)
        start = (width - keep) // 2
        return frames[:, :, start : start + keep]
    keep = round(width / wanted)  # too tall: take a full-width slice
    start = (height - keep) // 2
    return frames[:, start : start + keep, :]


def region(frames: np.ndarray, box) -> np.ndarray:
    """The part of every frame a dataset's own crop box keeps, ``[x0, x1, y0, y1]`` in the clip's
    pixels. Some sets cut their clips down to a region of the frame - past a subtitle, a watermark,
    a letterbox, or onto the part that holds the subject - and name the box rather than re-encode
    millions of files; whatever is laid over the clip frame for frame is cut with it."""

    if not box:
        return frames
    x0, x1, y0, y1 = (int(v) for v in box)
    height, width = frames.shape[1:3]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"crop box {list(box)} does not fit the {width}x{height} frame")
    return frames[:, y0:y1, x0:x1]


def within(image, limit: float = 4.0):
    """The centre of an image cut to the widest and tallest shapes H3 takes, 4:1 and 1:4."""

    width, height = image.size
    if width > limit * height:
        keep = int(limit * height)
        return image.crop(((width - keep) // 2, 0, (width - keep) // 2 + keep, height))
    if height > limit * width:
        keep = int(limit * width)
        return image.crop((0, (height - keep) // 2, width, (height - keep) // 2 + keep))
    return image


def aligned_image(image, canvas_shape: tuple[int, int], box=None):
    """An image laid over the target's canvas (a first frame, a still of an aligned stream):
    cut to the
    canvas's aspect and LANCZOS-resized to its size, as every frame of the target is. Handed to
    H3 as
    an image, a first frame became a 128x224 latent - off the target's 48x84 grid - and some seven
    thousand extra prompt tokens; ``reference`` hands it over as a one-frame video at this size."""

    from PIL import Image

    pixels = fit(region(np.asarray(image)[None], box), canvas_shape)[0]
    height, width = canvas_shape
    if pixels.shape[:2] == (height, width):
        return Image.fromarray(pixels)
    return Image.fromarray(pixels).resize((width, height), Image.Resampling.LANCZOS)


def reference(
    path: str,
    canvas_shape: tuple[int, int],
    video=None,
    crop: bool = True,
    start: float = 0.0,
    num_frames: int = 158,
    box=None,
):
    """One reference for H3. ``crop`` puts it on the canvas's shape; only what is laid over the
    target's canvas - the target itself, and an observation aligned with it frame for frame - has to
    be. A free reference (a portrait, an object, another shot) sits on its own spatial grid, as it
    does in H3 itself, so it keeps its own shape: cutting a full-length portrait to the canvas's
    7:4 kept a band across its middle, a torso with no face and no feet."""

    from diffusers.modular_pipelines.minimax_h3 import (
        MiniMaxH3ImageReference,
        MiniMaxH3VideoReference,
    )
    from PIL import Image

    if path.lower().endswith(IMAGE_EXT):
        image = Image.open(localize(path)).convert("RGB")
        if not crop:
            return MiniMaxH3ImageReference(image=within(image))
        # Encode aligned stills on the target's grid, using the same video VAE path.
        # are: as video, on the target's grid (as an image H3 brings it to its own still size). H3
        # takes no video shorter than STILL_FRAMES, so the still is held that long; the causal VAE's
        # first latent sees the first frame alone, and that latent is all the sample keeps.
        frame = np.asarray(aligned_image(image, canvas_shape, box))[None]
        return MiniMaxH3VideoReference(frames=np.repeat(frame, STILL_FRAMES, axis=0), fps=FPS)
    frames, fps = video if video is not None else read_video(path, start + num_frames / FPS + 0.5)
    if start:
        frames = frames[round(start * fps) :]
    # The target's own crop box, for a reference laid over it; a free one is never cut.
    frames = fit(region(frames, box), canvas_shape) if crop else frames
    return MiniMaxH3VideoReference(frames=prepared(frames, fps, num_frames), fps=FPS)


def build_pipelines(device: str):
    from diffusers import ComponentsManager, ModularPipeline
    from diffusers.modular_pipelines import SequentialPipelineBlocks

    blocks = ModularPipeline.from_pretrained(ROOT).blocks
    manager = ComponentsManager()
    # Weights stay on the card. Offloading moved the text encoder over PCIe on every call, which
    # cost little per sample once warm but made the first call of every process several times
    # slower. The heaviest sample, 47 latents with four observations, fits on the card;
    # offloading remains, behind a switch, for a shape that would not fit.
    offload = os.environ.get("OMNI_CPU_OFFLOAD") == "1"
    if offload:
        manager.enable_auto_cpu_offload(device=device)

    def chain(workflow: str, names: tuple[str, ...]):
        sub = blocks.get_workflow(workflow).sub_blocks
        return SequentialPipelineBlocks.from_blocks_dict(
            {name: sub[name] for name in names}
        ).init_pipeline(ROOT, components_manager=manager)

    conditioned = chain("ref2va", ("before_encode", "text_encoder", "vae_encoder"))
    conditioned.load_components(dtype=torch.bfloat16)
    if not offload:
        for name in conditioned.pretrained_component_names:
            component = getattr(conditioned, name, None)
            if isinstance(component, torch.nn.Module):
                component.to(device)
    # One copy of every weight: the other chains borrow the components (the text encoder is 62 GB).
    loaded = {name: getattr(conditioned, name) for name in conditioned.pretrained_component_names}
    latents_only = chain("ref2va", ("before_encode", "vae_encoder"))
    text_only = chain("t2va", ("text_encoder",))
    for pipe in (latents_only, text_only):
        pipe.update_components(
            **{k: v for k, v in loaded.items() if k in pipe.pretrained_component_names}
        )
    return conditioned, latents_only, text_only


def canvas(path: str, box=None) -> tuple[int, int]:
    """The canvas a clip is encoded on: landscape or portrait by the part of the frame that is
    kept."""

    stream = open_media(localize(path)).streams.video[0]
    width, height = stream.width, stream.height
    if box:
        width, height = int(box[1]) - int(box[0]), int(box[3]) - int(box[2])
    return (768, 1344) if width >= height else (1344, 768)


def source_size(path: str) -> tuple[int, int]:

    stream = open_media(localize(path)).streams.video[0]
    return stream.height, stream.width


def encode(sample: dict, pipes, num_frames: int) -> tuple[dict, dict]:
    conditioned, latents_only, text_only = pipes
    box = sample.get("crop")
    # A crop box moves the picture's origin and scales its field of view, and a camera track's
    # intrinsics would have to follow; nothing here does that, so the two are refused together
    # before any card time is spent.
    if box and sample.get("camera"):
        raise ValueError("a crop box and a camera track together are not supported")
    height, width = canvas(sample["target"], box)
    size = {"height": height, "width": width, "num_frames": num_frames}
    shape = (height, width)
    # Where in the clip the sample starts, in seconds. The encoder keeps 6.6 s of a clip; a sample
    # whose moment is later in a longer clip - a keyframe at 20 s of 22 - names where its stretch
    # begins, and the target, every observation on the clip's clock, the camera and the mask all
    # start there together. Free references are not on the clip's clock and are not moved.
    start = float(sample.get("start") or 0.0)
    # A start given as a frame number is turned into seconds by the clip's own rate: a list's
    # rounded "fps 23" for a 23.976 clip would put frame 736 1.3 s off.
    if sample.get("start_frame"):
        start = int(sample["start_frame"]) / frame_rate(sample["target"])
    span = start + num_frames / FPS + 0.5  # what of the clip the latents can hold, and a margin
    video = read_video(sample["target"], span)
    video = (region(video[0], box), video[1])
    # The clip as it was made, before anything is cut off: counted by the container, since only its
    # start was decoded. A camera track made one pose per frame of the whole clip needs it.
    whole_frames = max(frame_count(sample["target"]), int(video[0].shape[0]))
    if start:
        video = (video[0][round(start * video[1]) :], video[1])
        if len(video[0]) < 2:
            raise ValueError(f"start {start:.2f} s is past the end of the clip")
    # Where the sample's stretch of the clip ends, as a frame of the clip, when a set curates a
    # stretch shorter than the encoder's window: reading on past it would take in what the set cut
    # away - often the next shot. Everything on the clip's clock is held to the same stretch,
    # because it is counted from the target's frames below.
    if sample.get("end_frame"):
        keep = int(sample["end_frame"]) - round(start * video[1])
        if keep < 2:
            raise ValueError(
                f"end frame {sample['end_frame']} leaves {keep} frames after the start"
            )
        video = (video[0][:keep], video[1])
    clip_frames, clip_fps = int(video[0].shape[0]), float(video[1])
    target_state = latents_only(
        references=[reference(sample["target"], shape, video, num_frames=num_frames)],
        prompt="",
        **size,
    )
    del video
    target = target_state.get("condition_latents")[0][0]  # [24, F, h, w]
    frames = int(target.shape[1])
    # What of the clip the target latents hold: H3 puts the clip on 24 fps and keeps the first
    # num_frames, so a clip longer than that is encoded from its start and the rest is not there.
    # Anything else that runs on the clip's clock - a camera track, a mask that came with the
    # clip - has to be read over this same stretch, or it describes seconds the latents never saw.
    window = min(math.floor(clip_frames * FPS / clip_fps + 0.5), num_frames)

    def seconds(latent: int) -> float:
        """Where a latent sits in the clip, in seconds from its start."""
        return latent * (window - 1) / max(frames - 1, 1) / FPS

    # Two latents is five frames, a fifth of a second: a still with a timestamp. Nothing about
    # motion, continuation or timing can be learned from it, so it is refused by name here and
    # counted, rather than encoded and quietly drawn in training.
    if frames <= 2:
        raise ValueError(
            f"target is {frames} latents ({17 * (frames - 2) // 5 + 5} frames), "
            "too short to be a video"
        )
    tensors = {"target_latents": target.to(torch.bfloat16).contiguous()}
    observations = []
    items = sample.get("observations") or []
    if items:
        state = conditioned(
            prompt=sample["text"],
            references=[
                reference(
                    o["media"],
                    shape,
                    crop=o.get("frames") is not None,
                    start=start if o.get("frames") is not None else 0.0,
                    num_frames=num_frames,
                    box=box if o.get("frames") is not None else None,
                )
                for o in items
            ],
            **size,
        )
        for index, (item, latents) in enumerate(
            zip(items, state.get("condition_latents"), strict=True)
        ):
            latents = latents[0]
            if item["media"].lower().endswith(IMAGE_EXT) and item.get("frames") is not None:
                latents = latents[:, :1]  # an aligned still: its first latent (see ``reference``)
            aligned = item.get("frames")
            if aligned == "all":
                aligned = list(range(min(int(latents.shape[1]), frames)))
                latents = latents[:, : len(aligned)]
            if aligned is not None and len(aligned) != int(latents.shape[1]):
                raise ValueError(
                    f"observation {index} has {int(latents.shape[1])} latent frames "
                    f"for {len(aligned)} target frames"
                )
            tensors[f"observation_{index}_latents"] = latents.to(torch.bfloat16).contiguous()
            observations.append(
                {
                    "frames": aligned,
                    "degree": float(item.get("degree", 0.999)),
                    "boxes": item.get("boxes"),
                }
            )
    else:
        state = text_only(prompt=sample["text"])
    prompt = state.get("prompt_embeds")
    tensors["prompt_embeds"] = prompt.reshape(-1, prompt.shape[-1]).to(torch.bfloat16).contiguous()
    tensors["text_token_tags"] = (
        state.get("text_token_tags").reshape(-1).to(torch.int64).contiguous()
    )
    if sample.get("camera"):
        # The sample hands over absolute camera-to-world poses and normalised intrinsics; the
        # reader makes them relative.
        loaded = np.load(localize(sample["camera"]))
        views_all = torch.as_tensor(np.asarray(loaded["c2w"]), dtype=torch.float32)
        k_all = torch.as_tensor(np.asarray(loaded["K"]), dtype=torch.float32)
        total = int(views_all.shape[0])
        if k_all.ndim != 2 and int(k_all.shape[0]) < total:
            raise ValueError(f"camera holds {total} poses but {int(k_all.shape[0])} intrinsics")
        # One camera per latent, taken at the latent's time. A track says its own rate when it was
        # made over part of a clip; one without says nothing because it was made one pose per
        # frame of the whole clip. Spreading the whole track over the latents instead - as this
        # once did - is right only for a clip no longer than the window; for a 20 s clip it laid
        # 20 s of camera motion over the 6.6 s the latents hold.
        track_fps = (
            float(loaded["fps"]) if "fps" in loaded.files else total * clip_fps / whole_frames
        )
        # A track starts where the clip does unless it says otherwise, so a sample that starts
        # later reads it from that far in.
        offset = start - (float(loaded["start"]) if "start" in loaded.files else 0.0)
        if offset < 0:
            raise ValueError(f"camera track starts {-offset:.2f} s after the sample does")
        if offset + seconds(frames - 1) > (total - 1) / track_fps + 1.0 / FPS:
            raise ValueError(
                f"camera track covers {(total - 1) / track_fps:.2f} s, the sample needs "
                f"{offset + seconds(frames - 1):.2f} s"
            )
        picks = [
            min(round((offset + seconds(index)) * track_fps), total - 1) for index in range(frames)
        ]
        tensors["camera_c2w"] = views_all[picks].contiguous()
        intrinsics = (k_all if k_all.ndim == 2 else k_all[picks]).clone()
        # The intrinsics describe the clip as it was shot; the encoder sees a centre-crop of it.
        # A normalised focal length is pixels over the frame's own size, so cropping the frame
        # raises it by exactly the fraction cropped away - and a field of view that disagrees with
        # the picture is worse than no intrinsics at all.
        keep_h, keep_w = crop_scale(source_size(sample["target"]), (height, width))
        if keep_h != 1.0 or keep_w != 1.0:
            intrinsics[..., 0, 0] /= keep_w
            intrinsics[..., 1, 1] /= keep_h
        # A long lens on a frame that had to be cropped hard ends up narrower than the released
        # contract can express (normalised focal at most 4, about a 14 degree vertical view). The
        # sample is dropped by name here rather than written out for the reader to reject later,
        # because a rejected member is a silent loss and a skipped sample is a counted one.
        focal = torch.stack([intrinsics[..., 0, 0].max(), intrinsics[..., 1, 1].max()])
        if float(focal.max()) > 4.0 or float(focal.min()) <= 0.0:
            raise ValueError(
                f"cropped focal {float(focal.max()):.2f} is outside the contract's 0..4; "
                "this lens is narrower than the canvas can carry"
            )
        tensors["camera_K"] = intrinsics.contiguous()
    if sample.get("known_mask"):
        # Which clock the mask runs on. One that came with the clip runs on the clip's: its frame j
        # is the clip at j / mask_fps, so it is put on 24 fps and cut to the window exactly as the
        # target was, and frame for frame it covers what the latents cover. One from the generated
        # library says what is given over the window, whatever the clip's length, and is stretched
        # over it. A dataset's mask marks the thing in the clip, so that is its default.
        clock = sample.get("known_mask_clock") or (
            "clip" if sample.get("known_mask_marks_the_region") else "window"
        )
        # white = keep (given), black = generate. A mask on the window's clock is read whole.
        if sample["known_mask"].endswith(".npz"):
            mask, mask_fps = label_mask(
                sample["known_mask"],
                int(sample["known_mask_label"]),
                clip_fps,
                start,
                num_frames / FPS + 0.5,
            )
            skip = 0  # a label map is returned already cut at the start
        else:
            mask, mask_fps = read_video(sample["known_mask"], span if clock == "clip" else 0.0)
            skip = round(start * mask_fps)
        if clock == "clip":
            # A mask of the clip's own shares its pixels, so the target's crop box cuts it too.
            mask = on_grid(region(mask[skip:], box), mask_fps)[:window]
            if len(mask) < window:  # a mask a frame short of its clip holds its last frame
                mask = np.concatenate([mask, np.repeat(mask[-1:], window - len(mask), axis=0)])
        # The target is centre-cropped to the canvas, so a mask that names a real region has to be
        # cropped the same way or it points at the wrong part of the picture. A mask already at the
        # canvas's aspect - every mask in the generated library is - comes back unchanged.
        mask = fit(mask, (height, width))
        keep = torch.from_numpy(mask[..., 0].astype(np.float32) / 255.0)[None, None]
        # A mask that came with a dataset marks the thing, not what is kept around it. Saying so
        # on the sample is cheaper and more honest than re-encoding the file: the sample records
        # which way round its own mask is written, and the convention downstream stays one.
        if sample.get("known_mask_marks_the_region"):
            keep = 1.0 - keep
        grid = torch.nn.functional.interpolate(
            keep,
            size=(frames, target.shape[2] // 2, target.shape[3] // 2),
            mode="trilinear",
            align_corners=False,
        )
        # A row is given only where the whole patch is: VAE fields leak across a mask edge.
        tensors["known_degrees"] = (grid[0, 0] > 0.98).to(torch.float16).reshape(-1).contiguous()
    meta = {
        "format": "solarrenderer.ref2va.v1",
        "observations": observations,
        "latent_frames": frames,
        "canvas": [height, width],
        # How long the clip was and how much of it the latents hold, so a member that covers only
        # the start of a long clip says so.
        "clip_seconds": round(whole_frames / clip_fps, 3),
        "start_seconds": round(start, 3),
        "window_seconds": round(window / FPS, 3),
        "source": sample.get("source", ""),
        "crop": list(box) if box else None,
        # What this sample was made from, kept beside it. The text is the one thing that can be
        # rewritten later without re-encoding the pixels - a pass over the members recomputing
        # prompt_embeds costs the text encoder only - but that pass can only work if a member
        # still says which clip it came from. Without these two lines a rewrite would depend on
        # a sample list that has since been regenerated, and the corpus would have to be built
        # again to change a word of its text.
        "target": sample.get("target", ""),
        "media": [item.get("media", "") for item in items],
        "known_mask": sample.get("known_mask", ""),
        "text": sample["text"][:16000],
    }
    return tensors, meta


def sample_name(value: str) -> str:
    """Validate the relative sample name used inside the output prefix."""
    from pathlib import PurePosixPath

    name = PurePosixPath(value)
    if not value or name.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError(f"invalid relative sample ID: {value!r}")
    if "\\" in value or "\x00" in value:
        raise ValueError("sample ID contains a path separator or null byte")
    return str(name)


def main() -> None:
    import tempfile
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Encode SolarRenderer training samples.")
    parser.add_argument("--samples", required=True, help="input JSONL, local path or storage URL")
    parser.add_argument("--out", required=True, help="output directory or storage URL")
    parser.add_argument("--shard", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--shards", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--num-frames", type=int, default=158)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards:
        parser.error("--shard must lie in [0, --shards), with --shards positive")
    if args.num_frames < 22 or (args.num_frames - 5) % 17:
        parser.error("--num-frames must be 17n+5 and at least 22")
    with open(fetch_fresh(args.samples, ""), encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row.get("id"), str) for row in rows):
        raise ValueError("sample IDs must be strings")
    identifiers = [sample_name(str(row["id"])) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample IDs must be unique")
    selected = [
        row
        for row in rows
        if int(hashlib.sha256(str(row["id"]).encode()).hexdigest(), 16) % args.shards == args.shard
    ]
    if not selected:
        print(f"Shard {args.shard}/{args.shards} has no samples", flush=True)
        return
    sink = Sink(args.out, args.shard)
    completed = {}
    if Path(sink.index_path).is_file():
        with open(sink.index_path, encoding="utf-8") as stream:
            completed = {row["id"]: row for row in map(json.loads, filter(str.strip, stream))}

    def publish_index() -> None:
        path = Path(sink.index_path)
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for identifier in sorted(completed):
                stream.write(json.dumps(completed[identifier]) + "\n")
        os.replace(temporary, path)
        sink.publish_index()

    pending = []
    for row in selected:
        previous = completed.get(row["id"])
        if previous is not None:
            location = f"{sink.out}/{previous['member']}"
            exists = storage.stat(location) is not None if sink.remote else Path(location).is_file()
            if exists:
                continue
        pending.append(row)
    if not pending:
        print(f"Shard {args.shard}/{args.shards} is complete", flush=True)
        return

    from safetensors.torch import save_file

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    pipelines = build_pipelines(f"cuda:{local_rank}")
    began = time.monotonic()
    failures = 0
    try:
        with torch.no_grad():
            for position, row in enumerate(pending, 1):
                temporary = None
                try:
                    tensors, metadata = encode(row, pipelines, args.num_frames)
                    name = f"samples/{sample_name(str(row['id']))}.safetensors"
                    member = Path(sink.member(name))
                    member.parent.mkdir(parents=True, exist_ok=True)
                    handle, temporary = tempfile.mkstemp(dir=member.parent, prefix=".encoding-")
                    os.close(handle)
                    save_file(tensors, temporary, metadata={"omni": json.dumps(metadata)})
                    os.replace(temporary, member)
                    sink.publish(name)
                    completed[row["id"]] = {
                        "id": row["id"],
                        "member": name,
                        **{key: metadata[key] for key in ("latent_frames", "canvas", "source")},
                        "observations": len(metadata["observations"]),
                    }
                    publish_index()
                    print(
                        f"Shard {args.shard}/{args.shards}: {position}/{len(pending)} encoded "
                        f"in {time.monotonic() - began:.0f} s",
                        flush=True,
                    )
                except Exception as error:
                    failures += 1
                    print(f"Failed {row['id']}: {error}", file=sys.stderr, flush=True)
                    torch.cuda.empty_cache()
                finally:
                    if temporary is not None:
                        Path(temporary).unlink(missing_ok=True)
    finally:
        publish_index()
    if failures:
        raise SystemExit(
            f"{failures} samples failed; completed samples remain available for resume"
        )


if __name__ == "__main__":
    main()
