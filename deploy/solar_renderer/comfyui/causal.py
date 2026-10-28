"""Long clips, chunk by chunk: SolarRenderer's causal LoRA pair on ComfyUI's own MiniMax-H3.

The causal renderer generates a clip five latent frames (about 17 video frames) at a time, each chunk seeing the prompt,
the first frame, the soundtrack, the structure of its own and the five chunks before it, and the rendered frames of
those five chunks - nothing later. So a clip can run far past the 158 frames of a single render, with the memory of a
six-chunk window. This is the sampler of the released ``--causal`` path (solar.renderer.render, solar.renderer.causal):

* the prompt and the first frame (time 1 and 0.999) and the held silence (0.999) are written to a key/value cache once;
* each chunk's structure (0.999) is written once, seeing the prompt, the first frame and itself;
* each chunk is denoised in four evaluations against the cache, re-noised between them to the next level, and written
  back at 0.999 with its own clean frames.

Every forward runs only the rows it adds - one chunk - against the cached keys and values of the rest.
"""  # noqa: E501

import logging

import comfy.model_management as mm
import comfy.nested_tensor
import comfy.quant_ops
import comfy.utils
import latent_preview
import torch
from comfy.ldm.minimax.model import (
    PackedLayout,
    pack_audio,
    patchify_video,
    rope_rotation_table,
    unpatchify_video,
)

from .nodes import CATEGORY, SHIFT_AUDIO, SHIFT_VIDEO, SILENCE, _fa4, shifted_sigmas

STATIC_TIME = 1.0  # the prompt rows
CLEAN_LEVEL = 0.999  # the first frame, the structure, the soundtrack and every rendered chunk as the next ones see it  # noqa: E501
CHUNK, WINDOW, STEPS = 5, 6, 4
TAG_VIDEO, TAG_AUDIO = (
    0,
    2,
)  # ComfyUI's modality tags (text rows carry their own, from the text encoder)


def _silence(length: int, edge: int = 20):
    """H3's encoded silence over ``length`` audio latents, ``[1, 32, 2, length]``: away from its first and last
    ``edge`` columns the encoded silence is constant, so a longer one keeps both ends and repeats a middle column."""  # noqa: E501
    silence = (
        comfy.utils.load_torch_file(SILENCE)["silence_170f"].float().permute(1, 0, 2).unsqueeze(0)
    )
    columns = silence.shape[-1]
    if length <= columns:
        return silence[..., :length]
    middle = silence[..., columns // 2 : columns // 2 + 1].expand(
        *silence.shape[:-1], length - 2 * edge
    )
    return torch.cat(
        (silence[..., :edge], middle, silence[..., columns - edge :]), dim=-1
    ).contiguous()


def _attention_kernel():
    fa4 = _fa4()

    def attend(q, k, v):  # [1, tokens, heads, head_dim]
        if fa4 is not None and q.is_cuda:
            out = fa4(q, k, v)
            return out[0] if isinstance(out, tuple) else out
        from torch.nn.attention import SDPBackend, sdpa_kernel

        order = [
            SDPBackend.CUDNN_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
        with sdpa_kernel(order, set_priority=True):
            out = torch.nn.functional.scaled_dot_product_attention(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            )
        return out.transpose(1, 2)

    return attend


class _Cache:
    """Per layer, the keys and values (after the rotary) of written rows under ``(kind, chunk)``; chunks of a kind
    older than the window are dropped as new ones arrive. Kept on ``device`` (the GPU, or system memory for a small
    GPU: about 1 MB per row over all layers)."""  # noqa: E501

    def __init__(self, window: int, device):
        self.layers, self.window, self.device = {}, window, device

    def put(self, layer, name, key, value):
        entries = self.layers.setdefault(layer, {})
        entries[name] = (key.to(self.device, copy=True), value.to(self.device, copy=True))
        kind, chunk = name
        for old in [n for n in entries if n[0] == kind and n[1] <= chunk - self.window]:
            entries.pop(old)

    def get(self, layer, name, device):
        key, value = self.layers[layer][name]
        return key.to(device, non_blocking=True), value.to(device, non_blocking=True)


class CausalRun:
    """One clip on a loaded MiniMax-H3 (``dm``): its document's rows, the cache, and the chunk loop."""  # noqa: E501

    def __init__(self, dm, device, dtype, context, extra, latent_shapes, seed, cache_device):
        self.dm, self.device, self.dtype = dm, device, dtype
        self.attend = _attention_kernel()
        self.generator = torch.Generator(device=device).manual_seed(int(seed))
        (_, _, frames, height, width), audio_t = latent_shapes[0], latent_shapes[1][-1]
        if height % 2 or width % 2:
            raise ValueError(
                "the latent's height and width must be even (width and height multiples of 32)"
            )
        self.frames, self.height, self.width = frames, height, width
        self.per_frame = (height // 2) * (width // 2)

        keyframes = extra.get("minimax_keyframes") or []
        if not keyframes or keyframes[0]["latent"].shape[2] != frames:
            raise ValueError(
                "SolarRenderer Causal Sampler takes the conditioning of SolarRenderer Structure to Video"  # noqa: E501
            )
        context = context.to(device, dtype)
        layout = PackedLayout(context.shape[1], frames, height, width, audio_t, keyframes=keyframes)
        self.positions = layout.position_ids
        segments = layout.segments
        self.text = segments[0][:2]
        conds = [s[:2] for s in segments if s[2] == "cond"]
        self.structure_start = conds[0][0]
        self.first = conds[1] if len(conds) > 1 else None
        self.audio = next(s[:2] for s in segments if s[2] == "audio")
        self.video_start = next(s[0] for s in segments if s[2] == "video")

        # the condition rows (structure, then the first frame), as ComfyUI noises them to 0.999 for
        # every render
        self.cond_rows = dm._cond_video_rows(
            {"cond_video_latents": [k["latent"] for k in keyframes], "seed": int(seed)}, device
        )
        self.text_embed = dm.preprocess_text_embeds(context)[0]
        tags = extra.get("minimax_token_tags")
        self.text_tags = tags.view(-1).tolist() if tags is not None else [1] * context.shape[1]
        clean = _silence(audio_t).to(device)
        noise = torch.randn(
            clean.shape, generator=self.generator, device=device, dtype=torch.float32
        )
        self.audio_embed = dm.audio_patch_proj(
            pack_audio(CLEAN_LEVEL * clean + (1.0 - CLEAN_LEVEL) * noise)
        ).to(dtype)
        self.cache = _Cache(WINDOW, cache_device)

    # one forward

    def _video_embed(self, rows):
        return self.dm.video_patch_proj(rows.to(self.device, torch.float32)).to(self.dtype)

    def _forward(self, pieces, cached, commit, sigma=None):
        """Run the rows ``pieces`` [(embed, (start, stop) in the document, time, tag)] against the cached ``cached``;
        write their keys and values as ``commit``; with ``sigma``, return the last piece's velocity rows."""  # noqa: E501
        dm = self.dm
        h = torch.cat([p[0] for p in pieces]).contiguous()
        positions = torch.cat([self.positions[a:b] for _, (a, b), _, _ in pieces])
        times = sorted({p[2] for p in pieces})
        index = {t: i for i, t in enumerate(times)}
        mod, row = [], 0
        for embed, _, t, tag in pieces:
            n = embed.shape[0]
            if isinstance(tag, list):  # the prompt: runs of its rows' own tags
                start = 0
                for i in range(1, n + 1):
                    if i == n or tag[i] != tag[start]:
                        mod.append((row + start, row + i, index[t] * 3 + int(tag[start])))
                        start = i
            else:
                mod.append((row, row + n, index[t] * 3 + tag))
            row += n

        t_vals = torch.tensor(times, dtype=torch.float32, device=self.device)
        if dm.use_adaln_curves:
            table = mm.cast_to(dm.adaln_t_table, device=self.device)
            pos = t_vals.clamp(0.0, 1.0) * (table.shape[0] - 1)
            i0 = pos.floor().long().clamp(max=table.shape[0] - 2)
            t_emb = torch.lerp(table[i0], table[i0 + 1], (pos - i0).unsqueeze(1))
        else:
            t_emb = dm.time_embedder(t_vals).to(self.dtype)
        rope = rope_rotation_table(dm.rope_freqs(positions, self.device), self.dtype)

        for layer, block in enumerate(dm.blocks):
            h = block(
                h,
                t_emb,
                mod,
                rope,
                transformer_options={},
                attention=self._attention(block.attn, layer, cached, commit),
            )
        if sigma is None:
            return None
        n = pieces[-1][0].shape[0]
        a, b = row - n, row
        video, _ = dm.final_layer(
            h,
            t_emb,
            (a, b, index[pieces[-1][2]]),
            (b, b, index[pieces[-1][2]]),
            torch.tensor(sigma),
            None,
            (SHIFT_VIDEO, SHIFT_AUDIO),
        )
        return video

    def _attention(self, attn, layer, cached, commit):
        """The block's attention over its own rows and the cached ones: H3's q/k norm and rotary as ComfyUI computes
        them, the new keys and values written to the cache when the forward commits."""  # noqa: E501

        def run(x, rope_freqs=None, transformer_options=None):
            s, heads, dim = x.shape[0], attn.heads, attn.head_dim
            q, k, v = attn.qkv_proj(x).split(heads * dim, dim=-1)
            v = v.view(s, heads, dim)
            q, k = q.view(1, s, heads, dim), k.view(1, s, heads, dim)
            qw = mm.cast_to(attn.q_norm.weight, device=x.device)
            kw = mm.cast_to(attn.k_norm.weight, device=x.device)
            comfy.quant_ops.ck.rms_rope_split_half_(
                q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rope_freqs.shape[-3] * 2
            )
            k = k[0]
            if commit is not None:
                self.cache.put(layer, commit, k, v)
            keys, values = [], []
            for name in cached:
                ck, cv = self.cache.get(layer, name, x.device)
                keys.append(ck)
                values.append(cv)
            keys.append(k)
            values.append(v)
            out = self.attend(
                q.contiguous(),
                torch.cat(keys).unsqueeze(0).contiguous(),
                torch.cat(values).unsqueeze(0).contiguous(),
            )
            return attn.out_proj(out.reshape(s, heads * dim))

        return run

    # the clip

    def prefill(self):
        """The prompt and the first frame, then the soundtrack, into the cache."""
        static = [(self.text_embed, self.text, STATIC_TIME, self.text_tags)]
        if self.first is not None:
            n = self.frames * self.per_frame
            static.append(
                (
                    self._video_embed(self.cond_rows[n : n + self.per_frame]),
                    self.first,
                    CLEAN_LEVEL,
                    TAG_VIDEO,
                )
            )
        self._forward(static, (), ("static", 0))
        self._forward(
            [(self.audio_embed, self.audio, CLEAN_LEVEL, TAG_AUDIO)], [("static", 0)], ("audio", 0)
        )

    def chunk(self, c, f0, f1):
        """Render latent frames ``f0:f1`` (chunk ``c``); returns them ``[1, 24, f1 - f0, h, w]``."""
        rows = (f0 * self.per_frame, f1 * self.per_frame)
        structure = (self.structure_start + rows[0], self.structure_start + rows[1])
        target = (self.video_start + rows[0], self.video_start + rows[1])
        self._forward(
            [
                (
                    self._video_embed(self.cond_rows[rows[0] : rows[1]]),
                    structure,
                    CLEAN_LEVEL,
                    TAG_VIDEO,
                )
            ],
            [("static", 0)],
            ("structure", c),
        )
        window = range(max(0, c - WINDOW + 1), c + 1)
        history = [
            ("static", 0),
            ("audio", 0),
            *(("structure", j) for j in window),
            *(("clean", j) for j in window if j < c),
        ]

        shape = (1, 24, f1 - f0, self.height, self.width)

        def draw():
            return torch.randn(
                shape, generator=self.generator, device=self.device, dtype=torch.float32
            )

        sigmas = shifted_sigmas(STEPS)
        current = draw()
        for i in range(STEPS):
            sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
            embed = self._video_embed(patchify_video(current, self.dm.patch_size))
            velocity = self._forward(
                [(embed, target, 1.0 - sigma, TAG_VIDEO)], history, None, sigma=sigma
            )
            velocity = unpatchify_video(
                velocity, f1 - f0, self.height // 2, self.width // 2, 24, self.dm.patch_size
            ).float()
            estimate = current + sigma * velocity
            current = (
                estimate if i == STEPS - 1 else (1.0 - sigma_next) * estimate + sigma_next * draw()
            )
        written = CLEAN_LEVEL * current + (1.0 - CLEAN_LEVEL) * draw()
        self._forward(
            [
                (
                    self._video_embed(patchify_video(written, self.dm.patch_size)),
                    target,
                    CLEAN_LEVEL,
                    TAG_VIDEO,
                )
            ],
            history,
            ("clean", c),
        )
        return current


class SolarRendererCausalSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": "MiniMax-H3 with the two causal SolarRenderer LoRAs loaded "
                        "(solarrenderer_causal_4step_lora, then solarrenderer_causal_control_lora)."
                    },
                ),
                "positive": (
                    "CONDITIONING",
                    {"tooltip": "From SolarRenderer Structure to Video, at any length."},
                ),
                "latent": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cache": (
                    ["auto", "gpu", "system memory"],
                    {
                        "tooltip": "Where the six-chunk key/value cache lives (about 60 GB at 1344x768). 'system "  # noqa: E501
                        "memory' lets a smaller GPU render long clips, more slowly; 'auto' picks "
                        "the GPU when it has room."
                    },
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Renders the clip chunk by chunk with the causal LoRA pair: any length, the memory of a six-chunk "  # noqa: E501
        "window. Split the result with Separate AV Latent and decode the video with VAE Decode."
    )

    @torch.no_grad()
    def sample(self, model, positive, latent, seed, cache):
        streams = latent["samples"]
        shapes = (
            [t.shape for t in streams.tensors] if getattr(streams, "is_nested", False) else None
        )
        if shapes is None or len(shapes) < 2:
            raise ValueError(
                "SolarRenderer Causal Sampler takes the latent of SolarRenderer Structure to Video"
            )
        mm.load_models_gpu([model])
        device, dtype = model.load_device, model.model.get_dtype_inference()
        dm = model.model.diffusion_model
        context, extra = positive[0][0], positive[0][1]

        _, _, frames, height, width = shapes[0]
        rows = (height // 2) * (width // 2)
        need = (
            rows
            * CHUNK
            * (2 * WINDOW)
            * 2
            * dm.blocks[0].attn.heads
            * dm.blocks[0].attn.head_dim
            * 2
            * len(dm.blocks)
        )
        on_gpu = cache == "gpu" or (cache == "auto" and mm.get_free_memory(device) > need * 1.3)
        run = CausalRun(
            dm,
            device,
            dtype,
            context,
            extra,
            shapes,
            seed,
            device if on_gpu else torch.device("cpu"),
        )
        chunks = [(f, min(f + CHUNK, frames)) for f in range(0, frames, CHUNK)]
        logging.info(
            f"SolarRenderer: {len(chunks)} chunks, cache on {'the GPU' if on_gpu else 'system memory'} "  # noqa: E501
            f"(up to {need / 2**30:.0f} GB)"
        )

        callback = latent_preview.prepare_callback(model, len(chunks))
        run.prefill()
        clip = torch.zeros(shapes[0], dtype=torch.float32)
        for c, (f0, f1) in enumerate(chunks):
            mm.throw_exception_if_processing_interrupted()
            rendered = run.chunk(c, f0, f1)
            clip[:, :, f0:f1] = rendered.cpu()
            try:
                callback(c, rendered, None, len(chunks))
            except Exception as exc:
                logging.debug(f"SolarRenderer: no preview for chunk {c}: {exc}")
        del run
        audio = _silence(shapes[1][-1]).to(clip.dtype)
        out = latent.copy()
        out["samples"] = comfy.nested_tensor.NestedTensor(
            (clip.to(mm.intermediate_device()), audio)
        )
        return (out,)
