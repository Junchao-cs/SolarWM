"""Wan transformer, VAE, and text-encoder runtime adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from solarwm.errors import BackendContractError

from .assets import WanAssetLayout
from .camera import (
    CAMERA_TRANSLATION_LINEAR,
    normalize_camera_translation_transform,
    transform_relative_viewmats,
)
from .loader import WeightLoadReport, build_camera_transformer
from .scheduler import FlowMatchScheduler


class WanTextEncoder:
    """Frozen UMT5-XXL encoder with the model tokenization contract."""

    def __init__(self, weights: str | Path, tokenizer: str | Path) -> None:
        import torch

        from .modeling.t5 import umt5_xxl
        from .modeling.tokenizers import HuggingfaceTokenizer

        self.module = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )
        try:
            state = torch.load(str(weights), map_location="cpu", weights_only=True)
            self.module.load_state_dict(state, strict=True)
        except Exception as exc:
            raise BackendContractError(f"cannot load UMT5 weights {weights}: {exc}") from exc
        self.tokenizer = HuggingfaceTokenizer(name=str(tokenizer), seq_len=512, clean="whitespace")

    def to(self, device: Any, *, dtype: Any | None = None) -> WanTextEncoder:
        self.module.to(device=device, dtype=dtype)
        return self

    @property
    def device(self) -> Any:
        return next(self.module.parameters()).device

    def __call__(self, prompts: Sequence[str]) -> Mapping[str, Any]:
        ids, mask = self.tokenizer(list(prompts), return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        sequence_lengths = mask.gt(0).sum(dim=1).long()
        context = self.module(ids, mask)
        for row, length in zip(context, sequence_lengths, strict=True):
            row[length:] = 0.0
        return {"prompt_embeds": context}


class Wan5BVAE:
    """Frozen 48-channel Wan2.2 VAE with deterministic normalized moments."""

    mean = (
        -0.2289,
        -0.0052,
        -0.1323,
        -0.2339,
        -0.2799,
        0.0174,
        0.1838,
        0.1557,
        -0.1382,
        0.0542,
        0.2813,
        0.0891,
        0.1570,
        -0.0098,
        0.0375,
        -0.1825,
        -0.2246,
        -0.1207,
        -0.0698,
        0.5109,
        0.2665,
        -0.2108,
        -0.2158,
        0.2502,
        -0.2055,
        -0.0322,
        0.1109,
        0.1567,
        -0.0729,
        0.0899,
        -0.2799,
        -0.1230,
        -0.0313,
        -0.1649,
        0.0117,
        0.0723,
        -0.2839,
        -0.2083,
        -0.0520,
        0.3748,
        0.0152,
        0.1957,
        0.1433,
        -0.2944,
        0.3573,
        -0.0548,
        -0.1681,
        -0.0667,
    )
    std = (
        0.4765,
        1.0364,
        0.4514,
        1.1677,
        0.5313,
        0.4990,
        0.4818,
        0.5013,
        0.8158,
        1.0344,
        0.5894,
        1.0901,
        0.6885,
        0.6165,
        0.8454,
        0.4978,
        0.5759,
        0.3523,
        0.7135,
        0.6804,
        0.5833,
        1.4146,
        0.8986,
        0.5659,
        0.7069,
        0.5338,
        0.4889,
        0.4917,
        0.4069,
        0.4999,
        0.6866,
        0.4093,
        0.5709,
        0.6065,
        0.6415,
        0.4944,
        0.5726,
        1.2042,
        0.5458,
        1.6887,
        0.3971,
        1.0600,
        0.3943,
        0.5537,
        0.5444,
        0.4089,
        0.7468,
        0.7744,
    )

    def __init__(self, weights: str | Path) -> None:
        import torch

        from .modeling.vae import _video_vae

        self._mean = torch.tensor(self.mean, dtype=torch.float32)
        self._std = torch.tensor(self.std, dtype=torch.float32)
        try:
            self.module = (
                _video_vae(
                    pretrained_path=str(weights),
                    z_dim=48,
                    temperal_downsample=[False, True, True],
                )
                .eval()
                .requires_grad_(False)
            )
        except Exception as exc:
            raise BackendContractError(f"cannot load Wan VAE {weights}: {exc}") from exc

    def to(self, device: Any, *, dtype: Any | None = None) -> Wan5BVAE:
        self.module.to(device=device, dtype=dtype)
        return self

    def _scale(self, reference: Any) -> list[Any]:
        return [
            self._mean.to(device=reference.device, dtype=reference.dtype),
            1.0 / self._std.to(device=reference.device, dtype=reference.dtype),
        ]

    def encode(self, pixels_bcthw: Any) -> Any:
        import torch

        encoded = [
            self.module.encode(clip.unsqueeze(0), self._scale(clip)).float().squeeze(0)
            for clip in pixels_bcthw
        ]
        return torch.stack(encoded, dim=0).permute(0, 2, 1, 3, 4)

    def decode(self, latents_btchw: Any, *, use_cache: bool = False) -> Any:
        import torch

        clips = latents_btchw.permute(0, 2, 1, 3, 4)
        decode = self.module.cached_decode if use_cache else self.module.decode
        output = []
        for clip in clips:
            with torch.autocast(device_type=clip.device.type, dtype=clip.dtype):
                decoded = decode(clip.unsqueeze(0), self._scale(clip))
            output.append(decoded.float().clamp_(-1, 1).squeeze(0))
        return torch.stack(output, dim=0).permute(0, 2, 1, 3, 4)

    def decode_streaming(
        self,
        latents_btchw: Any,
        *,
        chunk_latent_frames: int = 60,
    ) -> Any:
        """Decode consecutive temporal tiles and concatenate them on CPU."""

        import torch

        if (
            latents_btchw.ndim != 5
            or int(latents_btchw.shape[0]) <= 0
            or int(latents_btchw.shape[1]) <= 0
        ):
            raise BackendContractError("Wan streaming VAE decode requires non-empty BTCHW latents")
        outputs = []
        for sample in latents_btchw.split(1, dim=0):
            chunks = list(
                self.decode_streaming_chunks(
                    sample,
                    chunk_latent_frames=chunk_latent_frames,
                )
            )
            if not chunks:
                raise BackendContractError("Wan streaming VAE decode produced no chunks")
            outputs.append(torch.cat(chunks, dim=1))
        return torch.cat(outputs, dim=0)

    def decode_streaming_chunks(
        self,
        latents_btchw: Any,
        *,
        chunk_latent_frames: int = 60,
    ) -> Any:
        """Yield CPU BTCHW tiles while preserving one continuous VAE cache."""

        import torch

        if latents_btchw.ndim != 5 or int(latents_btchw.shape[1]) <= 0:
            raise BackendContractError("Wan streaming VAE decode requires non-empty BTCHW latents")
        if int(latents_btchw.shape[0]) != 1:
            raise BackendContractError("Wan streaming VAE file decode requires batch size one")
        if chunk_latent_frames <= 0:
            raise BackendContractError("Wan streaming VAE chunk size must be positive")
        clear_cache = getattr(self.module, "clear_cache", None)
        cached_decode = getattr(self.module, "cached_decode", None)
        if not callable(clear_cache) or not callable(cached_decode):
            raise BackendContractError(
                "Wan streaming VAE decode requires cache-aware decoder methods"
            )

        clips = latents_btchw.permute(0, 2, 1, 3, 4)
        for clip in clips:
            clear_cache()
            try:
                for start in range(0, int(clip.shape[1]), chunk_latent_frames):
                    chunk = clip[:, start : start + chunk_latent_frames].contiguous()
                    with torch.autocast(device_type=chunk.device.type, dtype=chunk.dtype):
                        decoded = cached_decode(chunk.unsqueeze(0), self._scale(chunk))
                    if not bool(torch.isfinite(decoded).all().item()):
                        raise BackendContractError(
                            "Wan streaming VAE decode produced non-finite pixels"
                        )
                    yield (decoded.float().clamp_(-1, 1).permute(0, 2, 1, 3, 4).contiguous().cpu())
                    del decoded
            finally:
                clear_cache()


class WanA14BVAE:
    """Official deterministic 16-channel Wan2.1 VAE used by I2V-A14B."""

    mean = (
        -0.7571,
        -0.7089,
        -0.9113,
        0.1075,
        -0.1745,
        0.9653,
        -0.1517,
        1.5508,
        0.4134,
        -0.0715,
        0.5517,
        -0.3632,
        -0.1922,
        -0.9497,
        0.2503,
        -0.2921,
    )
    std = (
        2.8184,
        1.4541,
        2.3275,
        2.6558,
        1.2196,
        1.7708,
        2.6052,
        2.0743,
        3.2687,
        2.1526,
        2.8652,
        1.5579,
        1.6382,
        1.1253,
        2.8251,
        1.9160,
    )

    def __init__(self, weights: str | Path) -> None:
        import torch
        from diffusers import AutoencoderKLWan
        from diffusers.loaders.single_file_utils import (
            convert_wan_vae_to_diffusers,
        )

        self._mean = torch.tensor(self.mean, dtype=torch.float32)
        self._std = torch.tensor(self.std, dtype=torch.float32)
        try:
            state = torch.load(
                str(weights),
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            if not isinstance(state, Mapping) or not state:
                raise BackendContractError(
                    "Wan2.1 A14B VAE checkpoint must be a non-empty state dict"
                )
            with torch.device("meta"):
                module = AutoencoderKLWan()
            target_keys = set(module.state_dict())
            if target_keys.issubset(state):
                normalized = dict(state)
            else:
                normalized = convert_wan_vae_to_diffusers(dict(state))
            result = module.load_state_dict(normalized, strict=True, assign=True)
            if result.missing_keys or result.unexpected_keys:
                raise BackendContractError(
                    "Wan2.1 A14B VAE state load was not exact: "
                    f"missing={result.missing_keys[:8]} "
                    f"unexpected={result.unexpected_keys[:8]}"
                )
            if any(parameter.is_meta for parameter in module.parameters()):
                raise BackendContractError("Wan2.1 A14B VAE exact load left meta parameters")
            self.module = module.eval().requires_grad_(False)
        except Exception as exc:
            raise BackendContractError(f"cannot load Wan2.1 A14B VAE {weights}: {exc}") from exc

    def to(self, device: Any, *, dtype: Any | None = None) -> WanA14BVAE:
        self.module.to(device=device, dtype=dtype)
        return self

    def _scale(self, reference: Any) -> list[Any]:
        return [
            self._mean.to(device=reference.device, dtype=reference.dtype),
            1.0 / self._std.to(device=reference.device, dtype=reference.dtype),
        ]

    def encode(self, pixels_bcthw: Any) -> Any:
        import torch

        output = []
        for clip in pixels_bcthw:
            posterior = self.module.encode(clip.unsqueeze(0)).latent_dist
            mean = posterior.mode()
            if mean.shape[1] != 16:
                raise BackendContractError("Wan2.1 A14B VAE must produce 16 deterministic channels")
            center, inverse_std = self._scale(mean)
            normalized = (mean - center.view(1, 16, 1, 1, 1)) * inverse_std.view(1, 16, 1, 1, 1)
            output.append(normalized.float().squeeze(0))
        return torch.stack(output, dim=0).permute(0, 2, 1, 3, 4)

    def decode(self, latents_btchw: Any, *, use_cache: bool = False) -> Any:
        del use_cache
        import torch

        output = []
        for clip in latents_btchw.permute(0, 2, 1, 3, 4):
            with torch.autocast(device_type=clip.device.type, dtype=clip.dtype):
                center, inverse_std = self._scale(clip)
                raw = clip.unsqueeze(0) / inverse_std.view(1, 16, 1, 1, 1) + center.view(
                    1, 16, 1, 1, 1
                )
                decoded = self.module.decode(raw).sample
            finite = torch.isfinite(decoded)
            if not finite.all():
                bad_frames = (~finite).any(dim=(0, 1, 3, 4)).nonzero(as_tuple=False).flatten()
                first_bad_frame = int(bad_frames[0].item()) if bad_frames.numel() else -1
                raise BackendContractError(
                    "Wan2.1 A14B VAE decode produced non-finite pixels: "
                    f"normalized_latent_min={float(clip.min().item()):.6g} "
                    f"normalized_latent_max={float(clip.max().item()):.6g} "
                    f"normalized_latent_absmax={float(clip.abs().max().item()):.6g} "
                    f"raw_latent_min={float(raw.min().item()):.6g} "
                    f"raw_latent_max={float(raw.max().item()):.6g} "
                    f"raw_latent_absmax={float(raw.abs().max().item()):.6g} "
                    f"finite_fraction={float(finite.float().mean().item()):.9f} "
                    f"first_bad_pixel_frame={first_bad_frame}"
                )
            output.append(decoded.float().clamp_(-1, 1).squeeze(0))
        return torch.stack(output, dim=0).permute(0, 2, 1, 3, 4)


class WanDiffusion:
    """Layout adapter around :class:`CausalWanModel`."""

    def __init__(
        self,
        module: Any,
        *,
        timestep_shift: float,
        frame_sequence_length: int,
        num_output_frames: int,
        camera_translation_transform: str = CAMERA_TRANSLATION_LINEAR,
    ) -> None:
        self.module = module
        self.frame_sequence_length = int(frame_sequence_length)
        self.num_output_frames = int(num_output_frames)
        self.camera_translation_transform = normalize_camera_translation_transform(
            camera_translation_transform
        )
        self.scheduler = FlowMatchScheduler(shift=float(timestep_shift))

    def _camera(self, camera: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.camera_translation_transform == CAMERA_TRANSLATION_LINEAR:
            return camera
        if "viewmats" not in camera:
            raise BackendContractError("Wan camera input lacks viewmats")
        transformed = dict(camera)
        transformed["viewmats"] = transform_relative_viewmats(
            camera["viewmats"],
            self.camera_translation_transform,
        )
        return transformed

    def __call__(
        self,
        noisy_btchw: Any,
        condition: Mapping[str, Any],
        camera: Mapping[str, Any],
        timestep_tokens: Any,
        *,
        i2v_y: Any | None = None,
        sequence_length: int | None = None,
        r_timestep_tokens: Any | None = None,
        kv_cache: Any | None = None,
        crossattn_cache: Any | None = None,
        current_start: int = 0,
        cache_start: int = 0,
        cache_update_policy: str = "none",
    ) -> Any:
        sequence_length = sequence_length or (self.frame_sequence_length * self.num_output_frames)
        output = self.module(
            noisy_btchw.permute(0, 2, 1, 3, 4),
            t=timestep_tokens,
            context=condition["prompt_embeds"],
            y=i2v_y.permute(0, 2, 1, 3, 4) if i2v_y is not None else None,
            y_camera=self._camera(camera),
            seq_len=int(sequence_length),
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=int(current_start),
            cache_start=int(cache_start),
            cache_update_policy=str(cache_update_policy),
            r=r_timestep_tokens,
        )
        return output.permute(0, 2, 1, 3, 4)

    def forward_train_tf(
        self,
        noisy_btchw: Any,
        clean_btchw: Any,
        condition: Mapping[str, Any],
        camera: Mapping[str, Any],
        timestep_tokens: Any,
        augmentation_timestep_tokens: Any,
        *,
        num_frame_per_block: int,
        i2v_y: Any | None = None,
        sequence_length: int | None = None,
        r_timestep_tokens: Any | None = None,
        augmentation_r_timestep_tokens: Any | None = None,
    ) -> Any:
        """Run Stage1 TF through the root module's public FSDP forward.

        Calling ``module.forward_train_tf`` directly would bypass the root
        FSDP pre/post-forward hooks. The explicit mode is therefore dispatched
        by :class:`CausalWanModel.forward` while the wrapped root is active.
        """

        sequence_length = sequence_length or (self.frame_sequence_length * self.num_output_frames)
        output = self.module(
            noisy_btchw.permute(0, 2, 1, 3, 4),
            t=timestep_tokens,
            context=condition["prompt_embeds"],
            y=i2v_y.permute(0, 2, 1, 3, 4) if i2v_y is not None else None,
            y_camera=self._camera(camera),
            seq_len=int(sequence_length),
            r=r_timestep_tokens,
            training_mode="teacher_forcing",
            clean_x=clean_btchw.permute(0, 2, 1, 3, 4),
            aug_t=augmentation_timestep_tokens,
            aug_r=augmentation_r_timestep_tokens,
            num_frame_per_block=int(num_frame_per_block),
        )
        return output.permute(0, 2, 1, 3, 4)

    def forward_inference_window(
        self,
        noisy_btchw: Any,
        clean_history_btchw: Any | None,
        condition: Mapping[str, Any],
        camera: Mapping[str, Any],
        timestep_tokens: Any,
        *,
        num_frame_per_block: int,
        i2v_y: Any | None = None,
        sequence_length: int | None = None,
        clean_history_timestep: Any | None = None,
        r_timestep_tokens: Any | None = None,
        clean_history_r_timestep: Any | None = None,
    ) -> Any:
        """Run cache-free AR inference through the public FSDP root.

        The underlying model owns the specialized sliding-window forward, but
        invoking that method directly would skip FSDP pre/post-forward hooks.
        As with teacher forcing, an explicit root-forward mode keeps the
        layout conversion here and the distributed execution in one place.
        """

        sequence_length = sequence_length or (
            self.frame_sequence_length * int(noisy_btchw.shape[1])
        )
        output = self.module(
            noisy_btchw.permute(0, 2, 1, 3, 4),
            t=timestep_tokens,
            context=condition["prompt_embeds"],
            y=i2v_y.permute(0, 2, 1, 3, 4) if i2v_y is not None else None,
            y_camera=self._camera(camera),
            seq_len=int(sequence_length),
            r=r_timestep_tokens,
            training_mode="inference_window",
            clean_history=(
                clean_history_btchw.permute(0, 2, 1, 3, 4)
                if clean_history_btchw is not None
                else None
            ),
            num_frame_per_block=int(num_frame_per_block),
            clean_history_timestep=clean_history_timestep,
            clean_history_r_timestep=clean_history_r_timestep,
        )
        return output.permute(0, 2, 1, 3, 4)

    def flow_to_x0(self, noisy_btchw: Any, flow_btchw: Any, timestep_frames: Any) -> Any:
        """Recover ``x0`` from the Wan rectified-flow velocity.

        Preserve the frozen Wan wrapper's numerical contract: select sigma
        from the shifted scheduler grid and perform the subtraction in FP64
        before casting back to the model dtype.  In particular, legacy
        self-forcing inference supplies BF16 timestep tokens, so using the
        rounded token value directly as ``sigma * 1000`` changes the rollout.
        """

        import torch

        noisy = torch.as_tensor(noisy_btchw)
        flow = torch.as_tensor(flow_btchw, device=noisy.device)
        if noisy.shape != flow.shape:
            raise BackendContractError("Wan flow/x_t shapes differ while reconstructing x0")
        timestep = torch.as_tensor(timestep_frames, device=noisy.device)
        if timestep.shape != noisy.shape[:2]:
            raise BackendContractError(
                "Wan x0 reconstruction requires [batch, latent_frames] timesteps"
            )
        original_dtype = flow.dtype
        flat_noisy = noisy.flatten(0, 1).double()
        flat_flow = flow.flatten(0, 1).double()
        schedule_timesteps = self.scheduler.timesteps.to(device=noisy.device, dtype=torch.float64)
        schedule_sigmas = self.scheduler.sigmas.to(device=noisy.device, dtype=torch.float64)
        timestep_ids = (
            (schedule_timesteps.unsqueeze(0) - timestep.flatten().double().unsqueeze(1))
            .abs()
            .argmin(dim=1)
        )
        sigma = schedule_sigmas[timestep_ids].reshape(-1, 1, 1, 1)
        return (flat_noisy - sigma * flat_flow).to(original_dtype).unflatten(0, noisy.shape[:2])


def build_diffusion(config: Mapping[str, Any]) -> tuple[WanDiffusion, WeightLoadReport]:
    model_config = config["model"]
    architecture_config = dict(config)
    architecture_model = dict(model_config)
    architecture_model["camera_translation_transform"] = CAMERA_TRANSLATION_LINEAR
    architecture_config["model"] = architecture_model
    module, report = build_camera_transformer(architecture_config)
    return (
        WanDiffusion(
            module,
            timestep_shift=float(model_config["timestep_shift"]),
            frame_sequence_length=int(model_config["frame_sequence_length"]),
            num_output_frames=int(model_config["num_output_frames"]),
            camera_translation_transform=str(
                model_config.get("camera_translation_transform", CAMERA_TRANSLATION_LINEAR)
            ),
        ),
        report,
    )


def build_diffusion_architecture(config: Mapping[str, Any]) -> WanDiffusion:
    """Build an uninitialized diffusion role for exact full-state loading."""

    from .loader import build_camera_transformer_architecture

    model_config = config["model"]
    architecture_config = dict(config)
    architecture_model = dict(model_config)
    architecture_model["camera_translation_transform"] = CAMERA_TRANSLATION_LINEAR
    architecture_config["model"] = architecture_model
    module = build_camera_transformer_architecture(architecture_config)
    return WanDiffusion(
        module,
        timestep_shift=float(model_config["timestep_shift"]),
        frame_sequence_length=int(model_config["frame_sequence_length"]),
        num_output_frames=int(model_config["num_output_frames"]),
        camera_translation_transform=str(
            model_config.get("camera_translation_transform", CAMERA_TRANSLATION_LINEAR)
        ),
    )


def build_online_components(
    config: Mapping[str, Any],
) -> tuple[WanDiffusion, WanTextEncoder, Wan5BVAE | WanA14BVAE, WeightLoadReport]:
    text_encoder, vae = build_online_codec_components(config)
    diffusion, report = build_diffusion(config)
    return diffusion, text_encoder, vae, report


def build_online_codec_components(
    config: Mapping[str, Any],
) -> tuple[WanTextEncoder, Wan5BVAE | WanA14BVAE]:
    model = config.get("model", {})
    if not isinstance(model, Mapping):
        raise BackendContractError("model must be a mapping")
    family = str(model.get("family"))
    if family not in {"wan22_ti2v_5b", "wan22_i2v_a14b"}:
        raise BackendContractError(f"unsupported Wan online codec family {family!r}")
    layout = WanAssetLayout.from_config(config)
    vae = Wan5BVAE(layout.vae) if family == "wan22_ti2v_5b" else WanA14BVAE(layout.vae)
    return WanTextEncoder(layout.text_encoder, layout.tokenizer), vae


__all__ = [
    "Wan5BVAE",
    "WanA14BVAE",
    "WanDiffusion",
    "WanTextEncoder",
    "build_diffusion",
    "build_diffusion_architecture",
    "build_online_codec_components",
    "build_online_components",
]
