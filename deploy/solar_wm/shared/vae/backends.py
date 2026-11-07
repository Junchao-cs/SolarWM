"""Two explicit VAE choices on the DiT GPUs, with isolated compiler settings."""
from contextlib import contextmanager
from pathlib import Path
import time


@contextmanager
def decoder_context(cache, backend):
    import torch
    import torch._inductor.config as config
    dynamo = torch._dynamo.config
    name = "recompile_limit" if hasattr(dynamo, "recompile_limit") else "cache_size_limit"
    prior = (config.emulate_precision_casts, getattr(dynamo, name))
    # Legacy Boolean getters lose "none" (inherited) versus "ieee" state.
    # Restore the complete compiler-key state, not just its effective Boolean.
    precision_owners = (torch.backends, torch.backends.cuda.matmul,
                        torch.backends.cudnn, torch.backends.cudnn.conv,
                        torch.backends.cudnn.rnn)
    precision = tuple(owner.fp32_precision for owner in precision_owners)
    try:
        config.emulate_precision_casts = backend == "current"
        setattr(dynamo, name, max(128, prior[1]))
        if backend == "streamrender":
            # Exactly the StreamRender entry's effective decode policy.
            # Native streaming VAE did not override these backend settings.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        with cache.installed("vae-" + backend), torch.inference_mode():
            yield
    finally:
        config.emulate_precision_casts, limit = prior
        setattr(dynamo, name, limit)
        for owner, value in zip(precision_owners, precision):
            owner.fp32_precision = value


def finite_cpu(value):
    import numpy as np
    import torch
    if value.device.type != "cpu":
        raise ValueError("actual CPU output required")
    value = value.detach()
    if value.dtype == torch.bfloat16:
        bits = value.contiguous().view(torch.int16).numpy().view(np.uint16)
        return not bool(np.any((bits & 0x7f80) == 0x7f80))
    return bool(np.isfinite(value.numpy()).all())


class Backends:
    def __init__(self, deployment, model_config, *, device, rank, cache):
        import torch.distributed as dist
        self.config, self.model_config = deployment, model_config
        self.device, self.rank, self.cache = device, rank, cache
        self.current_group = dist.new_group(list(range(deployment["vae_ranks"])), backend="nccl")
        self.models = {}
        self.backend = None
        self.next_chunk = 0

    def load(self, backend):
        import torch
        if backend in self.models:
            return
        with decoder_context(self.cache, backend):
            if backend == "current":
                from diffusers import AutoencoderKLMiniMaxH3
                from .tile_group import GroupTileDecoder
                decoder = None
                if self.rank < self.config["vae_ranks"]:
                    vae = AutoencoderKLMiniMaxH3.from_pretrained(
                        self.model_config["model"]["checkpoint_path"], subfolder="vae",
                        torch_dtype=torch.float32)
                    vae.eval().requires_grad_(False).to(self.device)
                    vae.decoder.to(dtype=torch.bfloat16)
                    vae.post_quant_conv.to(dtype=torch.bfloat16)
                    decoder = GroupTileDecoder(vae, group=self.current_group, compile=True)
                self.models[backend] = decoder
            elif backend == "streamrender":
                from safetensors.torch import load_file
                from streamrender_h3 import load_config
                from streamrender_h3.decoder.causal_vit_decoder import CausalViTDecoder
                from .streamrender_tp import TensorParallelCausalViT
                weight = Path(self.config["streamrender_weights"]).resolve(strict=True)
                model = CausalViTDecoder.from_config(load_config(), chunk=1, cache_latents=4)
                model.load_state_dict(load_file(str(weight)), strict=True, assign=True)
                model.to(self.device).eval().requires_grad_(False)
                self.models[backend] = TensorParallelCausalViT(model, 48, 84,
                    cuda_graph=True, variant="fused_split_head")
            else:
                raise ValueError("unsupported decoder")

    def start(self, backend):
        self.load(backend)
        self.backend, self.next_chunk = backend, 0
        decoder = self.models[backend]
        if decoder is not None:
            with decoder_context(self.cache, backend):
                if backend == "streamrender":
                    decoder.init_state()
                else:
                    decoder.reset()

    def push(self, index, latent, on_piece):
        import torch
        import torch.distributed as dist
        if self.backend is None or index != self.next_chunk:
            raise RuntimeError("decoder request reordered")
        decoder = self.models[self.backend]
        started = time.perf_counter()
        with decoder_context(self.cache, self.backend):
            # All ranks already own the same SP output. The broadcast binds the
            # VAE input to rank zero, preserving the deployed decoder contract.
            dist.broadcast(latent, src=0)
            if decoder is not None:
                for offset in range(5):
                    z = latent[:, :, offset:offset + 1]
                    if self.backend == "streamrender":
                        output = decoder.decode_chunk(z.transpose(1, 2).contiguous())
                    else:
                        vae = decoder.vae
                        mean = z.new_tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
                        std = z.new_tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
                        output = decoder.push(z * std + mean)
                    if self.rank == 0:
                        expected = (1, 3, 1 if offset == 0 else 4, 768, 1344)
                        if tuple(output.shape) != expected:
                            raise RuntimeError("decoder frame geometry differs")
                        cpu = torch.empty(expected, dtype=output.dtype, device="cpu", pin_memory=True)
                        cpu.copy_(output, non_blocking=True)
                        torch.cuda.current_stream(self.device).synchronize()
                        if not finite_cpu(cpu):
                            raise FloatingPointError("nonfinite decoder pixels")
                        on_piece(index, offset, cpu, time.monotonic_ns())
            torch.cuda.synchronize(self.device)
        self.next_chunk += 1
        return (time.perf_counter() - started) * 1000

    def finish(self):
        self.backend = None

    def close(self):
        for decoder in self.models.values():
            graph = getattr(decoder, "graph", None)
            if graph is not None:
                graph.reset()
                decoder.graph = None
        self.models.clear()
