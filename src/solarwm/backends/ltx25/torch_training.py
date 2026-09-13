"""Real Stage0.5 training runtime used by the embedded LTX provider."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from safetensors.torch import load_file, save_file
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from solarwm.checkpoint import CheckpointTransaction, VerifiedCheckpoint
from solarwm.errors import BackendContractError
from solarwm.runtime.distributed import collective_call, collective_rank_zero_call
from solarwm.runtime.output_layout import checkpoint_model_dir
from solarwm.runtime.randomness import model_init_seed
from solarwm.runtime.serialization import canonical_json_bytes
from solarwm.training import BatchIdentity, GradientStatus, MicrobatchResult
from solarwm.training.ema import ShardedEMA
from solarwm.training.optim import FP32MasterAdamW
from solarwm.training.schedule import make_warmup_cosine

from .checkpoint import StrictModelLoadReceipt
from .runtime import (
    REQUIRED_CHECKPOINT_COMPONENTS,
    _host_rng_state,
    _restore_host_rng_state,
    checkpoint_contract,
)
from .torch_data import PreencodedBatchSource, TorchBatch
from .torch_distributed import (
    broadcast_sp_tensor,
    clip_replicated_gradient_norm,
    initialize,
    synchronize_replicated_gradients,
)
from .torch_distributed import (
    state as distributed_state,
)
from .torch_flow import prepare_objective, sample_shifted_logit_normal, velocity_loss
from .torch_model import (
    LTX25SequenceParallelModel,
    SolarLTX25VideoTransformerBlock,
    StrictLoadedModel,
    inject_lora,
)

ValidationHook = Callable[["LTX25TrainingRuntime", int], Mapping[str, Any]]


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _wrap_fsdp(
    model: torch.nn.Module,
    *,
    config: Mapping[str, Any],
    ignored: tuple[torch.nn.Parameter, ...],
) -> FSDP:
    import functools

    runtime = distributed_state()
    fsdp_config = config["train"]["fsdp"]
    strategy = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }[str(fsdp_config["sharding_strategy"])]
    wrapped = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={SolarLTX25VideoTransformerBlock},
        ),
        sharding_strategy=strategy,
        mixed_precision=MixedPrecision(
            param_dtype=None,
            reduce_dtype=torch.float32,
            buffer_dtype=None,
            cast_root_forward_inputs=True,
        ),
        cpu_offload=None,
        device_id=runtime.local_rank,
        use_orig_params=True,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sync_module_states=False,
        process_group=runtime.fsdp_process_group,
        limit_all_gathers=True,
        ignored_states=ignored,
    )
    blocks = [
        module
        for module in wrapped.modules()
        if isinstance(module, SolarLTX25VideoTransformerBlock)
    ]
    if len(blocks) != 48:
        raise BackendContractError(
            f"LTX activation-checkpoint block inventory is {len(blocks)}, expected 48"
        )
    wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    selected = {id(module) for module in blocks}
    apply_activation_checkpointing(
        wrapped,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: id(module) in selected,
    )
    return wrapped


class LTX25TrainingRuntime:
    """Shared-engine adapter around strict LTX-Core model math."""

    def __init__(
        self,
        config: Mapping[str, Any],
        loaded: StrictLoadedModel,
        model_receipt: StrictModelLoadReceipt,
        *,
        validation_hook: ValidationHook,
        resume_checkpoint: VerifiedCheckpoint | None = None,
        batch_source: Any | None = None,
    ) -> None:
        self.config = config
        self.model_receipt = model_receipt
        self.validation_hook = validation_hook
        self.distributed = initialize(dict(config))
        self.device = torch.device("cuda", self.distributed.local_rank)
        data = config["data"]
        train = config["train"]
        optimizer_config = train["optimizer"]
        seed = int(data.get("seed", 42))
        initialization_seed = model_init_seed("ltx25_video", seed)
        torch.manual_seed(initialization_seed)
        torch.cuda.manual_seed_all(initialization_seed)

        loaded.backbone.requires_grad_(False)
        core, self.lora = inject_lora(loaded.core)
        loaded.backbone.transformer = LTX25SequenceParallelModel(core)
        ignored = tuple(dict.fromkeys((*loaded.fp32_scale_tables, *self.lora.parameters)))
        if len(loaded.fp32_scale_tables) != 97:
            raise BackendContractError("strict LTX model did not expose 97 FP32 tables")
        self.generator = _wrap_fsdp(
            loaded.backbone,
            config=config,
            ignored=ignored,
        )
        self.lora.broadcast()
        self.optimizer = FP32MasterAdamW(
            self.lora.parameters,
            lr=float(optimizer_config["learning_rate"]),
            betas=tuple(float(item) for item in optimizer_config["betas"]),
            eps=float(optimizer_config["epsilon"]),
            weight_decay=float(optimizer_config["weight_decay"]),
        )
        self.scheduler = make_warmup_cosine(
            self.optimizer,
            warmup_steps=int(optimizer_config["warmup_steps"]),
            total_steps=int(train["max_steps"]),
            min_lr_ratio=float(optimizer_config.get("min_lr_ratio", 0.1)),
        )
        ema_config = config["checkpoint"]["ema"]
        self.ema = ShardedEMA(
            self.generator,
            decay=float(ema_config["decay"]),
            device=self.device,
            dtype=torch.float32,
            trainable_only=True,
        )
        root = getattr(self.generator, "module", self.generator)
        ema_name_by_parameter = {
            id(parameter): name
            for name, parameter in root.named_parameters()
            if parameter.requires_grad
        }
        try:
            self._ema_name_by_adapter_key = {
                key: ema_name_by_parameter[id(parameter)]
                for key, parameter in self.lora.parameter_by_key.items()
            }
        except KeyError as exc:
            raise BackendContractError(
                "LTX EMA cannot map one LoRA parameter to its FSDP name"
            ) from exc
        if set(self._ema_name_by_adapter_key.values()) != set(self.ema.shadow):
            raise BackendContractError("LTX EMA inventory differs from portable LoRA state")
        self.data = PreencodedBatchSource(config) if batch_source is None else batch_source
        self.grad_clip = float(optimizer_config["gradient_clip"])
        self._global_step = 0
        self._last_gradient = GradientStatus(True, 0.0)
        self._data_seed = seed * 100003 + self.distributed.dp_rank * 1024
        self._checkpoint_id = self.data.checkpoint_id
        torch.manual_seed(self._data_seed)
        torch.cuda.manual_seed_all(self._data_seed)
        if resume_checkpoint is not None:
            self.load_checkpoint(resume_checkpoint)

    @property
    def global_step(self) -> int:
        return self._global_step

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def _collective(self, call: Callable[[], Any], label: str) -> Any:
        return collective_call(
            call,
            dist=dist,
            rank=self.distributed.rank,
            world_size=self.distributed.world_size,
            label=f"LTX {label}",
            error_type=BackendContractError,
        )

    def _rank_zero_collective(self, call: Callable[[], Any], label: str) -> Any:
        return collective_rank_zero_call(
            call,
            dist=dist,
            rank=self.distributed.rank,
            world_size=self.distributed.world_size,
            label=f"LTX {label}",
            error_type=BackendContractError,
        )

    def _device_batch(self, batch: TorchBatch) -> dict[str, Any]:
        value = {
            "video_latent": batch.video_latent.to(
                self.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            ),
            "first_frame_latent": batch.first_frame_latent.to(
                self.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            ),
            "video_prompt_embeds": batch.video_prompt_embeds.to(
                self.device,
                dtype=torch.bfloat16,
                non_blocking=True,
            ),
            "prompt_attention_mask": batch.prompt_attention_mask.to(
                self.device,
                dtype=torch.int64,
                non_blocking=True,
            ),
            "viewmats": batch.relative_w2c.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            ),
            "camera_k": batch.camera_k.to(
                self.device,
                dtype=torch.float32,
                non_blocking=True,
            ),
        }
        return value

    def train_microbatch(self, micro_index: int, grad_accum: int) -> MicrobatchResult:
        def prepare() -> tuple[
            dict[str, Any],
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            BatchIdentity,
        ]:
            batch = self.data.next()
            device_batch = self._device_batch(batch)
            video = device_batch["video_latent"]
            noise = torch.randn_like(video)
            sampling = self.config["train"]["timestep_sampling"]
            sigma = sample_shifted_logit_normal(
                video.shape[0],
                device=self.device,
                std=float(sampling["std"]),
                epsilon=float(sampling["epsilon"]),
                uniform_probability=float(sampling["uniform_probability"]),
            )
            first_frame_mask = torch.zeros(
                video.shape[0],
                20,
                device=self.device,
                dtype=torch.bool,
            )
            first_frame_mask[:, 0] = True
            identity = BatchIdentity(
                sample_ids=(batch.sample_id,),
                start_frames=(batch.start_frame,),
                noise_seeds=(self._data_seed + self._global_step * grad_accum + micro_index,),
                checkpoint_id=self._checkpoint_id,
                plan_fingerprint=batch.plan_fingerprint,
            )
            return device_batch, noise, sigma, first_frame_mask, identity

        device_batch, noise, sigma, first_frame_mask, identity = prepare()
        for tensor in device_batch.values():
            broadcast_sp_tensor(tensor)
        broadcast_sp_tensor(noise)
        broadcast_sp_tensor(sigma)
        video = device_batch["video_latent"]
        objective = prepare_objective(
            video,
            device_batch["first_frame_latent"],
            noise,
            sigma,
        )
        synchronize = micro_index == grad_accum - 1
        context = nullcontext() if synchronize else self.generator.no_sync()
        with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            prediction = self.generator(
                video_latent=objective.noisy,
                sigma=objective.sigma,
                caption_embedding=device_batch["video_prompt_embeds"],
                first_frame_mask=first_frame_mask,
                camera={
                    "viewmats": device_batch["viewmats"],
                    "K": device_batch["camera_k"],
                },
                caption_mask=device_batch["prompt_attention_mask"],
            )
        # Keep the reduction outside autocast at the FP32 loss boundary.
        loss = velocity_loss(prediction, objective.target_velocity)
        if not bool(torch.isfinite(loss).item()):
            raise BackendContractError("LTX Stage0.5 produced a non-finite loss")
        (loss / (grad_accum * self.distributed.sp_size)).backward()
        return MicrobatchResult(identity, {"loss": float(loss.detach().item())})

    def assert_sp_peer_identity(self, identity: BatchIdentity) -> None:
        del identity

    def prepare_optimizer_step(self) -> GradientStatus:
        synchronize_replicated_gradients(self.lora.parameters)
        norm = clip_replicated_gradient_norm(self.lora.parameters, self.grad_clip)
        finite = bool(torch.isfinite(norm).item())
        self._last_gradient = GradientStatus(
            finite,
            float(norm.item()) if finite else float("nan"),
        )
        return self._last_gradient

    def optimizer_step(self) -> None:
        self.optimizer.step()

    def scheduler_step(self) -> None:
        self.scheduler.step()

    def ema_update(self, step: int) -> None:
        ema_config = self.config["checkpoint"]["ema"]
        if step >= int(ema_config["start_step"]) and (
            step % int(ema_config["update_every_steps"]) == 0
        ):
            self.ema.update(self.generator)

    def set_global_step(self, step: int) -> None:
        if step != self._global_step + 1:
            raise BackendContractError("LTX global step must advance by exactly one")
        self._global_step = step

    def _rank_runtime_state(self) -> dict[str, Any]:
        return {
            "schema": "solarwm.ltx25.rank-runtime.v2",
            "rank": self.distributed.rank,
            "world_size": self.distributed.world_size,
            "global_step": self._global_step,
            "checkpoint_id": self._checkpoint_id,
            **_host_rng_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(self.device),
            "reader": self.data.state_dict(),
        }

    def save_checkpoint(self, step: int) -> str:
        def validate_step() -> None:
            if step != self._global_step:
                raise BackendContractError("LTX checkpoint step differs from runtime")

        self._collective(validate_step, "checkpoint step preflight")
        local_rank_state = self._rank_runtime_state()
        if dist.is_initialized():
            rank_states: list[Any] = [None] * self.distributed.world_size
            dist.all_gather_object(rank_states, local_rank_state)
        else:
            rank_states = [local_rank_state]
        target = checkpoint_model_dir(
            str(self.config["runtime"]["output_dir"]),
            step=int(step),
            width=8,
        )
        transaction_holder: list[CheckpointTransaction] = []

        def setup_transaction() -> str:
            transaction = CheckpointTransaction(target)
            transaction.__enter__()
            for component in REQUIRED_CHECKPOINT_COMPONENTS:
                (transaction.path / component).mkdir(parents=True, exist_ok=False)
            transaction_holder.append(transaction)
            return str(transaction.path)

        staging = Path(
            str(
                self._rank_zero_collective(
                    setup_transaction,
                    "checkpoint staging setup",
                )
            )
        )

        def commit_transaction() -> str:
            if len(transaction_holder) != 1:
                raise BackendContractError("LTX rank-zero checkpoint transaction is missing")
            transaction = transaction_holder[0]
            for rank, rank_state in enumerate(rank_states):
                torch.save(
                    rank_state,
                    staging / "runtime" / f"rank-{rank:05d}.pt",
                )
            adapter_state = {
                key: value.detach().cpu().contiguous()
                for key, value in self.lora.state_dict().items()
            }
            save_file(adapter_state, staging / "adapter" / "model.safetensors")
            _atomic_json(staging / "adapter" / "metadata.json", self.lora.metadata())
            ema_state = {
                key: self.ema.shadow[name].detach().cpu().contiguous()
                for key, name in self._ema_name_by_adapter_key.items()
            }
            save_file(ema_state, staging / "ema" / "model.safetensors")
            _atomic_json(
                staging / "ema" / "metadata.json",
                {
                    "schema": "solarwm.ltx25.ema.v1",
                    "num_updates": self.ema.num_updates,
                    "decay": self.ema.decay,
                    "dtype": "float32",
                    "trainable_only": True,
                    "state_keys": list(self.lora.keys),
                },
            )
            torch.save(self.optimizer.state_dict(), staging / "optimizer" / "state.pt")
            torch.save(self.scheduler.state_dict(), staging / "scheduler" / "state.pt")
            committed = transaction.commit(
                step=step,
                contract=checkpoint_contract(self.config),
                required_components=REQUIRED_CHECKPOINT_COMPONENTS,
                metadata={
                    "schema": "solarwm.ltx25.training-checkpoint.v1",
                    "model_load_receipt": self.model_receipt.as_dict(),
                    "ema_num_updates": self.ema.num_updates,
                },
            )
            return committed.manifest_digest

        checkpoint_identity = str(
            self._rank_zero_collective(
                commit_transaction,
                "checkpoint corpus commit",
            )
        )
        if not checkpoint_identity:
            raise BackendContractError("LTX checkpoint commit returned no identity")
        self._checkpoint_id = checkpoint_identity
        return str(target)

    def load_checkpoint(self, checkpoint: VerifiedCheckpoint) -> None:
        """Restore every algorithm-bearing component and rank-local stream state."""

        root = checkpoint.path

        def read_state() -> tuple[Any, ...]:
            adapter_metadata_path = root / "adapter" / "metadata.json"
            ema_metadata_path = root / "ema" / "metadata.json"
            try:
                adapter_metadata = json.loads(adapter_metadata_path.read_text())
                ema_metadata = json.loads(ema_metadata_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise BackendContractError("LTX resume metadata is missing or invalid") from exc
            if adapter_metadata != self.lora.metadata():
                raise BackendContractError("LTX resume adapter metadata differs")
            if (
                ema_metadata.get("schema") != "solarwm.ltx25.ema.v1"
                or ema_metadata.get("dtype") != "float32"
                or ema_metadata.get("trainable_only") is not True
                or tuple(ema_metadata.get("state_keys", ())) != self.lora.keys
            ):
                raise BackendContractError("LTX resume EMA metadata differs")
            adapter = load_file(
                str(root / "adapter" / "model.safetensors"),
                device="cpu",
            )
            portable_ema = load_file(
                str(root / "ema" / "model.safetensors"),
                device="cpu",
            )
            if set(portable_ema) != set(self._ema_name_by_adapter_key):
                raise BackendContractError("LTX resume EMA key inventory differs")
            optimizer = torch.load(
                root / "optimizer" / "state.pt",
                map_location="cpu",
                weights_only=True,
            )
            scheduler = torch.load(
                root / "scheduler" / "state.pt",
                map_location="cpu",
                weights_only=True,
            )
            runtime_path = root / "runtime" / f"rank-{self.distributed.rank:05d}.pt"
            runtime_files = tuple((root / "runtime").glob("rank-*.pt"))
            if len(runtime_files) != self.distributed.world_size:
                raise BackendContractError("LTX resume raw-rank state inventory differs")
            state = torch.load(runtime_path, map_location="cpu", weights_only=True)
            if (
                state.get("schema")
                not in {
                    "solarwm.ltx25.rank-runtime.v1",
                    "solarwm.ltx25.rank-runtime.v2",
                }
                or int(state.get("rank", -1)) != self.distributed.rank
                or int(state.get("world_size", -1)) != self.distributed.world_size
                or int(state.get("global_step", -1)) != checkpoint.step
            ):
                raise BackendContractError("LTX resume rank runtime identity differs")
            return adapter, portable_ema, ema_metadata, optimizer, scheduler, state

        adapter, portable_ema, ema_metadata, optimizer, scheduler, state = self._collective(
            read_state,
            "resume checkpoint read",
        )

        def apply_state() -> None:
            self.lora.load_state_dict(adapter, broadcast=False)
            internal_ema = {
                self._ema_name_by_adapter_key[key]: value for key, value in portable_ema.items()
            }
            self.ema.load_state_dict(
                internal_ema,
                num_updates=int(ema_metadata["num_updates"]),
            )
            self.optimizer.load_state_dict(optimizer)
            self.scheduler.load_state_dict(scheduler)
            self.data.load_state_dict(state["reader"])
            if state["schema"] == "solarwm.ltx25.rank-runtime.v2":
                _restore_host_rng_state(state)
            torch.set_rng_state(state["torch_rng_state"])
            torch.cuda.set_rng_state(state["cuda_rng_state"], self.device)

        self._collective(apply_state, "resume checkpoint apply")
        self._global_step = checkpoint.step
        self._checkpoint_id = checkpoint.manifest_digest

    def validate(self, step: int) -> Mapping[str, Any]:
        def validate_step() -> None:
            if step != self._global_step:
                raise BackendContractError("LTX validation step differs from runtime")

        self._collective(validate_step, "validation step preflight")
        return self.validation_hook(self, step)

    def close(self) -> None:
        self.data.close()


__all__ = ["LTX25TrainingRuntime", "ValidationHook"]
