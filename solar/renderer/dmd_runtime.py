"""Train the released bidirectional and block-causal four-step SolarRenderer LoRAs.

Adapted from SolarRenderer, Copyright 2026 SolarRenderer authors, Apache-2.0.
Modified to use SolarWM's current checkpoint runtime and separate renderer contracts.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from solar.backends.minimax_h3.distributed import get_sp_size, sync_lora_gradients
from solar.backends.minimax_h3.runtime import _base_model_load_receipt
from solar.backends.minimax_h3.sgf import h3_sgf_critic_loss
from solar.training import JsonlEventSink, StepPolicy
from solar.training.sgf import compute_sgf_kl_gradient, sgf_student_loss, should_update_student

from .runtime import RendererTrainingRuntime, read_adapter


def build_optimizer(lora: Any, config: Any) -> Any:
    from solar.backends.minimax_h3.optimizer import FP32MasterAdamW

    return FP32MasterAdamW(
        lora.parameters,
        lr=float(config["learning_rate"]),
        betas=tuple(float(value) for value in config["betas"]),
        eps=float(config.get("epsilon", 1.0e-8)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


class H3OmniDMDRuntime(RendererTrainingRuntime):
    def _verify_resume_checkpoint(self, path: str) -> Any:
        from solar.errors import BackendContractError

        verified = super()._verify_resume_checkpoint(path)
        state = self.torch.load(verified.path / "runtime.pt", map_location="cpu", weights_only=True)
        ratio = int(self.dmd["critic_updates_per_student"])
        if int(state.get("student_step", -1)) != max(0, (verified.step - 1) // ratio):
            raise BackendContractError("renderer DMD checkpoint student update count differs")
        return verified

    def __init__(self, config: Any) -> None:
        self.dmd = dict(config["train"]["dmd"])
        self.mode = str(self.dmd["mode"])
        super().__init__(config, defer_resume=True)
        torch = self.torch
        from safetensors import safe_open

        from solar.backends.minimax_h3.lora import released_lora_state, seed_lora_from_released

        path = str(self.dmd["accel_init"] if self.mode == "causal" else self.dmd["init_lora"])
        with safe_open(path, framework="pt") as handle:
            metadata = handle.metadata() or {}
        state = released_lora_state(path)
        rank = int(
            next(value for key, value in state.items() if key.endswith("lora_A.weight")).shape[0]
        )
        if self.mode == "adapt_plug" and int(self.dmd.get("init_lora_rank", rank)) != rank:
            raise ValueError("train.dmd.init_lora_rank differs from the initializer tensors")
        alpha = float(metadata.get("alpha", self.dmd.get("init_lora_alpha", rank)))
        seed_lora_from_released(
            self.lora.parameter_by_key,
            state,
            live_scale=self.lora.alpha / self.lora.rank,
            released_scale=alpha / rank,
        )
        del state
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        try:
            self.teacher, self.teacher_lora = self._build_score_role(trainable=False)
            critic, critic_lora = self._build_score_role(trainable=True)
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
        optimizer = build_optimizer(critic_lora, self.train_cfg["critic_optimizer"])
        self.extra_roles["critic"] = {
            "model": critic,
            "lora": critic_lora,
            "optimizer": optimizer,
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        }
        resume = str(self.checkpoint_cfg.get("resume_from") or "")
        if resume:
            self.load_checkpoint(resume)

    def _before_adapter(self, transformer: Any) -> None:
        from solar.backends.minimax_h3.lora import adapter_state_as_released, merge_lora_delta

        control = self.dmd["student_init"] if self.mode == "causal" else self.dmd["control"]
        merge_lora_delta(
            transformer,
            adapter_state_as_released(read_adapter(str(control))),
            alpha=float(self.model_cfg["adapter"]["alpha"]),
            rank=int(self.model_cfg["adapter"]["rank"]),
        )

    def _make_core(self) -> Any:
        from .dmd import H3OmniDMDCore
        from .observations import observation_profile_from_train_config

        options = {"observations": observation_profile_from_train_config(self.train_cfg)}
        if self.mode == "causal":
            from .causal import H3OmniCausalDMDCore

            return H3OmniCausalDMDCore(
                self.model,
                self.silence,
                self.device,
                self.dmd,
                dict(self.dmd["causal_stream"]),
                **options,
            )
        return H3OmniDMDCore(self.model, self.silence, self.device, self.dmd, **options)

    def _build_score_role(self, *, trainable: bool) -> tuple[Any, Any]:
        from solar.backends.minimax_h3.fsdp import wrap_h3_fsdp
        from solar.backends.minimax_h3.lora import (
            adapter_state_as_released,
            inject_h3_lora,
            merge_lora_delta,
        )
        from solar.backends.minimax_h3.optional import load_transformer

        seed = 0x48330000 + int(self.config["data"]["seed"])
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        modules = load_transformer(self.model_cfg, device=self.device)
        modules.transformer.requires_grad_(False)
        control = read_adapter(str(self.dmd["control"]))
        adapter = self.model_cfg["adapter"]
        ranks = (384,)
        if trainable:
            merge_lora_delta(
                modules.transformer,
                adapter_state_as_released(control),
                alpha=float(adapter["alpha"]),
                rank=int(adapter["rank"]),
            )
            rank = int(self.dmd.get("critic_rank", 128))
            adapter = {"rank": rank, "alpha": rank, "dropout": 0.0}
            ranks = (rank,)
        model, lora = inject_h3_lora(
            modules.transformer,
            adapter,
            base_identity=_base_model_load_receipt(self.model_cfg, modules.transformer),
            ranks=ranks,
        )
        model = wrap_h3_fsdp(
            model,
            local_rank=self.topology.local_rank,
            transformer_block_cls=modules.transformer_block_cls,
            fp32_units=modules.fp32_fsdp_units,
            ignored_parameters=lora.parameters,
            activation_checkpointing=trainable,
            frozen_base_shard_size=self.train_cfg["fsdp"].get("frozen_base_shard_size"),
            saved_attention_blocks=int(self.train_cfg["fsdp"].get("saved_attention_blocks", 0)),
        )
        if not trainable:
            lora.load_state_dict(control, broadcast=False)
            model.requires_grad_(False)
        return model.eval(), lora

    @property
    def structure_guidance(self) -> float:
        return float(self.dmd.get("structure_guidance", 1.0))

    def _dmd_gradient(self, fake: Any, real: Any, output: Any) -> Any:
        return compute_sgf_kl_gradient(fake_x0=fake, real_x0=real, student_output=output)

    def _shared_exit(self) -> int:
        """One exit evaluation for every rank: FSDP gathers per forward, so all ranks run alike."""
        torch = self.torch
        value = torch.randint(0, int(self.dmd.get("forwards", 4)), (), device=self.device)
        if self.dist.is_initialized():
            self.dist.broadcast(value, src=0)
        return int(value.item())

    def _shared_choice(self, count: int) -> int:
        value = self.torch.randint(0, int(count), (), device=self.device)
        if self.dist.is_initialized():
            self.dist.broadcast(value, src=0)
        return int(value.item())

    def _stream_student(self, batch: Any, *, graph: bool) -> Any:
        """Stream a clip and optionally replay it with gradients through clean history."""
        from solar.backends.minimax_h3.distributed import get_dp_rank
        from solar.backends.minimax_h3.flow import make_shifted_schedule
        from solar.backends.minimax_h3.layout import unpatchify_video

        torch, core = (self.torch, self.core)
        forwards = int(self.dmd.get("forwards", 4))
        seed = self._shared_choice(2**30) + 7919 * int(get_dp_rank())
        schedule = make_shifted_schedule(forwards + 1, shift=12.0)
        exit_index = self._shared_exit()
        with torch.no_grad():
            clip, inputs, times, jitters = core.stream(
                batch,
                generator=torch.Generator(device=self.device).manual_seed(seed),
                schedule=schedule,
                exit_index=exit_index,
                model=self.model,
            )
        if not graph:
            return clip
        replay = torch.Generator(device=self.device).manual_seed(seed)
        document = core._document(batch, replay)
        rows = core.training_prediction(
            document, inputs, times, clip, generator=replay, jitter=jitters, model=self.model
        )
        frames, grid = document[4]
        velocity = unpatchify_video(rows, frames, *grid, channels=24).float()
        level = torch.ones((1, 1, frames, 1, 1), device=self.device)
        for (a, b), t in zip(document[3].frames, times, strict=True):
            level[:, :, a:b] = 1.0 - float(t)
        return inputs + level * velocity

    def _finish_update(
        self, model: Any, lora: Any, optimizer: Any, scheduler: Any, max_norm
    ) -> float:
        sync_lora_gradients(lora.parameters)
        finite, norm = self._finite_clip_norm(
            (parameter for parameter in model.parameters() if parameter.requires_grad), max_norm
        )
        self._check_finite(finite, f"non-finite H3 DMD gradient norm={norm}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return norm

    def _phase(self, name: str) -> None:
        if self.is_main:
            print(
                f"[h3-dmd-phase] outer_step={self.global_step + 1} {name} "
                f"elapsed_s={time.monotonic() - self._step_started:.1f}",
                flush=True,
            )

    def _student_update(self, stats: dict[str, float]) -> None:
        torch, core = (self.torch, self.core)
        critic = self.extra_roles["critic"]
        forwards = int(self.dmd.get("forwards", 4))
        batch = self._next_batch()
        inputs = core.prepare_inputs(batch)
        self._phase("student_rollout")
        if self.mode == "causal":
            output = self._stream_student(batch, graph=True)
        else:
            output = core.rollout(
                inputs, model=self.model, forwards=forwards, exit_index=self._shared_exit()
            )
        with torch.no_grad():
            self._phase("student_scores")
            noisy, _noise, video_t, audio_t = core.score_inputs(output.detach())
            audio_noise = core._random(tuple(inputs.audio_clean.shape), None)
            sigma = 1.0 - video_t
            fake = noisy + sigma * core.score_forward(
                critic["model"], inputs, noisy, video_t, audio_t, audio_noise
            )
            real = core.teacher_answer(
                self.teacher,
                inputs,
                batch,
                noisy,
                video_t,
                audio_t,
                audio_noise,
                guidance=self.structure_guidance,
            )
            self._check_finite(
                bool(torch.isfinite(fake).all() and torch.isfinite(real).all()),
                "non-finite H3 DMD score output",
            )
            gradient = self._dmd_gradient(fake, real, output.detach())
        loss = sgf_student_loss(output, gradient)
        self._phase("student_backward")
        (loss / get_sp_size()).backward()
        stats["loss_student"] = float(loss.detach())
        del output, noisy, fake, real, gradient, loss
        stats["student_grad_norm"] = self._finish_update(
            self.model,
            self.lora,
            self.optimizer,
            self.scheduler,
            float(self.train_cfg["optimizer"].get("gradient_clip", 10.0)),
        )
        self.student_step += 1
        if self.ema is not None:
            self.ema_update(self.student_step)

    def _critic_update(self, stats: dict[str, float]) -> None:
        torch, core = (self.torch, self.core)
        critic = self.extra_roles["critic"]
        batch = self._next_batch()
        inputs = core.prepare_inputs(batch)
        self._phase("critic_rollout")
        with torch.no_grad():
            if self.mode == "causal":
                clean = self._stream_student(batch, graph=False)
            else:
                clean = core.rollout(
                    inputs,
                    model=self.model,
                    forwards=int(self.dmd.get("forwards", 4)),
                    exit_index=self._shared_exit(),
                )
            noisy, noise, video_t, audio_t = core.score_inputs(clean)
        self._phase("critic_forward")
        velocity = core.score_forward(critic["model"], inputs, noisy, video_t, audio_t)
        loss = h3_sgf_critic_loss(velocity, noise=noise, clean=clean)
        self._check_finite(bool(torch.isfinite(loss)), "non-finite H3 DMD critic loss")
        self._phase("critic_backward")
        (loss / get_sp_size()).backward()
        stats["loss_critic"] = float(loss.detach())
        stats["critic_grad_norm"] = self._finish_update(
            critic["model"],
            critic["lora"],
            critic["optimizer"],
            critic["scheduler"],
            float(self.train_cfg["critic_optimizer"].get("gradient_clip", 10.0)),
        )

    def train_outer_step(self) -> dict[str, float]:
        self._step_started = time.monotonic()
        ratio = int(self.dmd.get("critic_updates_per_student", 5))
        update_student = should_update_student(self.global_step, ratio)
        stats: dict[str, float] = {"student_updated": float(update_student), "loss_student": 0.0}
        if update_student:
            self._student_update(stats)
        self._critic_update(stats)
        self._global_step += 1
        stats.update(
            student_step=float(self.student_step),
            loss=stats["loss_student"] if update_student else stats["loss_critic"],
            lr=float(self.optimizer.param_groups[0]["lr"]),
            critic_lr=float(self.extra_roles["critic"]["optimizer"].param_groups[0]["lr"]),
            compute_time_s=time.monotonic() - self._step_started,
        )
        return stats


def run_omni_dmd_training(config: Any) -> int:
    runtime = H3OmniDMDRuntime(config)
    policy = StepPolicy(
        max_steps=int(config["train"]["max_steps"]),
        save_every=int(config["checkpoint"]["save_every_steps"]),
        save_steps=tuple(config["checkpoint"].get("save_steps", ())),
        validate_every=int(config["validation"]["validate_every_steps"]),
        validation_steps=(),
    )
    sink = (
        JsonlEventSink(Path(str(config["runtime"]["output_dir"])) / "training-events.jsonl")
        if runtime.is_main
        else None
    )
    try:
        while runtime.global_step < policy.max_steps:
            stats = runtime.train_outer_step()
            step = runtime.global_step
            if sink is not None:
                sink({"event": "train_step", "step": step, **stats})
            if policy.should_save(step):
                runtime.save_checkpoint(step)
    finally:
        runtime.reader.close()
    return 0
