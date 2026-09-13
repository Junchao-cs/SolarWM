"""Three-role H3 SGF updates using the public checkpoint and validation lifecycle."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from solarwm.training import JsonlEventSink, StepPolicy
from solarwm.training.sgf import compute_sgf_kl_gradient, sgf_student_loss, should_update_student

from .distributed import get_sp_size, sync_lora_gradients
from .runtime import H3TrainingRuntime, _base_model_load_receipt
from .sgf import h3_sgf_critic_loss
from .sgf_rollout import h3_sgf_replay
from .stage2 import H3SGFCore
from .torch_flow import predict_clean_sample


class H3SGFTrainingRuntime(H3TrainingRuntime):
    def __init__(self, config: Any) -> None:
        super().__init__(config, defer_resume=True)
        self.initialization = {"student": self._weights_id}
        torch = self.torch
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
        try:
            self.teacher, _ = self._build_score_role(trainable=False)
            critic, critic_lora = self._build_score_role(trainable=True)
        finally:
            torch.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng, self.device)
        from .optimizer import FP32MasterAdamW

        cfg = self.train_cfg["critic_optimizer"]
        optimizer = FP32MasterAdamW(
            critic_lora.parameters,
            lr=float(cfg["learning_rate"]),
            betas=tuple(cfg["betas"]),
            eps=float(cfg["epsilon"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        self.extra_roles["critic"] = {
            "model": critic,
            "lora": critic_lora,
            "optimizer": optimizer,
            "scheduler": scheduler,
        }
        self.model.eval()
        self.contract = replace(
            self.contract,
            extras={**self.contract.extras, "initialization": dict(self.initialization)},
        )
        resume = str(self.checkpoint_cfg.get("resume_from") or "")
        if resume:
            self.load_checkpoint(resume)
        self._check_ema_boundary()

    def _make_core(self) -> H3SGFCore:
        return H3SGFCore(self.model, self.silence, self.device)

    def _build_score_role(self, *, trainable: bool) -> tuple[Any, Any]:
        from .fsdp import wrap_h3_fsdp
        from .lora import inject_h3_lora
        from .optional import load_transformer
        from .weights import load_initial_weights

        seed = 0x48330000 + int(self.config["data"]["seed"])
        self.torch.manual_seed(seed)
        self.torch.cuda.manual_seed_all(seed)
        modules = load_transformer(self.model_cfg, device=self.device)
        modules.transformer.requires_grad_(False)
        model, lora = inject_h3_lora(
            modules.transformer,
            self.model_cfg["adapter"],
            base_identity=_base_model_load_receipt(self.model_cfg, modules.transformer),
        )
        model = wrap_h3_fsdp(
            model,
            local_rank=self.topology.local_rank,
            transformer_block_cls=modules.transformer_block_cls,
            fp32_units=modules.fp32_fsdp_units,
            ignored_parameters=lora.parameters,
            activation_checkpointing=trainable,
            frozen_base_shard_size=8,
        )
        role = "critic" if trainable else "teacher"
        identity = load_initial_weights(self.checkpoint_cfg["initialization"][role], lora)
        self.initialization[role] = identity
        if self.is_main:
            print(f"[h3-sgf-init] {role}={identity}", flush=True)
        if not trainable:
            model.requires_grad_(False)
        return model.eval(), lora

    def _check_ema_boundary(self) -> None:
        expected = self.student_step >= int(self.checkpoint_cfg["ema"]["start_step"])
        if expected != (self.ema is not None):
            raise ValueError("H3 SGF EMA presence differs from the saved student-update boundary")

    def save_checkpoint(self, step: int) -> str:
        # The replay leaves large inactive allocations. Release them before
        # checkpoint collectives, retaining live parameters, optimizer and EMA.
        self.torch.cuda.synchronize(self.device)
        self.torch.cuda.empty_cache()
        return super().save_checkpoint(step)

    def _student_ema_update(self) -> None:
        from .ema import H3ShardedEMA

        cfg = self.checkpoint_cfg["ema"]
        if self.ema is None and self.student_step >= int(cfg["start_step"]):
            self.ema = H3ShardedEMA(self.model, decay=float(cfg["decay"]), device=self.device)
            if self.is_main:
                print(f"[h3-sgf-ema] created student_step={self.student_step}", flush=True)
        if self.ema is not None:
            self.ema.update(self.model)

    def _finish_update(self, model: Any, lora: Any, optimizer: Any, scheduler: Any) -> float:
        sync_lora_gradients(lora.parameters)
        finite, norm = self._finite_clip_norm(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            10.0,
        )
        if not finite:
            raise FloatingPointError(f"non-finite H3 SGF gradient norm={norm}")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        return norm

    def _phase(self, name: str) -> None:
        if self.is_main:
            print(
                f"[h3-sgf-phase] outer_step={self.global_step + 1} {name} "
                f"elapsed_s={time.monotonic() - self._step_started:.1f}",
                flush=True,
            )

    def _rollout(self, inputs: Any) -> Any:
        return self.core.rollout(
            inputs, progress=lambda chunk: self._phase(f"rollout_chunk={chunk}/10")
        )

    def train_outer_step(self) -> dict[str, float]:
        torch, core = self.torch, self.core
        self._step_started = time.monotonic()
        critic = self.extra_roles["critic"]
        update_student = should_update_student(self.global_step, 5)
        stats = {
            "student_updated": float(update_student),
            "loss_student": 0.0,
            "student_grad_norm": 0.0,
        }
        if update_student:
            self._phase("student_rollout")
            inputs = core.prepare_inputs(self._next_batch())
            rollout = self._rollout(inputs)
            self._phase("student_replay")
            output = h3_sgf_replay(student=self.model, inputs=inputs, rollout=rollout)[:, :, :47]
            with torch.no_grad():
                self._phase("student_scores")
                noisy, noise, vt, audio, at = core.score_inputs(output.detach())
                fake_velocity = core.score_forward(critic["model"], inputs, noisy, vt, audio, at)
                real_velocity = core.score_forward(self.teacher, inputs, noisy, vt, audio, at)
                fake = predict_clean_sample(noisy, fake_velocity, vt)
                real = predict_clean_sample(noisy, real_velocity, vt)
                if not bool(torch.isfinite(fake).all() and torch.isfinite(real).all()):
                    raise FloatingPointError("non-finite H3 SGF score output")
                gradient = compute_sgf_kl_gradient(
                    fake_x0=fake, real_x0=real, student_output=output.detach()
                )
            loss = sgf_student_loss(output, gradient)
            self._phase("student_backward")
            (loss / get_sp_size()).backward()
            stats["loss_student"] = float(loss.detach())
            stats["student_grad_norm"] = self._finish_update(
                self.model, self.lora, self.optimizer, self.scheduler
            )
            self.student_step += 1
            self._student_ema_update()
            del (
                inputs,
                rollout,
                output,
                noisy,
                noise,
                audio,
                fake_velocity,
                real_velocity,
                fake,
                real,
                gradient,
                loss,
            )
        inputs = core.prepare_inputs(self._next_batch())
        self._phase("critic_rollout")
        rollout = self._rollout(inputs)
        self._phase("critic_replay")
        with torch.no_grad():
            clean = h3_sgf_replay(student=self.model, inputs=inputs, rollout=rollout)[:, :, :47]
            noisy, noise, vt, audio, at = core.score_inputs(clean)
        self._phase("critic_forward")
        velocity = core.score_forward(critic["model"], inputs, noisy, vt, audio, at)
        loss = h3_sgf_critic_loss(velocity, noise=noise, clean=clean)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite H3 SGF critic loss")
        self._phase("critic_backward")
        (loss / get_sp_size()).backward()
        stats["critic_grad_norm"] = self._finish_update(
            critic["model"],
            critic["lora"],
            critic["optimizer"],
            critic["scheduler"],
        )
        self._global_step += 1
        stats.update(
            loss_critic=float(loss.detach()),
            student_step=float(self.student_step),
            loss=stats["loss_student"] if update_student else float(loss.detach()),
            lr=float(self.optimizer.param_groups[0]["lr"]),
            critic_lr=float(critic["optimizer"].param_groups[0]["lr"]),
            compute_time_s=time.monotonic() - self._step_started,
        )
        if self.is_main:
            print(
                f"[h3-sgf] outer_step={self.global_step} student_step={self.student_step} "
                f"student_loss={stats['loss_student']:.6f} critic_loss={stats['loss_critic']:.6f} "
                f"compute_time_s={stats['compute_time_s']:.2f} "
                f"ema={'present' if self.ema is not None else 'not_started'}",
                flush=True,
            )
        return stats


def run_sgf_training(config: Any) -> int:
    runtime = H3SGFTrainingRuntime(config)
    policy = StepPolicy(
        max_steps=int(config["train"]["max_steps"]),
        save_every=int(config["checkpoint"]["save_every_steps"]),
        save_steps=tuple(config["checkpoint"].get("save_steps", ())),
        validate_every=int(config["validation"]["validate_every_steps"]),
        validation_steps=(int(config["validation"]["smoke_step"]),)
        if int(config["validation"]["smoke_step"]) > 0
        else (),
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
            if policy.should_validate(step):
                report = runtime.validate(step)
                if sink is not None:
                    sink({"event": "validation", "step": step, "report": report})
    finally:
        runtime.reader.close()
    return 0
