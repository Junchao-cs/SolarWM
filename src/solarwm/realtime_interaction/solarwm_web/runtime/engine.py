"""Resident SP8 model, exact-length scenes and denoise/refresh Graph ownership."""
from contextlib import ExitStack, nullcontext
import importlib.util
import json
from pathlib import Path
import time

from .scene_math import NativeSceneMath
from .navigation import NavigationCamera
from .configuration import file_sha256
from .camera_window import local_window
from .compiler import create_compiler
from .operators import OperatorProfile, ResidentPhaseOptimizations, MetadataRebuildPhasePins
from ..acceleration.graph.resident import ResidentFullStudentRuntime


class Engine(NativeSceneMath):
    def __init__(self, deployment, *, rank, device, run, compiler_cache, control_group):
        import torch
        import torch.distributed as dist
        from solarwm.backends.minimax_h3.artifacts import load_silence_latents
        from solarwm.backends.minimax_h3.optional import load_transformer
        from solarwm.backends.minimax_h3.lora import inject_h3_lora
        from solarwm.backends.minimax_h3.runtime import _base_model_load_receipt
        from solarwm.backends.minimax_h3.weights import load_initial_weights
        from solarwm.backends.minimax_h3.sgf import h3_sgf_schedule
        from ..acceleration.attention.history import PackedHistoryCache

        self.deployment, self.rank, self.device, self.run = deployment, rank, device, Path(run)
        self.compiler_cache, self.request_group, self.request_dist = compiler_cache, control_group, dist
        self.cfg = json.loads(Path(deployment["model_config"]).read_text())
        if self.cfg["distributed"]["sequence_parallel_size"] != 8:
            raise ValueError("model configuration must use SP8")
        torch.set_num_threads(2)
        torch.manual_seed(42)
        modules = load_transformer(self.cfg["model"], device=device)
        receipt = _base_model_load_receipt(self.cfg["model"], modules.transformer)
        self.model, self.lora = inject_h3_lora(modules.transformer, self.cfg["model"]["adapter"],
                                              base_identity=receipt)
        checkpoint = self.cfg["checkpoint"]
        self.weights = load_initial_weights(dict(path=checkpoint["resume_from"], stage="stage2",
                    weight_source=checkpoint["weight_source"]), self.lora)
        expected = self.cfg["inference"].get("expected_step")
        if expected is not None and self.weights != f"stage2:{checkpoint['weight_source']}:step={expected}":
            raise RuntimeError("loaded checkpoint label differs from configuration")
        self.model.eval().requires_grad_(False)
        self.base = self.model.get_base_model()
        if type(self.base).__name__ != "SolarMiniMaxH3Transformer3DModel" or len(self.base.transformer_blocks) != 50:
            raise RuntimeError("50-block H3 model required")
        self.native_forward = self.base.forward
        silence, _ = load_silence_latents(Path(deployment["silence"]), pixel_frames=158)
        self.silence = silence.to(device).float().unsqueeze(0).permute(0, 1, 3, 2).reshape(1, -1, 32).contiguous()
        self.schedule, _ = h3_sgf_schedule(device=device)
        self.cache = PackedHistoryCache()
        self.inputs = self.noise = self.rng = self.scopes = self.owners = None
        self.index, self.session_count = 0, 0
        self.poses, self.session_times = [], []
        self.layout_key = self.condition = None
        self.failed = False
        self.compiler = create_compiler(self.native_forward,
            evidence_directory=self.run / f"compiler-rank{rank}", rank=rank)
        # The released entry uses a zero metadata-rebuild budget when reusing
        # its existing build. Keep that same fail-fast contract after relocation.
        from ..acceleration.compile.cache import MetadataRebuildBudget
        self.cache_reuse = MetadataRebuildBudget(0) if deployment.get("dit_cache_seed") else None
        self.dynamic_evidence = self.compiler.evidence
        self.full_resident = ResidentFullStudentRuntime(device=device,
            emit=self.dynamic_evidence.emit, capacity=deployment["graph_capacity"])
        self.steady_optim = self.pins = None
        if deployment["acceleration"] == "tuned":
            # An explicit build profile is required. Never silently claim the
            # untuned cold compiler is the previously qualified tuned recipe.
            fields = json.loads(Path(deployment["operator_profile"]).read_text())
            profile = OperatorProfile(**{k: Path(v) for k, v in fields.items()})
            self.steady_optim = ResidentPhaseOptimizations(profile=profile, rank=rank,
                device=device, emit=self.dynamic_evidence.emit,
                inductor_cache=compiler_cache.root / "dit/inductor",
                triton_cache=compiler_cache.root / "dit/triton", rebuild_metadata=True)
            self.pins = MetadataRebuildPhasePins(rank, profile.seed, profile.alternative,
                                                 self.dynamic_evidence.emit)
            self.pins.set_variant("B")
        torch.cuda.synchronize(device)

    def select_scene(self, record):
        if self.scopes is not None:
            raise RuntimeError("finish active session before switching scene")
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError("condition file changed")
        self.condition = dict(record)

    def start(self, *, seed=42):
        from safetensors import safe_open
        if self.failed or self.condition is None:
            raise RuntimeError("healthy engine and selected scene required")
        with safe_open(self.condition["path"], framework="numpy") as source:
            length = source.get_slice("prompt_embeds").get_shape()[-2]
        if not 1000 <= length <= 1300:
            raise ValueError("actual prompt length outside supported 1000..1300 domain")
        self.session_count += 1
        self.dynamic_evidence.scene = self.condition["scene_id"]
        self.dynamic_evidence.sequence = self.session_count
        self.session_times = []
        self.session_started = time.perf_counter()
        self._start_inputs(seed=seed)
        self.camera = NavigationCamera(self.camera)
        self.navigation_speeds = (1., 1.)
        self.compiler.begin_scene_checks(self.request_dist, self.request_group)
        if self.steady_optim:
            self.steady_optim.bind_request(self.inputs)
        self.full_resident.begin_session(key=self.layout_key, engine=self,
                                        rotaries=self.owners.core.rotaries)

    def _enter_graph_scope(self):
        from solarwm.backends.minimax_h3 import model as model_module, distributed as parallel
        from ..acceleration.attention.backend import installed_candidate as attention
        from ..acceleration.attention.history import installed_candidate as packed
        from ..acceleration.attention.scope import combined_core, hook_snapshot
        from ..acceleration.attention.collectives import install, validate_balanced
        from ..acceleration.compile.packed_window import install as sidecars
        from ..acceleration.attention.collectives import contiguous_packed_bounds
        stack = ExitStack()
        self.before_hooks = hook_snapshot(self.base)
        self.prior_forward_present = "forward" in self.base.__dict__
        self.prior_forward = self.base.forward
        try:
            prior = parallel.contiguous_packed_bounds
            parallel.contiguous_packed_bounds = contiguous_packed_bounds
            stack.callback(setattr, parallel, "contiguous_packed_bounds", prior)
            stack.callback(install().close)
            stack.callback(sidecars().close)
            self.owners = stack.enter_context(combined_core(self.inputs, self.base, enabled=True))
            validate_balanced()
            flash = importlib.util.find_spec("flash_attn.flash_attn_interface")
            stack.enter_context(attention(model_module, flash_source=Path(flash.origin)))
            stack.enter_context(packed())
            self.base.forward = self.compiler
            self.scopes = stack
        except BaseException:
            stack.close()
            self.failed = True
            raise

    def set_navigation(self, move, turn):
        from .navigation import speed
        self.navigation_speeds = (speed(move, maximum=10), speed(turn, maximum=3))

    def generate(self):
        import torch
        import torch._inductor.compile_fx as compile_fx
        from solarwm.backends.minimax_h3 import sgf_rollout
        from solarwm.backends.minimax_h3.torch_flow import scale_noise
        if self.failed or self.scopes is None:
            raise RuntimeError("active healthy scene required")
        index = self.index
        offset = index if index < 58 else (index - 58) % 10
        if index >= 58 and offset == 0:
            self.noise = self.random_like(torch.empty(1, 24, 50, 48, 84, device=self.device))
        current = self.noise[:, :, offset * 5:(offset + 1) * 5]
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        try:
            reuse = (self.cache_reuse.scope(compile_fx, self.dynamic_evidence.emit)
                     if self.cache_reuse else nullcontext())
            from .seeded_cache import require_seeded_aot
            aot_reuse = (require_seeded_aot(self.dynamic_evidence)
                         if self.cache_reuse else nullcontext())
            with self.compiler_cache.installed("dit"), torch.no_grad(), reuse, aot_reuse, local_window(self.inputs.camera_viewmats, index):
                if index < 5:
                    with self.pins if self.pins else nullcontext():
                        for ordinal, timestep in enumerate(self.schedule.timesteps):
                            x0 = sgf_rollout.h3_student_forward(self.model, self.inputs, current,
                                timestep, chunk_index=index, cache=self.cache)
                            if not bool(torch.isfinite(x0).all()):
                                raise FloatingPointError("nonfinite denoise")
                            if ordinal == 3:
                                final = x0.detach().clone()
                            else:
                                current = scale_noise(x0, self.random_like(x0),
                                                      self.schedule.timesteps[ordinal + 1])
                        sgf_rollout.h3_student_forward(self.model, self.inputs, final,
                            current.new_tensor(1.), chunk_index=index, cache=self.cache, commit_cache=True)
                else:
                    with self.steady_optim.installed() if self.steady_optim else nullcontext():
                        final = self.full_resident.step(engine=self, compiler=self.compiler, noisy=current)
            torch.cuda.synchronize(self.device)
            self.index += 1
            elapsed = (time.perf_counter() - start) * 1000
            self.session_times.append(elapsed)
            # Keep the normalized native latent on its device; VAE follows DiT
            # serially on the same eight devices, not through a CPU round trip.
            return final, elapsed
        except BaseException:
            self.failed = True
            raise

    def stop(self):
        import torch
        from ..acceleration.attention.scope import hook_snapshot
        if self.scopes is None:
            raise RuntimeError("no active scene")
        self.full_resident.end_session(self)
        torch.cuda.synchronize(self.device)
        report = dict(chunks=self.index, dit_ms=list(self.session_times),
            scene_id=self.condition["scene_id"], compiler_calls=dict(
                (str(k), v) for k, v in self.compiler.backend_calls.items()),
            graphs=self.full_resident.report(), acceleration=self.deployment["acceleration"])
        if self.prior_forward_present:
            self.base.forward = self.prior_forward
        else:
            del self.base.forward
        self.cache.clear()
        try:
            self.owners.validate()
        finally:
            self.scopes.close()
            self.scopes = self.inputs = self.noise = self.rng = self.owners = None
        if hook_snapshot(self.base) != self.before_hooks:
            self.failed = True
            raise RuntimeError("model hooks did not restore")
        return report

    def close(self):
        # Cleanup always attempts Graph release, even if a failed session
        # cannot safely be parked. The original error remains the caller's.
        try:
            if self.scopes is not None:
                if self.prior_forward_present:
                    self.base.forward = self.prior_forward
                elif "forward" in self.base.__dict__:
                    del self.base.forward
                self.scopes.close()
                self.scopes = None
        finally:
            self.full_resident.close()

    def recover_rejected_scene(self, error):
        import torch
        from ..acceleration.compile.guard import SceneRejected
        if not isinstance(error, SceneRejected) or not error.pre_call_collective_agreement:
            raise RuntimeError("only coordinated pre-call rejection is recoverable")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot recover interrupted CUDA capture")
        torch.cuda.synchronize(self.device)
        resident = self.full_resident
        if self.scopes is None:
            if resident.active is not None:
                raise RuntimeError("inconsistent inactive scene")
            return
        active = resident.active
        if active is None or active["engine"] is not self or active["slot"]["pool"].active:
            raise RuntimeError("not at a recoverable call boundary")
        slot = active["slot"]
        if (slot["denoise"] is None) != (slot["refresh"] is None):
            raise RuntimeError("partially captured scene cannot be reused")
        resident.failed = False
        try:
            report = self.stop()
        except BaseException:
            resident.failed = True
            raise
        self.failed = False
        self.dynamic_evidence.emit("scene_rejected_and_parked", stage=error.stage)
        return report
