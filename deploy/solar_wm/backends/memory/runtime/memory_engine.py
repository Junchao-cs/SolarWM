"""Independent native Memory web engine; no legacy DiT acceleration claims.

The released Memory forward and checkpoint loader are used unchanged. VAE and
media live outside this class. Absolute cameras are retained for retrieval and
PRoPE; the old recent-window-only camera rebase MUST NOT run on this path.
"""
from contextlib import ExitStack
from pathlib import Path
import json
import time
import numpy as np

from deploy.solar_wm.backends.memory.runtime.engine import Engine as OriginalEngine
from deploy.solar_wm.shared.runtime.navigation import NavigationCamera


def noise_window(index):
    if type(index) is not int or index < 0:
        raise ValueError("nonnegative chunk index required")
    offset = index if index < 58 else (index - 58) % 10
    return offset, index >= 58 and offset == 0


def check_selection(record):
    j = record['chunk']
    ids = record['memory_latent_indices']
    stop = max(0, j - 5) * 5
    if len(ids) != min(5, stop) or len(set(ids)) != len(ids):
        raise RuntimeError('retrieval count or uniqueness differs')
    if any(type(i) is not int or not 0 <= i < stop for i in ids):
        raise RuntimeError('retrieved recent, future, or invalid latent')
    if record['cached_chunks_per_layer'] > 5:
        raise RuntimeError('recent KV exceeded W5')


class MemoryEngine(OriginalEngine):
    def __init__(self, deployment, *, rank, device, run, compiler_cache, control_group):
        import torch
        import torch.distributed as dist
        from solar.backends.minimax_h3.optional import load_transformer
        from solar.backends.minimax_h3.runtime import _base_model_load_receipt
        from solar.backends.minimax_h3.lora import inject_h3_lora
        from solar.backends.minimax_h3.weights import load_initial_weights
        from solar.backends.minimax_h3.memory_inference import attach_memory, validate_memory_inference
        from solar.backends.minimax_h3.artifacts import load_silence_latents
        from solar.backends.minimax_h3.sgf import h3_sgf_schedule
        from solar.backends.minimax_h3.sgf_attention import H3RawKVCache
        if deployment['acceleration'] != 'memory_native':
            raise ValueError('unvalidated legacy DiT acceleration is forbidden')
        self.deployment, self.rank, self.device, self.run = deployment, rank, device, Path(run)
        self.compiler_cache, self.request_group, self.request_dist = compiler_cache, control_group, dist
        self.cfg = json.loads(Path(deployment['model_config']).read_text())
        validate_memory_inference(self.cfg)
        if not self.cfg['inference']['memory']['enabled'] or self.cfg['distributed']['sequence_parallel_size'] != 8:
            raise ValueError('Memory-enabled SP8 configuration required')
        torch.set_num_threads(2)
        torch.manual_seed(42)
        modules = load_transformer(self.cfg['model'], device=device)
        receipt = _base_model_load_receipt(self.cfg['model'], modules.transformer)
        self.model, self.lora = inject_h3_lora(modules.transformer, self.cfg['model']['adapter'], base_identity=receipt)
        checkpoint = self.cfg['checkpoint']
        self.weights = load_initial_weights(dict(path=checkpoint['resume_from'], stage='stage2', weight_source=checkpoint['weight_source']), self.lora)
        if self.weights != 'stage2:ema:step=3900':
            raise RuntimeError('FIX3900 EMA parent required')
        self.model, self.memory_identity = attach_memory(self.model, self.lora, self.cfg['inference']['memory'])
        self.base = self.model.get_base_model()
        self.model.eval().requires_grad_(False)
        if len(self.base.transformer_blocks) != 50:
            raise RuntimeError('full 50-layer model required')
        silence, _ = load_silence_latents(Path(deployment['silence']), pixel_frames=158)
        self.silence = silence.to(device).float().unsqueeze(0).permute(0, 1, 3, 2).reshape(1, -1, 32).contiguous()
        self.schedule, _ = h3_sgf_schedule(device=device)
        self.cache = H3RawKVCache()
        self.inputs = self.noise = self.rng = self.scopes = self.owners = None
        self.index = self.session_count = 0
        self.poses, self.session_times = [], []
        self.layout_key = self.condition = None
        self.failed = False
        self.memory_forward = None
        with (self.run / f'memory-weights-rank{rank}.json').open('x') as f:
            json.dump(self.memory_identity, f, indent=2)
        torch.cuda.synchronize(device)

    def _enter_graph_scope(self):
        # Session activity marker, intentionally no old attention/Graph hooks.
        self.scopes = ExitStack()

    def start(self, *, seed=42):
        import torch
        from solar.backends.minimax_h3.camera import wan_fixed_intrinsics_like
        from solar.backends.minimax_h3.memory_stage2_rollout import Stage2MemoryForward
        if self.failed or self.condition is None or self.scopes is not None:
            raise RuntimeError('inactive healthy scene required')
        self.session_count += 1
        self.session_times = []
        self.session_started = time.perf_counter()
        self._start_inputs(seed=seed)
        fixed = wan_fixed_intrinsics_like(self.inputs.camera_K[0].cpu().numpy()).copy()
        self.inputs.camera_K.copy_(torch.from_numpy(fixed).to(self.device))
        self.camera = NavigationCamera(self.camera)
        self.navigation_speeds = (1., 1.)
        self.memory_forward = Stage2MemoryForward(np.empty((0, 4, 4)), fixed[1:])

    def advance_camera(self, actions):
        import torch
        from solar.backends.minimax_h3.torch_prope import _invert_se3
        expected = 4 if self.index == 0 else 5
        if self.scopes is None or len(actions) != expected:
            raise ValueError('camera packet differs from active chunk')
        self.camera.prepare(actions, self.index, *self.navigation_speeds)
        if self.index == 0:
            self.poses.append(self.camera.c2w.copy())
        for keys in actions:
            keys = [key for key in keys if key not in 'qe']
            previous = len(self.poses) - 1
            for _ in range((1, 4, 4, 4, 4)[previous % 5]):
                self.camera.step(keys)
            self.poses.append(self.camera.c2w.copy())
        if len(self.poses) != (self.index + 1) * 5 or self.camera.vertical:
            raise RuntimeError('camera sampling/order changed')
        needed = len(self.poses) + 1
        views, intrinsics = self.inputs.camera_viewmats, self.inputs.camera_K
        if needed > views.shape[1]:
            capacity = max(needed, views.shape[1] * 2)
            extended = torch.eye(4, device=self.device).view(1, 1, 4, 4).repeat(1, capacity, 1, 1)
            extended[:, :views.shape[1]].copy_(views)
            new_K = intrinsics[:, -1:].repeat(1, capacity, 1, 1)
            new_K[:, :intrinsics.shape[1]].copy_(intrinsics)
            self.inputs.camera_viewmats, self.inputs.camera_K = extended, new_K
        start = self.index * 5
        value = torch.as_tensor(np.asarray(self.poses[start:]), device=self.device, dtype=torch.float32)
        self.inputs.camera_viewmats[:, start + 1:needed].copy_(_invert_se3(value).unsqueeze(0))
        if self.index == 0:
            self.inputs.camera_viewmats[:, :1].copy_(self.inputs.camera_viewmats[:, 1:2])
        # Match released inference: retrieval uses FP64 inverse of actual FP32
        # view matrices, not a different higher precision camera trajectory.
        view_cpu = self.inputs.camera_viewmats[0, 1:needed].cpu().numpy().astype(np.float64)
        self.memory_forward.c2w = np.linalg.inv(view_cpu)
        self.memory_forward.K = self.inputs.camera_K[0, 1:needed].cpu().numpy()

    def generate(self):
        import torch
        from solar.backends.minimax_h3.torch_flow import scale_noise
        if self.failed or self.scopes is None:
            raise RuntimeError('healthy active memory session required')
        index = self.index
        offset, refill = noise_window(index)
        if refill:
            self.noise = self.random_like(torch.empty(1, 24, 50, 48, 84, device=self.device))
        current = self.noise[:, :, offset * 5:(offset + 1) * 5]
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        try:
            with torch.no_grad():
                for ordinal, timestep in enumerate(self.schedule.timesteps):
                    x0 = self.memory_forward(self.model, self.inputs, current, timestep, chunk_index=index, cache=self.cache)
                    if not bool(torch.isfinite(x0).all()):
                        raise FloatingPointError('nonfinite memory denoise')
                    if ordinal == 3:
                        final = x0.detach().clone()
                    else:
                        current = scale_noise(x0, self.random_like(x0), self.schedule.timesteps[ordinal + 1])
                self.memory_forward(self.model, self.inputs, final, current.new_tensor(1.), chunk_index=index, cache=self.cache, commit_cache=True)
            torch.cuda.synchronize(self.device)
            elapsed = (time.perf_counter() - started) * 1000
            record = self.memory_forward.records[-1]
            check_selection(record)
            self.index += 1
            self.session_times.append(elapsed)
            if self.rank == 0:
                print('MEMORY_WEB_CHUNK ' + json.dumps(dict(session=self.session_count, dit_ms=elapsed, **record)), flush=True)
            return final, elapsed
        except BaseException:
            self.failed = True
            raise

    def stop(self):
        import torch
        if self.scopes is None:
            raise RuntimeError('no active memory session')
        torch.cuda.synchronize(self.device)
        report = dict(chunks=self.index, dit_ms=self.session_times, scene_id=self.condition['scene_id'],
                      acceleration='memory_native', dit_compile=False, dit_graph=False,
                      memory=self.memory_identity, retrieval=self.memory_forward.records,
                      camera_policy='released absolute first-camera coordinates; no local rebase')
        with (self.run / f'memory-session-{self.session_count}-rank{self.rank}.json').open('x') as f:
            json.dump(report, f)
        self.cache.clear()
        self.scopes.close()
        self.scopes = self.inputs = self.noise = self.rng = self.memory_forward = None
        self.poses = []
        return report

    def close(self):
        if self.scopes is not None:
            self.scopes.close()
        self.cache.clear()
        self.scopes = self.inputs = self.noise = self.rng = self.memory_forward = None
        self.poses = []
