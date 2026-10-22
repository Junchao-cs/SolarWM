"""Native input/RNG/camera construction, mechanically extracted without padding."""
import hashlib
from pathlib import Path
import numpy as np
from .configuration import file_sha256 as sha

class NativeSceneMath:
    def random_like(self, value):
        import torch
        from solarwm.backends.minimax_h3.distributed import broadcast_sp_tensor
        noise = torch.randn(value.shape, device=self.device, dtype=torch.float32, generator=self.rng)
        broadcast_sp_tensor(noise)
        return noise

    def _advance_initial_camera(self, actions: list[list[str]]) -> None:
        import torch
        from solarwm.backends.minimax_h3.torch_prope import _invert_se3
        if self.inputs is None:
            raise RuntimeError('interactive H3 request inactive or complete')
        expected = 4 if self.index == 0 else 5
        if type(actions) is not list or len(actions) != expected:
            raise ValueError('camera action count differs from H3 chunk contract')
        if self.index == 0:
            self.poses.append(self.camera.c2w.copy())
        for keys in actions:
            if type(keys) is not list or any((key not in 'wasdijkl' for key in keys)):
                raise ValueError('invalid H3 camera keys')
            previous = len(self.poses) - 1
            for _ in range((1, 4, 4, 4, 4)[previous % 5]):
                self.camera.step(keys)
            self.poses.append(self.camera.c2w.copy())
        if len(self.poses) != (self.index + 1) * 5:
            raise RuntimeError('H3 camera pose count differs')
        poses = torch.tensor(np.asarray(self.poses), device=self.device, dtype=torch.float32)
        views = _invert_se3(poses).unsqueeze(0)
        self.inputs.camera_viewmats[:, 0].copy_(views[:, 0])
        self.inputs.camera_viewmats[:, 1:len(self.poses) + 1].copy_(views)

    def _start_inputs(self, *, seed=42):
        import torch
        from safetensors.torch import load_file
        from solarwm.backends.minimax_h3.layout import patchify_video
        from solarwm.backends.minimax_h3.sgf_rollout import H3SGFInputs
        from solarwm.backends.minimax_h3.torch_flow import sample_shifted_timestep, scale_noise
        from solarwm.backends.minimax_h3.distributed import broadcast_sp_tensor
        from solarwm_web.runtime.camera import KeyboardCameraConfig, KeyboardCameraState
        if self.scopes is not None or self.inputs is not None:
            raise RuntimeError('prior session is still active')
        if sha(self.condition['path']) != self.condition['sha256']:
            raise ValueError('condition bytes changed')
        values = load_file(self.condition['path'], device='cpu')
        tags = values['text_token_tags']
        key = hashlib.sha256(tags.numpy().tobytes() + str(tuple(values['prompt_embeds'].shape)).encode()).hexdigest()
        self.layout_key = key
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        anchor = values['anchor_latents'].to(self.device).float()
        if anchor.ndim == 4:
            anchor = anchor.unsqueeze(0)
        anchor = scale_noise(anchor, self.random_like(anchor), 0.999)
        audio_t = sample_shifted_timestep(1, 3.0, self.device, torch.float32, generator=self.rng).reshape(())
        broadcast_sp_tensor(audio_t)
        audio = scale_noise(self.silence, self.random_like(self.silence), audio_t)
        prompt = values['prompt_embeds'].to(self.device).bfloat16()
        if prompt.ndim == 2:
            prompt = prompt.unsqueeze(0)
        views = torch.eye(4, device=self.device, dtype=torch.float32).view(1, 1, 4, 4).repeat(1, 291, 1, 1)
        source_K = values['K']
        if tuple(source_K.shape) != (290, 3, 3):
            raise ValueError('scene intrinsics shape differs')
        K = torch.cat((source_K[:1], source_K), dim=0).unsqueeze(0).to(self.device)
        self.inputs = H3SGFInputs(prompt, tags.to(self.device), patchify_video(anchor), audio, audio_t, views, K, 48, 84, 0.999, True)
        self.noise = self.random_like(torch.empty(1, 24, 290, 48, 84, device=self.device))
        self.cache.clear()
        self.index = 0
        self.poses = []
        self.camera = KeyboardCameraState(KeyboardCameraConfig(frames_per_second=16.0, move_speed_per_second=0.8, pitch_speed_degrees_per_second=15.0, yaw_speed_degrees_per_second=15.0, acceleration_seconds=0.25))
        self._enter_graph_scope()
        print('SOLARWM_SCENE_BOUND ' + __import__('json').dumps(dict(rank=self.rank, **self.condition, layout_key=key, prompt_shape=list(prompt.shape))), flush=True)

    def advance_camera(self, actions):
        self.camera.prepare(actions, self.index, *self.navigation_speeds)
        actions = [[key for key in keys if key not in 'qe'] for keys in actions]
        if self.index < 58:
            self._advance_initial_camera(actions)
        else:
            import numpy as np
            import torch
            from solarwm.backends.minimax_h3.torch_prope import _invert_se3
            if self.index == 58:
                if not torch.equal(self.inputs.camera_K[:, 1:], self.inputs.camera_K[:, -1:].expand_as(self.inputs.camera_K[:, 1:])):
                    raise RuntimeError('constant extension intrinsics required')
                self.pose_count = len(self.poses)
                self.poses = []
            previous = self.pose_count
            poses = []
            for keys in actions:
                for _ in range((1, 4, 4, 4, 4)[(self.pose_count - 1) % 5]):
                    self.camera.step(keys)
                poses.append(self.camera.c2w.copy())
                self.pose_count += 1
            if self.pose_count != (self.index + 1) * 5:
                raise RuntimeError('extended camera order differs')
            needed = self.pose_count + 1
            old_views, old_K = (self.inputs.camera_viewmats, self.inputs.camera_K)
            if needed > old_views.shape[1]:
                capacity = max(needed, old_views.shape[1] * 2)
                views = torch.eye(4, device=self.device).view(1, 1, 4, 4).repeat(1, capacity, 1, 1)
                views[:, :old_views.shape[1]].copy_(old_views)
                intrinsics = old_K[:, -1:].repeat(1, capacity, 1, 1)
                intrinsics[:, :old_K.shape[1]].copy_(old_K)
                self.inputs.camera_viewmats = views
                self.inputs.camera_K = intrinsics
            value = torch.tensor(np.asarray(poses), device=self.device, dtype=torch.float32)
            self.inputs.camera_viewmats[:, previous + 1:needed].copy_(_invert_se3(value).unsqueeze(0))
        if self.camera.vertical:
            raise RuntimeError('camera packet not fully consumed')
