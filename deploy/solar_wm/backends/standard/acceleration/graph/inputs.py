"""Pointer-stable full-student H3 denoise inputs for a future Graph probe.

This is the actual five-latent, 50-layer student call boundary. It does not
capture a CUDA Graph, install a route, or authorize the Graph stage. Callers
must keep refresh/commit, RNG, and finite-output checks outside capture."""

from __future__ import annotations
from ..compatibility import expected_digest
import hashlib
from pathlib import Path
import torch
from .buffers import StableSteadyInputs


SOURCE_ROOT = Path(__import__('solar.backends.minimax_h3.layout', fromlist=['__file__']).__file__).parent
SOURCE_SHA256 = {'layout.py': expected_digest("layout"), 'torch_layout.py': expected_digest("torch_layout"), 'sgf_rollout.py': expected_digest("sgf_rollout"), 'sgf_attention.py': expected_digest("sgf_attention"), 'torch_flow.py': expected_digest("torch_flow")}


def _check_sources():
    for name, expected in SOURCE_SHA256.items():
        if hashlib.sha256((SOURCE_ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"frozen H3 {name} source changed")


def _scalar(value, device):
    if not isinstance(value, torch.Tensor) or value.device != device:
        raise ValueError("H3 timestep must be an actual tensor on the input device")
    if value.numel() != 1 or value.requires_grad:
        raise ValueError("H3 timestep must be a detached scalar tensor")
    return value


class FullStudentTensorInputs:
    """Own stable model tensors while rebuilding native values outside capture."""

    def __init__(self, *, inputs, noisy, timestep):
        _check_sources()
        from solar.backends.minimax_h3.layout import patchify_video
        from solar.backends.minimax_h3.sgf_rollout import H3SGFInputs
        from solar.backends.minimax_h3.torch_layout import build_row_timesteps

        if type(inputs) is not H3SGFInputs:
            raise TypeError("exact H3SGFInputs object required")
        self._check_noisy(noisy)
        _scalar(timestep, noisy.device)
        if inputs.sp_enabled is not (noisy.device.type == "cuda"):
            raise ValueError("official SP8 requires CUDA; CPU fixtures disable SP")
        self.inputs = inputs
        self.layout = inputs.layout("rollout")
        if (self.layout.stage != "stage0p5" or self.layout.latent_height != 48
                or self.layout.latent_width != 84
                or self.layout.rows_per_video_frame != 1008
                or int(self.layout.noisy_video_indices.numel()) != 5040):
            raise ValueError("full H3 rollout layout differs")
        for name in ("anchor_rows", "audio_rows", "prompt", "audio_timestep"):
            value = getattr(inputs, name)
            if (not isinstance(value, torch.Tensor) or value.device != noisy.device
                    or value.requires_grad):
                raise ValueError(f"H3 {name} device or ownership differs")
        if (inputs.anchor_rows.ndim != 3 or inputs.anchor_rows.shape[:2] != (1, 1008)
                or inputs.audio_rows.ndim != 3 or inputs.audio_rows.shape[0] != 1
                or inputs.prompt.ndim != 3 or inputs.prompt.shape[0] != 1
                or inputs.audio_timestep.numel() != 1):
            raise ValueError("full H3 condition shape differs")
        self.anchor_rows = inputs.anchor_rows.detach().clone()
        self.audio_rows = inputs.audio_rows.detach().clone()
        self.prompt = inputs.prompt.detach().clone()
        self.audio_timestep = inputs.audio_timestep.detach().clone()
        self.keyframe_timestep = float(inputs.keyframe_timestep)
        self._patchify_video = patchify_video
        self._build_row_timesteps = build_row_timesteps
        hidden, times, indices = self._prepare(noisy, timestep)
        self.hidden_states = torch.empty_like(hidden)
        self.timestep = torch.empty_like(times)
        self.timestep_indices = torch.empty_like(indices)
        self.ready = False
        self.poisoned = False
        self.load(noisy, timestep)

    @staticmethod
    def _check_noisy(noisy):
        if (not isinstance(noisy, torch.Tensor) or noisy.requires_grad
                or tuple(noisy.shape) != (1, 24, 5, 48, 84)):
            raise ValueError("full H3 five-latent noisy tensor differs")

    def _prepare(self, noisy, timestep):
        hidden = torch.cat((self.anchor_rows, self._patchify_video(noisy)), dim=1)
        times, indices = self._build_row_timesteps(
            self.layout, timestep, self.audio_timestep,
            text_timestep=1.0,
            condition_video_timestep=self.keyframe_timestep,
            device=noisy.device,
        )
        return hidden, times, indices

    def load(self, noisy, timestep):
        if self.poisoned:
            raise ValueError("poisoned H3 tensor owner cannot be reused")
        self.ready = False
        try:
            self._check_noisy(noisy)
            _scalar(timestep, noisy.device)
            if noisy.device != self.hidden_states.device:
                raise ValueError("H3 input device changed")
            hidden, times, indices = self._prepare(noisy, timestep)
            for source, target in ((hidden, self.hidden_states),
                                   (times, self.timestep),
                                   (indices, self.timestep_indices)):
                if (source.shape != target.shape or source.dtype != target.dtype
                        or source.device != target.device):
                    raise ValueError("H3 Graph input specialization changed")
            with torch.no_grad():
                self.hidden_states.copy_(hidden)
                self.timestep.copy_(times)
                self.timestep_indices.copy_(indices)
        except BaseException:
            self.poisoned = True
            raise
        self.ready = True

    def model_tensor_kwargs(self):
        if not self.ready or self.poisoned:
            raise ValueError("H3 model tensors are not ready")
        return {
            "hidden_states": self.hidden_states,
            "audio_hidden_states": self.audio_rows,
            "encoder_hidden_states": self.prompt,
            "timestep": self.timestep,
            "timestep_indices": self.timestep_indices,
        }


class FullStudentReadOnlyCall:
    """Keep native H3 control fixed at chunk 5 while copy-in advances onward."""

    def __init__(self, *, student, tensor_owner: FullStudentTensorInputs,
                 history_owner: StableSteadyInputs, noisy, timestep):
        _check_sources()
        from solar.backends.minimax_h3.layout import unpatchify_video
        from solar.backends.minimax_h3.sgf_attention import H3SGFAttention
        from solar.backends.minimax_h3.torch_flow import predict_clean_sample

        if type(tensor_owner) is not FullStudentTensorInputs:
            raise TypeError("exact full-student tensor owner required")
        if type(history_owner) is not StableSteadyInputs:
            raise TypeError("exact full 50-layer H3 history owner required")
        if (not tensor_owner.ready or tensor_owner.poisoned
                or not history_owner.ready or history_owner.poisoned
                or history_owner.next_chunk != 6):
            raise ValueError("H3 chunk 5 inputs are not ready")
        tensor_owner._check_noisy(noisy)
        _scalar(timestep, noisy.device)
        if (tensor_owner.hidden_states.device != noisy.device
                or history_owner.viewmats.device != noisy.device
                or history_owner.intrinsics.device != noisy.device):
            raise ValueError("H3 Graph owners are on different devices")
        self.student = student
        self.tensor_owner = tensor_owner
        self.history_owner = history_owner
        self.noisy = noisy.detach().clone()
        self.step = timestep.detach().clone()
        self.actual_chunk = 5
        self.control = H3SGFAttention(
            tensor_owner.layout, history_owner.viewmats, history_owner.intrinsics,
            "rollout", chunk_index=5, cache=history_owner, commit_cache=False,
        )
        self._unpatchify_video = unpatchify_video
        self._predict_clean_sample = predict_clean_sample
        self.ready = True
        self.poisoned = False

    def load(self, *, actual_chunk, noisy, timestep):
        if self.poisoned:
            raise ValueError("poisoned H3 call cannot be reused")
        self.ready = False
        try:
            if (type(actual_chunk) is not int or actual_chunk not in
                    (self.actual_chunk, self.actual_chunk + 1)
                    or actual_chunk < 5):
                raise ValueError("H3 denoise chunk skipped or replayed")
            if (not self.history_owner.ready or self.history_owner.poisoned
                    or self.history_owner.next_chunk != actual_chunk + 1):
                raise ValueError("current H3 history was not copied before denoise")
            self.tensor_owner._check_noisy(noisy)
            _scalar(timestep, noisy.device)
            if (noisy.shape != self.noisy.shape or noisy.dtype != self.noisy.dtype
                    or noisy.device != self.noisy.device or timestep.shape != self.step.shape
                    or timestep.dtype != self.step.dtype or timestep.device != self.step.device):
                raise ValueError("H3 Graph noisy/timestep specialization changed")
            self.tensor_owner.load(noisy, timestep)
            with torch.no_grad():
                self.noisy.copy_(noisy)
                self.step.copy_(timestep)
            self.actual_chunk = actual_chunk
        except BaseException:
            self.poisoned = True
            raise
        self.ready = True

    def __call__(self):
        if (not self.ready or self.poisoned or not self.tensor_owner.ready
                or not self.history_owner.ready
                or self.history_owner.next_chunk != self.actual_chunk + 1):
            raise ValueError("H3 read-only denoise call is not ready")
        try:
            prediction, _ = self.student(
                **self.tensor_owner.model_tensor_kwargs(),
                fused_prope=True,
                packed_sequence_parallel=self.tensor_owner.inputs.sp_enabled,
                sgf_attention=self.control,
                return_dict=False,
                **self.tensor_owner.layout.transformer_kwargs(),
            )
            velocity = self._unpatchify_video(
                prediction[:, self.tensor_owner.layout.noisy_video_output_slice],
                5, 48, 84,
            )
            return self._predict_clean_sample(self.noisy, velocity, self.step)
        except BaseException:
            self.poisoned = True
            self.ready = False
            raise
