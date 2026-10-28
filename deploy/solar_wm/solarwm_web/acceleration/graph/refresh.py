"""Request-owned full-H3 refresh CUDA Graph, separate from denoise Graph.

This module owns only the fixed chunk-5 refresh forward and its captured K/V
writes. The enclosing rollout owns real chunk order, native cache publication,
all-rank admission, lifecycle, and comparison against a native request."""

from __future__ import annotations
import torch
from .writeback import RefreshWriteback
from .inputs import FullStudentTensorInputs, _check_sources, _scalar
from .denoise import _read_token
from .buffers import StableSteadyInputs


class RefreshStudentCall:
    """Full 50-layer student with fixed Graph-visible chunk and K/V slots."""

    def __init__(self, *, student, inputs, history_owner, global_cache, noisy, timestep):
        _check_sources()
        from solar.backends.minimax_h3.layout import unpatchify_video
        from solar.backends.minimax_h3.sgf_attention import H3SGFAttention
        from solar.backends.minimax_h3.torch_flow import predict_clean_sample

        if type(history_owner) is not StableSteadyInputs or not history_owner.ready:
            raise TypeError("exact ready full-H3 history owner required")
        if history_owner.next_chunk != 6:
            raise ValueError("refresh Graph must start with real chunk 5")
        self.tensor_owner = FullStudentTensorInputs(
            inputs=inputs, noisy=noisy, timestep=timestep,
        )
        if (self.tensor_owner.hidden_states.device.type != "cuda"
                or inputs.sp_enabled is not True):
            raise ValueError("refresh Graph requires actual official SP8 CUDA inputs")
        self.student = student
        self.history_owner = history_owner
        self.writeback = RefreshWriteback(history_owner, global_cache, actual_chunk=5)
        self.noisy = noisy.detach().clone()
        self.step = timestep.detach().clone()
        self.actual_chunk = 5
        self.control = H3SGFAttention(
            self.tensor_owner.layout, history_owner.viewmats, history_owner.intrinsics,
            "rollout", chunk_index=5, cache=self.writeback, commit_cache=True,
        )
        self._unpatchify_video = unpatchify_video
        self._predict_clean_sample = predict_clean_sample
        self.final_capture = False
        self.ready = True
        self.poisoned = False

    def load(self, *, actual_chunk, noisy, timestep):
        if self.poisoned or not self.ready or self.writeback.failed:
            raise RuntimeError("refresh Graph call failed closed")
        self.ready = False
        try:
            if (type(actual_chunk) is not int or actual_chunk != self.actual_chunk + 1
                    or actual_chunk != self.writeback.next_chunk
                    or self.history_owner.next_chunk != actual_chunk + 1):
                raise ValueError("refresh Graph chunk order/history changed")
            self.tensor_owner._check_noisy(noisy)
            _scalar(timestep, self.noisy.device)
            if (noisy.shape != self.noisy.shape or noisy.dtype != self.noisy.dtype
                    or noisy.device != self.noisy.device
                    or timestep.shape != self.step.shape
                    or timestep.dtype != self.step.dtype
                    or timestep.device != self.step.device):
                raise ValueError("refresh Graph input specialization changed")
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
            raise RuntimeError("refresh Graph student call is not ready")
        self.writeback.begin_capture_pass()
        try:
            prediction, _ = self.student(
                **self.tensor_owner.model_tensor_kwargs(),
                fused_prope=True,
                packed_sequence_parallel=True,
                sgf_attention=self.control,
                return_dict=False,
                **self.tensor_owner.layout.transformer_kwargs(),
            )
            velocity = self._unpatchify_video(
                prediction[:, self.tensor_owner.layout.noisy_video_output_slice],
                5, 48, 84,
            )
            result = self._predict_clean_sample(self.noisy, velocity, self.step)
            self.writeback.end_capture_pass(final=self.final_capture)
            return result
        except BaseException:
            self.poisoned = True
            self.ready = False
            self.writeback.failed = True
            raise


class ManualRefreshGraph:
    """One full-model capture; no eager fallback after capture failure."""

    def __init__(self, call: RefreshStudentCall):
        if type(call) is not RefreshStudentCall or not call.ready or call.poisoned:
            raise TypeError("exact ready refresh call required")
        self.call = call
        self.graph = None
        self.stream = None
        self.static_output = None
        self.capture_count = 0
        self.replay_count = 0
        self.prime_count = 0
        self.barrier_count = 0
        self.failed = False
        self.reference_max_abs = None
        self.kv_reference_max_abs = None

    def _barrier(self):
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() != 8:
            raise RuntimeError("refresh Graph needs initialized official SP8")
        torch.cuda.synchronize(self.call.noisy.device)
        dist.barrier()
        torch.cuda.synchronize(self.call.noisy.device)
        self.barrier_count += 1

    def _check(self):
        if (self.failed or self.call.poisoned or not self.call.ready
                or self.call.student.training or torch.is_autocast_enabled("cuda")):
            raise RuntimeError("refresh Graph must use ready eval student and native autocast")

    def _replay_once(self):
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        caller.wait_stream(self.stream)
        self.replay_count += 1
        return self.static_output.clone()

    def capture(self):
        self._check()
        if self.graph is not None or self.capture_count or self.call.actual_chunk != 5:
            raise RuntimeError("refresh Graph can capture only once at chunk 5")
        before = _read_token(self.call)
        try:
            reference = self.call().detach().clone()
            reference_kv = {
                layer: tuple(t.detach().clone() for t in self.call.writeback.slots[layer])
                for layer in range(50)
            }
            torch.cuda.synchronize(self.call.noisy.device)
            for _ in range(3):
                self.call()
                self.prime_count += 1
            self._barrier()
            self.stream = torch.cuda.Stream(device=self.call.noisy.device)
            caller = torch.cuda.current_stream()
            self.stream.wait_stream(caller)
            with torch.cuda.stream(self.stream):
                for _ in range(3):
                    self.call()
                    self.prime_count += 1
            caller.wait_stream(self.stream)
            torch.cuda.synchronize(self.call.noisy.device)
            if _read_token(self.call) != before:
                raise RuntimeError("refresh Graph priming mutated model inputs")
            self._barrier()
            self.static_output = torch.empty_like(reference)
            self.graph = torch.cuda.CUDAGraph()
            self.call.final_capture = True
            with torch.autocast("cuda", enabled=False):
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.static_output.copy_(self.call())
            self.capture_count = 1
            self._barrier()
            result = self._replay_once()
            if _read_token(self.call) != before:
                raise RuntimeError("refresh Graph capture/replay mutated model inputs")
            if not bool(torch.isfinite(result).all()):
                raise RuntimeError("refresh Graph output is nonfinite")
            self.reference_max_abs = (reference.float() - result.float()).abs().max().item()
            self.kv_reference_max_abs = max(
                (reference_kv[layer][side].float()
                 - self.call.writeback.slots[layer][side].float()).abs().max().item()
                for layer in range(50) for side in range(2)
            )
            self.call.writeback.mark_replayed(5)
            return result
        except BaseException:
            self.failed = True
            self.call.poisoned = True
            self.graph = None
            self.static_output = None
            raise

    def replay(self):
        self._check()
        if self.graph is None or self.capture_count != 1 or self.call.actual_chunk < 6:
            raise RuntimeError("refresh Graph replay before capture or copy-in")
        before = _read_token(self.call)
        try:
            result = self._replay_once()
            if _read_token(self.call) != before:
                raise RuntimeError("refresh Graph replay mutated model inputs")
            self.call.writeback.mark_replayed(self.call.actual_chunk)
            return result
        except BaseException:
            self.failed = True
            self.call.poisoned = True
            raise

    def release(self):
        had_graph = self.graph is not None
        if had_graph:
            torch.cuda.synchronize(self.call.noisy.device)
        self.graph = None
        self.static_output = None
        self.stream = None
        self.call = None
        return {"had_graph": had_graph, "device_synchronized": had_graph,
                "references_cleared": all(getattr(self, name) is None
                                          for name in ("graph", "static_output", "stream", "call"))}
