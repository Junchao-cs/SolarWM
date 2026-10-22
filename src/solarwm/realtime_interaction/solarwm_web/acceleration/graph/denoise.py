"""Full-student CUDA Graph capture with native disabled-autocast semantics."""

from __future__ import annotations
import torch
from .inputs import FullStudentReadOnlyCall


def _read_token(call):
    """Best-effort Python mutation guard, not proof against raw CUDA writes."""
    tensors = [
        call.tensor_owner.hidden_states,
        call.tensor_owner.timestep,
        call.tensor_owner.timestep_indices,
        call.tensor_owner.anchor_rows,
        call.tensor_owner.audio_rows,
        call.tensor_owner.prompt,
        call.tensor_owner.audio_timestep,
        call.history_owner.viewmats,
        call.history_owner.intrinsics,
        call.noisy,
        call.step,
    ]
    for layer in range(50):
        tensors.extend(call.history_owner.buffers[layer])
    return tuple((id(t), t.data_ptr(), t._version, tuple(t.shape), t.dtype)
                 for t in tensors)


class NativeAutocastManualFullStudentGraph:
    """Capture exactly the read-only full H3 student once, then replay.

    No eager fallback, recapture, pool reset, or implicit rank synchronization
    after failure. A failed instance cannot be reused in the same session.
    """


    def __init__(self, call: FullStudentReadOnlyCall):
        if type(call) is not FullStudentReadOnlyCall or not call.ready or call.poisoned:
            raise TypeError("exact ready H3 full-student call required")
        if call.tensor_owner.hidden_states.device.type != "cuda":
            raise ValueError("manual H3 Graph requires actual CUDA tensors")
        if call.tensor_owner.inputs.sp_enabled is not True:
            raise ValueError("manual H3 Graph requires official SP8")
        self.call = call
        self.graph = None
        self.stream = None
        self.static_output = None
        self.failed = False
        self.capture_count = 0
        self.caller_prime_count = 0
        self.side_prime_count = 0
        self.replay_count = 0
        self.barrier_count = 0
        self.input_fence_count = 0
        self.output_fence_count = 0
        self.capture_reference_max_abs = None
        self.capture_reference_mean_abs = None


    def _barrier(self):
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() != 8:
            raise RuntimeError("H3 manual Graph needs initialized official SP8")
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.synchronize()
        self.barrier_count += 1


    def _check_call(self):
        if self.failed or self.call.poisoned or not self.call.ready:
            raise RuntimeError("H3 Graph/call failed closed or not ready")
        if (not self.call.tensor_owner.ready or not self.call.history_owner.ready
                or self.call.history_owner.next_chunk != self.call.actual_chunk + 1):
            raise RuntimeError("H3 Graph inputs were not copied for current chunk")
        if self.call.student.training:
            raise RuntimeError("H3 Graph requires eval student")


    @staticmethod
    def _autocast_policy() -> None:
        if torch.is_autocast_enabled("cuda"):
            raise RuntimeError(
                "H3 Graph requires the original disabled CUDA autocast state"
            )


    def capture(self):
        if self.graph is not None or self.capture_count:
            raise RuntimeError("H3 full-student Graph may be captured only once")
        self._check_call()
        if self.call.actual_chunk != 5:
            raise RuntimeError("H3 Graph capture must start at real chunk 5")
        self._autocast_policy()
        before = _read_token(self.call)
        try:
            reference = self.call().detach().clone()
            torch.cuda.synchronize()
            for _ in range(3):
                self.call()
                self.caller_prime_count += 1
            torch.cuda.synchronize()
            if _read_token(self.call) != before:
                raise RuntimeError("H3 eager prime mutated read-only inputs")
            self._barrier()
            caller = torch.cuda.current_stream()
            self.stream = torch.cuda.Stream(device=reference.device)
            self.stream.wait_stream(caller)
            with torch.cuda.stream(self.stream):
                with torch.autocast("cuda", enabled=False):
                    for _ in range(3):
                        self.call()
                        self.side_prime_count += 1
            caller.wait_stream(self.stream)
            torch.cuda.synchronize()
            if _read_token(self.call) != before:
                raise RuntimeError("H3 side prime mutated read-only inputs")
            self._barrier()
            self.static_output = torch.empty_like(reference)
            self.graph = torch.cuda.CUDAGraph()
            with torch.autocast("cuda", enabled=False):
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.static_output.copy_(self.call())
            self.capture_count = 1
            self._barrier()
            replayed = self._replay_once()
            if _read_token(self.call) != before:
                raise RuntimeError("H3 capture/replay mutated read-only inputs")
            if not torch.isfinite(replayed).all().item():
                raise RuntimeError("H3 captured output is nonfinite")
            difference = (reference.float() - replayed.float()).abs()
            self.capture_reference_max_abs = difference.max().item()
            self.capture_reference_mean_abs = difference.mean().item()
            return replayed
        except BaseException:
            self.failed = True
            self.graph = None
            self.static_output = None
            raise


    def _replay_once(self):
        if self.graph is None or self.stream is None or self.static_output is None:
            raise RuntimeError("H3 Graph replay before capture")
        caller = torch.cuda.current_stream()
        self.stream.wait_stream(caller)
        self.input_fence_count += 1
        with torch.cuda.stream(self.stream):
            self.graph.replay()
        caller.wait_stream(self.stream)
        self.output_fence_count += 1
        self.replay_count += 1
        return self.static_output.clone()


    def replay(self):
        self._check_call()
        if self.graph is None or self.capture_count != 1:
            raise RuntimeError("H3 Graph replay requires one successful capture")
        if self.call.actual_chunk < 5:
            raise RuntimeError("H3 Graph chunk left steady window")
        before = _read_token(self.call)
        try:
            output = self._replay_once()
            if _read_token(self.call) != before:
                raise RuntimeError("H3 replay mutated read-only inputs")
            return output
        except BaseException:
            self.failed = True
            raise


    def summary(self):
        return {
            "scope": "full_50_layer_H3_student_read_only_denoise",
            "backend": "manual_cuda_graph",
            "capture_count": self.capture_count,
            "caller_prime_count": self.caller_prime_count,
            "side_prime_count": self.side_prime_count,
            "replay_count": self.replay_count,
            "barrier_count": self.barrier_count,
            "input_fence_count": self.input_fence_count,
            "output_fence_count": self.output_fence_count,
            "capture_reference_max_abs_diagnostic": self.capture_reference_max_abs,
            "capture_reference_mean_abs_diagnostic": self.capture_reference_mean_abs,
            "failed": self.failed,
            "stage_passed": False,
        }
