"""Request-owned rotary tables, restored when the inference scope exits."""

from contextlib import contextmanager
from dataclasses import replace
import torch
import solar.backends.minimax_h3.sgf_attention as attention
import solar.backends.minimax_h3.sgf_rollout as rollout


_ACTIVE = None

def _tensor_stamp(tensor):
    try:
        version = tensor._version
    except RuntimeError as exc:
        raise ValueError("unversioned inference tensor not supported by this candidate") from exc
    return (id(tensor), tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()),
            str(tensor.dtype), str(tensor.device), version)

class RequestRotaries:
    def __init__(self, control, rope, native_prepare):
        if control.mode != "rollout":
            raise ValueError("request cache is inference rollout only")
        if rope.training:
            raise ValueError("request rotary cache requires eval RoPE")
        self.layout, self.rope = control.layout, rope
        self.native_prepare = native_prepare
        self.closed = False
        self.hits = 0
        self._before = self.signature()
        with torch.no_grad():
            prepared = native_prepare(control, rope)
        self.prefix = prepared.prefix_rotary
        self.windows = prepared.window_rotaries
        if self.signature() != self._before:
            raise RuntimeError("RoPE/layout state changed during table preparation")
        self._tables_before = self.table_signature()

    def signature(self):
        l, rope = self.layout, self.rope
        fields = ("position_ids", "condition_indices", "audio_indices", "text_indices")
        tensors = tuple((name, _tensor_stamp(getattr(l, name))) for name in fields)
        module_state = tuple((name, _tensor_stamp(t)) for name, t in
                             list(rope.named_parameters()) + list(rope.named_buffers()))
        return (id(l), l.latent_height, l.latent_width, tuple(l.patch_size),
                id(rope), rope.training, rope.rope_freq_dim, tensors, module_state)

    def table_signature(self):
        return (tuple(_tensor_stamp(t) for t in self.prefix),
                tuple((k, tuple(_tensor_stamp(t) for t in value))
                      for k, value in sorted(self.windows.items())))

    def validate(self):
        if self.closed:
            raise RuntimeError("request rotary owner is closed")
        if self.signature() != self._before:
            raise RuntimeError("request rotary inputs mutated; recreate request owner")
        if self.table_signature() != self._tables_before:
            raise RuntimeError("prepared rotary tables mutated")

    def prepared(self, control, rope):
        # The outer h3_student_forward hook already validates state before each
        # model invocation. Keep mutation/version checks outside model.forward.
        if self.closed or control.mode != "rollout" or control.layout is not self.layout or rope is not self.rope:
            raise RuntimeError("request rotary identity mismatch")
        return replace(control, prefix_rotary=self.prefix, window_rotaries=self.windows)

    def close(self):
        self.closed = True
        self.prefix = None
        self.windows = None

@contextmanager
def request_rotaries(inputs, rope, *, enabled=False):
    """One nonconcurrent inference request. OFF changes no hook or tensor."""
    global _ACTIVE
    if not enabled:
        yield None
        return
    if _ACTIVE is not None:
        raise RuntimeError("overlapping rotary owners are unsupported")
    layout = inputs.layout("rollout") # explicit first-use, outside model boundary
    control = attention.H3SGFAttention(layout, inputs.camera_viewmats, inputs.camera_K, "rollout")
    native_prepare, native_forward = attention.prepare_sgf_rotaries, rollout.h3_student_forward
    owner = RequestRotaries(control, rope, native_prepare)
    def checked_forward(student, call_inputs, noisy, timestep, **kwargs):
        if call_inputs is not inputs or kwargs.get("clean") is not None:
            raise RuntimeError("cached rotary owner used by another request or replay")
        owner.validate()
        owner.hits += 1 # host diagnostic, outside model.forward compile region
        return native_forward(student, call_inputs, noisy, timestep, **kwargs)
    _ACTIVE = owner
    attention.prepare_sgf_rotaries = owner.prepared
    rollout.h3_student_forward = checked_forward
    try:
        yield owner
    finally:
        attention.prepare_sgf_rotaries = native_prepare
        rollout.h3_student_forward = native_forward
        owner.close()
        _ACTIVE = None
