"""Keep capture-external rotary constants alive for the CUDA graph lifetime."""

import hashlib
import torch


def pointers(call):
    owner=call.tensor_owner
    tensors=[owner.hidden_states,owner.timestep,owner.timestep_indices,
             owner.anchor_rows,owner.audio_rows,owner.prompt,owner.audio_timestep,
             call.noisy,call.step,call.history_owner.viewmats,call.history_owner.intrinsics]
    for pair in call.history_owner.buffers.values(): tensors.extend(pair)
    if hasattr(call,"writeback"):
        for pair in call.writeback.slots.values(): tensors.extend(pair)
    return tuple((t.data_ptr(),tuple(t.shape),tuple(t.stride()),t.dtype,t.device) for t in tensors)


def rebind_conditions(call, student, inputs):
    if call.student is not student or student.training or call.poisoned:
        raise RuntimeError("resident model/call changed or failed")
    owner=call.tensor_owner
    old=owner.inputs
    if (any(getattr(old,n)!=getattr(inputs,n) for n in
            ("latent_height","latent_width","keyframe_timestep","sp_enabled"))
            or not torch.equal(old.text_tags,inputs.text_tags)):
        raise RuntimeError("resident graph layout specialization changed")
    before=pointers(call)
    with torch.no_grad():
        for name in ("anchor_rows","audio_rows","prompt","audio_timestep"):
            target,source=getattr(owner,name),getattr(inputs,name)
            if (target.shape!=source.shape or target.dtype!=source.dtype
                    or target.device!=source.device or source.requires_grad):
                raise RuntimeError("resident condition specialization changed: "+name)
            target.copy_(source)
    owner.inputs=inputs
    if pointers(call)!=before:raise RuntimeError("resident pointer changed on reset")


def tables(owner):
    return tuple(owner.prefix) + tuple(t for k in sorted(owner.windows)
                                      for t in owner.windows[k])


def stamp(values):
    return tuple((t.data_ptr(), tuple(t.shape), tuple(t.stride()),
                  str(t.dtype), str(t.device), t._version) for t in values)


class GraphBank:
    def __init__(self):
        self.failed = False
        self.rotary_tables = None
        self.rotary_keys = None
        self.rotary_stamp = None
        self.rotary_bindings = 0

    def pin_rotaries(self, owner):
        if self.failed:
            raise RuntimeError('resident bank failed')
        owner.validate()
        values, keys = tables(owner), tuple(sorted(owner.windows))
        if self.rotary_tables is None:
            # Strong references, NOT clones: kernels captured later read these
            # exact addresses. RequestRotaries.close clears its own references.
            self.rotary_tables, self.rotary_keys = values, keys
            self.rotary_stamp = stamp(values)
        else:
            self.validate_rotaries()
            if keys != self.rotary_keys or len(values) != len(self.rotary_tables):
                raise RuntimeError('resident rotary layout changed')
            for old, new in zip(self.rotary_tables, values):
                if (old.shape != new.shape or old.dtype != new.dtype
                        or old.device != new.device or not torch.equal(old, new)):
                    raise RuntimeError('resident rotary values changed')
        self.rotary_bindings += 1

    def validate_rotaries(self):
        if self.rotary_tables is None or stamp(self.rotary_tables) != self.rotary_stamp:
            self.failed = True
            raise RuntimeError('captured rotary constants changed')

    def rotary_receipt(self):
        self.validate_rotaries()
        return dict(retained_tensors=len(self.rotary_tables),
                    retained_bytes=sum(t.numel()*t.element_size() for t in self.rotary_tables),
                    bindings=self.rotary_bindings,
                    pointer_digest=hashlib.sha256(repr(self.rotary_stamp).encode()).hexdigest(),
                    fresh_session_values_equal=True)

    def destroy(self):
        self.failed = True
        self.rotary_tables = self.rotary_stamp = self.rotary_keys = None
