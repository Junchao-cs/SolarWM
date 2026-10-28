"""Opt-in request-owned SP length registration; fixed official SP8 only.

Only registration metadata moves. Q/K/V all-to-all and output gathers remain
native. Every new request still performs native all-rank length validation."""

from contextlib import contextmanager
import solar.backends.minimax_h3.distributed as parallel
import solar.backends.minimax_h3.sgf_rollout as rollout


_ACTIVE = None

class RequestSPLengths:
    def __init__(self, inputs, native_register):
        if not inputs.sp_enabled:
            raise ValueError("SP request owner requires SP-enabled inputs")
        self.inputs = inputs
        self.layout = inputs.layout("rollout")
        self.total = self.layout.sequence_length
        self.state = parallel.state()
        if not self.state.initialized or self.state.sp_size != 8:
            raise ValueError("this candidate requires initialized official SP8")
        self._topology = self.topology()
        self.previous = self.state.sequence_lengths
        self.closed = False
        self.calls = 0
        bounds = [parallel.contiguous_packed_bounds(self.total, sp_size=8, sp_rank=i) for i in range(8)]
        self.expected = tuple(stop-start for start,stop in bounds)
        self.local = self.expected[self.state.sp_rank]
        try:
            actual = native_register(self.local)
            if actual != self.expected or self.state.sequence_lengths != self.expected:
                raise RuntimeError("native SP ranks disagree with frozen request layout")
        except BaseException:
            self.state.sequence_lengths = self.previous
            raise

    def topology(self):
        s = self.state
        return (id(s), s.initialized, s.sp_size, s.sp_rank, s.dp_rank, s.dp_world_size,
                id(s.sp_group), id(s.dp_group), tuple(s.sp_group_ranks), tuple(s.dp_group_ranks))

    def validate(self):
        if self.closed:
            raise RuntimeError("SP length request owner is closed")
        if parallel.state() is not self.state or self.topology() != self._topology:
            raise RuntimeError("SP topology changed inside fixed request")
        if not self.inputs.sp_enabled or self.inputs.layout("rollout") is not self.layout or self.layout.sequence_length != self.total:
            raise RuntimeError("SP input layout changed inside fixed request")
        if self.state.sequence_lengths != self.expected:
            raise RuntimeError("registered SP lengths changed inside fixed request")

    def register(self, local_length):
        # Host-boundary wrapper calls validate before model.forward. This hot
        # function performs no collective, item(), diagnostic counter or mutation.
        if self.closed or type(local_length) is not int or local_length != self.local:
            raise RuntimeError("local SP length differs from frozen request")
        return self.expected

    def close(self):
        self.state.sequence_lengths = self.previous
        self.closed = True

@contextmanager
def request_sp_lengths(inputs, *, enabled=False):
    global _ACTIVE
    if not enabled:
        yield None
        return
    if _ACTIVE is not None:
        raise RuntimeError("overlapping SP length owners are unsupported")
    native_register, native_forward = parallel.register_sequence_length, rollout.h3_student_forward
    owner = RequestSPLengths(inputs,native_register)
    def checked_forward(student,call_inputs,noisy,timestep,**kwargs):
        if call_inputs is not inputs or kwargs.get("clean") is not None:
            raise RuntimeError("SP length owner used by another request or replay")
        owner.validate()
        owner.calls += 1
        return native_forward(student,call_inputs,noisy,timestep,**kwargs)
    _ACTIVE = owner
    parallel.register_sequence_length = owner.register
    rollout.h3_student_forward = checked_forward
    try:
        yield owner
    finally:
        parallel.register_sequence_length = native_register
        rollout.h3_student_forward = native_forward
        owner.close()
        _ACTIVE = None
