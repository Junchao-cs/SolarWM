"""Scoped rotary, sequence, backend and communication bindings."""

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
from .rotary import request_rotaries
from .sequence import request_sp_lengths
from .refiner import canonical_refiner_backend
from .all_to_all import canonical_all_to_all
from .all_gather import canonical_all_gather
from .prope import canonical_prope_import


@dataclass(frozen=True)
class CoreOwners:
    rotaries: object = None
    sp_lengths: object = None

@contextmanager
def request_core(inputs,rope,*,enabled=False):
    with ExitStack() as stack:
        # LIFO restoration is part of the contract. Outer RoPE -> inner SP ->
        # observed/native h3_student_forward; both guards precede model.forward.
        rotary=stack.enter_context(request_rotaries(inputs,rope,enabled=enabled))
        lengths=stack.enter_context(request_sp_lengths(inputs,enabled=enabled))
        yield CoreOwners(rotary,lengths)


@dataclass(frozen=True)
class CombinedOwners:
    core: object = None
    extra: tuple = ()
    def validate(self):
        if self.core is not None:
            self.core.rotaries.validate()
            self.core.sp_lengths.validate()
        for owner in self.extra:
            owner.validate()

def hook_snapshot(base):
    import solarwm.backends.minimax_h3.sgf_attention as a
    import solarwm.backends.minimax_h3.sgf_rollout as r
    import solarwm.backends.minimax_h3.distributed as d
    import solarwm.backends.minimax_h3.model as m
    import solarwm.backends.minimax_h3.torch_prope as p
    backend = tuple((id(b.attn.processor),
                     "_attention_backend" in b.attn.processor.__dict__,
                     type(b.attn.processor._attention_backend),
                     b.attn.processor._attention_backend)
                    for b in base.token_refiner.refiner_blocks)
    return (a.prepare_sgf_rotaries, r.h3_student_forward, d.register_sequence_length,
            d.sequence_all_to_all, m.sequence_model_parallel_all_to_all_4D,
            d._all_gather_apply, d.sequence_all_gather,
            m.sequence_model_parallel_all_gather, p._invert_se3, backend)

@contextmanager
def combined_core(inputs, base, *, enabled=False):
    if type(enabled) is not bool:
        raise TypeError("explicit boolean candidate switch required")
    if not enabled:
        yield CombinedOwners()
        return
    before = hook_snapshot(base)
    try:
        with ExitStack() as stack:
            core = stack.enter_context(request_core(inputs, base.rope, enabled=True))
            extra = (
                stack.enter_context(canonical_refiner_backend(base.token_refiner, enabled=True, expected_backend="flash")),
                stack.enter_context(canonical_all_to_all(enabled=True)),
                stack.enter_context(canonical_all_gather(enabled=True)),
                stack.enter_context(canonical_prope_import(enabled=True)),
            )
            owners = CombinedOwners(core, extra)
            owners.validate()
            yield owners
            owners.validate()
    finally:
        if hook_snapshot(base) != before:
            raise RuntimeError("combined canonical scope did not restore entry hooks")
