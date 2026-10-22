"""Read cached Dynamo guards before allowing any rank into a compiled call.

This does not replace Dynamo guards, disable recompilation errors, or catch
CUDA/NCCL failures. Only a unanimous, pre-call rejection is recoverable."""

import types


class SceneRejected(RuntimeError):
    def __init__(self, stage, errors):
        super().__init__('scene rejected at '+stage+': '+'; '.join(errors))
        self.stage, self.errors = stage, tuple(errors)
        self.pre_call_collective_agreement = True


def consensus(stage, callback, *, dist, group, size=8):
    value = None
    try:
        value = callback()
        error = None
    except (ValueError, TypeError, KeyError, OSError, SceneRejected) as exc:
        error = type(exc).__name__+': '+str(exc)[:1600]
    # Deliberately don't catch RuntimeError from CUDA or distributed execution.
    rows = [None]*size
    dist.all_gather_object(rows, error, group=group)
    errors = [f'rank {r}: {e}' for r,e in enumerate(rows) if e is not None]
    if errors:
        raise SceneRejected(stage, errors)
    return value


def check_cached_call(compiled, kwargs):
    """Check the exact original function's locals plus its closure cells."""
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    original = compiled._torchdynamo_orig_callable
    if type(original) is not types.FunctionType:
        raise TypeError('owned Python compiled forward required')
    scope = {'kwargs': kwargs}
    scope.update(zip(original.__code__.co_freevars,
                     (c.cell_contents for c in original.__closure__ or ())))
    entries = _debug_get_cache_entry_list(original)
    if not entries:
        raise ValueError('compiled forward has no reusable guard entry')
    failures = []
    for entry in entries:
        manager = entry.guard_manager
        if manager.check(scope):
            return
        diagnostic = manager.check_verbose(scope)
        failures.append(str(diagnostic.verbose_code_parts)[:1200])
    raise ValueError('cached guard mismatch; no forward or recompile started: '+str(failures))


class GuardedCompilerMixin:
    def begin_scene_checks(self, dist, group):
        self.scene_guard_dist, self.scene_guard_group = dist, group
        self.scene_checked_phases = set()

    def entry(self, phase):
        existed = phase in self.entries
        function = super().entry(phase)
        if not existed or not hasattr(self, 'scene_checked_phases'):
            return function
        def guarded(**kwargs):
            if phase not in self.scene_checked_phases:
                import torch
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError('request guard must be checked before Graph capture')
                consensus('compiled phase '+str(phase),
                    lambda: check_cached_call(function, kwargs),
                    dist=self.scene_guard_dist, group=self.scene_guard_group)
                self.scene_checked_phases.add(phase)
            return function(**kwargs)
        return guarded
