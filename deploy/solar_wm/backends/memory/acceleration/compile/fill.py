"""Process-shared full50 D0; prompt axes and exact SP metadata are symbolic.

Reuses the deployed numerical Core and canonical collectives. No prompt padding,
precision change, reduced layer count, or per-shape compiler is introduced.
"""
from contextlib import contextmanager
from pathlib import Path
import json
import re
import time
import torch
from deploy.solar_wm.backends.memory.acceleration.compile.fill_math import Core
MODULE = "G['__import_solar_dot_backends_dot_minimax_h3_dot_distributed']"
OWNER = MODULE + '.register_sequence_length.__self__'
SOURCES = (OWNER + '.local', *(OWNER + f'.expected[{i}]' for i in range(8)), *(MODULE + f'._STATE.sequence_lengths[{i}]' for i in range(8)))
DYNAMIC_SOURCES = ','.join((re.escape(s) + '$' for s in SOURCES))

@contextmanager
def dynamic_metadata():
    import torch.compiler.config
    with torch.compiler.config.patch(dynamic_sources=DYNAMIC_SOURCES):
        yield

def mark_prompt(kwargs):
    c = kwargs['sgf_attention']
    n = kwargs['encoder_hidden_states'].shape[1]
    if not 1000 <= n <= 2048 or c.chunk_index != 5 or c.commit_cache or (len(c.bank_indices) != 5):
        raise ValueError('unsupported unpadded full50 Memory D0')
    torch._dynamo.mark_dynamic(kwargs['encoder_hidden_states'], 1, min=1000, max=2048)
    for name in ('position_ids', 'token_tags', 'timestep_indices', 'text_indices'):
        torch._dynamo.mark_dynamic(kwargs[name], 0)
    for name in ('position_ids', 'token_tags', 'text_indices', 'condition_indices', 'row_roles'):
        torch._dynamo.mark_dynamic(getattr(c.layout, name), 0)
    for t in c.prefix_rotary:
        torch._dynamo.mark_dynamic(t, 0)

class LengthBinding:
    """One owner identity; actual per-shape counts are checked outside tracing."""

    def __init__(self):
        from solar.backends.minimax_h3 import distributed as d
        self.d = d
        self.native = d.register_sequence_length
        self.verified = set()
        self.active = False
        self.state = d.state()
        self.group = self.state.sp_group

    def register(self, local):
        if local != self.local:
            raise RuntimeError('D0 local shard differs from live cohort')
        return self.expected

    def open(self, total):
        d = self.d
        if self.active or d.register_sequence_length is not self.native or d.state() is not self.state or (self.state.sp_group is not self.group):
            raise RuntimeError('D0 SP owner changed')
        self.previous = self.state.sequence_lengths
        self.expected = tuple((d.contiguous_packed_bounds(total, sp_size=8, sp_rank=i)[1] - d.contiguous_packed_bounds(total, sp_size=8, sp_rank=i)[0] for i in range(8)))
        self.local = self.expected[self.state.sp_rank]
        if total not in self.verified:
            if self.native(self.local) != self.expected:
                raise RuntimeError('real allrank SP length mismatch')
            self.verified.add(total)
        d.register_sequence_length = self.register
        self.active = True

    def close(self):
        if not self.active:
            return
        if self.d.register_sequence_length != self.register:
            raise RuntimeError('D0 SP binding overwritten')
        self.d.register_sequence_length = self.native
        self.state.sequence_lengths = self.previous
        self.active = False

class SharedD0:

    def __init__(self, native, folder, group):
        self.core = Core(native)
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.group = group
        self.fn = None
        self.calls = 0
        self.checked = set()
        self.lengths = LengthBinding()
        self.inputs_seen = set()
        self.invocations = 0

    def begin_request(self):
        self.checked.clear()

    def check(self, kwargs):
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list
        original = self.fn._torchdynamo_orig_callable
        scope = {'kwargs': kwargs}
        scope.update(zip(original.__code__.co_freevars, (c.cell_contents for c in original.__closure__ or ())))
        entries = _debug_get_cache_entry_list(original)
        if any((e.guard_manager.check(scope) for e in entries)):
            return
        failures = [str(e.guard_manager.check_verbose(scope).verbose_code_parts)[:1800] for e in entries]
        raise ValueError('shared D0 guard rejected before GPU call: ' + str(failures))

    def __call__(self, kwargs):
        import torch.distributed as dist
        from deploy.solar_wm.backends.memory.acceleration.compile.guard import consensus
        capturing = torch.cuda.is_current_stream_capturing()
        total = kwargs['sgf_attention'].layout.sequence_length
        token = id(kwargs['sgf_attention'].layout)
        if not capturing:
            mark_prompt(kwargs)
        if self.fn is None:
            core = self.core

            def forward(kwargs):
                return core(kwargs)

            def backend(gm, examples):
                self.calls += 1
                if self.calls != 1:
                    raise RuntimeError('D0 must compile once across prompt lengths')
                (self.folder / 'dynamic-d0-fx.py').write_text(gm.code)
                symbols = [str(x) for x in examples if isinstance(x, torch.SymInt)]
                print('D0_DYNAMIC_BACKEND_ENTER ' + json.dumps(dict(symbols=symbols)), flush=True)
                from torch._dynamo.backends.registry import lookup_backend
                started = time.perf_counter()
                result = lookup_backend('inductor')(gm, examples)
                with (self.folder / 'compile.json').open('x') as f:
                    json.dump(dict(calls=self.calls, symbols=symbols, seconds=time.perf_counter() - started), f)
                print('D0_DYNAMIC_BACKEND_RETURN', flush=True)
                return result
            with dynamic_metadata():
                self.fn = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
        self.lengths.open(total)
        try:
            if self.calls and token not in self.checked:
                if capturing:
                    raise RuntimeError('D0 guard must be checked before capture')
                consensus('dynamic Memory D0', lambda : self.check(kwargs), dist=dist, group=self.group)
                self.checked.add(token)
            with dynamic_metadata(), torch._dynamo.config.patch(error_on_recompile=bool(self.calls)):
                result = self.fn(kwargs)
            self.inputs_seen.add(int(kwargs['encoder_hidden_states'].shape[1]))
            self.invocations += 1
            return result
        finally:
            self.lengths.close()

    def report(self):
        return dict(compile_calls=self.calls, prompt_lengths=sorted(self.inputs_seen), invocations=self.invocations, padding=False, shared_across_shapes=True)
