"""Three process-shared full50 query programs with actual symbolic lengths."""
from pathlib import Path
import json
import time
import types
import torch
from deploy.solar_wm.backends.memory.acceleration.compile.query_math import FullQuery


def mark_call(args):
    if len(args) != 10:
        raise ValueError('full50 query ABI')
    control = args[4]
    layout = control.layout
    length = int(layout.text_indices.numel())
    if not 1000 <= length <= 2048 or control.chunk_index != 5 or len(control.bank_indices) != 5:
        raise ValueError('unsupported unpadded mature Memory domain')
    for tensor, axis in ((layout.position_ids, 0), (layout.text_indices, 0), (args[2], 0), (args[3], 0)):
        torch._dynamo.mark_dynamic(tensor, axis)
    for tensor in control.prefix_rotary:
        torch._dynamo.mark_dynamic(tensor, 0)
    for key, value in args[5]:
        torch._dynamo.mark_dynamic(key, 1)
        torch._dynamo.mark_dynamic(value, 1)
    if args[6] is not None:
        for key, value in args[6]:
            torch._dynamo.mark_dynamic(key, 2)
            torch._dynamo.mark_dynamic(value, 2)
    if args[9] is not None:
        torch._dynamo.mark_dynamic(args[9][1], 1)


def check_call(fn, args):
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    original = fn._torchdynamo_orig_callable
    scope = {'args': args}
    scope.update(zip(original.__code__.co_freevars, (c.cell_contents for c in original.__closure__ or ())))
    entries = _debug_get_cache_entry_list(original)
    if not entries:
        raise ValueError('shared compiler has no warmed guard')
    if any(entry.guard_manager.check(scope) for entry in entries):
        return
    failures = [str(entry.guard_manager.check_verbose(scope).verbose_code_parts)[:1400] for entry in entries]
    raise ValueError('scene rejected before GPU execution: ' + str(failures))


class SharedQuery:
    def __init__(self, base, rank, group, folder):
        self.group, self.rank = group, rank
        self.core = FullQuery(base, rank, torch.bfloat16)
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.entries, self.counts, self.checked = {}, {}, set()
        self.invocations = 0

    @property
    def calls(self):
        return dict(self.counts)

    def begin_request(self):
        self.checked.clear()

    def entry(self, phase):
        if phase not in ('derived-fill', 'reuse', 'refresh'):
            raise ValueError('query phase')
        if phase not in self.entries:
            core = self.core

            def forward(*args):
                return core(*args)

            forward = types.FunctionType(forward.__code__.replace(co_filename=__file__+'#'+phase),
                                         forward.__globals__, forward.__name__, forward.__defaults__, forward.__closure__)

            def backend(gm, examples):
                from torch._dynamo.backends.registry import lookup_backend
                self.counts[phase] = self.counts.get(phase, 0) + 1
                if self.counts[phase] != 1:
                    raise RuntimeError('unexpected query recompile: ' + phase)
                (self.folder/(phase+'-fx.py')).write_text(gm.code)
                start = time.perf_counter()
                result = lookup_backend('inductor')(gm, examples)
                with (self.folder/'events.jsonl').open('a') as f:
                    f.write(json.dumps(dict(rank=self.rank, phase=phase, seconds=time.perf_counter()-start))+'\n')
                return result

            self.entries[phase] = torch.compile(forward, backend=backend, fullgraph=True, dynamic=False)
        compiled = self.entries[phase]

        def call(*args):
            capturing = torch.cuda.is_current_stream_capturing()
            if not capturing:
                mark_call(args)
            if self.counts.get(phase, 0) and phase not in self.checked:
                if capturing:
                    raise RuntimeError('guard check must precede Graph capture')
                import torch.distributed as dist
                from deploy.solar_wm.backends.memory.acceleration.compile.guard import consensus
                consensus('Memory dynamic '+phase, lambda: check_call(compiled, args), dist=dist, group=self.group)
                self.checked.add(phase)
            with torch._dynamo.config.patch(error_on_recompile=bool(self.counts.get(phase, 0))):
                result = compiled(*args)
            self.invocations += 1
            return result
        return call
