"""Cross-length phase compilation, prompt axes and dynamic request metadata."""

from contextlib import contextmanager
import re
from pathlib import Path
import json
import time
import types


MODULE = "G['__import_solarwm_dot_backends_dot_minimax_h3_dot_distributed']"
OWNER = MODULE + '.register_sequence_length.__self__'
SOURCES = (OWNER + '.local', *(OWNER + f'.expected[{i}]' for i in range(8)),
           *(MODULE + f'._STATE.sequence_lengths[{i}]' for i in range(8)))
# Full-match regexes avoid unintentionally marking any topology/control scalar.
DYNAMIC_SOURCES = ','.join(re.escape(s) + '$' for s in SOURCES)

@contextmanager
def dynamic_metadata():
    import torch.compiler.config
    with torch.compiler.config.patch(dynamic_sources=DYNAMIC_SOURCES):
        yield


def mark_prompt_dimensions(kwargs, rotary_owner):
    import torch
    layout = kwargs['sgf_attention'].layout
    if rotary_owner is None or rotary_owner.layout is not layout or rotary_owner.closed:
        raise RuntimeError('current live rotary owner required')
    prompt = kwargs['encoder_hidden_states']
    if tuple(prompt.shape[::2]) != (1, 5120) or prompt.ndim != 3:
        raise ValueError('expected actual [1,L,5120] condition')
    torch._dynamo.mark_dynamic(prompt, 1, min=1000, max=1300)
    for name in ('position_ids', 'token_tags', 'timestep_indices', 'text_indices'):
        value = kwargs[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(name)
        torch._dynamo.mark_dynamic(value, 0)
    for name in ('position_ids', 'token_tags', 'text_indices', 'condition_indices', 'row_roles'):
        value = getattr(layout, name)
        if not isinstance(value, torch.Tensor):
            raise TypeError(name)
        torch._dynamo.mark_dynamic(value, 0)
    for value in rotary_owner.prefix:
        torch._dynamo.mark_dynamic(value, 0)
    # All fixed video/audio/camera dimensions, layer sizes, topology and scalar
    # configuration stay static under torch.compile(dynamic=False).


class CrossLengthRecompile(RuntimeError):
    pass


class Evidence:
    def __init__(self, folder, rank):
        self.folder = Path(folder)
        self.rank = rank
        self.scene = None
        self.sequence = None
        self.events = self.folder / f'rank-{rank}.jsonl'

    def emit(self, event, **fields):
        row = dict(event=event, rank=self.rank, scene=self.scene,
                   sequence=self.sequence, monotonic=time.monotonic(), **fields)
        with self.events.open('a') as out:
            out.write(json.dumps(row, sort_keys=True) + '\n')
        print(json.dumps(row, sort_keys=True), flush=True)


def phase_key(control):
    # Early KV-history depth and commit side effects are separate native regimes.
    # Prompt length and scene ID must NEVER be part of this key.
    return min(control.chunk_index, 5), bool(control.commit_cache)


class SharedCompiler:
    def __init__(self, native, evidence, *, mode):
        if mode not in ('trace', 'inductor'):
            raise ValueError(mode)
        self.native, self.evidence, self.mode = native, evidence, mode
        self.entries = {}
        self.backend_calls = {}
        self.allow_compile = True

    def entry(self, phase):
        import torch
        if phase not in self.entries:
            if not self.allow_compile:
                raise CrossLengthRecompile('new phase outside first scene')
            native = self.native

            def forward(**kwargs):
                return native(**kwargs)

            # One stable code object per phase, shared by ALL scenes and lengths.
            code = forward.__code__.replace(co_filename=__file__ + f'#phase-{phase}')
            forward = types.FunctionType(code, forward.__globals__, forward.__name__,
                                         forward.__defaults__, forward.__closure__)

            def backend(gm, examples):
                count = self.backend_calls.get(phase, 0) + 1
                self.backend_calls[phase] = count
                placeholders = []
                for node in gm.graph.nodes:
                    if node.op == 'placeholder':
                        value = node.meta.get('example_value')
                        placeholders.append(dict(name=node.name, type=type(value).__name__,
                            shape=[str(s) for s in value.shape] if isinstance(value, torch.Tensor) else str(value)))
                self.evidence.emit('backend_enter', phase=phase, count=count,
                    mode=self.mode, graph_nodes=len(list(gm.graph.nodes)),
                    placeholders=placeholders,
                    attention_operators=sorted({str(n.target) for n in gm.graph.nodes
                        if any(s in str(n.target).lower() for s in ('attention', 'flash_attn', 'sdpa'))}))
                # Do not silently pay for a new length specialization.
                if not self.allow_compile:
                    raise CrossLengthRecompile('different length retraced the full model')
                path = self.evidence.folder / f'fx-r{self.evidence.rank}-{phase[0]}-{int(phase[1])}-{count}.py'
                with path.open('x') as out:
                    out.write(gm.code)
                if self.mode == 'trace':
                    # Real full-model FX execution, not Inductor or a speed result.
                    return gm.forward
                from torch._dynamo.backends.registry import lookup_backend
                started = time.perf_counter()
                result = lookup_backend('inductor')(gm, examples)
                self.evidence.emit('inductor_return', phase=phase,
                                   seconds=time.perf_counter() - started)
                return result

            self.entries[phase] = torch.compile(forward, backend=backend,
                                                fullgraph=True, dynamic=False)
        return self.entries[phase]

    def __call__(self, **kwargs):
        import torch
        from .packed_window import PackedWindowCall
        control = kwargs['sgf_attention']
        private = PackedWindowCall(control, layer_count=50)
        prompt = kwargs['encoder_hidden_states']
        if prompt.ndim != 3 or prompt.shape[0] != 1 or prompt.shape[2] != 5120:
            raise ValueError('unexpected actual H3 prompt')
        # Fail rather than pretending a specialized length is dynamic.
        from solarwm.backends.minimax_h3 import sgf_attention
        rotary_owner = getattr(sgf_attention.prepare_sgf_rotaries, '__self__', None)
        mark_prompt_dimensions(dict(kwargs, sgf_attention=private.control), rotary_owner)
        fn = self.entry(phase_key(control))
        with dynamic_metadata(), torch._dynamo.config.patch(error_on_recompile=not self.allow_compile):
            result = fn(**dict(kwargs, sgf_attention=private.control))
        private.finish()
        return result
