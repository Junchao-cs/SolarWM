"""Request/chunk ownership and one real refresh publication after four denoises."""
from deploy.solar_wm.backends.memory.acceleration.attention import memory_window as window
from deploy.solar_wm.backends.memory.acceleration.cache.prefix import PrefixOwner
from deploy.solar_wm.backends.memory.acceleration.cache.prefix import stamp
from deploy.solar_wm.backends.memory.acceleration.graph.shared_io import BorrowPrefixMixin
from deploy.solar_wm.backends.memory.acceleration.cache.dependencies import Dependencies
from deploy.solar_wm.backends.memory.acceleration.cache.chunk_window import ChunkWindow
from deploy.solar_wm.backends.memory.acceleration.cache.refresh import publish


class DirectPrefix(BorrowPrefixMixin, PrefixOwner):
    pass


class Order:
    def __init__(self):
        self.chunk = None
        self.ordinal = 0

    def select(self, chunk, commit):
        if self.ordinal == 5:
            self.ordinal = 0
        if self.ordinal == 0:
            if commit or chunk == self.chunk:
                raise RuntimeError('new chunk fill required')
            self.chunk = chunk
        if chunk != self.chunk or bool(commit) != (self.ordinal == 4):
            raise RuntimeError('four denoise then one refresh required')
        return ('full-fill', 'derived-fill', 'reuse', 'reuse', 'refresh')[self.ordinal]

    def done(self):
        self.ordinal += 1

    def reset(self):
        self.chunk = None
        self.ordinal = 0


class Route:
    def __init__(self, engine, fill, compiler, request):
        from solar.backends.minimax_h3 import distributed as d
        self.base, self.inputs = engine.base, engine.inputs
        self.full = self.capture = fill
        self.compiler = compiler
        self.owner = DirectPrefix(request=request)
        self.dependencies = Dependencies(engine.base, engine.inputs, engine.memory_forward.bank)
        self.order = Order()
        self.fixed = self.geometry = self.output_stamps = None
        self.d = d
        self.full_lengths = tuple(d.state().sequence_lengths)
        self.query_lengths = (630,)*8
        self.prepared = ChunkWindow(window.Window)
        self.counts = {phase: 0 for phase in ('full-fill', 'derived-fill', 'reuse', 'refresh')}
        self.failed = False

    def reset(self):
        self.order.reset()
        self.owner.reset()
        self.dependencies.reset()
        self.prepared.reset()
        self.fixed = self.geometry = self.output_stamps = None
        self.failed = False

    def __call__(self, **kwargs):
        if self.failed:
            raise RuntimeError('failed route cannot be reused')
        source = kwargs['sgf_attention']
        try:
            phase = self.order.select(source.chunk_index, source.commit_cache)
            if kwargs['encoder_hidden_states'] is not self.inputs.prompt or kwargs['audio_hidden_states'] is not self.inputs.audio_rows:
                raise RuntimeError('conditioning binding changed')
            self.dependencies.check(source, fill=phase == 'full-fill')
            if phase == 'full-fill':
                self.fixed = self.geometry = self.output_stamps = None
                self.d.state().sequence_lengths = self.full_lengths
                with self.capture.filling(self.owner, source):
                    result = self.full(**kwargs)
                self.owner.finish_fill(result, int(source.layout.num_condition_video_rows))
            else:
                self.owner.verify(source)
                if self.fixed is not None:
                    tensors = tuple(t for pair in self.fixed for t in pair) + tuple(self.geometry)
                    if tuple(stamp(t) for t in tensors) != self.output_stamps:
                        raise RuntimeError('derived cache mutated')
                self.d.state().sequence_lengths = self.query_lengths
                prepared = self.prepared.prepare(source, self.base.rope)
                prefix = tuple(self.owner.kv[i] for i in range(50))
                args = (kwargs['hidden_states'], kwargs['timestep'], kwargs['timestep_indices'], kwargs['token_tags'],
                        prepared.control, prefix, self.fixed, self.owner.video, self.owner.audio, self.geometry)
                video, audio, derived, raw, geometry = self.compiler.entry(phase)(*args)
                if phase == 'derived-fill':
                    if len(derived) != 50 or self.fixed is not None or len(geometry) != 3:
                        raise RuntimeError('derived cache ABI')
                    self.fixed, self.geometry = derived, geometry
                    self.output_stamps = tuple(stamp(t) for pair in derived for t in pair) + tuple(stamp(t) for t in geometry)
                elif derived:
                    raise RuntimeError('reuse returned new fixed KV')
                if source.commit_cache:
                    if len(raw) != 50:
                        raise RuntimeError('incomplete raw refresh KV')
                    publish(prepared, raw)
                elif raw:
                    raise RuntimeError('denoise attempted refresh')
                self.prepared.finish()
                self.owner.reuses += 1
                if source.commit_cache:
                    self.owner.commits += 1
                result = video, audio
            self.counts[phase] += 1
            self.order.done()
            return result
        except BaseException:
            self.failed = True
            raise
        finally:
            self.d.state().sequence_lengths = self.full_lengths
