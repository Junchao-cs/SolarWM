"""Portable full-model Memory engine.

The native first six chunks, per-chunk local camera origin, full50 D0 and three
query programs retain the tested boundary. Process compilers are separate from
bounded shape Graphs and request/cache ownership. No legacy experiment imports.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
import hashlib
import json
import time
from deploy.solar_wm.backends.memory.runtime.memory_engine import MemoryEngine
from deploy.solar_wm.backends.memory.acceleration.graph.domains import Domains
from deploy.solar_wm.backends.memory.acceleration.graph.domains import scene_key
from deploy.solar_wm.backends.memory.acceleration.compile.fill import SharedD0
from deploy.solar_wm.backends.memory.acceleration.compile.query import SharedQuery
from deploy.solar_wm.backends.memory.acceleration.graph.shared_io import FullFill
from deploy.solar_wm.backends.memory.acceleration.graph.shared_io import Query
from deploy.solar_wm.backends.memory.acceleration.cache.route import Route
from deploy.solar_wm.backends.memory.runtime.gc_owner import RequestGC
from deploy.solar_wm.backends.memory.runtime.host_camera import install_for_request
from deploy.solar_wm.backends.memory.runtime.memory_window import local_memory_window
from deploy.solar_wm.backends.memory.runtime.memory_window import select_global
from deploy.solar_wm.backends.memory.acceleration.attention import memory_window as window


class Shape:
    def __init__(self, layout):
        self.layout = layout
        self.graph = self.query = None
        self.ready = self.closed = False

    def close(self):
        if self.closed:
            return
        if self.query is not None:
            self.query.close()
        if self.graph is not None:
            self.graph.close()
        self.closed = True


class FillPort:
    def __init__(self, engine, resident):
        self.engine, self.resident = engine, resident
        self.owner = self.source = None

    @contextmanager
    def filling(self, owner, source):
        if self.owner is not None:
            raise RuntimeError('nested D0 producer')
        owner.begin_fill(source)
        self.owner, self.source = owner, source
        try:
            yield
            if owner.seen != set(range(50)):
                raise RuntimeError('incomplete full50 prefix')
        finally:
            self.owner = self.source = None

    def __call__(self, **kwargs):
        if self.owner is None or kwargs['sgf_attention'] is not self.source:
            raise RuntimeError('D0 owner mismatch')
        prepared = window.Window(self.source, self.engine.base.rope)
        result = self.resident(dict(kwargs, sgf_attention=prepared.control), graph=self.engine.capture_enabled)
        if len(result) != 3 or len(result[2]) != 50:
            raise RuntimeError('full50 output ABI')
        for layer, (key, value) in enumerate(result[2]):
            self.owner.fill(layer, key, value, key.shape[1])
        prepared.finish()
        return result[:2]


class QueryPort:
    def __init__(self, engine, resident):
        self.engine, self.resident = engine, resident

    def entry(self, phase):
        def call(*args):
            if not self.engine.capture_enabled:
                return self.engine.query_compiler.entry(phase)(*args)
            return self.resident(phase, args, (self.engine.session_count, self.engine.index))
        return call


class Engine(MemoryEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from solar.backends.minimax_h3 import distributed as d, model, torch_prope
        from deploy.solar_wm.shared.runtime.cache import CompilerCache
        self.native_forward = self.base.forward
        self.native_attention = model.SolarMiniMaxH3Transformer3DModel.__dict__['_sgf_attention_forward']
        self.global_modules = d, model, torch_prope
        self.native_globals = [{k: v for k, v in vars(m).items() if callable(v)} for m in self.global_modules]
        self.process_scope = ExitStack()
        self.canonical = False
        self.route_scope = self.route = self.domain = self.full_compiler = None
        self.domains = Domains(self.deployment.get('graph_capacity', 2))
        # Use all actual packaged accelerator sources, not UI files or a fixed
        # historical experiment label, in this independent cache namespace.
        fingerprint = hashlib.sha256()
        package = Path(__file__).resolve().parents[1]
        from deploy.solar_wm.shared.runtime.sources import numerical_sources
        for name, path in sorted(numerical_sources(package).items()):
            fingerprint.update(name.encode())
            fingerprint.update(path.read_bytes())
        self.accel_cache = CompilerCache(Path(self.deployment['cache_directory'])/'memory-dynamic',
                                        key=fingerprint.hexdigest(), rank=self.rank)
        self.query_compiler = SharedQuery(self.base, self.rank, self.request_group, self.run/f'query-rank{self.rank}')
        self.rows = []
        self.warmed = self.capture_enabled = False
        self.gc_owner = RequestGC(self.run/f'gc-rank{self.rank}.jsonl',
            lambda: dict(session=self.session_count, chunk=self.index, rank=self.rank), exclusive_process=True)

    def select_scene(self, record):
        import torch.distributed as dist
        from safetensors import safe_open
        from deploy.solar_wm.backends.memory.acceleration.compile.guard import consensus
        def validate():
            super(Engine, self).select_scene(record)
            with safe_open(record['path'], framework='numpy') as f:
                shape = f.get_slice('prompt_embeds').get_shape()
                if len(shape) not in (2, 3) or shape[-1] != 5120 or not 1000 <= shape[-2] <= 2048:
                    raise ValueError('unsupported unpadded prompt shape')
                if f.get_slice('text_token_tags').get_shape() != [shape[-2]]:
                    raise ValueError('prompt/tag extent mismatch')
                if f.get_slice('K').get_shape() != [290, 3, 3]:
                    raise ValueError('intrinsics extent')
        consensus('Memory scene metadata', validate, dist=dist, group=self.request_group)

    def start(self, *, seed=42):
        from solar.backends.minimax_h3.memory_layout import memory_rollout_layout
        super().start(seed=seed)
        tags = self.inputs.text_tags.cpu()
        key = scene_key(int(self.inputs.prompt.shape[1]), tags.numpy().tobytes())
        self.domain = self.domains.acquire(key, lambda: Shape(memory_rollout_layout(
            tags, 5, self.inputs.audio_rows.shape[1]//2, self.device, height=48, width=84)))
        self.memory_forward.layouts[5] = self.domain.layout
        self.query_compiler.begin_request()
        if self.full_compiler is not None:
            self.full_compiler.begin_request()
        self.rows = []
        self.warmed = self.domain.ready
        install_for_request(self)
        self.gc_owner.begin(self.deployment.get('request_gc', 'freeze'))

    @contextmanager
    def native_growth(self):
        from solar.backends.minimax_h3 import model
        owner = model.SolarMiniMaxH3Transformer3DModel
        changed = []
        forward, attention = self.base.forward, owner.__dict__['_sgf_attention_forward']
        try:
            for module, saved in zip(self.global_modules, self.native_globals):
                for name, original in saved.items():
                    current = getattr(module, name)
                    if current is not original:
                        changed.append((module, name, current, original))
                        setattr(module, name, original)
            owner._sgf_attention_forward = self.native_attention
            self.base.forward = self.native_forward
            yield
        finally:
            corrupt = self.base.forward is not self.native_forward or owner.__dict__['_sgf_attention_forward'] is not self.native_attention
            self.base.forward = forward
            owner._sgf_attention_forward = attention
            for module, name, current, original in reversed(changed):
                corrupt |= getattr(module, name) is not original
                setattr(module, name, current)
            if corrupt:
                self.failed = True
                raise RuntimeError('native growth owner changed')

    def begin_route(self):
        from solar.backends.minimax_h3 import distributed as d, memory_stage2_rollout
        from solar.backends.minimax_h3.memory_kv_attention import MemoryKVAttention
        from deploy.solar_wm.backends.memory.acceleration.attention.memory_scope import installed
        from deploy.solar_wm.backends.memory.acceleration.cache.preparation import BankPreparation
        from deploy.solar_wm.backends.memory.acceleration.cache.preparation import RotaryPreparation
        if not self.canonical:
            self.process_scope.enter_context(installed(self.base))
            self.canonical = True
        layout = self.domain.layout
        d.state().sequence_lengths = tuple(d.contiguous_packed_bounds(layout.sequence_length, sp_size=8, sp_rank=i)[1]
                                          - d.contiguous_packed_bounds(layout.sequence_length, sp_size=8, sp_rank=i)[0] for i in range(8))
        self.route_scope = ExitStack()
        self.bankprep = BankPreparation(self.memory_forward.bank, memory_stage2_rollout.patchify_video)
        self.rotary = RotaryPreparation()
        self.route_scope.enter_context(self.bankprep.installed_forward(memory_stage2_rollout, self.memory_forward))
        self.route_scope.enter_context(self.rotary.installed(MemoryKVAttention))
        if self.full_compiler is None:
            self.full_compiler = SharedD0(self.native_forward, self.run/f'd0-rank{self.rank}', self.request_group)
        if self.domain.graph is None:
            self.domain.graph = FullFill(self.full_compiler, self.request_group, shared_history=True)
            self.domain.query = Query(self.query_compiler, self.request_group, self.domain.graph)
        self.route = Route(self, FillPort(self, self.domain.graph), QueryPort(self, self.domain.query),
                           request=f'{self.run.name}/{self.session_count}/{self.rank}')
        self.base.forward = self.route

    def generate(self):
        import torch
        callback = getattr(self, 'preparation_progress', None)
        index = self.index
        if callback is not None:
            stage = 'compiling' if index == 6 and (self.full_compiler is None or not self.full_compiler.calls) else (
                'capturing' if index >= 7 and (self.domain.graph is None or self.domain.graph.phase is None) else 'warming')
            callback(stage, index)
        selected = select_global(self.memory_forward, index)
        torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        with local_memory_window(self.inputs.camera_viewmats, index, selected):
            if index < 6:
                with self.native_growth():
                    latent, _ = super().generate()
            else:
                with self.accel_cache.installed('dit'):
                    if self.route is None:
                        self.begin_route()
                    self.capture_enabled = index >= 7
                    latent, _ = super().generate()
        torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter()-start)*1000
        self.session_times[-1] = elapsed
        self.warmed = bool(self.domain.graph and self.domain.graph.phase and len(self.domain.query.phases) == 3)
        self.domain.ready = self.warmed
        row = dict(chunk=index, dit_ms=elapsed, selected_memory=selected, graph_ready=self.warmed,
                   fullfill=None if self.domain.graph is None else self.domain.graph.report(),
                   query=None if self.domain.query is None else self.domain.query.report())
        self.rows.append(row)
        if self.rank == 0:
            print('MEMORY_ACCEL '+json.dumps(row), flush=True)
        return latent, elapsed

    def end_route(self):
        if self.route_scope is None:
            return
        self.base.forward = self.native_forward
        self.route_scope.close()
        self.route_scope = None
        if self.route is not None:
            self.route.reset()
        self.route = None

    def stop(self):
        import torch
        if self.scopes is None:
            raise RuntimeError('no active request')
        try:
            phases = None if self.route is None else dict(self.route.counts)
            self.end_route()
            torch.cuda.synchronize(self.device)
            report = dict(chunks=self.index, dit_ms=self.session_times, rows=self.rows, memory=self.memory_identity,
                          retrieval=self.memory_forward.records, camera_policy='local-memory-window',
                          full_compiler=None if self.full_compiler is None else self.full_compiler.report(),
                          query_compiler=self.query_compiler.calls, graph=self.warmed,
                          phases=phases, domains=self.domains.report())
            with (self.run/f'memory-session-{self.session_count}-rank{self.rank}.json').open('x') as f:
                json.dump(report, f)
            self.cache.clear()
            self.scopes.close()
            self.scopes = self.inputs = self.noise = self.rng = self.memory_forward = None
            self.poses = []
            self.domains.release()
            return report
        finally:
            self.gc_owner.end()

    def recover_rejected_scene(self, error):
        if not getattr(error, 'pre_call_collective_agreement', False):
            raise error
        key = self.domains.active
        if self.scopes is not None:
            self.stop()
        else:
            self.end_route()
        if key is not None:
            if self.domains.active is not None:
                self.domains.release()
            self.domains.entries.pop(key).close()
        self.domain = None
        self.warmed = self.failed = False

    def close(self):
        try:
            self.end_route()
            if self.domains.active is not None:
                self.domains.release()
            self.domains.close()
            self.process_scope.close()
            super().close()
        finally:
            self.gc_owner.close()
