"""Resident denoise/refresh Graphs, scene rebinding and owned cleanup."""

from collections import OrderedDict
from types import SimpleNamespace
from .pool import ChunkCopyPool
from .denoise_bridge import capture as capture_denoise, stage
from .refresh_bridge import capture as capture_refresh, replay_and_publish


def close_full_student_graphs(bundles, *, synchronize, emit):
    owners=[b['graph'] for b in bundles if not getattr(b['graph'],'closed',False)]
    if not owners:
        return 0
    if len({id(x) for x in owners})!=len(owners):
        raise RuntimeError('duplicate full-student Graph owner')
    synchronize()
    closed=0
    for owner in owners:
        # reset() releases the CUDA graph and executable even if Python still
        # holds the enclosing model/call/bundle for diagnostics. No replay is
        # allowed afterwards; its native graph pointer is removed.
        if owner.graph is not None:
            owner.graph.reset()
        owner.graph=None
        owner.static_output=None
        owner.stream=None
        owner.call.ready=False
        owner.closed=True
        closed+=1
        emit('full_student_graph_released',ordinal=closed)
    synchronize()
    return closed


class ResidentFullStudentRuntime:
    def __init__(self, *, device, emit, capacity=2):
        if type(capacity) is not int or capacity<1:
            raise ValueError('positive resident shape capacity required')
        self.device,self.emit,self.capacity=device,emit,capacity
        self.slots=OrderedDict()
        self.active=None
        self.sessions=0
        self.evictions=0
        self.failed=False
        self.closed=False

    def begin_session(self, *, key, engine, rotaries):
        import torch
        from .bank import GraphBank
        if self.closed or self.failed or self.active is not None:
            raise RuntimeError('resident runtime unavailable or already checked out')
        if type(key) is not str or not key or engine.index!=0 or engine.inputs is None:
            raise ValueError('fresh actual scene session required')
        hit=key in self.slots
        try:
            if not hit:
                if len(self.slots)>=self.capacity:
                    oldest=next(iter(self.slots))
                    self._destroy_slot(self.slots[oldest])
                    del self.slots[oldest]
                    self.evictions+=1
                self.slots[key]=dict(pool=ChunkCopyPool(),rotaries=GraphBank(),
                    denoise=None,refresh=None,bound_session=None,student=engine.model)
            slot=self.slots[key]
            if slot['student'] is not engine.model:
                raise RuntimeError('resident model identity changed')
            # Strong references keep capture-external constants alive. Reuse
            # requires equal actual rotary values, not only a matching shape.
            slot['rotaries'].pin_rotaries(rotaries)
            torch.cuda.synchronize(self.device)
            self.slots.move_to_end(key)
            self.sessions+=1
            self.active=dict(key=key,slot=slot,engine=engine,inputs=engine.inputs,
                cache=engine.cache,session=self.sessions,next_chunk=5)
            self.emit('resident_full_session_begin',shape_key=key,shape_hit=hit,
                captured_hit=slot['denoise'] is not None,session=self.sessions)
        except BaseException:
            self.failed=True
            raise

    def _active(self,engine,index):
        a=self.active
        if self.closed or self.failed or a is None or a['engine'] is not engine:
            raise RuntimeError('no owned resident session')
        if engine.inputs is not a['inputs'] or engine.cache is not a['cache']:
            raise RuntimeError('scene/cache owner changed during resident session')
        if type(index) is not int or index!=a['next_chunk'] or index<5:
            raise RuntimeError('resident chunk skipped or replayed')
        return a,a['slot']

    @staticmethod
    def _private_cache(cache):
        from ..attention.history import PackedHistoryCache
        if type(cache) is not PackedHistoryCache:
            raise TypeError('exact actual packed history required')
        private=PackedHistoryCache()
        private.layers={layer:list(rows) for layer,rows in cache.layers.items()}
        private.packed_history_by_layer=dict(cache.packed_history_by_layer)
        return private

    def _capture(self,slot,engine,compiler,noisy):
        from solar.backends.minimax_h3 import sgf_rollout
        pool=slot['pool']
        owner=SimpleNamespace(pool=pool,mode='plain')
        # Cache-restored compiler warm-up and capture use private publication
        # dictionaries. They must not commit a provisional latent to the live
        # session or advance its RNG/camera/index.
        step=engine.schedule.timesteps[0]
        warm=sgf_rollout.h3_student_forward(engine.model,engine.inputs,noisy,step,
            chunk_index=5,cache=self._private_cache(engine.cache))
        sgf_rollout.h3_student_forward(engine.model,engine.inputs,warm,warm.new_tensor(1.),
            chunk_index=5,cache=self._private_cache(engine.cache),commit_cache=True)
        pool.begin_chunk()
        try:
            slot['denoise']=capture_denoise(engine=engine,compiler=compiler,backend_owner=owner,
                cache=self._private_cache(engine.cache),noisy=noisy,timestep=step)
        finally:pool.end_chunk()
        slot['refresh']=capture_refresh(engine=engine,compiler=compiler,backend_owner=owner,
            denoise_bundle=slot['denoise'],cache=self._private_cache(engine.cache),
            noisy=slot['denoise']['first_output'],timestep=noisy.new_tensor(1.))
        self.emit('resident_full_shape_captured',shape_key=self.active['key'],
            denoise_captures=slot['denoise']['graph'].capture_count,
            refresh_captures=slot['refresh']['graph'].capture_count)

    def _bind(self,slot,engine):
        from .bank import rebind_conditions, pointers
        a=self.active
        if a['next_chunk']!=5 or slot['bound_session']==a['session']:
            raise RuntimeError('session binding is single-use and only before chunk 5')
        denoise,refresh=slot['denoise'],slot['refresh']
        history=denoise['history']
        if refresh['call'].history_owner is not history:
            raise RuntimeError('denoise/refresh no longer share KV storage')
        for bundle in (denoise,refresh):
            graph=bundle['graph']
            if graph.failed or getattr(graph,'closed',False) or graph.capture_count!=1 or graph.graph is None:
                raise RuntimeError('resident Graph is incomplete or failed')
        for layer in range(50):
            if tuple(row[0] for row in engine.cache.history(layer,5))!=tuple(range(5)):
                raise RuntimeError('new scene lacks its five actual history entries')
        writeback=refresh['call'].writeback
        if writeback.failed or writeback.phase!='captured':
            raise RuntimeError('resident refresh was not cleanly parked')
        before=(pointers(denoise['call']),pointers(refresh['call']))
        rebind_conditions(denoise['call'],engine.model,engine.inputs)
        rebind_conditions(refresh['call'],engine.model,engine.inputs)
        history.next_chunk=5
        denoise['call'].actual_chunk=4
        refresh['call'].actual_chunk=4
        writeback.global_cache=engine.cache
        writeback.next_chunk=5
        if before!=(pointers(denoise['call']),pointers(refresh['call'])):
            raise RuntimeError('Graph storage moved during scene rebinding')
        slot['bound_session']=a['session']
        self.emit('resident_full_scene_bound',shape_key=a['key'],session=a['session'],
            unchanged_storage=True,new_capture=False)
        # Staging cursor follows the completed canonical chunk 5.
        slot['denoise']['history'].next_chunk=6

    def step(self, *, engine, compiler, noisy):
        import torch
        from solar.backends.minimax_h3.torch_flow import scale_noise
        index=engine.index
        a,slot=self._active(engine,index)
        if torch.is_grad_enabled():raise RuntimeError('no-grad inference required')
        try:
            if slot['denoise'] is None:self._capture(slot,engine,compiler,noisy)
            if slot['bound_session']!=a['session']:self._bind(slot,engine)
            denoise,refresh=slot['denoise'],slot['refresh']
            counts=dict(compiler.backend_calls)
            replays=(denoise['graph'].replay_count,refresh['graph'].replay_count)
            pool=slot['pool'];pool.begin_chunk()
            try:
                stage(denoise,cache=engine.cache,index=index,views=engine.inputs.camera_viewmats,
                    intrinsics=engine.inputs.camera_K,pool=pool)
                current=noisy
                for ordinal,timestep in enumerate(engine.schedule.timesteps):
                    denoise['call'].load(actual_chunk=index,noisy=current,timestep=timestep)
                    x0=denoise['graph'].replay()
                    if not bool(torch.isfinite(x0).all()):raise FloatingPointError('nonfinite resident denoise')
                    if ordinal==3:final=x0.detach().clone()
                    else:current=scale_noise(x0,engine.random_like(x0),engine.schedule.timesteps[ordinal+1])
                replay_and_publish(refresh,index=index,noisy=final,timestep=noisy.new_tensor(1.))
            finally:pool.end_chunk()
            if dict(compiler.backend_calls)!=counts:
                raise RuntimeError('resident replay recompiled')
            if (denoise['graph'].replay_count-replays[0],refresh['graph'].replay_count-replays[1])!=(4,1):
                raise RuntimeError('resident denoise/refresh replay counts differ')
            if denoise['graph'].capture_count!=1 or refresh['graph'].capture_count!=1:
                raise RuntimeError('resident Graph recaptured')
            a['next_chunk']+=1
            return final
        except BaseException:
            self.failed=True
            raise

    def end_session(self,engine):
        import torch
        if self.active is None or self.active['engine'] is not engine or self.failed:
            raise RuntimeError('no clean resident session to park')
        a=self.active;slot=a['slot']
        if slot['pool'].active:raise RuntimeError('cannot park during chunk execution')
        if engine.index>=5 and engine.index!=a['next_chunk']:
            raise RuntimeError('session ended before current chunk publication completed')
        torch.cuda.synchronize(self.device)
        for name in ('denoise','refresh'):
            bundle=slot[name]
            if bundle is None:continue
            call=bundle['call']
            old=call.tensor_owner.inputs
            call.tensor_owner.inputs=SimpleNamespace(**{n:getattr(old,n) for n in
                ('latent_height','latent_width','keyframe_timestep','sp_enabled','text_tags')})
            bundle['bridge'].private.layers.clear()
            if hasattr(bundle['bridge'],'sources'):bundle['bridge'].sources.clear()
        if slot['refresh'] is not None:
            wb=slot['refresh']['call'].writeback
            if wb.failed or wb.phase!='captured':
                self.failed=True
                raise RuntimeError('refresh publication incomplete at session end')
            wb.global_cache=None
            slot['refresh'].pop('initial_published_cache',None)
        self.emit('resident_full_session_parked',shape_key=a['key'],session=a['session'],
            generated_steady_chunks=a['next_chunk']-5,graphs_retained=slot['denoise'] is not None)
        self.active=None

    def _destroy_slot(self,slot):
        import torch
        bundles=[slot[name] for name in ('denoise','refresh') if slot[name] is not None]
        close_full_student_graphs(bundles,synchronize=lambda:torch.cuda.synchronize(self.device),emit=self.emit)
        slot['pool'].close()
        slot['rotaries'].destroy()

    def report(self):
        return dict(sessions=self.sessions,evictions=self.evictions,failed=self.failed,
            active_key=None if self.active is None else self.active['key'],
            shapes={key:{name:None if slot[name] is None else dict(
                captures=slot[name]['graph'].capture_count,replays=slot[name]['graph'].replay_count)
                for name in ('denoise','refresh')} for key,slot in self.slots.items()},
            web_runtime_validated=False)

    def close(self):
        if self.closed:return
        for slot in self.slots.values():self._destroy_slot(slot)
        self.slots.clear();self.active=None;self.closed=True
