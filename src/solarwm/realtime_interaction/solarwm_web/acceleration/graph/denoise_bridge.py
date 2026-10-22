"""Denoise capture and staging with the same KV pool used by refresh."""

from .bridge import StableDynamicEntry, canonical_rows, installed_full_student_bridge


def transition(current, requested):
    if type(current) is not int or type(requested) is not int or current < 5:
        raise ValueError('actual steady index required')
    if requested not in (current, current+1):
        raise ValueError('cannot skip or rewind live history')
    return requested+1


def stage(bundle, *, cache, index, views, intrinsics, pool):
    import torch
    from ..compile.window import singleton_stride_view
    from .buffers import plan
    history, bridge = bundle['history'], bundle['bridge']
    if torch.is_grad_enabled() or not pool.active or history.poisoned:
        raise RuntimeError('owned no-grad chunk required')
    next_chunk = transition(history.next_chunk-1, index)
    mapping = plan(index)
    pairs = history._source_pairs(cache, mapping)
    history._check_camera(views, intrinsics, mapping.camera_source[-1]+1)
    jobs=[]
    for layer in range(50):
        for side in range(2):
            source=singleton_stride_view(pairs[layer][side])
            target=history.buffers[layer][side]
            if pool.target((layer,side),source) is not target:
                raise RuntimeError('full-student and refresh no longer share owned KV storage')
            pool.stage((layer,side),source,target,jobs)
    # The native five-history-plus-current-camera projection is exactly
    # contiguous here. Copy it and the anchor without 60 scalar slice jobs.
    if mapping.camera_destination != tuple(range(1,31)) or mapping.camera_source != tuple(range(mapping.camera_source[0],mapping.camera_source[0]+30)):
        raise RuntimeError('native camera map changed')
    start,stop=mapping.camera_source[0],mapping.camera_source[-1]+1
    for source,target in ((views,history.viewmats),(intrinsics,history.intrinsics)):
        if source.device!=target.device or source.dtype!=target.dtype:
            raise RuntimeError('camera specialization changed')
        jobs.extend(((target[:,:1],source[:,:1]),(target[:,1:31],source[:,start:stop])))
    history.ready=False
    try:
        targets,sources=zip(*jobs)
        torch._foreach_copy_(list(targets),list(sources))
        # Preserve actual raw-history references for this request. Captured
        # model reads the same packed buffers, not stale raw entry objects.
        bridge.private.layers={layer:[(i,singleton_stride_view(k),singleton_stride_view(v))
            for i,k,v in canonical_rows(cache.history(layer,index),index)] for layer in range(50)}
        bridge.sources=[t for rows in bridge.private.layers.values() for row in rows for t in row[1:]]
    except BaseException:
        history.poisoned=True
        raise
    history.next_chunk=next_chunk
    history.ready=True
    bundle['stage_calls']+=1


def capture(*,engine,compiler,backend_owner,cache,noisy,timestep):
    from ..compile.window import singleton_stride_view
    from .buffers import StableSteadyInputs
    from .inputs import FullStudentTensorInputs, FullStudentReadOnlyCall
    from .denoise import NativeAutocastManualFullStudentGraph
    pool=backend_owner.pool
    if not pool.active:
        raise RuntimeError('capture staging requires owned copy transaction')
    history=StableSteadyInputs(cache,5,engine.inputs.camera_viewmats,engine.inputs.camera_K)
    # Reuse the already captured backend-refresh targets, not a second
    # full-model packed copy. No captured pointer is replaced later.
    history.buffers={layer:tuple(pool.target((layer,side),singleton_stride_view(cache.packed_history_by_layer[layer][1][side]))
        for side in range(2)) for layer in range(50)}
    tensors=FullStudentTensorInputs(inputs=engine.inputs,noisy=noisy,timestep=timestep)
    call=FullStudentReadOnlyCall(student=engine.model,tensor_owner=tensors,history_owner=history,noisy=noisy,timestep=timestep)
    bridge=StableDynamicEntry(compiler,history,cache,5)
    graph=NativeAutocastManualFullStudentGraph(call)
    bundle=dict(history=history,tensors=tensors,call=call,bridge=bridge,graph=graph,stage_calls=0)
    stage(bundle,cache=cache,index=5,views=engine.inputs.camera_viewmats,intrinsics=engine.inputs.camera_K,pool=pool)
    with installed_full_student_bridge(engine.base,backend_owner,bridge):
        bundle['first_output']=graph.capture()
    if graph.capture_reference_max_abs != 0:
        raise RuntimeError('full-student capture/reference output mismatch')
    return bundle
