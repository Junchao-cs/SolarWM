"""Local-window camera coordinates for Memory-conditioned generation.

Global trajectory/retrieval remain unchanged. One bounded set (anchor, selected
bank, W5 history, current) shares the old recent-window translation origin for
all four denoises and refresh. Nonlinear PRoPE makes this different from the
upstream global-origin policy; retrieval still uses the global trajectory.
"""
from contextlib import contextmanager

def participating_rows(index, selected):
    if type(index) is not int or index < 0:
        raise ValueError('nonnegative integer chunk required')
    selected = tuple(selected)
    limit = max(0, index - 5) * 5
    if len(selected) != min(5, limit) or len(set(selected)) != len(selected):
        raise ValueError('exact native bank extent and uniqueness required')
    if any((type(i) is not int or not 0 <= i < limit for i in selected)):
        raise ValueError('bank must precede recent window')
    (start, stop) = (1 + limit, 1 + (index + 1) * 5)
    rows = (0,) + tuple((i + 1 for i in selected)) + tuple(range(start, stop))
    if len(rows) > 36 or len(set(rows)) != len(rows):
        raise ValueError('bounded disjoint anchor/bank/window required')
    return (start, stop, rows)

@contextmanager
def local_memory_window(views, index, selected):
    import torch
    (start, stop, rows) = participating_rows(index, selected)
    if views.ndim != 4 or views.shape[0] != 1 or tuple(views.shape[-2:]) != (4, 4) or (views.dtype != torch.float32) or views.requires_grad or (views.shape[1] < stop):
        raise ValueError('complete B1 FP32 inference camera table required')
    if index <= 5:
        yield dict(active=False, rows=(), origin_row=None)
        return
    ids = torch.tensor(rows, device=views.device, dtype=torch.long)
    with torch.no_grad():
        chosen = views.index_select(1, ids)
        first = views[:, start]
        origin = -(first[:, :3, :3].transpose(-1, -2) @ first[:, :3, 3:4])
        original = chosen[..., :3, 3].clone()
        translation = original + (chosen[..., :3, :3] @ origin[:, None]).squeeze(-1)
        views[..., :3, 3].index_copy_(1, ids, translation)
    try:
        yield dict(active=True, rows=rows, origin_row=start)
    finally:
        with torch.no_grad():
            views[..., :3, 3].index_copy_(1, ids, original)

def select_global(forward, index):
    """Same native selector and inputs, evaluated before local camera mutation."""
    from solar.backends.minimax_h3.memory_retrieval import select_memory
    if index not in forward.selected:
        (ids, scores) = select_memory(forward.c2w, forward.K, index)
        if not forward.enabled:
            (ids, scores) = (ids[:0], scores[:0])
        forward.selected[index] = (ids, scores)
    (ids, _) = forward.selected[index]
    selected = tuple((int(i) for i in ids))
    participating_rows(index, selected)
    return selected
