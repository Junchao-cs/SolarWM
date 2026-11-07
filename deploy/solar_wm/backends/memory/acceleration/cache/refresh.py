"""Publish borrowed compiled outputs once through the existing real commit.

Avoid a redundant private-cache clone: the public native commit still clones
each produced raw K/V exactly as before. Borrowed outputs must live through
Window.finish; no pointer can be reused by another replay before that finish.
"""

def publish(window, raw):
    if not window.source.commit_cache or len(raw) != 50:
        raise RuntimeError('full refresh outputs required')
    checked = []
    for (layer, (k, v)) in enumerate(raw):
        rows = window.private.history(layer, 5)
        if len(rows) != 5:
            raise RuntimeError('normalized raw history required')
        (_, hk, hv) = rows[-1]
        for (actual, expected) in ((k, hk), (v, hv)):
            if (actual.shape, actual.dtype, actual.device) != (expected.shape, expected.dtype, expected.device):
                raise RuntimeError('raw current KV ABI changed')
        checked.append((layer, rows, k, v))
    for (layer, rows, k, v) in checked:
        window.private.layers[layer] = [*rows[1:], (5, k, v)]
