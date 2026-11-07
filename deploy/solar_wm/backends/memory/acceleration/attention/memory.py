"""Scoped Memory attention backend; no changes to visibility or camera math."""
from contextlib import contextmanager
import ast
import inspect
import textwrap

def segment(q, k, v, control):
    import torch
    from flash_attn import flash_attn_func
    if q.shape[1] == 5 * control.layout.rows_per_video_frame:
        return torch.nn.functional.scaled_dot_product_attention(q.contiguous().transpose(1, 2), k.contiguous().transpose(1, 2), v.contiguous().transpose(1, 2), dropout_p=0.0, is_causal=False).transpose(1, 2)
    return flash_attn_func(q.contiguous(), k.contiguous(), v.contiguous(), dropout_p=0.0, causal=False)

def projected_function(original, module):
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    if not isinstance(function, ast.FunctionDef) or function.name != '_sgf_attention_forward':
        raise RuntimeError('unexpected native attention function')
    segments = [n for n in ast.walk(function) if isinstance(n, ast.FunctionDef) and n.name == 'segment_attention']
    if len(segments) != 1 or len(segments[0].body) != 2:
        raise RuntimeError('native segmented attention contract changed')
    segments[0].body = ast.parse('return _memory_segment(q, k, v, control)').body
    function.decorator_list = []
    namespace = dict(vars(module), _memory_segment=segment)
    exec(compile(ast.fix_missing_locations(tree), __file__ + '#native-projection', 'exec'), namespace)
    return namespace[function.name]

@contextmanager
def installed():
    from torch.nn.attention import sdpa_kernel, SDPBackend
    from solar.backends.minimax_h3 import model
    owner = model.SolarMiniMaxH3Transformer3DModel
    prior = owner.__dict__['_sgf_attention_forward']
    if not isinstance(prior, staticmethod):
        raise RuntimeError('native staticmethod required')
    candidate = projected_function(prior.__func__, model)
    owner._sgf_attention_forward = staticmethod(candidate)
    try:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            yield
    finally:
        changed = owner.__dict__['_sgf_attention_forward'].__func__ is not candidate
        owner._sgf_attention_forward = prior
        if changed:
            raise RuntimeError('scoped attention owner changed')
