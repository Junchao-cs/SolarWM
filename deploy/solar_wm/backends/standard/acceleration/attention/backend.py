"""Use SDPA for video attention; preserve native conditioning attention and dtype."""

from __future__ import annotations
from ..compatibility import expected_digest
import ast
import hashlib
import inspect
import sys
import textwrap
from contextlib import contextmanager
from pathlib import Path


MODEL_SHA = expected_digest("model")
FLASH_SHA = expected_digest("flash_attention")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


is_sequence_parallel_enabled = None
get_sp_size = None
sequence_model_parallel_all_to_all_4D = None
_apply_rotary_emb = None


def _sgf_attention_forward(attn, query, key, value, control):
    from solar.backends.minimax_h3.sgf_attention import sgf_attention as attend_sgf

    sp_enabled = is_sequence_parallel_enabled()
    if sp_enabled:
        if attn.heads % get_sp_size():
            raise ValueError("H3 SGF heads must be divisible by SP size")
        query = sequence_model_parallel_all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = sequence_model_parallel_all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = sequence_model_parallel_all_to_all_4D(value, scatter_dim=2, gather_dim=1)

    def segment_attention(q, k, v):
        from flash_attn import flash_attn_func

        if control.mode == "rollout" and q.shape[1] == 5 * int(control.layout.rows_per_video_frame):
            from torch.nn.functional import scaled_dot_product_attention

            return scaled_dot_product_attention(
                q.contiguous().transpose(1, 2),
                k.contiguous().transpose(1, 2),
                v.contiguous().transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)
        return flash_attn_func(
            q.contiguous(), k.contiguous(), v.contiguous(), dropout_p=0.0, causal=False
        )

    attended = attend_sgf(
        query,
        key,
        value,
        control=control,
        layer=attn.h3_layer_index,
        apply_rotary=_apply_rotary_emb,
        attend=segment_attention,
    )
    if sp_enabled:
        attended = sequence_model_parallel_all_to_all_4D(attended, scatter_dim=1, gather_dim=2)
    attended = attended.flatten(2, 3).type_as(query)
    return attn.to_out[1](attn.to_out[0](attended))


_VIDEO_BRANCH = """
if control.mode == "rollout" and q.shape[1] == 5 * int(control.layout.rows_per_video_frame):
    from torch.nn.functional import scaled_dot_product_attention
    return scaled_dot_product_attention(
        q.contiguous().transpose(1, 2),
        k.contiguous().transpose(1, 2),
        v.contiguous().transpose(1, 2),
        dropout_p=0.0,
        is_causal=False,
    ).transpose(1, 2)
"""


def projected_source_matches(original_source: str, candidate_source: str) -> bool:
    original = ast.parse(textwrap.dedent(original_source))
    candidate = ast.parse(textwrap.dedent(candidate_source))
    if len(original.body) != 1 or len(candidate.body) != 1:
        raise RuntimeError("H3 attention source projection is not one function")
    old, new = original.body[0], candidate.body[0]
    if not isinstance(old, ast.FunctionDef) or not isinstance(new, ast.FunctionDef):
        raise RuntimeError("H3 attention source projection is not a function")
    if (len(old.decorator_list) != 1 or not isinstance(old.decorator_list[0], ast.Name)
            or old.decorator_list[0].id != "staticmethod" or new.decorator_list):
        raise RuntimeError("H3 staticmethod ownership differs")
    old.decorator_list = []
    source_import, relocated = old.body[0], new.body[0]
    if not (isinstance(source_import, ast.ImportFrom)
            and source_import.module == "sgf_attention" and source_import.level == 1
            and isinstance(relocated, ast.ImportFrom)
            and relocated.module == "solar.backends.minimax_h3.sgf_attention"
            and relocated.level == 0 and len(source_import.names) == len(relocated.names) == 1
            and ast.dump(source_import.names[0], include_attributes=False)
            == ast.dump(relocated.names[0], include_attributes=False)):
        raise RuntimeError("H3 package import relocation differs")
    relocated.module, relocated.level = source_import.module, source_import.level
    segments = [node for node in ast.walk(old) if isinstance(node, ast.FunctionDef)
                and node.name == "segment_attention"]
    if len(segments) != 1 or len(segments[0].body) != 2:
        raise RuntimeError("original H3 segment attention body differs")
    segments[0].body.insert(1, ast.parse(textwrap.dedent(_VIDEO_BRANCH)).body[0])
    return ast.dump(original, include_attributes=False) == ast.dump(candidate, include_attributes=False)


@contextmanager
def installed_candidate(model_module: object, *, flash_source: Path):
    if sha(Path(model_module.__file__)) != MODEL_SHA or sha(flash_source) != FLASH_SHA:
        raise RuntimeError("original H3 or installed FlashAttention source changed")
    owner = model_module.SolarMiniMaxH3Transformer3DModel
    descriptor = owner.__dict__.get("_sgf_attention_forward")
    if not isinstance(descriptor, staticmethod):
        raise RuntimeError("original H3 SGF staticmethod owner differs")
    original = descriptor.__func__
    if not projected_source_matches(inspect.getsource(original),
                                    inspect.getsource(_sgf_attention_forward)):
        raise RuntimeError("candidate differs beyond video-only SDPA branch")
    global is_sequence_parallel_enabled, get_sp_size
    global sequence_model_parallel_all_to_all_4D, _apply_rotary_emb
    if any(value is not None for value in (is_sequence_parallel_enabled, get_sp_size,
                                          sequence_model_parallel_all_to_all_4D,
                                          _apply_rotary_emb)):
        raise RuntimeError("candidate module already owns a native scope")
    is_sequence_parallel_enabled = model_module.is_sequence_parallel_enabled
    get_sp_size = model_module.get_sp_size
    sequence_model_parallel_all_to_all_4D = model_module.sequence_model_parallel_all_to_all_4D
    _apply_rotary_emb = model_module._apply_rotary_emb
    owner._sgf_attention_forward = staticmethod(_sgf_attention_forward)
    try:
        yield
    finally:
        current = owner.__dict__.get("_sgf_attention_forward")
        changed = not isinstance(current, staticmethod) or current.__func__ is not _sgf_attention_forward
        owner._sgf_attention_forward = descriptor
        is_sequence_parallel_enabled = None
        get_sp_size = None
        sequence_model_parallel_all_to_all_4D = None
        _apply_rotary_emb = None
        if changed:
            primary = sys.exc_info()[1]
            if primary is None:
                raise RuntimeError("H3 candidate route changed during scoped rollout")
            if hasattr(primary, "add_note"):
                primary.add_note("H3 candidate route also changed during scoped rollout")
