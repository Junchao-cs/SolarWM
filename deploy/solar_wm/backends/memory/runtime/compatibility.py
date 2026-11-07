"""Check the required upstream API without pinning source-file contents.

This is an early interface check, not a numerical compatibility guarantee.
Runtime tensor shapes, phase ordering and cache ownership are checked where used.
"""
import ast
import importlib.util
from pathlib import Path

REQUIRED = {
    "model": ("SolarMiniMaxH3Transformer3DModel",),
    "memory_inference": ("attach_memory", "validate_memory_inference"),
    "memory_stage2_rollout": ("Stage2MemoryForward",),
    "memory_kv_attention": ("MemoryKVAttention",),
    "memory_layout": ("memory_rollout_layout",),
    "sgf_rollout": ("H3SGFInputs",),
    "torch_prope": ("_project", "_invert_se3", "_prepare_suffix_transforms"),
}


def verify(repository=None):
    if repository is None:
        spec = importlib.util.find_spec("solar")
        if spec is None or not spec.origin:
            raise RuntimeError("install SolarWM or run this folder from its checkout")
        repository = Path(spec.origin).resolve().parent.parent
    root = Path(repository).resolve(strict=True)
    backend = root / "solar" / "backends" / "minimax_h3"
    for module, names in REQUIRED.items():
        path = backend / (module + ".py")
        tree = ast.parse(path.read_text(), filename=str(path))
        defined = {node.name for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        missing = set(names) - defined
        if missing:
            raise RuntimeError("missing H3 Memory interface in " + module + ": " + ", ".join(sorted(missing)))
    return {"required_interfaces_present": True, "model_constructed": False}
