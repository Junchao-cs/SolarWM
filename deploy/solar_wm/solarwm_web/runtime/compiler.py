"""The deployed unpadded phase compiler without historical experiment paths.

This component does not load weights, change attention, install operator
profiles, or capture Graphs. Those owners must surround actual native calls.
Persisted files are observations of this build, not copied acceptance receipts.
"""
import hashlib
import json
from pathlib import Path

from ..acceleration.compile.compiler import Evidence, SharedCompiler, phase_key
from ..acceleration.compile.guard import GuardedCompilerMixin


PHASES = frozenset((chunk, refresh) for chunk in range(6) for refresh in (False, True))


class PhaseCompiler(GuardedCompilerMixin, SharedCompiler):
    """Share one full-model program per history/refresh phase across scenes."""

    def __init__(self, native, evidence):
        super().__init__(native, evidence, mode="inductor")
        self.frozen = False

    def __call__(self, **kwargs):
        phase = phase_key(kwargs["sgf_attention"])
        # Exactly the resident deployment's policy: a missing history regime
        # may compile during preparation; an existing one may never retrace.
        self.allow_compile = not self.frozen and phase not in self.entries
        return super().__call__(**kwargs)

    def freeze(self):
        if set(self.entries) != PHASES or set(self.backend_calls) != PHASES:
            raise RuntimeError("all twelve native denoise/refresh regimes must be prepared")
        if any(count != 1 for count in self.backend_calls.values()):
            raise RuntimeError("a phase retraced during preparation")
        self.frozen = True
        self.allow_compile = False

    def describe_build(self):
        """Read the actual emitted FX; does not assert numerical/quality parity."""
        if not self.frozen:
            raise RuntimeError("compiler preparation is incomplete")
        rows = []
        for chunk, refresh in sorted(PHASES):
            path = self.evidence.folder / (
                f"fx-r{self.evidence.rank}-{chunk}-{int(refresh)}-1.py")
            rows.append({
                "history_phase": chunk, "refresh": refresh,
                "backend_calls": self.backend_calls[(chunk, refresh)],
                "fx_file": path.name,
                "fx_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
        return {
            "schema": 1, "rank": self.evidence.rank,
            "status": "PREPARED_COMPILED_PHASES_NOT_QUALITY_ACCEPTANCE",
            "phases": rows, "prompt_padding": False,
            "cuda_graph_serialized": False,
            "compiler_disk_cache_hit": "not_measured",
            "numerical_verified": False, "performance_verified": False,
        }

    def save_build_description(self, path):
        data = self.describe_build()
        with Path(path).open("x", encoding="utf-8") as output:
            json.dump(data, output, indent=2)
            output.write("\n")
        return data


def create_compiler(native, *, evidence_directory, rank):
    """A fresh evidence directory is required even when reusing disk caches."""
    if type(rank) is not int or not 0 <= rank < 8:
        raise ValueError("one-node SP8 rank required")
    folder = Path(evidence_directory)
    folder.mkdir(parents=True, exist_ok=False)
    return PhaseCompiler(native, Evidence(folder, rank))
