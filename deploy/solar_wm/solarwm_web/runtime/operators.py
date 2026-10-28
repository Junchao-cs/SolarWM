"""The resident operator recipe with explicit, external build-profile paths.

Generated wrappers, pinned choices and validation results belong to the exact
runtime build that produced them. They are not included in the source release.
No historical acceptance is supplied implicitly and no missing pin falls back.
"""
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..acceleration.operators.norm import PhaseNormPins
from ..acceleration.compile.cache import MetadataRebuildPins
from ..acceleration.operators.fma import PhaseFma
from ..acceleration.operators.qkv import ScopedQkvDispatch
from ..acceleration.operators.layout import SpLayout
from ..acceleration.operators.fma_layout import restore_refresh_binaries
from ..acceleration.operators.resident import ResidentFmaLayoutDispatch, restore_layout_binaries
from ..acceleration.operators.dispatch import install as bind_rebuilt_wrappers


@dataclass(frozen=True)
class OperatorProfile:
    seed: Path
    alternative: Path
    qkv_manifest: Path
    layout_manifest: Path
    layout_result: Path
    denoise_manifest: Path
    refresh_manifest: Path
    denoise_result: Path
    refresh_result: Path
    refresh_prefix_provenance: Path

    def check_paths(self):
        for name, value in vars(self).items():
            path = Path(value)
            if not path.is_absolute():
                raise ValueError("operator profile paths must be explicit and absolute: " + name)
            if name in ("seed", "alternative"):
                if not path.is_dir():
                    raise FileNotFoundError(path)
            elif not path.is_file():
                raise FileNotFoundError(path)




class MetadataRebuildPhasePins(MetadataRebuildPins, PhaseNormPins):
    pass


class ResidentPhaseOptimizations:
    """Same QKV/layout/FMA/pin ordering as the deployed steady/refresh owner."""

    def __init__(self, *, profile, rank, device, emit, inductor_cache,
                 triton_cache, rebuild_metadata=False):
        if type(profile) is not OperatorProfile:
            raise TypeError("explicit OperatorProfile required")
        profile.check_paths()
        pins = MetadataRebuildPhasePins if rebuild_metadata else PhaseNormPins
        self.pins = pins(rank, profile.seed, profile.alternative, emit)
        self.pins.set_variant("B")
        self.qkv = ScopedQkvDispatch(profile.qkv_manifest, rank, emit)
        self.qkv.set_mode("overlap")
        self.layout = SpLayout(profile.layout_manifest, rank, device, emit)
        restore_layout_binaries(self.layout, profile.layout_result)
        self.layout.set_mode("replace")
        self.denoise = PhaseFma(profile.denoise_manifest, rank, device, emit)
        self.refresh = PhaseFma(profile.refresh_manifest, rank, device, emit)
        self.refresh.prefix_binary_provenance = profile.refresh_prefix_provenance
        self.denoise.restore_actual_denoise(profile.denoise_result)
        restore_refresh_binaries(self.refresh, profile.refresh_result)
        for owner in (self.denoise, self.refresh):
            owner.bind_wrapper(inductor_cache)
            owner.set_rounding(False)
            owner.set_mode("replace")
        self.combined = ResidentFmaLayoutDispatch(
            self.denoise, self.refresh, self.layout, self.qkv.record, triton_cache)
        self.active = False
        self.rebuilt_wrapper_bindings = bind_rebuilt_wrappers(
            self, cache=inductor_cache, triton_cache=triton_cache, rank=rank, emit=emit)

    def bind_request(self, inputs):
        if self.active:
            raise RuntimeError("cannot rebind during a live DiT scope")
        layout = inputs.layout("rollout")
        self.layout.bind_request(
            int(inputs.prompt.shape[1]), int(layout.condition_indices.numel()),
            layout.sequence_length)

    @contextmanager
    def installed(self):
        if self.active:
            raise RuntimeError("resident phase scope is not reentrant")
        self.active = True
        try:
            with self.pins, self.qkv, self.combined.installed(self.pins):
                yield self
        finally:
            self.active = False

    def report(self):
        return dict(pins=self.pins.report(), qkv=self.qkv.report(),
                    layout=self.layout.report(), denoise_fma=self.denoise.report(),
                    refresh_fma=self.refresh.report(), combined=self.combined.report(),
                    active=self.active, release_runtime_validated=False)
