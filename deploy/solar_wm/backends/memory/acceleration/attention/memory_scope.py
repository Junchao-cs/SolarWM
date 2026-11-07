"""Owned attention/collective bindings and exact history layout normalization."""
from contextlib import ExitStack, contextmanager
from deploy.solar_wm.backends.memory.acceleration.attention import memory as attention
from deploy.solar_wm.backends.memory.acceleration.attention import memory_window as window
from deploy.solar_wm.backends.memory.acceleration.attention.helper_binding import helper_binding
from deploy.solar_wm.backends.memory.acceleration.graph.history_layout import spec
from deploy.solar_wm.backends.memory.acceleration.graph.history_layout import match_singleton


@contextmanager
def installed(base):
    from solar.backends.minimax_h3 import model, distributed as d
    from deploy.solar_wm.backends.memory.acceleration.attention.collectives import install, contiguous_packed_bounds
    from deploy.solar_wm.backends.memory.acceleration.attention.all_to_all import canonical_all_to_all
    from deploy.solar_wm.backends.memory.acceleration.attention.all_gather import canonical_all_gather
    from deploy.solar_wm.backends.memory.acceleration.attention.prope import canonical_prope_import
    from deploy.solar_wm.backends.memory.acceleration.attention.refiner import canonical_refiner_backend
    with ExitStack() as stack:
        old = d.contiguous_packed_bounds
        d.contiguous_packed_bounds = contiguous_packed_bounds
        stack.callback(setattr, d, 'contiguous_packed_bounds', old)
        stack.callback(install().close)
        stack.enter_context(canonical_all_to_all(enabled=True))
        stack.enter_context(canonical_all_gather(enabled=True))
        stack.enter_context(canonical_prope_import(enabled=True))
        stack.enter_context(canonical_refiner_backend(base.token_refiner, enabled=True, expected_backend='flash'))
        stack.enter_context(helper_binding(model, attention.segment))
        stack.enter_context(attention.installed())
        original = window.Window
        expected = {}

        class StableWindow(original):
            def __init__(self, source, rope):
                super().__init__(source, rope)
                for layer, rows in self.private.layers.items():
                    normalized = []
                    for slot, (index, key, value) in enumerate(rows):
                        values = []
                        for kind, tensor in enumerate((key, value)):
                            token = layer, slot, kind
                            if token not in expected:
                                expected[token] = spec(tensor)
                            values.append(match_singleton(tensor, expected[token]))
                        normalized.append((index, *values))
                    self.private.layers[layer] = normalized

        window.Window = StableWindow
        try:
            yield
        finally:
            changed = window.Window is not StableWindow
            window.Window = original
            if changed:
                raise RuntimeError('Memory window ownership changed')
