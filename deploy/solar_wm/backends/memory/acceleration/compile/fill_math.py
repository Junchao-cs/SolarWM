from dataclasses import dataclass
from deploy.solar_wm.backends.memory.acceleration.attention.memory_window import PreparedMemoryAttention

@dataclass(frozen=True)
class ExportControl(PreparedMemoryAttention):
    outputs: object = None

    def attend(self, q, k, v, *, layer, apply_rotary, attend):
        prefix = self.layout.sequence_length - 5 * self.layout.rows_per_video_frame
        if layer != len(self.outputs):
            raise RuntimeError('ordered full50 prefix export required')
        self.outputs.append((k[:, :prefix], v[:, :prefix]))
        return super().attend(q, k, v, layer=layer, apply_rotary=apply_rotary, attend=attend)

class Core:

    def __init__(self, native):
        self.native = native

    def __call__(self, kwargs):
        source = kwargs['sgf_attention']
        outputs = []
        control = ExportControl(source.layout, source.viewmats, source.intrinsics, source.bank_indices, source.chunk_index, source.cache, source.commit_cache, source.prefix_rotary, source.window_rotary, outputs)
        (video, audio) = self.native(**dict(kwargs, sgf_attention=control))
        if len(outputs) != 50:
            raise RuntimeError('all 50 prefix layers must be returned')
        return (video, audio, tuple(outputs))
