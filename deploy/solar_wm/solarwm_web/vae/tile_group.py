"""Explicit subgroup ownership around the unchanged spatial tile decoder."""
import torch
import torch.distributed as dist
from .streaming import StreamingDecoder
from .tile_parallel import TileParallelDecoder


class GroupTileDecoder(TileParallelDecoder):
    def __init__(self, vae, *, group, **kwargs):
        self.group = group
        self.rank, self.world = dist.get_rank(group), dist.get_world_size(group)
        if self.world not in (7, 8):
            raise ValueError("seven or eight VAE ranks required")
        StreamingDecoder.__init__(self, vae, **kwargs)

    def _blend(self, tiles):
        count = max(hi - lo for lo, hi in self._parts)
        padded = tiles.new_zeros((count, *tiles.shape[1:]))
        padded[:tiles.shape[0]].copy_(tiles)
        gathered = [torch.empty_like(padded) for _ in self._parts] if self.rank == 0 else None
        dist.gather(padded, gather_list=gathered, dst=0, group=self.group)
        if self.rank:
            return tiles.new_empty((1, 3, self.ratio, 0, 0))
        ordered = torch.cat([p[:hi-lo] for p, (lo, hi) in zip(gathered, self._parts)], 0)
        return StreamingDecoder._blend(self, ordered)
