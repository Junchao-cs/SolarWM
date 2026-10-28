"""Canonical indexes, deterministic sampling, cameras, and shard transports."""

from .archive import RawSample, RawSampleReader, TarShardReader
from .index import (
    IndexInventory,
    IndexRow,
    ensure_disjoint,
    inventory,
    read_index,
    resolve_index_path,
    select_index_rows,
)
from .prefetch import ShardPrefetcher, build_shard_prefetcher, shard_prefetch_depth
from .sampling import CanonicalSampler, ReaderIdentity, SamplePlan, SamplingConfig
from .transport import GCSResolver, LocalResolver, resolver_from_config

__all__ = [
    "CanonicalSampler",
    "GCSResolver",
    "IndexInventory",
    "IndexRow",
    "LocalResolver",
    "RawSample",
    "RawSampleReader",
    "ReaderIdentity",
    "SamplePlan",
    "SamplingConfig",
    "ShardPrefetcher",
    "TarShardReader",
    "build_shard_prefetcher",
    "ensure_disjoint",
    "inventory",
    "read_index",
    "resolve_index_path",
    "resolver_from_config",
    "select_index_rows",
    "shard_prefetch_depth",
]
