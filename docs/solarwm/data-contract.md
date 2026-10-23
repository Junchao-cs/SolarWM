# Data contract

## Indexes and sample order

Published indexes are ordered JSONL or JSONL.GZ. Each row has a stable
`sample_id`, a relative POSIX `shard` key, the declared shard size, and the
member names required by its format. The index is authoritative: readers use
its rows rather than scanning tar files to discover samples.

`epoch_repeats` creates virtual occurrences without duplicating payloads.
Occurrences are ordered by index row and then repeat ordinal. With the same
index, seed, epoch, distributed topology, and worker count, local and
object-store transports produce the same logical sample sequence.

## Paths and storage

Paths inside public indexes must be relative and use `/` separators. The
runtime shard root may be a local directory or an object-store URI.
`data.index_root` points to the local release directory containing the relative
`train_index`, `index`, or `test_index` path named by the selected config.
`data.transport.root` is the base directory or URI used to resolve each
relative `shard` key.

For a local all-in-one tree, `data.index_root` may be omitted and the local
transport root is used. Object-store streaming requires a local
`data.index_root` for the recipe controls. The same indexes and shard-relative
paths work in both modes.

## Read-time validation

Runtime shard identity consists of a relative path and positive byte count.
Local reads check that shard paths remain under the configured root and that
their sizes match the index. Published digests are audit metadata and are not
rehash requirements at training start.

Preencoded readers validate the serialized format, tensor schema, shapes,
dtypes, finite values, and sample/camera semantics. Small semantic contract
fingerprints may also be checked without rereading every payload byte.

## Camera convention

Camera arrays are authoritative `c2w` unless an index schema explicitly says
otherwise. Camera translation uses its stored scale. `logd4` is a
model-conditioning transform applied after first-frame rebasing and never
rewrites stored camera tensors.
