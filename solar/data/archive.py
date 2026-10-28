"""Random access to indexed members in uncompressed WebDataset tar shards."""

from __future__ import annotations

import json
import tarfile
import threading
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from solar.errors import DataContractError

from .index import IndexRow, validate_relative_key
from .sampling import SamplePlan
from .transport import ShardResolver


class TarShardReader:
    """Small per-worker LRU of tar handles; never extracts paths to disk."""

    def __init__(self, resolver: ShardResolver, *, max_open: int = 4) -> None:
        if max_open < 1:
            raise DataContractError("tar max_open must be positive")
        self.resolver = resolver
        self.max_open = max_open
        self._handles: OrderedDict[Path, tarfile.TarFile] = OrderedDict()
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            while self._handles:
                _, handle = self._handles.popitem(last=False)
                handle.close()

    def __enter__(self) -> TarShardReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _handle(self, row: IndexRow) -> tarfile.TarFile:
        path = self.resolver.resolve(row).resolve()
        handle = self._handles.pop(path, None)
        if handle is None:
            try:
                # Published shards are uncompressed tars. Strict mode prevents a
                # transport-specific decompressor from hiding format drift.
                handle = tarfile.open(path, "r:")  # noqa: SIM115 - retained in the LRU
            except (OSError, tarfile.TarError) as exc:
                raise DataContractError(f"cannot open WDS shard {path}: {exc}") from exc
        self._handles[path] = handle
        while len(self._handles) > self.max_open:
            _, old = self._handles.popitem(last=False)
            old.close()
        return handle

    def read(self, row: IndexRow, member: Any) -> bytes:
        member_name = validate_relative_key(member, field="tar member")
        with self._lock:
            handle = self._handle(row)
            try:
                entry = handle.getmember(member_name)
                extracted = handle.extractfile(entry)
            except (KeyError, OSError, tarfile.TarError) as exc:
                raise DataContractError(
                    f"missing member {member_name!r} for sample {row.sample_id!r}"
                ) from exc
            if extracted is None:
                raise DataContractError(
                    f"member {member_name!r} for sample {row.sample_id!r} is not a file"
                )
            return extracted.read()


@dataclass(frozen=True)
class RawSample:
    """Transport-neutral raw payload handed to a model-family codec."""

    plan: SamplePlan
    index_values: Mapping[str, Any]
    caption: str
    scene: str
    manifest: Mapping[str, Any]
    members: Mapping[str, bytes]


class RawSampleReader:
    """Materialize bytes and metadata without performing model-specific encoding."""

    def __init__(
        self,
        rows: Sequence[IndexRow],
        shards: TarShardReader,
        *,
        member_fields: Sequence[str] = ("video_member", "camera_member"),
    ) -> None:
        self.rows = tuple(rows)
        self.shards = shards
        self.member_fields = tuple(member_fields)

    def materialize(self, plan: SamplePlan) -> RawSample:
        try:
            row = self.rows[plan.row_ordinal]
        except IndexError as exc:
            raise DataContractError(f"plan row ordinal is invalid: {plan.row_ordinal}") from exc
        if row.sample_id != plan.sample_id or row.key != plan.key or row.shard != plan.shard:
            raise DataContractError(f"sample plan identity drift for {plan.sample_id!r}")

        manifest = row.values.get("manifest")
        if manifest is None:
            name = row.values.get("manifest_member")
            if name is None:
                raise DataContractError(f"sample {row.sample_id!r} lacks manifest/manifest_member")
            try:
                manifest = json.loads(self.shards.read(row, name))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise DataContractError(
                    f"sample {row.sample_id!r} has an invalid manifest member"
                ) from exc
        if not isinstance(manifest, Mapping):
            raise DataContractError(f"sample {row.sample_id!r} manifest is not an object")

        members: dict[str, bytes] = {}
        for field_name in self.member_fields:
            member_name = row.values.get(field_name)
            if member_name is None:
                raise DataContractError(f"sample {row.sample_id!r} lacks required {field_name}")
            members[field_name] = self.shards.read(row, member_name)

        metadata = manifest.get("metadata", {})
        prompt = manifest.get("prompt", {})
        scene = (
            metadata.get("scene", row.values.get("dataset", row.key))
            if isinstance(metadata, Mapping)
            else row.values.get("dataset", row.key)
        )
        caption = row.values.get("caption") or (
            prompt.get("text", "") if isinstance(prompt, Mapping) else ""
        )
        return RawSample(
            plan=plan,
            index_values=dict(row.values),
            caption=str(caption),
            scene=str(scene),
            manifest=dict(manifest),
            members=members,
        )
