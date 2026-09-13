from __future__ import annotations

import base64
import fcntl
import hashlib
import http.client
import io
import json
import os
import tarfile
import threading
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from solarwm.data.archive import RawSampleReader, TarShardReader
from solarwm.data.index import IndexRow
from solarwm.data.sampling import SamplePlan
from solarwm.data.transport import (
    GCSDownloadTimeout,
    GCSResolver,
    GCSRestDownloader,
    LocalResolver,
    join_root,
)
from solarwm.errors import DataContractError


def _md5(value: bytes) -> str:
    digest = hashlib.md5(value, usedforsecurity=False).digest()
    return base64.b64encode(digest).decode()


def _digest(value: bytes) -> str:
    return hashlib.blake2s(value).hexdigest()


class FakeDownloader:
    def __init__(self, value: bytes) -> None:
        self.value = value
        self.calls: list[str] = []

    def download(self, uri: str, generation: str, output) -> None:
        del generation
        self.calls.append(uri)
        output.write(self.value)


class FakeStreamingDownloader(FakeDownloader):
    def download(self, uri: str, generation: str, output) -> int:
        super().download(uri, generation, output)
        return len(self.value)


def test_gcs_rest_download_uses_current_object_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls: list[str] = []

    def open_url(request, *, timeout: float):
        assert timeout == 120.0
        urls.append(request.full_url)
        return io.BytesIO(b"payload")

    monkeypatch.setattr("solarwm.data.transport.urllib.request.urlopen", open_url)
    output = io.BytesIO()
    downloader = GCSRestDownloader(token=lambda: "token", max_attempts=1)

    assert downloader.download("gs://bucket/path/object.tar", "stale-generation", output) == 7
    assert urls == [
        "https://storage.googleapis.com/download/storage/v1/b/bucket/o/path%2Fobject.tar?alt=media"
    ]


def test_gcs_rest_download_has_a_total_time_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    opens = 0

    class SlowResponse(io.BytesIO):
        def read1(self, _size: int = -1) -> bytes:
            now[0] += 0.6
            return b"x"

    def open_url(_request, *, timeout: float):
        nonlocal opens
        opens += 1
        assert timeout == 1.0
        return SlowResponse()

    monkeypatch.setattr("solarwm.data.transport.time.monotonic", lambda: now[0])
    monkeypatch.setattr("solarwm.data.transport.urllib.request.urlopen", open_url)
    downloader = GCSRestDownloader(
        token=lambda: "token",
        max_elapsed_seconds=1.0,
        max_attempts=12,
    )

    with pytest.raises(GCSDownloadTimeout, match="exceeded 1s"):
        downloader.download("gs://bucket/path/object.tar", "", io.BytesIO())

    assert opens == 1


@pytest.mark.parametrize(
    "error",
    [
        urllib.error.HTTPError("https://example", 429, "busy", {}, None),
        http.client.IncompleteRead(b"partial", 10),
    ],
)
def test_gcs_rest_download_classifies_transient_retry_exhaustion_as_timeout(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    monkeypatch.setattr(
        "solarwm.data.transport.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    downloader = GCSRestDownloader(token=lambda: "token", max_attempts=1)

    with pytest.raises(GCSDownloadTimeout, match="retry budget"):
        downloader.download("gs://bucket/path/object.tar", "", io.BytesIO())


def _row(**extra: object) -> IndexRow:
    values: dict[str, object] = {
        "sample_id": "sample-a",
        "key": "sample-key-a",
        "shard": "corpus/shards/000001.tar",
        "num_frames": 100,
    }
    values.update(extra)
    return IndexRow.from_mapping(0, values)


def test_join_root_preserves_relative_key() -> None:
    assert join_root("gs://bucket/prefix", "a/b.tar") == "gs://bucket/prefix/a/b.tar"
    with pytest.raises(DataContractError, match="relative"):
        join_root("gs://bucket", "/private/b.tar")


def test_local_and_cached_gcs_resolve_identical_bytes(tmp_path: Path) -> None:
    payload = b"immutable-shard-bytes"
    local_root = tmp_path / "local"
    local_path = local_root / "corpus/shards/000001.tar"
    local_path.parent.mkdir(parents=True)
    local_path.write_bytes(payload)
    row = _row(
        shard_generation="7",
        shard_size=len(payload),
        shard_md5_b64=_md5(payload),
        shard_digest=_digest(payload),
    )

    local = LocalResolver(local_root).resolve(row)
    downloader = FakeDownloader(payload)
    gcs = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=downloader,
    )
    streamed = gcs.resolve(row)
    cached = gcs.resolve(row)

    assert local.read_bytes() == streamed.read_bytes() == cached.read_bytes()
    assert downloader.calls == ["gs://example-root/corpus/shards/000001.tar"]


def test_local_resolver_uses_size_without_binding_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = b"immutable-local"
    mutated = b"CORRUPTED-local"
    assert len(mutated) == len(original)
    root = tmp_path / "local"
    target = root / "corpus/shards/000001.tar"
    target.parent.mkdir(parents=True)
    target.write_bytes(original)
    row = _row(shard_size=len(original), shard_digest=_digest(original))
    resolver = LocalResolver(root)

    original_open = Path.open

    def reject_payload_open(path: Path, *args, **kwargs):
        if path == target:
            pytest.fail("local resolution reread the shard payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_payload_open)
    assert resolver.resolve(row) == target
    verified = target.stat()
    with original_open(target, "wb") as handle:
        handle.write(mutated)
    os.utime(target, ns=(verified.st_atime_ns, verified.st_mtime_ns + 1))
    assert resolver.resolve(row) == target


def test_gcs_cache_redownloads_when_cached_file_stat_changes(tmp_path: Path) -> None:
    payload = b"immutable-gcs-cache"
    corrupted = payload[::-1]
    row = _row(
        shard_generation="7",
        shard_size=len(payload),
        shard_md5_b64=_md5(payload),
        shard_digest=_digest(payload),
    )
    downloader = FakeDownloader(payload)
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=downloader,
    )

    cached = resolver.resolve(row)
    cached.write_bytes(corrupted)
    resolved = resolver.resolve(row)

    assert resolved.read_bytes() == payload
    assert downloader.calls == [
        "gs://example-root/corpus/shards/000001.tar",
        "gs://example-root/corpus/shards/000001.tar",
    ]


def test_gcs_cache_does_not_open_unchanged_hot_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"large-hot-shard"
    row = _row(
        shard_generation="7",
        shard_size=len(payload),
        shard_md5_b64=_md5(payload),
        shard_digest=_digest(payload),
    )
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=FakeDownloader(payload),
    )

    first = resolver.resolve(row)
    original_open = Path.open

    def reject_payload_open(path: Path, *args, **kwargs):
        if path == first:
            pytest.fail("GCS cache hit reread the shard payload")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_payload_open)
    second = resolver.resolve(row)

    assert first == second


def test_gcs_download_uses_streamed_size_without_rereading(tmp_path: Path) -> None:
    payload = b"stream-verified-shard"
    row = _row(
        shard_generation="7",
        shard_size=len(payload),
        shard_md5_b64=_md5(payload),
        shard_digest=_digest(payload),
    )
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=FakeStreamingDownloader(payload),
    )

    assert resolver.resolve(row).read_bytes() == payload


def test_gcs_cache_verified_receipt_is_reused_by_a_new_resolver(
    tmp_path: Path,
) -> None:
    payload = b"node-shared-verified-shard"
    row = _row(
        shard_generation="7",
        shard_size=len(payload),
        shard_md5_b64=_md5(payload),
        shard_digest=_digest(payload),
    )
    cache = tmp_path / "cache"
    first_downloader = FakeDownloader(payload)
    first = GCSResolver(
        root="gs://example-root",
        cache_dir=cache,
        max_bytes=1024,
        downloader=first_downloader,
    )
    target = first.resolve(row)

    second_downloader = FakeDownloader(payload)
    second = GCSResolver(
        root="gs://example-root",
        cache_dir=cache,
        max_bytes=1024,
        downloader=second_downloader,
    )

    assert second.resolve(row) == target
    assert first_downloader.calls == ["gs://example-root/corpus/shards/000001.tar"]
    assert second_downloader.calls == []


def test_gcs_cache_evicts_only_after_a_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"bounded-cache-shard"
    row = _row(
        shard_generation="7",
        shard_size=len(payload),
        shard_digest=_digest(payload),
    )
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=FakeDownloader(payload),
    )
    calls: list[Path] = []
    monkeypatch.setattr(resolver, "_evict", lambda *, protect: calls.append(protect))

    target = resolver.resolve(row)
    assert resolver.resolve(row) == target
    assert calls == [target]


def test_gcs_cache_reaps_an_abandoned_large_partial_after_worker_exit(tmp_path: Path) -> None:
    payload = b"complete"
    cache = tmp_path / "cache"
    row = _row(shard_size=len(payload))
    uri = "gs://example-root/corpus/shards/000001.tar"
    key = hashlib.blake2s(uri.encode()).hexdigest()
    target_name = f"{key[:24]}-000001.tar"
    cache.mkdir()
    orphan = cache / f".{target_name}.999999.part"
    orphan.write_bytes(b"large abandoned partial")
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=cache,
        max_bytes=1024,
        downloader=FakeDownloader(payload),
    )

    assert resolver.resolve(row).read_bytes() == payload
    assert not orphan.exists()


def test_gcs_cache_lru_uses_explicit_receipt_access_time(tmp_path: Path) -> None:
    payload = b"data"
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=2 * len(payload),
        downloader=FakeDownloader(payload),
    )
    rows = tuple(
        _row(
            sample_id=f"sample-{index}",
            key=f"sample-{index}",
            shard=f"corpus/shards/{index:06d}.tar",
            shard_size=len(payload),
        )
        for index in range(3)
    )
    first = resolver.resolve(rows[0])
    second = resolver.resolve(rows[1])
    first_receipt = first.with_suffix(first.suffix + ".receipt.json")
    second_receipt = second.with_suffix(second.suffix + ".receipt.json")
    os.utime(first_receipt, ns=(1, 1))
    os.utime(second_receipt, ns=(2, 2))

    assert resolver.resolve(rows[0]) == first
    third = resolver.resolve(rows[2])

    assert first.is_file()
    assert not second.exists()
    assert third.is_file()


def test_gcs_identity_can_use_digest_without_md5(tmp_path: Path) -> None:
    payload = b"digest-only-immutable-object"
    row = _row(
        shard_generation="8",
        shard_size=len(payload),
        shard_digest=_digest(payload),
    )
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=FakeDownloader(payload),
    )

    assert resolver.resolve(row).read_bytes() == payload


def test_gcs_runtime_identity_needs_only_size(tmp_path: Path) -> None:
    payload = b"path-based-object"
    row = _row(shard_size=len(payload))
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=FakeDownloader(payload),
    )

    assert resolver.resolve(row).read_bytes() == payload


def test_gcs_ignores_non_provider_generation_metadata(tmp_path: Path) -> None:
    payload = b"local-staging-only"
    row = _row(
        shard_generation=f"local-digest:{_digest(payload)}",
        shard_size=len(payload),
        shard_digest=_digest(payload),
    )
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=FakeDownloader(payload),
    )

    assert resolver.resolve(row).read_bytes() == payload


def test_gcs_cache_identity_ignores_generation_changes(tmp_path: Path) -> None:
    payload = b"same-object-at-a-new-provider-generation"
    downloader = FakeDownloader(payload)
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=downloader,
    )

    first = resolver.resolve(_row(shard_generation="7", shard_size=len(payload)))
    second = resolver.resolve(_row(shard_generation="999999999", shard_size=len(payload)))

    assert first == second
    assert second.read_bytes() == payload
    assert downloader.calls == ["gs://example-root/corpus/shards/000001.tar"]


def test_gcs_path_cache_does_not_reuse_generation_receipt(tmp_path: Path) -> None:
    payload = b"current-object"
    stale = b"stale---object"
    assert len(stale) == len(payload)
    cache = tmp_path / "cache"
    cache.mkdir()
    uri = "gs://example-root/corpus/shards/000001.tar"
    key = hashlib.blake2s(uri.encode()).hexdigest()
    target = cache / f"{key[:24]}-000001.tar"
    target.write_bytes(stale)
    receipt = target.with_suffix(target.suffix + ".receipt.json")
    receipt.write_text(
        json.dumps(
            {
                "schema": "solarwm.gcs-cache-receipt.v4",
                "uri": uri,
                "generation": "7",
                "size": len(stale),
            }
        )
    )
    downloader = FakeDownloader(payload)
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=cache,
        max_bytes=1024,
        downloader=downloader,
    )

    assert resolver.resolve(_row(shard_size=len(payload))).read_bytes() == payload
    assert downloader.calls == [uri]
    assert json.loads(receipt.read_text())["schema"] == "solarwm.gcs-cache-receipt.v5"


def test_gcs_size_mismatch_never_commits_cache_entry(tmp_path: Path) -> None:
    payload = b"short"
    cache = tmp_path / "cache"
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=cache,
        max_bytes=1024,
        downloader=FakeStreamingDownloader(payload),
    )

    with pytest.raises(DataContractError, match="size mismatch"):
        resolver.resolve(_row(shard_size=len(payload) + 1))

    assert not list(cache.glob("*.receipt.json"))
    assert not list(cache.glob("*.part"))


def test_gcs_timeout_cleans_partial_releases_lock_and_defers_retry(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    row = _row(shard_size=10)

    class TimeoutDownloader:
        calls = 0

        def download(self, uri: str, generation: str, output) -> int:
            del uri, generation
            self.calls += 1
            output.write(b"partial")
            raise GCSDownloadTimeout("injected timeout")

    downloader = TimeoutDownloader()
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=cache,
        max_bytes=1024,
        downloader=downloader,
        failure_cooldown_seconds=300.0,
    )

    with pytest.raises(GCSDownloadTimeout, match="injected timeout"):
        resolver.resolve(row)

    assert downloader.calls == 1
    assert not list(cache.glob("*.part"))
    lock_path = next(cache.glob("*.lock"))
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    with pytest.raises(GCSDownloadTimeout, match="temporarily unavailable"):
        resolver.resolve(row)

    assert downloader.calls == 1


def test_gcs_timeout_retry_recovers_after_cooldown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = b"recovered"
    now = [100.0]

    class RecoveringDownloader:
        calls = 0

        def download(self, uri: str, generation: str, output) -> int:
            del uri, generation
            self.calls += 1
            if self.calls == 1:
                raise GCSDownloadTimeout("injected timeout")
            output.write(payload)
            return len(payload)

    monkeypatch.setattr("solarwm.data.transport.time.time", lambda: now[0])
    downloader = RecoveringDownloader()
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=downloader,
        failure_cooldown_seconds=30.0,
    )
    row = _row(shard_size=len(payload))

    with pytest.raises(GCSDownloadTimeout, match="injected timeout"):
        resolver.resolve(row)
    with pytest.raises(GCSDownloadTimeout, match="temporarily unavailable"):
        resolver.resolve(row)

    now[0] += 31.0
    assert resolver.resolve(row).read_bytes() == payload
    assert downloader.calls == 2
    assert not list((tmp_path / "cache").glob("*.failure.json"))


def test_gcs_timeout_default_retries_the_next_sample_without_a_shard_cooldown(
    tmp_path: Path,
) -> None:
    payload = b"recovered"

    class RecoveringDownloader:
        calls = 0

        def download(self, uri: str, generation: str, output) -> int:
            del uri, generation
            self.calls += 1
            if self.calls == 1:
                raise GCSDownloadTimeout("injected timeout")
            output.write(payload)
            return len(payload)

    downloader = RecoveringDownloader()
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=downloader,
    )
    row = _row(shard_size=len(payload))

    with pytest.raises(GCSDownloadTimeout, match="injected timeout"):
        resolver.resolve(row)
    assert resolver.resolve(row).read_bytes() == payload
    assert downloader.calls == 2
    assert not list((tmp_path / "cache").glob("*.failure.json"))


def test_gcs_timeout_is_shared_by_existing_lock_waiters_then_a_new_call_retries(
    tmp_path: Path,
) -> None:
    payload = b"recovered"
    first_entered = threading.Event()
    second_started = threading.Event()

    class RecoveringDownloader:
        calls = 0

        def download(self, uri: str, generation: str, output) -> int:
            del uri, generation
            self.calls += 1
            if self.calls == 1:
                first_entered.set()
                assert second_started.wait(timeout=2)
                time.sleep(0.1)
                raise GCSDownloadTimeout("injected timeout")
            output.write(payload)
            return len(payload)

    downloader = RecoveringDownloader()
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=downloader,
        max_elapsed_seconds=2.0,
    )
    row = _row(shard_size=len(payload))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(resolver.resolve, row)
        assert first_entered.wait(timeout=2)
        second = pool.submit(resolver.resolve, row)
        second_started.set()
        with pytest.raises(GCSDownloadTimeout, match="injected timeout"):
            first.result(timeout=2)
        with pytest.raises(GCSDownloadTimeout, match="while this sample waited"):
            second.result(timeout=2)

    assert downloader.calls == 1
    assert resolver.resolve(row).read_bytes() == payload
    assert downloader.calls == 2


def test_gcs_resolve_uses_one_deadline_for_lock_wait_and_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]

    class SlowResponse(io.BytesIO):
        def read1(self, _size: int = -1) -> bytes:
            now[0] += 0.25
            return b"x"

    def delayed_lock(lock, *, deadline: float, total_seconds: float, uri: str) -> None:
        del deadline, total_seconds, uri
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        now[0] = 0.8

    monkeypatch.setattr("solarwm.data.transport.time.monotonic", lambda: now[0])
    monkeypatch.setattr("solarwm.data.transport._acquire_flock", delayed_lock)
    monkeypatch.setattr(
        "solarwm.data.transport.urllib.request.urlopen",
        lambda *_args, **_kwargs: SlowResponse(),
    )
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=GCSRestDownloader(token=lambda: "token", max_attempts=1),
        max_elapsed_seconds=1.0,
    )

    with pytest.raises(GCSDownloadTimeout, match="exceeded 0s"):
        resolver.resolve(_row(shard_size=1))

    assert now[0] < 1.2


def test_gcs_runtime_does_not_enforce_index_md5(tmp_path: Path) -> None:
    payload = b"actual"
    row = _row(
        shard_generation="7",
        shard_size=len(payload),
        shard_md5_b64=_md5(b"wrong!"),
    )
    resolver = GCSResolver(
        root="gs://example-root",
        cache_dir=tmp_path / "cache",
        max_bytes=1024,
        downloader=FakeDownloader(payload),
    )
    assert resolver.resolve(row).read_bytes() == payload
    assert len(list((tmp_path / "cache").glob("*.receipt.json"))) == 1


def test_raw_reader_uses_index_members_and_preserves_identity(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "corpus/shards/000001.tar"
    target.parent.mkdir(parents=True)
    manifest = {"prompt": {"text": "manifest caption"}, "metadata": {"scene": "s"}}
    with tarfile.open(target, "w:") as archive:
        for name, value in {
            "sample/video.mp4": b"video",
            "sample/camera.npz": b"camera",
            "sample/manifest.json": json.dumps(manifest).encode(),
        }.items():
            info = tarfile.TarInfo(name)
            info.size = len(value)
            archive.addfile(info, io.BytesIO(value))

    row = _row(
        video_member="sample/video.mp4",
        camera_member="sample/camera.npz",
        manifest_member="sample/manifest.json",
    )
    plan = SamplePlan(
        sample_id=row.sample_id,
        key=row.key,
        shard=row.shard,
        row_ordinal=0,
        repeat_ordinal=0,
        epoch=0,
        start_frame=3,
        source_frame_indices=(3, 4),
        reader_rank=0,
        worker_id=0,
    )
    shards = TarShardReader(LocalResolver(root))
    sample = RawSampleReader((row,), shards).materialize(plan)
    shards.close()

    assert sample.plan.sample_id == "sample-a"
    assert sample.caption == "manifest caption"
    assert sample.scene == "s"
    assert sample.members == {
        "video_member": b"video",
        "camera_member": b"camera",
    }
