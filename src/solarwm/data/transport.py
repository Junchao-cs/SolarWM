"""Physical shard resolution below the canonical data plan.

The transport receives an :class:`~solarwm.data.index.IndexRow` only after
sample ownership, shuffle order, and frame starts have been decided. Local
files and GCS objects therefore cannot perturb the logical RNG stream.
"""

from __future__ import annotations

import fcntl
import hashlib
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Protocol
from urllib.parse import urlparse

from solarwm.errors import DataContractError

from .index import IndexRow, ShardIdentity, shard_identity, validate_relative_key


def join_root(root: str, relative_key: str) -> str:
    """Join a runtime root and portable index key without changing semantics."""

    key = validate_relative_key(relative_key)
    if root.startswith("gs://"):
        bucket, prefix = parse_gs_uri(root, allow_empty_object=True)
        object_name = "/".join(part for part in (prefix.rstrip("/"), key) if part)
        return f"gs://{bucket}/{object_name}"
    return str(Path(root).expanduser() / key)


class ShardResolver(Protocol):
    """Resolve one immutable index row to a readable local file."""

    def resolve(self, row: IndexRow) -> Path: ...


@dataclass(frozen=True)
class LocalResolver:
    """Resolve relative shard keys beneath a mounted directory."""

    root: Path

    def resolve(self, row: IndexRow) -> Path:
        target = (self.root / row.shard).resolve()
        root = self.root.resolve()
        if target != root and root not in target.parents:
            raise DataContractError(f"shard escapes local root: {row.shard}")
        if not target.is_file():
            raise DataContractError(f"local shard is missing: {target}")
        identity = shard_identity(row)
        if identity is not None:
            _verify_size(target, identity, label="local shard")
        return target


def parse_gs_uri(uri: str, *, allow_empty_object: bool = False) -> tuple[str, str]:
    parsed = urlparse(uri)
    object_name = parsed.path.lstrip("/")
    if (
        parsed.scheme != "gs"
        or not parsed.netloc
        or (not allow_empty_object and not object_name)
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise DataContractError(f"invalid GCS URI: {uri!r}")
    return parsed.netloc, object_name


@dataclass(frozen=True)
class ObjectIdentity:
    """Runtime identity required before an object may enter the cache."""

    size: int

    @classmethod
    def from_row(cls, row: IndexRow) -> ObjectIdentity:
        identity = shard_identity(row)
        if identity is None:
            raise DataContractError("bucket rows require shard_size")
        return cls(size=identity.size)


class ObjectDownloader(Protocol):
    """Download the object currently stored at a URI into an open binary file."""

    def download(
        self,
        uri: str,
        generation: str,
        output: BinaryIO,
        *,
        deadline: float | None = None,
    ) -> int | None: ...


class GCSDownloadTimeout(DataContractError):
    """A GCS shard did not finish within the bounded transfer window."""


def _transient_download_error(exc: BaseException | None) -> bool:
    """Classify retryable transport failures without masking permanent 4xx errors."""

    if exc is None:
        return False
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {408, 429} or exc.code >= 500
    if isinstance(
        exc,
        (
            TimeoutError,
            ConnectionError,
            http.client.IncompleteRead,
            urllib.error.URLError,
            OSError,
        ),
    ):
        return True
    return _transient_download_error(exc.__cause__)


@dataclass
class GoogleTokenProvider:
    """Obtain a bearer token without embedding infrastructure credentials."""

    token_file: Path | None = None
    metadata_url: str = (
        "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
    )

    def __call__(self) -> str:
        if self.token_file is not None and self.token_file.is_file():
            value = self.token_file.read_text(encoding="utf-8").strip()
            if value:
                return value
        explicit = os.environ.get("GOOGLE_OAUTH_ACCESS_TOKEN", "").strip()
        if explicit:
            return explicit
        request = urllib.request.Request(self.metadata_url, headers={"Metadata-Flavor": "Google"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                payload = json.load(response)
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            raise DataContractError(
                "cannot obtain a GCS access token; provide token_file, "
                "GOOGLE_OAUTH_ACCESS_TOKEN, or a metadata-enabled workload identity"
            ) from exc
        value = str(payload.get("access_token") or "")
        if not value:
            raise DataContractError("the metadata service returned an empty access token")
        return value


@dataclass
class GCSRestDownloader:
    """Path-based GCS JSON API downloader using only the stdlib."""

    token: Callable[[], str] = field(default_factory=GoogleTokenProvider)
    timeout_seconds: float = 120.0
    max_elapsed_seconds: float = 300.0
    max_attempts: int = 12
    max_auth_attempts: int = 72
    chunk_bytes: int = 16 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise DataContractError("GCS socket timeout must be positive")
        if self.max_elapsed_seconds <= 0:
            raise DataContractError("GCS download time limit must be positive")
        if self.max_attempts < 1 or self.max_auth_attempts < 1:
            raise DataContractError("GCS download attempts must be positive")
        if self.chunk_bytes < 1:
            raise DataContractError("GCS download chunk size must be positive")

    def download(
        self,
        uri: str,
        generation: str,
        output: BinaryIO,
        *,
        deadline: float | None = None,
    ) -> int:
        # Keep the injected-downloader call signature stable, but deliberately
        # ignore publication generation metadata for ordinary runtime reads.
        del generation
        bucket, object_name = parse_gs_uri(uri)
        encoded = urllib.parse.quote(object_name, safe="")
        query = urllib.parse.urlencode({"alt": "media"})
        url = (
            "https://storage.googleapis.com/download/storage/v1/b/"
            f"{urllib.parse.quote(bucket, safe='')}/o/{encoded}?{query}"
        )
        last_error: Exception | None = None
        auth_error_seen = False
        started = time.monotonic()
        own_deadline = started + self.max_elapsed_seconds
        deadline = own_deadline if deadline is None else min(own_deadline, float(deadline))
        budget_seconds = max(0.0, deadline - started)
        attempts = max(self.max_attempts, self.max_auth_attempts)
        for attempt in range(attempts):
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GCSDownloadTimeout(
                        f"GCS download exceeded {budget_seconds:.0f}s for {uri}"
                    )
                output.seek(0)
                output.truncate(0)
                size = 0
                request = urllib.request.Request(
                    url, headers={"Authorization": f"Bearer {self.token()}"}
                )
                request_timeout = max(0.001, min(self.timeout_seconds, remaining))
                with urllib.request.urlopen(request, timeout=request_timeout) as response:
                    read = getattr(response, "read1", response.read)
                    while True:
                        if time.monotonic() >= deadline:
                            raise GCSDownloadTimeout(
                                f"GCS download exceeded {budget_seconds:.0f}s "
                                f"after {attempt + 1} attempt(s) and {size} bytes for {uri}"
                            )
                        chunk = read(self.chunk_bytes)
                        if not chunk:
                            break
                        output.write(chunk)
                        size += len(chunk)
                        if time.monotonic() >= deadline:
                            raise GCSDownloadTimeout(
                                f"GCS download exceeded {budget_seconds:.0f}s "
                                f"after {attempt + 1} attempt(s) and {size} bytes for {uri}"
                            )
                return size
            except GCSDownloadTimeout:
                raise
            except Exception as exc:  # urllib raises environment-specific errors
                last_error = exc
                is_auth = isinstance(exc, urllib.error.HTTPError) and exc.code in {401, 403}
                auth_error_seen = auth_error_seen or is_auth
                budget = self.max_auth_attempts if auth_error_seen else self.max_attempts
                if attempt + 1 >= budget:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise GCSDownloadTimeout(
                        f"GCS download exceeded {budget_seconds:.0f}s "
                        f"after {attempt + 1} attempt(s) for {uri}"
                    ) from exc
                delay = min(10.0, 0.5 * (2 ** min(attempt, 5)))
                time.sleep(min(delay, remaining))
        if _transient_download_error(last_error):
            raise GCSDownloadTimeout(
                f"GCS download exhausted its retry budget for {uri}: {last_error}"
            ) from last_error
        raise DataContractError(f"failed to download {uri}: {last_error}") from last_error


def _verification_token(
    stat: os.stat_result, identity: ShardIdentity | ObjectIdentity
) -> tuple[object, ...]:
    return (
        stat.st_dev,
        stat.st_ino,
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        identity,
    )


def _verify_size(
    path: Path,
    identity: ShardIdentity | ObjectIdentity,
    *,
    label: str,
) -> None:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise DataContractError(f"cannot stat {label} {path}: {exc}") from exc
    if size != identity.size:
        raise DataContractError(f"{label} size mismatch for {path}: {size} != {identity.size}")


def _valid_receipt_values(
    target: Path,
    receipt: Path,
    identity: ObjectIdentity,
    *,
    uri: str,
) -> Mapping[str, object] | None:
    if not target.is_file() or not receipt.is_file():
        return None
    try:
        values = json.loads(receipt.read_text(encoding="utf-8"))
        size = target.stat().st_size
        receipt_size = int(values.get("size", -1))
    except (OSError, ValueError, TypeError):
        return None
    valid = (
        size == identity.size
        and values.get("schema") == "solarwm.gcs-cache-receipt.v5"
        and values.get("uri") == uri
        and receipt_size == identity.size
    )
    return values if valid else None


def _receipt_valid(
    target: Path,
    receipt: Path,
    identity: ObjectIdentity,
    *,
    uri: str,
) -> bool:
    return _valid_receipt_values(target, receipt, identity, uri=uri) is not None


def _verified_stat(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


def _receipt_stat_matches(values: Mapping[str, object], path: Path) -> bool:
    stored = values.get("verified_stat")
    return isinstance(stored, Mapping) and dict(stored) == _verified_stat(path)


def _receipt_payload(
    target: Path,
    identity: ObjectIdentity,
    *,
    uri: str,
) -> dict[str, object]:
    return {
        "schema": "solarwm.gcs-cache-receipt.v5",
        "uri": uri,
        "size": identity.size,
        "verified_stat": _verified_stat(target),
    }


def _atomic_json(path: Path, values: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (json.dumps(values, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _failure_retry_after(path: Path, *, uri: str) -> float | None:
    if not path.is_file():
        return None
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        if values.get("schema") != "solarwm.gcs-cache-failure.v1" or values.get("uri") != uri:
            return None
        return float(values["retry_after"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _failure_happened_during_call(path: Path, *, uri: str, started_ns: int) -> bool:
    """Tell lock waiters to share a timeout without delaying a later retry."""

    if not path.is_file():
        return False
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
        return bool(
            values.get("schema") == "solarwm.gcs-cache-failure.v1"
            and values.get("uri") == uri
            and int(values.get("failed_at_ns", -1)) >= int(started_ns)
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _acquire_flock(lock: BinaryIO, *, deadline: float, total_seconds: float, uri: str) -> None:
    """Acquire one shard lock within the same bounded transfer window."""

    while True:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GCSDownloadTimeout(
                    f"GCS resolve exceeded {total_seconds:.0f}s while waiting for {uri}"
                ) from None
            time.sleep(min(0.05, remaining))


@dataclass
class GCSResolver:
    """Node-shared, path-based, bounded cache for GCS shards."""

    root: str
    cache_dir: Path
    max_bytes: int
    downloader: ObjectDownloader = field(default_factory=GCSRestDownloader)
    failure_cooldown_seconds: float = 0.0
    max_elapsed_seconds: float = 300.0
    _verified: dict[Path, tuple[object, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        parse_gs_uri(self.root, allow_empty_object=True)
        if self.max_bytes < 1:
            raise DataContractError("GCS cache max_bytes must be positive")
        if self.failure_cooldown_seconds < 0:
            raise DataContractError("GCS failure cooldown must be non-negative")
        if self.max_elapsed_seconds <= 0:
            raise DataContractError("GCS resolve time limit must be positive")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def resolve(self, row: IndexRow, *, deadline: float | None = None) -> Path:
        identity = ObjectIdentity.from_row(row)
        uri = join_root(self.root, row.shard)
        key = hashlib.blake2s(uri.encode()).hexdigest()
        target = self.cache_dir / f"{key[:24]}-{Path(row.shard).name}"
        receipt = target.with_suffix(target.suffix + ".receipt.json")
        failure = target.with_suffix(target.suffix + ".failure.json")
        lock_path = target.with_suffix(target.suffix + ".lock")
        call_started_ns = time.time_ns()
        started = time.monotonic()
        own_deadline = started + self.max_elapsed_seconds
        deadline = own_deadline if deadline is None else min(own_deadline, float(deadline))
        total_seconds = max(0.0, deadline - started)
        if total_seconds <= 0:
            raise GCSDownloadTimeout(f"GCS resolve has no time remaining for {uri}")

        downloaded = False
        with lock_path.open("a+b") as lock:
            acquired = False
            try:
                _acquire_flock(
                    lock,
                    deadline=deadline,
                    total_seconds=total_seconds,
                    uri=uri,
                )
                acquired = True
                for orphan in self.cache_dir.glob(f".{target.name}.*.part"):
                    orphan.unlink(missing_ok=True)
                receipt_values = _valid_receipt_values(target, receipt, identity, uri=uri)
                cached_matches = False
                if receipt_values is not None:
                    before = target.stat()
                    token = _verification_token(before, identity)
                    if self._verified.get(target) == token or _receipt_stat_matches(
                        receipt_values, target
                    ):
                        self._verified[target] = token
                        cached_matches = True
                if cached_matches:
                    failure.unlink(missing_ok=True)
                    os.utime(receipt, None)
                if not cached_matches:
                    retry_after = _failure_retry_after(failure, uri=uri)
                    if retry_after is not None and time.time() < retry_after:
                        raise GCSDownloadTimeout(
                            f"GCS shard is temporarily unavailable after a timed-out "
                            f"download: {uri}"
                        )
                    if self.failure_cooldown_seconds == 0 and _failure_happened_during_call(
                        failure,
                        uri=uri,
                        started_ns=call_started_ns,
                    ):
                        raise GCSDownloadTimeout(
                            f"GCS shard download timed out while this sample waited: {uri}"
                        )
                    failure.unlink(missing_ok=True)
                    self._verified.pop(target, None)
                    target.unlink(missing_ok=True)
                    receipt.unlink(missing_ok=True)
                    part = target.with_name(f".{target.name}.{os.getpid()}.part")
                    part.unlink(missing_ok=True)
                    try:
                        with part.open("w+b") as output:
                            generation = str(row.values.get("shard_generation") or "")
                            if isinstance(self.downloader, GCSRestDownloader):
                                observed_size = self.downloader.download(
                                    uri,
                                    generation,
                                    output,
                                    deadline=deadline,
                                )
                            else:
                                observed_size = self.downloader.download(uri, generation, output)
                            output.flush()
                            os.fsync(output.fileno())
                        _verify_size(part, identity, label=f"GCS shard {uri}")
                        if observed_size is not None and observed_size != identity.size:
                            raise DataContractError(
                                f"GCS shard {uri} size mismatch while downloading: "
                                f"{observed_size} != {identity.size}"
                            )
                        os.replace(part, target)
                        _atomic_json(
                            receipt,
                            _receipt_payload(target, identity, uri=uri),
                        )
                        self._verified[target] = _verification_token(target.stat(), identity)
                        failure.unlink(missing_ok=True)
                        downloaded = True
                    except GCSDownloadTimeout as exc:
                        _atomic_json(
                            failure,
                            {
                                "schema": "solarwm.gcs-cache-failure.v1",
                                "uri": uri,
                                "failed_at_ns": time.time_ns(),
                                "retry_after": time.time() + self.failure_cooldown_seconds,
                                "error": str(exc),
                            },
                        )
                        raise
                    finally:
                        part.unlink(missing_ok=True)
            finally:
                if acquired:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

        if downloaded:
            self._evict(protect=target)
        return target

    def _evict(self, *, protect: Path) -> None:
        eviction_lock = self.cache_dir / ".evict.lock"
        with eviction_lock.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                self._reap_orphan_parts()
                total = 0
                candidates: list[tuple[int, Path, int]] = []
                for receipt in self.cache_dir.glob("*.receipt.json"):
                    suffix = ".receipt.json"
                    target = receipt.with_name(receipt.name[: -len(suffix)])
                    if not target.is_file():
                        continue
                    stat = target.stat()
                    total += stat.st_size
                    if target != protect:
                        candidates.append((receipt.stat().st_mtime_ns, target, stat.st_size))
                for _, candidate, size in sorted(candidates):
                    if total <= self.max_bytes:
                        break
                    item_lock_path = candidate.with_suffix(candidate.suffix + ".lock")
                    with item_lock_path.open("a+b") as item_lock:
                        try:
                            fcntl.flock(item_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            continue
                        candidate.unlink(missing_ok=True)
                        candidate.with_suffix(candidate.suffix + ".receipt.json").unlink(
                            missing_ok=True
                        )
                        self._verified.pop(candidate, None)
                        total -= size
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _reap_orphan_parts(self) -> None:
        """Delete abandoned large partials only when their item lock is free."""

        for part in self.cache_dir.glob(".*.*.part"):
            try:
                target_name, pid, suffix = part.name[1:].rsplit(".", 2)
            except ValueError:
                continue
            if suffix != "part" or not pid.isdigit() or not target_name:
                continue
            lock_path = (self.cache_dir / target_name).with_suffix(
                (self.cache_dir / target_name).suffix + ".lock"
            )
            with lock_path.open("a+b") as item_lock:
                try:
                    fcntl.flock(item_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                part.unlink(missing_ok=True)


def resolver_from_config(
    root: str, *, cache_dir: str | None, max_gib: float = 256.0
) -> ShardResolver:
    """Construct the only transport switch exposed to run configuration."""

    if root.startswith("gs://"):
        if not cache_dir:
            raise DataContractError("a gs:// data root requires data.cache_dir")
        return GCSResolver(
            root=root,
            cache_dir=Path(cache_dir).expanduser(),
            max_bytes=max(1, int(float(max_gib) * 1024**3)),
        )
    if cache_dir:
        raise DataContractError("data.cache_dir is only valid for a gs:// data root")
    return LocalResolver(Path(root).expanduser())
