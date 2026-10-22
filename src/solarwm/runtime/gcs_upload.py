"""Create-only GCS publication with bounded requests and private upload sessions."""

from __future__ import annotations

import base64
import hashlib
import http.client
import io
import json
import mimetypes
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote, urlencode, urlsplit

from solarwm.data.transport import GoogleTokenProvider
from solarwm.errors import SolarWMError

_HOST = "storage.googleapis.com"
_CHUNK = 8 * 1024 * 1024
_RETRYABLE = {401, 408, 429, 500, 502, 503, 504}


class PublicationError(SolarWMError):
    """A fixed diagnostic code; never include transport bodies or credentials."""

    def __init__(self, code: str, *, retryable: bool = False) -> None:
        super().__init__(code)
        self.retryable = retryable


@dataclass(frozen=True)
class ObjectInfo:
    size: int
    md5: str

    @classmethod
    def from_bytes(cls, payload: bytes) -> ObjectInfo:
        digest = hashlib.md5(payload, usedforsecurity=False).digest()
        return cls(len(payload), base64.b64encode(digest).decode("ascii"))

    @classmethod
    def from_file(cls, path: Path) -> ObjectInfo:
        digest = hashlib.md5(usedforsecurity=False)
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(_CHUNK):
                size += len(chunk)
                digest.update(chunk)
        return cls(size, base64.b64encode(digest.digest()).decode("ascii"))


def _relative_key(key: str) -> str:
    if (
        not key
        or "\\" in key
        or any(ord(char) < 32 or ord(char) == 127 for char in key)
        or any(part in {"", ".", ".."} for part in key.split("/"))
    ):
        raise PublicationError("invalid_object_key")
    return key


def _json(payload: bytes) -> dict:
    try:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, UnicodeError):
        raise PublicationError("invalid_storage_response") from None


def _info(value: dict) -> ObjectInfo:
    try:
        size, md5 = int(value["size"]), value["md5Hash"]
        if size < 0 or len(base64.b64decode(md5, validate=True)) != 16:
            raise ValueError
        return ObjectInfo(size, md5)
    except (KeyError, TypeError, ValueError):
        raise PublicationError("invalid_object_metadata") from None


class GCSStore:
    """Store relative keys under one explicit run prefix, without overwrites.

    Session URLs stay in memory. Restarting resumes at immutable file boundaries;
    transient failures within a live session resume from the acknowledged offset.
    """

    def __init__(
        self,
        destination: str,
        *,
        token: Callable[[], str] | None = None,
        timeout: float = 60.0,
        max_attempts: int = 4,
    ) -> None:
        uri = urlsplit(destination)
        if (
            uri.scheme != "gs"
            or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", uri.netloc)
            or uri.query
            or uri.fragment
            or not uri.path.startswith("/")
        ):
            raise PublicationError("invalid_destination")
        self.prefix = _relative_key(uri.path[1:].rstrip("/")) + "/"
        self.bucket = uri.netloc
        self.token = token if token is not None else GoogleTokenProvider()
        self.timeout, self.max_attempts = timeout, max_attempts

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: bytes = b"",
        headers: dict[str, str] | None = None,
        accepted: tuple[int, ...] = (200,),
        retry: bool = True,
    ) -> tuple[int, dict[str, str], bytes]:
        attempts = self.max_attempts if retry else 1
        for attempt in range(attempts):
            connection = http.client.HTTPSConnection(_HOST, timeout=self.timeout)
            try:
                try:
                    credential = self.token()
                    if not credential or "\r" in credential or "\n" in credential:
                        raise ValueError
                except Exception:
                    raise PublicationError("storage_authentication_failed") from None
                request_headers = {"Authorization": "Bearer " + credential, **(headers or {})}
                connection.request(method, path, body=body, headers=request_headers)
                response = connection.getresponse()
                response_headers = {key.lower(): value for key, value in response.getheaders()}
                payload = response.read(16 * 1024 * 1024 + 1)
                if len(payload) > 16 * 1024 * 1024:
                    raise PublicationError("storage_response_too_large")
                if response.status in accepted:
                    return response.status, response_headers, payload
                error = PublicationError(
                    f"storage_http_{response.status}", retryable=response.status in _RETRYABLE
                )
            except (OSError, http.client.HTTPException):
                error = PublicationError("storage_transport_failed", retryable=True)
            finally:
                connection.close()
            if not error.retryable or attempt + 1 == attempts:
                raise error from None
            time.sleep(min(2**attempt, 8))
        raise PublicationError("storage_retry_exhausted")  # pragma: no cover

    def _object_path(self, key: str) -> str:
        return f"/storage/v1/b/{self.bucket}/o/" + quote(self.prefix + _relative_key(key), safe="")

    def metadata(self, key: str) -> ObjectInfo | None:
        status, _, payload = self._request(
            "GET", self._object_path(key) + "?fields=size,md5Hash", accepted=(200, 404)
        )
        return None if status == 404 else _info(_json(payload))

    def read(self, key: str) -> bytes:
        return self._request("GET", self._object_path(key) + "?alt=media")[2]

    def list(self, prefix: str) -> dict[str, ObjectInfo]:
        prefix = _relative_key(prefix.rstrip("/")) + "/"
        result: dict[str, ObjectInfo] = {}
        page = ""
        while True:
            query = {
                "prefix": self.prefix + prefix,
                "fields": "items(name,size,md5Hash),nextPageToken",
                "maxResults": "1000",
            }
            if page:
                query["pageToken"] = page
            payload = _json(
                self._request("GET", f"/storage/v1/b/{self.bucket}/o?" + urlencode(query))[2]
            )
            for item in payload.get("items", []):
                name = item.get("name", "")
                if not name.startswith(self.prefix + prefix):
                    raise PublicationError("storage_prefix_mismatch")
                result[name[len(self.prefix) :]] = _info(item)
            page = payload.get("nextPageToken", "")
            if not page:
                return result

    def put_file(self, key: str, path: Path, expected: ObjectInfo) -> None:
        with path.open("rb") as stream:
            self._put(key, stream, expected)

    def put_bytes(self, key: str, payload: bytes) -> None:
        self._put(key, io.BytesIO(payload), ObjectInfo.from_bytes(payload))

    def _present(self, key: str, expected: ObjectInfo) -> bool:
        existing = self.metadata(key)
        if existing is not None and existing != expected:
            raise PublicationError("object_content_conflict")
        return existing is not None

    def _put(self, key: str, stream: BinaryIO, expected: ObjectInfo) -> None:
        if expected.size <= 0:
            raise PublicationError("empty_publication_object")
        if self._present(key, expected):
            return
        content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
        query = urlencode({"uploadType": "resumable", "ifGenerationMatch": "0"})
        status, headers, _ = self._request(
            "POST",
            f"/upload/storage/v1/b/{self.bucket}/o?{query}",
            body=json.dumps(
                {
                    "name": self.prefix + _relative_key(key),
                    "md5Hash": expected.md5,
                    "contentType": content_type,
                }
            ).encode(),
            headers={
                "Content-Type": "application/json",
                "X-Upload-Content-Type": content_type,
                "X-Upload-Content-Length": str(expected.size),
            },
            accepted=(200, 201, 412),
        )
        if status != 412:
            session = urlsplit(headers.get("location", ""))
            if (
                session.scheme != "https"
                or session.netloc != _HOST
                or not session.path.startswith("/upload/")
                or session.fragment
            ):
                raise PublicationError("invalid_upload_session")
            session_path = session.path + "?" + session.query
            offset, stalls = 0, 0
            while offset < expected.size:
                stream.seek(offset)
                chunk = stream.read(min(_CHUNK, expected.size - offset))
                if len(chunk) != min(_CHUNK, expected.size - offset):
                    raise PublicationError("local_file_changed")
                try:
                    status, headers, _ = self._request(
                        "PUT",
                        session_path,
                        body=chunk,
                        headers={
                            "Content-Range": (
                                f"bytes {offset}-{offset + len(chunk) - 1}/{expected.size}"
                            )
                        },
                        accepted=(200, 201, 308, 412),
                        retry=False,
                    )
                except PublicationError as exc:
                    if not exc.retryable:
                        raise
                    status, headers, _ = self._request(
                        "PUT",
                        session_path,
                        headers={"Content-Range": f"bytes */{expected.size}"},
                        accepted=(200, 201, 308, 412),
                    )
                if status != 308:
                    break
                acknowledged = headers.get("range", "")
                match = re.fullmatch(r"bytes=0-(\d+)", acknowledged)
                if acknowledged and match is None:
                    raise PublicationError("invalid_upload_offset")
                following = int(match[1]) + 1 if match else 0
                if following < offset or following > offset + len(chunk):
                    raise PublicationError("invalid_upload_offset")
                stalls = stalls + 1 if following == offset else 0
                if stalls >= self.max_attempts:
                    raise PublicationError("upload_stalled", retryable=True)
                offset = following
        if not self._present(key, expected):
            raise PublicationError("upload_not_committed", retryable=True)
