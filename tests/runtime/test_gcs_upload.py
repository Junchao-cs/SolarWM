from __future__ import annotations

import json
from collections import deque
from urllib.parse import parse_qs, urlsplit

import pytest

from solar.runtime.gcs_upload import GCSStore, ObjectInfo, PublicationError

DESTINATION = "gs://example-bucket/example-run"
SESSION = "https://storage.googleapis.com/upload/storage/v1/b/example-bucket/o?upload_id=private"


def metadata(data):
    info = ObjectInfo.from_bytes(data)
    return {"size": str(info.size), "md5Hash": info.md5}


def response(status, data=b"", headers=None):
    return status, headers or {}, data if isinstance(data, bytes) else json.dumps(data).encode()


class Wire:
    def __init__(self, monkeypatch, responses):
        self.responses = deque(responses)
        self.requests = []
        wire = self

        class Connection:
            def __init__(self, host, *, timeout):
                assert host == "storage.googleapis.com"
                assert timeout > 0

            def request(self, method, path, *, body, headers):
                wire.requests.append((method, path, body, headers))

            def getresponse(self):
                item = wire.responses.popleft()
                if isinstance(item, Exception):
                    raise item
                status, headers, data = item

                class Response:
                    def getheaders(self):
                        return list(headers.items())

                    def read(self, limit):
                        return data[:limit]

                result = Response()
                result.status = status
                return result

            def close(self):
                pass

        monkeypatch.setattr("solar.runtime.gcs_upload.http.client.HTTPSConnection", Connection)
        monkeypatch.setattr("solar.runtime.gcs_upload.time.sleep", lambda _seconds: None)


def store(**kwargs):
    return GCSStore(DESTINATION, token=lambda: "synthetic-credential", **kwargs)


def test_file_upload_is_create_only_checksum_verified_and_session_not_persisted(
    tmp_path, monkeypatch
):
    data = b"synthetic-video"
    path = tmp_path / "sample.mp4"
    path.write_bytes(data)
    wire = Wire(
        monkeypatch,
        [
            response(404),
            response(200, headers={"Location": SESSION}),
            response(200),
            response(200, metadata(data)),
        ],
    )
    store().put_file(
        "validation/step_000200_live/compare/sample.mp4", path, ObjectInfo.from_bytes(data)
    )
    initiation = wire.requests[1]
    assert parse_qs(urlsplit(initiation[1]).query)["ifGenerationMatch"] == ["0"]
    body = json.loads(initiation[2])
    assert body["md5Hash"] == metadata(data)["md5Hash"]
    assert body["name"].startswith("example-run/validation/")
    assert wire.requests[2][2] == data
    assert list(tmp_path.iterdir()) == [path]
    assert not wire.responses


def test_idempotency_checks_contents_not_only_size(monkeypatch):
    data = b"same-size"
    wire = Wire(monkeypatch, [response(200, metadata(data))])
    store().put_bytes("receipt.json", data)
    assert len(wire.requests) == 1
    Wire(monkeypatch, [response(200, metadata(b"different"))])
    with pytest.raises(PublicationError, match="object_content_conflict"):
        store().put_bytes("receipt.json", data)


def test_lost_chunk_response_queries_acknowledged_offset(monkeypatch):
    data = b"abcdef"
    monkeypatch.setattr("solar.runtime.gcs_upload._CHUNK", 4)
    wire = Wire(
        monkeypatch,
        [
            response(404),
            response(200, headers={"Location": SESSION}),
            OSError("transport details must remain private"),
            response(308, headers={"Range": "bytes=0-1"}),
            response(200),
            response(200, metadata(data)),
        ],
    )
    store().put_bytes("video.mp4", data)
    assert wire.requests[2][2] == b"abcd"
    assert wire.requests[3][2] == b""
    assert wire.requests[3][3]["Content-Range"] == "bytes */6"
    assert wire.requests[4][2] == b"cdef"
    assert wire.requests[4][3]["Content-Range"] == "bytes 2-5/6"


def test_multi_chunk_upload_obeys_acknowledged_range(monkeypatch):
    data = b"abcdef"
    monkeypatch.setattr("solar.runtime.gcs_upload._CHUNK", 4)
    wire = Wire(
        monkeypatch,
        [
            response(404),
            response(200, headers={"Location": SESSION}),
            response(308, headers={"Range": "bytes=0-3"}),
            response(200),
            response(200, metadata(data)),
        ],
    )
    store().put_bytes("video.mp4", data)
    assert wire.requests[3][2] == b"ef"


def test_create_race_accepts_only_identical_winner(monkeypatch):
    data = b"receipt"
    Wire(monkeypatch, [response(404), response(412), response(200, metadata(data))])
    store().put_bytes("receipt.json", data)
    Wire(monkeypatch, [response(404), response(412), response(200, metadata(b"other"))])
    with pytest.raises(PublicationError, match="object_content_conflict"):
        store().put_bytes("receipt.json", data)


def test_listing_is_prefix_scoped_and_paginated(monkeypatch):
    wire = Wire(
        monkeypatch,
        [
            response(
                200,
                {
                    "items": [{"name": "example-run/validation/a", **metadata(b"a")}],
                    "nextPageToken": "next",
                },
            ),
            response(200, {"items": [{"name": "example-run/validation/b", **metadata(b"b")}]}),
        ],
    )
    objects = store().list("validation/")
    assert set(objects) == {"validation/a", "validation/b"}
    assert parse_qs(urlsplit(wire.requests[0][1]).query)["prefix"] == ["example-run/validation/"]
    assert parse_qs(urlsplit(wire.requests[1][1]).query)["pageToken"] == ["next"]


@pytest.mark.parametrize(
    "session",
    [
        "https://example.invalid/upload/private",
        "http://storage.googleapis.com/upload/private",
        "https://storage.googleapis.com@other.invalid/upload/private",
        SESSION + "#private",
    ],
)
def test_untrusted_session_destinations_are_rejected(monkeypatch, session):
    wire = Wire(monkeypatch, [response(404), response(200, headers={"Location": session})])
    with pytest.raises(PublicationError, match="invalid_upload_session") as caught:
        store().put_bytes("receipt.json", b"receipt")
    assert session not in str(caught.value)
    assert len(wire.requests) == 2


def test_authentication_refresh_retries_and_errors_do_not_echo_private_data(monkeypatch):
    credentials = iter(["first-synthetic-credential", "second-synthetic-credential"])
    wire = Wire(monkeypatch, [response(401, b"sensitive response"), response(404)])
    assert GCSStore(DESTINATION, token=lambda: next(credentials)).metadata("missing") is None
    assert wire.requests[0][3]["Authorization"] != wire.requests[1][3]["Authorization"]
    Wire(monkeypatch, [response(403, b"sensitive response " + SESSION.encode())])
    with pytest.raises(PublicationError) as caught:
        store().metadata("object")
    assert str(caught.value) == "storage_http_403"


def test_redirects_do_not_forward_credentials(monkeypatch):
    wire = Wire(monkeypatch, [response(302, headers={"Location": "https://example.invalid"})])
    with pytest.raises(PublicationError, match="storage_http_302"):
        store().metadata("object")
    assert len(wire.requests) == 1


def test_transport_retries_are_bounded(monkeypatch):
    wire = Wire(monkeypatch, [OSError(SESSION), response(503, b"private"), response(503)])
    with pytest.raises(PublicationError) as caught:
        store(max_attempts=3).metadata("object")
    assert str(caught.value) == "storage_http_503"
    assert len(wire.requests) == 3


@pytest.mark.parametrize(
    "destination",
    [
        "gs://example-bucket",
        "gs://example-bucket/run?token=private",
        "gs://user:private@example-bucket/run",
        "gs://example-bucket/run/../other",
    ],
)
def test_invalid_destination_has_no_reflected_input(destination):
    with pytest.raises(PublicationError) as caught:
        GCSStore(destination)
    assert destination not in str(caught.value)
