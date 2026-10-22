"""Loopback web/control and independent resumable media endpoints."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import struct
import threading
import time
from urllib.parse import parse_qs, urlsplit

from .media.store import MediaStore

STATIC = Path(__file__).parent / "static"


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(state, output):
    options = state.options
    deployment = state.row["deployment_id"]
    store = MediaStore(Path(output), deployment)
    origin = f"http://{options['host']}:{options['port']}"

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def allowed_origin(self, value):
            if value is None or value == origin or value in options.get("allowed_origins", []):
                return True
            request = urlsplit("http://" + self.headers.get("Host", ""))
            parsed = urlsplit(value)
            return (request.hostname in ("127.0.0.1", "::1", "localhost")
                    and parsed.scheme == "http" and parsed.netloc == request.netloc)

        def log_message(self, *_):
            pass

        def end_headers(self):
            request_origin = self.headers.get("Origin")
            if request_origin and self.allowed_origin(request_origin):
                self.send_header("Access-Control-Allow-Origin", request_origin)
                self.send_header("Vary", "Origin")
                self.send_header("Access-Control-Expose-Headers", "X-Deployment,X-Next-Cursor,X-Stream-Done")
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()

        def reply(self, value, code=200, kind="application/json"):
            data = json.dumps(value, allow_nan=False).encode() if not isinstance(value, bytes) else value
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            target = urlsplit(self.path)
            path = target.path
            if path in ("/status", "/health"):
                return self.reply(state.snapshot())
            if path in ("/clock", "/delivery-clock"):
                now = time.monotonic_ns() / 1e6
                return self.reply(dict(server_monotonic_ms=now, received_ms=now, sent_ms=now))
            if path == "/runtime-config.js":
                # Relative same-origin URLs also work behind authenticated proxies.
                media = json.dumps(options["media_origin"]) if options.get("media_origin") else "location.origin"
                value = f"window.SOLARWM_MEDIA_ORIGIN={media};window.SOLARWM_CONTROL_ORIGIN=location.origin;"
                return self.reply(value.encode(), kind="text/javascript")
            if path == "/stream":
                return self.stream(target)
            if path == "/gallery/manifest.json":
                return self.reply(dict(items=[dict(id=k, title=v.get("title", k),
                    filename=v.get("title", k), scene=v.get("title", k), prompt=v.get("prompt", ""),
                    thumbnail="/gallery/" + k + ".png", preview="/gallery/" + k + ".png",
                    width=1344, height=768) for k, v in state.scene_snapshot().items()]))
            if path.startswith("/gallery/") and path.endswith(".png"):
                scene = path[len("/gallery/"):-4]
                row = state.scene_snapshot().get(scene)
                if row and row.get("first"):
                    return self.reply(Path(row["first"]).read_bytes(),
                        kind=mimetypes.guess_type(row["first"])[0] or "application/octet-stream")
                return self.reply({"error": "scene image unavailable"}, 404)
            if path == "/first.png":
                row = state.scene_snapshot()[state.snapshot()["scene_id"]]
                image = Path(row["first"]) if row.get("first") else STATIC / "solar_teaser.png"
                return self.reply(image.read_bytes(), kind=mimetypes.guess_type(image.name)[0] or "image/png")
            if path == "/video":
                row = state.snapshot()
                file = Path(output) / f"{row['scene_id']}-upgraded-streaming-session-{row['stream_sequence']}.mp4"
                if not row["video_ready"] or not file.is_file():
                    return self.reply({"error": "video not completed"}, 404)
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(file.stat().st_size))
                self.end_headers()
                try:
                    with file.open("rb") as source:
                        for block in iter(lambda: source.read(256 * 1024), b""):
                            self.wfile.write(block)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            name = "index.html" if path == "/" else path[1:]
            if "/" in name or "\\" in name or name.startswith("."):
                return self.reply({}, 404)
            file = STATIC / name
            if file.is_file() and file.suffix in (".html", ".js", ".png", ".css"):
                kind = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                        ".png": "image/png", ".css": "text/css"}[file.suffix]
                return self.reply(file.read_bytes(), kind=kind)
            return self.reply({}, 404)

        def do_POST(self):
            # Never allow arbitrary websites to command this loopback model.
            if not self.allowed_origin(self.headers.get("Origin")):
                return self.reply({"error": "origin refused"}, 403)
            try:
                route = urlsplit(self.path).path
                from .scene_inputs import MAX_BODY_BYTES
                limit = MAX_BODY_BYTES if route == "/scenes" else 4096
                size = self.headers.get("Content-Length", "")
                if not size.isdecimal() or not 0 < int(size) <= limit:
                    self.close_connection = True
                    raise ValueError("bounded JSON body required")
                self.connection.settimeout(30)
                body = json.loads(self.rfile.read(int(size)))
                if type(body) is not dict:
                    raise ValueError("JSON object required")
                return self.reply(state.upload(body) if route == "/scenes" else state.request(route, body))
            except (ValueError, RuntimeError, OSError) as error:
                return self.reply(dict(error=str(error)), 409)

        def stream(self, target):
            try:
                q = parse_qs(target.query, strict_parsing=True)
                if set(q) != {"scene", "session", "cursor"} or any(len(v) != 1 for v in q.values()):
                    raise ValueError("exact cursor parameters required")
                scene, session, cursor = q["scene"][0], int(q["session"][0]), int(q["cursor"][0])
                if scene not in state.scenes or session < 1 or cursor < 0:
                    raise ValueError("invalid stream identity")
            except ValueError as error:
                return self.reply(dict(error=str(error)), 409)
            self.close_connection = True
            self.connection.settimeout(5)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-h3-fragments")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.send_header("X-Deployment", deployment)
            self.end_headers()
            def send(data):
                self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
            def notice(kind):
                data = json.dumps(dict(kind=kind, scene=scene, session=session,
                    cursor=cursor, deployment_id=deployment)).encode()
                send(struct.pack(">II", len(data), 0) + data)
            start = heartbeat = time.monotonic()
            try:
                while time.monotonic() - start < 90:
                    try:
                        data, cursor, done = store.fragments(scene, session, cursor, 1)
                    except FileNotFoundError:
                        data, done = b"", False
                    if data:
                        send(data)
                    if done:
                        notice("done")
                        break
                    if data:
                        continue
                    if time.monotonic() - heartbeat >= 1:
                        notice("heartbeat")
                        heartbeat = time.monotonic()
                    time.sleep(.008)
                else:
                    notice("lease")
                self.wfile.write(b"0\r\n\r\n")
            except (OSError, ValueError):
                pass

    servers = []
    try:
        for port in (options["port"], options["media_port"]):
            server = Server((options["host"], port), Handler)
            servers.append(server)
            threading.Thread(target=server.serve_forever, daemon=True).start()
        return servers
    except BaseException:
        for server in servers:
            server.shutdown()
            server.server_close()
        raise
