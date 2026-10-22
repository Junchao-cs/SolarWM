"""CPU-only UI state. HTTP controls never wait for GPU or video I/O."""
from collections import deque
import copy
import math
import queue
import threading
import time
import uuid


class State:
    def __init__(self, scenes, deployment, options):
        self.lock = threading.RLock()
        self.commands = queue.Queue(maxsize=1)
        self.scenes, self.options = dict(scenes), options
        self.held = set()
        self.produced = deque(maxlen=256)
        vae_options = [dict(id="current", label="Original streaming VAE", ranks=options["vae_ranks"])]
        if options.get("streamrender_weights"):
            vae_options.append(dict(id="streamrender", label="StreamRender VAE", ranks=8))
        self.row = dict(deployment_id=deployment, ready=False, active=False, start_pending=False,
            paused=False, phase="loading", chunk=0, frames=0, max_chunks=None, error=None,
            scene_error=None, scene_preparing=False, scene_uploading=False, scene_id=next(iter(scenes)),
            scene_preparation=None, scene_preparation_required=True, gallery_generation=True,
            stream_sequence=0, stream_ready=False, stream_frames=0, video_ready=False,
            stop_requested=False, preview_resolution="full", preview_width=1344, preview_height=768,
            dit_ms=None, vae_ms=None, warmup_chunk=0, warmup_total=9, warmup_verified=False,
            move_speed=1., turn_speed=1., held_keys="", navigation_controls=True,
            movement_speed_max=10, turning_speed_max=3, continuous=True,
            vae_backend=options["vae_backend"], vae_options_ready=False,
            vae_selection_locked=False, vae_options=vae_options,
            dit_backend=options["acceleration"], dit_vae_order="serial on same eight GPUs")

    def update(self, **values):
        with self.lock:
            self.row.update(values)

    def scene_snapshot(self):
        with self.lock:
            return copy.deepcopy(self.scenes)

    def upload(self, body):
        from ..scene_inputs import store_upload
        with self.lock:
            if (self.row["active"] or self.row["start_pending"] or self.row["scene_uploading"]
                    or self.commands.full() or (self.row["scene_preparation"] or {}).get("busy")):
                raise RuntimeError("stop the current session or preparation before adding a scene")
            self.row["scene_uploading"] = True
        try:
            scene = store_upload(self.options, body)
            with self.lock:
                self.scenes[scene["scene_id"]] = scene
            return dict(scene_id=scene["scene_id"])
        finally:
            self.update(scene_uploading=False)

    def snapshot(self):
        with self.lock:
            row = copy.deepcopy(self.row)
            now = time.monotonic()
            samples = [x for x in self.produced if x[0] >= now - 8]
            row["generated_fps"] = ((samples[-1][1] - samples[0][1]) /
                (samples[-1][0] - samples[0][0])) if len(samples) > 1 and now - samples[-1][0] < 2 else 0.
            row["held_keys"] = "".join(sorted(self.held))
            return row

    def prepare_progress(self, stage, chunk=0, ready=False, error=None):
        with self.lock:
            p = self.row["scene_preparation"]
            if p is None:
                raise RuntimeError("owned preparation required")
            elapsed = time.monotonic() - p["started"]
            p.update(stage=stage, chunk=chunk, busy=not ready and not error and stage not in ("cancelled", "failed"),
                     ready=ready, error=error, elapsed_s=elapsed,
                     estimated_remaining_s=None, estimate_overrun=False)
            self.row.update(scene_preparing=p["busy"], scene_error=error,
                            warmup_chunk=chunk, warmup_verified=ready)

    def request(self, route, body):
        with self.lock:
            r = self.row
            if route == "/stop":
                self.held.clear()
                r["stop_requested"] = True
                if r["scene_preparation"]:
                    r["scene_preparation"]["cancel_requested"] = True
                return self.snapshot()
            if route == "/control":
                key, down = body.get("key"), body.get("down")
                if type(key) is not str or len(key) != 1 or key not in "qweasdijkl" or type(down) is not bool:
                    raise ValueError("one camera key and boolean required")
                if r["active"] and not r["stop_requested"]:
                    self.held.add(key) if down else self.held.discard(key)
                return self.snapshot()
            if route == "/navigation":
                if body.get("idle") is True:
                    self.held.clear()
                else:
                    for name, maximum in (("move_speed", 10), ("turn_speed", 3)):
                        value = body.get(name)
                        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value <= maximum:
                            raise ValueError("invalid " + name)
                    r.update(move_speed=body["move_speed"], turn_speed=body["turn_speed"])
                return self.snapshot()
            if route in ("/pause", "/resume"):
                if not r["active"]:
                    raise ValueError("no active session")
                r["paused"] = route == "/pause"
                self.held.clear()
                return self.snapshot()
            if route not in ("/prepare", "/start"):
                raise ValueError("unknown command")
            if not r["ready"] or r["active"] or r["start_pending"] or r["scene_uploading"] or self.commands.full():
                raise RuntimeError("model is busy")
            if (r["scene_preparation"] or {}).get("busy"):
                raise RuntimeError("scene is preparing")
            scene = body.get("scene_id")
            if scene not in self.scenes:
                raise ValueError("unknown scene")
            if route == "/prepare":
                r.update(stop_requested=False, scene_id=scene, scene_error=None, video_ready=False,
                    scene_preparation=dict(id=uuid.uuid4().hex, scene_id=scene, stage="queued",
                        started=time.monotonic(), busy=True, ready=False, chunk=0,
                        warmup_min_chunks=9, warmup_max_chunks=14, elapsed_s=0,
                        estimated_remaining_s=None, estimate_overrun=False, cancel_requested=False, error=None))
            else:
                p = r["scene_preparation"]
                if not p or not p["ready"] or p["scene_id"] != scene or body.get("preparation_id") != p["id"]:
                    raise ValueError("prepare this scene before starting")
                backend = body.get("vae_backend", self.options["vae_backend"])
                if backend not in ("current", "streamrender") or (
                    backend == "streamrender" and not self.options.get("streamrender_weights")):
                    raise ValueError("VAE backend is unavailable")
                resolution = body.get("preview_resolution", "full")
                if resolution not in ("full", "half"):
                    raise ValueError("invalid resolution")
                r.update(start_pending=True, stop_requested=False, video_ready=False, error=None,
                         vae_backend=backend, preview_resolution=resolution,
                         preview_width=1344 if resolution == "full" else 672,
                         preview_height=768 if resolution == "full" else 384)
            self.commands.put_nowait(dict(op=route[1:], scene_id=scene))
            return self.snapshot()

    def next_command(self, index):
        with self.lock:
            r = self.row
            if r["active"]:
                if r["stop_requested"]:
                    return dict(op="stop")
                if r["paused"]:
                    return dict(op="idle")
                return dict(op="chunk", actions=[sorted(self.held) for _ in range(4 if index == 0 else 5)],
                            move=r["move_speed"], turn=r["turn_speed"])
            try:
                command = self.commands.get_nowait()
                if command["op"] == "start" and r["stop_requested"]:
                    r["start_pending"] = False
                    return dict(op="idle")
                command.update(backend=r["vae_backend"], resolution=r["preview_resolution"])
                return command
            except queue.Empty:
                return dict(op="idle")

    def record_frame(self, index):
        with self.lock:
            if index != self.row["stream_frames"]:
                raise RuntimeError("video frame order changed")
            self.row["stream_frames"] += 1
            self.row["stream_ready"] = True

    def generated(self, chunks, dit, vae):
        with self.lock:
            self.row.update(chunk=chunks, frames=chunks * 17, dit_ms=round(dit, 2), vae_ms=round(vae, 2))
            self.produced.append((time.monotonic(), chunks * 17))
