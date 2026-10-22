"""Deployment data is explicit JSON, never an experiment-directory import."""
import hashlib
import json
import re
from pathlib import Path

LEGACY_BUNDLED_SCENES = frozenset({
    "snow", "city_street", "indoor_architecture", "natural_landscape",
    "industrial_facility", "mountain", "rail_station", "harbor_road",
    "fantasy_world", "cinematic_3d_environment",
})


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    path = Path(path).resolve(strict=True)
    data = json.loads(path.read_text())
    required = {"model_config", "silence", "scenes", "state_directory", "cache_directory"}
    if not required <= data.keys():
        raise ValueError("missing deployment fields: " + str(sorted(required - data.keys())))
    for name in required | ({"streamrender_weights", "operator_profile"} & data.keys()):
        value = Path(data[name]).expanduser()
        data[name] = str((path.parent / value).resolve()) if not value.is_absolute() else str(value.resolve())
    for name in ("model_config", "silence", "scenes"):
        if not Path(data[name]).is_file():
            raise FileNotFoundError(data[name])
    data.setdefault("host", "127.0.0.1")
    data.setdefault("port", 8900)
    data.setdefault("media_port", 8901)
    data.setdefault("vae_backend", "current")
    data.setdefault("vae_ranks", 8)
    data.setdefault("graph_capacity", 2)
    data.setdefault("acceleration", "compile_graph")
    if data["acceleration"] not in ("compile_graph", "tuned"):
        raise ValueError("acceleration must be compile_graph or tuned")
    if data["vae_backend"] not in ("current", "streamrender"):
        raise ValueError("unknown VAE")
    if data["vae_backend"] == "streamrender" and not data.get("streamrender_weights"):
        raise ValueError("streamrender_weights is required for StreamRender")
    if data["acceleration"] == "tuned" and not data.get("operator_profile"):
        raise ValueError("tuned acceleration requires an explicit operator_profile")
    if type(data["graph_capacity"]) is not int or data["graph_capacity"] < 1:
        raise ValueError("positive graph_capacity required")
    if data["vae_ranks"] not in (7, 8):
        raise ValueError("current VAE supports seven or eight ranks")
    if data["host"] not in ("127.0.0.1", "::1"):
        raise ValueError("use an authenticated reverse proxy for shared access")
    if data["port"] == data["media_port"]:
        raise ValueError("control and media ports must be separate")
    data["configuration_path"] = str(path)
    return data


def load_scenes(path):
    path = Path(path).resolve()
    rows = json.loads(path.read_text())
    rows = rows.get("items", []) if isinstance(rows, dict) else rows
    scenes = {}
    seen = set()
    examples = None
    for row in rows:
        row = dict(row)
        identity = row.get("scene_id", row.get("id"))
        if not isinstance(identity, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", identity) or identity in seen:
            raise ValueError("unique scene id required")
        seen.add(identity)
        row["scene_id"] = identity
        for key in ("path", "first"):
            if key in row:
                p = Path(row[key])
                row[key] = str((path.parent / p).resolve() if not p.is_absolute() else p.resolve())
        legacy = ("solarwm", "realtime_interaction", "cases", identity, "first.png")
        if identity in LEGACY_BUNDLED_SCENES and Path(row.get("first", "")).parts[-5:] == legacy:
            # Adapt old deployment manifests in memory. Keep user prompts,
            # prepared conditions, uploads and the manifest on disk untouched.
            if examples is None:
                from ..examples import bundled_cases
                examples = bundled_cases()
            if identity not in examples:
                continue
            row["first"] = examples[identity]["first"]
        if "path" in row:
            digest = file_sha256(row["path"])
            if row.get("sha256", digest) != digest:
                raise ValueError("scene condition hash differs: " + identity)
            row["sha256"] = digest
        elif (not isinstance(row.get("prompt"), str) or not row["prompt"].strip()
              or not row.get("first") or not Path(row["first"]).is_file()):
            raise ValueError("an image and nonempty prompt are required for " + identity)
        scenes[identity] = row
    if not scenes:
        raise ValueError("at least one image/prompt scene is required")
    return scenes
