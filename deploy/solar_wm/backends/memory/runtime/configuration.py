"""Validate memory deployment configuration; scene utilities are shared."""
import json
from pathlib import Path
from deploy.solar_wm.shared.configuration import file_sha256 as file_sha256, load_scenes as load_scenes


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
    if data["acceleration"] not in ("compile_graph", "tuned", "memory_native"):
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
    if data["acceleration"] != "memory_native":
        raise ValueError("this entry requires the official Memory loader, not a legacy non-Memory engine")
    data.setdefault("request_gc", "freeze")
    if data["request_gc"] not in ("observe", "freeze"):
        raise ValueError("request_gc must be observe or freeze")
    data["configuration_path"] = str(path)
    return data
