"""Image/prompt storage and isolated, cancellable conditioning preparation."""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import warnings

from .runtime.configuration import file_sha256, load_scenes

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_BODY_BYTES = 12 * 1024 * 1024


class PreparationCancelled(Exception):
    pass


def write_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(value, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def store_upload(config, body):
    """Accept bytes, never browser-supplied server paths or remote URLs."""
    from PIL import Image, ImageOps
    prompt, encoded = body.get("prompt"), body.get("image")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 4000:
        raise ValueError("enter a prompt (1–4000 characters)")
    if not isinstance(encoded, str) or len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
        raise ValueError("image must be at most 8 MiB")
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise ValueError("invalid image encoding") from error
    if not 0 < len(data) <= MAX_IMAGE_BYTES:
        raise ValueError("image must be at most 8 MiB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as source:
                if source.format not in ("PNG", "JPEG", "WEBP") or getattr(source, "n_frames", 1) != 1:
                    raise ValueError("use a single PNG, JPEG or WebP image")
                if source.width * source.height > 16_000_000:
                    raise ValueError("image must be at most 16 megapixels")
                image = ImageOps.exif_transpose(source).convert("RGB")
            normalized = io.BytesIO()
            image.save(normalized, format="PNG")
    except (OSError, Image.DecompressionBombWarning, Image.DecompressionBombError) as error:
        raise ValueError("invalid or oversized image") from error
    image_bytes = normalized.getvalue()
    prompt = prompt.strip()
    identity = "upload-" + hashlib.sha256(image_bytes + b"\0" + prompt.encode()).hexdigest()[:24]
    manifest = Path(config["scenes"])
    # Only CPU I/O under this lock; the encoder never holds it.
    import fcntl
    with manifest.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        scenes = load_scenes(manifest)
        if identity in scenes:
            return scenes[identity]
        directory = Path(config["state_directory"]) / "uploads"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (identity + ".png")
        # A previous interrupted registration may have already saved these bytes.
        if target.exists():
            if target.read_bytes() != image_bytes:
                raise ValueError("stored upload differs")
        else:
            with target.open("xb") as file:
                file.write(image_bytes)
                file.flush()
                os.fsync(file.fileno())
        row = dict(scene_id=identity, title="My scene " + identity[-6:],
                   first=str(target), prompt=prompt)
        scenes[identity] = row
        write_json(manifest, list(scenes.values()))
        return row


def cache_identity(config, row):
    model_path = Path(config["model_config"])
    model = json.loads(model_path.read_text())
    base = Path(model["model"]["checkpoint_path"])
    # Invalidate when installed conditioning files change in place. This is a
    # local cache stamp, not an attestation of checkpoint payload identity.
    installed = []
    for name in ("text_encoder", "tokenizer", "processor", "vae"):
        for path in sorted((base / name).rglob("*")):
            if path.is_file() and ".cache" not in path.parts:
                stat = path.stat()
                installed.append((str(path.relative_to(base)), stat.st_size, stat.st_mtime_ns))
    from . import prepare_scene
    inputs = dict(image=file_sha256(row["first"]), prompt=row["prompt"].strip(),
                  model=file_sha256(model_path), encoder=file_sha256(prepare_scene.__file__),
                  installed=installed)
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return Path(config["state_directory"]) / "scene-features" / key, inputs


def cached_scene(config, row):
    if row.get("path"):
        if file_sha256(row["path"]) != row["sha256"]:
            raise ValueError("prepared scene condition changed")
        return row
    directory, inputs = cache_identity(config, row)
    if not directory.exists():
        return None
    receipt = json.loads((directory / "encoding.json").read_text())
    result = json.loads((directory / "scene.json").read_text())
    if (receipt.get("cache_inputs") != json.loads(json.dumps(inputs))
            or result.get("path") != str(directory / "condition.safetensors")
            or result.get("sha256") != receipt.get("condition_sha256")
            or file_sha256(result["path"]) != result["sha256"]
            or file_sha256(directory / "first.png") != receipt.get("first_sha256")):
        raise ValueError("incomplete or changed scene cache; inspect " + str(directory))
    return dict(result, scene_id=row["scene_id"], title=row.get("title", row["scene_id"]))


def encode_cached(config, row, device="cuda:0"):
    """Called only in the short-lived encoder process (or optional prepare CLI)."""
    from .prepare_scene import encode_scene
    directory, inputs = cache_identity(config, row)
    directory.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (directory.parent / (directory.name + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = cached_scene(config, row)
        if existing is not None:
            return existing
        # Interrupted jobs never publish a cache entry. Keep their private
        # staging files for diagnosis; a subsequent request uses a fresh job.
        job = Path(tempfile.mkdtemp(prefix=".encoding-", dir=directory.parent))
        staging = job / "result"
        result = encode_scene(config["model_config"], row["first"], row["prompt"],
                              row["scene_id"], staging, device)
        if cache_identity(config, row) != (directory, inputs):
            raise ValueError("scene inputs changed during encoding")
        receipt = json.loads((staging / "encoding.json").read_text())
        receipt.update(cache_inputs=inputs, first_sha256=file_sha256(staging / "first.png"))
        result.update(path=str(directory / "condition.safetensors"), first=str(directory / "first.png"),
                      title=row.get("title", row["scene_id"]))
        write_json(staging / "scene.json", result)
        write_json(staging / "encoding.json", receipt)
        staging.rename(directory)
        job.rmdir()
        return cached_scene(config, row)


def encode_job(path):
    job = Path(path)
    request = json.loads(job.read_text())
    try:
        from .runtime.dependencies import install
        install(request["config"])
        row = encode_cached(request["config"], request["scene"])
        write_json(job.parent / "result.json", dict(scene=row))
    except Exception as error:
        write_json(job.parent / "result.json", dict(error=f"{type(error).__name__}: {error}"))
        raise


def prepare_isolated(config, row, cancelled, progress):
    """Rank zero only. Never enters a distributed collective or reloads DiT."""
    if cancelled():
        raise PreparationCancelled()
    existing = cached_scene(config, row)
    if existing is not None:
        return existing
    progress("encoding")
    root = Path(config["state_directory"]) / "scene-jobs"
    root.mkdir(parents=True, exist_ok=True)
    job = Path(tempfile.mkdtemp(prefix="request-", dir=root))
    write_json(job / "request.json", dict(config=config, scene=row))
    env = {k: v for k, v in os.environ.items() if k not in {
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
        "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
        "TORCHELASTIC_ERROR_FILE"}}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    with (job / "encoder.log").open("wb") as log:
        child = subprocess.Popen([sys.executable, "-B", "-m", "solarwm_web.prepare_scene",
            "--job", str(job / "request.json")], env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
        started = time.monotonic()
        try:
            while child.poll() is None:
                if cancelled():
                    raise PreparationCancelled()
                if time.monotonic() - started > 1800:
                    raise RuntimeError("scene encoding timed out; log: " + str(job / "encoder.log"))
                time.sleep(.1)
        finally:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=5)
        if cancelled():
            raise PreparationCancelled()
        result = json.loads((job / "result.json").read_text()) if (job / "result.json").is_file() else {}
        if child.returncode or result.get("error"):
            raise RuntimeError(result.get("error", "encoder exited " + str(child.returncode))
                               + "; log: " + str(job / "encoder.log"))
    prepared = cached_scene(config, row)
    if prepared is None or result.get("scene") != prepared:
        raise RuntimeError("encoder did not publish a valid scene")
    return prepared
