"""Shared scene manifest validation and content identity; no model selection."""
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
                from deploy.solar_wm.shared.examples import bundled_cases
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
