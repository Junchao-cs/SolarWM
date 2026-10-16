"""Safe YAML loading, environment expansion, overrides, and provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from solarwm.errors import ConfigurationError

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_TOP_LEVEL_KEYS = frozenset(
    {
        "schema",
        "action",
        "name",
        "model",
        "data",
        "distributed",
        "train",
        "validation",
        "checkpoint",
        "preencode",
        "inference",
        "runtime",
        "metadata",
    }
)


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate and non-string mapping keys."""


def _construct_mapping(
    loader: _StrictSafeLoader, node: yaml.nodes.MappingNode, deep: bool = False
) -> dict[str, Any]:
    loader.flatten_mapping(node)
    result: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigurationError(
                f"YAML mapping keys must be strings at line {key_node.start_mark.line + 1}"
            )
        if key in result:
            raise ConfigurationError(
                f"duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}"
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping
)


def _construct_json_mapping(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ConfigurationError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _expand_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in os.environ:
            raise ConfigurationError(f"environment variable {name!r} is not set")
        return os.environ[name]

    return _ENV.sub(replace, value)


def _parse_override(raw: str) -> tuple[list[str], Any]:
    if "=" not in raw:
        raise ConfigurationError(f"override must be KEY=VALUE, got {raw!r}")
    dotted, value = raw.split("=", 1)
    keys = [part for part in dotted.split(".") if part]
    if not keys:
        raise ConfigurationError(f"override has an empty key: {raw!r}")
    try:
        parsed = yaml.load(value, Loader=_StrictSafeLoader)
    except (yaml.YAMLError, ConfigurationError) as exc:
        raise ConfigurationError(f"invalid override {raw!r}: {exc}") from exc
    return keys, parsed


def _apply_override(config: MutableMapping[str, Any], raw: str) -> None:
    keys, value = _parse_override(raw)
    node: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        child = node.get(key)
        if child is None:
            child = {}
            node[key] = child
        if not isinstance(child, MutableMapping):
            raise ConfigurationError(f"override {raw!r} crosses non-mapping key {key!r}")
        node = child
    node[keys[-1]] = value


def canonical_json(config: Mapping[str, Any]) -> bytes:
    """Return stable bytes used by manifests and checkpoint provenance."""

    return (
        json.dumps(
            config,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


@dataclass(frozen=True)
class ResolvedConfig:
    """A validated config plus its exact source and resolved identities."""

    path: Path
    source_digest: str
    resolved_digest: str
    values: Mapping[str, Any]

    def mutable_copy(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.values))

    def write_json(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target.parent,
                prefix=f".{target.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = handle.name
                handle.write(canonical_json(self.values))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary = None
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


def _validate_json_value(value: Any, path: tuple[str, ...] = ()) -> None:
    label = ".".join(path) or "<root>"
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigurationError(f"non-finite number at {label}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(child, (*path, str(index)))
        return
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(f"non-string mapping key at {label}")
            _validate_json_value(child, (*path, key))
        return
    raise ConfigurationError(
        f"unsupported YAML value {type(value).__name__} at {label}; "
        "configs must be canonical JSON-compatible values"
    )


def _validate_top_level(config: Mapping[str, Any]) -> None:
    unknown = sorted(set(config) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigurationError(f"unknown top-level config keys: {unknown}")
    if config.get("schema") != "solarwm.run.v1":
        raise ConfigurationError("schema must be exactly 'solarwm.run.v1'")
    action = str(config.get("action", "")).strip().lower()
    if action not in {"train", "infer", "preencode"}:
        raise ConfigurationError("action must be one of: train, infer, preencode")
    if not str(config.get("name", "")).strip():
        raise ConfigurationError("name must be a non-empty run identifier")
    for section in ("model", "data", "runtime"):
        if not isinstance(config.get(section), Mapping):
            raise ConfigurationError(f"{section} must be a mapping")


def load_config(
    path: str | Path,
    overrides: Sequence[str] = (),
    *,
    route_validator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> ResolvedConfig:
    """Load a config without executing constructors or arbitrary code."""

    source_path = Path(path).resolve()
    raw = source_path.read_bytes()
    try:
        if source_path.suffix.lower() == ".json":
            loaded = json.loads(raw, object_pairs_hook=_construct_json_mapping)
        else:
            loaded = yaml.load(raw, Loader=_StrictSafeLoader)
    except (json.JSONDecodeError, UnicodeDecodeError, yaml.YAMLError, ConfigurationError) as exc:
        raise ConfigurationError(f"cannot parse {source_path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"{source_path} must contain a YAML mapping")
    values = _expand_env(loaded)
    for override in overrides:
        _apply_override(values, override)
    _validate_top_level(values)
    _validate_json_value(values)

    # Import lazily to keep configuration loading usable without torch.
    from .routes import validate_route

    (route_validator or validate_route)(values)
    resolved = canonical_json(values)
    return ResolvedConfig(
        path=source_path,
        source_digest=hashlib.blake2s(raw).hexdigest(),
        resolved_digest=hashlib.blake2s(resolved).hexdigest(),
        values=values,
    )
