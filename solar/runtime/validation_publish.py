"""Publish node-local comparison outputs into one canonical validation tree."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from solar.errors import ConfigurationError
from solar.runtime.gcs_upload import ObjectInfo, PublicationError
from solar.runtime.provenance import reject_inline_secrets

_STEP = re.compile(r"step_(\d{6,})_([A-Za-z0-9][A-Za-z0-9_.-]*)")
_VIDEO = re.compile(r"rank(\d+)_[A-Za-z0-9_.-]+\.mp4")
_MANIFEST = re.compile(r"rank_(\d+)\.json")
_LAYOUT = "ground_truth_left_generated_right"
_RECEIPT_SCHEMA = "solarwm.validation-node-receipt.v1"
_SENSITIVE_KEYS = {"authorization", "private_key", "client_secret", "session_url", "signed_url"}


class PublicationStore(Protocol):
    """Only the immutable object operations needed by validation publication."""

    def put_file(self, key: str, path: Path, expected: ObjectInfo) -> None: ...
    def put_bytes(self, key: str, payload: bytes) -> None: ...
    def read(self, key: str) -> bytes: ...
    def list(self, prefix: str) -> dict[str, ObjectInfo]: ...


def _encode(value: Mapping) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _read_json(payload: bytes) -> dict:
    try:
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise ValueError
        return value
    except (ValueError, UnicodeError):
        raise PublicationError("invalid_publication_json") from None


def _check_credentials(value: object) -> None:
    try:
        reject_inline_secrets(value)
    except ConfigurationError:
        raise PublicationError("credential_in_publication_json") from None
    if isinstance(value, dict):
        for key, child in value.items():
            if key.lower() in _SENSITIVE_KEYS and child:
                raise PublicationError("credential_in_publication_json")
            _check_credentials(child)
    elif isinstance(value, list):
        for child in value:
            _check_credentials(child)


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PublicationError("invalid_publication_integer")
    return value


def _regular(path: Path, root: Path, *, directory: bool = False) -> None:
    current = path
    while current != root:
        if root not in current.parents or current.is_symlink():
            raise PublicationError("unsafe_publication_path")
        current = current.parent
    if not (path.is_dir() if directory else path.is_file()):
        raise PublicationError("missing_publication_file")


def _files(directory: Path, group: str, root: Path) -> dict[int, Path]:
    folder = directory / group
    if not folder.exists() and not folder.is_symlink():
        return {}
    _regular(folder, root, directory=True)
    pattern = _VIDEO if group == "compare" else _MANIFEST
    result = {}
    for path in sorted(folder.iterdir()):
        match = pattern.fullmatch(path.name)
        if match is None:
            # Never recurse into, or upload, arbitrary adjacent files.
            continue
        _regular(path, root)
        slot = int(match[1])
        if slot in result:
            raise PublicationError("duplicate_local_slot")
        result[slot] = path
    return result


def _sample_identity(value: dict, *, slot: int, step: int, pass_name: str) -> None:
    if (
        _integer(value.get("logical_validation_rank")) != slot
        or _integer(value.get("step")) != step
        or value.get("active_pass_name") != pass_name
    ):
        raise PublicationError("sample_identity_mismatch")


@dataclass(frozen=True)
class Partition:
    prefix: str
    receipt: dict
    files: tuple[tuple[str, Path, ObjectInfo], ...]


def _partition(root: Path, directory: Path, node_rank: int, num_nodes: int) -> Partition:
    match = _STEP.fullmatch(directory.name)
    if match is None:
        raise PublicationError("invalid_validation_directory")
    step, pass_name = int(match[1]), match[2]
    _regular(directory / "COMPLETE.json", root)
    marker = _read_json((directory / "COMPLETE.json").read_bytes())
    if (
        marker.get("schema") != "solarwm.validation-comparison.v1"
        or _integer(marker.get("step")) != step
        or marker.get("pass") != pass_name
        or marker.get("compare_layout") != _LAYOUT
    ):
        raise PublicationError("local_marker_mismatch")
    total = _integer(marker.get("global_slots"), minimum=1)
    videos = _files(directory, "compare", root)
    manifests = _files(directory, "manifests", root)
    errors = _files(directory, "errors", root)
    if (
        videos.keys() != manifests.keys()
        or len(manifests) != _integer(marker.get("local_slots"))
        or videos.keys() & errors.keys()
        or any(slot >= total for slot in {*videos, *errors})
    ):
        raise PublicationError("local_slot_inventory_mismatch")
    declared_failures = marker.get("failed_slots", [])
    if not isinstance(declared_failures, list) or not set(errors) <= {
        _integer(slot) for slot in declared_failures
    }:
        raise PublicationError("local_failure_inventory_mismatch")
    for slot, path in manifests.items():
        value = _read_json(path.read_bytes())
        _check_credentials(value)
        _sample_identity(value, slot=slot, step=step, pass_name=pass_name)
        if value.get("error") or value.get("artifact_valid") is False:
            raise PublicationError("invalid_successful_sample")
        expected_video = videos[slot].relative_to(root).as_posix()
        if value.get("compare_mp4") != expected_video:
            raise PublicationError("sample_video_path_mismatch")
    for slot, path in errors.items():
        value = _read_json(path.read_bytes())
        _check_credentials(value)
        _sample_identity(value, slot=slot, step=step, pass_name=pass_name)
        if not value.get("error"):
            raise PublicationError("invalid_failure_sample")
    prefix = directory.relative_to(root).as_posix()
    files = []
    identities = {}
    for group in (videos, manifests, errors):
        for path in group.values():
            info = ObjectInfo.from_file(path)
            if info.size == 0:
                raise PublicationError("empty_publication_object")
            key = path.relative_to(directory).as_posix()
            identities[key] = asdict(info)
            files.append((prefix + "/" + key, path, info))
    receipt = {
        "schema": _RECEIPT_SCHEMA,
        "node_rank": node_rank,
        "num_nodes": num_nodes,
        "step": step,
        "pass": pass_name,
        "global_slots": total,
        "successful_slots": sorted(videos),
        "failed_slots": sorted(errors),
        "files": identities,
    }
    return Partition(prefix, receipt, tuple(files))


def _receipt_name(prefix: str, rank: int) -> str:
    return f"{prefix}/receipts/node_{rank:03d}.json"


def _validate_receipt(receipt: dict, reference: dict, rank: int) -> None:
    for key in ("schema", "num_nodes", "step", "pass", "global_slots"):
        if type(receipt.get(key)) is not type(reference[key]) or receipt[key] != reference[key]:
            raise PublicationError("node_receipt_identity_mismatch")
    if _integer(receipt.get("node_rank")) != rank:
        raise PublicationError("node_receipt_identity_mismatch")
    successes, failures = receipt.get("successful_slots"), receipt.get("failed_slots")
    if not isinstance(successes, list) or not isinstance(failures, list):
        raise PublicationError("invalid_node_receipt_slots")
    slots = [_integer(slot) for slot in successes + failures]
    if len(set(slots)) != len(slots) or any(slot >= reference["global_slots"] for slot in slots):
        raise PublicationError("invalid_node_receipt_slots")
    files = receipt.get("files")
    if not isinstance(files, dict):
        raise PublicationError("invalid_node_receipt_files")
    groups: dict[str, list[int]] = {"compare": [], "manifests": [], "errors": []}
    for key, value in files.items():
        parts = key.split("/")
        if len(parts) != 2 or parts[0] not in groups:
            raise PublicationError("invalid_node_receipt_files")
        pattern = _VIDEO if parts[0] == "compare" else _MANIFEST
        match = pattern.fullmatch(parts[1])
        if match is None or not isinstance(value, dict):
            raise PublicationError("invalid_node_receipt_files")
        _integer(value.get("size"), minimum=1)
        if not isinstance(value.get("md5"), str):
            raise PublicationError("invalid_node_receipt_files")
        groups[parts[0]].append(int(match[1]))
    if (
        sorted(groups["compare"]) != sorted(successes)
        or sorted(groups["manifests"]) != sorted(successes)
        or sorted(groups["errors"]) != sorted(failures)
    ):
        raise PublicationError("node_receipt_slot_files_mismatch")


def finalize_validation(store: PublicationStore, prefix: str, reference: dict) -> str | None:
    """Commit only the exact union of all node partitions; return None while pending."""

    objects = store.list(prefix + "/")
    receipt_names = {_receipt_name(prefix, rank) for rank in range(reference["num_nodes"])}
    present_receipts = {key for key in objects if key.startswith(prefix + "/receipts/")}
    if present_receipts - receipt_names:
        raise PublicationError("unexpected_node_receipt")
    if not receipt_names <= objects.keys():
        if prefix + "/COMPLETE.json" in objects:
            raise PublicationError("global_marker_without_receipts")
        return None
    successes, failures, files = [], [], {}
    for rank in range(reference["num_nodes"]):
        key = _receipt_name(prefix, rank)
        raw = store.read(key)
        if ObjectInfo.from_bytes(raw) != objects[key]:
            raise PublicationError("node_receipt_changed")
        receipt = _read_json(raw)
        _validate_receipt(receipt, reference, rank)
        successes.extend(receipt["successful_slots"])
        failures.extend(receipt["failed_slots"])
        for relative, identity in receipt["files"].items():
            key = prefix + "/" + relative
            if key in files:
                raise PublicationError("duplicate_global_slot")
            files[key] = ObjectInfo(identity["size"], identity["md5"])
    if sorted(successes + failures) != list(range(reference["global_slots"])):
        raise PublicationError("global_slot_inventory_mismatch")
    artifact_objects = {
        key: value
        for key, value in objects.items()
        if any(
            key.startswith(prefix + "/" + group + "/")
            for group in ("compare", "manifests", "errors")
        )
    }
    if artifact_objects != files:
        raise PublicationError("global_object_inventory_mismatch")
    store.put_bytes(
        prefix + "/COMPLETE.json",
        _encode(
            {
                "schema": "solarwm.gcs-validation-complete.v1",
                "step": reference["step"],
                "pass": reference["pass"],
                "global_slots": reference["global_slots"],
                "successful_slots": len(successes),
                "failed_slots": sorted(failures),
                "status": "partial" if failures else "complete",
                "nodes": reference["num_nodes"],
                "compare_layout": _LAYOUT,
            }
        ),
    )
    return "partial" if failures else "complete"


class ValidationPublisher:
    """One independent publisher per local output tree; node zero coordinates."""

    def __init__(
        self, run_dir: Path, store: PublicationStore, *, node_rank: int, num_nodes: int
    ) -> None:
        if (
            not run_dir.is_absolute()
            or type(num_nodes) is not int
            or num_nodes < 1
            or type(node_rank) is not int
            or not 0 <= node_rank < num_nodes
        ):
            raise PublicationError("invalid_publisher_arguments")
        self.root = run_dir.resolve()
        self.store, self.node_rank, self.num_nodes = store, node_rank, num_nodes
        self._published: dict[str, tuple[bytes, dict]] = {}
        self._finalized: dict[str, str] = {}

    def scan(self) -> dict:
        result: dict = {
            "node_rank": self.node_rank,
            "partitions": [],
            "errors": 0,
            "pending": 0,
            "partial": 0,
        }
        validation = self.root / "validation"
        if not validation.exists() and not validation.is_symlink():
            result["pending"] = 1
            return result
        _regular(validation, self.root, directory=True)
        for directory in sorted(validation.iterdir()):
            if not _STEP.fullmatch(directory.name):
                continue
            status = {"partition": directory.name, "status": "pending"}
            try:
                _regular(directory, self.root, directory=True)
                marker_path = directory / "COMPLETE.json"
                if not marker_path.exists():
                    result["pending"] += 1
                    result["partitions"].append(status)
                    continue
                _regular(marker_path, self.root)
                marker = marker_path.read_bytes()
                cached = self._published.get(directory.name)
                if cached is not None and cached[0] != marker:
                    raise PublicationError("local_marker_changed")
                if cached is None:
                    partition = _partition(self.root, directory, self.node_rank, self.num_nodes)
                    for key, path, info in partition.files:
                        self.store.put_file(key, path, info)
                    self.store.put_bytes(
                        _receipt_name(partition.prefix, self.node_rank), _encode(partition.receipt)
                    )
                    self._published[directory.name] = (marker, partition.receipt)
                reference = self._published[directory.name][1]
                if self.node_rank == 0:
                    state = self._finalized.get(directory.name)
                    if state is None:
                        state = finalize_validation(
                            self.store, "validation/" + directory.name, reference
                        )
                        if state is not None:
                            self._finalized[directory.name] = state
                    status["status"] = state or "pending"
                    if state is None:
                        result["pending"] += 1
                    elif state == "partial":
                        result["partial"] += 1
                else:
                    status["status"] = "uploaded"
            except PublicationError as exc:
                status.update(status="error", error=str(exc))
                result["errors"] += 1
            except (OSError, ValueError, TypeError, KeyError):
                status.update(status="error", error="publication_io_or_schema_error")
                result["errors"] += 1
            result["partitions"].append(status)
        if not result["partitions"]:
            result["pending"] = 1
        return result
