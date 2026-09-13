"""Command-line interface for training, inference, preencoding, and data inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from solarwm import __version__
from solarwm.config import load_config
from solarwm.config.routes import supported_routes, validate_route
from solarwm.data.index import inventory, read_index
from solarwm.data.sampling import (
    CanonicalSampler,
    ReaderIdentity,
    SamplingConfig,
    plan_fingerprint,
)
from solarwm.errors import SolarWMError
from solarwm.runtime.create_only import write_file_create_only
from solarwm.runtime.images import probe_python_runtime
from solarwm.runtime.output_layout import (
    camera_inference_output_layout,
    invocation_output_dir,
)
from solarwm.runtime.provenance import build_launch_manifest, reject_inline_secrets


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one dotted config value; may be repeated",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="solarwm")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    for action in ("train", "infer", "preencode"):
        child = commands.add_parser(action, help=f"run the unified {action} entrypoint")
        _common_config(child)

    config = commands.add_parser("config", help="resolve and inspect a run config")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    resolve = config_commands.add_parser("resolve")
    _common_config(resolve)
    resolve.add_argument("--output", type=Path)
    config_commands.add_parser("routes")

    data = commands.add_parser("data", help="inspect canonical index semantics")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    inspect = data_commands.add_parser("inspect")
    inspect.add_argument("index", type=Path)
    warm = data_commands.add_parser(
        "warm-cache", help="optionally prepare an explicit shard working set before training"
    )
    warm.add_argument("index", type=Path, help="recipe index or node-specific planned subset")
    warm.add_argument("--root", required=True, help="GCS root used by training")
    warm.add_argument("--cache-dir", required=True, type=Path)
    warm.add_argument("--max-gib", required=True, type=float)
    warm.add_argument("--workers", type=int, default=4)
    warm.add_argument("--timeout-seconds", type=float, default=7200.0)
    windows = data_commands.add_parser(
        "materialize-wan153f",
        help="join raw Wan 153f sources to the released fixed-window indexes",
    )
    windows.add_argument("--train-index", required=True, type=Path)
    windows.add_argument("--test-index", required=True, type=Path)
    windows.add_argument("--train-window-index", required=True, type=Path)
    windows.add_argument("--test-window-index", required=True, type=Path)
    windows.add_argument("--output", required=True, type=Path)
    plan = data_commands.add_parser("plan")
    plan.add_argument("index", type=Path)
    plan.add_argument("--seed", type=int, required=True)
    plan.add_argument("--pixel-frames", type=int, required=True)
    plan.add_argument("--rank", type=int, default=0)
    plan.add_argument("--world-size", type=int, default=1)
    plan.add_argument("--worker-id", type=int, default=0)
    plan.add_argument("--num-workers", type=int, default=1)
    plan.add_argument("--epoch", type=int, default=0)
    plan.add_argument("--limit", type=int, default=16)
    plan.add_argument("--shuffle-buffer", type=int, default=4096)

    environment = commands.add_parser("environment", help="inspect the current runtime")
    environment_commands = environment.add_subparsers(dest="environment_command", required=True)
    environment_commands.add_parser("probe")

    return parser


def _write_json(value: Any, output: Path | None = None) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if output is None:
        sys.stdout.write(payload)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _run_backend(args: argparse.Namespace) -> int:
    resolved = load_config(args.config, args.set)
    reject_inline_secrets(resolved.values)
    configured_action = str(resolved.values["action"])
    if configured_action != args.command:
        raise SolarWMError(
            f"command {args.command!r} does not match config action {configured_action!r}"
        )
    route = validate_route(resolved.values)

    # Wan's masked training paths select whether FlexAttention is compiled
    # when the model implementation is imported.  Backends are intentionally
    # loaded lazily below, so make the validated config authoritative before
    # that import rather than requiring a second launcher-only switch.
    runtime = resolved.values.get("runtime", {})
    if route.family.startswith("wan22_"):
        if runtime.get("compile_flex") is True:
            os.environ["SOLARWM_COMPILE_FLEX"] = "1"
        else:
            os.environ.pop("SOLARWM_COMPILE_FLEX", None)

    from solarwm.backends import load_backend

    backend = load_backend(route.family)
    backend.validate_config(resolved.values)
    if not str(runtime.get("output_dir", "")).strip():
        raise SolarWMError("runtime.output_dir is required")
    publication_layout = camera_inference_output_layout(resolved.values)
    output_dir = invocation_output_dir(resolved.values)
    launcher_rank = int(os.environ.get("RANK", "0"))
    if launcher_rank == 0:
        if publication_layout is not None:
            publication_layout.publish_root.mkdir(parents=True, exist_ok=True)
            output_dir.parent.mkdir(parents=True, exist_ok=True)
            try:
                output_dir.mkdir()
            except FileExistsError as exc:
                raise SolarWMError(f"camera inference run already exists: {output_dir}") from exc
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            # A terminal result describes only the current invocation. Remove
            # a result left by an earlier legacy invocation before work starts.
            (output_dir / "run-result.json").unlink(missing_ok=True)
        resolved.write_json(output_dir / "resolved-config.json")
        manifest = build_launch_manifest(
            config=resolved.values,
            source_config=resolved.path,
            source_digest=resolved.source_digest,
            resolved_digest=resolved.resolved_digest,
            route=route.key,
            repository=Path(__file__).resolve().parents[2],
        )
        _write_json(manifest, output_dir / "launch-manifest.json")
    operation = getattr(backend, args.command)
    try:
        result = int(operation(resolved.values) or 0)
    except Exception as exc:
        if launcher_rank == 0:
            _write_json(
                {
                    "schema": "solarwm.run-result.v1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
                output_dir / "run-result.json",
            )
        raise
    if launcher_rank == 0:
        _write_json(
            {"schema": "solarwm.run-result.v1", "status": "complete", "exit_code": result},
            output_dir / "run-result.json",
        )
        if publication_layout is not None and result == 0:
            publication_complete = output_dir / "publication" / "COMPLETE.json"
            if not publication_complete.is_file():
                raise SolarWMError(
                    "camera inference returned without a publication completion marker"
                )
            publication_payload = publication_complete.read_bytes()
            run_result_payload = (output_dir / "run-result.json").read_bytes()
            complete_payload = (
                json.dumps(
                    {
                        "schema": "solarwm.camera-inference-run-complete.v1",
                        "layout": publication_layout.layout,
                        "run_id": publication_layout.run_id,
                        "resolved_config_digest": resolved.resolved_digest,
                        "publication_complete_digest": hashlib.blake2s(
                            publication_payload
                        ).hexdigest(),
                        "run_result_digest": hashlib.blake2s(run_result_payload).hexdigest(),
                    },
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                + "\n"
            ).encode("utf-8")
            write_file_create_only(
                output_dir / "COMPLETE.json",
                complete_payload,
                error_type=SolarWMError,
                label="camera inference run completion marker",
            )
    return result


def _config_command(args: argparse.Namespace) -> int:
    if args.config_command == "routes":
        _write_json([route.key for route in supported_routes()])
        return 0
    resolved = load_config(args.config, args.set)
    reject_inline_secrets(resolved.values)
    route = validate_route(resolved.values)
    from solarwm.backends import load_backend

    load_backend(route.family).validate_config(resolved.values)
    payload = {
        "source": str(resolved.path),
        "source_digest": resolved.source_digest,
        "resolved_digest": resolved.resolved_digest,
        "route": route.key,
        "config": resolved.values,
    }
    _write_json(payload, args.output)
    return 0


def _data_command(args: argparse.Namespace) -> int:
    if args.data_command == "warm-cache":
        from dataclasses import asdict

        from solarwm.data.prefetch import warm_shard_cache
        from solarwm.data.transport import GCSResolver

        if not math.isfinite(args.max_gib) or args.max_gib <= 0:
            raise SolarWMError("--max-gib must be finite and positive")
        max_bytes = int(args.max_gib * 1024**3)
        resolver = GCSResolver(root=args.root, cache_dir=args.cache_dir, max_bytes=max_bytes)
        result = warm_shard_cache(
            read_index(args.index),
            resolver,
            max_bytes=max_bytes,
            max_workers=args.workers,
            timeout_seconds=args.timeout_seconds,
            progress=lambda done, total: print(
                f"cache warmup: {done}/{total} shards ready", file=sys.stderr, flush=True
            ),
        )
        _write_json(asdict(result))
        return 0

    if args.data_command == "materialize-wan153f":
        from solarwm.backends.wan22.windows import write_wan153f_window_index

        summary = write_wan153f_window_index(
            args.train_index,
            args.test_index,
            args.train_window_index,
            args.test_window_index,
            args.output,
        )
        _write_json(summary.as_dict())
        return 0

    rows = read_index(args.index)
    report = inventory(args.index, rows)
    if args.data_command == "inspect":
        _write_json(
            {
                "index": str(args.index.resolve()),
                "rows": report.rows,
                "virtual_occurrences": report.virtual_occurrences,
                "ordered_sample_id_digest": report.ordered_sample_id_digest,
                "ordered_row_digest": report.ordered_row_digest,
                "decompressed_digest": report.decompressed_digest,
            }
        )
        return 0

    config = SamplingConfig(
        seed=args.seed,
        pixel_frames=args.pixel_frames,
        shuffle_buffer=args.shuffle_buffer,
    )
    identity = ReaderIdentity(
        rank=args.rank,
        world_size=args.world_size,
        worker_id=args.worker_id,
        num_workers=args.num_workers,
    )
    plans = list(CanonicalSampler(rows, config, identity).iter_epoch(args.epoch))
    selected = plans[: max(0, args.limit)]
    _write_json(
        {
            "full_plan_fingerprint": plan_fingerprint(plans),
            "full_plan_count": len(plans),
            "shown": [
                {
                    "sample_id": plan.sample_id,
                    "key": plan.key,
                    "shard": plan.shard,
                    "repeat_ordinal": plan.repeat_ordinal,
                    "start_frame": plan.start_frame,
                    "source_frame_indices": plan.source_frame_indices,
                }
                for plan in selected
            ],
        }
    )
    return 0


def _environment_command(args: argparse.Namespace) -> int:
    if args.environment_command == "probe":
        _write_json(probe_python_runtime())
        return 0
    raise SolarWMError(f"unhandled environment command {args.environment_command!r}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command in {"train", "infer", "preencode"}:
            return _run_backend(args)
        if args.command == "config":
            return _config_command(args)
        if args.command == "data":
            return _data_command(args)
        if args.command == "environment":
            return _environment_command(args)
        raise SolarWMError(f"unhandled command {args.command!r}")
    except (SolarWMError, OSError, ValueError) as exc:
        print(f"solarwm: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
