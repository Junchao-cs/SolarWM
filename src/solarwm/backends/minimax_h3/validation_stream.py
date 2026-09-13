"""Fixed, source-balanced H3 validation directly from the released test indexes."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np

from solarwm.data import IndexRow, read_index, resolve_index_path
from solarwm.data.sampling import SamplePlan
from solarwm.errors import DataContractError
from solarwm.runtime import Topology

from .artifacts import H3CameraFilterError, H3PreencodedStream
from .raw_data import decode_camera
from .validation_inputs import validation_cameras


def source_video_id(row: IndexRow) -> str:
    return str(row.values.get("source_sample_id") or row.sample_id.split("/h3-158f-occ-")[0])


def validation_candidates(
    rows: Sequence[IndexRow], *, sample_count: int, seed: int, balance: bool
) -> tuple[tuple[IndexRow, ...], ...]:
    """Give each slot disjoint source videos and same-dataset reserve candidates."""
    groups: dict[str, dict[str, IndexRow]] = defaultdict(dict)
    for row in rows:
        source = row.sample_id.split("/")[0] if balance else "all"
        groups[source].setdefault(source_video_id(row), row)
    sources = sorted(groups)
    if not sources or sample_count < len(sources):
        raise DataContractError("H3 validation needs at least one slot per eligible dataset")
    slots = []
    for slot in range(sample_count):
        source = sources[slot % len(sources)]
        quota = sample_count // len(sources) + (sources.index(source) < sample_count % len(sources))
        ordered = sorted(
            groups[source].values(),
            key=lambda row: hashlib.sha256(f"{seed}:{row.sample_id}".encode()).hexdigest(),
        )
        if len(ordered) < quota:
            raise DataContractError(f"H3 validation has too few distinct source videos in {source}")
        slots.append(tuple(ordered[slot // len(sources) :: quota]))
    return tuple(slots)


class H3IndexedValidationStream:
    """Materialize complete logical-DP waves without preparing separate tensor files."""

    def __init__(self, config: Mapping[str, Any], topology: Any, *, frozen_ids=None) -> None:
        self.config = config
        data, validation = config["data"], config["validation"]
        transport = data["transport"]
        self.stage = config["train"]["stage"]
        self.topology = topology
        self.cursor = int(topology.dp_rank)
        self.count = int(validation["sample_count"])
        if self.count < 1 or self.count % int(topology.dp_world_size):
            raise DataContractError(
                "H3 validation sample count must form complete logical-DP waves"
            )
        self.noise_seed = int(validation["noise_seed"])
        if self.noise_seed < 0 or self.noise_seed + (self.count - 1) * 1009 >= 2**63:
            raise DataContractError("H3 validation noise seed is outside [0, 2**63)")
        self.reader = H3PreencodedStream(
            root=str(transport["root"]),
            index=str(resolve_index_path(data, "test_index")),
            # The validation slots own sampling; this reader only materializes artifacts.
            topology=Topology(1, 0, 1, 0, sp_size=1),
            seed=int(data.get("seed", 42)),
            encoder_contract_path=str(data["encoder_contract_path"]),
            cache_dir=transport.get("cache_dir"),
            cache_max_gib=float(transport.get("cache_max_gib", 256)),
            num_workers=1,
            gcs_prefetch_shards=0,
            camera_audit_latents=47,
        )
        self.encoder_contract = self.reader.encoder_contract
        self.encoder_profile = self.reader.encoder_profile
        try:
            self.raw_rows = {}
            rows = self.reader.rows
            if self.stage == "stage1":
                self.raw_rows = {
                    row.sample_id: row
                    for row in read_index(resolve_index_path(data, "raw_test_index"))
                }
                missing = [
                    row.sample_id for row in rows if source_video_id(row) not in self.raw_rows
                ]
                if missing:
                    raise DataContractError(
                        f"H3 raw test index lacks encoded sources: {missing[:3]}"
                    )
                rows = tuple(
                    row
                    for row in rows
                    if int(self.raw_rows[source_video_id(row)].values["num_frames"])
                    >= int(row.values["start_frame"]) + 170
                )
            self.frozen = frozen_ids is not None
            if self.frozen:
                by_id = {row.sample_id: row for row in rows}
                if len(frozen_ids) != self.count or len(set(frozen_ids)) != self.count:
                    raise DataContractError(
                        "H3 frozen validation IDs differ from the configured slots"
                    )
                try:
                    self.candidates = tuple((by_id[sample_id],) for sample_id in frozen_ids)
                except KeyError as exc:
                    raise DataContractError(
                        "H3 frozen validation source left the eligible test index"
                    ) from exc
            else:
                self.candidates = validation_candidates(
                    rows,
                    sample_count=self.count,
                    seed=int(validation["selection_seed"]),
                    balance=bool(validation.get("balance_by_dataset", False)),
                )
            self.digest = hashlib.sha256(
                json.dumps(
                    [[dict(row.values) for row in reserve] for reserve in self.candidates],
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest()
        except Exception:
            self.close()
            raise

    def _stage1_camera(self, batch: Any, row: IndexRow) -> Any:
        import torch

        raw = self.raw_rows[source_video_id(row)]
        manifest = json.loads(self.reader.shards.read(row, str(row.values["manifest_member"])))
        encoding = manifest.get("h3_preencoding", {})
        transform = encoding.get("resize_transform")
        if not isinstance(transform, Mapping):
            # Reconstruct the center crop from source dimensions when metadata omits it.
            height, width = int(raw.values["height"]), int(raw.values["width"])
            scale = max(1344 / width, 768 / height)
            resized_w, resized_h = max(1344, round(width * scale)), max(768, round(height * scale))
            transform = dict(
                source_h=height,
                source_w=width,
                resized_h=resized_h,
                resized_w=resized_w,
                crop_left=(resized_w - 1344) // 2,
                crop_top=(resized_h - 768) // 2,
                target_h=768,
                target_w=1344,
            )
        intrinsics_member = raw.values.get("intrinsics_member")
        camera_bytes = self.reader.shards.read(raw, str(raw.values["camera_member"]))
        intrinsics = (
            self.reader.shards.read(raw, str(intrinsics_member)) if intrinsics_member else None
        )
        indices = tuple(range(batch.start_frame, batch.start_frame + 170))
        c2w, K = decode_camera(camera_bytes, indices, transform, intrinsics_bytes=intrinsics)
        views, K = validation_cameras(
            {"camera_c2w": c2w, "camera_K": K, "source_frame_indices": np.asarray(indices)},
            {
                "validation_start_frame": batch.start_frame,
                "camera_convention": "absolute_c2w+normalized_K",
            },
            stage="stage1",
        )
        encoded_views, encoded_K = batch.camera_viewmats, batch.camera_K
        if encoded_views.shape[0] == 47 * 1008:
            for value in (encoded_views, encoded_K):
                grouped = value.unflatten(0, (47, 1008))
                if not torch.equal(grouped, grouped[:, :1].expand_as(grouped)):
                    raise DataContractError(
                        "H3 Stage1 validation requires spatially constant cameras"
                    )
            encoded_views, encoded_K = encoded_views[::1008], encoded_K[::1008]
        if not torch.allclose(views[:47], encoded_views, atol=1e-5, rtol=1e-5):
            raise DataContractError("H3 raw validation cameras disagree with the encoded window")
        if not torch.allclose(K[:47], encoded_K, atol=1e-6, rtol=1e-5):
            raise DataContractError("H3 raw validation intrinsics disagree with the encoded window")
        if (
            float(views.abs().max()) > 20
            or float(torch.linalg.vector_norm(views[:, :3, 3], dim=-1).max()) > 20
        ):
            raise H3CameraFilterError("H3 rollout camera exceeds the magnitude guards")
        return replace(
            batch,
            camera_viewmats=views[:47],
            camera_K=K[:47],
            rollout_camera_viewmats=views,
            rollout_camera_K=K,
        )

    def _select(self, slot: int, candidates: Sequence[IndexRow]) -> Any:
        for row in candidates:
            start = int(row.values["start_frame"])
            plan = SamplePlan(
                row.sample_id,
                row.key,
                row.shard,
                row.ordinal,
                0,
                0,
                start,
                tuple(range(start, start + 158)),
                self.topology.dp_rank,
                0,
            )
            try:
                batch = self.reader.read_artifact(row, plan)
                if self.stage == "stage1":
                    batch = self._stage1_camera(batch, row)
            except H3CameraFilterError:
                if self.frozen:
                    raise
                continue
            return replace(
                batch,
                validation_slot=slot,
                validation_noise_seed=self.noise_seed + slot * 1009,
                plan_fingerprint=self.digest,
            )
        return None

    def prepare(self, dist: Any) -> None:
        """Freeze all slots, refilling exhausted reserves without reusing other videos."""
        from solarwm.runtime.distributed import collective_call

        def gather(local: Any) -> list[Any]:
            if not dist.is_initialized():
                if int(self.topology.dp_world_size) != 1:
                    raise DataContractError("H3 distributed validation requires a process group")
                return [local]
            output = [None] * dist.get_world_size()
            dist.all_gather_object(output, local)
            return output

        slots = range(int(self.topology.dp_rank), self.count, int(self.topology.dp_world_size))
        self.prepared = collective_call(
            lambda: {slot: self._select(slot, self.candidates[slot]) for slot in slots},
            dist=dist,
            label="H3 validation candidate selection",
        )
        records = gather(
            {slot: batch.sample_id if batch else None for slot, batch in self.prepared.items()}
        )
        accepted = {}
        for record in records:
            for slot, sample_id in record.items():
                if slot in accepted and accepted[slot] != sample_id:
                    raise DataContractError("H3 SP peers selected different validation sources")
                accepted[slot] = sample_id
        if set(accepted) != set(range(self.count)):
            raise DataContractError("H3 validation selection did not cover all slots")
        used = {value.split("/h3-158f-occ-")[0] for value in accepted.values() if value}
        for slot in sorted(slot for slot, value in accepted.items() if value is None):
            if self.frozen:
                raise DataContractError(
                    "H3 frozen validation source no longer passes camera guards"
                )
            source = self.candidates[slot][0].sample_id.split("/")[0]
            candidates = sorted(
                (
                    row
                    for pool in self.candidates
                    for row in pool
                    if source_video_id(row) not in used
                    and (
                        not self.config["validation"].get("balance_by_dataset", False)
                        or row.sample_id.split("/")[0] == source
                    )
                ),
                key=lambda row: hashlib.sha256(
                    f"{self.config['validation']['selection_seed']}:{row.sample_id}".encode()
                ).hexdigest(),
            )

            def refill(slot: int = slot, candidates: Any = candidates) -> Any:
                if slot not in self.prepared:
                    return None
                batch = self._select(slot, candidates)
                if batch is None:
                    raise DataContractError(
                        f"H3 validation has no unused camera-safe source for slot {slot}"
                    )
                self.prepared[slot] = batch
                return batch.sample_id

            sample_id = collective_call(refill, dist=dist, label="H3 validation reserve refill")
            selected = {value for value in gather(sample_id) if value is not None}
            if len(selected) != 1:
                raise DataContractError("H3 validation refill differs between SP peers")
            used.add(selected.pop().split("/h3-158f-occ-")[0])

    def next(self) -> Any:
        slot = self.cursor
        if slot >= self.count:
            raise DataContractError("H3 validation requested more cases than configured")
        prepared = getattr(self, "prepared", None)
        batch = (
            prepared.pop(slot)
            if prepared is not None
            else self._select(slot, self.candidates[slot])
        )
        if batch is None:
            raise DataContractError(f"H3 validation slot {slot} needs collective reserve refill")
        self.cursor += int(self.topology.dp_world_size)
        return batch

    def close(self) -> None:
        self.reader.close()
