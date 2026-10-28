from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from solar.data.archive import RawSampleReader, TarShardReader
from solar.data.index import IndexRow
from solar.data.sampling import SamplePlan
from solar.data.transport import LocalResolver
from solar.errors import DataContractError


def test_raw_sample_preserves_authoritative_index_values(tmp_path: Path) -> None:
    shard = tmp_path / "data/sample.tar"
    shard.parent.mkdir()
    members = {
        "sample.video.mp4": b"video",
        "sample.camera.npz": b"camera",
        "sample.manifest.json": json.dumps(
            {"video": {"fps": 24.0}, "prompt": {"text": "manifest caption"}}
        ).encode(),
    }
    with tarfile.open(shard, "w:") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mtime = 0
            archive.addfile(info, io.BytesIO(payload))
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample",
            "key": "sample",
            "shard": "data/sample.tar",
            "video_member": "sample.video.mp4",
            "camera_member": "sample.camera.npz",
            "manifest_member": "sample.manifest.json",
            "caption": "index caption",
            "fps": 16.0,
            "num_frames": 81,
        },
    )
    plan = SamplePlan(
        sample_id="sample",
        key="sample",
        shard="data/sample.tar",
        row_ordinal=0,
        repeat_ordinal=0,
        epoch=0,
        start_frame=0,
        source_frame_indices=tuple(range(81)),
        reader_rank=0,
        worker_id=0,
    )
    with TarShardReader(LocalResolver(tmp_path)) as shards:
        sample = RawSampleReader((row,), shards).materialize(plan)

    assert sample.index_values["fps"] == 16.0
    assert sample.index_values["caption"] == "index caption"
    assert sample.caption == "index caption"


@pytest.mark.parametrize(
    "member",
    [
        None,
        7,
        "",
        "/sample/video.mp4",
        "//host/sample/video.mp4",
        "gs://bucket/sample/video.mp4",
        "file:/sample/video.mp4",
        "sample//video.mp4",
        "sample/./video.mp4",
        "sample/../video.mp4",
        "sample/video.mp4/",
        "sample\\video.mp4",
    ],
)
def test_tar_members_use_the_canonical_relative_path_contract(
    tmp_path: Path,
    member: object,
) -> None:
    row = IndexRow.from_mapping(
        0,
        {"sample_id": "sample", "key": "sample", "shard": "data/sample.tar"},
    )
    with (
        TarShardReader(LocalResolver(tmp_path)) as shards,
        pytest.raises(DataContractError, match="tar member"),
    ):
        shards.read(row, member)
