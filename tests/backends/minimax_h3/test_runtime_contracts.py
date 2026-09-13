from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import pytest

from solarwm.backends.minimax_h3.artifacts import (
    H3ArtifactBatch,
    H3CameraFilterError,
    H3PlanMultiplexer,
    H3PreencodedStream,
    h3_encoder_contract,
    h3_silence_profile,
    load_silence_latents,
    read_encoder_contract,
    validate_fixed_validation_rows,
)
from solarwm.backends.minimax_h3.distributed import (
    contiguous_packed_bounds,
    logical_node_data_world_size,
)
from solarwm.backends.minimax_h3.fsdp import finite_clip_norm
from solarwm.backends.minimax_h3.inference import _finite_generation_metrics
from solarwm.backends.minimax_h3.lora import H3LoRARuntime
from solarwm.backends.minimax_h3.preencode_runner import (
    H3_COMPLETE_PATH,
    H3_ENCODER_CONTRACT_PATH,
    H3_INDEX_PATH,
    H3_SILENCE_PATH,
    _published_rows,
)
from solarwm.backends.minimax_h3.raw_data import normalize_raw_source_windows
from solarwm.backends.minimax_h3.runtime import (
    _assert_encoder_silence_identity,
    _base_model_load_receipt,
    _base_model_profile,
    _checkpoint_contract,
    _load_lora_checkpoint,
    _publish_h3_validation_complete,
    _validation_schedule,
)
from solarwm.data import IndexRow, SamplingConfig
from solarwm.data.sampling import SamplePlan
from solarwm.errors import BackendContractError, DataContractError
from solarwm.runtime import Topology


def test_h3_finite_generation_metrics_are_explicit() -> None:
    torch = pytest.importorskip("torch")
    metrics = _finite_generation_metrics(
        latents=torch.tensor([0.0, 1.0]),
        decoded=torch.tensor([0.25, 0.75]),
        reference_decoded=torch.tensor([0.0, 1.0]),
    )
    assert metrics["finite_fraction"] == 1.0
    assert metrics["latent_finite_fraction"] == 1.0
    assert metrics["decoded_finite_fraction"] == 1.0
    assert metrics["reference_decoded_finite_fraction"] == 1.0


def test_generic_h3_raw_rows_use_the_first_contiguous_window() -> None:
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample",
            "key": "sample",
            "shard": "raw/sample.tar",
            "num_frames": 200,
        },
    )
    normalized = normalize_raw_source_windows((row,))[0]
    assert normalized.values["start_frame"] == 0
    assert normalized.values["source_frame_indices"] == list(range(158))


@pytest.mark.parametrize("field", ("latents", "decoded", "reference_decoded"))
def test_h3_finite_generation_metrics_reject_nonfinite(field: str) -> None:
    torch = pytest.importorskip("torch")
    values = {
        "latents": torch.tensor([0.0]),
        "decoded": torch.tensor([0.0]),
        "reference_decoded": torch.tensor([0.0]),
    }
    values[field] = torch.tensor([float("inf")])
    with pytest.raises(RuntimeError, match="contains NaN or Inf"):
        _finite_generation_metrics(**values)


@pytest.mark.parametrize(
    ("weight_source", "sidecar_name"),
    (("live", "adapter_model.safetensors"), ("ema", "adapter_model_ema.safetensors")),
)
def test_h3_loader_selects_role_sidecar_and_returns_inventory(
    tmp_path, weight_source: str, sidecar_name: str
) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    values = {"block.lora_A.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}
    save_file(values, str(tmp_path / sidecar_name))
    (tmp_path / "model.pt").touch()

    class LoRA:
        loaded = None

        def load_state_dict(self, state, *, broadcast):
            self.loaded = (state, broadcast)

    lora = LoRA()
    identity = _load_lora_checkpoint(str(tmp_path / "model.pt"), lora, weight_source=weight_source)
    assert identity.startswith(f"inventory:file={sidecar_name}:bytes=")
    assert identity.endswith(f":role={weight_source}")
    assert "digest" not in identity
    assert lora.loaded is not None
    assert torch.equal(lora.loaded[0]["block.lora_A.weight"], values["block.lora_A.weight"])
    assert lora.loaded[1] is False


def test_h3_loader_does_not_reopen_loaded_sidecar_for_identity(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    values = {"block.lora_A.weight": torch.arange(4, dtype=torch.float32).reshape(2, 2)}
    sidecar = tmp_path / "adapter_model.safetensors"
    sidecar.write_bytes(b"loaded-by-test-double")

    calls = []
    monkeypatch.setattr(
        safetensors_torch,
        "load_file",
        lambda path, *, device: calls.append((path, device)) or values,
    )
    real_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == sidecar:
            raise AssertionError("loaded LoRA payload was reopened")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    class LoRA:
        def load_state_dict(self, state, *, broadcast):
            assert state is values
            assert broadcast is False

    identity = _load_lora_checkpoint(str(tmp_path), LoRA(), weight_source="live")
    assert calls == [(str(sidecar), "cpu")]
    assert identity == (
        f"inventory:file=adapter_model.safetensors:bytes={len(b'loaded-by-test-double')}:role=live"
    )


def test_h3_loader_requires_explicit_weight_source(tmp_path) -> None:
    with pytest.raises(BackendContractError, match="weight_source must be live or ema"):
        _load_lora_checkpoint(str(tmp_path / "model.pt"), object(), weight_source="")


@pytest.mark.parametrize("stage", ["stage0p5", "stage1"])
def test_h3_stage0p5_inference_loads_published_ema_and_checks_stage(tmp_path, stage) -> None:
    torch = pytest.importorskip("torch")
    from solarwm.checkpoint import CheckpointTransaction

    values = {"block.lora_A.weight": torch.full((2, 2), 1.2345)}
    parameter = torch.zeros(2, 2, dtype=torch.bfloat16)
    lora = SimpleNamespace(parameter_by_key={"block.lora_A.weight": parameter})
    contract = _checkpoint_contract(
        encoder_profile={"pixel_frames": 158, "height": 768, "width": 1344},
        silence_profile={},
        base_model={},
        config={"train": {"stage": stage}},
    )
    root = tmp_path / "model"
    with CheckpointTransaction(root) as transaction:
        torch.save(
            {"schema": "solarwm.minimax-h3-ema.v1", "shadow": values},
            transaction.path / "ema.pt",
        )
        transaction.commit(
            step=10500, contract=contract, required_components=["ema.pt"], metadata={}
        )
    if stage == "stage1":
        with pytest.raises(BackendContractError, match="contract"):
            _load_lora_checkpoint(str(root), lora, weight_source="ema")
        assert torch.count_nonzero(parameter) == 0
    else:
        identity = _load_lora_checkpoint(str(root), lora, weight_source="ema")
        assert identity == "stage0p5:ema:step=10500"
        assert torch.equal(parameter, values["block.lora_A.weight"].to(torch.bfloat16))


def test_h3_base_identity_is_readable_and_path_independent() -> None:
    first = {
        "checkpoint_path": "/models/first/MiniMax-H3",
        "architecture": "minimax-h3-33b",
        "transformer_subfolder": "transformer",
        "revision": "release-2026-08",
    }
    second = {**first, "checkpoint_path": "/another/mount/MiniMax-H3"}
    assert _base_model_profile(first) == _base_model_profile(second)
    rendered = json.dumps(_base_model_profile(first), sort_keys=True)
    assert "checkpoint_path" not in rendered
    assert "digest" not in rendered
    assert "minimax-h3-33b" in rendered
    assert "release-2026-08" in rendered


def test_h3_base_load_receipt_records_strict_inventory_without_a_digest() -> None:
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(
        OrderedDict(
            (
                ("bf16", torch.nn.Linear(2, 3, dtype=torch.bfloat16)),
                ("fp32", torch.nn.Linear(3, 1, dtype=torch.float32)),
            )
        )
    )
    receipt = _base_model_load_receipt(
        {
            "checkpoint_path": "/models/MiniMax-H3",
            "architecture": "minimax-h3-33b",
            "transformer_subfolder": "transformer",
        },
        model,
    )
    assert receipt["strict_state_load"] == {
        "missing_keys": 0,
        "unexpected_keys": 0,
        "mismatched_shapes": 0,
        "load_errors": 0,
    }
    assert receipt["parameter_inventory"]["tensors"] == 4
    assert set(receipt["parameter_inventory"]["dtypes"]) == {"bfloat16", "float32"}
    assert "digest" not in json.dumps(receipt)
    assert "/models" not in json.dumps(receipt)


def test_h3_checkpoint_contract_keeps_structured_resume_semantics() -> None:
    encoder = h3_encoder_contract(encoder_identity="official-minimax-h3-codec").as_dict()
    silence = h3_silence_profile()
    base = {
        "schema": "solarwm.minimax-h3-base-load.v1",
        "profile": _base_model_profile({"architecture": "minimax-h3-33b"}),
        "strict_state_load": {
            "missing_keys": 0,
            "unexpected_keys": 0,
            "mismatched_shapes": 0,
            "load_errors": 0,
        },
        "parameter_inventory": {
            "tensors": 1,
            "parameters": 2,
            "dtypes": {"bfloat16": {"tensors": 1, "parameters": 2}},
        },
    }
    contract = _checkpoint_contract(
        encoder_profile=encoder,
        silence_profile=silence,
        base_model=base,
    )
    assert contract.extras["encoder_profile"] == json.loads(json.dumps(encoder))
    assert contract.extras["silence_profile"] == silence
    assert contract.extras["base_model"] == base
    assert not any("digest" in key for key in contract.extras)


def _unit_lora_runtime() -> H3LoRARuntime:
    torch = pytest.importorskip("torch")
    parameters = OrderedDict(
        (("block.lora_A.weight", torch.nn.Parameter(torch.zeros(2, 2, dtype=torch.bfloat16))),)
    )
    return H3LoRARuntime(
        model=object(),
        targets=("block",),
        parameter_by_key=parameters,
        peft_config=object(),
        peft_module=SimpleNamespace(__version__="unit-test"),
        base_identity={"profile": "minimax-h3-33b"},
        rank=384,
        alpha=384,
    )


def test_h3_lora_metadata_keeps_explicit_state_without_target_digest() -> None:
    metadata = _unit_lora_runtime().metadata()
    assert metadata["target_modules"] == ["block"]
    assert metadata["state_keys"] == ["block.lora_A.weight"]
    assert metadata["state_shapes"] == {"block.lora_A.weight": [2, 2]}
    assert metadata["state_dtypes"] == {"block.lora_A.weight": "bfloat16"}
    assert "target_modules_digest" not in metadata


def test_h3_lora_state_load_is_strict_for_keys_shapes_and_dtypes() -> None:
    torch = pytest.importorskip("torch")
    runtime = _unit_lora_runtime()
    key = "block.lora_A.weight"
    runtime.load_state_dict({key: torch.ones(2, 2, dtype=torch.bfloat16)}, broadcast=False)
    assert torch.equal(runtime.parameter_by_key[key], torch.ones(2, 2, dtype=torch.bfloat16))
    with pytest.raises(BackendContractError, match="keys differ"):
        runtime.load_state_dict({"other": torch.ones(2, 2)}, broadcast=False)
    with pytest.raises(BackendContractError, match="shape differs"):
        runtime.load_state_dict({key: torch.ones(1, 2, dtype=torch.bfloat16)}, broadcast=False)
    with pytest.raises(BackendContractError, match="dtype differs"):
        runtime.load_state_dict({key: torch.ones(2, 2, dtype=torch.float32)}, broadcast=False)


def test_h3_silence_loader_validates_key_shape_dtype_and_finite_values(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    path = tmp_path / "silence.safetensors"
    save_file(
        {
            "silence_153f": torch.zeros((2, 32, 255), dtype=torch.bfloat16),
            "silence_158f": torch.zeros((2, 32, 263), dtype=torch.bfloat16),
        },
        str(path),
    )
    value, profile = load_silence_latents(path)
    assert tuple(value.shape) == (2, 32, 263)
    assert profile == h3_silence_profile()
    invalid = torch.zeros((2, 32, 263), dtype=torch.bfloat16)
    invalid[0, 0, 0] = float("inf")
    save_file({"silence_158f": invalid}, str(tmp_path / "nonfinite.safetensors"))
    with pytest.raises(DataContractError, match="non-finite"):
        load_silence_latents(tmp_path / "nonfinite.safetensors")


def test_training_stream_skips_camera_filter_rejections() -> None:
    stream = object.__new__(H3PreencodedStream)
    stream.fixed_validation = False
    stream.plan = SimpleNamespace(next_worker=0, num_workers=4)
    selected = object()
    outcomes = iter((H3CameraFilterError("filtered"), selected))
    workers = []

    def next_once():
        workers.append(stream.plan.next_worker)
        stream.plan.next_worker = (stream.plan.next_worker + 1) % stream.plan.num_workers
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    stream._next_once = next_once
    assert stream.next() is selected
    assert workers == [0, 0]
    assert stream.plan.next_worker == 1


def test_training_stream_keeps_retrying_the_rejected_worker() -> None:
    stream = object.__new__(H3PreencodedStream)
    stream.fixed_validation = False
    stream.plan = SimpleNamespace(next_worker=2, num_workers=4)
    selected = object()
    outcomes = iter(
        (
            H3CameraFilterError("filtered once"),
            H3CameraFilterError("filtered twice"),
            selected,
        )
    )
    workers = []

    def next_once():
        workers.append(stream.plan.next_worker)
        stream.plan.next_worker = (stream.plan.next_worker + 1) % stream.plan.num_workers
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    stream._next_once = next_once
    assert stream.next() is selected
    assert workers == [2, 2, 2]
    assert stream.plan.next_worker == 3


def test_training_stream_does_not_rotate_after_a_hard_failure() -> None:
    stream = object.__new__(H3PreencodedStream)
    stream.fixed_validation = False
    stream.plan = SimpleNamespace(next_worker=1, num_workers=4)

    def next_once():
        stream.plan.next_worker = 2
        raise DataContractError("broken artifact")

    stream._next_once = next_once
    with pytest.raises(DataContractError, match="broken artifact"):
        stream.next()
    assert stream.plan.next_worker == 1


def test_fixed_h3_validation_skips_camera_rejections_before_assigning_a_slot() -> None:
    stream = object.__new__(H3PreencodedStream)
    stream.fixed_validation = True
    stream.plan = SimpleNamespace(next_worker=0, num_workers=1)
    stream.rows = (object(), object())
    stream.topology = SimpleNamespace(dp_world_size=8, dp_rank=3)
    stream.fixed_validation_sample_count = 16
    stream.fixed_validation_noise_seed = 100
    stream.fixed_validation_successes = 0
    selected = H3ArtifactBatch(
        sample_id="selected",
        start_frame=0,
        plan_fingerprint="plan",
        target_latents=None,
        anchor_latents=None,
        prompt_embeds=None,
        text_token_tags=None,
        source_frame_indices=None,
        camera_viewmats=None,
        camera_K=None,
        source_fps=None,
    )
    outcomes = iter((H3CameraFilterError("filtered"), selected, selected))

    def next_once():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    stream._next_once = next_once
    first = stream.next()
    second = stream.next()
    assert (first.validation_slot, first.validation_noise_seed) == (3, 103)
    assert (second.validation_slot, second.validation_noise_seed) == (11, 111)


def test_fixed_h3_validation_selects_seeded_recipe_rows_and_reindexes_them() -> None:
    rows = tuple(
        IndexRow.from_mapping(
            slot,
            {
                "sample_id": f"sample-{slot}",
                "key": f"sample-{slot}",
                "shard": f"shards/{slot}.tar",
                "shard_size": 1,
                "start_frame": slot,
            },
        )
        for slot in range(8)
    )
    selected = validate_fixed_validation_rows(
        rows,
        logical_world_size=2,
        sample_count=4,
        selection_seed=42,
        noise_seed=100,
    )
    repeated = validate_fixed_validation_rows(
        rows,
        logical_world_size=2,
        sample_count=4,
        selection_seed=42,
        noise_seed=100,
    )
    assert [row.sample_id for row in selected] == [row.sample_id for row in repeated]
    assert len(selected) == 8
    assert [row.ordinal for row in selected] == list(range(8))
    assert all("validation_slot" not in row.values for row in selected)
    assert all("validation_noise_seed" not in row.values for row in selected)
    with pytest.raises(DataContractError, match="fewer rows"):
        validate_fixed_validation_rows(rows[:1], logical_world_size=2)
    with pytest.raises(DataContractError, match="complete logical-DP waves"):
        validate_fixed_validation_rows(rows, logical_world_size=2, sample_count=3)


def test_h3_validation_schedule_maps_sixteen_samples_to_two_sp2_waves() -> None:
    topology = Topology(16, 0, 8, 0, sp_size=2)
    assert _validation_schedule({"sample_count": 16}, topology) == (16, 2)
    with pytest.raises(BackendContractError, match="complete logical-DP waves"):
        _validation_schedule({"sample_count": 15}, topology)


def test_h3_validation_complete_requires_all_sixteen_unique_slots(tmp_path) -> None:
    destination = tmp_path / "validation" / "step_000020_live"
    compare = destination / "compare"
    manifests = destination / "manifests"
    compare.mkdir(parents=True)
    manifests.mkdir()
    for slot in range(16):
        (compare / f"rank{slot:03d}_round{slot // 8:02d}_dp{slot % 8:03d}.mp4").write_bytes(b"mp4")
        (manifests / f"rank_{slot:03d}.json").write_text(
            json.dumps({"logical_validation_rank": slot}),
            encoding="utf-8",
        )
    (compare / "rank015_round01_dp007.mp4").unlink()
    with pytest.raises(BackendContractError, match="output count differs"):
        _publish_h3_validation_complete(
            destination,
            step=20,
            pass_name="live",
            expected_local_slots=tuple(range(16)),
            global_slots=16,
        )
    (compare / "rank015_round01_dp007.mp4").write_bytes(b"mp4")
    _publish_h3_validation_complete(
        destination,
        step=20,
        pass_name="live",
        expected_local_slots=tuple(range(16)),
        global_slots=16,
    )
    complete = json.loads((destination / "COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["local_slots"] == 16
    assert complete["global_slots"] == 16


def test_h3_validation_complete_accepts_one_nodes_eight_of_sixteen_slots(tmp_path) -> None:
    destination = tmp_path / "validation" / "step_000050_live"
    compare = destination / "compare"
    manifests = destination / "manifests"
    compare.mkdir(parents=True)
    manifests.mkdir()
    local_slots = (0, 1, 2, 3, 8, 9, 10, 11)
    for slot in local_slots:
        (compare / f"rank{slot:03d}.mp4").write_bytes(b"mp4")
        (manifests / f"rank_{slot:03d}.json").write_text(
            json.dumps({"logical_validation_rank": slot}),
            encoding="utf-8",
        )
    _publish_h3_validation_complete(
        destination,
        step=50,
        pass_name="live",
        expected_local_slots=local_slots,
        global_slots=16,
    )
    complete = json.loads((destination / "COMPLETE.json").read_text(encoding="utf-8"))
    assert complete["local_slots"] == 8
    assert complete["global_slots"] == 16


def test_sp2_bounds_are_contiguous_and_cover_uneven_packed_rows() -> None:
    assert contiguous_packed_bounds(11, sp_size=2, sp_rank=0) == (0, 6)
    assert contiguous_packed_bounds(11, sp_size=2, sp_rank=1) == (6, 11)
    assert logical_node_data_world_size(world_size=256, local_world_size=8, sp_size=2) == 4
    with pytest.raises(ValueError, match="divisible"):
        logical_node_data_world_size(world_size=16, local_world_size=7, sp_size=2)


def test_replicated_lora_norm_is_overflow_safe_and_clips_in_place() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    parameter.grad = torch.tensor((3.0e19, 4.0e19), dtype=torch.bfloat16)
    finite, norm = finite_clip_norm((parameter,), 1.0)
    assert finite
    assert norm == pytest.approx(5.0e19, rel=0.01)
    clipped = torch.linalg.vector_norm(parameter.grad.float()).item()
    assert clipped == pytest.approx(1.0, rel=0.01)


def test_worker_plan_multiplexer_is_round_robin_and_exactly_resumable() -> None:
    rows = tuple(
        IndexRow.from_mapping(
            ordinal,
            {
                "sample_id": f"sample-{ordinal}",
                "key": f"sample-{ordinal}",
                "shard": f"shards/{ordinal:03d}.tar",
                "num_frames": 158,
                "start_frame": 0,
            },
        )
        for ordinal in range(16)
    )
    sampling = SamplingConfig(
        seed=42,
        pixel_frames=158,
        random_start=False,
        fixed_start_from_index=True,
        shuffle_buffer=4,
        partition_mode="node_shard",
    )

    def build() -> H3PlanMultiplexer:
        return H3PlanMultiplexer(
            rows,
            sampling,
            Topology(1, 0, 1, 0),
            num_workers=4,
            state_schema="test.h3-workers.v1",
        )

    first = build()
    prefix = [first.next_plan() for _ in range(5)]
    assert [plan.worker_id for plan in prefix] == [0, 1, 2, 3, 0]
    profile = {"schema": "test.h3-encoder.v1", "format": "h3.158f.v1"}
    state = first.state_dict(encoder_profile=profile)
    expected = [first.next_plan().sample_id for _ in range(12)]
    resumed = build()
    resumed.load_state_dict(state, encoder_profile=profile)
    assert [resumed.next_plan().sample_id for _ in range(12)] == expected
    with pytest.raises(DataContractError, match="encoder profile changed"):
        build().load_state_dict(state, encoder_profile={**profile, "format": "h3.other"})


def test_published_asset_layout_and_sampler_controls_are_explicit() -> None:
    indices = list(range(100, 258))
    receipt = SimpleNamespace(
        rows=(
            {
                "sample_id": "sample-a",
                "key": "sample-a",
                "source_sample_id": "source-a",
                "shard": "shards/rank-00000-part-000000.tar",
                "start_frame": 100,
                "source_frame_indices": indices,
                "encoder_contract_digest": "e" * 64,
                "members": {
                    "tensors.safetensors": "samples/a/tensors.safetensors",
                    "manifest.json": "samples/a/manifest.json",
                },
                "provenance_member": "samples/a/provenance.json",
                "metadata": {
                    "source_fps": 29.97,
                    "source_num_frames": 300,
                    "source_epoch_repeats": 3,
                    "member_digest": {
                        "tensors.safetensors": "a" * 64,
                        "manifest.json": "b" * 64,
                    },
                },
            },
        )
    )
    rows = _published_rows([receipt])
    assert rows[0]["num_frames"] == 300
    assert rows[0]["epoch_repeats"] == 3
    assert rows[0]["fps"] == pytest.approx(29.97)
    assert "source_num_frames" not in rows[0]["metadata"]
    assert "source_epoch_repeats" not in rows[0]["metadata"]
    assert not any("validation" in key for key in rows[0])
    assert rows[0]["tensor_digest"] == "a" * 64
    assert rows[0]["manifest_digest"] == "b" * 64
    assert len(rows[0]["provenance_digest"]) == 64
    assert "latent_recipe_sample_id" not in rows[0]
    assert (
        H3_ENCODER_CONTRACT_PATH,
        H3_SILENCE_PATH,
        H3_INDEX_PATH,
        H3_COMPLETE_PATH,
    ) == (
        "encoder-contract.json",
        "silence/h3-silence-158f.safetensors",
        "index.jsonl",
        "COMPLETE.json",
    )
    broken = SimpleNamespace(rows=({**receipt.rows[0], "source_frame_indices": indices[:-1]},))
    with pytest.raises(DataContractError, match="158 contiguous"):
        _published_rows([broken])


def test_preencoded_reader_replays_finalized_index_without_num_frames_or_indices(
    tmp_path,
) -> None:
    contract = h3_encoder_contract(encoder_identity="unit-test")
    contract_path = tmp_path / "encoder-contract.json"
    contract_path.write_text(json.dumps(contract.as_dict()), encoding="utf-8")
    index_path = tmp_path / "index.jsonl"
    source_indices = list(range(7, 165))
    index_path.write_text(
        json.dumps(
            {
                "sample_id": "sample-a",
                "key": "sample-a",
                "shard": "shards/a.tar",
                "start_frame": 7,
                "members": {"tensors.safetensors": "samples/a/tensors.safetensors"},
                "encoder_contract_digest": contract.digest,
                "metadata": {"source_fps": 24.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    stream = H3PreencodedStream(
        root=str(tmp_path),
        index=str(index_path),
        topology=Topology(2, 0, 2, 0, sp_size=2),
        seed=42,
        encoder_contract_path=str(contract_path),
        shuffle_buffer=1,
    )
    try:
        assert stream.rows[0].values["num_frames"] == 165
        assert stream.rows[0].values["fps"] == 24.0
        assert stream._plans[0].source_frame_indices == tuple(source_indices)
    finally:
        stream.close()


def test_provider_encoder_contract_resolves_to_readable_semantics(tmp_path) -> None:
    payload = {
        "schema": "solarwm_minimax_h3_encoder_contract_v1",
        "silence": {
            "path": "support/h3_silence_153_158_170.json",
            "generation": "123",
            "digest": "0" * 64,
        },
    }
    path = tmp_path / "provider-encoder-contract.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    loaded, profile = read_encoder_contract(path)
    assert loaded == payload
    assert profile == json.loads(
        json.dumps(h3_encoder_contract(encoder_identity="official-minimax-h3-codec").as_dict())
    )
    assert "digest" not in json.dumps(profile)


def test_fixed_validation_reader_maps_two_waves_to_distinct_logical_slots(tmp_path) -> None:
    contract = h3_encoder_contract(encoder_identity="unit-test")
    contract_path = tmp_path / "encoder-contract.json"
    contract_path.write_text(json.dumps(contract.as_dict()), encoding="utf-8")
    index_path = tmp_path / "fixed-validation.jsonl"
    rows = []
    for slot in range(16):
        start = slot * 10
        rows.append(
            {
                "sample_id": f"sample-{slot}",
                "key": f"sample-{slot}",
                "shard": f"shards/{slot}.tar",
                "num_frames": start + 158,
                "start_frame": start,
                "epoch_repeats": 1,
                "shard_size": 123,
                "shard_generation": "123456",
                "h3_preencoded_member": f"samples/{slot}.h3.safetensors",
                "h3_provenance_member": f"samples/{slot}.h3.provenance.json",
                "manifest_member": f"samples/{slot}.manifest.json",
            }
        )
    index_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    stream = H3PreencodedStream(
        root=str(tmp_path),
        index=str(index_path),
        topology=Topology(16, 4, 8, 4, sp_size=2),
        seed=42,
        encoder_contract_path=str(contract_path),
        fixed_validation=True,
        fixed_validation_sample_count=16,
        fixed_validation_selection_seed=42,
        fixed_validation_noise_seed=100,
    )
    try:
        assert len(stream._plans) == 2
        assert [plan.sample_id for plan in stream._plans] == [
            stream.rows[2].sample_id,
            stream.rows[10].sample_id,
        ]
        assert [plan.start_frame for plan in stream._plans] == [
            int(stream.rows[2].values["start_frame"]),
            int(stream.rows[10].values["start_frame"]),
        ]
    finally:
        stream.close()

    frozen_ids = tuple(f"sample-{slot}" for slot in reversed(range(16)))
    restored = H3PreencodedStream(
        root=str(tmp_path),
        index=str(index_path),
        topology=Topology(16, 4, 8, 4, sp_size=2),
        seed=42,
        encoder_contract_path=str(contract_path),
        fixed_validation=True,
        fixed_validation_sample_count=16,
        fixed_validation_selection_seed=999,
        fixed_validation_noise_seed=100,
        fixed_validation_sample_ids=frozen_ids,
    )
    try:
        assert [row.sample_id for row in restored.rows] == list(frozen_ids)
        assert [plan.sample_id for plan in restored._plans] == [frozen_ids[2], frozen_ids[10]]
    finally:
        restored.close()

    repeated_ids = list(frozen_ids)
    repeated_ids[10] = repeated_ids[2]
    repeated = H3PreencodedStream(
        root=str(tmp_path),
        index=str(index_path),
        topology=Topology(16, 4, 8, 4, sp_size=2),
        seed=42,
        encoder_contract_path=str(contract_path),
        fixed_validation=True,
        fixed_validation_sample_count=16,
        fixed_validation_selection_seed=999,
        fixed_validation_noise_seed=100,
        fixed_validation_sample_ids=repeated_ids,
    )
    try:
        assert [row.sample_id for row in repeated.rows] == repeated_ids
        assert [plan.sample_id for plan in repeated._plans] == [
            repeated_ids[2],
            repeated_ids[10],
        ]
    finally:
        repeated.close()


def test_h3_reader_treats_tensor_digest_as_offline_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    plan = SamplePlan(
        sample_id="sample",
        key="sample",
        shard="sample.tar",
        row_ordinal=0,
        repeat_ordinal=0,
        epoch=0,
        start_frame=0,
        source_frame_indices=tuple(range(158)),
        reader_rank=0,
        worker_id=0,
    )
    row = IndexRow.from_mapping(
        0,
        {
            "sample_id": "sample",
            "key": "sample",
            "shard": "sample.tar",
            "start_frame": 0,
            "source_frame_indices": list(range(158)),
            "h3_preencoded_member": "sample.h3.safetensors",
            "tensor_digest": "0" * 64,
        },
    )
    camera_c2w = torch.eye(4, dtype=torch.float32).repeat(158, 1, 1)
    camera_k = torch.eye(3, dtype=torch.float32).repeat(158, 1, 1)
    tensors = {
        "target_latents": torch.zeros((24, 47, 48, 84), dtype=torch.bfloat16),
        "anchor_latents": torch.zeros((24, 1, 48, 84), dtype=torch.bfloat16),
        "prompt_embeds": torch.zeros((2, 5120), dtype=torch.bfloat16),
        "text_token_tags": torch.zeros(2, dtype=torch.int64),
        "source_frame_indices": torch.arange(158, dtype=torch.int64),
        "camera_c2w": camera_c2w,
        "camera_K": camera_k,
    }
    monkeypatch.setattr(safetensors, "load", lambda _: tensors)

    class Shards:
        @staticmethod
        def read(_: IndexRow, member: str) -> bytes:
            assert member == "sample.h3.safetensors"
            return b"serialized tensor payload"

    prepared: list[SamplePlan] = []

    class Prefetcher:
        @staticmethod
        def prepare(value: SamplePlan) -> None:
            prepared.append(value)

    stream = object.__new__(H3PreencodedStream)
    stream.rows = (row,)
    stream.encoder_profile = h3_encoder_contract(encoder_identity="unit-test").as_dict()
    stream.shards = Shards()
    stream.plan = SimpleNamespace(
        next_plan=lambda: plan,
        current_plan_fingerprint="plan-fingerprint",
    )
    stream._shard_prefetcher = Prefetcher()
    stream.fixed_validation = False
    stream.camera_audit_latents = 47

    batch = stream._next_once()
    assert prepared == [plan]
    assert batch.sample_id == "sample"
    assert batch.plan_fingerprint == "plan-fingerprint"


def test_encoder_contract_can_bind_the_global_silence_artifact() -> None:
    plain = h3_encoder_contract(encoder_identity="unit-test")
    bound = h3_encoder_contract(
        encoder_identity="unit-test",
        silence_artifact_profile=h3_silence_profile(),
    )
    assert bound.extras["silence_artifact_profile"] == h3_silence_profile()
    assert bound.digest != plain.digest
    with pytest.raises(DataContractError, match="profile is unsupported"):
        h3_encoder_contract(
            encoder_identity="unit-test",
            silence_artifact_profile={**h3_silence_profile(), "posterior": "sample"},
        )


def test_runtime_rejects_silence_from_a_different_encoder_contract() -> None:
    profile = h3_silence_profile()
    contract = h3_encoder_contract(
        encoder_identity="unit-test",
        silence_artifact_profile=profile,
    )
    _assert_encoder_silence_identity(contract.as_dict(), profile)
    with pytest.raises(BackendContractError, match="different silence artifact"):
        _assert_encoder_silence_identity(
            contract.as_dict(),
            {**profile, "artifact": "synthetic-zero-latents"},
        )
