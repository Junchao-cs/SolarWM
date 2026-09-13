from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from solarwm.backends.wan22 import create_backend
from solarwm.backends.wan22.runtime.assets import WanAssetLayout
from solarwm.backends.wan22.runtime.data import (
    CAMERA_CONVENTION,
    _source_fps,
    build_camera_tokens,
    latent_aligned_pixel_indices,
)
from solarwm.backends.wan22.runtime.stage0p5 import deterministic_i2v_drop_mask
from solarwm.config import load_config
from solarwm.data.camera import CameraGuards
from solarwm.errors import BackendContractError, DataContractError

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "configs/examples/wan22_ti2v_5b/train_stage0p5_fm_81f.yaml"


def test_i2v_dropout_preserves_public_stochastic_contract() -> None:
    torch = pytest.importorskip("torch")
    observed = [
        bool(
            deterministic_i2v_drop_mask(
                probability=0.1,
                batch_size=1,
                seed=42,
                global_step=0,
                logical_rank=rank,
                micro_index=0,
                device="cpu",
            ).item()
        )
        for rank in range(4)
    ]
    assert observed == [False, False, True, False]
    assert torch.equal(
        deterministic_i2v_drop_mask(
            probability=0.1,
            batch_size=1,
            seed=42,
            global_step=1,
            logical_rank=0,
            micro_index=0,
            device="cpu",
        ),
        torch.tensor([False]),
    )


def test_builtin_architecture_is_packaged_and_resolved_outside_weight_tree() -> None:
    config = load_config(EXAMPLE).values
    layout = WanAssetLayout.from_config(config)
    assert layout.transformer_config.name == "ti2v_5b.json"
    assert layout.transformer_config.is_file()
    assert layout.transformer_config.parent.name == "architectures"
    assert hashlib.blake2s(layout.transformer_config.read_bytes()).hexdigest() == (
        "e03bea7cd0c0d5608c89e8439ce63b483403f959ac5afb386bcb7b04bade4858"
    )
    payload = json.loads(layout.transformer_config.read_text())
    assert (payload["model_type"], payload["dim"], payload["num_layers"]) == (
        "ti2v",
        3072,
        30,
    )


def test_readiness_returns_all_placeholder_asset_and_index_issues() -> None:
    config = load_config(EXAMPLE).values
    report = create_backend(family="wan22_ti2v_5b").readiness(config)
    assert report.ready is False
    assert report.assets["transformer_config"].endswith("architectures/ti2v_5b.json")
    codes = {issue.code for issue in report.issues}
    assert "placeholder_path" in codes
    assert "index_missing" in codes


def test_training_readiness_can_defer_full_index_inventory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from solarwm.backends.wan22.runtime import readiness

    index = tmp_path / "train-index.jsonl.gz"
    index.write_bytes(b"staged-control")
    monkeypatch.setattr(
        readiness,
        "_index_paths",
        lambda _config: ({"train_index": index}, []),
    )
    monkeypatch.setattr(
        readiness,
        "read_index",
        lambda _path: pytest.fail("runtime readiness must not scan the complete index"),
    )
    issues: list[readiness.ReadinessIssue] = []
    inventories = readiness._probe_indexes({}, issues, validate_contents=False)
    assert issues == []
    assert inventories == {"train_index": {"bytes": len(b"staged-control"), "validation": "reader"}}


def test_training_runtime_uses_the_lightweight_gate_for_full_resume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import readiness

    captured: dict[str, object] = {}

    class Ready:
        def require_ready(self) -> None:
            captured["required"] = True

    def probe(_config: object, **kwargs: object) -> Ready:
        captured.update(kwargs)
        return Ready()

    monkeypatch.setattr(readiness, "probe_runtime", probe)
    report = readiness.require_training_runtime(
        {"checkpoint": {"mode": "full_resume"}},
        family="wan22_i2v_a14b",
    )
    assert isinstance(report, Ready)
    assert captured == {
        "family": "wan22_i2v_a14b",
        "require_cuda": True,
        "require_transformer_weights": False,
        "validate_index_contents": False,
        "required": True,
    }


def test_model_asset_closure_is_explicit() -> None:
    config = load_config(EXAMPLE).mutable_copy()
    del config["model"]["assets"]["vae"]
    with pytest.raises(BackendContractError, match=r"model\.assets fields"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_raw_config_requires_explicit_c2w_array_key() -> None:
    config = load_config(EXAMPLE).mutable_copy()
    del config["data"]["camera_array_key"]
    with pytest.raises(BackendContractError, match=r"camera_array_key=c2w"):
        create_backend(family="wan22_ti2v_5b").validate_config(config)


def test_raw_source_fps_uses_index_first_and_rejects_drift() -> None:
    assert _source_fps({"fps": 16.0}, {"video": {"fps": 16.0}}) == 16.0
    assert _source_fps({}, {"video": {"fps": 24.0}}) == 24.0
    with pytest.raises(DataContractError, match="conflicts"):
        _source_fps({"fps": 16.0}, {"video": {"fps": 15.0}})


def test_wan_causal_vae_temporal_alignment() -> None:
    assert latent_aligned_pixel_indices(81).tolist() == [0, *range(1, 81, 4)]
    assert latent_aligned_pixel_indices(153).tolist() == [0, *range(1, 153, 4)]
    with pytest.raises(DataContractError, match=r"1 \+ 4"):
        latent_aligned_pixel_indices(80)


def test_frozen_manifest_uses_explicit_config_fallback_without_axis_flip() -> None:
    torch = pytest.importorskip("torch")
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 5, axis=0)
    c2w[:, 0, 3] = np.arange(5, dtype=np.float32)
    payload = io.BytesIO()
    np.savez(payload, c2w=c2w)
    manifest = {
        "camera": {
            "convention": CAMERA_CONVENTION,
            "dtype": "float32",
            "finite": True,
            "magnitude_audit_seconds": 10.0,
            "max_camera_abs": 20.0,
            "max_rel_translation": 20.0,
            "shape": [5, 4, 4],
        }
    }
    camera = build_camera_tokens(
        payload.getvalue(),
        range(5),
        manifest,
        source_fps=16.0,
        output_fps=16.0,
        frame_sequence_length=2,
        guards=CameraGuards(max_rel_translation=20.0, max_camera_abs=20.0),
        configured_array_key="c2w",
    )
    assert tuple(camera["viewmats"].shape) == (4, 4, 4)
    assert torch.equal(camera["viewmats"][0], torch.eye(4))
    assert camera["viewmats"][2, 0, 3].item() == -1.0
    assert torch.equal(camera["viewmats"][2, :3, :3], torch.eye(3))

    manifest["camera"]["array_key"] = "w2c"
    with pytest.raises(DataContractError, match="conflicts"):
        build_camera_tokens(
            payload.getvalue(),
            range(5),
            manifest,
            source_fps=16.0,
            output_fps=16.0,
            frame_sequence_length=2,
            guards=CameraGuards(max_rel_translation=20.0, max_camera_abs=20.0),
            configured_array_key="c2w",
        )


def test_validation_can_disable_runtime_guards_without_changing_manifest_attestation() -> None:
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], 9, axis=0)
    c2w[5:, 0, 3] = 25.0
    payload = io.BytesIO()
    np.savez(payload, c2w=c2w)
    manifest = {
        "camera": {
            "array_key": "c2w",
            "convention": CAMERA_CONVENTION,
            "dtype": "float32",
            "finite": True,
            "magnitude_audit_seconds": 10.0,
            "max_camera_abs": 20.0,
            "max_rel_translation": 20.0,
            "shape": [9, 4, 4],
        }
    }
    camera = build_camera_tokens(
        payload.getvalue(),
        range(9),
        manifest,
        source_fps=16.0,
        output_fps=16.0,
        frame_sequence_length=2,
        guards=CameraGuards(max_rel_translation=None, max_camera_abs=None),
        manifest_guards=CameraGuards(max_rel_translation=20.0, max_camera_abs=20.0),
        configured_array_key="c2w",
    )
    assert camera["viewmats"].shape == (6, 4, 4)


def test_wan_camera_tokens_preserve_fp32_operation_order() -> None:
    torch = pytest.importorskip("torch")

    def invert_numpy(matrices: np.ndarray) -> np.ndarray:
        rotation = matrices[..., :3, :3]
        rotation_t = np.swapaxes(rotation, -1, -2)
        result = np.zeros_like(matrices)
        result[..., :3, :3] = rotation_t
        result[..., :3, 3] = -np.einsum(
            "...ij,...j->...i",
            rotation_t,
            matrices[..., :3, 3],
        )
        result[..., 3, 3] = 1.0
        return result

    def invert_torch(matrices):
        rotation = matrices[..., :3, :3]
        rotation_t = rotation.transpose(-1, -2)
        result = torch.zeros_like(matrices)
        result[..., :3, :3] = rotation_t
        result[..., :3, 3] = -torch.einsum(
            "...ij,...j->...i",
            rotation_t,
            matrices[..., :3, 3],
        )
        result[..., 3, 3] = 1.0
        return result

    poses = []
    for index, angle in enumerate(np.linspace(0.1, 0.9, 9, dtype=np.float32)):
        cosine = np.float32(np.cos(angle))
        sine = np.float32(np.sin(angle))
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = ((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1))
        pose[:3, 3] = (
            np.float32(index * 0.17),
            np.float32(index * -0.03),
            np.float32(index * 0.01),
        )
        poses.append(pose)
    c2w = np.stack(poses)
    payload = io.BytesIO()
    np.savez(payload, c2w=c2w)
    manifest = {
        "camera": {
            "array_key": "c2w",
            "convention": CAMERA_CONVENTION,
            "dtype": "float32",
            "finite": True,
            "magnitude_audit_seconds": 10.0,
            "max_camera_abs": 20.0,
            "max_rel_translation": 20.0,
            "shape": [9, 4, 4],
        }
    }
    camera = build_camera_tokens(
        payload.getvalue(),
        range(9),
        manifest,
        source_fps=16.0,
        output_fps=16.0,
        frame_sequence_length=2,
        guards=CameraGuards(max_rel_translation=20.0, max_camera_abs=20.0),
        configured_array_key="c2w",
    )

    selected = c2w[[0, 1, 5]]
    selected_w2c = invert_numpy(selected)
    anchor = np.eye(4, dtype=np.float32) @ selected_w2c[0]
    relative_c2w = np.empty_like(selected)
    relative_c2w[0] = np.eye(4, dtype=np.float32)
    for index in range(1, len(selected)):
        relative_c2w[index] = anchor @ selected[index]
    expected = invert_torch(torch.as_tensor(relative_c2w, dtype=torch.float32))
    expected = expected.unsqueeze(1).expand(-1, 2, -1, -1).reshape(-1, 4, 4)
    assert torch.equal(camera["viewmats"], expected)

    direct_shortcut = torch.as_tensor(
        np.matmul(invert_numpy(selected), selected[0]),
        dtype=torch.float32,
    )
    assert not torch.equal(expected[::2], direct_shortcut)
    assert not torch.equal(expected[::2].to(torch.bfloat16), direct_shortcut.to(torch.bfloat16))


def test_stage0p5_sp_identity_uses_the_sequence_parallel_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import stage0p5
    from solarwm.training.engine import BatchIdentity

    group = object()
    observed: dict[str, object] = {}
    monkeypatch.setattr(stage0p5, "get_sp_group", lambda: group)
    monkeypatch.setattr(
        stage0p5,
        "gather_and_assert_sp_identity",
        lambda value, *, sp_size, group: observed.update(
            value=value,
            sp_size=sp_size,
            group=group,
        ),
    )
    runtime = SimpleNamespace(topology=SimpleNamespace(sp_size=2))
    identity = BatchIdentity(
        sample_ids=("sample",),
        start_frames=(0,),
        noise_seeds=(7,),
        checkpoint_id="digest:checkpoint",
        plan_fingerprint="plan",
    )

    stage0p5.Wan5BStage0p5Runtime.assert_sp_peer_identity(runtime, identity)

    assert observed["sp_size"] == 2
    assert observed["group"] is group
    assert observed["value"] == {
        "sample_ids": ("sample",),
        "start_frames": (0,),
        "noise_seeds": (7,),
        "plan_fingerprint": "plan",
    }


def test_initialization_receipt_records_inventory_without_reading_payloads(
    tmp_path: Path,
) -> None:
    from solarwm.backends.wan22.runtime.loader import (
        WeightLoadReport,
        _bind_initialization_receipt,
    )

    architecture = tmp_path / "architecture.json"
    architecture.write_text('{"model_type":"tiny"}\n')
    shard = tmp_path / "weights.safetensors"
    shard.write_bytes(b"immutable-shard")
    report = WeightLoadReport(
        shards=(str(shard),),
        source_keys=1,
        target_keys=2,
        stripped_model_prefix=False,
        missing_keys=("extension.weight",),
        unexpected_keys=(),
    )
    bound = _bind_initialization_receipt(architecture=architecture, report=report)
    assert bound.initialization_id.startswith("inventory:")
    assert bound.architecture_inventory == ("architecture.json", architecture.stat().st_size)
    assert bound.shard_inventory == (("weights.safetensors", len(b"immutable-shard")),)
    assert bound.initialized_extension_keys == ("extension.weight",)
    receipt = bound.initialization_receipt()
    assert receipt["schema"] == "solarwm.wan22-initialization.v2"
    assert receipt["shards"] == [{"name": "weights.safetensors", "bytes": len(b"immutable-shard")}]
    assert all("digest" not in key for key in receipt)

    shard.write_bytes(b"same-byte-count")
    repeated = _bind_initialization_receipt(architecture=architecture, report=report)
    assert repeated.initialization_id == bound.initialization_id
    shard.write_bytes(b"different-byte-count")
    changed = _bind_initialization_receipt(architecture=architecture, report=report)
    assert changed.initialization_id != bound.initialization_id


def test_wan5b_vae_reciprocal_is_computed_after_bfloat16_cast() -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime.components import Wan5BVAE

    vae = object.__new__(Wan5BVAE)
    vae._mean = torch.tensor(Wan5BVAE.mean, dtype=torch.float32)
    vae._std = torch.tensor(Wan5BVAE.std, dtype=torch.float32)
    reference = torch.empty((), dtype=torch.bfloat16)
    scale = vae._scale(reference)
    expected = 1.0 / torch.tensor(Wan5BVAE.std, dtype=torch.float32).to(torch.bfloat16)
    precomputed_then_cast = (1.0 / torch.tensor(Wan5BVAE.std)).to(torch.bfloat16)
    assert torch.equal(scale[1], expected)
    assert not torch.equal(expected, precomputed_then_cast)


def test_weight_readiness_inventories_shard_headers_without_digests(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from solarwm.backends.wan22.runtime.readiness import _probe_weights

    safetensors.save_file({"a": torch.ones(2)}, tmp_path / "part-01.safetensors")
    safetensors.save_file({"b": torch.zeros(3)}, tmp_path / "part-02.safetensors")
    issues = []
    inventory = _probe_weights(tmp_path, issues)
    assert issues == []
    assert inventory["tensor_keys"] == 2
    assert [row["name"] for row in inventory["shards"]] == [
        "part-01.safetensors",
        "part-02.safetensors",
    ]
    assert inventory["inspection"] == "safetensors-headers"
    assert all("digest" not in row for row in inventory["shards"])


def test_weight_source_inventory_uses_safe_open_keys_api(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    from solarwm.backends.wan22.runtime.loader import _source_keys

    first = tmp_path / "part-01.safetensors"
    second = tmp_path / "part-02.safetensors"
    safetensors.save_file({"block.weight": torch.ones(2)}, first)
    safetensors.save_file({"head.bias": torch.zeros(3)}, second)

    assert _source_keys((first, second)) == {"block.weight", "head.bias"}


def test_anyflow_external_embedding_readiness_checks_the_real_file(tmp_path: Path) -> None:
    from solarwm.backends.wan22.runtime.readiness import _check_assets

    example = ROOT / "configs/examples/wan22_ti2v_5b/train_stage1_tf_anyflow_v1_5_81f.yaml"
    config = load_config(example).mutable_copy()
    config["model"]["base_path"] = str(tmp_path)
    negative = tmp_path / "conditioning/wan_negemb_cn.pth"
    negative.parent.mkdir()
    negative.write_bytes(b"validity-is-checked-when-the-tensor-is-loaded")
    layout = WanAssetLayout.from_config(config)
    assert layout.anyflow_negative_embedding == negative
    issues = []
    _check_assets(layout, online=False, issues=issues)
    assert [issue for issue in issues if issue.path == str(negative)] == []


def test_tiny_camera_transformer_executes_forward_and_backward_when_installed() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("einops")
    if not torch.cuda.is_available():
        pytest.skip("the Wan attention kernel requires CUDA")

    from solarwm.backends.wan22.runtime.modeling.causal_model import CausalWanModel

    torch.manual_seed(7)
    device = torch.device("cuda")
    model = CausalWanModel(
        model_type="ti2v",
        patch_size=(1, 2, 2),
        text_len=4,
        in_dim=4,
        dim=32,
        ffn_dim=64,
        freq_dim=16,
        text_dim=8,
        out_dim=4,
        num_heads=2,
        num_layers=1,
        local_attn_size=2,
        sink_size=0,
        add_control_adapter=True,
        cam_method="prope",
        cam_self_attn_layers=[0],
        frame_seq_length=4,
        camera_attention_mode="fused_prope",
    ).to(device=device, dtype=torch.bfloat16)
    inputs = torch.randn(
        1,
        4,
        2,
        4,
        4,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    timestep = torch.full((1, 8), 500.0, device=device)
    context = torch.randn(1, 4, 8, device=device, dtype=torch.bfloat16)
    viewmats = torch.eye(4, device=device).repeat(1, 8, 1, 1)
    intrinsics = torch.eye(3, device=device).repeat(1, 8, 1, 1)
    output = model(
        inputs,
        t=timestep,
        context=context,
        seq_len=8,
        y_camera={"viewmats": viewmats, "K": intrinsics},
        cache_update_policy="none",
    )
    assert output.shape == inputs.shape
    output.float().square().mean().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()


def test_wan_activation_checkpointing_wraps_every_block_non_reentrantly() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    from solarwm.backends.wan22.runtime.distributed import (
        apply_wan_activation_checkpointing,
    )
    from solarwm.backends.wan22.runtime.modeling.causal_model import (
        CausalWanAttentionBlock,
    )

    class TinyRoot(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for index in range(3):
                block = CausalWanAttentionBlock.__new__(CausalWanAttentionBlock)
                torch.nn.Module.__init__(block)
                self.add_module(f"block_{index}", block)

    root = TinyRoot()
    assert apply_wan_activation_checkpointing(root) == 3
    assert all(
        hasattr(getattr(root, f"block_{index}"), "_checkpoint_wrapped_module") for index in range(3)
    )


def test_successful_training_step_returns_process_exit_zero_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from solarwm.backends.wan22.runtime import stage0p5

    runtime = SimpleNamespace(topology=SimpleNamespace(raw_rank=0))
    calls: list[str] = []

    class FakeEngine:
        def __init__(self, runtime_arg: object, policy: object, *, event_sink: object) -> None:
            assert runtime_arg is runtime
            assert policy.max_steps == 1
            assert event_sink is not None

        def run(self) -> int:
            calls.append("run")
            return 1

    monkeypatch.setattr(stage0p5, "build_stage0p5_runtime", lambda _: runtime)
    monkeypatch.setattr(stage0p5, "TrainingEngine", FakeEngine)
    monkeypatch.setattr(stage0p5, "cleanup_torchrun", lambda: calls.append("cleanup"))
    config = {
        "train": {"max_steps": 1, "grad_accum": 1},
        "runtime": {"output_dir": str(tmp_path)},
    }
    assert stage0p5.run_stage0p5_training(config) == 0
    assert calls == ["run", "cleanup"]


def test_checkpoint_commit_updates_inline_validation_weight_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import checkpoint
    from solarwm.backends.wan22.runtime.stage0p5 import Wan5BStage0p5Runtime

    runtime = Wan5BStage0p5Runtime.__new__(Wan5BStage0p5Runtime)
    runtime.config = {}
    runtime._global_step = 1000
    runtime.checkpoint_id = "digest:initial"
    runtime.diffusion = object()
    runtime.optimizer = object()
    runtime.lr_scheduler = object()
    runtime.ema = object()
    runtime.data = object()
    runtime.device = object()
    monkeypatch.setattr(
        checkpoint,
        "save_full_checkpoint",
        lambda **kwargs: "a" * 64,
    )
    assert runtime.save_checkpoint(1000) == "a" * 64
    assert runtime.checkpoint_id == f"digest:{'a' * 64}"


def test_rank_runtime_state_restores_reader_and_all_cpu_rng_streams() -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime.checkpoint import (
        _local_rank_runtime_state,
        _restore_rank_runtime_state,
    )

    class Reader:
        def __init__(self) -> None:
            self.position = 17

        def state_dict(self) -> dict[str, int]:
            return {"position": self.position}

        def load_state_dict(self, value: dict[str, int]) -> None:
            self.position = int(value["position"])

    reader = Reader()
    random.seed(123)
    np.random.seed(456)
    torch.manual_seed(789)
    state = _local_rank_runtime_state(reader, torch.device("cpu"))
    expected = (random.random(), float(np.random.random()), torch.rand(4))

    reader.position = 99
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    _restore_rank_runtime_state([state], reader=reader, device=torch.device("cpu"))
    assert reader.position == 17
    assert random.random() == expected[0]
    assert float(np.random.random()) == expected[1]
    assert torch.equal(torch.rand(4), expected[2])


def test_full_resume_rejects_checkpoint_without_rank_runtime_state() -> None:
    torch = pytest.importorskip("torch")
    from solarwm.backends.wan22.runtime.checkpoint import _restore_rank_runtime_state

    with pytest.raises(BackendContractError, match="rank-local reader and RNG state"):
        _restore_rank_runtime_state(None, reader=object(), device=torch.device("cpu"))


def test_training_failure_still_cleans_up_distributed_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import stage0p5

    calls: list[str] = []

    def fail(_: object) -> object:
        raise RuntimeError("construction failed")

    monkeypatch.setattr(stage0p5, "build_stage0p5_runtime", fail)
    monkeypatch.setattr(stage0p5, "cleanup_torchrun", lambda: calls.append("cleanup"))
    with pytest.raises(RuntimeError, match="construction failed"):
        stage0p5.run_stage0p5_training({})
    assert calls == ["cleanup"]


def test_training_defers_cleanup_to_explicit_caller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from solarwm.backends.wan22.runtime import stage0p5

    runtime = SimpleNamespace(topology=SimpleNamespace(raw_rank=0))
    calls: list[str] = []

    class FakeEngine:
        def __init__(self, runtime_arg: object, policy: object, *, event_sink: object) -> None:
            assert runtime_arg is runtime

        def run(self) -> int:
            calls.append("run")
            return 1

    monkeypatch.setenv("SOLARWM_TORCHRUN_LIFECYCLE_OWNER", "caller")
    monkeypatch.setattr(stage0p5, "build_stage0p5_runtime", lambda _: runtime)
    monkeypatch.setattr(stage0p5, "TrainingEngine", FakeEngine)
    monkeypatch.setattr(stage0p5, "cleanup_torchrun", lambda: calls.append("cleanup"))
    config = {
        "train": {"max_steps": 1, "grad_accum": 1},
        "runtime": {"output_dir": str(tmp_path)},
    }
    assert stage0p5.run_stage0p5_training(config) == 0
    assert calls == ["run"]


def test_training_rejects_unknown_lifecycle_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from solarwm.backends.wan22.runtime import stage0p5

    monkeypatch.setenv("SOLARWM_TORCHRUN_LIFECYCLE_OWNER", "somebody")
    with pytest.raises(BackendContractError, match="must be backend or caller"):
        stage0p5.run_stage0p5_training({})
