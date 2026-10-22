"""One model process per GPU; HTTP and RGB processing stay off its control lane."""
from datetime import timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import sys
import threading
import time
import uuid

from .runtime.configuration import file_sha256, load_config, load_scenes


def run(args):
    import torch
    import torch.distributed as dist
    from solarwm.backends.minimax_h3.fsdp import initialize_distributed
    from .runtime.cache import CompilerCache, cache_key
    from .runtime.engine import Engine
    from .runtime.state import State
    from .vae.backends import Backends
    from .media.pipeline import Pipeline
    from .server import serve

    config = load_config(args.config)
    rank, local, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if rank != local or world != 8:
        raise RuntimeError("one-node SP8 torchrun required")
    scenes = load_scenes(config["scenes"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.set_num_threads(2)
    # Preserve the proven entry's inherited backend precision settings.
    # In Torch 2.10, writing the deprecated allow_tf32=False changes the
    # serialized matmul setting from "none" to "ieee", invalidating both
    # AOT/FX caches even when effective matmul precision was already IEEE.
    # The deployed DiT cache uses False. Native VAE's independent context uses
    # True; leaking that setting into DiT changes its kernels and cache key.
    torch._inductor.config.emulate_precision_casts = False
    torch._inductor.config.compile_threads = 1
    os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
    initialize_distributed(sp_size=8, local_rank=local)
    control = dist.new_group(list(range(8)), backend="gloo", timeout=timedelta(hours=1))
    identity = [f"release-{uuid.uuid4().hex[:16]}" if rank == 0 else None]
    dist.broadcast_object_list(identity, src=0, group=control)
    deployment = identity[0]
    root = Path(config["state_directory"]) / deployment
    root.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).parent
    # UI/server edits must not invalidate a compiled model. Include the actual
    # model code as well as the compile/Graph adapters in this cache namespace.
    from solarwm.backends.minimax_h3 import optional as h3_optional
    model_root = Path(h3_optional.__file__).parent
    relevant = list((package / "acceleration").rglob("*.py"))
    relevant.append(package / "acceleration" / "compatibility.json")
    relevant += [package / "runtime" / (name + ".py") for name in ("compiler", "scene_math", "engine", "operators")]
    sources = {str(p.relative_to(package)): file_sha256(p) for p in relevant}
    sources.update({"model/" + p.name: file_sha256(p) for p in model_root.glob("*.py")})
    key, fingerprint = cache_key(sources_sha256=sources,
        runtime=dict(python=platform.python_version(), torch=torch.__version__,
            triton=importlib.metadata.version("triton"), cuda=torch.version.cuda,
            compute_capability=str(torch.cuda.get_device_capability(device))),
        inference=dict(sp_size=8, dtype="bfloat16", latent_height=48, latent_width=84,
            denoise_steps=4, attention_backend="video_sdpa", emulate_precision_casts=False,
            prompt_min=1000, prompt_max=1300))
    compiler_cache = CompilerCache(Path(config["cache_directory"]), key=key, rank=rank)
    if config.get("dit_cache_seed"):
        from .runtime.cache_import import import_dit
        import_dit(compiler_cache, {name: value.format(rank=rank)
                   for name, value in config["dit_cache_seed"].items()})
    if config.get("vae_cache_seed"):
        from .runtime.cache_import import import_component
        for backend, seed in config["vae_cache_seed"].items():
            if backend == "current" and rank >= config["vae_ranks"]:
                continue
            import_component(compiler_cache, "vae-" + backend,
                             {name: value.format(rank=rank) for name, value in seed.items()})
    state = State(scenes, deployment, config) if rank == 0 else None
    output = root / "output"
    output.mkdir(exist_ok=True)
    servers = serve(state, output) if rank == 0 and args.command == "serve" else []
    if rank == 0:
        from .media.pixels import prepare as prepare_pixels
        prepare_pixels(config["cache_directory"])
    stopping = threading.Event()
    def stop_signal(*_):
        stopping.set()
        if state:
            state.request("/stop", {})
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    engine = vaes = pipeline = None
    normal_return = False
    from .runtime.cleanup import cleanup
    from .acceleration.compile.guard import SceneRejected
    try:
        with compiler_cache.installed("dit"):
            engine = Engine(config, rank=rank, device=device, run=root,
                            compiler_cache=compiler_cache, control_group=control)
        vaes = Backends(config, engine.cfg, device=device, rank=rank, cache=compiler_cache)
        enabled = ["current"] + (["streamrender"] if config.get("streamrender_weights") else [])
        if state:
            state.update(ready=True, phase="ready", vae_options_ready=False)
        first = args.scene or next(iter(scenes))
        warmed_backends = set()

        def cancelled():
            flag = [bool(stopping.is_set() or (state and state.snapshot()["stop_requested"])) if rank == 0 else None]
            dist.broadcast_object_list(flag, src=0, group=control)
            return flag[0]

        def prepare(scene):
            # All peers wait on one CPU control broadcast while rank zero's
            # disposable encoder runs. Cancellation here is NOT a collective.
            packet = [None]
            if rank == 0:
                from .scene_inputs import prepare_isolated, PreparationCancelled
                try:
                    row = state.scene_snapshot()[scene]
                    row = prepare_isolated(config, row,
                        lambda: stopping.is_set() or state.snapshot()["stop_requested"],
                        state.prepare_progress)
                    packet[0] = dict(scene=row)
                except PreparationCancelled:
                    packet[0] = dict(cancelled=True)
                except Exception as error:
                    packet[0] = dict(error=f"{type(error).__name__}: {error}")
            dist.broadcast_object_list(packet, src=0, group=control)
            if "scene" not in packet[0]:
                if state:
                    state.prepare_progress("cancelled" if packet[0].get("cancelled") else "failed",
                                           error=packet[0].get("error"))
                return False
            scenes[scene] = packet[0]["scene"]
            if state:
                state.prepare_progress("initializing")
            engine.select_scene(scenes[scene])
            engine.start(seed=args.seed)
            # Preserve the working entry's initialization order, independently
            # of the decoder initially selected in the UI. In Torch 2.10,
            # capturing StreamRender under inference_mode before DiT's first
            # no_grad Graph capture makes the latter fail at capture_begin.
            # Native VAE does not manually capture a Graph here. Finish DiT's
            # history/Graph preparation before loading/warming StreamRender.
            preparation_backend = "current"
            vaes.start(preparation_backend)
            trace = []
            retained = []
            aborted = False
            for index in range(9):
                if cancelled():
                    aborted = True
                    break
                if state:
                    state.prepare_progress("capturing" if index == 5 else "warming", index)
                engine.advance_camera([[] for _ in range(4 if index == 0 else 5)])
                latent, dit = engine.generate()
                if any(b not in warmed_backends and b != preparation_backend for b in enabled):
                    retained.append(latent.detach().cpu())
                vae = vaes.push(index, latent, lambda *a: None)
                trace.append(dict(chunk=index, dit_ms=dit, vae_ms=vae))
            report = engine.stop()
            vaes.finish()
            if not aborted:
                engine.compiler.freeze()
                warmed_backends.add(preparation_backend)
                # Reuse the just-produced latents. Warming a second decoder
                # never reruns DiT or manufactures an unrelated test input.
                for backend in enabled:
                    if backend in warmed_backends:
                        continue
                    vaes.start(backend)
                    for index, latent in enumerate(retained):
                        vaes.push(index, latent.to(device), lambda *a: None)
                    vaes.finish()
                    warmed_backends.add(backend)
            with (root / f"prepare-{engine.session_count}-rank{rank}.json").open("x") as file:
                json.dump(dict(trace=trace, report=report, cancelled=aborted,
                    preparation_backend=preparation_backend), file)
            if state:
                state.update(vae_options_ready=set(enabled) <= warmed_backends)
                state.prepare_progress("cancelled" if aborted else "ready",
                                       len(trace), ready=not aborted)
            return not aborted

        if state:
            state.request("/prepare", dict(scene_id=first))
            state.commands.get_nowait()
        prepared = prepare(first)
        if args.command == "smoke":
            if not prepared:
                raise RuntimeError("initial scene preparation failed or was cancelled")
            # Each backend sees the same seed and camera commands, in separate
            # real sessions. Includes first-history, steady and KV refresh.
            for backend in enabled:
                engine.select_scene(scenes[first])
                engine.start(seed=args.seed)
                vaes.start(backend)
                rows = []
                retained = []
                from safetensors.torch import save_file
                for i in range(args.chunks):
                    engine.advance_camera([[] for _ in range(4 if i == 0 else 5)])
                    latent, dit = engine.generate()
                    if rank == 0:
                        save_file({"latent": latent.detach().float().cpu().contiguous()},
                                  str(root / f"{backend}-latent-{i}.safetensors"))
                    def keep(chunk, offset, cpu, stamp):
                        if rank == 0:
                            retained.append(cpu)
                    vae = vaes.push(i, latent, keep)
                    rows.append(dict(chunk=i, dit_ms=dit, vae_ms=vae))
                report = engine.stop()
                vaes.finish()
                with (root / f"smoke-{backend}-rank{rank}.json").open("x") as file:
                    json.dump(dict(rows=rows, report=report), file)
                # Encoding is outside the timed DiT/VAE calls; retain every
                # generated frame for the short visual deployment check.
                if rank == 0:
                    for chunk in (0, args.chunks - 1):
                        save_file({"decoded": retained[chunk * 5 + 4].contiguous()},
                                  str(root / f"{backend}-chunk{chunk}-piece4.safetensors"))
                    from .media.archive import Archive
                    from .media.pixels import rgb
                    archive = Archive(root / f"smoke-{backend}.mp4", config["ffmpeg"])
                    try:
                        for piece in retained:
                            for frame in piece[0].unbind(dim=1):
                                archive.append(rgb(frame))
                        archive.finish()
                    except BaseException:
                        archive.abort()
                        raise
                dist.barrier(group=control)
            with (root / f"smoke-model-complete-rank{rank}.json").open("x") as file:
                json.dump(dict(rank=rank, backends=enabled, chunks_per_backend=args.chunks,
                               serving_after_smoke=args.serve_after_smoke), file)
            dist.barrier(group=control)
            if not args.serve_after_smoke:
                normal_return = True
                return
            if rank == 0:
                servers = serve(state, output)
                print(json.dumps(dict(event="smoke_models_verified_serving",
                    deployment_id=deployment, run=str(root),
                    port=config["port"], media_port=config["media_port"])), flush=True)
        while True:
            if rank == 0:
                command = dict(op="shutdown") if stopping.is_set() else state.next_command(engine.index)
                if command["op"] == "idle":
                    stopping.wait(.05)
                packet = [command]
            else:
                packet = [None]
            dist.broadcast_object_list(packet, src=0, group=control)
            command = packet[0]
            op = command["op"]
            if op == "shutdown":
                break
            if op == "idle":
                continue
            if op == "prepare":
                try:
                    prepare(command["scene_id"])
                except SceneRejected as error:
                    engine.recover_rejected_scene(error)
                    vaes.finish()
                    if state:
                        state.prepare_progress("failed", 0, error=str(error))
                        state.update(active=False, start_pending=False)
            elif op == "start":
                engine.select_scene(scenes[command["scene_id"]])
                engine.start(seed=args.seed)
                vaes.start(command["backend"])
                if state:
                    sequence = state.snapshot()["stream_sequence"] + 1
                    pipeline = Pipeline(output / f"{command['scene_id']}-upgraded-streaming-session-{sequence}-live.mp4",
                        state=state, ffmpeg=config.get("ffmpeg", "ffmpeg"), sequence=sequence,
                        resolution=command["resolution"])
                    state.produced.clear()
                    state.update(active=True, start_pending=False, paused=False,
                        phase="generating", chunk=0, frames=0, stream_frames=0,
                        stream_sequence=sequence, stream_ready=True, video_ready=False)
            elif op == "chunk":
                engine.set_navigation(command["move"], command["turn"])
                engine.advance_camera(command["actions"])
                latent, dit = engine.generate()
                vae = vaes.push(engine.index - 1, latent, pipeline.push if rank == 0 else lambda *a: None)
                if state:
                    state.generated(engine.index, dit, vae)
            elif op == "stop":
                report = engine.stop()
                vaes.finish()
                if state:
                    if engine.index:
                        pipeline.finish()
                    else:
                        pipeline.abort()
                    pipeline = None
                    state.update(active=False, paused=False, phase="ready", video_ready=engine.index > 0)
                    with (output / f"session-{state.snapshot()['stream_sequence']}.json").open("x") as file:
                        json.dump(report, file)
        normal_return = True
    except BaseException as error:
        if state:
            state.update(error=f"{type(error).__name__}: {error}", phase="error", ready=False)
        raise
    finally:
        primary = sys.exc_info()[1]
        actions = []
        if pipeline is not None:
            actions.append(("video", pipeline.abort))
        if engine is not None:
            actions.append(("model_and_graphs", engine.close))
        if vaes is not None:
            actions.append(("decoders", vaes.close))
        for server in servers:
            actions.extend((("http_shutdown", server.shutdown), ("http_close", server.server_close)))
        destroyed = dict(attempts=0, succeeded=False)
        def destroy():
            if dist.is_initialized():
                destroyed["attempts"] += 1
                dist.destroy_process_group()
                destroyed["succeeded"] = True
        actions.append(("process_group", destroy))
        failures = []
        try:
            failures = cleanup(actions, primary)
        except BaseException as error:
            failures = [("cleanup_primary", error)] + getattr(error, "solarwm_cleanup_failures", [])
            raise
        finally:
            with (root / f"exit-rank{rank}.json").open("x") as file:
                json.dump(dict(rank=rank, normal_return=normal_return,
                    process_group_destroy=destroyed,
                    primary=None if primary is None else repr(primary),
                    cleanup_failures=[dict(stage=stage, error=repr(error)) for stage, error in failures]),
                    file)
