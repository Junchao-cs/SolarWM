"""Attach Standard compiler progress to the shared HTTP service."""
from pathlib import Path
from deploy.solar_wm.web.server import serve as serve_web
from .runtime.preparation_progress import PreparationProgress


def serve(state, output):
    progress = PreparationProgress(Path(output).parent, state.row["deployment_id"])
    return serve_web(state, output, preparation_progress=progress)
