"""Memory capabilities for the shared control/session state."""
from deploy.solar_wm.shared.state import State as SharedState


class State(SharedState):
    def __init__(self, scenes, deployment, options):
        super().__init__(scenes, deployment, options, warmup_total=11,
                         fixed_vae_backend=False)
