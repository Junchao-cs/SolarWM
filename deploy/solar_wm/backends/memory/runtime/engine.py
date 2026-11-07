"""Shared scene selection and camera-control interface for the Memory runtime."""
from deploy.solar_wm.shared.runtime.scene_math import NativeSceneMath
from .configuration import file_sha256


class Engine(NativeSceneMath):
    def select_scene(self, record):
        if self.scopes is not None:
            raise RuntimeError("finish active session before switching scene")
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError("condition file changed")
        self.condition = dict(record)

    def set_navigation(self, move, turn):
        from deploy.solar_wm.shared.runtime.navigation import speed
        self.navigation_speeds = (speed(move, maximum=10), speed(turn, maximum=3))
