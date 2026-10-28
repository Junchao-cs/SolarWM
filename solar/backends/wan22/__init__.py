"""Wan2.2 backend plugin.

Importing this module is intentionally light-weight. Torch and the model
implementation are loaded lazily by the selected training, inference, or
preencoding action.
"""

from .backend import Wan22Backend, create_backend

__all__ = ["Wan22Backend", "create_backend"]
