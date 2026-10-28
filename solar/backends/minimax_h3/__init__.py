"""MiniMax-H3 backend plugin.

Importing this package is intentionally safe on CPU-only installations.  The
33B model runtime is not imported or constructed at plugin discovery time.
"""

from .backend import MiniMaxH3Backend, create_backend

__all__ = ["MiniMaxH3Backend", "create_backend"]
