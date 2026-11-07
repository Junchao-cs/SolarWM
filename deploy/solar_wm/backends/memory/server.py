"""Use the shared web/control/media server with Memory capabilities.

The shared service supplies the player and static assets. Importing this
module does not import its model worker or its acceleration implementation.
Authentication and leases remain responsibilities of the shared gateway.
"""
from deploy.solar_wm.web.server import serve

__all__ = ["serve"]
