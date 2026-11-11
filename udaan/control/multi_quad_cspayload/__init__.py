"""Multi-quadrotor cable-suspended payload controllers (M3).

``nQ`` quadrotors cooperatively tracking one point-mass payload, each through
its own taut cable. The single controller here is the generic-``nQ`` sibling of
:class:`~udaan.control.quadrotor_cspayload.QuadCSPayloadController`; it reuses
:class:`~udaan.flatness.TensionSplitPolicy` to distribute the commanded net
cable force across the cables.
"""

from .gains import MultiCableCascadeGains as MultiCableCascadeGains
from .gains import multi_cable_cascade_gains as multi_cable_cascade_gains
from .payload_controller import MultiQuadCsPayloadController as MultiQuadCsPayloadController

__all__ = [
    "MultiCableCascadeGains",
    "MultiQuadCsPayloadController",
    "multi_cable_cascade_gains",
]
