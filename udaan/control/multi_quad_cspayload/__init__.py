"""Multi-quadrotor cable-suspended payload controllers (M3).

``nQ`` quadrotors cooperatively tracking one point-mass payload, each through
its own taut cable. The single controller here is the generic-``nQ`` sibling of
:class:`~udaan.control.quadrotor_cspayload.QuadCSPayloadController`; it reuses
:class:`~udaan.flatness.TensionSplitPolicy` to distribute the commanded net
cable force across the cables.
"""

from .payload_controller import MultiQuadCsPayloadController as MultiQuadCsPayloadController

__all__ = [
    "MultiQuadCsPayloadController",
]
