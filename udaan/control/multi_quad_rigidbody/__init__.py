"""Multi-quadrotor cable-suspended rigid-body payload controllers (M4).

``nQ`` quadrotors cooperatively tracking the full SE(3) pose of one rigid-body
payload, each through its own taut cable attached at a distinct body-frame
point.  The single controller here is the rigid-body sibling of
:class:`~udaan.control.multi_quad_cspayload.MultiQuadCsPayloadController`: where
M3 splits a commanded *force*, M4 splits a commanded *wrench* through the grasp
map :func:`udaan.utils.grasp.grasp_map`.
"""

from .payload_controller import MultiQuadRigidbodyController as MultiQuadRigidbodyController

__all__ = [
    "MultiQuadRigidbodyController",
]
