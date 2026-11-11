"""Multi-quadrotor cable-suspended rigid-body payload controllers (M4).

``nQ`` quadrotors cooperatively tracking the full SE(3) pose of one rigid-body
payload, each through its own taut cable attached at a distinct body-frame
point.  The single controller here is the rigid-body sibling of
:class:`~udaan.control.multi_quad_cspayload.MultiQuadCsPayloadController`: where
M3 splits a commanded *force*, M4 splits a commanded *wrench* through the grasp
map :func:`udaan.utils.grasp.grasp_map`.
"""

from .gains import RigidPayloadCascadeGains as RigidPayloadCascadeGains
from .gains import rigid_payload_cascade_gains as rigid_payload_cascade_gains
from .payload_controller import MultiQuadRigidbodyController as MultiQuadRigidbodyController
from .payload_controller import cable_reference_rates as cable_reference_rates

__all__ = [
    "MultiQuadRigidbodyController",
    "RigidPayloadCascadeGains",
    "cable_reference_rates",
    "rigid_payload_cascade_gains",
]
