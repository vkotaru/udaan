"""Multi-quadrotor cable-suspended rigid-body payload models (M4).

``nQ`` rigid-body quadrotors cooperatively carry one **rigid-body** payload,
each through its own taut cable attached at a distinct body-frame point of the
load.  Unlike M3's point mass the payload has an attitude, and the team can
therefore exert a moment on it.  Base dynamics are always available; the MuJoCo
backend needs the ``mujoco`` package.
"""

from .base import MultiQuadRigidbodyBase as MultiQuadRigidbodyBase
from .base import MultiQuadRigidbodyState as MultiQuadRigidbodyState
from .base import cable_state_from_geometry as cable_state_from_geometry
from .base import formation_cable_attitudes as formation_cable_attitudes
from .base import grasp_map as grasp_map
from .base import hover_wrench_map as hover_wrench_map
from .base import ring_attachment_points as ring_attachment_points

__all__ = [
    "MultiQuadRigidbodyBase",
    "MultiQuadRigidbodyState",
    "cable_state_from_geometry",
    "formation_cable_attitudes",
    "grasp_map",
    "hover_wrench_map",
    "ring_attachment_points",
]

# Optional import — MuJoCo backend needs mujoco
try:
    from .mujoco import MultiQuadRigidbodyMujoco as MultiQuadRigidbodyMujoco

    __all__.append("MultiQuadRigidbodyMujoco")
except ImportError:
    pass
