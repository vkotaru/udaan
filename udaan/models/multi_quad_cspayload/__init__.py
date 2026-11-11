"""Multi-quadrotor cable-suspended point-mass payload models (M3).

``nQ`` rigid-body quadrotors cooperatively carry one point-mass payload, each
through its own taut cable. Base dynamics are always available; the MuJoCo
backend needs the ``mujoco`` package.
"""

from .base import MultiQuadCsPayloadBase as MultiQuadCsPayloadBase
from .base import MultiQuadCsPayloadState as MultiQuadCsPayloadState
from .base import cable_state_from_geometry as cable_state_from_geometry
from .base import formation_cable_attitudes as formation_cable_attitudes

__all__ = [
    "MultiQuadCsPayloadBase",
    "MultiQuadCsPayloadState",
    "cable_state_from_geometry",
    "formation_cable_attitudes",
]

# Optional import — MuJoCo backend needs mujoco
try:
    from .mujoco import MultiQuadCsPayloadMujoco as MultiQuadCsPayloadMujoco

    __all__.append("MultiQuadCsPayloadMujoco")
except ImportError:
    pass
