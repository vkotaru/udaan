"""Quadrotor with cable-suspended payload models — base dynamics, VPython visualization, and MuJoCo physics."""

from .base import QuadrotorCsPayloadBase as QuadrotorCsPayloadBase
from .base import QuadrotorCsPayloadState as QuadrotorCsPayloadState

__all__ = [
    "QuadrotorCsPayloadBase",
    "QuadrotorCsPayloadState",
]

# Optional imports — VFX needs vpython, MuJoCo needs mujoco
try:
    from .vfx import QuadrotorCsPayloadVfx as QuadrotorCsPayloadVfx

    __all__.append("QuadrotorCsPayloadVfx")
except ImportError:
    pass

try:
    from .mujoco import QuadrotorCsPayloadMujoco as QuadrotorCsPayloadMujoco

    __all__.append("QuadrotorCsPayloadMujoco")
except ImportError:
    pass
