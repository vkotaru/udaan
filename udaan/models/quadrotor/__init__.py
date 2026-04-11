"""Quadrotor models — base dynamics, VPython visualization, and MuJoCo physics."""

from .base import QuadrotorBase as QuadrotorBase
from .base import QuadrotorState as QuadrotorState

__all__ = [
    "QuadrotorBase",
    "QuadrotorState",
]

# Optional imports — VFX needs vpython, MuJoCo needs mujoco
try:
    from .vfx import QuadrotorVfx as QuadrotorVfx

    __all__.append("QuadrotorVfx")
except ImportError:
    pass

try:
    from .mujoco import QuadrotorMujoco as QuadrotorMujoco

    __all__.append("QuadrotorMujoco")
except ImportError:
    pass
