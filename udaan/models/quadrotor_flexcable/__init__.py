"""Quadrotor with a flexible (``n``-link) cable-suspended payload — M5.

Base dynamics are always importable; the MuJoCo backend is optional.
"""

from .base import QuadrotorFlexCableBase as QuadrotorFlexCableBase
from .base import QuadrotorFlexCableState as QuadrotorFlexCableState

__all__ = [
    "QuadrotorFlexCableBase",
    "QuadrotorFlexCableState",
]

# Optional import — the MuJoCo backend needs the `mujoco` package.
try:
    from .mujoco import QuadrotorFlexCableMujoco as QuadrotorFlexCableMujoco

    __all__.append("QuadrotorFlexCableMujoco")
except ImportError:
    pass
