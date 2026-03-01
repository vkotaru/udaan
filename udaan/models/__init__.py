"""Physics models for Udaan.

Provides quadrotor and payload models with multiple backend support:
- base: Analytical dynamics (always available)
- mujoco: MuJoCo physics (optional, requires `pip install udaan[mujoco]`)
- bullet: PyBullet physics (optional, requires `pip install udaan[bullet]`)
"""

from . import base, bullet

# MuJoCo is optional
try:
    from . import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    mujoco = None  # type: ignore
    MUJOCO_AVAILABLE = False

__all__ = ["base", "bullet", "mujoco", "MUJOCO_AVAILABLE"]
