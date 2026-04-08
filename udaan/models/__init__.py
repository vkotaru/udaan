"""Physics models for Udaan.

Provides quadrotor and payload models with multiple backend support:
- base: Analytical dynamics (always available)
- mujoco: MuJoCo physics (requires mujoco package)
"""

from . import base

# MuJoCo is optional
try:
    from . import mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    import logging as _logging

    _logging.getLogger(__name__).info(
        "MuJoCo not available. Install with: pip install udaan[mujoco]"
    )
    mujoco = None  # type: ignore
    MUJOCO_AVAILABLE = False

__all__ = ["base", "mujoco", "MUJOCO_AVAILABLE"]
