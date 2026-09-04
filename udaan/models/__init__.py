"""Physics models for Udaan.

Each system gets a package of its own — ``quadrotor``, ``quadrotor_cspayload``,
``multi_quad_cspayload``, ``multi_quad_rigidbody``, ``quadrotor_flexcable``,
``multi_quad_hose`` — laid out the same way: ``base.py`` holds the pure
analytical dynamics, and ``mujoco.py`` / ``vfx.py`` subclass it to delegate to a
backend.

Two package-level modules sit beside them:

* :mod:`udaan.models.backend` — the MuJoCo plumbing (``MujocoModel``, the GLFW
  viewer) every per-system ``mujoco.py`` builds on.
* :mod:`udaan.models.fleet` — multi-agent models with no analytical
  counterpart: quadrotor fleets and comparison rigs.

``udaan.models.mujoco`` remains as a compatibility re-export of both.
"""

from . import (
    multi_quad_cspayload,
    multi_quad_hose,
    multi_quad_rigidbody,
    quadrotor,
    quadrotor_cspayload,
    quadrotor_flexcable,
)

# MuJoCo is optional
try:
    from . import backend, fleet, mujoco

    MUJOCO_AVAILABLE = True
except ImportError:
    import logging as _logging

    _logging.getLogger(__name__).info(
        "MuJoCo not available. Install with: pip install udaan[mujoco]"
    )
    backend = None  # type: ignore
    fleet = None  # type: ignore
    mujoco = None  # type: ignore
    MUJOCO_AVAILABLE = False

__all__ = [
    "MUJOCO_AVAILABLE",
    "backend",
    "fleet",
    "mujoco",
    "multi_quad_cspayload",
    "multi_quad_hose",
    "multi_quad_rigidbody",
    "quadrotor",
    "quadrotor_cspayload",
    "quadrotor_flexcable",
]
