"""Multiple quadrotors carrying one flexible hose (M6).

``nQ`` rigid-body quadrotors bolted to chosen vertices of a **single** hose,
discretised into ``n`` rigid links joined by ideal spherical joints.  It is M5's
chain with the airframe at its head replaced by an arbitrary attachment
pattern, so ``n`` and ``nQ`` are independent.  Base dynamics are always
importable; the MuJoCo backend needs the ``mujoco`` package.
"""

from .base import MultiQuadHoseBase as MultiQuadHoseBase
from .base import MultiQuadHoseState as MultiQuadHoseState

__all__ = [
    "MultiQuadHoseBase",
    "MultiQuadHoseState",
]

# Optional import — the MuJoCo backend needs the `mujoco` package.
try:
    from .mujoco import MultiQuadHoseMujoco as MultiQuadHoseMujoco

    __all__.append("MultiQuadHoseMujoco")
except ImportError:
    pass
