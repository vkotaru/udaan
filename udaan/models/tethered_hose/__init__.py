"""Quadrotors on a hose pinned to a ground anchor: M7 (one) and M8 (several in series)."""

from .base import TetheredHoseBase, TetheredQuadrotorBase

__all__ = ["TetheredHoseBase", "TetheredQuadrotorBase"]

try:
    from .mujoco import TetheredHoseMujoco, TetheredQuadrotorMujoco

    __all__ += ["TetheredHoseMujoco", "TetheredQuadrotorMujoco"]
except ImportError:  # pragma: no cover - mujoco is a hard dependency in practice
    pass
