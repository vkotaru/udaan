"""LQR for quadrotors on a hose pinned to a ground anchor (M7 / M8)."""

from .tethered_controller import TetheredHoseController, default_top_position

__all__ = ["TetheredHoseController", "default_top_position"]
