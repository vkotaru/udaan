"""Multi-quadrotor flexible-hose controllers (M6).

``nQ`` quadrotors bolted to chosen vertices of one ``n``-link flexible hose.
Like M5's, and for the same reason — a serial chain has no per-link body to
command — the single controller here is a linear-quadratic regulator on the
variational error dynamics rather than a geometric cascade, feedforwarded by
:class:`udaan.flatness.MultiQuadHose`.
"""

from .hose_controller import MultiQuadHoseController as MultiQuadHoseController

__all__ = [
    "MultiQuadHoseController",
]
