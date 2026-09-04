"""Quadrotor flexible-cable payload controllers (M5).

One quadrotor tracking a payload trajectory through a cable modelled as a
serial chain of ``n`` rigid links. The single controller here is the first in
this package that is **not** a geometric cascade: a chain has no per-link body
to command, so the published law — and this implementation — is a
linear-quadratic regulator on the variational error dynamics, feedforwarded by
:class:`udaan.flatness.QuadrotorFlexCable`.
"""

from .payload_controller import QuadrotorFlexCableController as QuadrotorFlexCableController

__all__ = [
    "QuadrotorFlexCableController",
]
