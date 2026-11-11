"""Differential-flatness state recovery for the systems in ``udaan``.

Central type
------------
:class:`Jet` — a value and all its time derivatives up to some order,
used as the input to every flat-to-state map in this package.

Per-system maps
---------------
:class:`Quadrotor` — flat-to-state recovery for a rigid-body quadrotor
with flat output ``(x_Q, ψ)``.

:class:`QuadrotorCsPayload` — flat-to-state recovery for a quadrotor
with a cable-suspended point-mass payload, with flat output
``(x_L, ψ)``.

Feasibility
-----------
:mod:`~udaan.flatness.feasibility` — whether a reference keeps every airframe
upright: :func:`is_upright`, :func:`scan_reference` and
:class:`ReferenceFeasibility`.
"""

from .base import Flat2State as Flat2State
from .feasibility import ReferenceFeasibility as ReferenceFeasibility
from .feasibility import infeasible_intervals as infeasible_intervals
from .feasibility import is_upright as is_upright
from .feasibility import scan_reference as scan_reference
from .jet import Jet as Jet
from .jet import yaw_jet as yaw_jet
from .quadrotor import (
    Quadrotor as Quadrotor,
)
from .quadrotor import (
    QuadrotorFlats as QuadrotorFlats,
)
from .quadrotor import (
    QuadrotorInputs as QuadrotorInputs,
)
from .quadrotor import (
    QuadrotorRefState as QuadrotorRefState,
)
from .quadrotor_cspayload import (
    QuadrotorCsPayload as QuadrotorCsPayload,
)
from .quadrotor_cspayload import (
    QuadrotorCsPayloadFlats as QuadrotorCsPayloadFlats,
)
from .quadrotor_cspayload import (
    QuadrotorCsPayloadInputs as QuadrotorCsPayloadInputs,
)
from .quadrotor_cspayload import (
    QuadrotorCsPayloadRefState as QuadrotorCsPayloadRefState,
)

__all__ = [
    "Flat2State",
    "Jet",
    "Quadrotor",
    "QuadrotorCsPayload",
    "QuadrotorCsPayloadFlats",
    "QuadrotorCsPayloadInputs",
    "QuadrotorCsPayloadRefState",
    "QuadrotorFlats",
    "QuadrotorInputs",
    "QuadrotorRefState",
    "ReferenceFeasibility",
    "infeasible_intervals",
    "is_upright",
    "scan_reference",
    "yaw_jet",
]
