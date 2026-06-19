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
"""

from .base import Flat2State as Flat2State
from .jet import Jet as Jet
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
]
