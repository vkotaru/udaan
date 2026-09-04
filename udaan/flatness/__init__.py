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

:class:`MultiQuadCsPayload` — flat-to-state recovery for ``nQ``
quadrotors sharing one cable-suspended point-mass payload, with flat
output ``(x_L, {ψ_i}, {internal cable forces})``.  The internal-force
part is supplied by a pluggable :class:`TensionSplitPolicy`
(:class:`RingTensionSplit` by default).
"""

from .base import Flat2State as Flat2State
from .cable import cable_direction_jet as cable_direction_jet
from .cable import point_mass_cable_force as point_mass_cable_force
from .cable import tension_direction_jet as tension_direction_jet
from .derivatives import leibniz_dot as leibniz_dot
from .derivatives import leibniz_product as leibniz_product
from .derivatives import normalize_derivatives as normalize_derivatives
from .jet import Jet as Jet
from .multi_quad_cspayload import (
    DEFAULT_MIN_TENSION as DEFAULT_MIN_TENSION,
)
from .multi_quad_cspayload import (
    AffineTensionSplit as AffineTensionSplit,
)
from .multi_quad_cspayload import (
    LegacyFixedTensionSplit as LegacyFixedTensionSplit,
)
from .multi_quad_cspayload import (
    MultiQuadCsPayload as MultiQuadCsPayload,
)
from .multi_quad_cspayload import (
    MultiQuadCsPayloadFlats as MultiQuadCsPayloadFlats,
)
from .multi_quad_cspayload import (
    MultiQuadCsPayloadInputs as MultiQuadCsPayloadInputs,
)
from .multi_quad_cspayload import (
    MultiQuadCsPayloadRefState as MultiQuadCsPayloadRefState,
)
from .multi_quad_cspayload import (
    RingTensionSplit as RingTensionSplit,
)
from .multi_quad_cspayload import (
    TensionSplitContractError as TensionSplitContractError,
)
from .multi_quad_cspayload import (
    TensionSplitPolicy as TensionSplitPolicy,
)
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
    "DEFAULT_MIN_TENSION",
    "AffineTensionSplit",
    "Flat2State",
    "Jet",
    "LegacyFixedTensionSplit",
    "MultiQuadCsPayload",
    "MultiQuadCsPayloadFlats",
    "MultiQuadCsPayloadInputs",
    "MultiQuadCsPayloadRefState",
    "Quadrotor",
    "RingTensionSplit",
    "TensionSplitContractError",
    "TensionSplitPolicy",
    "cable_direction_jet",
    "leibniz_dot",
    "leibniz_product",
    "normalize_derivatives",
    "point_mass_cable_force",
    "tension_direction_jet",
    "QuadrotorCsPayload",
    "QuadrotorCsPayloadFlats",
    "QuadrotorCsPayloadInputs",
    "QuadrotorCsPayloadRefState",
    "QuadrotorFlats",
    "QuadrotorInputs",
    "QuadrotorRefState",
]
