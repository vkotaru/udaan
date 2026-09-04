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

:class:`QuadrotorFlexCable` — flat-to-state recovery for a quadrotor
carrying a payload through a **flexible cable of ``n`` rigid links**, with
flat output ``(x_L, ψ)`` — the same flat output as
:class:`QuadrotorCsPayload`, of which it is the ``n = 1`` case.  Unlike
every other map here its required jet depth *grows with the system size*:
``2n + 4``.  The chain half is the reusable primitive
:func:`cable_chain_cascade`.

:class:`MultiQuadRigidbody` — flat-to-state recovery for ``nQ``
quadrotors sharing one cable-suspended **rigid-body** payload, with flat
output ``(x_L, R_L, {ψ_i}, λ)``.  The payload now has an attitude, so the
free part of the split is the null space of the grasp map, resolved by a
:class:`TensionDistributionPolicy` (:class:`RingTensionDistribution` by
default) through the :class:`WrenchAllocation` primitive.

:class:`MultiQuadHose` — flat-to-state recovery for ``nQ`` quadrotors
bolted at chosen vertices of **one flexible hose** of ``n`` links, with
flat output ``(x_0, {𝛕_a}, {ψ_k})``.  The only map here whose flat output
is not built from positions alone: the free coordinates are the *vector
tensions* in the links hanging off all but the last vehicle.  Like M5 its
jet depth grows as ``2n + 4``, and unlike every other map it is flat only
when a vehicle sits at the last vertex — :class:`NotFlatError` otherwise.
It reuses :func:`cable_chain_cascade` on the reversed chain, through the
``link_forces`` hook.
"""

from .allocation import (
    FixedTensionDistribution as FixedTensionDistribution,
)
from .allocation import (
    RingTensionDistribution as RingTensionDistribution,
)
from .allocation import (
    TensionDistributionPolicy as TensionDistributionPolicy,
)
from .allocation import (
    WrenchAllocation as WrenchAllocation,
)
from .allocation import (
    WrenchAllocationError as WrenchAllocationError,
)
from .allocation import (
    null_space_basis as null_space_basis,
)
from .base import Flat2State as Flat2State
from .cable import cable_direction_jet as cable_direction_jet
from .cable import point_mass_cable_force as point_mass_cable_force
from .cable import tension_direction_jet as tension_direction_jet
from .chain import CableChain as CableChain
from .chain import cable_chain_cascade as cable_chain_cascade
from .chain import chain_jet_order as chain_jet_order
from .derivatives import leibniz_dot as leibniz_dot
from .derivatives import leibniz_matmul as leibniz_matmul
from .derivatives import leibniz_product as leibniz_product
from .derivatives import normalize_derivatives as normalize_derivatives
from .jet import Jet as Jet
from .jet import yaw_jet as yaw_jet
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
from .multi_quad_hose import (
    HoseCascade as HoseCascade,
)
from .multi_quad_hose import (
    MultiQuadHose as MultiQuadHose,
)
from .multi_quad_hose import (
    MultiQuadHoseFlats as MultiQuadHoseFlats,
)
from .multi_quad_hose import (
    MultiQuadHoseInputs as MultiQuadHoseInputs,
)
from .multi_quad_hose import (
    MultiQuadHoseRefState as MultiQuadHoseRefState,
)
from .multi_quad_hose import (
    NotFlatError as NotFlatError,
)
from .multi_quad_rigidbody import (
    MultiQuadRigidbody as MultiQuadRigidbody,
)
from .multi_quad_rigidbody import (
    MultiQuadRigidbodyFlats as MultiQuadRigidbodyFlats,
)
from .multi_quad_rigidbody import (
    MultiQuadRigidbodyInputs as MultiQuadRigidbodyInputs,
)
from .multi_quad_rigidbody import (
    MultiQuadRigidbodyRefState as MultiQuadRigidbodyRefState,
)
from .multi_quad_rigidbody import (
    rotation_jet as rotation_jet,
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
from .quadrotor_flexcable import (
    QuadrotorFlexCable as QuadrotorFlexCable,
)
from .quadrotor_flexcable import (
    QuadrotorFlexCableFlats as QuadrotorFlexCableFlats,
)
from .quadrotor_flexcable import (
    QuadrotorFlexCableInputs as QuadrotorFlexCableInputs,
)
from .quadrotor_flexcable import (
    QuadrotorFlexCableRefState as QuadrotorFlexCableRefState,
)

__all__ = [
    "DEFAULT_MIN_TENSION",
    "AffineTensionSplit",
    "CableChain",
    "FixedTensionDistribution",
    "Flat2State",
    "HoseCascade",
    "Jet",
    "LegacyFixedTensionSplit",
    "MultiQuadCsPayload",
    "MultiQuadCsPayloadFlats",
    "MultiQuadCsPayloadInputs",
    "MultiQuadCsPayloadRefState",
    "MultiQuadHose",
    "MultiQuadHoseFlats",
    "MultiQuadHoseInputs",
    "MultiQuadHoseRefState",
    "MultiQuadRigidbody",
    "MultiQuadRigidbodyFlats",
    "MultiQuadRigidbodyInputs",
    "MultiQuadRigidbodyRefState",
    "NotFlatError",
    "Quadrotor",
    "RingTensionDistribution",
    "RingTensionSplit",
    "TensionDistributionPolicy",
    "TensionSplitContractError",
    "TensionSplitPolicy",
    "WrenchAllocation",
    "WrenchAllocationError",
    "cable_chain_cascade",
    "cable_direction_jet",
    "chain_jet_order",
    "leibniz_dot",
    "leibniz_matmul",
    "leibniz_product",
    "normalize_derivatives",
    "null_space_basis",
    "point_mass_cable_force",
    "rotation_jet",
    "tension_direction_jet",
    "yaw_jet",
    "QuadrotorCsPayload",
    "QuadrotorCsPayloadFlats",
    "QuadrotorCsPayloadInputs",
    "QuadrotorCsPayloadRefState",
    "QuadrotorFlexCable",
    "QuadrotorFlexCableFlats",
    "QuadrotorFlexCableInputs",
    "QuadrotorFlexCableRefState",
    "QuadrotorFlats",
    "QuadrotorInputs",
    "QuadrotorRefState",
]
