"""Differential flatness for ``nQ`` quadrotors carrying one cable-suspended
point-mass payload.

Flat output::

    y = (x_L, {ψ_i}_{i=1..nQ}, {internal cable forces})

where ``x_L ∈ R³`` is the payload position, ``ψ_i ∈ R`` is quadrotor ``i``'s
yaw, and the *internal* (self-cancelling) part of the cable-force split makes up
the remaining ``3(nQ − 1)`` flat components.  Counting:
``3 + nQ + 3(nQ − 1) = 4 nQ = dim u`` — the dimension-matching condition for
differential flatness (see ``docs/theory/differential-flatness/``).

Why the split is a flat output
------------------------------
Newton's law on the point-mass payload only constrains the *sum* of the cable
forces::

    Σ_i 𝐓_i = F_net := -m_L (a_L + g e₃)

which is 3 equations in ``3 nQ`` unknowns.  For ``nQ > 1`` the shortfall — a
``3(nQ − 1)``-dimensional internal-force space — is a free design choice, and
choosing it is what this module calls a **tension-split policy**
(:class:`TensionSplitPolicy`).  A policy must supply the split *and all of its
time derivatives* analytically, because the flat-to-state map consumes jets, not
point values.

Sign convention
---------------
``𝐓_i = T_i q_i`` is the **force the cable exerts on quadrotor** ``i`` (it pulls
the quadrotor towards the payload), so ``q_i`` points from the quadrotor to the
payload and the force on the *payload* is ``-𝐓_i``.  This is the same convention
as :mod:`udaan.flatness.cable`, :mod:`udaan.flatness.quadrotor_cspayload` and
:mod:`udaan.models.multi_quad_cspayload`; at ``nQ = 1`` it reduces exactly to
``𝐓 = -m_L (ẍ_L + g e₃)`` (:func:`~udaan.flatness.cable.point_mass_cable_force`).

Derivative orders required::

    x_L through pop  (x_L, …, x_L^(6))
    ψ_i through ψ̈_i  (ψ_i, ψ̇_i, ψ̈_i)

— identical to the single-quadrotor case, because the default policies keep the
internal forces *constant*, so ``𝐓_i^(k) = α_i F_net^(k) = -α_i m_L x_L^(k+2)``.

References:
    K. Sreenath and V. Kumar, "Dynamics, Control and Planning for Cooperative
    Manipulation of Payloads Suspended by Cables from Multiple Quadrotor
    Robots", Robotics: Science and Systems (RSS), 2013.
    P. Kotaru, G. Wu, K. Sreenath, "Differential-Flatness and Control of
    Quadrotor(s) with a Payload Suspended through Flexible Cable(s)", ICC 2018.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_FORMATION_RADIUS,
    DEFAULT_NUM_QUADROTORS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from ..core.types import Vec3
from ..manif import S2, SO3, TS2, TSO3
from .base import Flat2State
from .cable import tension_direction_jet
from .jet import Jet
from .quadrotor import _attitude_from_thrust_vector

__all__ = [
    "DEFAULT_MIN_TENSION",
    "AffineTensionSplit",
    "LegacyFixedTensionSplit",
    "MultiQuadCsPayload",
    "MultiQuadCsPayloadFlats",
    "MultiQuadCsPayloadInputs",
    "MultiQuadCsPayloadRefState",
    "RingTensionSplit",
    "TensionSplitContractError",
    "TensionSplitPolicy",
]

_E3 = np.array([0.0, 0.0, 1.0])

#: Floor on ``‖𝐓_i‖`` below which a cable counts as slack and the map refuses to
#: return a reference (mirrors the zero-norm guard in
#: :func:`~udaan.flatness.derivatives.normalize_derivatives`, but with physical
#: units and a knob).
DEFAULT_MIN_TENSION = 1e-6  # N


class TensionSplitContractError(ValueError):
    """A :class:`TensionSplitPolicy` failed ``Σ_i 𝐓_i^(k) = F_net^(k)``.

    This is a *programming* error in the policy, not a physically infeasible
    reference: unlike the slack-cable and free-fall refusals — which a caller
    may legitimately absorb by dropping the feedforward for that step — there is
    no trajectory for which a policy that does not add up is acceptable.  It is
    a distinct type so that callers catching :class:`ValueError` around
    :meth:`MultiQuadCsPayload.recover` can let it through instead of silently
    degrading to feedback-only control (see
    :meth:`udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase._desired_attitudes`).
    """


def _zeros3() -> Vec3:
    return np.zeros(3)


# ─── Tension-split policies ───────────────────────────────────────────


class TensionSplitPolicy(ABC):
    """Strategy that resolves the payload force balance into per-cable forces.

    The payload equation ``Σ_i 𝐓_i = F_net`` under-determines the ``nQ`` cable
    forces for ``nQ > 1``; a policy picks one solution *and* differentiates it,
    since the flat-to-state map needs jets rather than point values.

    Implementations must satisfy, at every derivative order ``k``::

        Σ_i 𝐓_i^(k) = F_net^(k)

    which is what makes the recovered reference an exact solution of the
    equations of motion.  :meth:`MultiQuadCsPayload.recover` re-checks it at
    every order on every call — a policy that violates it would otherwise
    return a silently wrong reference — and the open-loop round-trip tests in
    ``tests/flatness/test_multi_quad_cspayload_feedforward.py`` catch the
    subtler failures the identity cannot see.
    """

    @property
    @abstractmethod
    def num_quadrotors(self) -> int:
        """Number of cables this policy splits across."""

    @property
    @abstractmethod
    def reference_net_force(self) -> np.ndarray:
        """``F̄_net`` — the net cable force the policy's reference is built at.

        :meth:`net_force_margin` is a bound on ``‖F_net − F̄_net‖``, so the
        reference point has to be part of the contract.
        """

    @property
    @abstractmethod
    def hover_directions(self) -> np.ndarray:
        """``(nQ, 3)`` unit cable directions ``q̄_i`` at the reference condition.

        :meth:`MultiQuadCsPayload.recover` requires the recovered ``q_i`` to
        keep a positive inner product with these, which is what stops it from
        silently crossing the slack boundary and inverting a cable.
        """

    @abstractmethod
    def split(self, net_force: list[np.ndarray]) -> list[list[np.ndarray]]:
        """Split a net-cable-force jet into per-cable force jets.

        Args:
            net_force: ``[F_net, Ḟ_net, …, F_net^(K)]``, the derivative tuple of
                ``F_net = -m_L (a_L + g e₃)``.

        Returns:
            A length-``nQ`` list; entry ``i`` is ``[𝐓_i, 𝐓̇_i, …, 𝐓_i^(K)]``.
        """

    @abstractmethod
    def net_force_margin(self, min_tension: float = 0.0) -> float:
        """Largest ``‖F_net − F̄_net‖`` for which every cable is provably taut.

        ``F̄_net`` is the policy's hover/reference net force.  A finite value is
        a *sufficient* (conservative) certificate; ``inf`` means the policy
        cannot slacken a cable at any net force.  See
        :meth:`MultiQuadCsPayload.payload_acceleration_bound` for the same
        statement expressed as a bound on ``‖a_L‖``.
        """


class AffineTensionSplit(TensionSplitPolicy):
    """Constant internal force plus a constant share of the net force.

    The whole family is::

        𝐓_i(t) = 𝐓̄_i + α_i · (F_net(t) − F̄_net),      F̄_net := Σ_j 𝐓̄_j

    with ``Σ_i α_i = 1``.  Equivalently, in the ``F_i = α_i F_net + N_i``
    decomposition, ``N_i = 𝐓̄_i − α_i F̄_net`` and ``Σ_i N_i = 0`` by
    construction.

    Two facts make this the right shape for a *trajectory* (as opposed to
    setpoint) allocator:

    * **The constraint holds at every order, exactly.**
      ``Σ_i 𝐓_i = F̄_net + (F_net − F̄_net) = F_net``, and for ``k ≥ 1``
      ``Σ_i 𝐓_i^(k) = (Σ_i α_i) F_net^(k) = F_net^(k)``.
    * **The derivatives are free.** ``𝐓̄_i`` and ``F̄_net`` are constants, so
      ``𝐓_i^(k) = α_i F_net^(k)`` for ``k ≥ 1`` — no numerical differentiation,
      no per-timestep optimizer, exact to arbitrary order.

    ``𝐓̄_i`` is the **static anchor**: the cable force at the reference
    (hover) condition, which fixes the formation geometry, since
    ``q̄_i = 𝐓̄_i / ‖𝐓̄_i‖`` and ``x_{Q_i} = x_L − l_i q̄_i``.  ``α_i`` is the
    **dynamic share**: the fraction of the trajectory's inertial load quadrotor
    ``i`` absorbs.

    Shares are *not* required to be non-negative: ``α_i < 0`` is physically
    meaningful (quadrotor ``i`` counteracts while another over-absorbs), and
    only ``Σ_i α_i = 1`` is needed for the contract.  A negative share moves
    ``𝐓_i`` just as fast as a positive one, which is why
    :meth:`net_force_margin` bounds by ``|α_i|``.

    Note that ``F̄_net = Σ_j 𝐓̄_j`` is whatever the anchors happen to sum to;
    this class does not require it to balance the payload weight.  Anchors that
    do not sum to ``-m_L g e₃`` describe a formation that is *not* in
    equilibrium at rest, and
    :meth:`MultiQuadCsPayload.payload_acceleration_bound` accounts for the
    resulting constant offset.

    Args:
        anchors: ``(nQ, 3)`` hover cable forces ``𝐓̄_i`` (pulling each quadrotor
            towards the payload, i.e. pointing *down* into the payload).
        shares: ``(nQ,)`` dynamic shares ``α_i``, summing to 1.
    """

    def __init__(self, anchors, shares):
        anchor_arr = np.asarray(anchors, dtype=float)
        if anchor_arr.ndim != 2 or anchor_arr.shape[1] != 3:
            raise ValueError(f"anchors must have shape (nQ, 3); got {anchor_arr.shape}")
        share_arr = np.asarray(shares, dtype=float).reshape(-1)
        if share_arr.shape[0] != anchor_arr.shape[0]:
            raise ValueError(
                f"shares must have one entry per anchor; got {share_arr.shape[0]} "
                f"shares for {anchor_arr.shape[0]} anchors"
            )
        if not np.isclose(float(share_arr.sum()), 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"shares must sum to 1; got {float(share_arr.sum())!r}")
        norms = np.linalg.norm(anchor_arr, axis=1)
        if np.any(norms <= 0.0):
            raise ValueError(
                "every anchor must be a non-zero cable force — a zero anchor "
                "leaves the cable direction undefined at hover"
            )
        self._anchors = anchor_arr
        self._shares = share_arr
        self._anchor_sum: np.ndarray = anchor_arr.sum(axis=0)

    @property
    def num_quadrotors(self) -> int:
        return int(self._anchors.shape[0])

    @property
    def anchors(self) -> np.ndarray:
        """``(nQ, 3)`` static anchor forces ``𝐓̄_i``."""
        return self._anchors

    @property
    def shares(self) -> np.ndarray:
        """``(nQ,)`` dynamic shares ``α_i``."""
        return self._shares

    @property
    def reference_net_force(self) -> np.ndarray:
        """``F̄_net = Σ_i 𝐓̄_i`` — the net force the anchors are in balance with."""
        return self._anchor_sum

    @property
    def hover_directions(self) -> np.ndarray:
        """``(nQ, 3)`` unit cable directions ``q̄_i`` implied by the anchors."""
        return np.asarray(self._anchors / np.linalg.norm(self._anchors, axis=1, keepdims=True))

    def split(self, net_force: list[np.ndarray]) -> list[list[np.ndarray]]:
        delta = np.asarray(net_force[0], dtype=float) - self._anchor_sum
        out: list[list[np.ndarray]] = []
        for i in range(self.num_quadrotors):
            jet = [self._anchors[i] + self._shares[i] * delta]
            jet.extend(
                self._shares[i] * np.asarray(net_force[k], dtype=float)
                for k in range(1, len(net_force))
            )
            out.append(jet)
        return out

    def net_force_margin(self, min_tension: float = 0.0) -> float:
        # ‖𝐓_i‖ = ‖𝐓̄_i + α_i Δ‖ ≥ ‖𝐓̄_i‖ − |α_i| ‖Δ‖, so every cable holds
        # ‖𝐓_i‖ > min_tension as long as ‖Δ‖ < min_i (‖𝐓̄_i‖ − min_tension)/|α_i|.
        # Only α_i == 0 exempts a cable: its force is constant.  A *negative*
        # share moves 𝐓_i exactly as fast as a positive one — just the other
        # way — so it binds through |α_i|.
        norms = np.linalg.norm(self._anchors, axis=1)
        bound = np.inf
        for i in range(self.num_quadrotors):
            share = abs(float(self._shares[i]))
            if share == 0.0:
                continue  # this cable's force never moves — always taut
            bound = min(bound, (norms[i] - min_tension) / share)
        return float(max(bound, 0.0))


class RingTensionSplit(AffineTensionSplit):
    """Default policy — a ring formation anchor with mass-weighted dynamics.

    **Static anchor (geometry, length-weighted).** The quadrotors hover on a
    horizontal circle of ``radius`` centred above the payload, quadrotor ``i``
    at bearing ``θ_i = 2πi/nQ`` and height ``dz_i = √(l_i² − radius²)``.  The
    unique anchor set that holds that formation in equilibrium with *equal
    horizontal pull* per cable is

    .. math::

        \\bar{\\mathbf{T}}_i = -c\\,\\bigl[r\\cos θ_i,\\; r\\sin θ_i,\\; dz_i\\bigr],
        \\qquad c = \\frac{m_L\\, g}{\\sum_j dz_j},

    whose sum is ``-m_L g e₃`` because the ring's horizontal terms cancel
    (``Σ cos θ_i = Σ sin θ_i = 0`` for ``nQ ≥ 2``).  This reproduces
    :func:`udaan.models.multi_quad_cspayload.formation_cable_attitudes`
    exactly, so the flat reference at hover *is* the model's default reset
    configuration.  Tension magnitudes come out ``T̄_i = c · l_i``: longer
    cables are more vertical and therefore carry proportionally more.

    **Dynamic share (mass-weighted).** ``α_i = m_i / Σ_j m_j``.  Every
    quadrotor participates at *every* derivative order, in proportion to its
    own mass — a proxy for lift capacity — so heavier vehicles absorb more of
    the trajectory's inertial load and the formation stays balanced through the
    maneuver.

    **Why the ring is not optional.** Minimising ``Σ_i ‖𝐓_i‖²`` subject to
    ``Σ_i 𝐓_i = F_net`` alone gives ``𝐓_i = F_net/nQ`` — every cable parallel,
    every quadrotor stacked vertically above the payload at the same point.
    The min-norm objective must be constrained by geometry to be useful (as in
    the static solver of the flexible-hose codebase, which adds unit-norm,
    endpoint-closure and per-vertex balance constraints).  Here the ring
    radius plays that role directly: it *is* the physical knob, and pushing it
    to zero collapses the formation.

    **``nQ = 1`` has no ring.** A lone quadrotor hovers directly above the
    payload (``q̄ = -e₃``), so ``radius`` is *silently ignored* — including the
    ``0 < radius < min_i l_i`` validation, which does not apply.  This matches
    :func:`udaan.models.multi_quad_cspayload.formation_cable_attitudes`, which
    special-cases ``nQ = 1`` the same way, and it is why
    :attr:`radius` reads back as ``0.0`` there rather than as the value passed.

    Args:
        num_quadrotors: number of quadrotors ``nQ ≥ 1``.
        payload_mass: ``m_L``.
        masses: ``(nQ,)`` quadrotor masses (mass-weighted shares).
        cable_lengths: ``(nQ,)`` cable lengths.
        radius: formation radius; must satisfy ``0 < radius < min_i l_i`` for
            ``nQ ≥ 2``.  Ignored at ``nQ = 1``, where the single quadrotor
            hovers directly above the payload.
        gravity: ``g``.
    """

    def __init__(
        self,
        num_quadrotors: int,
        payload_mass: float,
        masses,
        cable_lengths,
        radius: float = DEFAULT_FORMATION_RADIUS,
        gravity: float = GRAVITY,
    ):
        nQ = int(num_quadrotors)
        if nQ < 1:
            raise ValueError(f"num_quadrotors must be >= 1, got {nQ}")
        mass_arr = np.asarray(masses, dtype=float).reshape(-1)
        length_arr = np.asarray(cable_lengths, dtype=float).reshape(-1)
        if mass_arr.shape[0] != nQ or length_arr.shape[0] != nQ:
            raise ValueError(
                f"masses and cable_lengths must have {nQ} entries; got "
                f"{mass_arr.shape[0]} and {length_arr.shape[0]}"
            )
        if np.any(mass_arr <= 0.0):
            raise ValueError(f"masses must be positive; got {mass_arr.tolist()}")
        if np.any(length_arr <= 0.0):
            raise ValueError(f"cable_lengths must be positive; got {length_arr.tolist()}")
        if payload_mass <= 0.0:
            raise ValueError(f"payload_mass must be positive; got {payload_mass}")

        if nQ == 1:
            offsets = np.array([[0.0, 0.0, length_arr[0]]])
        else:
            r = float(radius)
            if r <= 0.0:
                raise ValueError(
                    "formation radius must be strictly positive: at radius 0 every "
                    "cable is vertical, the quadrotors stack on top of each other, "
                    "and the formation is degenerate"
                )
            if np.any(length_arr <= r):
                raise ValueError(
                    f"formation radius {r} must be strictly below every cable "
                    f"length {length_arr.tolist()}"
                )
            angles = np.linspace(0.0, 2 * np.pi, nQ, endpoint=False)
            offsets = np.stack(
                [
                    r * np.cos(angles),
                    r * np.sin(angles),
                    np.sqrt(length_arr**2 - r**2),
                ],
                axis=1,
            )

        c = float(payload_mass) * float(gravity) / float(offsets[:, 2].sum())
        anchors = -c * offsets
        shares = mass_arr / float(mass_arr.sum())
        super().__init__(anchors, shares)
        self.radius = 0.0 if nQ == 1 else float(radius)


class LegacyFixedTensionSplit(AffineTensionSplit):
    """The published ``nQ = 3`` fixed split — reproducibility only, not a default.

    Ports ``Flat2state_mQpoint.m`` (lines 21–31) of the
    ``quadrotor-flexible-cables-payload`` GeoControl toolbox, which hard-codes

    .. code-block:: matlab

        traj.T(3).T = (1/3)*mL*g*RotX( pi/6)*[0;0;1];
        traj.T(2).T = (1/3)*mL*g*RotX(-pi/6)*[0;0;1];
        traj.T(1).T = mL*ddxL - T3 - T2 + mL*g*e3;
        traj.T(3).dT{j} = 0;  traj.T(2).dT{j} = 0;  traj.T(1).dT{j} = mL*xL^(j+2);

    In this module's sign convention (``𝐓_i`` is the force *on the quadrotor*,
    the negative of the MATLAB variable) that is exactly an
    :class:`AffineTensionSplit` with ``α = (1, 0, 0)``.  Four reasons it is not
    the default:

    1. **Asymmetric.** Quadrotors 2 and 3 are frozen at a constant world-frame
       force and contribute *nothing at any derivative order*; quadrotor 1
       absorbs 100 % of the trajectory dynamics. Its taut-cable margin is
       therefore the whole system's margin.
    2. **Mass-agnostic.** The shares ignore ``m_i`` entirely, even though the
       same reference deliberately gives the three quadrotors unequal masses.
    3. **``nQ = 3`` only.** There is no generic-``nQ`` statement of it.
    4. **Geometrically degenerate.** ``q̄_1 = -e₃`` and ``q̄_{2,3}`` tilt by 30°
       about the *x*-axis, so all three quadrotors lie in the ``x = 0`` plane —
       the formation has no extent along ``x``.

    Args:
        payload_mass: ``m_L``.
        gravity: ``g``.
    """

    def __init__(self, payload_mass: float = DEFAULT_PAYLOAD_MASS, gravity: float = GRAVITY):
        if payload_mass <= 0.0:
            raise ValueError(f"payload_mass must be positive; got {payload_mass}")
        mLg = float(payload_mass) * float(gravity)
        # RotX(φ) e₃ = [0, -sin φ, cos φ]; the MATLAB tensions are the forces on
        # the *payload*, so negate them for this module's convention.
        tilt = np.pi / 6.0
        anchor_3 = -(mLg / 3.0) * np.array([0.0, -np.sin(tilt), np.cos(tilt)])
        anchor_2 = -(mLg / 3.0) * np.array([0.0, np.sin(tilt), np.cos(tilt)])
        anchor_1 = -mLg * _E3 - anchor_2 - anchor_3
        super().__init__(
            anchors=np.stack([anchor_1, anchor_2, anchor_3]),
            shares=np.array([1.0, 0.0, 0.0]),
        )


# ─── Flat output / reference state / inputs ───────────────────────────


@dataclass
class MultiQuadCsPayloadFlats:
    """Flat output of ``nQ`` quadrotors sharing one cable-suspended payload.

    The internal-force part of the flat output is carried by the map's
    :class:`TensionSplitPolicy` rather than by this struct: the shipped policies
    hold it constant, which is what keeps the required jet orders at those of
    the single-quadrotor map.

    Attributes:
        x_L: dim-3 :class:`Jet` of the payload position, derivatives through
             pop (order ≥ 6) — ``q_i^(4)`` and hence the moment feedforward
             consume the 6th derivative of ``x_L``.
        psi: length-``nQ`` list of scalar :class:`Jet` s, one yaw angle per
             quadrotor, each with derivatives through ``ψ̈`` (order ≥ 2).
    """

    x_L: Jet  # dim 3, order >= 6 (through pop)
    psi: list[Jet]  # nQ scalar jets, each order >= 2 (through ψ̈)

    def __post_init__(self):
        if self.x_L.dim != 3:
            raise ValueError(f"x_L must be a 3-D jet; got dim={self.x_L.dim}")
        if self.x_L.order < 6:
            raise ValueError(f"x_L.order must be >= 6 (pop); got {self.x_L.order}")
        if isinstance(self.psi, Jet):
            raise ValueError("psi must be a list of scalar jets, one per quadrotor")
        self.psi = list(self.psi)
        if not self.psi:
            raise ValueError("psi must hold at least one yaw jet")
        for i, psi_i in enumerate(self.psi):
            if not isinstance(psi_i, Jet):
                raise ValueError(f"psi[{i}] must be a Jet; got {type(psi_i).__name__}")
            if not psi_i.is_scalar:
                raise ValueError(f"psi[{i}] must be a scalar jet; got dim={psi_i.dim}")
            if psi_i.order < 2:
                raise ValueError(f"psi[{i}].order must be >= 2; got {psi_i.order}")

    @property
    def num_quadrotors(self) -> int:
        """Number of quadrotors, inferred from the yaw list."""
        return len(self.psi)


@dataclass
class MultiQuadCsPayloadRefState:
    """Recovered kinematic reference state for the multi-quad cspayload system.

    Field naming mirrors
    :class:`udaan.models.multi_quad_cspayload.base.MultiQuadCsPayloadState`
    where the two overlap; the ``*_acceleration(s)`` entries are extras that the
    flatness map also produces.  Every per-quadrotor field is a list indexed by
    quadrotor.
    """

    # Payload kinematics
    payload_position: Vec3 = field(default_factory=_zeros3)
    payload_velocity: Vec3 = field(default_factory=_zeros3)
    payload_acceleration: Vec3 = field(default_factory=_zeros3)

    # Per-quadrotor centre-of-mass kinematics
    positions: list[Vec3] = field(default_factory=list)
    velocities: list[Vec3] = field(default_factory=list)
    accelerations: list[Vec3] = field(default_factory=list)

    # Per-cable kinematics on TS²
    cable_attitudes: list[Vec3] = field(default_factory=list)
    cable_angular_velocities: list[Vec3] = field(default_factory=list)
    cable_angular_accelerations: list[Vec3] = field(default_factory=list)

    # Per-quadrotor attitude
    orientations: list[SO3] = field(default_factory=list)
    angular_velocities: list[TSO3] = field(default_factory=list)
    angular_accelerations: list[Vec3] = field(default_factory=list)

    @property
    def num_quadrotors(self) -> int:
        return len(self.positions)

    def as_state(self):
        """Project onto the model's ``MultiQuadCsPayloadState`` shape.

        Drops the ``*_acceleration(s)`` extras and wraps the cable fields as
        :class:`~udaan.manif.S2` / :class:`~udaan.manif.TS2`.
        """
        from ..models.multi_quad_cspayload.base import MultiQuadCsPayloadState

        return MultiQuadCsPayloadState(
            num_quadrotors=self.num_quadrotors,
            payload_position=self.payload_position,
            payload_velocity=self.payload_velocity,
            positions=[np.asarray(p).copy() for p in self.positions],
            velocities=[np.asarray(v).copy() for v in self.velocities],
            orientations=[SO3(np.asarray(R)) for R in self.orientations],
            angular_velocities=[TSO3(np.asarray(w)) for w in self.angular_velocities],
            cable_attitudes=[S2(np.asarray(q)) for q in self.cable_attitudes],
            cable_angular_velocities=[TS2(np.asarray(w)) for w in self.cable_angular_velocities],
        )


@dataclass
class MultiQuadCsPayloadInputs:
    """Feedforward inputs recovered from the flat output, one entry per quadrotor.

    Attributes:
        thrusts:  collective thrust magnitudes ``f_i``.
        moments:  body-frame moments ``M_i``.
        tensions: cable tensions ``T_i = ‖𝐓_i‖``.  Positive throughout the taut
                  regime; ``T_i → 0`` is the slack-cable boundary.
                  :meth:`MultiQuadCsPayload.recover` refuses to cross it, but
                  the norm alone cannot see the crossing (it is positive again
                  on the far side), so the refusal rests on the *sign* check
                  ``𝐓_i · q̄_i > 0`` against the policy's hover direction, not
                  on ``T_i`` alone.
    """

    thrusts: list[float] = field(default_factory=list)
    moments: list[Vec3] = field(default_factory=list)
    tensions: list[float] = field(default_factory=list)

    @property
    def wrench(self) -> np.ndarray:
        """``(nQ, 4)`` stack of ``[f_i, M_i]`` — the ``input="wrench"`` shape of
        :meth:`udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase.step`."""
        return np.array([[f, *M] for f, M in zip(self.thrusts, self.moments, strict=True)])


# ─── The map ──────────────────────────────────────────────────────────


class MultiQuadCsPayload(Flat2State):
    """Differential-flatness recovery for ``nQ`` quadrotors carrying one
    cable-suspended point-mass payload through taut, massless cables.

    Args:
        num_quadrotors: ``nQ ≥ 1``.
        masses: scalar (broadcast) or ``(nQ,)`` quadrotor masses.
        payload_mass: ``m_L``.
        inertias: ``(3,)`` diagonal, ``(3, 3)`` tensor (both broadcast) or
            ``(nQ, 3, 3)`` stack.
        cable_lengths: scalar (broadcast) or ``(nQ,)`` cable lengths.
        gravity: ``g``.
        formation_radius: radius of the default :class:`RingTensionSplit`.
        tension_policy: an explicit :class:`TensionSplitPolicy`; defaults to
            :class:`RingTensionSplit` built from the arguments above.
        min_tension: cables below this tension are reported as slack.
    """

    Flats = MultiQuadCsPayloadFlats
    RefState = MultiQuadCsPayloadRefState
    Inputs = MultiQuadCsPayloadInputs

    def __init__(
        self,
        *,
        num_quadrotors: int = DEFAULT_NUM_QUADROTORS,
        masses=DEFAULT_QUAD_MASS,
        payload_mass: float = DEFAULT_PAYLOAD_MASS,
        inertias=DEFAULT_QUAD_INERTIA,
        cable_lengths=DEFAULT_CABLE_LENGTH,
        gravity: float = GRAVITY,
        formation_radius: float = DEFAULT_FORMATION_RADIUS,
        tension_policy: TensionSplitPolicy | None = None,
        min_tension: float = DEFAULT_MIN_TENSION,
    ):
        nQ = int(num_quadrotors)
        if nQ < 1:
            raise ValueError(f"num_quadrotors must be >= 1, got {nQ}")
        self._num_quadrotors = nQ

        self.masses = self._broadcast(masses, "masses")
        if np.any(self.masses <= 0.0):
            raise ValueError(f"masses must be positive; got {self.masses.tolist()}")

        self.payload_mass = float(payload_mass)
        if self.payload_mass <= 0.0:
            raise ValueError(f"payload_mass must be positive; got {self.payload_mass}")

        self.cable_lengths = self._broadcast(cable_lengths, "cable_lengths")
        if np.any(self.cable_lengths <= 0.0):
            raise ValueError(f"cable_lengths must be positive; got {self.cable_lengths.tolist()}")

        self.inertias = self._broadcast_inertias(inertias)
        self.gravity = float(gravity)
        self.formation_radius = float(formation_radius)
        self.min_tension = float(min_tension)

        if tension_policy is None:
            tension_policy = RingTensionSplit(
                num_quadrotors=nQ,
                payload_mass=self.payload_mass,
                masses=self.masses,
                cable_lengths=self.cable_lengths,
                radius=self.formation_radius,
                gravity=self.gravity,
            )
        if tension_policy.num_quadrotors != nQ:
            raise ValueError(
                f"tension_policy is built for {tension_policy.num_quadrotors} "
                f"quadrotors, but this map has {nQ}"
            )
        self.tension_policy = tension_policy

    # ─── Parameter plumbing ────────────────────────────────────────

    def _broadcast(self, value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self._num_quadrotors, float(arr))
        arr = arr.reshape(-1)
        if arr.shape[0] != self._num_quadrotors:
            raise ValueError(f"{name} must have {self._num_quadrotors} entries, got {arr.shape[0]}")
        return arr

    def _broadcast_inertias(self, value) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        nQ = self._num_quadrotors
        if arr.shape == (3,):
            return np.tile(np.diag(arr), (nQ, 1, 1))
        if arr.shape == (3, 3):
            return np.tile(arr, (nQ, 1, 1))
        if arr.shape == (nQ, 3, 3):
            return arr
        raise ValueError(f"inertias must be (3,), (3, 3) or (nQ, 3, 3); got {arr.shape}")

    @property
    def num_quadrotors(self) -> int:
        return self._num_quadrotors

    @classmethod
    def from_model(cls, model, tension_policy: TensionSplitPolicy | None = None):
        """Build from a
        :class:`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase`-like model.

        The default policy is then the ring at the model's own
        ``formation_radius``, so the flat reference at hover coincides with the
        model's ``reset()`` configuration.
        """
        return cls(
            num_quadrotors=model.num_quadrotors,
            masses=model.masses,
            payload_mass=model.payload_mass,
            inertias=model.inertias,
            cable_lengths=model.cable_lengths,
            formation_radius=model.formation_radius,
            tension_policy=tension_policy,
        )

    # ─── Taut-cable certificate ────────────────────────────────────

    def payload_acceleration_bound(self, min_tension: float | None = None) -> float:
        """Sufficient bound on ``‖a_L‖`` for every cable to stay taut.

        The policy certifies tautness for ``‖Δ‖ < margin``, where
        ``Δ := F_net − F̄_net`` and ``margin`` is
        :meth:`~TensionSplitPolicy.net_force_margin`.  Since
        ``F_net = -m_L (a_L + g e₃)``,

        .. math::

            Δ = -m_L a_L + d, \\qquad
            d := -m_L g e₃ - \\bar F_{\\mathrm{net}},

        so ``‖Δ‖ ≤ m_L ‖a_L‖ + ‖d‖`` and every cable stays taut whenever

        .. math::

            \\|a_L\\| < \\frac{\\mathrm{margin} - \\|d\\|}{m_L}.

        The offset ``d`` measures how far the policy's anchors are from
        balancing the payload weight: it vanishes for both shipped policies
        (and then the bound is simply ``margin / m_L``), but a hand-built
        :class:`AffineTensionSplit` need not be in equilibrium at rest, and
        then part of the margin is already spent at hover.  ``d`` alone
        exceeding the margin means no acceleration is certified, and the bound
        is 0.

        The result is conservative (two triangle inequalities), not tight —
        :meth:`recover` still checks the true tension at every sample.
        """
        margin = self.tension_policy.net_force_margin(
            self.min_tension if min_tension is None else float(min_tension)
        )
        if not np.isfinite(margin):
            return float("inf")
        offset = float(
            np.linalg.norm(
                -self.payload_mass * self.gravity * _E3
                - np.asarray(self.tension_policy.reference_net_force, dtype=float)
            )
        )
        return max(margin - offset, 0.0) / self.payload_mass

    # ─── Recovery ──────────────────────────────────────────────────

    def recover(
        self, flats: MultiQuadCsPayloadFlats
    ) -> tuple[MultiQuadCsPayloadRefState, MultiQuadCsPayloadInputs]:
        nQ = self._num_quadrotors
        if flats.num_quadrotors != nQ:
            raise ValueError(
                f"flats carry {flats.num_quadrotors} yaw jets but the map has {nQ} quadrotors"
            )
        g = self.gravity
        mL = self.payload_mass

        # ── Payload-position derivatives, x_L through pop ──────────────
        x_L: list[np.ndarray] = [np.asarray(flats.x_L[k], dtype=float) for k in range(7)]

        # ── Step 1: net cable force on the payload side, with derivatives.
        # Newton on the point mass: m_L ẍ_L = -Σ_i 𝐓_i - m_L g e₃, so
        #   F_net = Σ_i 𝐓_i = -m_L (ẍ_L + g e₃),   F_net^(k) = -m_L x_L^(k+2).
        # Four derivative levels are needed because q_i^(4) closes the chain.
        net_force = [-mL * (x_L[2] + g * _E3)]
        net_force.extend(-mL * x_L[k] for k in range(3, 7))

        # ── Step 2: split it — the free 3(nQ−1)-dimensional choice.
        cable_forces = self.tension_policy.split(net_force)

        # The policy's contract, Σ_i 𝐓_i^(k) = F_net^(k), is what makes the
        # recovered reference an exact solution; a policy that breaks it
        # produces a plausible-looking but wrong reference, so check it here
        # rather than letting it escape into a controller.
        for k, net_k in enumerate(net_force):
            total_k = np.sum([f[k] for f in cable_forces], axis=0)
            if not np.allclose(total_k, net_k, rtol=1e-9, atol=1e-9):
                raise TensionSplitContractError(
                    f"{type(self.tension_policy).__name__} violates the split "
                    f"contract at derivative order {k}: Σ_i 𝐓_i^({k}) = "
                    f"{np.asarray(total_k).tolist()} but F_net^({k}) = "
                    f"{np.asarray(net_k).tolist()}"
                )

        hover_dirs = np.asarray(self.tension_policy.hover_directions, dtype=float)

        ref = MultiQuadCsPayloadRefState(
            payload_position=x_L[0],
            payload_velocity=x_L[1],
            payload_acceleration=x_L[2],
        )
        inputs = MultiQuadCsPayloadInputs()

        for i in range(nQ):
            m_i = float(self.masses[i])
            l_i = float(self.cable_lengths[i])
            J_i = self.inertias[i]
            force = cable_forces[i]

            tension = float(np.linalg.norm(force[0]))
            if tension < self.min_tension:
                raise ValueError(
                    f"MultiQuadCsPayload.recover: cable {i} would go slack "
                    f"(T = {tension:.3e} N < min_tension = {self.min_tension:.3e} N) — "
                    "the requested payload trajectory is too aggressive for this "
                    "tension-split policy. Reduce the payload acceleration below "
                    f"{self.payload_acceleration_bound():.3f} m/s², enlarge the "
                    "formation radius, or supply a different tension_policy."
                )

            # ‖𝐓_i‖ is a norm, so it turns positive again the instant the
            # trajectory *overshoots* the slack point: the tension floor alone
            # only refuses to land on the boundary, never to cross it, and past
            # the crossing q_i has flipped and quadrotor i sits on the far side
            # of the payload.  Requiring 𝐓_i to keep its sign against the
            # anchor 𝐓̄_i is what makes the refusal two-sided.  It is a
            # sufficient condition, deliberately conservative: it also rejects
            # a cable that has rotated more than 90° away from its hover
            # direction without ever going slack.
            if float(np.dot(force[0], hover_dirs[i])) <= 0.0:
                raise ValueError(
                    f"MultiQuadCsPayload.recover: cable {i} has crossed the "
                    "slack boundary — the recovered cable force points away "
                    f"from its hover direction q̄_{i} = {hover_dirs[i].tolist()}, "
                    f"so quadrotor {i} would sit on the far side of the "
                    "payload. The requested payload trajectory is too "
                    "aggressive for this tension-split policy: reduce the "
                    f"payload acceleration below "
                    f"{self.payload_acceleration_bound():.3f} m/s², enlarge the "
                    "formation radius, or supply a different tension_policy."
                )

            # ── Step 3: per-cable tension magnitude and direction jets.
            # 𝐓_i = T_i q_i with q_i pointing quadrotor → payload; identical
            # machinery to the single-quadrotor map, only the force jet differs.
            T_derivs, q_d = tension_direction_jet(force)
            q, qdot, qddot, qd3, qd4 = q_d[0], q_d[1], q_d[2], q_d[3], q_d[4]

            # ── Step 4: quadrotor position derivatives, x_{Q_i} = x_L - l_i q_i.
            xQ = x_L[0] - l_i * q
            vQ = x_L[1] - l_i * qdot
            aQ = x_L[2] - l_i * qddot
            jQ = x_L[3] - l_i * qd3
            sQ = x_L[4] - l_i * qd4

            # ── Step 5: cable rates on T_{q_i}S². ω = q × q̇, ω̇ = q × q̈.
            omega_cable = np.cross(q, qdot)
            domega_cable = np.cross(q, qddot)

            # ── Step 6: thrust vector. Newton on quadrotor i, whose cable
            # reaction is +𝐓_i (Newton's third law against the payload):
            #   m_i ẍ_{Q_i} = f_i R_i e₃ - m_i g e₃ + 𝐓_i
            #   ⇒ B_i = f_i R_i e₃ = m_i (ẍ_{Q_i} + g e₃) - 𝐓_i.
            B = m_i * (aQ + g * _E3) - force[0]
            dB = m_i * jQ - force[1]
            d2B = m_i * sQ - force[2]

            if float(np.linalg.norm(B)) < 1e-6:
                raise ValueError(
                    f"MultiQuadCsPayload.recover: quadrotor {i} has a vanishing "
                    "thrust vector (free fall) — no unique attitude."
                )

            # ── Step 7: R_i, Ω_i, Ω̇_i, M_i from (B_i, ψ_i).
            psi_i = flats.psi[i]
            f, R, Omega, dOmega, moment = _attitude_from_thrust_vector(
                B,
                dB,
                d2B,
                float(psi_i.value),
                float(psi_i.velocity),
                float(psi_i.acceleration),
                J_i,
            )

            ref.positions.append(xQ)
            ref.velocities.append(vQ)
            ref.accelerations.append(aQ)
            ref.cable_attitudes.append(q)
            ref.cable_angular_velocities.append(omega_cable)
            ref.cable_angular_accelerations.append(domega_cable)
            ref.orientations.append(SO3(R))
            ref.angular_velocities.append(TSO3(Omega))
            ref.angular_accelerations.append(dOmega)

            inputs.thrusts.append(f)
            inputs.moments.append(moment)
            inputs.tensions.append(float(T_derivs[0]))

        return ref, inputs
