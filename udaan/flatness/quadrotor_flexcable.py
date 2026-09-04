r"""Differential flatness for a quadrotor with an ``n``-link flexible cable — M5.

Flat output::

    y = (x_L, ψ)

where ``x_L ∈ R³`` is the **payload** position (the tip of the chain) and
``ψ ∈ R`` is the quadrotor yaw — the same flat output as M2
(:mod:`udaan.flatness.quadrotor_cspayload`), which is the ``n = 1`` case of
this map.  The dynamics being inverted are
:class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase`.

Derivative orders required::

    x_L through its (2n + 4)-th derivative
    ψ   through ψ̈

**The jet depth grows with the link count.**  That is the qualitative
difference from every map in this package so far: M1 needs snap, M2 needs pop,
M3 and M4 need order 6 no matter how many quadrotors are added.  Here each
extra link costs two more derivatives of the payload trajectory, because the
chain cascade (:mod:`udaan.flatness.chain`) spends two on every rung — the
force in link ``i`` is built from the *acceleration* of the node below it.  At
``n = 1`` the requirement is 6, reproducing M2 exactly; at ``n = 5`` it is 14.
See :func:`~udaan.flatness.chain.chain_jet_order` and the "Derivative-order
accounting" section of ``docs/theory/differential-flatness/quadrotor-flexcable.md``.

Structure of the recovery::

    x_L jet  ──chain cascade──▶  q_i, T_i, node positions, F_0
                                              │
                                x_Q = node 0 ─┤
                                              ▼
                        f R e₃ = m_Q (ẍ_Q + g e₃) − F_0
                                              │
                                              ▼
                          _attitude_from_thrust_vector  ──▶  (f, R, Ω, Ω̇, M)

Only the first box is new; the second and third are the shared primitives every
thrust-vector map in this package already uses.

Sign convention.  ``q_i`` points **from the quadrotor towards the payload**
(``-e₃`` at rest), matching
:attr:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableState.link_attitudes`
and the rest of the codebase — and, as it happens, the legacy MATLAB reference
too.  What that reference carries is the *opposite force*, ``B_i = -F_i``, and
so a minus sign in every direction formula; :mod:`udaan.flatness.chain` folds
it into the force once and nothing downstream carries a stray sign.

Reference:
    P. Kotaru, G. Wu, K. Sreenath, "Differential-Flatness and Control of
    Quadrotor(s) with a Payload Suspended through Flexible Cable(s)",
    Indian Control Conference, 2018.
    (Dynamics: F. A. Goodarzi, D. Lee, T. Lee, "Geometric Control of a
    Quadrotor UAV Transporting a Payload Connected via Flexible Cable",
    ACC 2014.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_CABLE_MASS,
    DEFAULT_NUM_CABLE_LINKS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from ..core.types import Mat3, Vec3
from ..manif import S2, SO3, TS2, TSO3
from .base import Flat2State
from .chain import CableChain, cable_chain_cascade, chain_jet_order
from .jet import Jet
from .quadrotor import _attitude_from_thrust_vector

_E3 = np.array([0.0, 0.0, 1.0])

__all__ = [
    "QuadrotorFlexCable",
    "QuadrotorFlexCableFlats",
    "QuadrotorFlexCableInputs",
    "QuadrotorFlexCableRefState",
]


def _zeros3() -> Vec3:
    return np.zeros(3)


@dataclass
class QuadrotorFlexCableFlats:
    """Flat output of M5: payload-position jet and quadrotor-yaw jet.

    Attributes:
        x_L:  dim-3 :class:`Jet` of the **payload** position.  The order it
              must reach depends on the number of links —
              :func:`~udaan.flatness.chain.chain_jet_order` gives ``2n + 4`` —
              so the binding check lives in
              :meth:`QuadrotorFlexCable.recover`, which knows ``n``.  What is
              checked here is the ``n``-independent floor: dim 3 and order
              ≥ 6, the ``n = 1`` requirement.
        psi:  scalar :class:`Jet` of the yaw angle, order ≥ 2.
    """

    x_L: Jet  # dim 3, order >= 2n + 4 (checked against n in `recover`)
    psi: Jet  # dim 1 scalar, order >= 2 (through ψ̈)

    def __post_init__(self):
        if self.x_L.dim != 3:
            raise ValueError(f"x_L must be a 3-D jet; got dim={self.x_L.dim}")
        if self.x_L.order < 6:
            raise ValueError(
                f"x_L.order must be >= 6 (the n = 1 floor; an n-link chain needs "
                f"2n + 4); got {self.x_L.order}"
            )
        if not self.psi.is_scalar:
            raise ValueError(f"psi must be a scalar jet; got dim={self.psi.dim}")
        if self.psi.order < 2:
            raise ValueError(f"psi.order must be >= 2; got {self.psi.order}")


@dataclass
class QuadrotorFlexCableRefState:
    """Recovered kinematic reference state for a quadrotor with an ``n``-link cable.

    Field naming mirrors
    :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableState` where the
    two overlap; the ``*_acceleration`` fields and the node kinematics are
    extras the flatness map produces for free.

    Per-link lists are ordered **quadrotor → payload**, like the model's.
    """

    # Payload (chain tip) kinematics
    payload_position: Vec3 = field(default_factory=_zeros3)
    payload_velocity: Vec3 = field(default_factory=_zeros3)
    payload_acceleration: Vec3 = field(default_factory=_zeros3)

    # Quadrotor centre-of-mass kinematics
    position: Vec3 = field(default_factory=_zeros3)
    velocity: Vec3 = field(default_factory=_zeros3)
    acceleration: Vec3 = field(default_factory=_zeros3)

    # Per-link direction / rate / rate-derivative
    link_attitudes: list[Vec3] = field(default_factory=list)
    link_angular_velocities: list[Vec3] = field(default_factory=list)
    link_angular_accelerations: list[Vec3] = field(default_factory=list)

    # Lumped-mass (node) kinematics, ``(n, 3)`` each; row i is the distal end
    # of link i, so the last row repeats the payload.
    link_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    link_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    # Quadrotor attitude
    orientation: SO3 = field(default_factory=SO3)
    angular_velocity: TSO3 = field(default_factory=TSO3)
    angular_acceleration: Vec3 = field(default_factory=_zeros3)

    def as_state(self):
        """Project onto the model's ``QuadrotorFlexCableState`` shape.

        Drops the ``*_acceleration`` extras and the node kinematics, and wraps
        the per-link entries as :class:`~udaan.manif.S2` / :class:`~udaan.manif.TS2`.
        """
        from ..models.quadrotor_flexcable.base import QuadrotorFlexCableState

        return QuadrotorFlexCableState(
            num_links=len(self.link_attitudes),
            payload_position=self.payload_position,
            payload_velocity=self.payload_velocity,
            position=self.position,
            velocity=self.velocity,
            orientation=self.orientation,
            angular_velocity=self.angular_velocity,
            link_attitudes=[S2(np.asarray(q)) for q in self.link_attitudes],
            link_angular_velocities=[TS2(np.asarray(w)) for w in self.link_angular_velocities],
        )


@dataclass
class QuadrotorFlexCableInputs:
    """Feedforward inputs recovered from the flat output.

    Attributes:
        thrust:   collective thrust magnitude ``f``.
        moment:   body-frame moment ``M``.
        tensions: ``(n,)`` array of link tensions, ``tensions[i] = ‖F_i‖ ≥ 0``,
                  ordered quadrotor → payload.  Not an actuator input — the
                  chain's internal forces — but the slack diagnostic:
                  ``tensions[i] → 0`` is the hybrid-mode boundary at which link
                  ``i`` goes slack and the map ceases to apply.  It is a
                  **one-sided** margin: being a norm it turns positive again
                  past the crossing, so it says how close the trajectory is to
                  the boundary but cannot tell which side of it the recovery
                  landed on.  Crossing is refused separately, by the
                  hanging-branch check in :mod:`udaan.flatness.chain`.
    """

    thrust: float = 0.0
    moment: Vec3 = field(default_factory=_zeros3)
    tensions: np.ndarray = field(default_factory=lambda: np.zeros(0))


class QuadrotorFlexCable(Flat2State):
    """Differential-flatness recovery for a quadrotor carrying a point-mass
    payload through a taut ``n``-link flexible cable (M5).

    Parameters mirror
    :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase`: link
    masses and lengths are per-index, and the ``cable_mass`` / ``payload_mass``
    / ``cable_length`` convenience builds the default arrays (cable mass spread
    evenly, payload added to the last link) exactly as the model does, so
    :meth:`from_model` and a hand-built map agree.
    """

    Flats = QuadrotorFlexCableFlats
    RefState = QuadrotorFlexCableRefState
    Inputs = QuadrotorFlexCableInputs

    def __init__(
        self,
        *,
        num_links: int = DEFAULT_NUM_CABLE_LINKS,
        mass_q: float = DEFAULT_QUAD_MASS,
        inertia: Mat3 = DEFAULT_QUAD_INERTIA,
        link_masses=None,
        link_lengths=None,
        cable_length: float = DEFAULT_CABLE_LENGTH,
        cable_mass: float = DEFAULT_CABLE_MASS,
        payload_mass: float = DEFAULT_PAYLOAD_MASS,
        gravity: float = GRAVITY,
    ):
        self.num_links = int(num_links)
        if self.num_links < 1:
            raise ValueError(f"num_links must be >= 1, got {self.num_links}")
        n = self.num_links

        self.mass_q = float(mass_q)
        if self.mass_q <= 0.0:
            raise ValueError(f"mass_q must be positive, got {self.mass_q}")

        self.inertia = np.asarray(inertia, dtype=float)
        if self.inertia.shape != (3, 3):
            raise ValueError(f"inertia must be 3×3, got {self.inertia.shape}")

        if link_lengths is None:
            lengths = np.full(n, float(cable_length) / n)
        else:
            lengths = self._broadcast(link_lengths, "link_lengths")
        if link_masses is None:
            masses = np.full(n, float(cable_mass) / n)
            masses[-1] += float(payload_mass)
        else:
            masses = self._broadcast(link_masses, "link_masses")

        if not np.all(lengths > 0.0):
            raise ValueError(f"link_lengths must be strictly positive; got {lengths.tolist()}")
        if not np.all(masses > 0.0):
            raise ValueError(f"link_masses must be strictly positive; got {masses.tolist()}")

        self.link_lengths = lengths
        self.link_masses = masses
        self.gravity = float(gravity)

    def _broadcast(self, value, name: str) -> np.ndarray:
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.num_links, float(arr))
        if arr.shape != (self.num_links,):
            raise ValueError(
                f"{name} must be a scalar or have {self.num_links} entries; got {arr.shape}"
            )
        return arr.astype(float)

    # ─── Derived parameters ────────────────────────────────────────

    @property
    def jet_order(self) -> int:
        """Payload-jet order this map requires: ``2n + 4``.

        Grows with the link count — see the module docstring and
        :func:`~udaan.flatness.chain.chain_jet_order`.
        """
        return chain_jet_order(self.num_links)

    @property
    def cable_length(self) -> float:
        """Total taut cable length ``Σ_i ℓ_i``."""
        return float(np.sum(self.link_lengths))

    @property
    def total_mass(self) -> float:
        """Quadrotor + chain + payload, in kg."""
        return float(self.mass_q + np.sum(self.link_masses))

    @classmethod
    def from_model(cls, model) -> QuadrotorFlexCable:
        """Build from a
        :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase`-like model."""
        return cls(
            num_links=model.num_links,
            mass_q=model.mass,
            inertia=model.inertia,
            link_masses=np.asarray(model.link_masses, dtype=float).copy(),
            link_lengths=np.asarray(model.link_lengths, dtype=float).copy(),
        )

    # ─── Recovery ──────────────────────────────────────────────────

    def chain(self, flats: QuadrotorFlexCableFlats) -> CableChain:
        """Run only the chain cascade, returning the raw
        :class:`~udaan.flatness.chain.CableChain`.

        Exposed because the chain half of the map is worth inspecting (and
        testing) on its own: link directions, tensions and node positions with
        every derivative the recursion produced, before the attitude machinery
        collapses them to a state and a wrench.

        Raises:
            ValueError: if ``x_L`` is shallower than ``2n + 4``, if a link goes
                slack, or if a link leaves the hanging branch (the trajectory
                crossed the slack boundary rather than stopping at it) — see
                :func:`~udaan.flatness.chain.cable_chain_cascade`.
        """
        required = self.jet_order
        if flats.x_L.order < required:
            raise ValueError(
                f"QuadrotorFlexCable.chain: an {self.num_links}-link chain needs "
                f"x_L through its {required}-th derivative (2n + 4); got order "
                f"{flats.x_L.order}."
            )
        return cable_chain_cascade(
            [np.asarray(flats.x_L[k], dtype=float) for k in range(required + 1)],
            self.link_masses,
            self.link_lengths,
            gravity=self.gravity,
        )

    def recover(
        self, flats: QuadrotorFlexCableFlats
    ) -> tuple[QuadrotorFlexCableRefState, QuadrotorFlexCableInputs]:
        g = self.gravity
        mQ = self.mass_q
        J = self.inertia
        n = self.num_links

        # ── Step 1: the chain cascade, payload → quadrotor ─────────────
        # Every link direction, tension and node position, with the depth each
        # rung supports.  Raises on a slack link.
        chain = self.chain(flats)

        # ── Step 2: quadrotor kinematics = node 0, through snap ────────
        x_Q = chain.top_position  # order 4 exactly, by construction

        # ── Step 3: thrust vector.  Newton on the airframe alone,
        #     m_Q (ẍ_Q + g e₃) = f R e₃ + F_0,
        # with F_0 the force the chain pulls on it with (`chain.top_force`).
        # At n = 1, F_0 = -m_L(ẍ_L + g e₃) and this is M2's
        # B = m_Q(ẍ_Q + g e₃) + m_L(ẍ_L + g e₃), term for term.
        top_force = chain.top_force
        B = mQ * (x_Q[2] + g * _E3) - top_force[0]
        dB = mQ * x_Q[3] - top_force[1]
        d2B = mQ * x_Q[4] - top_force[2]

        if float(np.linalg.norm(B)) < 1e-6:
            raise ValueError(
                "QuadrotorFlexCable.recover: net thrust direction is undefined "
                "(m_Q (ẍ_Q + g e₃) − F_0 ≈ 0) — the airframe is in free fall."
            )

        # ── Step 4: (f, R, Ω, Ω̇, M) from (B, ψ) — the shared primitive ──
        f, R, Omega, dOmega, moment = _attitude_from_thrust_vector(
            B,
            dB,
            d2B,
            float(flats.psi.value),
            float(flats.psi.velocity),
            float(flats.psi.acceleration),
            J,
        )

        # ── Step 5: package ────────────────────────────────────────────
        link_attitudes = [chain.directions[i][0] for i in range(n)]
        link_omegas = [chain.angular_velocity(i) for i in range(n)]
        link_domegas = [chain.angular_acceleration(i) for i in range(n)]
        node_positions = np.stack([chain.node_positions[i + 1][0] for i in range(n)])
        node_velocities = np.stack([chain.node_positions[i + 1][1] for i in range(n)])

        ref = QuadrotorFlexCableRefState(
            payload_position=np.asarray(flats.x_L[0], dtype=float),
            payload_velocity=np.asarray(flats.x_L[1], dtype=float),
            payload_acceleration=np.asarray(flats.x_L[2], dtype=float),
            position=x_Q[0],
            velocity=x_Q[1],
            acceleration=x_Q[2],
            link_attitudes=link_attitudes,
            link_angular_velocities=link_omegas,
            link_angular_accelerations=link_domegas,
            link_positions=node_positions,
            link_velocities=node_velocities,
            orientation=SO3(R),
            angular_velocity=TSO3(Omega),
            angular_acceleration=dOmega,
        )
        inputs = QuadrotorFlexCableInputs(
            thrust=f,
            moment=moment,
            tensions=np.array([chain.tensions[i][0] for i in range(n)]),
        )
        return ref, inputs
