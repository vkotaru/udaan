"""Differential flatness for a quadrotor with a cable-suspended point-mass payload.

Flat output:
    y = (x_L, ψ)
where x_L ∈ R^3 is the payload position and ψ ∈ R is the quadrotor yaw.
Derivative orders required:
    x_L through pop          (x_L, ẋ_L, ẍ_L, x_L^(3), x_L^(4), x_L^(5), x_L^(6))
    ψ   through ψ̈           (ψ, ψ̇, ψ̈)
The map recovers the full state (x_L, ẋ_L, q, ω, x_Q, ẋ_Q, R, Ω) plus the
feedforward thrust and moment (f, M).

The cable unit vector ``q`` points from the quadrotor to the payload —
codebase convention; matches ``QuadrotorCsPayloadState.cable_attitude``,
whose default ``[0, 0, -1]`` places the payload directly below the
quadrotor at rest.

Reference:
    K. Sreenath, N. Michael, V. Kumar, "Trajectory Generation and
    Control of a Quadrotor with a Cable-Suspended Load — A
    Differentially-Flat Hybrid System", IEEE ICRA, 2013.
    (Companion: K. Sreenath, T. Lee, V. Kumar, "Geometric Control and
    Differential Flatness of a Quadrotor UAV with a Cable-Suspended
    Load", 52nd IEEE CDC, 2013.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb

import numpy as np

from ..core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from ..core.types import Mat3, Vec3
from ..manif import S2, SO3, TS2, TSO3
from .base import Flat2State
from .jet import Jet
from .quadrotor import _attitude_from_thrust_vector

_E3 = np.array([0.0, 0.0, 1.0])


def _zeros3() -> Vec3:
    return np.zeros(3)


def _normalize_derivatives(v: list[np.ndarray]) -> tuple[list[float], list[np.ndarray]]:
    """Given the time-derivative tuple ``v = [v, v̇, v̈, …, v^(K)]`` of a
    non-vanishing vector signal, return ``(n, u)`` where ``n[k] = (d/dt)^k ‖v‖``
    and ``u[k] = (d/dt)^k (v / ‖v‖)`` for k = 0, 1, …, K.

    Uses the Leibniz expansions
        (n²)^(k) = sum_{j=0}^{k} C(k,j) v^(j) · v^(k-j),
        u^(k)   = (v^(k) - sum_{j=1}^{k} C(k,j) n^(j) u^(k-j)) / n,
    and the closed-form recursion for n^(k) by differentiating ``n² = n·n``.

    Raises ``ValueError`` on a vanishing v (free-fall singularity in the
    flatness map).
    """
    K = len(v) - 1
    n_val = float(np.linalg.norm(v[0]))
    if n_val < 1e-9:
        raise ValueError(
            "_normalize_derivatives: input vector has zero norm — singularity in the flatness map."
        )

    # n²[k] := (d/dt)^k (‖v‖²)
    n_sq = []
    for k in range(K + 1):
        s = np.zeros_like(v[0], dtype=float)[..., 0] if False else 0.0
        s = 0.0
        for j in range(k + 1):
            s += comb(k, j) * float(v[j].dot(v[k - j]))
        n_sq.append(s)

    # n[k] from the Leibniz expansion of (n·n) = n²:
    #   (n²)^(k) = sum_{j=0}^{k} C(k,j) n^(j) n^(k-j)
    #   2 n n^(k) = (n²)^(k) - sum_{j=1}^{k-1} C(k,j) n^(j) n^(k-j)
    n = [n_val]
    for k in range(1, K + 1):
        s = n_sq[k]
        for j in range(1, k):
            s -= comb(k, j) * n[j] * n[k - j]
        n.append(s / (2.0 * n_val))

    # u[k] from u·n = v:
    #   v^(k) = sum_{j=0}^{k} C(k,j) u^(j) n^(k-j)
    #   u^(k) = (v^(k) - sum_{j=1}^{k} C(k,j) n^(j) u^(k-j)) / n
    u = [v[0] / n_val]
    for k in range(1, K + 1):
        s = v[k].copy()
        for j in range(1, k + 1):
            s -= comb(k, j) * n[j] * u[k - j]
        u.append(s / n_val)

    return n, u


@dataclass
class QuadrotorCsPayloadFlats:
    """Flat output of a quadrotor with cable-suspended payload: payload-position
    jet (order ≥ 6) and yaw-angle jet (order ≥ 2).

    Attributes:
        x_L:  dim-3 :class:`Jet` of the payload position, with derivatives
              through pop (order ≥ 6) — ``q^(4)`` and hence the moment
              feedforward consume the 6th derivative of ``x_L``.
        psi:  scalar :class:`Jet` of the yaw angle, with derivatives through
              yaw acceleration (order ≥ 2).
    """

    x_L: Jet  # dim 3, order >= 6 (through pop)
    psi: Jet  # dim 1 scalar, order >= 2 (through ψ̈)

    def __post_init__(self):
        if self.x_L.dim != 3:
            raise ValueError(f"x_L must be a 3-D jet; got dim={self.x_L.dim}")
        if self.x_L.order < 6:
            raise ValueError(f"x_L.order must be >= 6 (pop); got {self.x_L.order}")
        if not self.psi.is_scalar:
            raise ValueError(f"psi must be a scalar jet; got dim={self.psi.dim}")
        if self.psi.order < 2:
            raise ValueError(f"psi.order must be >= 2; got {self.psi.order}")


@dataclass
class QuadrotorCsPayloadRefState:
    """Recovered kinematic reference state for a quadrotor with cable-suspended
    payload.

    Field naming mirrors :class:`udaan.models.quadrotor_cspayload.base.QuadrotorCsPayloadState`
    where the two overlap; ``acceleration``, ``payload_acceleration``, and
    ``angular_acceleration`` are extras that the flatness map also produces.
    """

    # Payload kinematics
    payload_position: Vec3 = field(default_factory=_zeros3)
    payload_velocity: Vec3 = field(default_factory=_zeros3)
    payload_acceleration: Vec3 = field(default_factory=_zeros3)

    # Quadrotor centre-of-mass kinematics
    position: Vec3 = field(default_factory=_zeros3)
    velocity: Vec3 = field(default_factory=_zeros3)
    acceleration: Vec3 = field(default_factory=_zeros3)

    # Cable kinematics — q is a unit 3-vector, ω its angular velocity on T_q S^2
    cable_attitude: Vec3 = field(default_factory=lambda: np.array([0.0, 0.0, -1.0]))
    cable_angular_velocity: Vec3 = field(default_factory=_zeros3)
    cable_angular_acceleration: Vec3 = field(default_factory=_zeros3)

    # Quadrotor attitude
    orientation: SO3 = field(default_factory=SO3)
    angular_velocity: TSO3 = field(default_factory=TSO3)
    angular_acceleration: Vec3 = field(default_factory=_zeros3)

    def as_state(self):
        """Project onto the model's ``QuadrotorCsPayloadState`` shape (drops
        all *_acceleration extras; wraps cable fields as :class:`S2`/:class:`TS2`)."""
        from ..models.quadrotor_cspayload.base import QuadrotorCsPayloadState

        return QuadrotorCsPayloadState(
            position=self.position,
            velocity=self.velocity,
            orientation=self.orientation,
            angular_velocity=self.angular_velocity,
            payload_position=self.payload_position,
            payload_velocity=self.payload_velocity,
            cable_attitude=S2(np.asarray(self.cable_attitude)),
            cable_angular_velocity=TS2(np.asarray(self.cable_angular_velocity)),
        )


@dataclass
class QuadrotorCsPayloadInputs:
    """Feedforward inputs recovered from the flat output.

    Attributes:
        thrust:   collective thrust magnitude ``f``.
        moment:   body-frame moment ``M``.
        tension:  cable tension ``T = m_L ‖ẍ_L + g e_3‖``. Falls out of the
                  Newton–Euler form of Step 1 in the recovery; useful as a
                  slack-cable diagnostic (``T → 0`` marks the hybrid-mode
                  boundary at which the cable goes slack).
    """

    thrust: float = 0.0
    moment: Vec3 = field(default_factory=_zeros3)
    tension: float = 0.0


class QuadrotorCsPayload(Flat2State):
    """Differential-flatness recovery for a quadrotor with a cable-suspended,
    point-mass payload (taut cable, single rigid-body quadrotor)."""

    Flats = QuadrotorCsPayloadFlats
    RefState = QuadrotorCsPayloadRefState
    Inputs = QuadrotorCsPayloadInputs

    def __init__(
        self,
        *,
        mass_q: float = DEFAULT_QUAD_MASS,
        mass_l: float = DEFAULT_PAYLOAD_MASS,
        inertia: Mat3 = DEFAULT_QUAD_INERTIA,
        cable_length: float = DEFAULT_CABLE_LENGTH,
        gravity: float = GRAVITY,
    ):
        self.mass_q = float(mass_q)
        self.mass_l = float(mass_l)
        self.inertia = np.asarray(inertia, dtype=float)
        if self.inertia.shape != (3, 3):
            raise ValueError(f"inertia must be 3×3, got {self.inertia.shape}")
        self.cable_length = float(cable_length)
        if self.cable_length <= 0.0:
            raise ValueError(f"cable_length must be positive, got {self.cable_length}")
        self.gravity = float(gravity)

    @classmethod
    def from_model(cls, model) -> QuadrotorCsPayload:
        """Build from a :class:`udaan.models.quadrotor_cspayload.QuadrotorCsPayloadBase`-like model."""
        return cls(
            mass_q=model.mass,
            mass_l=model.payload_mass,
            inertia=model.inertia,
            cable_length=model.cable_length,
        )

    def recover(
        self, flats: QuadrotorCsPayloadFlats
    ) -> tuple[QuadrotorCsPayloadRefState, QuadrotorCsPayloadInputs]:
        g = self.gravity
        mQ = self.mass_q
        mL = self.mass_l
        L = self.cable_length
        J = self.inertia

        # ── Payload-position derivatives, x_L through pop ──────────────
        x_L = [flats.x_L[k] for k in range(7)]  # rows 0..6

        # ── Newton–Euler on the payload alone: m_L ẍ_L = -T q - m_L g e_3.
        # Build the tension vector  T_vec := T q = -m_L (ẍ_L + g e_3)
        # (doc symbol: bold T; see eq-payload-Tvec). Then
        #   T = ‖T_vec‖   (scalar cable tension; slack-cable diagnostic),
        #   q = T_vec / T (unit cable direction, quadrotor → payload).
        # Derivative chain: T_vec^(k) = -m_L x_L^(k+2) for k >= 1.
        Tvec = [-mL * (x_L[2] + g * _E3)]
        for k in range(1, 5):  # k = 1..4 → uses x_L^(3..6)
            Tvec.append(-mL * x_L[k + 2])

        if float(np.linalg.norm(Tvec[0])) < 1e-6:
            # Payload free-fall: cable goes slack, q is undefined.
            raise ValueError(
                "QuadrotorCsPayload.recover: payload is in free fall "
                "(ẍ_L + g e_3 ≈ 0) — cable would go slack, q undefined."
            )

        # ── q = T_vec / T and its derivatives through q^(4) ────────────
        T_derivs, q_d = _normalize_derivatives(Tvec)
        tension = T_derivs[0]
        # q_d[k] is the k-th time derivative of q.
        q = q_d[0]
        qdot = q_d[1]
        qddot = q_d[2]
        qd3 = q_d[3]
        qd4 = q_d[4]

        # ── Quadrotor position and its derivatives through snap ────────
        # x_Q^(k) = x_L^(k) - L q^(k), for k = 0..4.
        xQ = x_L[0] - L * q
        vQ = x_L[1] - L * qdot
        aQ = x_L[2] - L * qddot
        jQ = x_L[3] - L * qd3
        sQ = x_L[4] - L * qd4

        # ── Cable angular velocity and acceleration on T_q S² ──────────
        # ω = q × q̇,  ω̇ = q × q̈   (since q × (ω × q) = ω).
        omega_cable = np.cross(q, qdot)
        domega_cable = np.cross(q, qddot)

        # ── Total thrust vector: B = f R e_3 = m_Q (ẍ_Q + g e_3) + m_L (ẍ_L + g e_3).
        # Sum of the quadrotor and payload Newton's laws (the ±𝐓 cable
        # reactions cancel); equivalently, B = m_Q (ẍ_Q + g e_3) - 𝐓.
        B = mQ * (aQ + g * _E3) + mL * (x_L[2] + g * _E3)
        dB = mQ * jQ + mL * x_L[3]
        d2B = mQ * sQ + mL * x_L[4]

        if float(np.linalg.norm(B)) < 1e-6:
            # Quadrotor free-fall — no unique thrust direction.
            raise ValueError(
                "QuadrotorCsPayload.recover: net thrust direction is undefined "
                "(m_Q (ẍ_Q + g e_3) + m_L (ẍ_L + g e_3) ≈ 0)."
            )

        # ── R, Ω, dΩ, M from (B, ψ) — same machinery as the standalone
        # quadrotor map; see Steps 2–4 of `quadrotor.md` for the derivation.
        f, R, Omega, dOmega, moment = _attitude_from_thrust_vector(
            B,
            dB,
            d2B,
            flats.psi.value,
            flats.psi.velocity,
            flats.psi.acceleration,
            J,
        )

        ref = QuadrotorCsPayloadRefState(
            payload_position=x_L[0],
            payload_velocity=x_L[1],
            payload_acceleration=x_L[2],
            position=xQ,
            velocity=vQ,
            acceleration=aQ,
            cable_attitude=q,
            cable_angular_velocity=omega_cable,
            cable_angular_acceleration=domega_cable,
            orientation=SO3(R),
            angular_velocity=TSO3(Omega),
            angular_acceleration=dOmega,
        )
        inputs = QuadrotorCsPayloadInputs(thrust=f, moment=moment, tension=tension)
        return ref, inputs
