"""Differential flatness for the single rigid-body quadrotor.

Flat output:
    y = (x_Q, ψ)
where x_Q ∈ R^3 is the centre-of-mass position and ψ ∈ R is the yaw angle.
Derivative orders required:
    x_Q through snap (x, ẋ, ẍ, x⃛, x⁽⁴⁾)
    ψ through ψ̈
The map recovers the full state (x_Q, v_Q, R, Ω) plus the feedforward
thrust and moment (f, M).

Flat outputs are carried as :class:`Jet` bundles — ``x`` is a dim-3 jet
of order ≥ 4 (through snap); ``psi`` is a scalar jet of order ≥ 2.

Reference:
    D. Mellinger and V. Kumar, "Minimum Snap Trajectory Generation and
    Control for Quadrotors", ICRA 2011.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.defaults import GRAVITY
from ..core.types import Mat3, Vec3
from ..manif import SO3, TSO3, vee
from .base import Flat2State
from .jet import Jet

_E3 = np.array([0.0, 0.0, 1.0])


def _zeros3() -> Vec3:
    return np.zeros(3)


def _skew(m: np.ndarray) -> np.ndarray:
    """Skew-symmetric part of a 3×3 matrix — ensures vee() is well-defined
    even under floating-point noise."""
    return 0.5 * (m - m.T)


def _attitude_from_thrust_vector(
    B: np.ndarray,
    dB: np.ndarray,
    d2B: np.ndarray,
    psi: float,
    dpsi: float,
    d2psi: float,
    J: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recover ``(f, R, Ω, dΩ, M)`` from the thrust-vector chain ``(B, dB, d2B)``
    and yaw chain ``(ψ, ψ̇, ψ̈)``.

    Used by every flat-to-state map whose thrust acts along the body z-axis:
    once the caller has assembled ``B = f R e_3`` (in the form
    ``m(ẍ + g e_3)`` for the standalone quadrotor, or
    ``m_Q(ẍ_Q + g e_3) - 𝐓`` for the cspayload), the b_3-projector chain,
    yaw-hint frame construction, skew-projector body rates, and Newton–Euler
    moment are all the same.

    Args:
        B:     thrust vector ``f R e_3`` (R^3); must be non-vanishing.
        dB:    first time derivative of B (R^3).
        d2B:   second time derivative of B (R^3).
        psi:   yaw angle.
        dpsi:  yaw rate.
        d2psi: yaw acceleration.
        J:     quadrotor inertia matrix (3×3).

    Returns:
        ``(f, R, Ω, dΩ, M)`` where ``f = ‖B‖``, ``R ∈ SO(3)`` (3×3 ndarray),
        ``Ω, dΩ ∈ R^3`` are body-frame rate / angular acceleration, and
        ``M = J dΩ + Ω × J Ω`` is the body-frame moment.

    Raises:
        ValueError: if ``b_3 := B/‖B‖`` is parallel to the yaw-hint
            ``[cos ψ, sin ψ, 0]ᵀ`` (yaw-aligned singularity). The caller
            is responsible for the ``‖B‖ ≈ 0`` (free-fall) check.
    """
    norm_B = float(np.linalg.norm(B))
    f = norm_B
    b3 = B / norm_B

    # Yaw-frame heading vector and its first two derivatives
    cpsi, spsi = np.cos(psi), np.sin(psi)
    b1d = np.array([cpsi, spsi, 0.0])
    db1d = np.array([-spsi, cpsi, 0.0]) * dpsi
    d2b1d = np.array([-cpsi, -spsi, 0.0]) * dpsi**2 + np.array([-spsi, cpsi, 0.0]) * d2psi

    # Derivatives of ‖B‖ via d/dt(‖v‖) = v·v̇/‖v‖
    dnorm_B = B.dot(dB) / norm_B
    d2norm_B = (dB.dot(dB) + B.dot(d2B) - dnorm_B**2) / norm_B

    # Derivatives of b_3 = B/‖B‖ (projector form, automatically perpendicular)
    db3 = (dB - b3 * dnorm_B) / norm_B
    d2b3 = (d2B - 2.0 * db3 * dnorm_B - b3 * d2norm_B) / norm_B

    # Build R = [b1 b2 b3] with yaw-hint b1d
    b3_x_b1d = np.cross(b3, b1d)
    norm_b3_x_b1d = float(np.linalg.norm(b3_x_b1d))
    if norm_b3_x_b1d < 1e-6:
        raise ValueError(
            "Thrust direction aligned with yaw hint — cannot construct a "
            "unique desired attitude. Offset the yaw."
        )
    b2 = b3_x_b1d / norm_b3_x_b1d
    b1 = np.cross(b2, b3)
    R = np.column_stack([b1, b2, b3])

    # First derivatives of {b1, b2} for Ω
    db3_x_b1d = np.cross(db3, b1d) + np.cross(b3, db1d)
    dnorm_b3_x_b1d = b3_x_b1d.dot(db3_x_b1d) / norm_b3_x_b1d
    db2 = (db3_x_b1d - b2 * dnorm_b3_x_b1d) / norm_b3_x_b1d
    db1 = np.cross(db2, b3) + np.cross(b2, db3)
    dR = np.column_stack([db1, db2, db3])

    # Body-frame angular velocity: Ω̂ = Rᵀ Ṙ
    Omega = vee(_skew(R.T @ dR))

    # Second derivatives of {b1, b2} for dΩ
    d2b3_x_b1d = np.cross(d2b3, b1d) + 2.0 * np.cross(db3, db1d) + np.cross(b3, d2b1d)
    d2norm_b3_x_b1d = (
        db3_x_b1d.dot(db3_x_b1d) + b3_x_b1d.dot(d2b3_x_b1d) - dnorm_b3_x_b1d**2
    ) / norm_b3_x_b1d
    d2b2 = (d2b3_x_b1d - 2.0 * db2 * dnorm_b3_x_b1d - b2 * d2norm_b3_x_b1d) / norm_b3_x_b1d
    d2b1 = np.cross(d2b2, b3) + 2.0 * np.cross(db2, db3) + np.cross(b2, d2b3)
    d2R = np.column_stack([d2b1, d2b2, d2b3])

    # dΩ̂ = skew(Rᵀ R̈) — symmetric Ω̂² piece cancels under the projector
    dOmega = vee(_skew(R.T @ d2R))

    # Feedforward moment: M = J dΩ + Ω × J Ω
    M = J @ dOmega + np.cross(Omega, J @ Omega)

    return f, R, Omega, dOmega, M


@dataclass
class QuadrotorFlats:
    """Flat output of a rigid-body quadrotor: position jet (order ≥ 4)
    and yaw-angle jet (order ≥ 2).

    Attributes:
        x:    dim-3 :class:`Jet` of the centre-of-mass position, with
              derivatives through snap (order ≥ 4).
        psi:  scalar :class:`Jet` of the yaw angle, with derivatives
              through yaw acceleration (order ≥ 2).
    """

    x: Jet  # dim 3, order >= 4 (through snap)
    psi: Jet  # dim 1 scalar, order >= 2 (through d2ψ̈)

    def __post_init__(self):
        if self.x.dim != 3:
            raise ValueError(f"x must be a 3-D jet; got dim={self.x.dim}")
        if self.x.order < 4:
            raise ValueError(f"x.order must be >= 4 (snap); got {self.x.order}")
        if not self.psi.is_scalar:
            raise ValueError(f"psi must be a scalar jet; got dim={self.psi.dim}")
        if self.psi.order < 2:
            raise ValueError(f"psi.order must be >= 2; got {self.psi.order}")


@dataclass
class QuadrotorRefState:
    """Recovered kinematic reference state for a quadrotor.

    The first four fields mirror :class:`udaan.models.quadrotor.base.QuadrotorState`;
    ``acceleration`` and ``angular_acceleration`` are extras that the
    flatness map also makes available.
    """

    position: Vec3 = field(default_factory=_zeros3)
    velocity: Vec3 = field(default_factory=_zeros3)
    acceleration: Vec3 = field(default_factory=_zeros3)
    orientation: SO3 = field(default_factory=SO3)
    angular_velocity: TSO3 = field(default_factory=TSO3)
    angular_acceleration: Vec3 = field(default_factory=_zeros3)

    def as_state(self):
        """Project onto the model's ``QuadrotorState`` shape (drops
        acceleration + angular_acceleration)."""
        from ..models.quadrotor.base import QuadrotorState

        return QuadrotorState(
            position=self.position,
            velocity=self.velocity,
            orientation=self.orientation,
            angular_velocity=self.angular_velocity,
        )


@dataclass
class QuadrotorInputs:
    """Feedforward inputs recovered from the flat output."""

    thrust: float = 0.0
    moment: Vec3 = field(default_factory=_zeros3)


class Quadrotor(Flat2State):
    """Differential-flatness recovery for the single rigid-body quadrotor."""

    Flats = QuadrotorFlats
    RefState = QuadrotorRefState
    Inputs = QuadrotorInputs

    def __init__(self, mass: float, inertia: Mat3):
        self.mass = float(mass)
        self.inertia = np.asarray(inertia, dtype=float)
        if self.inertia.shape != (3, 3):
            raise ValueError(f"inertia must be 3×3, got {self.inertia.shape}")

    @classmethod
    def from_model(cls, model) -> Quadrotor:
        """Build from a :class:`udaan.models.quadrotor.QuadrotorBase`-like model."""
        return cls(mass=model.mass, inertia=model.inertia)

    def recover(self, flats: QuadrotorFlats) -> tuple[QuadrotorRefState, QuadrotorInputs]:
        g = GRAVITY
        m = self.mass
        J = self.inertia

        # Pull named derivatives off the jets — position through snap,
        # yaw through yaw-acceleration.
        pos = flats.x.value
        vel = flats.x.velocity
        acc = flats.x.acceleration
        jerk = flats.x.jerk
        snap = flats.x.snap
        psi = flats.psi.value
        dpsi = flats.psi.velocity
        d2psi = flats.psi.acceleration

        # Thrust vector  B = f R e_3 = m (ẍ + g e_3)  and its time derivatives.
        B = m * (acc + g * _E3)
        dB = m * jerk
        d2B = m * snap

        if float(np.linalg.norm(B)) < 1e-6:
            # Free-fall singularity — no unique thrust direction. Return a
            # degraded reference (kinematics only, zero feedforward).
            ref = QuadrotorRefState(position=pos, velocity=vel, acceleration=acc)
            return ref, QuadrotorInputs()

        f, R, Omega, dOmega, moment = _attitude_from_thrust_vector(B, dB, d2B, psi, dpsi, d2psi, J)

        ref = QuadrotorRefState(
            position=pos,
            velocity=vel,
            acceleration=acc,
            orientation=SO3(R),
            angular_velocity=TSO3(Omega),
            angular_acceleration=dOmega,
        )
        inputs = QuadrotorInputs(thrust=f, moment=moment)
        return ref, inputs
