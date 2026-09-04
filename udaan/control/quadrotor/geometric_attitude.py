"""Geometric attitude controller on SE(3) for quadrotor.

Taeyoung Lee, Melvin Leok, and N. Harris McClamroch,
"Geometric Tracking Control of a Quadrotor UAV on SE(3)", CDC 2010.
Paper: http://www.math.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf
"""

from __future__ import annotations

import numpy as np

from ...control import Controller, Gains
from ...core.defaults import (
    DEFAULT_ATT_KD,
    DEFAULT_ATT_KP,
    MIN_ATTITUDE_HINT_CONDITIONING,
)
from ...core.types import Vec3
from ...manif import SO3, TSO3
from ...utils import hat
from ...utils.logging import get_logger

_logger = get_logger(__name__)

_WORLD_X = np.array([1.0, 0.0, 0.0])

#: Below this the two axes are parallel to working precision and there is no
#: heading left in the hint to blend with.
_HINT_EXHAUSTED = 1e-12

#: ``1 + b₃·b₃_ref`` below this means the commanded thrust is antiparallel to
#: the reference's, where the minimal rotation between them is undefined.
_THRUST_INVERTED = 1e-8


def _swung_heading(b3: Vec3, reference: np.ndarray) -> Vec3 | None:
    r"""The reference's ``b₁`` carried onto the plane orthogonal to ``b₃``.

    Rotates the whole reference frame by the *minimal* rotation taking its own
    ``b₃`` onto the commanded ``b₃`` (Rodrigues, built from the cross product),
    and returns the image of its ``b₁``. The result is orthogonal to ``b₃`` by
    construction, so ``‖b₃ × hint‖ = 1`` exactly — it is the well-conditioned
    heading hint that :data:`~udaan.core.defaults.MIN_ATTITUDE_HINT_CONDITIONING`
    blends toward, and it carries the same heading as the raw ``b₁`` whenever
    the two commands agree at all.

    Undefined only when ``b₃`` is antiparallel to the reference's ``b₃`` — a
    thrust command pointing the opposite way from the reference's, which is a
    tracking failure long before it is a conditioning problem. ``None`` is
    returned there, and the caller falls back to the world ``x`` hint.
    """
    b3_ref = reference[:, 2]
    axis = np.cross(b3_ref, b3)
    cos = float(b3_ref @ b3)
    if 1.0 + cos < _THRUST_INVERTED:
        return None
    K = hat(axis)
    rotation = np.eye(3) + K + K @ K / (1.0 + cos)
    return np.asarray(rotation @ reference[:, 0], dtype=float)


class GeometricAttitudeController(Controller):
    """Geometric tracking control of a quadrotor UAV on SE(3).

    Control law (Lee, Leok, McClamroch 2010, Eq. 18-19):
        M = -kR·eR - kΩ·eΩ + Ω × J·Ω
            - J·(hat(Ω)·Rᵀ·Rd·Ωd - Rᵀ·Rd·dΩd)
    where:
        eR = 0.5·vee(Rdᵀ·R - Rᵀ·Rd)   (attitude error on so(3))
        eΩ = Ω - Rᵀ·Rd·Ωd              (angular velocity error)
        f  = F · R·e3                    (scalar thrust)
    """

    def __init__(self, **kwargs):
        self._gains = Gains(kp=DEFAULT_ATT_KP.copy(), kd=DEFAULT_ATT_KD.copy())

        self._inertia = np.eye(3)
        self._inertia_inv = np.eye(3)

        if "inertia" in kwargs:
            self.inertia = kwargs["inertia"]
        else:
            _logger.error("Inertia not provided")

    @property
    def inertia(self):
        return self._inertia

    @inertia.setter
    def inertia(self, inertia):
        self._inertia = inertia
        self._inertia_inv = np.linalg.inv(inertia)

    def _cmd_accel_to_cmd_att(self, accel: Vec3, reference: SO3 | None = None) -> SO3:
        r"""Convert a commanded acceleration into a commanded attitude.

        The commanded thrust direction ``b₃ = a/‖a‖`` fixes two of the three
        attitude degrees of freedom; the third — the heading of ``b₁`` about
        ``b₃`` — is free, and :meth:`SO3.from_two_vectors` resolves it from a
        *hint*.

        ``reference`` is where the differential-flatness reference attitude
        enters. Its own ``b₁`` becomes the hint, so the recovered attitude
        supplies the heading while ``b₃`` still comes from the position loop's
        force command. That is the composition: the two commands meet on
        orthogonal degrees of freedom rather than one overwriting the other.
        Passing the reference attitude *whole* instead would replace ``b₃`` as
        well, which is what severs position feedback — see :meth:`compute`.

        With no reference the hint falls back to the world ``x`` axis, which
        is heading-hold at ψ = 0.

        **Conditioning.** ``from_two_vectors`` divides by ``‖b₃ × hint‖``, so
        the hint has to stay clear of ``b₃``. The world ``x`` axis is fixed and
        benign; the reference's ``b₁`` is not, because it tilts with the
        reference and lines up with ``b₃`` whenever the commanded thrust
        direction approaches the reference's forward axis — reachable with a
        yawing reference at high tilt. Measured on an M1 spiral at 8 m/s,
        radius 2 m, ψ̇ = 3 rad/s: the raw hint fell to ``‖b₃ × hint‖ = 0.0065``
        and produced a **153°** step-to-step jump in the commanded attitude,
        straight into ``eR``, where the world-``x`` hint's worst was 31°.
        ``from_two_vectors``'s own ``1e-10`` guard is no help — it does not
        fire at 0.0065, and firing is itself a 90° discontinuity.

        Below :data:`~udaan.core.defaults.MIN_ATTITUDE_HINT_CONDITIONING` the
        hint is therefore rotated, in the plane orthogonal to ``b₃``, toward
        :func:`_swung_heading` — the reference frame carried onto ``b₃`` by the
        minimal rotation, which is orthogonal to ``b₃`` by construction. The
        interpolation weight is ``‖b₃ × hint‖`` over the threshold, so the
        blend is continuous at the threshold (weight 1, the raw hint) and
        complete at the singularity (weight 0). Guarded, the same run floors
        conditioning at 0.30 and the worst jump at 31°, with the mean angle to
        the reference attitude unchanged (32.6° → 32.2°).

        Returns identity for near-zero acceleration to avoid the singularity.
        """
        MIN_ACCEL_NORM = 0.001

        norm_accel = np.linalg.norm(accel)
        if norm_accel < MIN_ACCEL_NORM:
            return SO3()
        b3 = accel / norm_accel
        if reference is None:
            return SO3.from_two_vectors(b3, _WORLD_X)
        return SO3.from_two_vectors(b3, self._heading_hint(b3, np.asarray(reference)))

    @staticmethod
    def _heading_hint(b3: Vec3, reference: np.ndarray) -> Vec3:
        """The reference's ``b₁``, guarded against near-alignment with ``b₃``.

        See :meth:`_cmd_accel_to_cmd_att` for why, and
        :data:`~udaan.core.defaults.MIN_ATTITUDE_HINT_CONDITIONING` for the
        threshold.
        """
        hint = reference[:, 0]
        projected = hint - float(b3 @ hint) * b3
        conditioning = float(np.linalg.norm(projected))  # == ‖b₃ × hint‖
        if conditioning >= MIN_ATTITUDE_HINT_CONDITIONING:
            return hint

        swung = _swung_heading(b3, reference)
        if swung is None:
            return _WORLD_X
        if conditioning < _HINT_EXHAUSTED:
            return swung
        # Rotate `swung` toward the raw hint's direction, about b3, by the
        # fraction of the threshold the conditioning has left. Both vectors lie
        # in the plane orthogonal to b3, so the result is a unit vector there
        # and its own conditioning is exactly 1.
        target = projected / conditioning
        angle = (conditioning / MIN_ATTITUDE_HINT_CONDITIONING) * np.arctan2(
            float(b3 @ np.cross(swung, target)), float(swung @ target)
        )
        blended = np.cos(angle) * swung + np.sin(angle) * np.cross(b3, swung)
        return np.asarray(blended, dtype=float)

    def compute(
        self,
        t: float,
        state: tuple[SO3, TSO3],
        thrust_force: Vec3,
        desired_att: tuple[SO3, TSO3, Vec3] | None = None,
    ) -> tuple[float, Vec3]:
        """Compute scalar thrust and torque vector.

        Args:
            state: (R, Omega) — rotation matrix and body angular velocity.
            thrust_force: desired thrust force vector in world frame.
            desired_att: optional ``(Rd, Omegad, dOmegad)`` from differential
                flatness. All three are used, but ``Rd`` contributes only its
                *heading* — see below.

        Returns:
            (f, M) — scalar thrust and 3D torque vector.

        Note:
            The reference attitude **composes with** the thrust-derived one
            rather than replacing it, and the split is by degree of freedom.
            ``thrust_force`` is the position loop's entire output, and its
            direction fixes ``b₃``; the reference attitude then resolves the
            one remaining degree of freedom, the heading of ``b₁`` about
            ``b₃``, through :meth:`_cmd_accel_to_cmd_att`'s hint.

            Replacing ``Rd`` outright — the obvious reading of "apply the
            reference attitude" — overwrites ``b₃`` too, which severs the
            position loop from the attitude loop. Measured on the brisk spiral
            of ``tests/test_multi_quad_rigidbody_control.py``, whole-``Rd``
            substitution takes tracking RMS from 0.170 m to 4.39 m (M1),
            0.197 m to 4.06 m (M3) and 0.306 m to 4.00 m (M4). Composing on
            the heading alone costs none of that.

            On a ψ ≡ 0 reference the composition is close to a no-op, because
            the hint it supplies is then near the ``[1, 0, 0]`` the fallback
            uses anyway: brisk-spiral RMS 0.170003 → 0.170003 (M1),
            0.197249 → 0.197094 (M3), 0.306325 → 0.305917 (M4). It earns its
            place when the reference actually yaws. At ψ̇ = 1 rad/s on the same
            spiral, the mean angle between the flown and reference attitudes
            drops from 94.3° to 4.05° and position RMS from 0.2886 m to
            0.1698 m — without it the heading command is simply unreachable,
            and the ``Ω_d``/``Ω̇_d`` feedforward computed for that heading
            fights the loop instead of helping it.

            A yawing reference is reached through the models' own
            :attr:`~udaan.models.quadrotor.QuadrotorBase.yaw_setpoint`, which
            is what the flat map's ψ flat output is bound to; unbound it is
            ψ ≡ 0 and the composition stays the near no-op above. Flown at
            ψ̇ = 3 rad/s on the brisk spiral, the mean heading error is 1.4°
            against 91.1° with the reference discarded, and position RMS
            0.165 m against 0.286 m.

            **What it costs.** The hint is no longer a fixed world axis, so
            its conditioning is no longer fixed either: ``R_d`` is built by
            dividing by ``‖b₃ × hint‖``, and the reference's ``b₁`` tilts with
            the reference and can swing onto ``b₃``. Left unguarded that is a
            152° jump in ``R_d`` in one 5 ms step, straight into ``eR`` — worse
            than anything the fixed hint produced. It is guarded, at
            :data:`~udaan.core.defaults.MIN_ATTITUDE_HINT_CONDITIONING`; see
            :meth:`_cmd_accel_to_cmd_att` for the construction and the measured
            numbers on both sides of it.
        """
        R, Omega = state
        Rd: SO3 = self._cmd_accel_to_cmd_att(
            thrust_force, reference=None if desired_att is None else desired_att[0]
        )
        if desired_att is not None:
            _, Omegad, dOmegad = desired_att
        else:
            Omegad: TSO3 = TSO3()
            dOmegad: Vec3 = np.zeros(3)

        eR: TSO3 = R - Rd
        eOm: TSO3 = Omega - Omegad.transport(Rd, R)

        RtRd = R.T @ Rd
        M = (
            -self._gains.kp * eR.vector
            - self._gains.kd * eOm.vector
            + np.cross(Omega, self.inertia @ Omega)
        )
        M += -self.inertia @ (hat(Omega) @ RtRd @ Omegad - RtRd @ dOmegad)
        f = thrust_force.dot(R[:, 2])
        return f, M
