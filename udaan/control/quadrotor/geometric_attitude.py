"""Geometric attitude controller on SE(3) for quadrotor.

Taeyoung Lee, Melvin Leok, and N. Harris McClamroch,
"Geometric Tracking Control of a Quadrotor UAV on SE(3)", CDC 2010.
Paper: http://www.math.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf
"""

from __future__ import annotations

import numpy as np

from ...control import Controller, Gains
from ...core.defaults import DEFAULT_ATT_KD, DEFAULT_ATT_KP
from ...core.types import Vec3
from ...manif import SO3, TSO3
from ...utils import hat
from ...utils.logging import get_logger

_logger = get_logger(__name__)


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

    def _cmd_accel_to_cmd_att(self, accel: Vec3) -> SO3:
        """Convert desired acceleration to desired attitude.

        Returns identity for near-zero acceleration to avoid singularity.
        """
        MIN_ACCEL_NORM = 0.001

        norm_accel = np.linalg.norm(accel)
        if norm_accel < MIN_ACCEL_NORM:
            return SO3()
        b3 = accel / norm_accel
        b1d = np.array([1.0, 0.0, 0.0])
        return SO3.from_two_vectors(b3, b1d)

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
                flatness. **Only the last two are used.** ``Rd`` is always
                recomputed from the commanded thrust direction and the supplied
                one is discarded — see below.

        Returns:
            (f, M) — scalar thrust and 3D torque vector.

        Note:
            The reference attitude channel is **inert**, and deliberately so.
            ``Rd`` is what closes the position loop: the outer loop's whole
            output is the *direction* of ``thrust_force``, and
            :meth:`_cmd_accel_to_cmd_att` is the only place that direction
            becomes an attitude command. Substituting the flat map's
            open-loop ``R_i`` for it severs the position feedback from the
            attitude loop, and the closed loop then diverges: measured on the
            brisk spiral of ``tests/test_multi_quad_rigidbody_control.py``,
            payload/position tracking RMS goes from 0.170 m to 4.39 m (M1),
            0.197 m to 4.06 m (M3) and 0.306 m to 4.00 m (M4).

            So the feedforward this method actually applies is ``Ω_d`` and
            ``Ω̇_d``. That is not a small residue — passing ``R_i = I`` and
            keeping the rates gives a *bit-identical* run at every ``nQ``, and
            the rates alone are worth the whole 1.38x M4 tracking improvement
            (``Ω̇_d`` on its own, 0.07%).

            Making ``Rd`` live is a cascade-wide change — the outer loop would
            have to hand over an attitude *and* a thrust magnitude rather than
            a force vector — so it belongs to a structural pass over M1-M4
            together, not to one caller.
        """
        R, Omega = state
        Rd: SO3 = self._cmd_accel_to_cmd_att(thrust_force)
        if desired_att is not None:
            # The reference attitude is dropped on purpose: Rd above is the
            # position loop's only path into the attitude command. See the
            # Note in this method's docstring for the measured cost of using
            # the supplied one instead.
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
