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
        M = -kR * eR - kOm * eOm + Om x J @ Om
            - J @ (hat(Om) @ R.T @ Rd @ Omd - R.T @ Rd @ dOmd)
    where:
        eR   = 0.5 * vee(Rd.T @ R - R.T @ Rd)   (attitude error on so(3))
        eOm  = Om - R.T @ Rd @ Omd               (angular velocity error)
        f    = F . R @ e3                          (scalar thrust)
    """

    def __init__(self, **kwargs):
        self._gains = Gains(kp=DEFAULT_ATT_KP.copy(), kd=DEFAULT_ATT_KD.copy())

        self._inertia = np.eye(3)
        self._inertia_inv = np.eye(3)
        self.mass = 1.0

        if "inertia" in kwargs:
            self.inertia = kwargs["inertia"]
        else:
            _logger.error("Inertia not provided")
        if "mass" in kwargs:
            self.mass = kwargs["mass"]

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

    def compute(self, t: float, state: tuple[SO3, TSO3], thrust_force: Vec3) -> tuple[float, Vec3]:
        """Compute scalar thrust and torque vector.

        Args:
            state: (R, Omega) — rotation matrix and body angular velocity.
            thrust_force: desired thrust force vector in world frame.

        Returns:
            (f, M) — scalar thrust and 3D torque vector.
        """
        R, Omega = state
        Rd: SO3 = self._cmd_accel_to_cmd_att(thrust_force)
        Omegad: TSO3 = TSO3()
        dOmegad: Vec3 = np.zeros(3)

        eR: TSO3 = R - Rd
        eOmega: TSO3 = Omega - Omegad.transport(Rd, R)

        RtRd = R.T @ Rd
        M = (
            -self._gains.kp * eR.vector
            - self._gains.kd * eOmega.vector
            + np.cross(Omega, self.inertia @ Omega)
        )
        M += -self.inertia @ (hat(Omega) @ RtRd @ Omegad - RtRd @ dOmegad)
        f = thrust_force.dot(R[:, 2])
        return f, M
