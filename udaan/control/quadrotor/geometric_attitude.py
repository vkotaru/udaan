"""Geometric attitude controller on SE(3) for quadrotor.

Lee, Leok, McClamroch (2010) https://ieeexplore.ieee.org/document/5717652
"""

import numpy as np

from ...control import Controller, Gains
from ...utils import hat, printc_fail


class GeometricAttitudeController(Controller):
    """Geometric tracking control of a quadrotor UAV on SE(3)."""

    def __init__(self, **kwargs):
        self._gains = Gains(kp=np.array([2.4, 2.4, 1.35]), kd=np.array([0.35, 0.35, 0.225]))

        self._inertia = np.eye(3)
        self._inertia_inv = np.eye(3)
        self.mass = 1.0

        if "inertia" in kwargs:
            self.inertia = kwargs["inertia"]
        else:
            printc_fail("Inertia not provided")
        if "mass" in kwargs:
            self.mass = kwargs["mass"]

    @property
    def inertia(self):
        return self._inertia

    @inertia.setter
    def inertia(self, inertia):
        self._inertia = inertia
        self._inertia_inv = np.linalg.inv(inertia)

    def _cmd_accel_to_cmd_att(self, accel):
        """Convert desired acceleration to desired attitude.

        Returns identity for near-zero acceleration to avoid singularity.
        """
        MIN_ACCEL_NORM = 0.001

        norm_accel = np.linalg.norm(accel)
        if norm_accel < MIN_ACCEL_NORM:
            return np.eye(3)

        b3 = accel / norm_accel

        b1d = np.array([1.0, 0.0, 0.0])
        b3_b1d = np.cross(b3, b1d)
        norm_b3_b1d = np.linalg.norm(b3_b1d)

        if norm_b3_b1d < 1e-6:
            b1d = np.array([0.0, 1.0, 0.0])
            b3_b1d = np.cross(b3, b1d)
            norm_b3_b1d = np.linalg.norm(b3_b1d)

        b1 = (-1 / norm_b3_b1d) * np.cross(b3, b3_b1d)
        b2 = np.cross(b3, b1)

        Rd = np.column_stack([b1, b2, b3])
        return Rd

    def compute(self, *args):
        t = args[0]
        R, Omega = args[1][0], args[1][1]
        thrust_force = args[2]
        Rd = self._cmd_accel_to_cmd_att(thrust_force)
        Omegad = np.zeros(3)
        dOmegad = np.zeros(3)

        tmp = 0.5 * (Rd.T @ R - R.T @ Rd)
        eR = np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])  # vee-map
        eOmega = Omega - R.T @ Rd @ Omegad
        M = -self._gains.kp * eR - self._gains.kd * eOmega + np.cross(Omega, self.inertia @ Omega)
        M += -1 * self.inertia @ (hat(Omega) @ R.T @ Rd @ Omegad - R.T @ Rd @ dOmegad)
        f = thrust_force.dot(R[:, 2])
        return f, M
