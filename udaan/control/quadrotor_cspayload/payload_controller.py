"""Payload position + cable attitude controller.

Sreenath, Lee, Kumar (2013) https://ieeexplore.ieee.org/abstract/document/6760219
"""

import numpy as np

from ...control import Gains, PDController
from ...core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_PAYLOAD_CABLE_KD,
    DEFAULT_PAYLOAD_CABLE_KP,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_PAYLOAD_POS_KD,
    DEFAULT_PAYLOAD_POS_KP,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
)
from ...utils import hat


class QuadCSPayloadController(PDController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gain_pos = Gains(kp=DEFAULT_PAYLOAD_POS_KP.copy(), kd=DEFAULT_PAYLOAD_POS_KD.copy())
        self._gain_cable = Gains(
            kp=DEFAULT_PAYLOAD_CABLE_KP.copy(), kd=DEFAULT_PAYLOAD_CABLE_KD.copy()
        )

        self._quad_mass = DEFAULT_QUAD_MASS
        self._quad_inertia = DEFAULT_QUAD_INERTIA.copy()
        self._quad_inertia_inv = np.linalg.inv(self._quad_inertia)
        self._payload_mass = DEFAULT_PAYLOAD_MASS
        self._cable_length = DEFAULT_CABLE_LENGTH

    def compute(self, *args):
        """payload position control"""
        t = args[0]
        s = args[1]  # quadrotor payload state
        sd, dsd, d2sd = self.setpoint(t)

        ex = s.payload_position - sd
        ev = s.payload_velocity - dsd

        q = s.cable_attitude
        dq = s.dq()

        mQ = self._quad_mass
        mL = self._payload_mass
        l = self._cable_length

        Fff = (mQ + mL) * (d2sd + self._ge3) + mQ * l * np.dot(dq, dq) * q
        Fpd = -self._gain_pos.kp * ex - self._gain_pos.kd * ev
        A = Fff + Fpd

        #  desired load attitude
        qc = -A / np.linalg.norm(A)
        # load-attitude ctrl
        qd = qc
        dqd = np.zeros(3)  # TODO update from differential flatness
        d2qd = np.zeros(3)  # TODO update from differential flatness
        # calculating errors
        err_q = hat(q) @ hat(q) @ qd
        err_dq = dq - np.cross(np.cross(qd, dqd), q)

        Fpd = -self._gain_cable.kp * err_q - self._gain_cable.kd * err_dq
        Fff = (mQ * l) * np.dot(q, np.cross(qd, dqd)) * np.cross(q, dq) + (mQ * l) * np.cross(
            np.cross(qd, d2qd), q
        )
        Fn = np.dot(A, q) * q
        thrust_force = -Fpd - Fff + Fn
        return thrust_force
