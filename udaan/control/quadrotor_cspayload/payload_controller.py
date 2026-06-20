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
from ...flatness.quadrotor_cspayload import cable_direction_jet
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
        sp = self.setpoint(t)
        sd, dsd, d2sd = sp[0], sp[1], sp[2]
        # Higher payload derivatives (jerk, snap) feed the cable-attitude
        # feedforward below; default to zero when the setpoint omits them
        # (e.g. a stationary set-point), which recovers the prior behaviour.
        d3sd = sp[3] if len(sp) > 3 else np.zeros(3)
        d4sd = sp[4] if len(sp) > 4 else np.zeros(3)

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
        # Desired cable-attitude rates from differential flatness of the
        # payload trajectory: the flat cable direction is
        # q_des = -(a_L + g e3)/‖·‖ with payload-acceleration jet
        # [a_L, ȧ_L, ä_L] = [d2sd, d3sd, d4sd]; dqd, d2qd are its 1st/2nd time
        # derivatives, from the same flat-to-state map as
        # udaan.flatness.quadrotor_cspayload. These supply the cable
        # feedforward that was previously stubbed to zero; q_des coincides with
        # qc in the nominal (feedback-free) case, so qd is left as qc.
        try:
            _, q_flat = cable_direction_jet([d2sd, d3sd, d4sd], mL)
            dqd, d2qd = q_flat[1], q_flat[2]
        except ValueError:
            # Payload near free-fall: cable direction undefined, drop feedforward.
            dqd = np.zeros(3)
            d2qd = np.zeros(3)
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
