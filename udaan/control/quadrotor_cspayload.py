import numpy as np

from ..control import Gains, PDController
from ..utils import hat


class QuadCSPayloadController(PDController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gain_pos = Gains(kp=np.array([4.0, 4.0, 8.0]), kd=np.array([3.0, 3.0, 6.0]))
        self._gain_cable = Gains(kp=np.array([24.0, 24.0, 24.0]), kd=np.array([8.0, 8.0, 8.0]))

        self._quad_mass = 0.9  # kg
        self._quad_inertia = np.array(
            [[0.0023, 0.0, 0.0], [0.0, 0.0023, 0.0], [0.0, 0.0, 0.004]]
        )  # kg m^2
        self._quad_inertia_inv = np.linalg.inv(self._quad_inertia)
        self._payload_mass = 0.2  # kg
        self._cable_length = 1.0  # m

    def compute(self, *args):
        """payload position control"""
        t = args[0]
        s = args[1]  # quadrotor payload state
        sd, dsd, d2sd = self.setpoint(t)
        # TODO add differential flatness

        ex = s.payload_position - sd
        ev = s.payload_velocity - dsd

        q = s.cable_attitude
        dq = s.dq()

        # payload position control
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
        # % calculating errors
        err_q = hat(q) @ hat(q) @ qd
        err_dq = dq - np.cross(np.cross(qd, dqd), q)

        Fpd = -self._gain_cable.kp * err_q - self._gain_cable.kd * err_dq
        Fff = (mQ * l) * np.dot(q, np.cross(qd, dqd)) * np.cross(q, dq) + (
            mQ * l
        ) * np.cross(np.cross(qd, d2qd), q)
        Fn = np.dot(A, q) * q
        thrust_force = -Fpd - Fff + Fn
        return thrust_force
