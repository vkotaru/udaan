from ..control import Gains, Controller, PDController
import numpy as np
from ..utils import printc_fail, hat, vee


class QuadCSPayloadController(PDController):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gain_pos = Gains(kp=np.array([4., 4., 8.]),
                               kd=np.array([3., 3., 6.]))
        self._gain_cable = Gains(kp=np.array([24., 24., 24.]),
                                 kd=np.array([8., 8., 8.]))

        self.mQ = 0.9  # kg
        self.JQ = np.array([[0.0023, 0., 0.], [0., 0.0023, 0.],
                            [0., 0., 0.004]])  # kg m^2
        self.JQinv = np.linalg.inv(self.JQ)
        self.mL = 0.2  # kg
        self.l = 1.0  # m

        return

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
        Fff = (self.mQ + self.mL) * (
            d2sd + self._ge3) + self.mQ * self.l * np.dot(dq, dq) * q
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
        Fff = (self.mQ*self.l)*np.dot(q, np.cross(qd,dqd))*np.cross(q,dq)+\
                    (self.mQ*self.l)*np.cross(np.cross(qd, d2qd), q)
        Fn = np.dot(A, q) * q
        thrust_force = -Fpd - Fff + Fn
        return thrust_force
