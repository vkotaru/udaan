from ..control import Gains, Controller, PDController
import numpy as np
from ..utils import printc_fail, hat, vee

class QuadPosPD(PDController):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mass = 1.
        if "mass" in kwargs.keys():
            self.mass = kwargs["mass"]
        return

    def compute(self, *args):
        t = args[0]
        s, ds = args[1][0], args[1][1]
        sd, dsd, d2sd = self.setpoint(t)
        e = s - sd
        de = ds - dsd
        u = -self._gains.kp * e - self._gains.kd * de + d2sd + self._ge3
        u = self.mass * u
        return u
    

class QuadAttGeoPD(Controller):
    def __init__(self, **kwargs):
        self._gains = Gains(kp = np.array([2.4, 2.4, 1.35]), kd=np.array([0.35, 0.35, 0.225]))

        self._inertia = np.eye(3)
        self._inertia_inv = np.eye(3)
        self.mass = 1.

        if "inertia" in kwargs.keys():
            self.inertia = kwargs["inertia"]
        else:
            printc_fail("Inertia not provided")
        if "mass" in kwargs.keys():
            self.mass = kwargs["mass"]
        return

    @property
    def inertia(self):
        return self._inertia
    @inertia.setter
    def inertia(self, inertia):
        self._inertia = inertia
        self._inertia_inv = np.linalg.inv(inertia)

    def _cmd_accel_to_cmd_att(self, accel):
        """command attitude
        """
        norm_accel = np.linalg.norm(accel)
        b1d = np.array([1., 0., 0.])
        b3 = accel / norm_accel
        b3_b1d = np.cross(b3, b1d)
        norm_b3_b1d = np.linalg.norm(b3_b1d)
        b1 = (-1 / norm_b3_b1d) * np.cross(b3, b3_b1d)
        b2 = np.cross(b3, b1)
        Rd = np.hstack([
            np.expand_dims(b1, axis=1),
            np.expand_dims(b2, axis=1),
            np.expand_dims(b3, axis=1)
        ])
        return Rd

    def compute(self, *args):
        t = args[0]
        R, Omega = args[1][0], args[1][1]
        thrust_force = args[2]
        Rd = self._cmd_accel_to_cmd_att(thrust_force)
        # TODO: add angular velocity feedforward
        Omegad = np.zeros(3)  # self.des_state['Omega']
        dOmegad = np.zeros(3)  # self.des_state['dOmega']

        tmp = 0.5 * (Rd.T@ R - R.T@ Rd)
        eR = np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])  # vee-map
        eOmega = Omega - R.T@ Rd@ Omegad
        M = -self._gains.kp * eR - self._gains.kd * eOmega + np.cross(Omega, self.inertia @ Omega)
        M += -1 * self.inertia@(hat(Omega)@R.T@Rd@Omegad - R.T@Rd@dOmegad)
        f = thrust_force.dot(R[:, 2])
        return f, M



