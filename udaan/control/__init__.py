import numpy as np


class Gains(object):
    def __init__(self, kp=np.zeros(3), kd=np.zeros(3), ki=np.zeros(3)):
        self.kp = kp
        self.kd = kd
        self.ki = ki
        return


class Controller(object):
    def __init__(self):
        self.freq = 500.0
        self._g = 9.81
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._e1 = np.array([1.0, 0.0, 0.0])
        self._e2 = np.array([0.0, 1.0, 0.0])
        self._e3 = np.array([0.0, 0.0, 1.0])
        self._gravity = np.array([0.0, 0.0, -self._g])
        return

    def compute(self, *args):
        raise NotImplementedError


class PDController(Controller):
    def __init__(self, **kwargs):
        super().__init__()
        self._gains = Gains()

        if "kp" in kwargs.keys():
            self._gains.kp = kwargs["kp"]
        if "kd" in kwargs.keys():
            self._gains.kd = kwargs["kd"]

        if "setpoint" in kwargs.keys():
            self.setpoint = kwargs["setpoint"]
        else:
            self.setpoint = lambda t: (
                np.array([0.0, 0.0, 1.0]),
                np.zeros(3),
                np.zeros(3),
            )
        return

    def compute(self, *args):
        t = args[0]
        s, ds = args[1][0], args[1][1]
        sd, dsd, d2sd = self.setpoint(t)
        e = s - sd
        de = ds - dsd
        u = -self._gains.kp * e - self._gains.kd * de + d2sd
        return u


from .quadrotor import *
from .quadrotor_cspayload import *
