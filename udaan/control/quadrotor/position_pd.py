"""Position PD controller for quadrotor."""

import numpy as np

from ...control import PDController
from ...core.defaults import DEFAULT_POS_KD, DEFAULT_POS_KP


class PositionPDController(PDController):
    """Implements a PD controller for position control of a quadrotor."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mass = 1.0 if "mass" not in kwargs else kwargs["mass"]

        self._gains.kp = DEFAULT_POS_KP.copy()
        self._gains.kd = DEFAULT_POS_KD.copy()

        self.pos_setpoint = np.array([0.0, 0.0, 0.0])

    def compute(self, *args):
        t = args[0]
        s, ds = args[1][0], args[1][1]
        sd, dsd, d2sd = self.setpoint(t)
        e = s - sd
        de = ds - dsd
        u = -self._gains.kp * e - self._gains.kd * de + d2sd + self._ge3
        # scale acceleration to force
        u = self.mass * u

        # store setpoint for visualization
        self.pos_setpoint = sd
        return u
