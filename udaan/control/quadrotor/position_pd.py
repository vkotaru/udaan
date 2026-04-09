"""Position PD controller for quadrotor."""

import numpy as np

from ...control import PDController


class PositionPDController(PDController):
    """Implements a PD controller for position control of a quadrotor."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mass = 1.0 if "mass" not in kwargs else kwargs["mass"]

        self._gains.kp = np.array([4.1, 4.1, 8.1])
        self._gains.kd = 1.5 * np.array([2.0, 2.0, 6.0])

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
