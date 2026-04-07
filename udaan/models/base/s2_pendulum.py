import numpy as np

from ..base import BaseModel
from ... import manif


class S2Pendulum(BaseModel):
    """Pendulum model with attitude on S2.

    State: attitude (S2 unit vector, 3), angular_velocity (3)
    -> n_state=6, n_action=3
    """

    class State(object):
        def __init__(self):
            self.attitude = manif.S2(np.array([0.0, 0.0, -1.0]))
            self.angular_velocity = np.zeros(3)
            return

        def reset(self):
            self.attitude = manif.S2(np.array([0.0, 0.0, -1.0]))
            self.angular_velocity = np.zeros(3)
            return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_state = 6
        self._n_action = 3
        self.mass = 0.1
        self.length = 1.0
        self.state = S2Pendulum.State()
        self._parse_args(**kwargs)
        return

    def _zoh(self, torque):
        """Zero-order hold integration on S2."""
        dt = self.sim_timestep
        q = self.state.attitude
        om = self.state.angular_velocity

        # S2 geodesic update for attitude
        self.state.attitude = q.step(om * dt)
        # angular velocity update
        dom = -(self._g / self.length) * np.cross(
            self.state.attitude, self._e3
        ) + torque / (self.mass * self.length**2)
        self.state.angular_velocity = om + dom * dt

    def step(self, action):
        self._zoh(action)
        self.t += self.sim_timestep

    def reset(self, **kwargs):
        self.t = 0.0
        self.state.reset()
        if "attitude" in kwargs:
            self.state.attitude = manif.S2(kwargs["attitude"])
        if "angular_velocity" in kwargs:
            self.state.angular_velocity = kwargs["angular_velocity"]
        return

    def get_rand_init_state(self, rand=True):
        if rand:
            phi = -np.pi + 2 * np.pi * np.random.randn(1)
            th = np.pi * np.random.rand(1)
            init_att = manif.S2.fromEuler(phi[0], th[0])
        else:
            init_att = manif.S2(np.array([1.0, 0.0, 0.0]))
        return {"attitude": init_att, "angular_velocity": np.zeros(3)}
