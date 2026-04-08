import numpy as np

from ..base import BaseModel


class FloatingPointmass(BaseModel):
    """Floating point-mass model: a 3D point mass subject to gravity and
    an external force input.

    State: position (3), velocity (3) -> n_state=6, n_action=3
    """

    class State:
        def __init__(self):
            self.position = np.zeros(3)
            self.velocity = np.zeros(3)
            return

        def reset(self):
            self.position = np.zeros(3)
            self.velocity = np.zeros(3)
            return

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._n_state = 6
        self._n_action = 3
        self.mass = 1.0
        self.state = FloatingPointmass.State()
        self._parse_args(**kwargs)
        return

    def _zoh(self, force):
        """Zero-order hold Euler integration."""
        dt = self.sim_timestep
        net_accel = self._gravity + force / self.mass
        self.state.position += dt * self.state.velocity + 0.5 * net_accel * dt * dt
        self.state.velocity += dt * net_accel

    def step(self, action):
        self._zoh(action)
        self.t += self.sim_timestep

    def reset(self, **kwargs):
        self.t = 0.0
        self.state.reset()
        for key in ["position", "velocity"]:
            if key in kwargs:
                setattr(self.state, key, kwargs[key])
        return

    def get_rand_init_state(self, rand=True):
        rng = np.random.default_rng()
        init_pos = -5 + 10 * rng.random(3) if rand else np.zeros(3)
        return {"position": init_pos, "velocity": np.zeros(3)}
