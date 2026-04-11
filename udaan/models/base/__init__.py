import numpy as np

from ...core.defaults import GRAVITY
from ...utils.logging import get_logger

_logger = get_logger(__name__)


class BaseModel:
    def __init__(self, **kwargs):
        self._g = GRAVITY
        self._ge3 = np.array([0.0, 0.0, self._g])
        self._e1 = np.array([1.0, 0.0, 0.0])
        self._e2 = np.array([0.0, 1.0, 0.0])
        self._e3 = np.array([0.0, 0.0, 1.0])
        self._gravity = np.array([0.0, 0.0, -self._g])
        self.sim_timestep = 0.002

        self._n_state = 0
        self._n_action = 0

        self.t = 0.0

        self.verbose = False
        self.render = False
        # matched disturbance, i.e., disturbance if added to the input before updating the dynamics
        self.disturbance = False
        return

    def _parse_args(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.__dict__.keys():
                if type(self.__dict__[key]) is dict:
                    self.__dict__[key].update(value)
                else:
                    self.__dict__[key] = value
            else:
                _logger.warning(f"Key {key} not found in environment")

    def reset(self):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def simulate(self, **kwargs):
        raise NotImplementedError

    def get_action_size(self):
        return self._n_action

    def get_state_size(self):
        return self._n_state


from .floating_pointmass import FloatingPointmass as FloatingPointmass
from .multi_pointmass_suspended_payload import (
    MultiPointmassSuspendedPayload as MultiPointmassSuspendedPayload,
)
from .pointmass_suspended_payload import (
    PointmassSuspendedPayload as PointmassSuspendedPayload,
)
from .quadrotor_cspayload import QuadrotorCSPayload as QuadrotorCSPayload
from .s2_pendulum import S2Pendulum as S2Pendulum

__all__ = [
    "BaseModel",
    "FloatingPointmass",
    "MultiPointmassSuspendedPayload",
    "PointmassSuspendedPayload",
    "QuadrotorCSPayload",
    "S2Pendulum",
]
