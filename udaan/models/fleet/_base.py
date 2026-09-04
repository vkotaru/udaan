"""The small shared base of the multi-agent models in this package.

``BaseModel`` used to live in ``udaan.models.base`` next to four standalone
legacy models (``FloatingPointmass``, ``S2Pendulum``,
``PointmassSuspendedPayload``, ``MultiPointmassSuspendedPayload``). Those four
were superseded — the last two are point-mass reductions of M2 and M3, which
now exist properly with backends, controllers, flatness maps and CLI commands —
and were removed. ``BaseModel`` itself is still the base of
:class:`~udaan.models.fleet.QuadrotorFleet`,
:class:`~udaan.models.fleet.QuadrotorCsPayloadFleet` and
:class:`~udaan.models.fleet.QuadrotorComparison`, so it moved here with them.

Keeping the name ``base`` at package level was also the source of a standing
confusion with the per-system ``models/<system>/base.py`` files, which are
something else entirely: the pure-dynamics parent of that system's backends.
"""

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
        # Legacy default. The two fleets override it with
        # :data:`~udaan.core.defaults.DEFAULT_SIM_DT`, the outer loop every
        # other model in the package runs at and the one the 1.25 ms MJCF
        # timestep divides exactly; `QuadrotorComparison` keeps 2 ms, which its
        # hand-written asset's own 2 ms timestep divides exactly too.
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
