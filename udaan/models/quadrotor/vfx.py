"""QuadrotorVfx — adds VPython real-time visualization to QuadrotorBase."""

import time as _time

import numpy as np

from .base import QuadrotorBase


class QuadrotorVfx(QuadrotorBase):
    """Quadrotor with VPython 3D visualization and real-time sync."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ...utils.vfx.quadrotor_vfx import QuadrotorVFX

        self._vfx = QuadrotorVFX()
        self._wall_start = None

    def step(self, u):
        super().step(u)
        if self._wall_start is None:
            self._wall_start = _time.monotonic()
        wall_elapsed = _time.monotonic() - self._wall_start
        sleep = self.t - wall_elapsed
        if sleep > 0:
            _time.sleep(sleep)
        self._vfx.update(self.state.position, np.asarray(self.state.orientation))

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._wall_start = None
        self._vfx.reset(self.state.position, np.asarray(self.state.orientation))
