"""QuadrotorCsPayloadVfx — adds VPython real-time visualization to QuadrotorCsPayloadBase."""

import time as _time

import numpy as np

from .base import QuadrotorCsPayloadBase


class QuadrotorCsPayloadVfx(QuadrotorCsPayloadBase):
    """Quadrotor with cable-suspended payload and VPython 3D visualization."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        from ...utils.vfx.quadrotor_cspayload_vfx import QuadrotorCSPayloadVFX

        self._vfx = QuadrotorCSPayloadVFX(l=self._cable_length)
        self._wall_start = None

    def step(self, u, desired_att=None):
        super().step(u, desired_att=desired_att)
        if self._wall_start is None:
            self._wall_start = _time.monotonic()
        wall_elapsed = _time.monotonic() - self._wall_start
        sleep = self.t - wall_elapsed
        if sleep > 0:
            _time.sleep(sleep)
        self._vfx.update(
            self.state.payload_position,
            np.asarray(self.state.cable_attitude),
            np.asarray(self.state.orientation),
        )

    def reset(self, **kwargs):
        super().reset(**kwargs)
        self._wall_start = None
        self._vfx.reset(
            self.state.payload_position,
            np.asarray(self.state.cable_attitude),
            np.asarray(self.state.orientation),
        )
