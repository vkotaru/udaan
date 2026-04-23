import numpy as np
import vpython as vp

from . import VFXHandler


class QuadrotorVFX(VFXHandler):
    ARM_LENGTH = 0.1
    PROP_RADIUS = 0.125

    def __init__(self, rate=200, retain=100):
        super().__init__(title="Quadrotor", rate=rate)
        super().create_env()
        self._rate = rate
        self._retain = retain
        self.point = vp.sphere(
            color=vp.color.blue, radius=0.025, make_trail=True, retain=self._retain, opacity=0.75
        )
        self.goal = vp.sphere(color=vp.color.red, radius=0.05, make_trail=True, retain=1)

        self._init_quadrotor_geometry()
        self._traj = vp.curve(color=vp.color.black, radius=0.025)
        QuadrotorVFX.update(self)

    def _init_quadrotor_geometry(self):
        """Create arms, propellers, and curves shared by all quadrotor visualizers."""
        l = self.ARM_LENGTH
        r = self.PROP_RADIUS

        self._arm0 = [
            np.array([l, 0.0, 0.05]),
            np.array([l, 0.0, 0.0]),
            np.array([-l, 0.0, 0.0]),
            np.array([-l, 0.0, 0.05]),
        ]
        self._arm1 = [
            np.array([0.0, l, 0.05]),
            np.array([0.0, l, 0.0]),
            np.array([0.0, -l, 0.0]),
            np.array([0.0, -l, 0.05]),
        ]
        self._prop_offsets = [
            np.array([l, 0.0, 0.05]),
            np.array([0.0, l, 0.05]),
            np.array([0.0, -l, 0.05]),
            np.array([-l, 0.0, 0.05]),
        ]

        self.curve1 = vp.curve(color=vp.color.red, radius=0.01)
        self.curve2 = vp.curve(color=vp.color.blue, radius=0.01)

        self.props = [
            vp.ellipsoid(
                length=r,
                height=0.1 * r,
                width=r,
                color=vp.color.cyan if i == 0 else vp.color.black,
                opacity=0.5,
            )
            for i in range(4)
        ]

    def _update_quadrotor(self, x, R):
        """Update arm curves and propeller positions."""
        self.curve1.clear()
        self.curve2.clear()

        for pt in self._arm0:
            p = x + R @ pt
            self.curve1.append(vp.vector(p[0], p[1], p[2]))
        for pt in self._arm1:
            p = x + R @ pt
            self.curve2.append(vp.vector(p[0], p[1], p[2]))
        for i in range(4):
            p = x + R @ self._prop_offsets[i]
            ax = R @ np.array([0.0, 0.0, 1.0])
            self.props[i].pos = vp.vector(p[0], p[1], p[2])
            self.props[i].up = vp.vector(ax[0], ax[1], ax[2])

    def reset(self, x=np.zeros(3), R=np.eye(3)):
        self.point.clear_trail()
        self.update(x, R)

    def update(self, x=np.zeros(3), R=np.eye(3)):
        self.point.pos = vp.vector(x[0], x[1], x[2])
        self._update_quadrotor(x, R)

    def update_goal(self, xd=np.zeros(3)):
        self.goal.pos = vp.vector(xd[0], xd[1], xd[2])

    def update_traj(self, points=None):
        if points is not None:
            self._traj.clear()
            for p in points:
                self._traj.append(vp.vector(p[0], p[1], p[2]))
