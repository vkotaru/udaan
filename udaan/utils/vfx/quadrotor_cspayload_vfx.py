import numpy as np
import vpython as vp

from .quadrotor_vfx import QuadrotorVFX


class QuadrotorCSPayloadVFX(QuadrotorVFX):
    def __init__(self, l=1, rate=200, retain=100):
        super().__init__(rate=rate, retain=retain)
        self._title = "udaan::Quadrotor payload"
        self.payload = vp.sphere(
            color=vp.color.blue, radius=0.05, make_trail=True, retain=self._retain
        )
        self.cable = vp.arrow(color=vp.color.black, shaftwidth=0.01, retain=1)
        self.l = l

    def reset(self, x=np.zeros(3), q=np.array([0.0, 0.0, -1.0]), R=np.eye(3)):
        self.payload.clear_trail()
        self.goal.clear_trail()
        self.update(x, q, R)

    def update(self, pL=np.zeros(3), q=np.array([0.0, 0.0, -1.0]), R=np.eye(3)):
        self.payload.pos = vp.vector(pL[0], pL[1], pL[2])
        vect = self.l * q
        x = pL - vect

        self.point.pos = vp.vector(x[0], x[1], x[2])
        self.cable.pos = self.point.pos
        self.cable.axis = vp.vector(vect[0], vect[1], vect[2])

        self._update_quadrotor(x, R)

    def update_goal(self, xd=np.zeros(3)):
        self.goal.pos = vp.vector(xd[0], xd[1], xd[2])
