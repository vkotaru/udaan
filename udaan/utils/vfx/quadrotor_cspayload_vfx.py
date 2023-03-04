from . import *


class QuadrotorCSPayloadVFX(VFXHandler):

    def __init__(self, l=1, rate=200, retain=100):
        super().__init__(title='Quadrotor payload', rate=rate)
        super().create_env()
        self._rate = rate
        self._retain = retain
        self.point = vp.sphere(color=vp.color.blue,
                               radius=0.025,
                               make_trail=True,
                               retain=1,
                               opacity=0.75)
        self.goal = vp.sphere(color=vp.color.red,
                              radius=0.05,
                              make_trail=True,
                              retain=self._retain)
        self.payload = vp.sphere(color=vp.color.blue,
                                 radius=0.05,
                                 make_trail=True,
                                 retain=self._retain)
        self.cable = vp.arrow(color=vp.color.black, shaftwidth=0.01, retain=1)
        self.l = l

        self.arm_l = 0.1  # length of quadrotor boom
        self.prop_r = 0.125  # radius of propeller prop

        self._arm0 = [
            np.array([self.arm_l, 0., 0.05]),
            np.array([self.arm_l, 0., 0.0]),
            np.array([-self.arm_l, 0., 0.0]),
            np.array([-self.arm_l, 0.0, 0.05])
        ]

        self._arm1 = [
            np.array([0., self.arm_l, 0.05]),
            np.array([0., self.arm_l, 0.0]),
            np.array([0., -self.arm_l, 0.0]),
            np.array([0., -self.arm_l, 0.05])
        ]

        self._props = [
            np.array([self.arm_l, 0., 0.05]),
            np.array([0., self.arm_l, 0.05]),
            np.array([0., -self.arm_l, 0.05]),
            np.array([-self.arm_l, 0., 0.05])
        ]

        self.curve1 = vp.curve(color=vp.color.red, radius=0.01)
        self.curve2 = vp.curve(color=vp.color.blue, radius=0.01)

        self.props = [
            vp.ellipsoid(length=self.prop_r,
                         height=0.1 * self.prop_r,
                         width=self.prop_r,
                         color=vp.color.cyan,
                         opacity=0.5),
            vp.ellipsoid(length=self.prop_r,
                         height=0.1 * self.prop_r,
                         width=self.prop_r,
                         color=vp.color.black,
                         opacity=0.5),
            vp.ellipsoid(length=self.prop_r,
                         height=0.1 * self.prop_r,
                         width=self.prop_r,
                         color=vp.color.black,
                         opacity=0.5),
            vp.ellipsoid(length=self.prop_r,
                         height=0.1 * self.prop_r,
                         width=self.prop_r,
                         color=vp.color.black,
                         opacity=0.5)
        ]

        self._traj = vp.curve(color=vp.color.red, radius=0.025)
        self.update()

    def reset(self, x=np.zeros(3), q=np.array([0., 0., -1.]), R=np.eye(3)):
        self.payload.clear_trail()
        self.goal.clear_trail()
        self.update(x, q, R)

    def update(self, pL=np.zeros(3), q=np.array([0., 0., -1.]), R=np.eye(3)):
        self.curve1.clear()
        self.curve2.clear()

        self.payload.pos = vp.vector(pL[0], pL[1], pL[2])
        vect = self.l * q
        x = pL - vect

        self.point.pos = vp.vector(x[0], x[1], x[2])
        self.cable.pos = self.point.pos
        self.cable.axis = vp.vector(vect[0], vect[1], vect[2])

        for _ in self._arm0:
            p = x + R @ _
            self.curve1.append(vp.vector(p[0], p[1], p[2]))

        for _ in self._arm1:
            p = x + R @ _
            self.curve2.append(vp.vector(p[0], p[1], p[2]))

        for i in range(4):
            p = x + R @ self._props[i]
            ax = R @ np.array([0., 0., 1.])
            self.props[i].pos = vp.vector(p[0], p[1], p[2])
            self.props[i].up = vp.vector(ax[0], ax[1], ax[2])

    def update_goal(self, xd=np.zeros(3)):
        self.goal.pos = vp.vector(xd[0], xd[1], xd[2])

    def update_traj(self, points=None):
        if points is not None:
            self._traj.clear()
            for p in points:
                self._traj.append(vp.vector(p[0], p[1], p[2]))
