import math
import warnings

import numpy as np

PI = math.pi


class Trajectory:
    def __init__(self):
        self._tf = 10

    def compute_traj_params(self):
        raise NotImplementedError

    def get(self, t):
        raise NotImplementedError


class SmoothTraj(Trajectory):
    def __init__(self, x0, xf, tf):
        self._x0 = x0
        self._xf = xf
        self._tf = tf
        self._pos_params = []
        self._vel_params = []
        self._acc_params = []

        self._t = lambda s: np.array([1.0, s, s**2, s**3, s**4, s**5])

        self.compute_traj_params()

    def compute_traj_params(self):
        raise NotImplementedError

    def get(self, t):
        if t >= self._tf:
            return self._xf, np.zeros(3), np.zeros(3)
        elif t <= 0:
            warnings.warn("cannot have t < 0")
            return self._x0, np.zeros(3), np.zeros(3)
        else:
            s = t / self._tf
            return (
                (np.array([self._t(s)]) @ self._pos_params)[0],
                (np.array([self._t(s)]) @ self._vel_params)[0],
                (np.array([self._t(s)]) @ self._acc_params)[0],
            )


class SmoothTraj5(SmoothTraj):
    """5th-order smooth trajectory with zero velocity and acceleration BCs."""

    def __init__(self, x0, xf, tf):
        super().__init__(x0, xf, tf)
        self._t = lambda s: np.array([1.0, s, s**2, s**3, s**4, s**5])

    def compute_traj_params(self):
        a = self._xf - self._x0
        self._pos_params = np.array([self._x0, np.zeros(3), np.zeros(3), 10 * a, -15 * a, 6 * a])
        self._vel_params = np.array(
            [
                np.zeros(3),
                2 * np.zeros(3),
                3 * 10 * a,
                4 * -15 * a,
                5 * 6 * a,
                np.zeros(3),
            ]
        )
        self._acc_params = np.array(
            [
                np.zeros(3),
                6 * 10 * a,
                12 * -15 * a,
                20 * 6 * a,
                np.zeros(3),
                np.zeros(3),
            ]
        )


class SmoothTraj3(SmoothTraj):
    """3rd-order smooth trajectory with zero velocity and acceleration BCs."""

    def __init__(self, x0, xf, tf):
        super().__init__(x0, xf, tf)
        self._t = lambda s: np.array([1.0, s, s**2, s**3])

    def compute_traj_params(self):
        a = self._xf - self._x0
        self._pos_params = np.array([self._x0, np.zeros(3), 3 * a, -2 * a])
        self._vel_params = np.array([np.zeros(3), 6 * a, -6 * a, np.zeros(3)])
        self._acc_params = np.array([6 * a, -12 * a, np.zeros(3), np.zeros(3)])


class SmoothTraj1(SmoothTraj):
    """1st-order (linear) smooth trajectory."""

    def __init__(self, x0, xf, tf):
        super().__init__(x0, xf, tf)
        self._t = lambda s: np.array([1.0, s])

    def compute_traj_params(self):
        a = self._xf - self._x0
        self._pos_params = np.array([self._x0, a])
        self._vel_params = np.array([a, np.zeros(3)])
        self._acc_params = np.array([np.zeros(3), np.zeros(3)])


class SmoothSineTraj(SmoothTraj):
    """Sine-based smooth trajectory."""

    def __init__(self, x0, xf, tf):
        self._pos_offset = np.zeros(3)
        self._pos_amp = np.zeros(3)
        self._vel_amp = np.zeros(3)
        self._acc_amp = np.zeros(3)
        super().__init__(x0, xf, tf)

    def compute_traj_params(self):
        self._pos_offset = 0.5 * (self._xf + self._x0)
        self._pos_amp = 0.5 * (self._xf - self._x0)
        self._vel_amp = 0.5 * (self._xf - self._x0) * (np.pi / self._tf)
        self._acc_amp = -0.5 * (self._xf - self._x0) * (np.pi / self._tf) ** 2

    def get(self, t):
        if t >= self._tf:
            return self._xf, np.zeros(3), np.zeros(3)
        elif t <= 0:
            warnings.warn("cannot have t < 0")
            return self._x0, np.zeros(3), np.zeros(3)
        else:
            x = self._pos_offset + self._pos_amp * np.sin(t * np.pi / self._tf - np.pi / 2)
            v = self._vel_amp * np.cos(t * np.pi / self._tf - np.pi / 2)
            a = self._acc_amp * np.sin(t * np.pi / self._tf - np.pi / 2)
            return x, v, a


class PolyTraj5(SmoothTraj):
    """5th-order polynomial with arbitrary boundary conditions."""

    def __init__(self, x0, xf, tf, v0=np.zeros(3), vf=np.zeros(3), a0=np.zeros(3), af=np.zeros(3)):
        self._v0 = v0
        self._vf = vf
        self._a0 = a0
        self._af = af
        super().__init__(x0, xf, tf)
        self._t = lambda s: np.array([1.0, s, s**2, s**3, s**4, s**5])

    def solve_params(self, p0, v0, a0, p1, v1, a1):
        b = np.array([[p1 - p0 - v0 - 0.5 * a0], [v1 - v0 - a0], [a1 - a0]])
        A = np.array([[1.0, 1.0, 1.0], [3.0, 4.0, 5.0], [6.0, 12.0, 20.0]])
        x = np.linalg.pinv(A) @ b
        return x

    def get(self, t):
        if t >= self._tf:
            return self._xf, self._vf, self._af
        elif t <= 0:
            return self._x0, self._v0, self._a0
        else:
            s = t / self._tf
            return (
                (np.array([self._t(s)]) @ self._pos_params)[0],
                (np.array([self._t(s)]) @ self._vel_params)[0],
                (np.array([self._t(s)]) @ self._acc_params)[0],
            )

    def compute_traj_params(self):
        a3 = np.zeros(3)
        a4 = np.zeros(3)
        a5 = np.zeros(3)
        for i in range(3):
            p = self.solve_params(
                self._x0[i],
                self._v0[i],
                self._a0[i],
                self._xf[i],
                self._vf[i],
                self._af[i],
            )
            a3[i], a4[i], a5[i] = p[0][0], p[1][0], p[2][0]

        self._pos_params = np.array([self._x0, self._v0, 0.5 * self._a0, a3, a4, a5])
        self._vel_params = np.array([self._v0, self._a0, 3.0 * a3, 4.0 * a4, 5.0 * a5, np.zeros(3)])
        self._acc_params = np.array(
            [self._a0, 6.0 * a3, 12.0 * a4, 20.0 * a5, np.zeros(3), np.zeros(3)]
        )


class CircularTraj(Trajectory):
    """Circular trajectory in the XY plane."""

    def __init__(self, center=np.zeros(3), radius=1, speed=1, th0=0, tf=10.0):
        self._center = center
        self._radius = radius
        self._speed = speed
        self._tf = tf
        self._w = self._speed / self._radius
        self._th0 = th0
        self._compute_params()

    def _compute_params(self):
        self._sint = lambda t: np.sin(self._th0 + self._w * t)
        self._cost = lambda t: np.cos(self._th0 + self._w * t)
        self._x = lambda t: (
            self._center
            + np.array([self._radius * self._cost(t), self._radius * self._sint(t), 0.0])
        )
        self._dx = lambda t: np.array(
            [
                -self._radius * self._sint(t) * self._w,
                self._radius * self._cost(t) * self._w,
                0.0,
            ]
        )
        self._d2x = lambda t: np.array(
            [
                -self._radius * self._cost(t) * self._w**2,
                -self._radius * self._sint(t) * self._w**2,
                0.0,
            ]
        )

    def get(self, t):
        return self._x(t), self._dx(t), self._d2x(t)


class CrazyTrajectory(Trajectory):
    """3-axis sinusoidal Lissajous-style trajectory."""

    def __init__(
        self,
        tf=10,
        ax=2,
        ay=2.5,
        az=1.5,
        f1=1 / 4,
        f2=1 / 5,
        f3=1 / 7,
        phix=0.0,
        phiy=0.0,
        phiz=0,
    ):
        super().__init__()
        self._tf = tf
        self.ax = ax
        self.ay = ay
        self.az = az
        self.f1 = f1
        self.f2 = f2
        self.f3 = f3
        self.phix = phix
        self.phiy = phiy
        self.phiz = phiz

    def get(self, t):
        w1 = 2 * np.pi * self.f1
        w2 = 2 * np.pi * self.f2
        w3 = 2 * np.pi * self.f3
        x = np.array(
            [
                self.ax * (1 - np.cos(w1 * t + self.phix)),
                self.ay * np.sin(w2 * t + self.phiy),
                self.az * np.cos(w3 * t + self.phiz),
            ]
        )
        dx = np.array(
            [
                self.ax * np.sin(w1 * t + self.phix) * w1,
                self.ay * np.cos(w2 * t + self.phiy) * w2,
                -self.az * np.sin(w3 * t + self.phiz) * w3,
            ]
        )
        d2x = np.array(
            [
                self.ax * np.cos(w1 * t + self.phix) * w1 * w1,
                -self.ay * np.sin(w2 * t + self.phiy) * w2 * w2,
                -self.az * np.cos(w3 * t + self.phiz) * w3 * w3,
            ]
        )
        return x, dx, d2x


def setpoint(t, sp=np.array([0.0, 0.0, 1.0])):
    traj = {}
    traj["x"] = sp
    traj["dx"] = np.zeros(3)
    traj["d2x"] = np.zeros(3)
    traj["d3x"] = np.zeros(3)
    traj["d4x"] = np.zeros(3)
    traj["d5x"] = np.zeros(3)
    traj["d6x"] = np.zeros(3)
    return traj


def circleXY(t, r=1, c=np.zeros(3), w=0.1 * PI):
    traj = {}
    traj["x"] = c + r * np.array([math.cos(w * t), math.sin(w * t), 0])
    traj["dx"] = r * np.array([-1 * w * math.sin(w * t), w * math.cos(w * t), 0])
    traj["d2x"] = r * np.array([-1 * w**2 * math.cos(w * t), -1 * w**2 * math.sin(w * t), 0])
    traj["d3x"] = r * np.array([w**3 * math.sin(w * t), -1 * w**3 * math.cos(w * t), 0])
    traj["d4x"] = r * np.array([w**4 * math.cos(w * t), w**4 * math.sin(w * t), 0])
    traj["d5x"] = r * np.array([-(w**5) * math.sin(w * t), w**5 * math.cos(w * t), 0])
    traj["d6x"] = r * np.array([-(w**6) * math.cos(w * t), -(w**6) * math.sin(w * t), 0])
    return traj
