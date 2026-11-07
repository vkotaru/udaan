"""Figure-8 (lemniscate of Gerono) with a swinging heading.

Figure8Trajectory: a horizontal figure-8 whose phase ramps up from rest,
cruises and ramps back down, with a heading ``ψ(t)`` that swings with the
phase. The position derivatives through snap and the yaw derivatives through
``ψ̈`` are analytic.
"""

import math

import numpy as np
from numpy.polynomial import polynomial as P

from .trajectory import Trajectory

#: ``S(τ) = 35τ⁴ − 84τ⁵ + 70τ⁶ − 20τ⁷``: 0 → 1 with zero first to third
#: derivatives at both ends. Coefficients lowest power first.
_SMOOTHSTEP7 = np.array([0, 0, 0, 0, 35, -84, 70, -20], dtype=float)


class _PhaseProfile:
    """``θ(t)``: ``θ̇`` ramps from 0 to ``ω`` over ``Tr`` through :data:`_SMOOTHSTEP7`,
    cruises, and ramps back down, ending at rest at ``θ_end``.

    :meth:`get` returns ``(θ, θ̇, θ̈, θ⁽³⁾, θ⁽⁴⁾)``, continuous everywhere.
    """

    def __init__(self, omega, theta_end, Tr):
        self.w, self.Tr = float(omega), float(Tr)
        self.Tc = theta_end / omega - Tr
        if self.Tc < 0:
            raise ValueError(
                f"ramp_time {Tr:g} s is too long for {theta_end:g} rad at {omega:g} rad/s: "
                f"it must be at most {theta_end / omega:g} s"
            )
        S = _SMOOTHSTEP7
        self.S = [S]
        for _ in range(3):
            self.S.append(P.polyder(self.S[-1]))
        self.intS = P.polyint(S)
        self.theta_ramp = self.w * self.Tr * P.polyval(1.0, self.intS)
        self.duration = 2 * self.Tr + self.Tc
        self.theta_end = theta_end

    def get(self, t):
        w, Tr = self.w, self.Tr
        if t <= 0:
            return np.zeros(5)
        if t < Tr:
            tau = t / Tr
            return np.array(
                [w * Tr * P.polyval(tau, self.intS)]
                + [w * P.polyval(tau, self.S[k]) / Tr**k for k in range(4)]
            )
        if t < Tr + self.Tc:
            return np.array([self.theta_ramp + w * (t - Tr), w, 0, 0, 0])
        if t < self.duration:
            # The ramp-down: θ̇ = ω(1 − S(τ)).
            tau = (t - Tr - self.Tc) / Tr
            th = self.theta_ramp + w * self.Tc + w * Tr * (tau - P.polyval(tau, self.intS))
            return np.array(
                [th, w * (1 - P.polyval(tau, self.S[0]))]
                + [-w * P.polyval(tau, self.S[k]) / Tr**k for k in range(1, 4)]
            )
        return np.array([self.theta_end, 0, 0, 0, 0])


def _compose(g, th):
    """Time derivatives 0..4 of ``g(θ(t))`` by Faà di Bruno; ``g = (g, g′, …, g⁗)`` at
    ``θ``, ``th = (θ, θ̇, …, θ⁽⁴⁾)``."""
    _, t1, t2, t3, t4 = th
    g0, g1, g2, g3, g4 = g
    return np.array(
        [
            g0,
            g1 * t1,
            g2 * t1**2 + g1 * t2,
            g3 * t1**3 + 3 * g2 * t1 * t2 + g1 * t3,
            g4 * t1**4 + 6 * g3 * t1**2 * t2 + 3 * g2 * t2**2 + 4 * g2 * t1 * t3 + g1 * t4,
        ]
    )


def _sin_derivs(x):
    """``sin`` and its first four derivatives at ``x``."""
    s, c = math.sin(x), math.cos(x)
    return (s, c, -s, -c, s)


class Figure8Trajectory(Trajectory):
    r"""Horizontal figure-8 about ``center``, with a heading that swings ``±yaw_amplitude``.

    .. math::

        p(t) = c + \bigl[A\sin\theta,\; \tfrac{A}{2}\sin 2\theta,\; 0\bigr],
        \qquad \psi(t) = \psi_{\max}\sin\theta,

    a lemniscate of Gerono traversed ``n_loops`` times. The phase rate
    :math:`\dot\theta` ramps from rest to ``omega`` over ``ramp_time`` through
    the smoothstep :math:`S(\tau) = 35\tau^4 - 84\tau^5 + 70\tau^6 - 20\tau^7`,
    cruises and ramps back down; :math:`S` has zero first to third derivatives
    at both ends, so the reference is continuous through snap — what the
    quadrotor's flat map reads. The vehicle holds ``center`` for ``hold_time``
    before and after. At the defaults the peak speed is about 2.5 m/s and the
    peak yaw rate 1.9 rad/s.

    Heading is the quadrotor's second flat output, so it is not part of
    :meth:`get_full`: bind :meth:`yaw` as the model's ``yaw_setpoint``
    beside the position setpoint::

        f8 = Figure8Trajectory(center=[0.0, 0.0, 1.5])
        mdl.position_controller.setpoint = f8.get
        mdl.yaw_setpoint = f8.yaw

    Args:
        center: centre of the figure-8, and where it starts and ends [x, y, z].
        amplitude: half-width ``A`` along x, in m (half that along y).
        omega: cruise phase rate, in rad/s.
        n_loops: number of full figure-8s.
        ramp_time: duration of each of the ramp-in and ramp-out, in s.
        hold_time: hover at ``center`` before and after, in s.
        yaw_amplitude: ``ψ_max``, in rad; 0 flies it at ψ ≡ 0.

    Raises:
        ValueError: if the ramps take longer than the loops at full rate.
    """

    def __init__(
        self,
        center=np.array([0.0, 0.0, 1.5]),
        amplitude=1.5,
        omega=1.2,
        n_loops=2,
        ramp_time=2.0,
        hold_time=1.0,
        yaw_amplitude=math.pi / 2,
    ):
        super().__init__()
        self._center = np.asarray(center, dtype=float)
        self._A = amplitude
        self._hold = hold_time
        self._phase = _PhaseProfile(omega, 2 * math.pi * n_loops, ramp_time)
        self._yaw_amp = yaw_amplitude
        self._tf = hold_time + self._phase.duration + hold_time

    def _theta(self, t):
        return self._phase.get(t - self._hold)

    def get_full(self, t):
        """Return (position, velocity, acceleration, jerk, snap) at time t."""
        th = self._theta(t)
        x = self._A * _compose(_sin_derivs(th[0]), th)
        s2 = _sin_derivs(2 * th[0])
        y = 0.5 * self._A * _compose(tuple(2**k * s2[k] for k in range(5)), th)
        out = np.zeros((5, 3))
        out[:, 0], out[:, 1] = x, y
        out[0] += self._center
        return tuple(out)

    def get(self, t):
        """Return (position, velocity, acceleration) at time t."""
        return self.get_full(t)[:3]

    def yaw(self, t):
        """Heading ``(ψ, ψ̇, ψ̈)`` at time t — a model's ``yaw_setpoint``."""
        th = self._theta(t)
        d = self._yaw_amp * _compose(_sin_derivs(th[0]), th)
        return float(d[0]), float(d[1]), float(d[2])

    @property
    def duration(self):
        return self._tf
