"""Setpoint step and rest-to-rest waypoint missions.

StepSetpoint: hold one point, then jump to another — a true step, for step
responses. It has no ``get_full``: a step has no honest higher derivatives, so
it never switches on the flatness feedforward.

WaypointTrajectory: take off from the first waypoint, visit the rest, stopping
at each, and come to rest at the last. Each leg is a 7th-order rest-to-rest
polynomial, so the reference is continuous through jerk.
"""

import math

import numpy as np
from numpy.polynomial import polynomial as P

from .trajectory import Trajectory


class StepSetpoint(Trajectory):
    """Hold ``start`` until ``step_time``, then hold ``end``: a step, no smoothing.

    Args:
        start: position held before the step [x, y, z].
        end: position held from ``step_time`` on [x, y, z].
        step_time: when the setpoint jumps, in s.
    """

    def __init__(self, start, end, step_time=2.0):
        super().__init__()
        self._start = np.asarray(start, dtype=float)
        self._end = np.asarray(end, dtype=float)
        self._t_step = float(step_time)

    def get(self, t):
        """Return (position, velocity, acceleration) at time t; velocity and acceleration are 0."""
        p = self._start if t < self._t_step else self._end
        return p.copy(), np.zeros(3), np.zeros(3)

    @property
    def step_time(self):
        return self._t_step

    @property
    def end_position(self):
        return self._end.copy()


class _RestToRest:
    r"""7th-order polynomial from ``p0`` to ``p1`` over ``[0, T]``, at rest at both ends.

    Velocity, acceleration and jerk vanish at both ends. Solved in normalised
    time ``s = t/T`` and evaluated with the ``1/T^k`` chain factor on the
    ``k``-th derivative.
    """

    def __init__(self, p0, p1, duration):
        self.T = float(duration)
        A = np.zeros((8, 8))
        for k in range(4):
            for n in range(k, 8):
                # d^k/ds^k s^n at s = 0 is k! for n = k and 0 otherwise; at s = 1 it is n!/(n-k)!.
                A[k, n] = math.factorial(n) / math.factorial(n - k) * (0.0 if n - k else 1.0)
                A[4 + k, n] = math.factorial(n) / math.factorial(n - k)
        b = np.zeros((8, 3))
        b[0] = np.asarray(p0, dtype=float)
        b[4] = np.asarray(p1, dtype=float)
        coeffs = np.linalg.solve(A, b)  # (8, 3), ascending powers of s
        self._derivs = [coeffs]
        for _ in range(4):
            self._derivs.append(P.polyder(self._derivs[-1], axis=0))

    def get_full(self, t):
        s = np.clip(t / self.T, 0.0, 1.0)
        return tuple(P.polyval(s, c) / self.T**k for k, c in enumerate(self._derivs))


class WaypointTrajectory(Trajectory):
    r"""Visit ``waypoints`` in order, coming to rest at each; a take-off-to-landing mission.

    The vehicle holds the first waypoint for ``start_hold``, flies each leg as a
    rest-to-rest 7th-order polynomial lasting
    :math:`T = \max(2,\; 2\sqrt{d})` s for a leg of length :math:`d` m, holds
    each intermediate waypoint for ``hold_time``, and holds the last for
    ``end_hold``. Start and end on the ground for a take-off and a landing.

    Velocity, acceleration and jerk vanish at every waypoint, so the reference
    is continuous through jerk. Snap is not: a 7th-order rest-to-rest leg starts
    and ends with non-zero snap, so it steps where a leg meets a hold, and with
    it the feedforward's angular acceleration. :meth:`get_full` is exact
    between those instants, and the flatness feedforward runs on it.

    Args:
        waypoints: two or more positions [x, y, z], visited in order.
        hold_time: hover at each intermediate waypoint, in s.
        start_hold: hover at the first waypoint before the first leg, in s.
        end_hold: hover at the last waypoint after the last leg, in s.

    Raises:
        ValueError: for fewer than two waypoints.
    """

    def __init__(self, waypoints, hold_time=0.6, start_hold=1.0, end_hold=2.0):
        super().__init__()
        points = [np.asarray(w, dtype=float) for w in waypoints]
        if len(points) < 2 or any(p.shape != (3,) for p in points):
            raise ValueError("waypoints must be two or more positions [x, y, z]")
        pieces: list = [(start_hold, points[0])]
        for a, b in zip(points[:-1], points[1:], strict=True):
            d = float(np.linalg.norm(np.subtract(b, a)))
            T = max(2.0, 2.0 * math.sqrt(d))
            pieces.append((T, _RestToRest(a, b, T)))
            pieces.append((hold_time, b))
        pieces[-1] = (end_hold, points[-1])
        self._pieces = pieces
        self._starts = np.cumsum([0.0] + [p[0] for p in pieces])
        self._tf = self._starts[-1].item()  # a Python float
        self._waypoints = points

    def get_full(self, t):
        """Return (position, velocity, acceleration, jerk, snap) at time t."""
        i = int(np.searchsorted(self._starts, t, side="right") - 1)
        i = min(max(i, 0), len(self._pieces) - 1)
        T, seg = self._pieces[i]
        tau = min(max(t - self._starts[i], 0.0), T)
        if t >= self._tf:
            T, seg = self._pieces[-1]
            tau = T
        if isinstance(seg, np.ndarray):
            return seg.copy(), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
        return seg.get_full(tau)

    def get(self, t):
        """Return (position, velocity, acceleration) at time t."""
        return self.get_full(t)[:3]

    @property
    def duration(self):
        return self._tf

    @property
    def waypoints(self):
        return [p.copy() for p in self._waypoints]
