"""Aggressive trajectories for quadrotor — flip and spiral maneuvers.

FlipTrajectory: vertical loop (360° pitch rotation) via circular
acceleration in the xz-plane. Requires differential flatness feedforward.

SpiralTrajectory: helical path with circular motion in yz-plane
and linear advance along x.

All derivatives up to snap (4th order) are analytic.
"""

from math import comb, factorial

import numpy as np

from .trajectory import Trajectory

_g = 9.81


#: Degree-11 smoothstep ``S(u) = 462u⁶ − 1980u⁷ + 3465u⁸ − 3080u⁹ + 1386u¹⁰ − 252u¹¹``:
#: ``S(0) = 0``, ``S(1) = 1``, and its first five derivatives vanish at both
#: ends. Coefficients lowest power first.
_SMOOTHSTEP = np.polynomial.Polynomial([0, 0, 0, 0, 0, 0, 462, -1980, 3465, -3080, 1386, -252])
_SMOOTHSTEP_INTEGRAL = _SMOOTHSTEP.integ()


def _ramp_up(u, w, duration, order=4):
    """``φ`` and its first ``order`` time derivatives for ``φ̇ = w·S(u)``,
    ``u = t/duration`` in ``[0, 1]``.

    With :data:`_SMOOTHSTEP` the rate's first five derivatives vanish where the
    ramp meets rest and where it meets the cruise, so ``φ`` is continuous
    through its sixth derivative there — what a payload flat map of order six
    (M3, M4) needs. Over a full ramp ``φ`` advances ``w·duration/2``.
    """
    out = np.empty(order + 1)
    out[0] = w * duration * _SMOOTHSTEP_INTEGRAL(u)
    poly = _SMOOTHSTEP
    for k in range(1, order + 1):
        out[k] = w / duration ** (k - 1) * poly(u)
        poly = poly.deriv()
    return out


#: Gauss–Legendre nodes and weights on [−1, 1], and the longest panel the
#: composite rule uses. 16 nodes integrate a degree-31 polynomial exactly; a
#: flip-rate panel of ≤ 25 ms spans ≤ 0.31 rad at 12.6 rad/s, where the
#: integrands are resolved to round-off.
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(16)
_GL_PANEL = 0.025


def _integrate(fn, lo, hi, breaks=()):
    """Composite Gauss–Legendre quadrature of ``fn`` (scalar or array) over [lo, hi].

    ``breaks`` are points where ``fn`` loses smoothness; panels never straddle
    one, because Gauss–Legendre's accuracy rests on the integrand being smooth
    across the panel.
    """
    if hi <= lo:
        return 0.0 * fn(lo)
    knots = [lo, *sorted(b for b in breaks if lo < b < hi), hi]
    edges = np.concatenate(
        [
            np.linspace(a, b, max(1, int(np.ceil((b - a) / _GL_PANEL))) + 1)[:-1]
            for a, b in zip(knots[:-1], knots[1:], strict=True)
        ]
        + [[hi]]
    )
    total = 0.0 * fn(lo)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        half, mid = 0.5 * (b - a), 0.5 * (a + b)
        total = total + half * sum(
            w * fn(mid + half * x) for x, w in zip(_GL_NODES, _GL_WEIGHTS, strict=True)
        )
    return total


def _hermite_displacement(x0, duration, v0, vf, a0, af):
    r"""End position whose velocity is the cubic Hermite interpolant of the BCs.

    .. math::

        x_f = x_0 + \tfrac{T}{2}(v_0 + v_f) + \tfrac{T^2}{12}(a_0 - a_f)

    This is the integral of the cubic that matches ``v`` and ``a`` at both ends
    (exact by Euler–Maclaurin), so a segment ending here needs no velocity
    reversal to meet its boundary conditions: the quintic through these BCs
    has the Hermite cubic as its velocity.
    """
    return x0 + 0.5 * duration * (v0 + vf) + duration**2 / 12.0 * (a0 - af)


class _SnapMatchedSegment:
    """Degree-9 polynomial meeting position through snap at both ends.

    Ten conditions — ``(p, v, a, j, s)`` at each end — fix the ten
    coefficients, so a phase built from these joins its neighbours C⁴. Solved
    in normalised time ``u = t/T``, where the k-th derivative boundary value is
    scaled by ``Tᵏ``; evaluated back in time units.
    """

    _DEGREE = 9

    def __init__(self, start_jet, end_jet, duration):
        n = self._DEGREE + 1
        self._T = float(duration)
        A = np.zeros((n, n))
        for k in range(5):
            A[k] = self._basis(0.0, k)
            A[5 + k] = self._basis(1.0, k)
        scale = self._T ** np.arange(5)[:, None]
        B = np.vstack((np.asarray(start_jet) * scale, np.asarray(end_jet) * scale))
        self._coef = np.linalg.solve(A, B)

    @classmethod
    def _basis(cls, u, k):
        row = np.zeros(cls._DEGREE + 1)
        for i in range(k, cls._DEGREE + 1):
            row[i] = factorial(i) / factorial(i - k) * u ** (i - k)
        return row

    def get_full(self, t):
        u = t / self._T
        return tuple(self._basis(u, k) @ self._coef / self._T**k for k in range(5))


class FlipTrajectory(Trajectory):
    r"""Vertical loop trajectory: hold -> climb -> 360° pitch flip -> recover -> hold.

    **Flip.** A constant-magnitude thrust whose direction turns in the
    xz-plane through the angle :math:`\theta(t)`::

        a_x = T sin θ,    a_z = T cos θ − g

    so ``j = T θ̇ (cos θ, 0, −sin θ)`` and
    ``s = T θ̈ (cos θ, 0, −sin θ) − T θ̇² (sin θ, 0, cos θ)``. The rate
    :math:`\dot\theta` ramps from 0 to :math:`\omega = 2\pi/` ``flip_duration``
    over ``flip_ramp_time`` through the cubic smoothstep
    :math:`S(u) = 3u^2 - 2u^3`, holds, and ramps back down; the ramps together
    cover :math:`\omega T_r` of the :math:`2\pi n` total, so the flip phase lasts
    :math:`2\pi n/\omega + T_r`. At its ends :math:`\dot\theta = \ddot\theta = 0`,
    so jerk and snap vanish there, and :math:`\ddot\theta` is continuous at the
    ramp–cruise joins, so snap is too. A constant-rate flip would begin with a
    body-rate step (jerk ``(Tω, 0, 0)`` from rest), which no snap-matched climb
    can meet without reversing.

    During the inverted phase (90°-270°) the thrust's vertical projection is
    negative — the quadrotor is in free-fall. This is physically correct; the
    attitude controller maintains rotation via torque feedforward, from the
    fixed-axis reference attitude of :meth:`reference_attitude` (the generic
    flat map's heading convention is singular on a pitch loop). The flip's position and velocity are integrals of
    ``a`` with no closed form under the ramps; they are evaluated by composite
    Gauss–Legendre quadrature (:func:`_integrate_accel`).

    **Launch velocity.** The climb hands the flip a vertical velocity ``V``
    chosen so the flip ends with ``v_z = 0``,
    :math:`V = g\,t_{flip} - T\int_0^{t_{flip}} \cos\theta\,dt`. (With a
    constant rate the integral vanishes and this reduces to ``g·t_flip``.)
    The ramps are mirror images, so :math:`\int \sin\theta = 0` and the flip
    also ends with ``v_x = 0``.

    **Climb and recovery** are :class:`_SnapMatchedSegment`\ s, joining their
    neighbours through snap. Each ends at the Hermite displacement of its own
    velocity and acceleration boundary conditions (see
    :func:`_hermite_displacement`), so the climb's ``ż`` never reverses.

    **Where it ends.** The manoeuvre does not return to ``start``. At the
    defaults it ends **5.66 m above and 0.62 m along x** from it:

    - the climb rises 2.67 m;
    - the flip rises 1.35 m and drifts the 0.62 m in x;
    - the recovery rises 1.64 m.

    That is the price of ``ż`` never reversing. A climb that must hand the flip
    an upward launch velocity, and a recovery that must absorb the flip's
    ``T − g`` upward acceleration, each end at the Hermite displacement of
    their own boundary conditions rather than at the height they started from.
    Read the final hover from :attr:`end_position`.

    Args:
        start: starting hover position [x, y, z].
        flip_duration: time for one 360° rotation at full rate (seconds).
        n_loops: number of rotations.
        thrust_ratio: thrust / (m*g) during the flip.
        climb_time: duration of the climb phase.
        recover_time: duration of the recovery phase.
        hold_time: hover duration before and after the maneuver.
        flip_ramp_time: duration of each of the flip-rate ramps (seconds).
            Default ``flip_duration / 4``.

    Raises:
        ValueError: if ``flip_ramp_time`` is not positive or exceeds what the
            loops take at full rate.
    """

    def __init__(
        self,
        start=np.array([0.0, 0.0, 2.0]),
        flip_duration=0.5,
        n_loops=1,
        thrust_ratio=1.5,
        climb_time=2.0,
        recover_time=2.0,
        hold_time=1.0,
        flip_ramp_time=None,
    ):
        super().__init__()
        self._start = np.array(start, dtype=float)
        self._T = thrust_ratio * _g
        self._omega = 2 * np.pi / flip_duration
        self._theta_total = 2 * np.pi * n_loops
        full_rate = self._theta_total / self._omega
        ramp = 0.25 * flip_duration if flip_ramp_time is None else float(flip_ramp_time)
        if not 0.0 < ramp <= full_rate:
            raise ValueError(
                f"flip_ramp_time must lie in (0, {full_rate:.6g}] s, got {flip_ramp_time}"
            )
        self._t_ramp = ramp
        self._t_flip = full_rate + ramp
        self._theta_breaks = (ramp, self._t_flip - ramp)  # where θ⃛ jumps
        self._t_climb = climb_time
        self._t_recover = recover_time

        # Phase boundaries
        self._t1 = hold_time
        self._t2 = self._t1 + climb_time
        self._t3 = self._t2 + self._t_flip
        self._t4 = self._t3 + recover_time
        self._tf = self._t4 + hold_time

        # Launch velocity: chosen so the flip ends with v_z = 0.
        cos_integral = _integrate(
            lambda s: np.cos(self._theta(s)[0]), 0.0, self._t_flip, self._theta_breaks
        )
        self._v_launch = np.array([0.0, 0.0, _g * self._t_flip - self._T * cos_integral])
        flip_accel = np.array([0.0, 0.0, self._T - _g])  # a at both ends of the flip
        z = np.zeros(3)

        # The climb ends, and the flip begins, at the height the boundary
        # velocities and accelerations imply (see `_hermite_displacement`), so
        # a climb that must arrive moving up never dips below its start.
        self._flip_start = _hermite_displacement(
            self._start, climb_time, z, self._v_launch, z, flip_accel
        )
        self._climb_traj = _SnapMatchedSegment(
            [self._start, z, z, z, z],
            [self._flip_start, self._v_launch, flip_accel, z, z],
            climb_time,
        )

        self._flip_end_pos, self._flip_end_vel = self._flip_state(self._t_flip)[:2]
        recover_end = _hermite_displacement(
            self._flip_end_pos, recover_time, self._flip_end_vel, z, flip_accel, z
        )
        self._recover_traj = _SnapMatchedSegment(
            [self._flip_end_pos, self._flip_end_vel, flip_accel, z, z],
            [recover_end, z, z, z, z],
            recover_time,
        )
        self._end_pos = recover_end.copy()

    def _theta(self, tp):
        """``(θ, θ̇, θ̈)`` at ``tp`` into the flip."""
        w, Tr = self._omega, self._t_ramp
        if tp < Tr:
            u = tp / Tr
            return np.array(
                [
                    w * Tr * (u**3 - 0.5 * u**4),
                    w * (3 * u**2 - 2 * u**3),
                    w / Tr * (6 * u - 6 * u**2),
                ]
            )
        if tp <= self._t_flip - Tr:
            return np.array([0.5 * w * Tr + w * (tp - Tr), w, 0.0])
        u = max(self._t_flip - tp, 0.0) / Tr
        return np.array(
            [
                self._theta_total - w * Tr * (u**3 - 0.5 * u**4),
                w * (3 * u**2 - 2 * u**3),
                -w / Tr * (6 * u - 6 * u**2),
            ]
        )

    def _accel(self, tp):
        theta = self._theta(tp)[0]
        return np.array([self._T * np.sin(theta), 0.0, self._T * np.cos(theta) - _g])

    def _flip_state(self, tp):
        theta, d1, d2 = self._theta(tp)
        st, ct = np.sin(theta), np.cos(theta)
        T = self._T
        v = self._v_launch + _integrate(self._accel, 0.0, tp, self._theta_breaks)
        # p(tp) = p0 + v0 tp + ∫₀^tp (tp − s) a(s) ds
        p = (
            self._flip_start
            + self._v_launch * tp
            + _integrate(lambda s: (tp - s) * self._accel(s), 0.0, tp, self._theta_breaks)
        )
        a = np.array([T * st, 0.0, T * ct - _g])
        j = T * d1 * np.array([ct, 0.0, -st])
        s = T * d2 * np.array([ct, 0.0, -st]) - T * d1**2 * np.array([st, 0.0, ct])
        return p, v, a, j, s

    def reference_attitude(self, t):
        r"""Closed-form reference attitude during the flip, else ``None``.

        The flip is a rotation about the fixed body axis :math:`e_2` (pitch),
        through the thrust-direction angle :math:`\theta(t)`::

            R_ref = exp(θ ê₂),    Ω_ref = θ̇ e₂,    Ω̇_ref = θ̈ e₂

        Its ``b₃ = (sin θ, 0, cos θ)`` is the thrust direction the flat map
        recovers. The flat map's heading convention ``b₁d = (cos ψ, sin ψ, 0)``
        is singular on a pitch loop — ``b₃`` passes through ``e₁`` — and its
        ``b₂ = sign(cos θ)·e₂`` turns the reference 180° about ``b₃`` for the
        whole inverted half. While upright the two agree exactly, and the flip
        starts and ends upright at rest, so the hand-over is continuous in
        ``R``, ``Ω`` and ``Ω̇``. Outside the flip phase this returns ``None``
        and :meth:`~udaan.models.quadrotor.QuadrotorBase.simulate` falls back
        to the flat map. The flip lies in the xz-plane, so its heading is 0.

        Returns:
            ``(R, Ω, Ω̇)`` while ``t`` is inside the flip phase, else ``None``.
        """
        if not self._t2 < t <= self._t3:
            return None
        theta, d1, d2 = self._theta(t - self._t2)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        e2 = np.array([0.0, 1.0, 0.0])
        return R, d1 * e2, d2 * e2

    def get(self, t):
        return self.get_full(t)[:3]

    def get_full(self, t):
        z5 = np.zeros(3)
        if t <= self._t1:
            return self._start.copy(), z5, z5, z5, z5
        elif t <= self._t2:
            return self._climb_traj.get_full(t - self._t1)
        elif t <= self._t3:
            return self._flip_state(t - self._t2)
        elif t <= self._t4:
            return self._recover_traj.get_full(t - self._t3)
        else:
            return self._end_pos.copy(), z5, z5, z5, z5

    @property
    def duration(self):
        return self._tf

    @property
    def end_position(self):
        return self._end_pos.copy()


class SpiralTrajectory(Trajectory):
    r"""Helical spiral: circle in yz-plane while advancing along x.

    The trajectory holds, executes ``n_loops`` loops of a helix, then holds
    again. The helix is a fixed path :math:`p(\varphi)` traversed with a
    time-warped phase :math:`\varphi(t)`:

    .. math::

        p(\varphi) = x_0 + \Bigl[\tfrac{A}{2\pi n}\varphi,\;
            r\sin\varphi,\; r(1 - \cos\varphi)\Bigr]

    whose rate :math:`\dot\varphi` ramps from 0 to :math:`\omega = s/r` over
    ``ramp_time`` :math:`T_r` through the degree-11 smoothstep
    :math:`S(u) = 462u^6 - 1980u^7 + 3465u^8 - 3080u^9 + 1386u^{10} - 252u^{11}`,
    holds :math:`\omega`, and ramps back down the same way. :math:`S` has
    zero first to fifth derivatives at both ends, so :math:`\varphi` is
    continuous through its sixth derivative, and so is the reference at every
    phase boundary — the depth M3's and M4's flat maps read. The time
    derivatives of :math:`p(\varphi(t))` follow by Faà di Bruno, to any order
    (see :meth:`_spiral_jet` and :meth:`get_jet`). The ramps together cover
    :math:`\omega T_r` of the :math:`2\pi n` total, so the cruise lasts
    :math:`2\pi n/\omega - T_r` and the helix phase
    :math:`2\pi n/\omega + T_r`.

    With ``ramp_time = 0`` the reference steps from rest to
    :math:`(v_x, r\omega, 0)` — 2 m/s and 5 m/s² at the defaults — at both
    ends of the helix.

    Args:
        start: starting hover position [x, y, z].
        radius: radius of the circular motion in yz-plane (meters).
        speed: tangential speed of the circular motion (m/s).
        n_loops: number of full loops.
        advance: total x-distance traveled over all loops (meters).
        hold_time: hover duration before and after the maneuver.
        ramp_time: duration of each of the ramp-in and ramp-out (seconds).

    Raises:
        ValueError: if ``ramp_time`` is negative, or longer than the loops take
            at full speed (``2π n_loops / ω``), which leaves no cruise.
    """

    def __init__(
        self,
        start=np.array([0.0, 0.0, 2.0]),
        radius=0.8,
        speed=2.0,
        n_loops=2,
        advance=3.0,
        hold_time=1.0,
        ramp_time=2.0,
    ):
        super().__init__()
        self._start = np.array(start, dtype=float)
        self._r = radius
        self._n_loops = n_loops
        self._advance = advance
        self._hold = hold_time

        self._w = speed / radius
        self._phi_total = 2 * np.pi * n_loops
        t_full_speed = self._phi_total / self._w
        if ramp_time < 0 or ramp_time > t_full_speed:
            raise ValueError(
                f"ramp_time must lie in [0, {t_full_speed:.6g}] s (the loops at full "
                f"speed), got {ramp_time}"
            )
        self._t_ramp = float(ramp_time)
        self._t_cruise = t_full_speed - self._t_ramp
        self._t_spiral = self._t_cruise + 2 * self._t_ramp
        self._x_per_rad = advance / self._phi_total

        # Phase boundaries
        self._t1 = hold_time  # end of initial hold
        self._t2 = self._t1 + self._t_spiral  # end of spiral
        self._tf = self._t2 + hold_time

        # End position
        self._end_pos = self._start + np.array([advance, 0.0, 0.0])

    def _phase(self, tp, order=4):
        """``φ`` and its first ``order`` derivatives at ``tp`` into the helix phase."""
        w, Tr = self._w, self._t_ramp
        if Tr > 0 and tp < Tr:
            return _ramp_up(tp / Tr, w, Tr, order)
        if tp <= Tr + self._t_cruise or Tr == 0:
            # In the cruise (and, with no ramps, at the end: `tp` can pass it by
            # round-off, so hold the final phase).
            out = np.zeros(order + 1)
            phi_in = 0.5 * w * Tr  # what the ramp-in covered
            out[0] = min(phi_in + w * (tp - Tr), self._phi_total)
            out[1] = w
            return out
        # The ramp-out is the ramp-in reflected in time about the phase's end:
        # odd derivatives keep their sign, even ones flip.
        out = _ramp_up(max(self._t_spiral - tp, 0.0) / Tr, w, Tr, order)
        out[0] = self._phi_total - out[0]
        out[2::2] *= -1
        return out

    def _spiral_state(self, tp):
        """Full state (through snap) at time tp into the spiral phase."""
        return tuple(self._spiral_jet(tp, 4))

    def _spiral_jet(self, tp, order):
        """``(order + 1, 3)``: the helix position and its time derivatives.

        Faà di Bruno for ``p(φ(t))``, by the recursion on ``g = e^{iφ}``:
        ``ġ = iφ̇ g``, so ``g⁽ⁿ⁾ = Σₖ C(n−1, k) (iφ)⁽ᵏ⁺¹⁾ g⁽ⁿ⁻¹⁻ᵏ⁾`` (Leibniz on
        that product). Then ``y = r Im g``, ``z = r(1 − Re g)``, and ``x`` is
        linear in ``φ``.
        """
        phi = self._phase(tp, order)
        g = np.empty(order + 1, dtype=complex)
        g[0] = np.exp(1j * phi[0])
        for n in range(1, order + 1):
            g[n] = sum(comb(n - 1, k) * 1j * phi[k + 1] * g[n - 1 - k] for k in range(n))
        jet = np.zeros((order + 1, 3))
        jet[:, 0] = self._x_per_rad * phi
        jet[:, 1] = self._r * g.imag
        jet[:, 2] = -self._r * g.real
        jet[0] += self._start + np.array([0.0, 0.0, self._r])
        return jet

    def get_jet(self, t, order):
        """Position jet ``(order + 1, 3)`` at ``t``, every row exact.

        The deeper protocol a flat map of order above four reads (see
        :meth:`~udaan.utils.trajectory.CrazyTrajectory.get_jet`). The holds are
        exactly at rest, so only row 0 is non-zero there.
        """
        if order < 0:
            raise ValueError(f"order must be >= 0, got {order}")
        if self._t1 < t <= self._t2:
            return self._spiral_jet(t - self._t1, order)
        jet = np.zeros((order + 1, 3))
        jet[0] = self._start if t <= self._t1 else self._end_pos
        return jet

    def get(self, t):
        """Return (position, velocity, acceleration) at time t."""
        if t <= self._t1:
            return self._start.copy(), np.zeros(3), np.zeros(3)
        elif t <= self._t2:
            p, v, a, _j, _s = self._spiral_state(t - self._t1)
            return p, v, a
        else:
            return self._end_pos.copy(), np.zeros(3), np.zeros(3)

    def get_full(self, t):
        """Return (position, velocity, acceleration, jerk, snap) at time t."""
        z5 = np.zeros(3)
        if t <= self._t1:
            return self._start.copy(), z5, z5, z5, z5
        elif t <= self._t2:
            return self._spiral_state(t - self._t1)
        else:
            return self._end_pos.copy(), z5, z5, z5, z5

    @property
    def duration(self):
        return self._tf

    @property
    def end_position(self):
        return self._end_pos.copy()
