"""Aggressive trajectories for quadrotor — flip and spiral maneuvers.

FlipTrajectory: vertical loop (360° pitch rotation) via circular
acceleration in the xz-plane. Requires differential flatness feedforward.

SpiralTrajectory: helical path with circular motion in yz-plane
and linear advance along x.

All derivatives up to snap (4th order) are analytic.
"""

import numpy as np

from .trajectory import PolyTraj5, Trajectory

_g = 9.81


class FlipTrajectory(Trajectory):
    """Vertical loop trajectory: hold -> climb -> 360° pitch flip -> recover -> hold.

    The flip phase uses a constant-thrust circular acceleration profile:
        a_x(t') = T * sin(w * t')
        a_z(t') = T * cos(w * t') - g
    The thrust vector rotates 360° in the xz-plane. During the inverted
    phase (90°-270°), thrust projection is negative — the quadrotor is
    in free-fall. This is physically correct; the attitude controller
    maintains rotation via torque feedforward from flat2state.

    Args:
        start: starting hover position [x, y, z].
        flip_duration: time for the 360° rotation (seconds).
        thrust_ratio: thrust / (m*g) during the flip.
        climb_time: duration of the climb phase.
        recover_time: duration of the recovery phase.
        hold_time: hover duration before and after the maneuver.
    """

    def __init__(
        self,
        start=np.array([0.0, 0.0, 2.0]),
        flip_duration=0.5,
        n_loops=3,
        thrust_ratio=1.5,
        climb_time=2.0,
        recover_time=2.0,
        hold_time=1.0,
    ):
        super().__init__()
        self._start = np.array(start, dtype=float)
        self._t_flip = flip_duration * n_loops
        self._T = thrust_ratio * _g
        self._omega = 2 * np.pi / flip_duration
        self._t_climb = climb_time
        self._t_recover = recover_time

        # Phase boundaries
        self._t1 = hold_time
        self._t2 = self._t1 + climb_time
        self._t3 = self._t2 + flip_duration
        self._t4 = self._t3 + recover_time
        self._tf = self._t4 + hold_time

        # Launch velocity: v_z0 = g * t_flip_total so flip ends with v_z = 0
        self._v_launch = np.array([0.0, 0.0, _g * self._t_flip])

        # Flip end state
        self._flip_end_pos, self._flip_end_vel = self._compute_flip_end()

        # Smooth transitions
        self._climb_traj = PolyTraj5(
            x0=self._start,
            xf=self._start,
            tf=climb_time,
            v0=np.zeros(3),
            vf=self._v_launch,
            a0=np.zeros(3),
            af=np.array([0.0, 0.0, self._T - _g]),
        )

        self._recover_traj = PolyTraj5(
            x0=self._flip_end_pos,
            xf=self._flip_end_pos.copy(),
            tf=recover_time,
            v0=self._flip_end_vel,
            vf=np.zeros(3),
            a0=np.array([0.0, 0.0, self._T - _g]),
            af=np.zeros(3),
        )

        self._end_pos = self._flip_end_pos.copy()

    def _compute_flip_end(self):
        T, w, tf = self._T, self._omega, self._t_flip
        v0z = self._v_launch[2]
        dx = T / w * tf - T / w**2 * np.sin(w * tf)
        dz = v0z * tf + T / w**2 * (1 - np.cos(w * tf)) - _g * tf**2 / 2
        vx = T / w * (1 - np.cos(w * tf))
        vz = v0z + T / w * np.sin(w * tf) - _g * tf
        return self._start + np.array([dx, 0.0, dz]), np.array([vx, 0.0, vz])

    def _flip_state(self, tp):
        T, w = self._T, self._omega
        v0z = self._v_launch[2]
        sw, cw = np.sin(w * tp), np.cos(w * tp)

        p = self._start + np.array([
            T / w * tp - T / w**2 * sw,
            0.0,
            v0z * tp + T / w**2 * (1 - cw) - _g * tp**2 / 2,
        ])
        v = np.array([T / w * (1 - cw), 0.0, v0z + T / w * sw - _g * tp])
        a = np.array([T * sw, 0.0, T * cw - _g])
        j = np.array([T * w * cw, 0.0, -T * w * sw])
        s = np.array([-T * w**2 * sw, 0.0, -T * w**2 * cw])
        return p, v, a, j, s

    def get(self, t):
        if t <= self._t1:
            return self._start.copy(), np.zeros(3), np.zeros(3)
        elif t <= self._t2:
            return self._climb_traj.get(t - self._t1)
        elif t <= self._t3:
            p, v, a, _, _ = self._flip_state(t - self._t2)
            return p, v, a
        elif t <= self._t4:
            return self._recover_traj.get(t - self._t3)
        else:
            return self._end_pos.copy(), np.zeros(3), np.zeros(3)

    def get_full(self, t):
        z5 = np.zeros(3)
        if t <= self._t1:
            return self._start.copy(), z5, z5, z5, z5
        elif t <= self._t2:
            p, v, a = self._climb_traj.get(t - self._t1)
            return p, v, a, z5, z5
        elif t <= self._t3:
            return self._flip_state(t - self._t2)
        elif t <= self._t4:
            p, v, a = self._recover_traj.get(t - self._t3)
            return p, v, a, z5, z5
        else:
            return self._end_pos.copy(), z5, z5, z5, z5

    @property
    def duration(self):
        return self._tf

    @property
    def end_position(self):
        return self._end_pos.copy()


class SpiralTrajectory(Trajectory):
    """Helical spiral: circle in yz-plane while advancing along x.

    The trajectory smoothly accelerates from hover, executes N loops
    of a helix, then smoothly decelerates back to hover.

    Args:
        start: starting hover position [x, y, z].
        radius: radius of the circular motion in yz-plane (meters).
        speed: tangential speed of the circular motion (m/s).
        n_loops: number of full loops.
        advance: total x-distance traveled over all loops (meters).
        hold_time: hover duration before and after the maneuver.
    """

    def __init__(
        self,
        start=np.array([0.0, 0.0, 2.0]),
        radius=0.8,
        speed=2.0,
        n_loops=2,
        advance=3.0,
        hold_time=1.0,
    ):
        super().__init__()
        self._start = np.array(start, dtype=float)
        self._r = radius
        self._n_loops = n_loops
        self._advance = advance
        self._hold = hold_time

        # Angular velocity and duration of the spiral phase
        self._w = speed / radius
        self._t_spiral = n_loops * 2 * np.pi / self._w
        self._vx = advance / self._t_spiral  # constant x velocity during spiral

        # Phase boundaries
        self._t1 = hold_time  # end of initial hold
        self._t2 = self._t1 + self._t_spiral  # end of spiral
        self._tf = self._t2 + hold_time

        # End position
        self._end_pos = self._start + np.array([advance, 0.0, 0.0])

    def _spiral_state(self, tp):
        """Full state (through snap) at time tp into the spiral phase."""
        r = self._r
        w = self._w
        vx = self._vx
        sw, cw = np.sin(w * tp), np.cos(w * tp)

        # Position: x advances linearly, yz traces a circle
        # Circle starts at (y=0, z=center) and goes through (y=+r, z=center)
        p = self._start + np.array([
            vx * tp,
            r * sw,
            r * (1 - cw),
        ])

        v = np.array([vx, r * w * cw, r * w * sw])
        a = np.array([0.0, -r * w**2 * sw, r * w**2 * cw])
        j = np.array([0.0, -r * w**3 * cw, -r * w**3 * sw])
        s = np.array([0.0, r * w**4 * sw, -r * w**4 * cw])

        return p, v, a, j, s

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
