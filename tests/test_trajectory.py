import numpy as np
import pytest

from udaan.utils.trajectory import (
    CircularTraj,
    CrazyTrajectory,
    PolyTraj5,
    SmoothSineTraj,
    SmoothTraj1,
    SmoothTraj3,
    SmoothTraj5,
    circleXY,
    setpoint,
)


class TestSmoothTraj5:
    def test_boundary_conditions(self):
        x0 = np.array([0.0, 0.0, 0.0])
        xf = np.array([1.0, 2.0, 3.0])
        tf = 5.0
        traj = SmoothTraj5(x0, xf, tf)

        # At t=0, should return x0
        x, v, a = traj.get(0.001)
        np.testing.assert_allclose(x, x0, atol=0.01)

        # At t=tf, should return xf
        x, v, a = traj.get(tf)
        np.testing.assert_allclose(x, xf)
        np.testing.assert_allclose(v, np.zeros(3))
        np.testing.assert_allclose(a, np.zeros(3))

    def test_midpoint(self):
        x0 = np.zeros(3)
        xf = np.ones(3)
        traj = SmoothTraj5(x0, xf, 10.0)
        x, v, a = traj.get(5.0)
        np.testing.assert_allclose(x, 0.5 * np.ones(3), atol=1e-10)

    def test_returns_three_arrays(self):
        traj = SmoothTraj5(np.zeros(3), np.ones(3), 1.0)
        result = traj.get(0.5)
        assert len(result) == 3
        for arr in result:
            assert arr.shape == (3,)


class TestSmoothTraj3:
    def test_boundary_conditions(self):
        x0 = np.array([1.0, 2.0, 3.0])
        xf = np.array([4.0, 5.0, 6.0])
        tf = 2.0
        traj = SmoothTraj3(x0, xf, tf)

        x, v, a = traj.get(tf)
        np.testing.assert_allclose(x, xf)


class TestSmoothTraj1:
    def test_linear_interpolation(self):
        x0 = np.zeros(3)
        xf = np.array([2.0, 4.0, 6.0])
        tf = 2.0
        traj = SmoothTraj1(x0, xf, tf)

        x, v, a = traj.get(1.0)
        np.testing.assert_allclose(x, xf / 2.0, atol=1e-10)


class TestSmoothSineTraj:
    def test_boundary_conditions(self):
        x0 = np.zeros(3)
        xf = np.array([1.0, 1.0, 1.0])
        tf = 5.0
        traj = SmoothSineTraj(x0, xf, tf)

        x_start, v_start, a_start = traj.get(0.001)
        np.testing.assert_allclose(x_start, x0, atol=0.01)

        x_end, v_end, a_end = traj.get(tf)
        np.testing.assert_allclose(x_end, xf)
        np.testing.assert_allclose(v_end, np.zeros(3))


class TestPolyTraj5:
    def test_zero_bc(self):
        x0 = np.zeros(3)
        xf = np.ones(3)
        traj = PolyTraj5(x0, xf, 5.0)
        x, v, a = traj.get(5.0)
        np.testing.assert_allclose(x, xf)

    def test_custom_bc(self):
        x0 = np.zeros(3)
        xf = np.ones(3)
        v0 = np.array([0.1, 0.0, 0.0])
        vf = np.array([0.0, 0.1, 0.0])
        traj = PolyTraj5(x0, xf, 5.0, v0=v0, vf=vf)
        x_end, v_end, a_end = traj.get(5.0)
        np.testing.assert_allclose(x_end, xf)
        np.testing.assert_allclose(v_end, vf)


class TestCircularTraj:
    def test_returns_on_circle(self):
        center = np.zeros(3)
        radius = 2.0
        traj = CircularTraj(center=center, radius=radius, speed=1.0)

        for t in [0, 1, 2, 5]:
            x, dx, d2x = traj.get(t)
            dist = np.linalg.norm(x[:2] - center[:2])
            np.testing.assert_allclose(dist, radius, atol=1e-10)
            assert x[2] == pytest.approx(0.0)

    def test_velocity_magnitude(self):
        speed = 3.0
        traj = CircularTraj(radius=1.0, speed=speed)
        _, dx, _ = traj.get(1.0)
        np.testing.assert_allclose(np.linalg.norm(dx), speed, atol=1e-10)

    def test_jet_matches_get_and_finite_differences(self):
        traj = CircularTraj(center=np.array([0.5, -1.0, 2.0]), radius=1.5, speed=2.0, th0=0.3)
        t, h = 1.7, 1e-5
        jet = traj.get_jet(t, 6)
        assert jet.shape == (7, 3)
        for row, ref in zip(jet[:3], traj.get(t), strict=True):
            np.testing.assert_allclose(row, ref, atol=1e-12)
        for k in range(1, 7):
            fd = (traj.get_jet(t + h, k - 1)[k - 1] - traj.get_jet(t - h, k - 1)[k - 1]) / (2 * h)
            np.testing.assert_allclose(jet[k], fd, rtol=1e-6, atol=1e-6)
        for row, ref in zip(traj.get_full(t), jet[:5], strict=True):
            np.testing.assert_allclose(row, ref)

    def test_jet_refuses_negative_order(self):
        with pytest.raises(ValueError):
            CircularTraj().get_jet(0.0, -1)


class TestCrazyTrajectory:
    def test_returns_three_arrays(self):
        traj = CrazyTrajectory()
        x, dx, d2x = traj.get(1.0)
        assert x.shape == (3,)
        assert dx.shape == (3,)
        assert d2x.shape == (3,)

    def test_different_params(self):
        traj1 = CrazyTrajectory(ax=1.0, f1=0.5)
        traj2 = CrazyTrajectory(ax=3.0, f1=0.5)
        x1, _, _ = traj1.get(1.0)
        x2, _, _ = traj2.get(1.0)
        assert not np.allclose(x1, x2)


class TestSetpoint:
    def test_returns_dict_with_all_keys(self):
        traj = setpoint(0)
        for key in ["x", "dx", "d2x", "d3x", "d4x", "d5x", "d6x"]:
            assert key in traj

    def test_derivatives_zero(self):
        sp = np.array([1.0, 2.0, 3.0])
        traj = setpoint(0, sp=sp)
        np.testing.assert_array_equal(traj["x"], sp)
        for key in ["dx", "d2x", "d3x", "d4x", "d5x", "d6x"]:
            np.testing.assert_array_equal(traj[key], np.zeros(3))


class TestCircleXY:
    def test_returns_dict_with_all_keys(self):
        traj = circleXY(0)
        for key in ["x", "dx", "d2x", "d3x", "d4x", "d5x", "d6x"]:
            assert key in traj

    def test_on_circle(self):
        r = 2.0
        c = np.array([1.0, 0.0, 0.0])
        traj = circleXY(0, r=r, c=c)
        dist = np.linalg.norm(traj["x"][:2] - c[:2])
        np.testing.assert_allclose(dist, r, atol=1e-10)


def _central_difference(fn, t, row, h=1e-6):
    """``d/dt`` of row ``row`` of ``fn(t)``, by central difference."""
    return (fn(t + h)[row] - fn(t - h)[row]) / (2 * h)


class TestDerivativesAreTimeDerivatives:
    """Each returned derivative row is ``d/dt`` of the row above it.

    These polynomials are written in normalised time ``s = t/t_f``, so each
    row carries a ``1/t_f`` chain-rule factor per derivative. Every case here
    uses ``t_f != 1``, where ``d/ds`` and ``d/dt`` differ.
    """

    @pytest.mark.parametrize("cls", [SmoothTraj5, SmoothTraj3, SmoothTraj1])
    @pytest.mark.parametrize("tf", [0.5, 4.0])
    def test_smooth_traj(self, cls, tf):
        traj = cls(np.zeros(3), np.array([2.0, -1.0, 0.5]), tf)
        for t in np.linspace(0.1, 0.9, 5) * tf:
            np.testing.assert_allclose(
                traj.get(t)[1], _central_difference(traj.get, t, 0), atol=1e-6
            )
            np.testing.assert_allclose(
                traj.get(t)[2], _central_difference(traj.get, t, 1), atol=1e-6
            )

    def test_poly_traj5_meets_its_boundary_conditions_in_time_units(self):
        tf = 4.0
        v0, a0 = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.0])
        vf, af = np.array([0.0, 0.0, 2.0]), np.array([0.3, 0.0, 0.0])
        traj = PolyTraj5(np.zeros(3), np.ones(3), tf, v0=v0, vf=vf, a0=a0, af=af)
        eps = 1e-9
        p0, v0_got, a0_got = traj.get(eps)
        p1, vf_got, af_got = traj.get(tf - eps)
        np.testing.assert_allclose(p0, np.zeros(3), atol=1e-8)
        np.testing.assert_allclose(p1, np.ones(3), atol=1e-8)
        np.testing.assert_allclose(v0_got, v0, atol=1e-7)
        np.testing.assert_allclose(vf_got, vf, atol=1e-7)
        np.testing.assert_allclose(a0_got, a0, atol=1e-7)
        np.testing.assert_allclose(af_got, af, atol=1e-7)
        # Matching the endpoints is not enough: solving and reporting both in
        # s-units also matches them. The interior is where it shows.
        for t in (0.7, 2.0, 3.3):
            np.testing.assert_allclose(
                traj.get(t)[1], _central_difference(traj.get, t, 0), atol=1e-6
            )
            np.testing.assert_allclose(
                traj.get(t)[2], _central_difference(traj.get, t, 1), atol=1e-6
            )

    def test_poly_traj5_full_jet_is_consistent(self):
        traj = PolyTraj5(
            np.zeros(3), np.ones(3), 3.0, v0=np.array([1.0, 0.0, 0.0]), af=np.array([0, 0, 1.0])
        )
        for t in (0.4, 1.5, 2.7):
            full = traj.get_full(t)
            for row in range(4):
                np.testing.assert_allclose(
                    full[row + 1], _central_difference(traj.get_full, t, row, h=1e-5), atol=1e-5
                )


class TestFlipTrajectory:
    @staticmethod
    def _boundaries(flip):
        ramp = flip._t_ramp
        return (flip._t1, flip._t2, flip._t2 + ramp, flip._t3 - ramp, flip._t3, flip._t4)

    @pytest.mark.parametrize("n_loops", [1, 3])
    def test_reference_is_continuous_through_snap_at_every_boundary(self, n_loops):
        """Every phase meets its neighbours C⁴: position through snap is continuous
        at each boundary, including the flip's ends, where the rate ramps make
        jerk and snap vanish."""
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory(n_loops=n_loops)
        for boundary in self._boundaries(flip):
            before, after = flip.get_full(boundary - 1e-9), flip.get_full(boundary + 1e-9)
            for row in range(5):
                np.testing.assert_allclose(before[row], after[row], atol=1e-3 if row == 4 else 1e-6)

    @staticmethod
    def _flat_reference(flip, t):
        """``(R, Ω, Ω̇)`` the generic flat map recovers, heading ψ ≡ 0."""
        from udaan.flatness import Jet, Quadrotor, QuadrotorFlats, yaw_jet

        flats = QuadrotorFlats(x=Jet(np.array(flip.get_full(t))), psi=yaw_jet(None, t))
        ref, _ = Quadrotor(mass=1.0, inertia=np.diag([2e-3, 2e-3, 4e-3])).recover(flats)
        return (
            np.asarray(ref.orientation),
            np.asarray(ref.angular_velocity),
            np.asarray(ref.angular_acceleration),
        )

    def _reference(self, flip, t):
        """What ``simulate()`` feeds forward: the flip's own attitude, else the flat map."""
        own = flip.reference_attitude(t)
        return own if own is not None else self._flat_reference(flip, t)

    @staticmethod
    def _angle(Ra, Rb):
        return float(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0)))

    @pytest.mark.parametrize("n_loops", [1, 3])
    def test_reference_attitude_never_jumps(self, n_loops):
        """The flat map's ``b₁d = (cos ψ, sin ψ, 0)`` heading is singular on a
        pitch loop: ``b₂ = sign(cos θ)·e₂`` turns the reference 180° about
        ``b₃`` for the whole inverted half. At 1 ms the full flip rate turns it
        by 0.72°."""
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory(n_loops=n_loops)
        ts = np.arange(flip._t1, flip._t4, 1e-3)
        Rs = [self._reference(flip, t)[0] for t in ts]
        worst = max(self._angle(a, b) for a, b in zip(Rs[:-1], Rs[1:], strict=True))
        assert np.degrees(worst) < 1.0

    def test_reference_b3_is_the_flat_map_thrust_direction(self):
        """``b₃`` is the flat map's ``(a + g e₃)/‖a + g e₃‖`` — read off the
        acceleration, since the flat map itself raises where ``b₃ = e₁``."""
        from udaan.core.defaults import GRAVITY
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory()
        for t in np.linspace(flip._t2, flip._t3, 401)[1:]:
            R, _, _ = flip.reference_attitude(t)
            thrust = flip.get(t)[2] + np.array([0.0, 0.0, GRAVITY])
            np.testing.assert_allclose(R[:, 2], thrust / np.linalg.norm(thrust), atol=1e-12)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)

    def test_reference_rates_are_derivatives_of_the_attitude(self):
        """``Ω̂ = Rᵀ Ṙ`` and ``Ω̇`` against central differences, ramps and cruise."""
        from udaan.utils import vee
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory()
        h = 1e-6
        for tp in (0.03, 0.1, 0.25, 0.4, 0.55):
            t = flip._t2 + tp
            R, Om, dOm = flip.reference_attitude(t)
            dR = (flip.reference_attitude(t + h)[0] - flip.reference_attitude(t - h)[0]) / (2 * h)
            np.testing.assert_allclose(Om, vee(R.T @ dR), atol=1e-6)
            dOm_fd = (flip.reference_attitude(t + h)[1] - flip.reference_attitude(t - h)[1]) / (
                2 * h
            )
            np.testing.assert_allclose(dOm, dOm_fd, atol=1e-4)

    def test_reference_hands_over_continuously_at_the_flip_ends(self):
        """Upright at rest the two conventions agree, so ``R``, ``Ω`` and ``Ω̇``
        are continuous where the flip phase meets the climb and the recovery."""
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory()
        for boundary in (flip._t2, flip._t3):
            before = self._reference(flip, boundary - 1e-9)
            after = self._reference(flip, boundary + 1e-9)
            for b, a in zip(before, after, strict=True):
                np.testing.assert_allclose(b, a, atol=1e-4)
        assert flip.reference_attitude(flip._t2) is None
        assert flip.reference_attitude(flip._t3) is not None
        assert flip.reference_attitude(flip._t3 + 1e-9) is None

    def test_reference_agrees_with_the_flat_map_while_upright(self):
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory()
        for tp in (0.02, 0.08, 0.58, 0.61):  # θ well under 90° or past 270°
            t = flip._t2 + tp
            for own, flat in zip(
                flip.reference_attitude(t), self._flat_reference(flip, t), strict=True
            ):
                np.testing.assert_allclose(own, flat, atol=1e-9)

    def test_jerk_and_snap_vanish_at_the_flip_ends(self):
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory()
        for t in (flip._t2, flip._t3):
            _, _, _, jerk, snap = flip.get_full(t)
            np.testing.assert_allclose(jerk, 0.0, atol=1e-9)
            np.testing.assert_allclose(snap, 0.0, atol=1e-9)

    @pytest.mark.parametrize("n_loops", [1, 3])
    def test_climb_rises_monotonically_and_the_run_never_dips_below_start(self, n_loops):
        """The climb ends at the Hermite displacement of its boundary conditions,
        so ż never reverses."""
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory(n_loops=n_loops)
        ts = np.linspace(flip._t1, flip._t2, 2001)
        assert min(flip.get(t)[1][2] for t in ts) >= -1e-9
        whole = np.linspace(0.0, flip.duration, 4001)
        assert min(flip.get(t)[0][2] for t in whole) >= flip.get(0.0)[0][2] - 1e-9

    def test_flip_ends_at_rest_and_the_run_ends_at_rest(self):
        """The launch velocity is chosen so the ramped flip ends with v = 0."""
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory()
        np.testing.assert_allclose(flip.get(flip._t3)[1], 0.0, atol=1e-9)
        p, v, a, j, s = flip.get_full(flip.duration)
        np.testing.assert_allclose(p, flip.end_position)
        for row in (v, a, j, s):
            np.testing.assert_allclose(row, 0.0)

    def test_stays_inside_the_m1_actuator_limits(self):
        """Peak thrust ``m‖a + g e₃‖`` against the 20 N clip, and peak body rate."""
        from udaan.core.defaults import DEFAULT_M1_QUAD_MASS, DEFAULT_MAX_THRUST, GRAVITY
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory()
        ts = np.linspace(0.0, flip.duration, 6001)
        acc = np.array([flip.get(t)[2] for t in ts])
        thrust = DEFAULT_M1_QUAD_MASS * np.linalg.norm(acc + [0.0, 0.0, GRAVITY], axis=1)
        assert thrust.max() <= DEFAULT_MAX_THRUST
        assert (
            max(flip._theta(t - flip._t2)[1] for t in ts if flip._t2 < t < flip._t3) <= flip._omega
        )

    def test_full_jet_rows_are_derivatives_of_each_other(self):
        from udaan.utils.trajectory import FlipTrajectory

        flip = FlipTrajectory()
        for t in (flip._t1 + 1.0, flip._t2 + 0.05, flip._t2 + 0.3, flip._t3 + 0.7):
            full = flip.get_full(t)
            for row in range(4):
                np.testing.assert_allclose(
                    full[row + 1], _central_difference(flip.get_full, t, row, h=1e-5), atol=1e-4
                )

    def test_rejects_a_ramp_longer_than_the_loops(self):
        from udaan.utils.trajectory import FlipTrajectory

        with pytest.raises(ValueError, match="flip_ramp_time"):
            FlipTrajectory(flip_duration=0.5, flip_ramp_time=0.6)


class TestSpiralTrajectory:
    """The helix phase is time-warped, so the reference ramps from rest to
    ``(v_x, rω, 0)`` rather than stepping to it at both ends."""

    @staticmethod
    def _boundaries(spiral):
        return (spiral._t1, spiral._t1 + spiral._t_ramp, spiral._t2 - spiral._t_ramp, spiral._t2)

    def test_reference_is_continuous_through_snap_at_every_boundary(self):
        from udaan.utils.trajectory import SpiralTrajectory

        spiral = SpiralTrajectory()
        for boundary in self._boundaries(spiral):
            before, after = spiral.get_full(boundary - 1e-9), spiral.get_full(boundary + 1e-9)
            for row in range(5):
                np.testing.assert_allclose(before[row], after[row], atol=1e-5)

    def test_jet_is_continuous_through_its_sixth_derivative_at_every_boundary(self):
        """The order M3's and M4's payload flat maps read."""
        from udaan.utils.trajectory import SpiralTrajectory

        spiral = SpiralTrajectory()
        for boundary in self._boundaries(spiral):
            before, after = spiral.get_jet(boundary - 1e-9, 6), spiral.get_jet(boundary + 1e-9, 6)
            near = [np.abs(spiral.get_jet(boundary + d, 7)).max(axis=1) for d in (-1e-3, 1e-3)]
            scale = np.maximum(1.0, np.maximum(*near))
            for row in range(7):
                # Row k changes by at most row k+1 times the 2e-9 window.
                assert np.abs(before[row] - after[row]).max() <= 1e-8 * scale[min(row + 1, 7)]

    def test_get_jet_agrees_with_get_full_and_with_itself(self):
        from udaan.utils.trajectory import SpiralTrajectory

        spiral = SpiralTrajectory()
        for t in (0.2, spiral._t1 + 0.4, 3.0, spiral._t2 - 0.4, spiral._t2 + 0.2):
            np.testing.assert_allclose(
                spiral.get_jet(t, 4), np.array(spiral.get_full(t)), atol=1e-12
            )
        t, h = spiral._t1 + 0.6, 1e-5
        for row in range(6):
            fd = (spiral.get_jet(t + h, 7)[row] - spiral.get_jet(t - h, 7)[row]) / (2 * h)
            np.testing.assert_allclose(spiral.get_jet(t, 7)[row + 1], fd, rtol=1e-5, atol=1e-5)

    def test_full_jet_rows_are_derivatives_of_each_other(self):
        from udaan.utils.trajectory import SpiralTrajectory

        spiral = SpiralTrajectory()
        ramp_mid = spiral._t1 + 0.5 * spiral._t_ramp
        for t in (ramp_mid, 0.5 * (spiral._t1 + spiral._t2), spiral._t2 - 0.3):
            full = spiral.get_full(t)
            for row in range(4):
                np.testing.assert_allclose(
                    full[row + 1], _central_difference(spiral.get_full, t, row, h=1e-5), atol=1e-5
                )

    def test_completes_its_loops_and_ends_at_rest(self):
        from udaan.utils.trajectory import SpiralTrajectory

        spiral = SpiralTrajectory()
        p, v, a, j, s = spiral.get_full(spiral._t2 - 1e-9)
        np.testing.assert_allclose(p, spiral.end_position, atol=1e-6)
        for row in (v, a, j, s):
            np.testing.assert_allclose(row, np.zeros(3), atol=1e-5)

    def test_rejects_a_ramp_longer_than_the_loops(self):
        from udaan.utils.trajectory import SpiralTrajectory

        with pytest.raises(ValueError, match="ramp_time"):
            SpiralTrajectory(radius=1.0, speed=2 * np.pi, n_loops=1, ramp_time=1.5)

    def test_zero_ramp_is_the_unramped_helix_and_stays_finite_at_its_end(self):
        from udaan.utils.trajectory import SpiralTrajectory

        spiral = SpiralTrajectory(ramp_time=0.0)
        mid = 0.5 * (spiral._t1 + spiral._t2)
        np.testing.assert_allclose(np.linalg.norm(spiral.get(mid)[1][1:]), spiral._r * spiral._w)
        for t in (spiral._t2 - 1e-12, spiral._t2, spiral._t2 + 1e-12):
            assert all(np.isfinite(row).all() for row in spiral.get_full(t))


class TestStepSetpoint:
    def test_holds_then_steps_with_no_feedforward(self):
        from udaan.utils.trajectory import StepSetpoint

        step = StepSetpoint([0.0, 0.0, 1.5], [1.0, 1.0, 2.0], step_time=2.0)
        np.testing.assert_array_equal(step.get(1.999)[0], [0.0, 0.0, 1.5])
        np.testing.assert_array_equal(step.get(2.0)[0], [1.0, 1.0, 2.0])
        assert all(np.array_equal(row, np.zeros(3)) for t in (0.0, 5.0) for row in step.get(t)[1:])
        # No get_full: a step has no honest derivatives, so simulate() runs no feedforward.
        assert not hasattr(step, "get_full")


MISSION = [
    (0, 0, 0.026),
    (0, 0, 1.5),
    (2, 0, 1.5),
    (2, 2, 2.5),
    (0, 2, 1.5),
    (0, 0, 1.5),
    (0, 0, 0.026),
]


class TestWaypointTrajectory:
    @staticmethod
    def _mission():
        from udaan.utils.trajectory import WaypointTrajectory

        return WaypointTrajectory(MISSION)

    def test_derivative_rows_are_time_derivatives(self):
        wm = self._mission()
        for t in np.linspace(0.05, 0.95, 37) * wm.duration:
            full = wm.get_full(t)
            for row in range(4):
                np.testing.assert_allclose(
                    full[row + 1], _central_difference(wm.get_full, t, row, h=1e-5), atol=1e-4
                )

    def test_rests_at_every_waypoint_and_is_continuous_through_jerk(self):
        """Snap is not: a 7th-order rest-to-rest leg ends with non-zero snap."""
        wm = self._mission()
        for boundary in wm._starts[1:-1]:
            before, after = wm.get_full(boundary - 1e-9), wm.get_full(boundary + 1e-9)
            for row in range(4):
                np.testing.assert_allclose(before[row], after[row], atol=1e-5)
        visited = {tuple(np.round(wm.get(t)[0], 9)) for t in wm._starts[1::2]}
        assert visited == {tuple(map(float, w)) for w in MISSION}
        for t in wm._starts[1::2]:
            for row in wm.get_full(t)[1:4]:
                np.testing.assert_allclose(row, 0.0, atol=1e-9)

    def test_starts_and_ends_on_its_end_waypoints(self):
        wm = self._mission()
        np.testing.assert_allclose(wm.get(0.0)[0], MISSION[0])
        np.testing.assert_allclose(wm.get(wm.duration + 5.0)[0], MISSION[-1])

    def test_needs_two_waypoints(self):
        from udaan.utils.trajectory import WaypointTrajectory

        with pytest.raises(ValueError):
            WaypointTrajectory([(0.0, 0.0, 1.0)])


class TestFigure8Trajectory:
    @staticmethod
    def _boundaries(f8):
        hold, ph = f8._hold, f8._phase
        return (hold, hold + ph.Tr, hold + ph.Tr + ph.Tc, hold + ph.duration)

    def test_derivative_rows_are_time_derivatives(self):
        from udaan.utils.trajectory import Figure8Trajectory

        f8 = Figure8Trajectory()
        for t in np.linspace(0.05, 0.95, 37) * f8.duration:
            full = f8.get_full(t)
            for row in range(4):
                np.testing.assert_allclose(
                    full[row + 1], _central_difference(f8.get_full, t, row, h=1e-5), atol=1e-4
                )

    def test_yaw_rows_are_time_derivatives(self):
        from udaan.utils.trajectory import Figure8Trajectory

        f8 = Figure8Trajectory()
        for t in np.linspace(0.05, 0.95, 37) * f8.duration:
            psi = f8.yaw(t)
            for row in range(2):
                assert psi[row + 1] == pytest.approx(_central_difference(f8.yaw, t, row), abs=1e-5)

    def test_reference_and_yaw_are_continuous_at_every_boundary(self):
        from udaan.utils.trajectory import Figure8Trajectory

        f8 = Figure8Trajectory()
        for boundary in self._boundaries(f8):
            before, after = f8.get_full(boundary - 1e-9), f8.get_full(boundary + 1e-9)
            for row in range(5):
                np.testing.assert_allclose(before[row], after[row], atol=1e-5)
            np.testing.assert_allclose(f8.yaw(boundary - 1e-9), f8.yaw(boundary + 1e-9), atol=1e-6)

    def test_is_a_closed_figure8_about_its_centre(self):
        from udaan.utils.trajectory import Figure8Trajectory

        c = np.array([1.0, -2.0, 3.0])
        f8 = Figure8Trajectory(center=c, amplitude=1.0, yaw_amplitude=0.5)
        ts = np.linspace(0.0, f8.duration, 4001)
        p = np.array([f8.get(t)[0] for t in ts])
        np.testing.assert_allclose(p[0], c, atol=1e-12)
        np.testing.assert_allclose(p[-1], c, atol=1e-9)
        np.testing.assert_allclose(p[:, 2], c[2])
        assert np.abs(p[:, 0] - c[0]).max() == pytest.approx(1.0, abs=1e-4)
        assert np.abs(p[:, 1] - c[1]).max() == pytest.approx(0.5, abs=1e-4)
        assert max(abs(f8.yaw(t)[0]) for t in ts) == pytest.approx(0.5, abs=1e-4)

    def test_ramp_longer_than_the_loops_is_refused(self):
        from udaan.utils.trajectory import Figure8Trajectory

        with pytest.raises(ValueError):
            Figure8Trajectory(omega=1.0, n_loops=1, ramp_time=10.0)
