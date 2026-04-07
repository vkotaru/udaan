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
