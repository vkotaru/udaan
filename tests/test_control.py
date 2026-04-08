"""Tests for udaan.control — controller instantiation and output dimensions."""

import numpy as np

from udaan.control import Gains, PDController
from udaan.control.quadrotor import (
    DirectPropellerForceController,
    GeometricAttitudeController,
    PositionPDController,
)
from udaan.control.quadrotor_cspayload import QuadCSPayloadController


class TestGains:
    def test_default(self):
        g = Gains()
        np.testing.assert_allclose(g.kp, np.zeros(3))
        np.testing.assert_allclose(g.kd, np.zeros(3))
        np.testing.assert_allclose(g.ki, np.zeros(3))

    def test_custom(self):
        g = Gains(kp=np.ones(3), kd=2 * np.ones(3))
        np.testing.assert_allclose(g.kp, np.ones(3))
        np.testing.assert_allclose(g.kd, 2 * np.ones(3))


class TestPDController:
    def test_compute_output_shape(self):
        ctrl = PDController(kp=np.ones(3), kd=np.ones(3))
        t = 0.0
        state = (np.array([1.0, 0.0, 0.0]), np.zeros(3))
        u = ctrl.compute(t, state)
        assert u.shape == (3,)

    def test_at_setpoint_gives_zero(self):
        sp = np.array([0.0, 0.0, 1.0])
        ctrl = PDController(kp=np.ones(3), kd=np.ones(3))
        u = ctrl.compute(0.0, (sp, np.zeros(3)))
        np.testing.assert_allclose(u, np.zeros(3), atol=1e-15)


class TestPositionPDController:
    def test_output_shape(self):
        ctrl = PositionPDController(mass=1.0)
        u = ctrl.compute(0.0, (np.zeros(3), np.zeros(3)))
        assert u.shape == (3,)


class TestGeometricAttitudeController:
    def test_output_shape(self):
        inertia = np.diag([0.01, 0.01, 0.02])
        ctrl = GeometricAttitudeController(inertia=inertia)
        R = np.eye(3)
        Omega = np.zeros(3)
        thrust_force = np.array([0.0, 0.0, 9.81])
        f, M = ctrl.compute(0.0, (R, Omega), thrust_force)
        assert np.isscalar(f) or f.shape == ()
        assert M.shape == (3,)

    def test_hover_gives_gravity_thrust(self):
        mass = 1.0
        inertia = np.diag([0.01, 0.01, 0.02])
        ctrl = GeometricAttitudeController(mass=mass, inertia=inertia)
        thrust_force = np.array([0.0, 0.0, 9.81])
        f, M = ctrl.compute(0.0, (np.eye(3), np.zeros(3)), thrust_force)
        assert f > 0


class TestDirectPropellerForceController:
    def test_output_shape(self):
        ctrl = DirectPropellerForceController(mass=1.0, inertia=np.diag([0.01, 0.01, 0.02]))
        state = (np.zeros(3), np.zeros(3), np.eye(3), np.zeros(3))
        u = ctrl.compute(0.0, state)
        assert u.shape == (4,)


class TestQuadCSPayloadController:
    def test_output_shape(self):
        ctrl = QuadCSPayloadController()

        class MockState:
            payload_position = np.array([0.0, 0.0, 0.0])
            payload_velocity = np.zeros(3)
            cable_attitude = np.array([0.0, 0.0, -1.0])

            def dq(self):
                return np.zeros(3)

        u = ctrl.compute(0.0, MockState())
        assert u.shape == (3,)
