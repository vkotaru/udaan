import numpy as np
import pytest

from udaan.models.base.floating_pointmass import FloatingPointmass
from udaan.models.base.multi_pointmass_suspended_payload import (
    MultiPointmassSuspendedPayload,
)
from udaan.models.base.pointmass_suspended_payload import PointmassSuspendedPayload
from udaan.models.base.s2_pendulum import S2Pendulum


class TestFloatingPointmass:
    def test_init_defaults(self):
        mdl = FloatingPointmass()
        assert mdl.get_state_size() == 6
        assert mdl.get_action_size() == 3

    def test_reset(self):
        mdl = FloatingPointmass()
        mdl.reset()
        np.testing.assert_allclose(mdl.state.position, np.zeros(3))
        np.testing.assert_allclose(mdl.state.velocity, np.zeros(3))

    def test_step_gravity(self):
        mdl = FloatingPointmass()
        mdl.reset()
        mdl.state.position = np.array([0.0, 0.0, 1.0])
        pos_before = mdl.state.position.copy()
        mdl.step(np.zeros(3))
        # gravity pulls down
        assert mdl.state.position[2] < pos_before[2]

    def test_step_thrust_counteracts_gravity(self):
        mdl = FloatingPointmass()
        mdl.reset()
        mdl.state.position = np.array([0.0, 0.0, 1.0])
        # apply force equal to gravity
        hover_force = np.array([0.0, 0.0, mdl.mass * 9.81])
        mdl.step(hover_force)
        # should stay roughly at same height
        np.testing.assert_allclose(mdl.state.position[2], 1.0, atol=1e-2)

    def test_get_rand_init_state(self):
        mdl = FloatingPointmass()
        s = mdl.get_rand_init_state()
        assert "position" in s
        assert "velocity" in s
        assert s["position"].shape == (3,)


class TestS2Pendulum:
    def test_init_defaults(self):
        mdl = S2Pendulum()
        assert mdl.get_state_size() == 6
        assert mdl.get_action_size() == 3

    def test_reset(self):
        mdl = S2Pendulum()
        mdl.reset()
        np.testing.assert_allclose(np.linalg.norm(mdl.state.attitude), 1.0, atol=1e-10)
        np.testing.assert_allclose(mdl.state.angular_velocity, np.zeros(3), atol=1e-10)

    def test_attitude_stays_unit_after_step(self):
        mdl = S2Pendulum()
        mdl.reset()
        for _ in range(100):
            mdl.step(np.random.default_rng(42).standard_normal(3) * 0.1)
        np.testing.assert_allclose(np.linalg.norm(mdl.state.attitude), 1.0, atol=1e-6)

    def test_gravity_pulls_to_equilibrium(self):
        mdl = S2Pendulum()
        # perturb attitude slightly from -e3
        perturbed = np.array([0.1, 0.1, -np.sqrt(1 - 0.02)])
        perturbed /= np.linalg.norm(perturbed)
        mdl.reset(attitude=perturbed)
        # step with no torque many times (should oscillate around -e3)
        for _ in range(500):
            mdl.step(np.zeros(3))
        assert np.linalg.norm(mdl.state.attitude) == pytest.approx(1.0, abs=1e-6)


class TestPointmassSuspendedPayload:
    def test_init_defaults(self):
        mdl = PointmassSuspendedPayload()
        assert mdl.get_state_size() == 12
        assert mdl.get_action_size() == 3

    def test_reset(self):
        mdl = PointmassSuspendedPayload()
        mdl.reset()
        np.testing.assert_allclose(np.linalg.norm(mdl.state.cable_attitude), 1.0, atol=1e-10)

    def test_cable_stays_unit_after_steps(self):
        mdl = PointmassSuspendedPayload()
        mdl.reset()
        for _ in range(100):
            mdl.step(np.zeros(3))
        np.testing.assert_allclose(np.linalg.norm(mdl.state.cable_attitude), 1.0, atol=1e-6)

    def test_quadrotor_position_derived(self):
        mdl = PointmassSuspendedPayload()
        mdl.reset()
        mdl.state.payload_position = np.array([1.0, 2.0, 3.0])
        mdl.state.cable_attitude = np.array([0.0, 0.0, -1.0])
        xQ = mdl.quadrotor_position
        # quad should be above payload by cable_length
        expected = mdl.state.payload_position - mdl.cable_length * mdl.state.cable_attitude
        np.testing.assert_allclose(xQ, expected, atol=1e-10)

    def test_gravity_moves_payload(self):
        mdl = PointmassSuspendedPayload()
        mdl.reset()
        mdl.state.payload_position = np.array([0.0, 0.0, 5.0])
        pos_before = mdl.state.payload_position.copy()
        mdl.step(np.zeros(3))
        # payload should fall under gravity
        assert mdl.state.payload_position[2] < pos_before[2]

    def test_err_q(self):
        q = np.array([0.0, 0.0, -1.0])
        qd = np.array([0.0, 0.0, -1.0])
        err = PointmassSuspendedPayload.err_q(q, qd)
        np.testing.assert_allclose(err, np.zeros(3), atol=1e-10)


class TestMultiPointmassSuspendedPayload:
    def test_init_defaults(self):
        mdl = MultiPointmassSuspendedPayload(nQ=2)
        assert mdl.get_state_size() == 6 + 6 * 2
        assert mdl.get_action_size() == 3 * 2

    def test_reset(self):
        mdl = MultiPointmassSuspendedPayload(nQ=3)
        mdl.reset()
        for i in range(3):
            q_i = mdl.state.cable_attitudes[:, i]
            np.testing.assert_allclose(np.linalg.norm(q_i), 1.0, atol=1e-10)

    def test_cable_stays_unit_after_steps(self):
        mdl = MultiPointmassSuspendedPayload(nQ=2)
        mdl.reset()
        for _ in range(50):
            mdl.step(np.zeros((3, 2)))
        for i in range(2):
            q_i = mdl.state.cable_attitudes[:, i]
            np.testing.assert_allclose(np.linalg.norm(q_i), 1.0, atol=1e-4)

    def test_step_accepts_flat_input(self):
        mdl = MultiPointmassSuspendedPayload(nQ=2)
        mdl.reset()
        # should accept (6,) flat vector as well as (3,2) matrix
        mdl.step(np.zeros(6))

    def test_gravity_moves_payload(self):
        mdl = MultiPointmassSuspendedPayload(nQ=2)
        mdl.reset()
        mdl.state.payload_position = np.array([0.0, 0.0, 5.0])
        pos_before = mdl.state.payload_position.copy()
        mdl.step(np.zeros((3, 2)))
        assert mdl.state.payload_position[2] < pos_before[2]
