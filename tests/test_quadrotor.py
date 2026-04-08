"""Tests for base Quadrotor and QuadrotorCSPayload models."""

import numpy as np

from udaan.models.base.quadrotor import Quadrotor
from udaan.models.base.quadrotor_cspayload import QuadrotorCSPayload


class TestQuadrotor:
    def test_instantiate(self):
        mdl = Quadrotor(render=False)
        assert mdl.state is not None

    def test_reset(self):
        mdl = Quadrotor(render=False)
        mdl.reset(position=np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(mdl.state.position, np.array([1.0, 2.0, 3.0]))

    def test_step_accel_input(self):
        mdl = Quadrotor(render=False)
        mdl.reset()
        pos_before = mdl.state.position.copy()
        mdl.step(np.zeros(3))
        # Should have moved (gravity pulls down)
        assert not np.allclose(mdl.state.position, pos_before)

    def test_step_updates_time(self):
        mdl = Quadrotor(render=False)
        mdl.reset()
        assert mdl.t == 0.0
        mdl.step(np.zeros(3))
        assert mdl.t > 0.0

    def test_simulate(self):
        mdl = Quadrotor(render=False)
        mdl.simulate(tf=1.0, position=np.array([0.0, 0.0, 1.0]))
        assert mdl.t >= 1.0

    def test_mass_property(self):
        mdl = Quadrotor(render=False)
        original = mdl.mass
        mdl.mass = 2.0
        assert mdl.mass == 2.0
        assert mdl.mass != original

    def test_state_shape(self):
        mdl = Quadrotor(render=False)
        mdl.reset()
        assert mdl.state.position.shape == (3,)
        assert mdl.state.velocity.shape == (3,)
        assert mdl.state.orientation.shape == (3, 3)
        assert mdl.state.angular_velocity.shape == (3,)


class TestQuadrotorCSPayload:
    def test_instantiate(self):
        mdl = QuadrotorCSPayload(render=False)
        assert mdl.state is not None

    def test_reset(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(mdl.state.position, np.array([0.0, 0.0, 2.0]))

    def test_step_updates_state(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        pos_before = mdl.state.payload_position.copy()
        mdl.step(np.zeros(3))
        assert not np.allclose(mdl.state.payload_position, pos_before)

    def test_simulate(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.simulate(tf=1.0, position=np.array([0.0, 0.0, 2.0]))
        assert mdl.t >= 1.0

    def test_properties(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.qrotor_mass = 1.5
        assert mdl.qrotor_mass == 1.5
        mdl.payload_mass = 0.3
        assert mdl.payload_mass == 0.3
        mdl.cable_length = 2.0
        assert mdl.cable_length == 2.0

    def test_state_shape(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.reset()
        assert mdl.state.position.shape == (3,)
        assert mdl.state.payload_position.shape == (3,)
        assert mdl.state.cable_attitude.shape == (3,)
        assert mdl.state.orientation.shape == (3, 3)

    def test_cable_attitude_is_unit(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.reset()
        norm = np.linalg.norm(mdl.state.cable_attitude)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
