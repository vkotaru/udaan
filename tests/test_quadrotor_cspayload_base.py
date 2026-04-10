"""Tests for base QuadrotorCSPayload model."""

import numpy as np
import pytest

from udaan.models.base.quadrotor_cspayload import QuadrotorCSPayload


class TestQuadrotorCSPayloadUnit:
    """Basic instantiation, reset, properties."""

    def test_instantiate(self):
        mdl = QuadrotorCSPayload(render=False)
        assert mdl.state is not None

    def test_reset(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(mdl.state.position, np.array([0.0, 0.0, 2.0]))

    def test_properties(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.qrotor_mass = 1.5
        assert mdl.qrotor_mass == 1.5
        mdl.payload_mass = 0.3
        assert mdl.payload_mass == 0.3
        mdl.cable_length = 2.0
        assert mdl.cable_length == 2.0

    def test_cable_attitude_is_unit(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.reset()
        norm = np.linalg.norm(mdl.state.cable_attitude)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)


@pytest.mark.xfail(reason="Attitude controller caller migration in progress")
@pytest.mark.integration
class TestQuadrotorCSPayloadIntegration:
    """Simulate and verify convergence."""

    TARGET = np.array([0.0, 0.0, 1.0])
    SIM_TIME = 5.0
    TOL_POS = 0.2

    def test_simulate_hover(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.simulate(tf=self.SIM_TIME, position=np.array([0.0, 0.0, 2.0]))
        pos_err = np.linalg.norm(mdl.state.payload_position - self.TARGET)
        assert pos_err < self.TOL_POS, f"Payload position error {pos_err:.4f}m"

    def test_step_updates_state(self):
        mdl = QuadrotorCSPayload(render=False)
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        pos_before = mdl.state.payload_position.copy()
        mdl.step(np.zeros(3))
        assert not np.allclose(mdl.state.payload_position, pos_before)
