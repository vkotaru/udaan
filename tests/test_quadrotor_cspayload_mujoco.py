"""Tests for MuJoCo QuadrotorCSPayload model."""

import numpy as np
import pytest

import udaan as U


class TestMujocoCSPayloadUnit:
    """Basic instantiation."""

    def test_instantiate_tendon(self):
        mdl = U.models.mujoco.QuadrotorCSPayload(render=False, model="tendon")
        assert mdl.state is not None

    def test_instantiate_links(self):
        mdl = U.models.mujoco.QuadrotorCSPayload(render=False, model="links")
        assert mdl.state is not None


@pytest.mark.xfail(reason="Attitude controller caller migration in progress")
@pytest.mark.integration
class TestMujocoCSPayloadIntegration:
    """Simulate and verify convergence."""

    SIM_TIME = 5.0
    TOL_POS = 0.2

    def test_tendon_hover(self):
        mdl = U.models.mujoco.QuadrotorCSPayload(render=False, model="tendon")
        mdl.simulate(tf=self.SIM_TIME, payload_position=np.array([0.0, 0.0, 0.5]))
        pos_err = np.linalg.norm(mdl.state.payload_position - np.array([0.0, 0.0, 1.0]))
        assert pos_err < self.TOL_POS, f"Payload position error {pos_err:.4f}m"

    def test_links_hover(self):
        mdl = U.models.mujoco.QuadrotorCSPayload(render=False, model="links")
        mdl.simulate(tf=self.SIM_TIME, payload_position=np.array([0.0, 0.0, 0.5]))
        pos_err = np.linalg.norm(mdl.state.payload_position - np.array([0.0, 0.0, 1.0]))
        assert pos_err < self.TOL_POS, f"Payload position error {pos_err:.4f}m"
