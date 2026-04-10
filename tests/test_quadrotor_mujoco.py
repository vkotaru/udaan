"""Tests for MuJoCo Quadrotor model — unit + integration."""

import numpy as np
import pytest

import udaan as U

TRAJECTORIES = [
    (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), "hover at setpoint"),
    (np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0]), "climb from 0.5m"),
    (np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 1.0]), "lateral + climb"),
    (np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), "large offset"),
]


class TestMujocoQuadrotorUnit:
    """Basic instantiation and state."""

    def test_instantiate(self):
        mdl = U.models.mujoco.Quadrotor(render=False)
        assert mdl.state is not None

    def test_reset(self):
        mdl = U.models.mujoco.Quadrotor(render=False)
        mdl.reset(position=np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(mdl.state.position, np.array([1.0, 2.0, 3.0]), atol=0.01)

    def test_time_syncs(self):
        mdl = U.models.mujoco.Quadrotor(render=False)
        mdl.reset()
        mdl.step(np.zeros(3))
        assert mdl.t > 0.0


@pytest.mark.integration
class TestMujocoQuadrotorIntegration:
    """Simulate and verify convergence."""

    SIM_TIME = 5.0
    TOL_POS = 0.15

    @pytest.mark.parametrize("start,target,desc", TRAJECTORIES)
    def test_converge(self, start, target, desc):
        mdl = U.models.mujoco.Quadrotor(render=False)
        mdl._pos_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf=self.SIM_TIME, position=start)
        pos_err = np.linalg.norm(mdl.state.position - target)
        assert pos_err < self.TOL_POS, f"[{desc}] Position error {pos_err:.4f}m > {self.TOL_POS}m"
