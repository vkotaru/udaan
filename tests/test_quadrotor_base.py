"""Tests for QuadrotorBase — unit + integration."""

import numpy as np
import pytest

from udaan.manif import SO3, TSO3
from udaan.models.quadrotor import QuadrotorBase

# All valid (input_type, force_type) combinations
INPUT_FORCE_COMBOS = [
    ({}, "accel+wrench (default)"),
    ({"force": "prop_forces"}, "accel+propforces"),
    ({"input": "wrench"}, "wrench+wrench"),
    ({"input": "wrench", "force": "prop_forces"}, "wrench+propforces"),
    ({"input": "prop_forces"}, "propforces+wrench"),
    ({"input": "prop_forces", "force": "prop_forces"}, "propforces+propforces"),
]

# (start, target, description)
TRAJECTORIES = [
    (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), "hover at setpoint"),
    (np.array([0.0, 0.0, 0.8]), np.array([0.0, 0.0, 1.0]), "0.2m below"),
    (np.array([0.0, 0.0, 0.2]), np.array([0.0, 0.0, 1.0]), "climb from 0.2m"),
    (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), "lateral + climb"),
    (np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), "diagonal + climb"),
]


class TestQuadrotorUnit:
    """Basic instantiation, reset, properties."""

    def test_instantiate(self):
        mdl = QuadrotorBase()
        assert mdl.state is not None

    def test_reset(self):
        mdl = QuadrotorBase()
        mdl.reset(position=np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(mdl.state.position, np.array([1.0, 2.0, 3.0]))

    def test_step_updates_time(self):
        mdl = QuadrotorBase()
        mdl.reset()
        assert mdl.t == 0.0
        mdl.step(np.zeros(3))
        assert mdl.t > 0.0

    def test_mass_property(self):
        mdl = QuadrotorBase()
        mdl.mass = 2.0
        assert mdl.mass == 2.0

    def test_state_types(self):
        mdl = QuadrotorBase()
        mdl.reset()
        assert isinstance(mdl.state.orientation, SO3)
        assert isinstance(mdl.state.angular_velocity, TSO3)

    def test_state_types_after_step(self):
        """State types should be preserved after dynamics integration."""
        mdl = QuadrotorBase()
        mdl.reset()
        mdl.step(np.zeros(3))
        assert isinstance(mdl.state.orientation, SO3)
        assert isinstance(mdl.state.angular_velocity, TSO3)


@pytest.mark.integration
class TestQuadrotorIntegration:
    """Simulate and verify convergence for all input/force × trajectory combos."""

    SIM_TIME = 5.0
    TOL_POS = 0.05

    @pytest.mark.parametrize("kwargs,mode", INPUT_FORCE_COMBOS)
    @pytest.mark.parametrize("start,target,desc", TRAJECTORIES)
    def test_converge(self, kwargs, mode, start, target, desc):
        """Quadrotor should converge from start to target."""
        mdl = QuadrotorBase(**kwargs)
        mdl._pos_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf=self.SIM_TIME, position=start)
        pos_err = np.linalg.norm(mdl.state.position - target)
        assert pos_err < self.TOL_POS, (
            f"[{mode}, {desc}] Position error {pos_err:.4f}m > {self.TOL_POS}m"
        )
