"""Tests for QuadrotorCsPayloadMujoco — unit + integration across cable models."""

import numpy as np
import pytest

from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadMujoco

# Cable model variants. (cable_model kwarg, descriptive name, payload tolerance [m])
# Tendon's spring-damper has a known steady-state offset, so its tolerance is looser.
CABLE_MODELS = [
    ("links", "links", 0.10),
    ("tendon", "tendon", 0.30),
]

# (payload start, payload target, description)
TRAJECTORIES = [
    (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), "hover at setpoint"),
    (np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0]), "climb 0.5m"),
    (np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 1.0]), "lateral + climb"),
    (np.array([1.0, 1.0, 0.5]), np.array([0.0, 0.0, 1.0]), "diagonal + climb"),
]


class TestMujocoCsPayloadUnit:
    """Basic instantiation."""

    @pytest.mark.parametrize("cable_model", ["links", "tendon"])
    def test_instantiate(self, cable_model):
        mdl = QuadrotorCsPayloadMujoco(render=False, cable_model=cable_model)
        assert mdl.state is not None

    def test_state_types(self):
        mdl = QuadrotorCsPayloadMujoco(render=False, cable_model="links")
        mdl.reset()
        assert isinstance(mdl.state.orientation, SO3)
        assert isinstance(mdl.state.angular_velocity, TSO3)
        assert isinstance(mdl.state.cable_attitude, S2)
        assert isinstance(mdl.state.cable_angular_velocity, TS2)

    def test_invalid_cable_model_raises(self):
        with pytest.raises(ValueError):
            QuadrotorCsPayloadMujoco(render=False, cable_model="bogus")


@pytest.mark.integration
@pytest.mark.mujoco
class TestMujocoCsPayloadIntegration:
    """Closed-loop payload tracking for each cable model variant."""

    SIM_TIME = 6.0

    @pytest.mark.parametrize("cable_model,name,tol", CABLE_MODELS)
    @pytest.mark.parametrize("start,target,desc", TRAJECTORIES)
    def test_payload_converges(self, cable_model, name, tol, start, target, desc):
        mdl = QuadrotorCsPayloadMujoco(render=False, cable_model=cable_model)
        mdl._payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf=self.SIM_TIME, payload_position=start)
        pos_err = np.linalg.norm(mdl.state.payload_position - target)
        assert pos_err < tol, f"[{name}, {desc}] Payload position error {pos_err:.4f}m > {tol}m"
