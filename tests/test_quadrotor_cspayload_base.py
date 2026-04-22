"""Tests for QuadrotorCsPayloadBase — unit + integration."""

import numpy as np
import pytest

from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.quadrotor_cspayload import (
    QuadrotorCsPayloadBase,
    QuadrotorCsPayloadState,
)

# (payload start, payload target, description)
TRAJECTORIES = [
    (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), "hover at setpoint"),
    (np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0]), "climb 0.5m"),
    (np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 1.0]), "lateral + climb"),
    (np.array([1.0, 1.0, 0.5]), np.array([0.0, 0.0, 1.0]), "diagonal + climb"),
]


class TestQuadrotorCsPayloadUnit:
    """Basic instantiation, reset, properties."""

    def test_instantiate(self):
        mdl = QuadrotorCsPayloadBase()
        assert mdl.state is not None
        assert isinstance(mdl.state, QuadrotorCsPayloadState)

    def test_reset_payload_position(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(mdl.state.payload_position, np.array([0.0, 0.0, 1.0]))
        # Quadrotor should sit one cable length above the payload (q = -e3).
        np.testing.assert_allclose(mdl.state.position, np.array([0.0, 0.0, 1.0 + mdl.cable_length]))

    def test_reset_does_not_mutate_caller_array(self):
        # Regression: the integrator does in-place += on payload_position, so
        # reset() must copy ndarray kwargs; otherwise a shared start array
        # drifts across repeated sims (e.g. EVAL_STARTS in CMA-ES tuning).
        start = np.array([1.0, 1.0, 0.5])
        snapshot = start.copy()
        mdl = QuadrotorCsPayloadBase()
        mdl.reset(payload_position=start)
        mdl._payload_controller.setpoint = lambda t: (np.zeros(3), np.zeros(3), np.zeros(3))
        for _ in range(50):
            u = mdl._payload_controller.compute(mdl.t, mdl.state)
            mdl.step(u)
        # Internal state must have advanced; caller's array must not have.
        assert not np.allclose(mdl.state.payload_position, snapshot)
        np.testing.assert_array_equal(start, snapshot)

    def test_reset_quad_position(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(mdl.state.position, np.array([0.0, 0.0, 2.0]))
        # Payload should hang one cable length below the quadrotor (q = -e3).
        np.testing.assert_allclose(
            mdl.state.payload_position, np.array([0.0, 0.0, 2.0 - mdl.cable_length])
        )

    def test_state_types(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.reset()
        assert isinstance(mdl.state.orientation, SO3)
        assert isinstance(mdl.state.angular_velocity, TSO3)
        assert isinstance(mdl.state.cable_attitude, S2)
        assert isinstance(mdl.state.cable_angular_velocity, TS2)

    def test_cable_attitude_unit(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.reset()
        np.testing.assert_allclose(
            np.linalg.norm(np.asarray(mdl.state.cable_attitude)), 1.0, atol=1e-10
        )

    def test_step_advances_time(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.reset()
        mdl.step(np.array([0.0, 0.0, mdl.mass * 9.81]))
        assert mdl.t > 0.0


@pytest.mark.integration
class TestQuadrotorCsPayloadIntegration:
    """Closed-loop payload tracking via QuadCSPayloadController."""

    SIM_TIME = 6.0
    TOL_POS = 0.05

    @pytest.mark.parametrize("start,target,desc", TRAJECTORIES)
    def test_payload_converges(self, start, target, desc):
        mdl = QuadrotorCsPayloadBase()
        mdl._payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf=self.SIM_TIME, payload_position=start)
        pos_err = np.linalg.norm(mdl.state.payload_position - target)
        assert pos_err < self.TOL_POS, (
            f"[{desc}] Payload position error {pos_err:.4f}m > {self.TOL_POS}m"
        )
