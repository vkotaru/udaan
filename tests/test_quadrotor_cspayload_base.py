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


class TestTangency:
    """``q · ω`` is a conserved quantity of the continuous dynamics.

    M2 had no such test, which is how the ZOH tangency drift survived here
    while M3-M6 each measured and documented it. The two models agree to five
    significant figures at ``nQ = 1`` — 3.1517e-03 at ``dt = 1/200`` over 2 s
    of thrusting flight, before the fix — precisely because they share the
    update ordering, which is why this belongs in both suites.
    """

    @staticmethod
    def _worst_drift(dt):
        mdl = QuadrotorCsPayloadBase(input="wrench", dt=dt)
        q0 = np.array([0.0, 0.0, -1.0])
        tilt = 0.35
        mdl.reset(
            payload_position=np.array([0.0, 0.0, 10.0]),
            cable_attitude=S2(q0),
            cable_angular_velocity=TS2(np.cross(q0, np.array([0.3, -0.2, 0.15]))),
            orientation=SO3.from_two_vectors(
                np.array([np.sin(tilt), 0.0, np.cos(tilt)]), np.array([1.0, 0.0, 0.0])
            ),
        )
        wrench = np.zeros(4)
        wrench[0] = 1.5 * (mdl.mass + mdl.payload_mass) * 9.81
        worst = 0.0
        for _ in range(int(round(2.0 / dt))):
            mdl.step(wrench)
            worst = max(
                worst,
                abs(
                    float(
                        np.dot(
                            np.asarray(mdl.state.cable_attitude),
                            np.asarray(mdl.state.cable_angular_velocity),
                        )
                    )
                ),
            )
        return worst

    @pytest.mark.parametrize("dt", [1.0 / 200.0, 1.0 / 400.0])
    def test_cable_rate_stays_tangent(self, dt):
        worst = self._worst_drift(dt)
        assert worst < 1e-12, f"q·ω drifted to {worst:.3e} at dt={dt}"

    def test_cable_direction_stays_on_the_sphere(self):
        mdl = QuadrotorCsPayloadBase(input="wrench")
        mdl.reset(payload_position=np.array([0.0, 0.0, 10.0]))
        for _ in range(400):
            mdl.step(np.array([mdl.mass * 9.81, 0.0, 0.0, 0.0]))
            norm = float(np.linalg.norm(np.asarray(mdl.state.cable_attitude)))
            assert abs(norm - 1.0) < 1e-12


class TestParameterValidation:
    """Non-physical parameters are refused where they are set, not in ``_zoh``.

    ``payload_mass = 0`` used to be accepted silently. With ``m_L = 0`` the
    payload acceleration's ``m_Q + m_L`` denominator is just ``m_Q``, so M2
    limped on integrating a model that is not the one asked for; the
    multi-quadrotor form of the same assignment makes the mass matrix singular
    and raises a ``LinAlgError`` out of the middle of the integrator instead.
    """

    @pytest.mark.parametrize("attribute", ["mass", "payload_mass", "cable_length"])
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_non_positive_is_refused(self, attribute, value):
        mdl = QuadrotorCsPayloadBase()
        with pytest.raises(ValueError, match="strictly positive"):
            setattr(mdl, attribute, value)

    @pytest.mark.parametrize("attribute", ["mass", "payload_mass", "cable_length"])
    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_non_finite_is_refused(self, attribute, value):
        mdl = QuadrotorCsPayloadBase()
        with pytest.raises(ValueError, match="finite"):
            setattr(mdl, attribute, value)

    def test_a_refused_assignment_leaves_the_model_untouched(self):
        mdl = QuadrotorCsPayloadBase()
        before = mdl.payload_mass
        with pytest.raises(ValueError):
            mdl.payload_mass = -1.0
        assert mdl.payload_mass == before

    def test_valid_assignment_still_works(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.payload_mass = 0.35
        mdl.cable_length = 1.4
        mdl.mass = 0.9
        assert (mdl.payload_mass, mdl.cable_length, mdl.mass) == (0.35, 1.4, 0.9)

    def test_asymmetric_inertia_is_refused(self):
        mdl = QuadrotorCsPayloadBase()
        with pytest.raises(ValueError, match="symmetric"):
            mdl.inertia = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_indefinite_inertia_is_refused(self):
        mdl = QuadrotorCsPayloadBase()
        with pytest.raises(ValueError, match="positive definite"):
            mdl.inertia = np.diag([1.0, -1.0, 1.0])

    def test_diagonal_inertia_is_still_accepted(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.inertia = np.array([0.01, 0.02, 0.03])
        np.testing.assert_allclose(mdl.inertia, np.diag([0.01, 0.02, 0.03]))
