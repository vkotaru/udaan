"""Tests for QuadrotorCsPayloadBase — unit + integration."""

import numpy as np
import pytest

from udaan.control.quadrotor_cspayload import QuadCSPayloadController
from udaan.core import defaults as D
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.quadrotor_cspayload import (
    QuadrotorCsPayloadBase,
    QuadrotorCsPayloadState,
)
from udaan.utils.trajectory import CrazyTrajectory, StepSetpoint

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
        # The integrator does in-place += on payload_position, so reset() must
        # copy ndarray kwargs; otherwise a shared start array drifts across
        # repeated sims.
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

    def test_reset_velocity_survives_one_step(self):
        """The airframe's reset velocity is kept: the whole system moves at it."""
        mdl = QuadrotorCsPayloadBase(input="wrench")
        v = np.array([0.3, -0.2, 0.1])
        mdl.reset(position=np.array([0.0, 0.0, 2.0]), velocity=v)
        np.testing.assert_allclose(mdl.state.payload_velocity, v, atol=1e-12)
        mdl.step(np.array([(mdl.mass + mdl.payload_mass) * 9.81, 0.0, 0.0, 0.0]))
        np.testing.assert_allclose(mdl.state.velocity[:2], v[:2], atol=1e-9)
        np.testing.assert_allclose(mdl.state.payload_velocity[:2], v[:2], atol=1e-9)

    def test_reset_derives_the_cable_rate_from_the_two_velocities(self):
        mdl = QuadrotorCsPayloadBase()
        q = np.array([0.0, 0.0, -1.0])
        v_Q, v_L = np.array([0.1, 0.0, 0.0]), np.array([0.6, 0.0, 0.0])
        mdl.reset(velocity=v_Q, payload_velocity=v_L, cable_attitude=S2(q))
        omega = np.cross(q, v_L - v_Q) / mdl.cable_length
        np.testing.assert_allclose(np.asarray(mdl.state.cable_angular_velocity), omega, atol=1e-12)

    def test_reset_refuses_a_payload_velocity_with_a_cable_rate(self):
        mdl = QuadrotorCsPayloadBase()
        with pytest.raises(ValueError, match="not both"):
            mdl.reset(
                payload_velocity=np.zeros(3), cable_angular_velocity=TS2(np.array([0.0, 1.0, 0.0]))
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

    The ZOH substep must keep ``ω`` tangent at the new ``q``; a first-order
    drift would grow with ``dt``, so the check runs at two step sizes.
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

    With ``m_L = 0`` the payload acceleration's ``m_Q + m_L`` denominator is
    just ``m_Q``, so the model would integrate dynamics that are not the ones
    asked for.
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


INPUT_PATHS = [{}, {"input": "wrench"}, {"input": "prop_forces"}]


def attitude_calls_per_step(mdl, tf=0.1):
    """Fly ``simulate(tf)`` and return attitude-controller calls per outer step."""
    calls, steps = [0], [0]
    compute, step = mdl.attitude_controller.compute, mdl.step

    def counted_compute(*args, **kwargs):
        calls[0] += 1
        return compute(*args, **kwargs)

    def counted_step(u, desired_att=None):
        steps[0] += 1
        return step(u, desired_att=desired_att)

    mdl.attitude_controller.compute = counted_compute
    mdl.step = counted_step
    mdl.simulate(tf, payload_position=np.array([0.0, 0.0, 1.0]))
    return calls[0] / steps[0]


def probe_attitude_command(mdl):
    """Record ``∠(R, R_d)`` in degrees and the thrust at every attitude update."""
    controller = mdl.attitude_controller
    compute = controller.compute
    angles, thrusts = [], []

    def probed(t, state, thrust_force, desired_att=None):
        reference = None if desired_att is None else desired_att[0]
        Rd = np.asarray(controller._cmd_accel_to_cmd_att(thrust_force, reference))
        R = np.asarray(state[0])
        cos = np.clip((np.trace(R.T @ Rd) - 1.0) / 2.0, -1.0, 1.0)
        angles.append(float(np.degrees(np.arccos(cos))))
        f, M = compute(t, state, thrust_force, desired_att=desired_att)
        thrusts.append(float(f))
        return f, M

    controller.compute = probed
    return angles, thrusts


def fly_lissajous(mdl, tf=10.0, bind=None):
    """Fly a lissajous about (1, 1, 1.5); return the RMS payload error and the
    mean ∠(R, R_d) over its second half."""
    traj = CrazyTrajectory(tf=tf, center=np.array([1.0, 1.0, 1.5]), ax=1.0, ay=1.0, az=0.75)
    mdl.payload_controller.setpoint = traj.get_full if bind is None else bind(traj)
    angles, _ = probe_attitude_command(mdl)
    errors = []
    step = mdl.step

    def logged_step(u, desired_att=None):
        step(u, desired_att=desired_att)
        errors.append(float(np.linalg.norm(mdl.state.payload_position - traj.get(mdl.t)[0])))

    mdl.step = logged_step
    mdl.simulate(tf, payload_position=traj.get(0.0)[0])
    half = len(errors) // 2
    rms = float(np.sqrt(np.mean(np.square(errors[half:]))))
    return rms, float(np.mean(angles[len(angles) // 2 :]))


def assert_rotor_force_path_matches(make, tf=5.0):
    """``force="prop_forces"`` flies the ``force="wrench"`` run while no rotor clips.

    The payload starts 0.3 m beside its setpoint, where every rotor force stays
    inside its bounds; the round trip through the rotor forces is then exact.
    """
    target = np.array([0.0, 0.0, 1.0])
    final, out_of_bounds = [], [0]
    for force in ("wrench", "prop_forces"):
        mdl = make(force=force)
        to_props = mdl._wrench_to_propforces

        def counted(wrench, _f=to_props, _m=mdl):
            props = _f(wrench)
            lo, hi = _m._prop_min_force, _m._prop_max_force
            out_of_bounds[0] += int(np.any((props < lo - 1e-12) | (props > hi + 1e-12)))
            return props

        mdl._wrench_to_propforces = counted
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf, payload_position=np.array([0.3, 0.0, 1.0]))
        final.append(mdl.state.payload_position.copy())
    assert out_of_bounds[0] == 0
    np.testing.assert_allclose(final[1], final[0], atol=1e-9)


class TestMachine:
    """The analytical model and its controllers fly one machine."""

    def test_default_machine_is_the_m1_airframe_with_the_m2_payload(self):
        mdl = QuadrotorCsPayloadBase()
        assert mdl.mass == D.DEFAULT_M1_QUAD_MASS == 0.75
        np.testing.assert_array_equal(mdl.inertia, D.DEFAULT_M1_QUAD_INERTIA)
        assert mdl.payload_mass == D.DEFAULT_M2_PAYLOAD_MASS == 0.15
        assert mdl.cable_length == D.DEFAULT_M2_CABLE_LENGTH == 1.0

    def test_the_cable_is_massless(self):
        mdl = QuadrotorCsPayloadBase()
        assert mdl.cable_mass == 0.0
        assert mdl.payload_controller.mass_l == mdl.payload_mass

    def test_default_controllers_are_built_for_the_model(self):
        mdl = QuadrotorCsPayloadBase()
        mdl.mass, mdl.payload_mass, mdl.cable_length = 0.8, 0.3, 1.5
        mdl.inertia = np.diag([0.006, 0.006, 0.01])
        mdl._init_default_controllers()
        ctrl = mdl.payload_controller
        assert (ctrl.mass_q, ctrl.mass_l, ctrl.cable_length) == (0.8, 0.3, 1.5)
        np.testing.assert_array_equal(ctrl.inertia, mdl.inertia)
        np.testing.assert_array_equal(mdl.attitude_controller.inertia, mdl.inertia)

    def test_a_heavier_payload_hovers_on_its_setpoint(self):
        """The controller's ``(m_Q + m_L) g`` is the model's, so hover has no offset."""
        mdl = QuadrotorCsPayloadBase()
        mdl.payload_mass = 0.3
        mdl._init_default_controllers()
        target = np.array([0.0, 0.0, 1.0])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(8.0, payload_position=target)
        assert np.linalg.norm(mdl.state.payload_position - target) < 0.005

    def test_controller_keywords_and_sync(self):
        ctrl = QuadCSPayloadController(mass_q=1.0, mass_l=0.4, cable_length=2.0, inertia=[1, 2, 3])
        assert (ctrl.mass_q, ctrl.mass_l, ctrl.cable_length) == (1.0, 0.4, 2.0)
        np.testing.assert_array_equal(ctrl.inertia, np.diag([1.0, 2.0, 3.0]))
        mdl = QuadrotorCsPayloadBase()
        ctrl.sync_from_model(mdl)
        assert (ctrl.mass_q, ctrl.mass_l, ctrl.cable_length) == (0.75, 0.15, 1.0)
        np.testing.assert_array_equal(ctrl.inertia, mdl.inertia)
        default = QuadCSPayloadController()
        assert (default.mass_q, default.mass_l, default.cable_length) == (0.75, 0.15, 1.0)

    @pytest.mark.parametrize("key", ["mass_q", "mass_l", "cable_length"])
    def test_controller_refuses_a_non_positive_parameter(self, key):
        with pytest.raises(ValueError, match="strictly positive"):
            QuadCSPayloadController(**{key: 0.0})


class TestInputPaths:
    """``simulate()`` flies the payload controller on every input path."""

    def test_every_path_flies_the_same_run(self):
        target = np.array([0.0, 0.0, 1.0])
        final = []
        for kwargs in INPUT_PATHS:
            mdl = QuadrotorCsPayloadBase(**kwargs)
            mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
            mdl.simulate(10.0, payload_position=np.array([-1.0, 2.0, 0.5]))
            final.append(mdl.state.payload_position.copy())
        assert np.linalg.norm(final[0] - target) < 0.05
        np.testing.assert_allclose(final[1], final[0], atol=1e-9)
        np.testing.assert_allclose(final[2], final[0], atol=1e-9)

    def test_the_rotor_force_round_trip_flies_the_wrench_run_where_no_rotor_clips(self):
        assert_rotor_force_path_matches(lambda **kw: QuadrotorCsPayloadBase(**kw))

    def test_a_wrench_input_is_not_clipped_in_place(self):
        mdl = QuadrotorCsPayloadBase(input="wrench")
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        u = np.array([100.0, 0.0, 0.0, 0.0])
        mdl.step(u)
        assert u[0] == 100.0


class TestRates:
    """M1's rate scheme: 200 Hz payload loop, 800 Hz attitude, 1600 Hz physics."""

    def test_defaults(self):
        mdl = QuadrotorCsPayloadBase()
        assert mdl.dt == 0.005
        assert mdl.physics_rate_hz == pytest.approx(1600.0)
        assert mdl.physics_steps == 8
        assert mdl.attitude_rate_hz == pytest.approx(800.0)

    @pytest.mark.parametrize("kwargs", INPUT_PATHS)
    def test_attitude_loop_runs_four_times_per_step_on_every_path(self, kwargs):
        assert attitude_calls_per_step(QuadrotorCsPayloadBase(**kwargs)) == 4

    @pytest.mark.parametrize("rate,per_step", [(None, 1), (400.0, 2), (1600.0, 8)])
    def test_an_explicit_rate_is_honoured(self, rate, per_step):
        mdl = QuadrotorCsPayloadBase(input="wrench", attitude_rate_hz=rate)
        assert attitude_calls_per_step(mdl) == per_step

    def test_a_step_advances_exactly_dt(self):
        mdl = QuadrotorCsPayloadBase(physics_rate_hz=1000.0, attitude_rate_hz=1000.0)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        hover = np.array([0.0, 0.0, (mdl.mass + mdl.payload_mass) * 9.81])
        for _ in range(2000):
            mdl.step(hover)
        assert mdl.t == 2000 * 0.005

    @pytest.mark.parametrize("rate", [300.0, 1000.0, 0.0, float("nan")])
    def test_an_attitude_rate_that_does_not_divide_the_substeps_is_refused(self, rate):
        with pytest.raises(ValueError, match="attitude_rate_hz"):
            QuadrotorCsPayloadBase(attitude_rate_hz=rate)

    @pytest.mark.parametrize("rate", [400.0, 2400.0, 900.0])
    def test_a_physics_rate_outside_the_range_or_not_whole_is_refused(self, rate):
        with pytest.raises(ValueError, match="physics_rate_hz"):
            QuadrotorCsPayloadBase(physics_rate_hz=rate)

    def test_the_inner_loop_steps_keyword_is_refused(self):
        with pytest.raises(TypeError, match="physics_rate_hz"):
            QuadrotorCsPayloadBase(inner_loop_steps=4)


class TestFeedforward:
    """The airframe's flatness feed-forward, wired as on M1."""

    def test_the_feedforward_closes_the_attitude_gap(self):
        """On a trajectory exposing ``get_jet`` the attitude command follows the
        recovered reference; behind a plain callable it does not."""
        rms_ff, angle_ff = fly_lissajous(QuadrotorCsPayloadBase())
        rms_no, angle_no = fly_lissajous(
            QuadrotorCsPayloadBase(), bind=lambda traj: lambda t: traj.get_full(t)
        )
        assert angle_ff < 0.5 < angle_no
        assert rms_ff < 0.6 * rms_no

    @pytest.mark.parametrize("kwargs", INPUT_PATHS)
    def test_every_path_receives_the_reference(self, kwargs):
        mdl = QuadrotorCsPayloadBase(**kwargs)
        traj = CrazyTrajectory(tf=1.0, center=np.array([1.0, 1.0, 1.5]))
        mdl.payload_controller.setpoint = traj.get
        seen = []
        step = mdl.step

        def recorded(u, desired_att=None):
            seen.append(desired_att)
            step(u, desired_att=desired_att)

        mdl.step = recorded
        mdl.simulate(0.05, payload_position=traj.get(0.0)[0])
        assert seen and all(d is not None for d in seen)

    def test_yaw_setpoint_turns_the_airframe(self):
        mdl = QuadrotorCsPayloadBase()
        traj = CrazyTrajectory(tf=6.0, center=np.array([0.0, 0.0, 1.0]), ax=0.2, ay=0.2, az=0.1)
        mdl.payload_controller.setpoint = traj.get_full
        mdl.yaw_setpoint = lambda t: (0.5, 0.0, 0.0)
        mdl.simulate(6.0, payload_position=traj.get(0.0)[0])
        R = np.asarray(mdl.state.orientation)
        assert np.arctan2(R[1, 0], R[0, 0]) == pytest.approx(0.5, abs=0.01)

    def test_a_non_callable_yaw_setpoint_is_refused(self):
        with pytest.raises(TypeError, match="yaw_setpoint"):
            QuadrotorCsPayloadBase().yaw_setpoint = 0.5


class TestMinimumTension:
    """A downward command keeps the cable taut and the commanded attitude upright."""

    def test_a_drop_never_commands_an_inverted_airframe(self):
        mdl = QuadrotorCsPayloadBase()
        drop = StepSetpoint(np.array([0.0, 0.0, 2.0]), np.array([0.0, 0.0, 0.5]), 1.0)
        mdl.payload_controller.setpoint = drop.get
        angles, thrusts = probe_attitude_command(mdl)
        mdl.simulate(6.0, payload_position=np.array([0.0, 0.0, 2.0]))
        assert max(angles) < 30.0
        assert min(thrusts) >= 0.99 * mdl.payload_controller.min_tension
        assert np.linalg.norm(mdl.state.payload_position - [0.0, 0.0, 0.5]) < 0.01

    def test_min_tension_is_a_tenth_of_the_payload_weight(self):
        ctrl = QuadrotorCsPayloadBase().payload_controller
        assert D.MIN_CABLE_TENSION_FRACTION == 0.1
        assert ctrl.min_tension == pytest.approx(0.1 * 0.15 * D.GRAVITY)
