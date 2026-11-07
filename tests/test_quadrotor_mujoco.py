"""Tests for QuadrotorMujoco — unit + integration."""

import numpy as np
import pytest

from udaan.manif import SO3, TSO3
from udaan.models.quadrotor import QuadrotorMujoco

INPUT_FORCE_COMBOS = [
    ({}, "accel+wrench (default)"),
    ({"force": "prop_forces"}, "accel+propforces"),
    ({"input": "wrench"}, "wrench+wrench"),
    ({"input": "wrench", "force": "prop_forces"}, "wrench+propforces"),
    ({"input": "prop_forces"}, "propforces+wrench"),
    ({"input": "prop_forces", "force": "prop_forces"}, "propforces+propforces"),
]

TRAJECTORIES = [
    (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), "hover at setpoint"),
    (np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 1.0]), "climb from 0.5m"),
    (np.array([0.5, 0.5, 0.5]), np.array([0.0, 0.0, 1.0]), "lateral + climb"),
    (np.array([1.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), "large offset"),
]


class TestMujocoQuadrotorUnit:
    """Basic instantiation and state."""

    def test_instantiate(self):
        mdl = QuadrotorMujoco(render=False)
        assert mdl.state is not None

    def test_reset(self):
        mdl = QuadrotorMujoco(render=False)
        mdl.reset(position=np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(mdl.state.position, np.array([1.0, 2.0, 3.0]), atol=0.01)

    def test_time_syncs(self):
        mdl = QuadrotorMujoco(render=False)
        mdl.reset()
        mdl.step(np.zeros(3))
        assert mdl.t > 0.0

    def test_state_types(self):
        mdl = QuadrotorMujoco(render=False)
        mdl.reset()
        assert isinstance(mdl.state.orientation, SO3)
        assert isinstance(mdl.state.angular_velocity, TSO3)


@pytest.mark.integration
class TestMujocoQuadrotorIntegration:
    """Simulate and verify convergence."""

    SIM_TIME = 5.0
    TOL_POS = 0.15

    @pytest.mark.parametrize("kwargs,mode", INPUT_FORCE_COMBOS)
    @pytest.mark.parametrize("start,target,desc", TRAJECTORIES)
    def test_converge(self, kwargs, mode, start, target, desc):
        mdl = QuadrotorMujoco(render=False, **kwargs)
        mdl._pos_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(tf=self.SIM_TIME, position=start)
        pos_err = np.linalg.norm(mdl.state.position - target)
        assert pos_err < self.TOL_POS, (
            f"[{mode}, {desc}] Position error {pos_err:.4f}m > {self.TOL_POS}m"
        )


@pytest.mark.mujoco
@pytest.mark.parametrize("kwargs,mode", INPUT_FORCE_COMBOS)
def test_every_input_path_flies_the_bound_trajectory(kwargs, mode):
    """Every input path flies the trajectory bound to ``position_controller``."""
    target = np.array([3.0, 0.0, 2.0])
    mdl = QuadrotorMujoco(render=False, **kwargs)
    mdl.position_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
    mdl.simulate(tf=8.0, position=np.array([0.0, 0.0, 2.0]))
    assert np.linalg.norm(mdl.state.position - target) < 0.05, mode


@pytest.mark.mujoco
class TestBackendAgreesWithItsModel:
    """The MJCF and the model's own constants say the same thing."""

    def test_step_advances_dt(self):
        """One ``step()`` advances the model's clock by exactly ``dt``."""
        mdl = QuadrotorMujoco(render=False)
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        hover = np.array([0.0, 0.0, mdl.mass * 9.81])
        for _ in range(100):
            mdl.step(hover)
        assert mdl.dt == pytest.approx(0.005)
        assert mdl.t == pytest.approx(100 * mdl.dt, abs=1e-9)

    def test_wrench_actuator_ranges_are_the_model_limits(self):
        """The MJCF's ranges equal the model's limits, so MuJoCo never re-clips a command."""
        from udaan.core import defaults as D

        mdl = QuadrotorMujoco(render=False)
        ranges = mdl._mjMdl.model.actuator_ctrlrange
        np.testing.assert_allclose(ranges[4], [D.DEFAULT_MIN_THRUST, D.DEFAULT_MAX_THRUST])
        np.testing.assert_allclose(ranges[5:8, 0], D.DEFAULT_MIN_TORQUE)
        np.testing.assert_allclose(ranges[5:8, 1], D.DEFAULT_MAX_TORQUE)
        for i in range(4):
            np.testing.assert_allclose(
                ranges[i], [D.DEFAULT_PROP_MIN_FORCE, D.DEFAULT_PROP_MAX_FORCE]
            )

    def test_rotor_geometry_is_the_allocation_matrix(self):
        """Each rotor's site and yaw gear reproduce its column of the model's
        allocation matrix to 1e-10."""
        mdl = QuadrotorMujoco(render=False)
        m = mdl._mjMdl.model
        for i in range(4):
            x, y, _ = m.site_pos[m.actuator_trnid[i, 0]]
            column = np.array([1.0, y, -x, m.actuator_gear[i, 5]])
            np.testing.assert_allclose(column, mdl._alloc[:, i], atol=1e-10)

    def test_base_model_flies_the_mjcf_airframe(self):
        """``QuadrotorBase`` flies the mass and inertia this backend reads from the MJCF."""
        from udaan.models.quadrotor import QuadrotorBase

        base, mj = QuadrotorBase(), QuadrotorMujoco(render=False)
        body = mj._mjMdl.model
        assert base.mass == pytest.approx(float(body.body_mass[mj._mj_quad_index]))
        np.testing.assert_allclose(
            np.diag(base.inertia), body.body_inertia[mj._mj_quad_index], rtol=1e-12
        )
        assert base.mass == pytest.approx(mj.mass)
        np.testing.assert_allclose(base.inertia, mj.inertia)


def _attitude_calls_per_step(mdl, tf=0.1):
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
    mdl.simulate(tf, position=np.array([0.0, 0.0, 1.0]))
    return calls[0] / steps[0]


@pytest.mark.mujoco
class TestAttitudeRate:
    """One attitude-loop rate for every input path, as on QuadrotorBase."""

    @pytest.mark.parametrize("kwargs,mode", INPUT_FORCE_COMBOS)
    def test_default_is_800_hz_on_every_path(self, kwargs, mode):
        """The default attitude rate is four updates per 5 ms step on every input path."""
        from udaan.core.defaults import DEFAULT_ATTITUDE_RATE_HZ

        mdl = QuadrotorMujoco(render=False, **kwargs)
        assert DEFAULT_ATTITUDE_RATE_HZ == 800.0
        assert _attitude_calls_per_step(mdl) == 4, mode
        assert mdl.attitude_rate_hz == pytest.approx(800.0)

    @pytest.mark.parametrize("kwargs,mode", INPUT_FORCE_COMBOS)
    def test_none_is_once_per_outer_step_on_every_path(self, kwargs, mode):
        mdl = QuadrotorMujoco(render=False, attitude_rate_hz=None, **kwargs)
        assert _attitude_calls_per_step(mdl) == 1, mode
        assert mdl.attitude_rate_hz == pytest.approx(1.0 / mdl.dt)

    @pytest.mark.parametrize("rate,per_step", [(400.0, 2), (800.0, 4)])
    @pytest.mark.parametrize("kwargs,mode", INPUT_FORCE_COMBOS)
    def test_a_faster_rate_is_honoured_on_every_path(self, kwargs, mode, rate, per_step):
        mdl = QuadrotorMujoco(render=False, attitude_rate_hz=rate, **kwargs)
        assert _attitude_calls_per_step(mdl) == per_step, mode
        assert mdl.attitude_rate_hz == pytest.approx(rate)

    def test_acceleration_and_wrench_paths_fly_the_same_run(self):
        """Same loops at the same rate: the two paths differ only in who calls the attitude law."""
        target = np.array([1.0, 0.5, 1.5])
        final = []
        for kwargs in ({}, {"input": "wrench"}):
            mdl = QuadrotorMujoco(render=False, **kwargs)
            mdl.position_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
            mdl.simulate(1.0, position=np.array([0.0, 0.0, 1.0]))
            final.append(np.asarray(mdl.state.position).copy())
        np.testing.assert_array_equal(final[0], final[1])

    @pytest.mark.parametrize("rate", [300.0, 1000.0, 0.0, float("nan")])
    def test_a_rate_that_does_not_divide_the_substeps_is_refused(self, rate):
        with pytest.raises(ValueError, match="attitude_rate_hz"):
            QuadrotorMujoco(render=False, attitude_rate_hz=rate)

    def test_the_attitude_zoh_keyword_is_refused(self):
        with pytest.raises(TypeError, match="attitude_rate_hz"):
            QuadrotorMujoco(render=False, attitude_zoh=True)


def _fly_flip(mdl):
    """Fly the default flip; return max ∠(R, R_d) inside the flip [deg] and the
    position errors over the whole run [m]."""
    from udaan.utils.trajectory import FlipTrajectory

    flip = FlipTrajectory()
    mdl.position_controller.setpoint = flip.get
    controller = mdl.attitude_controller
    command = controller._cmd_accel_to_cmd_att
    angles, errors = [], []

    def probe(accel, reference=None):
        Rd = command(accel, reference=reference)
        if flip._t2 < mdl.t <= flip._t3:
            R = np.asarray(mdl.state.orientation)
            cos = (np.trace(np.asarray(Rd).T @ R) - 1.0) / 2.0
            angles.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
        return Rd

    controller._cmd_accel_to_cmd_att = probe
    step = mdl.step

    def instrumented(u, desired_att=None):
        errors.append(float(np.linalg.norm(mdl.state.position - flip.get(mdl.t)[0])))
        return step(u, desired_att=desired_att)

    mdl.step = instrumented
    mdl.simulate(flip.duration, position=np.array([0.0, 0.0, 2.0]))
    return max(angles), np.array(errors)


@pytest.mark.mujoco
def test_flip_tracks_its_commanded_attitude_while_inverted():
    """The flat map's heading convention would turn the reference — and through
    the heading hint the command R_d — 180° about b₃ for the inverted half of
    the flip; the flip supplies its own fixed-axis attitude instead, so
    ∠(R, R_d) stays small."""
    # The bounds below hold at 200 / 200 / 800 Hz (position / attitude / physics).
    worst, errors = _fly_flip(
        QuadrotorMujoco(render=False, attitude_rate_hz=None, physics_rate_hz=800.0)
    )
    assert worst < 3.0
    assert np.sqrt(np.mean(errors**2)) < 1.22e-3
    assert errors.max() < 3.4e-3


class TestRotorParametersFromMjcf:
    """On MuJoCo the MJCF is the source of the rotor parameters: it is read, and
    an explicitly passed value that disagrees with it is refused rather than
    silently flown against a different machine."""

    def test_mjcf_rotors_are_todays_constants(self):
        from udaan.core import defaults as D

        read = QuadrotorMujoco(render=False)._read_mjcf_rotors()
        assert read["arm_length"] == pytest.approx(D.DEFAULT_ARM_LENGTH, rel=1e-9)
        wrapped = (read["rotor_angles"] - D.DEFAULT_ROTOR_ANGLES + np.pi) % (2 * np.pi) - np.pi
        np.testing.assert_allclose(wrapped, 0.0, atol=1e-9)
        np.testing.assert_array_equal(read["spin_directions"], D.DEFAULT_SPIN_DIRECTIONS)
        assert read["moment_ratio"] == pytest.approx(
            D.DEFAULT_TORQUE_CONSTANT / D.DEFAULT_FORCE_CONSTANT, rel=1e-9
        )
        assert (read["prop_min_force"], read["prop_max_force"]) == (
            D.DEFAULT_PROP_MIN_FORCE,
            D.DEFAULT_PROP_MAX_FORCE,
        )

    def test_agreeing_defaults_keep_their_exact_form(self):
        """The MJCF's ten digits agree with the defaults, so nothing moves."""
        from udaan.models.quadrotor import QuadrotorBase

        mj, base = QuadrotorMujoco(render=False), QuadrotorBase()
        assert np.array_equal(mj.allocation_matrix, base.allocation_matrix)
        for key, value in base.rotor_parameters.items():
            assert np.array_equal(mj.rotor_parameters[key], value), key

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"arm_length": 0.25},
            {"rotor_angles": [0.0, np.pi / 2, np.pi, 3 * np.pi / 2]},
            {"spin_directions": [1.0, -1.0, 1.0, -1.0]},
            {"prop_max_force": 8.0},
            {"prop_min_force": 1.0},
            {"force_constant": 1.0e-6, "torque_constant": 1.0e-7},
        ],
    )
    def test_a_disagreeing_explicit_value_is_refused(self, kwargs):
        with pytest.raises(ValueError, match="quadrotor_mj.xml"):
            QuadrotorMujoco(render=False, **kwargs)

    def test_agreeing_explicit_values_are_accepted(self):
        mdl = QuadrotorMujoco(
            render=False,
            arm_length=0.2,
            rotor_angles=[np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, -np.pi / 4],
            prop_max_force=10.0,
        )
        np.testing.assert_allclose(
            mdl.allocation_matrix, QuadrotorMujoco(render=False).allocation_matrix, atol=1e-15
        )

    def test_one_motor_constant_takes_the_other_from_the_mjcf_ratio(self):
        mdl = QuadrotorMujoco(render=False, force_constant=1.0e-5)
        params = mdl.rotor_parameters
        assert params["force_constant"] == 1.0e-5
        ratio = mdl._read_mjcf_rotors()["moment_ratio"]
        assert params["torque_constant"] / params["force_constant"] == pytest.approx(ratio)

    def test_a_different_mjcf_replaces_the_defaults(self, monkeypatch):
        """A default the MJCF contradicts is replaced by the MJCF's value."""
        read = QuadrotorMujoco._read_mjcf_rotors

        def wider(self):
            out = read(self)
            out["arm_length"] = 0.3
            out["prop_max_force"] = 7.0
            return out

        monkeypatch.setattr(QuadrotorMujoco, "_read_mjcf_rotors", wider)
        mdl = QuadrotorMujoco(render=False)
        assert mdl.rotor_parameters["arm_length"] == 0.3
        assert mdl.rotor_parameters["prop_max_force"] == 7.0
        np.testing.assert_allclose(np.abs(mdl.allocation_matrix[1:3]), 0.3 / np.sqrt(2), rtol=1e-12)


def _lissajous_rms(mdl):
    """Position RMS over a 10 s CLI-geometry lissajous, started on the reference."""
    from udaan.utils.trajectory import CrazyTrajectory

    traj = CrazyTrajectory(tf=10.0, center=np.array([1.0, 1.0, 2.0]))
    mdl.position_controller.setpoint = traj.get
    errors = []
    step = mdl.step

    def instrumented(u, desired_att=None):
        step(u, desired_att=desired_att)
        errors.append(np.linalg.norm(mdl.state.position - traj.get(mdl.t)[0]))

    mdl.step = instrumented
    p, v, _ = traj.get(0.0)
    mdl.simulate(tf=10.0, position=p, velocity=v)
    return float(np.sqrt(np.mean(np.square(errors))))


MODEL, MODEL_KWARGS = QuadrotorMujoco, {"render": False}


class TestPhysicsRate:
    """``physics_rate_hz`` sets the substeps: ``N = rate · dt`` of ``h = dt/N``."""

    def _make(self, **kwargs):
        return MODEL(**MODEL_KWARGS, **kwargs)

    def test_default_is_eight_substeps_of_0_625_ms(self):
        mdl = self._make()
        assert mdl.physics_steps == 8
        assert mdl.physics_rate_hz == 1600.0
        assert mdl.h == 0.000625 and mdl.dt == 0.005
        assert mdl.h == self._make(physics_rate_hz=1600.0).h

    @pytest.mark.parametrize("rate", [600.0, 1000.0, 2000.0])
    def test_a_rate_the_default_attitude_rate_does_not_divide_says_so(self, rate):
        with pytest.raises(ValueError, match="is the default"):
            self._make(physics_rate_hz=rate)

    @pytest.mark.parametrize("rate,n", [(600.0, 3), (1000.0, 5), (2000.0, 10)])
    def test_a_rate_sets_the_substeps_and_a_step_advances_dt(self, rate, n):
        mdl = self._make(physics_rate_hz=rate, attitude_rate_hz=None)
        assert mdl.physics_steps == n
        assert mdl.h == pytest.approx(mdl.dt / n, rel=1e-15)
        assert mdl.dt == 0.005
        mdl.reset(position=np.array([0.0, 0.0, 1.0]))
        mdl.step(np.array([0.0, 0.0, mdl.mass * 9.81]))
        assert mdl.t == pytest.approx(mdl.dt, rel=1e-12)

    @pytest.mark.parametrize("rate", [500.0, 2500.0, 700.0, 400.0, 0.0, float("nan")])
    def test_an_invalid_rate_is_refused(self, rate):
        with pytest.raises(ValueError, match="physics_rate_hz"):
            self._make(physics_rate_hz=rate)

    def test_the_refusal_lists_the_valid_rates_and_the_500_hz_rule(self):
        with pytest.raises(
            ValueError, match="600, 800, 1000, 1200, 1400, 1600, 1800, 2000 Hz"
        ) as e:
            self._make(physics_rate_hz=500.0)
        assert "2 ms" in str(e.value)

    def test_500_hz_needs_a_dt_that_is_a_multiple_of_2_ms(self):
        assert self._make(physics_rate_hz=500.0, dt=0.004, attitude_rate_hz=None).physics_steps == 2

    @pytest.mark.parametrize(
        "physics,attitude,updates", [(1000.0, 1000.0, 5), (2000.0, 400.0, 2), (2000.0, 1000.0, 5)]
    )
    def test_attitude_rate_divides_the_resulting_substeps(self, physics, attitude, updates):
        mdl = self._make(physics_rate_hz=physics, attitude_rate_hz=attitude)
        assert mdl._attitude_updates == updates

    @pytest.mark.parametrize("physics,attitude", [(1000.0, 400.0), (1000.0, 800.0), (600.0, 400.0)])
    def test_attitude_rate_not_dividing_the_resulting_substeps_is_refused(self, physics, attitude):
        with pytest.raises(ValueError, match="attitude_rate_hz"):
            self._make(physics_rate_hz=physics, attitude_rate_hz=attitude)

    @pytest.mark.parametrize("rate", [1000.0, 2000.0])
    def test_a_non_default_rate_still_tracks(self, rate):
        """Hover from an offset, and the CLI lissajous (39 mm RMS at 800 Hz)."""
        hover = self._make(physics_rate_hz=rate, attitude_rate_hz=None)
        target = np.array([0.0, 0.0, 1.0])
        hover.position_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        hover.simulate(tf=6.0, position=np.array([0.5, -0.5, 0.5]))
        assert np.linalg.norm(hover.state.position - target) < 5e-3
        rms = _lissajous_rms(self._make(physics_rate_hz=rate, attitude_rate_hz=None))
        assert rms < 0.05
        assert rms == pytest.approx(_lissajous_rms(self._make()), abs=2e-3)

    @pytest.mark.parametrize("rate", [800.0, 1000.0, 2000.0])
    def test_the_rate_is_written_into_the_mujoco_timestep(self, rate):
        """MuJoCo's timestep is ``dt / N`` whatever the MJCF says; nothing is rounded."""
        mdl = self._make(physics_rate_hz=rate, attitude_rate_hz=None)
        model = mdl._mjMdl.model
        assert model.opt.timestep == mdl.dt / mdl.physics_steps
        mdl.reset(position=np.array([0.0, 0.0, 1.0]))
        mdl.step(np.array([0.0, 0.0, mdl.mass * 9.81]))
        assert mdl._mjMdl.data.time == pytest.approx(mdl.dt, rel=1e-12)
        assert mdl._step_iter == mdl.physics_steps


def _count_steps(mdl, tf, **reset):
    """``simulate(tf)`` from hover; return the number of ``step()`` calls."""
    calls = [0]
    step = mdl.step

    def counted(u, desired_att=None):
        calls[0] += 1
        step(u, desired_att=desired_att)

    mdl.step = counted
    target = np.array([0.0, 0.0, 1.0])
    mdl.position_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
    mdl.simulate(tf, position=target, **reset)
    return calls[0]


class TestClock:
    """The clock counts whole substeps; it does not accumulate ``t += h``.

    A running sum of ``h`` drifts when ``h`` is not a binary fraction: at 1000
    or 1600 Hz it falls ~1e-13 short of 10 s after 2000 steps, enough for a
    10 s ``simulate()`` to take a 2001st step.
    """

    def _make(self, **kwargs):
        return MODEL(**MODEL_KWARGS, **kwargs)

    @pytest.mark.parametrize("rate", [800.0, 1000.0, 1600.0, 2000.0])
    def test_a_10_s_simulate_ends_at_exactly_10_s(self, rate):
        mdl = self._make(physics_rate_hz=rate, attitude_rate_hz=None)
        assert _count_steps(mdl, 10.0) == 2000
        assert mdl.t == 10.0

    def test_a_10_ms_step_ends_at_exactly_10_s(self):
        mdl = self._make(dt=0.01, physics_rate_hz=1000.0, attitude_rate_hz=500.0)
        assert _count_steps(mdl, 10.0) == 1000
        assert mdl.t == 10.0

    @pytest.mark.parametrize("rate", [1000.0, 1600.0])
    def test_every_step_lands_on_its_multiple_of_dt(self, rate):
        mdl = self._make(physics_rate_hz=rate, attitude_rate_hz=None)
        mdl.reset(position=np.array([0.0, 0.0, 1.0]))
        hover = np.array([0.0, 0.0, mdl.mass * 9.81])
        worst = 0.0
        for k in range(1, 2001):
            mdl.step(hover)
            worst = max(worst, abs(mdl.t - k * 0.005))
        assert worst <= 2e-15

    def test_reset_restarts_the_clock(self):
        mdl = self._make(physics_rate_hz=1000.0, attitude_rate_hz=None)
        assert _count_steps(mdl, 1.0) == 200
        assert _count_steps(mdl, 1.0) == 200
        assert mdl.t == 1.0


class TestStepPeriod:
    """``dt`` is the position-loop period, a constructor keyword validated with the rates.

    ``N = physics_rate_hz · dt`` must be a whole number of substeps in range, and
    the attitude updates per step must divide ``N``; anything else raises.
    """

    def _make(self, **kwargs):
        return MODEL(**MODEL_KWARGS, **kwargs)

    def test_a_10_ms_step(self):
        mdl = self._make(dt=0.01, physics_rate_hz=1000.0, attitude_rate_hz=500.0)
        assert mdl.dt == 0.01
        assert mdl.physics_steps == 10
        assert mdl._attitude_updates == 5
        assert mdl.attitude_rate_hz == pytest.approx(500.0)
        mdl.reset(position=np.array([0.0, 0.0, 1.0]))
        mdl.step(np.array([0.0, 0.0, mdl.mass * 9.81]))
        assert mdl.t == 0.01

    def test_the_default_attitude_rate_holds_at_another_dt(self):
        mdl = self._make(dt=0.01, physics_rate_hz=800.0)
        assert mdl.physics_steps == 8
        assert mdl._attitude_updates == 8
        assert mdl.attitude_rate_hz == pytest.approx(800.0)

    def test_none_is_once_per_step_at_any_dt(self):
        mdl = self._make(dt=0.01, physics_rate_hz=1000.0, attitude_rate_hz=None)
        assert mdl._attitude_updates == 1
        assert mdl.attitude_rate_hz == pytest.approx(100.0)

    @pytest.mark.parametrize("dt", [0.0, -0.01, float("nan"), float("inf"), 0.0033, 0.0001, 0.0045])
    def test_an_invalid_dt_is_refused(self, dt):
        with pytest.raises(ValueError):
            self._make(dt=dt)

    @pytest.mark.parametrize("attitude", [300.0, 400.0, 700.0])
    def test_an_attitude_rate_not_dividing_the_substeps_at_10_ms_is_refused(self, attitude):
        with pytest.raises(ValueError, match="attitude_rate_hz"):
            self._make(dt=0.01, physics_rate_hz=1000.0, attitude_rate_hz=attitude)

    def test_a_10_ms_step_still_tracks(self):
        """Hover from an offset, and the CLI lissajous, at 100 / 500 / 1000 Hz."""
        kw = {"dt": 0.01, "physics_rate_hz": 1000.0, "attitude_rate_hz": 500.0}
        hover = self._make(**kw)
        target = np.array([0.0, 0.0, 1.0])
        hover.position_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        hover.simulate(tf=6.0, position=np.array([0.5, -0.5, 0.5]))
        assert np.linalg.norm(hover.state.position - target) < 5e-3
        assert _lissajous_rms(self._make(**kw)) < 0.05

    def test_mujoco_time_is_the_model_clock(self):
        mdl = self._make(physics_rate_hz=1000.0, attitude_rate_hz=None)
        assert _count_steps(mdl, 10.0) == 2000
        assert mdl._mjMdl.data.time == mdl.t == 10.0
