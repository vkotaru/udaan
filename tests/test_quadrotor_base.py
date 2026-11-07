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


def _spiral():
    """The brisk helix M3 and M4 use, at M1's scale."""
    from udaan.utils.trajectory import SpiralTrajectory

    return SpiralTrajectory(
        start=np.array([0.0, 0.0, 2.0]),
        radius=0.6,
        speed=1.5,
        n_loops=2,
        advance=2.0,
        hold_time=0.5,
    )


def _fly(yaw_rate, use_reference=True):
    """Fly the spiral at ``ψ̇ = yaw_rate`` and report how well the heading is held.

    ``use_reference=False`` discards the reference attitude the flat map
    recovers, which leaves the attitude loop with the fixed ``[1, 0, 0]``
    heading hint.
    """
    traj = _spiral()
    mdl = QuadrotorBase()
    mdl.position_controller.setpoint = traj.get
    if yaw_rate is not None:
        mdl.yaw_setpoint = lambda t: (yaw_rate * t, yaw_rate, 0.0)
    if not use_reference:
        controller = mdl.attitude_controller
        recompute = controller._cmd_accel_to_cmd_att
        controller._cmd_accel_to_cmd_att = lambda accel, reference=None: recompute(accel)

    heading_err, pos_err = [], []
    step = mdl.step

    def instrumented(u, desired_att=None):
        b1 = np.asarray(mdl.state.orientation)[:, 0]
        horizontal = np.array([b1[0], b1[1], 0.0])
        norm = np.linalg.norm(horizontal)
        if norm > 1e-6:
            psi = (yaw_rate or 0.0) * mdl.t
            want = np.array([np.cos(psi), np.sin(psi), 0.0])
            flown = horizontal / norm
            heading_err.append(
                abs(np.degrees(np.arctan2(np.cross(want, flown)[2], float(want @ flown))))
            )
        reference, *_ = traj.get(mdl.t)
        pos_err.append(float(np.linalg.norm(mdl.state.position - reference)))
        return step(u, desired_att=desired_att)

    mdl.step = instrumented
    mdl.simulate(traj._tf, position=np.array([0.0, 0.0, 2.0]))
    return {
        "heading_mean": float(np.mean(heading_err)),
        "position_rms": float(np.sqrt(np.mean(np.square(pos_err)))),
        "state": (
            np.asarray(mdl.state.position).copy(),
            np.asarray(mdl.state.orientation).copy(),
        ),
    }


@pytest.mark.integration
@pytest.mark.slow
class TestYawFeedforward:
    """A commanded yaw is reachable through the public API, and is flown.

    :attr:`~udaan.models.quadrotor.QuadrotorBase.yaw_setpoint` is the whole
    switch. It exists because the flatness feedforward's reference attitude
    contributes exactly one thing the position loop cannot — the heading — and
    without a yaw setpoint the flat map is handed ψ ≡ 0, which makes the
    composition a near no-op and leaves the capability untestable end to end.

    These are closed-loop runs on purpose. The unit tests in
    ``test_control.py`` pin the geometry of the composition; only a flown run
    shows that a fixed ``[1, 0, 0]`` hint costs anything.
    """

    def test_a_yawing_reference_is_flown(self):
        """Mean heading error under 10°, against over 45° with the reference discarded.

        That gap is the point of the whole channel: with the fixed world-``x``
        hint the commanded heading is not merely tracked badly, it is
        *unreachable* — the attitude command never yaws at all, so the error
        grows with ψ and averages a quarter turn.
        """
        with_reference = _fly(3.0)
        without = _fly(3.0, use_reference=False)
        assert with_reference["heading_mean"] < 10.0, with_reference["heading_mean"]
        assert without["heading_mean"] > 45.0, without["heading_mean"]
        assert with_reference["position_rms"] < without["position_rms"]

    def test_a_yawing_reference_does_not_cost_position_tracking(self):
        """Heading is bought on the free degree of freedom, not out of ``b₃``."""
        still = _fly(0.0)
        yawing = _fly(3.0)
        assert yawing["position_rms"] < 1.2 * still["position_rms"]

    def test_no_yaw_setpoint_is_exactly_psi_zero(self):
        """The default changes nothing: unbound and ψ ≡ 0 are the same run."""
        unbound = _fly(None)
        explicit_zero = _fly(0.0)
        for a, b in zip(unbound["state"], explicit_zero["state"], strict=True):
            np.testing.assert_array_equal(a, b)

    def test_a_yaw_setpoint_must_be_callable(self):
        mdl = QuadrotorBase()
        with pytest.raises(TypeError):
            mdl.yaw_setpoint = 0.5

    def test_a_yaw_setpoint_must_return_three_derivatives(self):
        traj = _spiral()
        mdl = QuadrotorBase()
        mdl.position_controller.setpoint = traj.get
        mdl.yaw_setpoint = lambda t: (0.0, 0.0)
        with pytest.raises(ValueError, match="psi, dpsi, ddpsi"):
            mdl.simulate(0.05, position=np.array([0.0, 0.0, 2.0]))


@pytest.mark.parametrize("kwargs,mode", INPUT_FORCE_COMBOS)
def test_every_input_path_flies_the_bound_trajectory(kwargs, mode):
    """Every input path flies the trajectory bound to ``position_controller``."""
    target = np.array([3.0, 0.0, 2.0])
    mdl = QuadrotorBase(**kwargs)
    mdl.position_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
    mdl.simulate(tf=8.0, position=np.array([0.0, 0.0, 2.0]))
    assert np.linalg.norm(mdl.state.position - target) < 0.05, mode


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


class TestAttitudeRate:
    """One attitude-loop rate for every input path, overridable in the constructor.

    ``tests/test_quadrotor_mujoco.py`` holds the same checks for that backend.
    """

    @pytest.mark.parametrize("kwargs,mode", INPUT_FORCE_COMBOS)
    def test_default_is_800_hz_on_every_path(self, kwargs, mode):
        """The default attitude rate is four updates per 5 ms step on every input path."""
        from udaan.core.defaults import DEFAULT_ATTITUDE_RATE_HZ

        mdl = QuadrotorBase(**kwargs)
        assert DEFAULT_ATTITUDE_RATE_HZ == 800.0
        assert _attitude_calls_per_step(mdl) == 4, mode
        assert mdl.attitude_rate_hz == pytest.approx(800.0)

    @pytest.mark.parametrize("kwargs,mode", INPUT_FORCE_COMBOS)
    def test_none_is_once_per_outer_step_on_every_path(self, kwargs, mode):
        mdl = QuadrotorBase(attitude_rate_hz=None, **kwargs)
        assert _attitude_calls_per_step(mdl) == 1, mode
        assert mdl.attitude_rate_hz == pytest.approx(1.0 / mdl.dt)

    @pytest.mark.parametrize("rate,per_step", [(400.0, 2), (800.0, 4)])
    @pytest.mark.parametrize("kwargs,mode", INPUT_FORCE_COMBOS)
    def test_a_faster_rate_is_honoured_on_every_path(self, kwargs, mode, rate, per_step):
        mdl = QuadrotorBase(attitude_rate_hz=rate, **kwargs)
        assert _attitude_calls_per_step(mdl) == per_step, mode
        assert mdl.attitude_rate_hz == pytest.approx(rate)

    def test_step_alone_closes_the_loop_at_the_rate(self):
        """The path an RL environment takes: ``step()`` without ``simulate()``."""
        mdl = QuadrotorBase(attitude_rate_hz=800.0)
        calls = [0]
        compute = mdl.attitude_controller.compute

        def counted(*args, **kwargs):
            calls[0] += 1
            return compute(*args, **kwargs)

        mdl.attitude_controller.compute = counted
        mdl.reset()
        mdl.step(np.array([0.0, 0.0, mdl.mass * 9.81]))
        assert calls[0] == 4
        assert mdl.t == pytest.approx(mdl.dt)

    @pytest.mark.parametrize("rate", [300.0, 1000.0, 100.0, 0.0, -200.0, float("nan")])
    def test_a_rate_that_does_not_divide_the_substeps_is_refused(self, rate):
        with pytest.raises(ValueError, match="attitude_rate_hz"):
            QuadrotorBase(attitude_rate_hz=rate)


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


def test_flip_tracks_its_commanded_attitude_while_inverted():
    """The flat map's heading convention would turn the reference — and through
    the heading hint the command R_d — 180° about b₃ for the inverted half of
    the flip; the flip supplies its own fixed-axis attitude instead, so
    ∠(R, R_d) stays small. (base model: 1.25 mm RMS, 3.8 mm max.)"""
    # The bounds below hold at 200 / 200 / 800 Hz (position / attitude / physics).
    worst, errors = _fly_flip(QuadrotorBase(attitude_rate_hz=None, physics_rate_hz=800.0))
    assert worst < 3.0
    assert np.sqrt(np.mean(errors**2)) < 1.30e-3
    assert errors.max() < 4.0e-3


class TestRotorParameters:
    """The rotor layout, constants and force limits are constructor keywords."""

    @staticmethod
    def _torque_from_geometry(arm, angle, spin, ratio, force):
        """The wrench of one rotor, from r × F and its drag moment."""
        r = arm * np.array([np.cos(angle), np.sin(angle), 0.0])
        torque = np.cross(r, np.array([0.0, 0.0, force]))
        return np.array([force, torque[0], torque[1], spin * ratio * force])

    def test_defaults_give_the_x_layout_matrix(self):
        """The defaults reproduce the X-layout matrix built column by column, bit for bit."""
        from udaan.core import defaults as D

        c = D.DEFAULT_TORQUE_CONSTANT / D.DEFAULT_FORCE_CONSTANT
        ang = [np.pi / 4, 3 * np.pi / 4, 5 * np.pi / 4, 7 * np.pi / 4]
        d = [-1.0, 1.0, -1.0, 1.0]
        expected = np.zeros((4, 4))
        for i in range(4):
            expected[:, i] = [
                1.0,
                D.DEFAULT_ARM_LENGTH * np.sin(ang[i]),
                -D.DEFAULT_ARM_LENGTH * np.cos(ang[i]),
                c * d[i],
            ]
        mdl = QuadrotorBase()
        assert np.array_equal(mdl.allocation_matrix, expected)
        assert np.array_equal(mdl._alloc_inv, np.linalg.pinv(expected))
        assert (mdl._prop_min_force, mdl._prop_max_force) == (0.0, 10.0)

    def test_custom_layout_is_each_rotors_own_wrench(self):
        """Every column is r × F plus d·c·f for the configured rotor."""
        arm, angles, spins = 0.31, [0.0, np.pi / 2, np.pi, 3 * np.pi / 2], [1.0, -1.0, 1.0, -1.0]
        kf, km = 2.0e-6, 1.0e-7
        mdl = QuadrotorBase(
            arm_length=arm,
            rotor_angles=angles,
            spin_directions=spins,
            force_constant=kf,
            torque_constant=km,
        )
        for i in range(4):
            expected = self._torque_from_geometry(arm, angles[i], spins[i], km / kf, 1.0)
            np.testing.assert_allclose(mdl.allocation_matrix[:, i], expected, atol=1e-15)
        assert not np.allclose(mdl.allocation_matrix, QuadrotorBase().allocation_matrix)

    def test_rotor_forces_spin_the_body_about_the_configured_arm(self):
        """Flown: one rotor's extra force on the PROP_FORCES path accelerates the
        body by I⁻¹(r × F), at the configured arm length."""
        for arm in (0.2, 0.35):
            mdl = QuadrotorBase(input="prop_forces", arm_length=arm)
            mdl.reset(position=np.array([0.0, 0.0, 1.0]))
            hover = mdl.mass * 9.81 / 4
            forces = np.full(4, hover)
            forces[0] += 0.5
            mdl.step(forces)
            r = arm * np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0])
            torque = np.cross(r, [0.0, 0.0, 0.5])
            expected = np.linalg.solve(mdl.inertia, torque) * mdl.dt
            np.testing.assert_allclose(
                np.asarray(mdl.state.angular_velocity)[:2], expected[:2], rtol=1e-3
            )

    def test_prop_force_limit_bounds_the_delivered_thrust(self):
        """Under force='prop_forces', no rotor exceeds the configured limit."""
        for limit in (10.0, 3.0):
            mdl = QuadrotorBase(input="wrench", force="prop_forces", prop_max_force=limit)
            mdl.reset(position=np.array([0.0, 0.0, 1.0]))
            mdl.step(np.array([19.0, 0.0, 0.0, 0.0]))
            assert mdl._last_wrench[0] == pytest.approx(min(19.0, 4 * limit))

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"arm_length": 0.0},
            {"arm_length": float("nan")},
            {"force_constant": -1.0},
            {"torque_constant": 0.0},
            {"rotor_angles": [0.0, 1.0, 2.0]},
            {"rotor_angles": [0.0, 1.0, 2.0, float("inf")]},
            {"rotor_angles": [0.0, 0.0, 0.0, 0.0]},
            {"spin_directions": [1.0, 1.0, -1.0, 0.5]},
            {"spin_directions": [1.0, 1.0, 1.0, 1.0]},
            {"prop_min_force": 5.0, "prop_max_force": 5.0},
            {"prop_max_force": float("nan")},
        ],
    )
    def test_invalid_rotor_parameters_are_refused(self, kwargs):
        with pytest.raises(ValueError):
            QuadrotorBase(**kwargs)


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


MODEL, MODEL_KWARGS = QuadrotorBase, {}


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

    def test_the_inner_loop_steps_keyword_is_refused(self):
        with pytest.raises(TypeError, match="physics_rate_hz"):
            QuadrotorBase(inner_loop_steps=4)


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
