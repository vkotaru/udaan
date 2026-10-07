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

    ``use_reference=False`` reproduces the pre-composition behaviour by
    discarding the reference attitude the flat map recovers, which is exactly
    the ``[1, 0, 0]`` heading hint the attitude loop used to have no way past.
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
    shows that reverting the hint to ``[1, 0, 0]`` costs anything.
    """

    def test_a_yawing_reference_is_flown(self):
        """Mean heading error 1.4°, against 91° with the reference discarded.

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
