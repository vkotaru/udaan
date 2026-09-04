"""Tests for :mod:`udaan.control.multi_quad_rigidbody` — the M4 payload cascade.

The rigid-body sibling of ``tests/test_multi_quad_cspayload_control.py``, and
the split is the same one: algebra at hover, algebra off it, and the closed
loop. What M4 adds is everything the payload's *attitude* brings with it, and
that is what most of this file is about:

* the cascade commands a **wrench**, not a force, so the allocation has a null
  space and a rank. Both are exercised — the rank-6 ring at ``nQ >= 3``, and
  the structurally underactuated ``nQ <= 2`` where part of every moment command
  is dropped and has to be *reported* rather than silently lost;
* :attr:`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController.attitude_setpoint`
  is a second, independent reference. It is the headline feature of the
  cascade, so it is flown here against non-level targets rather than only
  read back through :meth:`desired_wrench`;
* the flatness feedforward is built from the controller's **own**
  :attr:`~udaan.control.multi_quad_rigidbody.MultiQuadRigidbodyController.tension_policy`.
  A reference allocated at a different null-space bias describes a different
  formation, so the tests below pin both halves: that the shared policy makes
  tracking measurably better, and that a controller which does not publish one
  gets no feedforward at all.

Every ``ValueError`` the controller can raise has a case in
:class:`TestParameterValidation`; they are cheap and they are the difference
between a mis-specified team failing at construction and failing as a slowly
toppling payload ten seconds into a run.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_PHYSICAL
from udaan.control.multi_quad_rigidbody import MultiQuadRigidbodyController
from udaan.core.defaults import (
    DEFAULT_MULTI_PAYLOAD_CABLE_KD,
    DEFAULT_MULTI_PAYLOAD_CABLE_KP,
    DEFAULT_MULTI_PAYLOAD_POS_KD,
    DEFAULT_MULTI_PAYLOAD_POS_KP,
    DEFAULT_RIGID_PAYLOAD_ATT_KD,
    DEFAULT_RIGID_PAYLOAD_ATT_KP,
    GRAVITY,
)
from udaan.flatness import MultiQuadRigidbody
from udaan.flatness.allocation import FixedTensionDistribution, RingTensionDistribution
from udaan.manif import SO3, hat, rodrigues_expm
from udaan.models.multi_quad_rigidbody import MultiQuadRigidbodyBase, hover_wrench_map
from udaan.utils.trajectory import SpiralTrajectory

#: Counts whose default ring grasp has full rank 6, hence a controllable
#: payload attitude.
CONTROLLABLE_COUNTS = [3, 4, 5]

#: Counts where the grasp is rank-deficient and the attitude is structurally
#: uncontrollable — asserted, not fixed.
UNDERACTUATED_COUNTS = [1, 2]

_E3 = np.array([0.0, 0.0, 1.0])

_CONTROL_LOGGER = "udaan.control.multi_quad_rigidbody.payload_controller"
_MODEL_LOGGER = "udaan.models.multi_quad_rigidbody.base"


# ─── Helpers ──────────────────────────────────────────────────────────


def hover_setpoint(x):
    """A stationary payload setpoint at ``x``."""
    target = np.asarray(x, dtype=float).copy()
    return lambda t: (target, np.zeros(3), np.zeros(3))


def fixed_attitude(R_d):
    """A stationary payload *attitude* setpoint at ``R_d``."""
    R = np.asarray(R_d, dtype=float).copy()
    return lambda t: (R, np.zeros(3), np.zeros(3))


class YawRamp:
    """A payload attitude that yaws through ``angle`` on a smooth quintic.

    The reference and both its derivatives are exact, so the ``Ω_d``/``Ω̇_d``
    entries of the setpoint protocol carry real content rather than zeros —
    which is the point: a step in ``R_d`` exercises only the proportional
    channel of the geometric PD, and the two feedforward terms of
    :meth:`desired_wrench` would stay dead.
    """

    def __init__(self, angle: float = 0.6, duration: float = 6.0):
        self.angle = float(angle)
        self.duration = float(duration)

    def _profile(self, t: float) -> tuple[float, float, float]:
        T = self.duration
        if t >= T:
            return self.angle, 0.0, 0.0
        s = t / T
        # Quintic with zero velocity and acceleration at both ends.
        psi = self.angle * (10 * s**3 - 15 * s**4 + 6 * s**5)
        dpsi = self.angle * (30 * s**2 - 60 * s**3 + 30 * s**4) / T
        d2psi = self.angle * (60 * s - 180 * s**2 + 120 * s**3) / T**2
        return psi, dpsi, d2psi

    def get(self, t):
        psi, dpsi, d2psi = self._profile(t)
        # A pure yaw: R_d = R_z(ψ), so Ω_d = ψ̇ e₃ in the body frame too.
        return rodrigues_expm(psi * _E3), dpsi * _E3, d2psi * _E3


def attitude_angle_deg(R, R_d) -> float:
    """Rotation angle between two attitudes, in degrees."""
    err = np.asarray(R_d).T @ np.asarray(R)
    return float(np.degrees(np.arccos(np.clip((np.trace(err) - 1.0) / 2.0, -1.0, 1.0))))


def brisk_spiral():
    """Fast enough that the attitude loops visibly lag without feedforward."""
    return SpiralTrajectory(
        start=np.array([0.0, 0.0, 2.0]),
        radius=0.6,
        speed=1.5,
        n_loops=2,
        advance=2.0,
        hold_time=0.5,
    )


def hover_trim_tensions(mdl):
    """Scalar hover tensions of the model's current formation, from scratch.

    Solves the static conditions of
    :func:`~udaan.utils.grasp.hover_wrench_map` directly, so it is a second
    route to the same numbers the controller caches — not a read of them.
    """
    q = np.array([np.asarray(x) for x in mdl.state.cable_attitudes])
    wrench_map = hover_wrench_map(q, mdl.attachment_points, mdl.state.payload_orientation)
    target = np.concatenate([-mdl.payload_mass * GRAVITY * _E3, np.zeros(3)])
    tensions, *_ = np.linalg.lstsq(wrench_map, target, rcond=None)
    return tensions


def fly(mdl, trajectory, tf, feedforward: bool):
    """Run the cascade against ``trajectory``, returning tracking-error stats.

    ``feedforward=True`` binds the trajectory's ``get_full``, which is what
    trips the flat-map sniffing in
    :meth:`MultiQuadRigidbodyBase.simulate`. ``False`` binds a plain closure
    returning only ``(pos, vel, acc)`` — no ``__self__``, so no feedforward.
    That is the whole switch, and it is deliberately the one a user flips.

    This re-implements ``simulate``'s stepping loop so the per-step error can
    be sampled; nothing here covers the wiring *inside* ``simulate``.
    :meth:`TestFlatnessFeedforward.test_simulate_itself_runs_the_feedforward`
    does.
    """
    ctrl = mdl.payload_controller
    if feedforward:
        ctrl.setpoint = trajectory.get_full
    else:
        ctrl.setpoint = lambda t: trajectory.get_full(t)[:3]

    mdl.reset(payload_position=trajectory.get(0.0)[0])
    flat_fn = mdl._flatness_feedforward_fn()
    assert (flat_fn is not None) == feedforward
    flat_map = (
        MultiQuadRigidbody.from_model(mdl, tension_policy=ctrl.tension_policy)
        if flat_fn is not None
        else None
    )

    errors = []
    while mdl.t < tf:
        u = mdl.compute_default_input()
        desired_att = None if flat_map is None else mdl._desired_attitudes(flat_map, flat_fn, mdl.t)
        mdl.step(u, desired_att=desired_att)
        errors.append(float(np.linalg.norm(mdl.state.payload_position - trajectory.get(mdl.t)[0])))

    err = np.asarray(errors)
    return {"rms": float(np.sqrt(np.mean(err**2))), "max": float(err.max())}


# ─── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    @pytest.mark.parametrize("nQ", [1, 2, 3, 4, 5, 6])
    def test_output_shape(self, nQ):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        u = mdl.payload_controller.compute(0.0, mdl.state)
        assert u.shape == (nQ, 3)

    def test_from_model_adopts_model_parameters(self):
        mdl = MultiQuadRigidbodyBase(
            num_quadrotors=3,
            masses=[0.7, 1.1, 0.5],
            cable_lengths=[1.3, 1.2, 1.5],
            payload_mass=0.6,
        )
        ctrl = MultiQuadRigidbodyController.from_model(mdl)
        np.testing.assert_allclose(ctrl._masses, mdl.masses, atol=TOL_EXACT)
        np.testing.assert_allclose(ctrl._cable_lengths, mdl.cable_lengths, atol=TOL_EXACT)
        np.testing.assert_allclose(ctrl.attachment_points, mdl.attachment_points, atol=TOL_EXACT)
        assert ctrl._payload_mass == pytest.approx(mdl.payload_mass)

    def test_model_default_controller_is_the_cascade(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        assert isinstance(mdl.payload_controller, MultiQuadRigidbodyController)

    def test_attitude_setpoint_defaults_to_level_and_still(self):
        ctrl = MultiQuadRigidbodyController(num_quadrotors=4)
        R_d, Omega_d, dOmega_d = ctrl.attitude_setpoint(1.234)
        np.testing.assert_allclose(R_d, np.eye(3), atol=TOL_EXACT)
        np.testing.assert_allclose(Omega_d, np.zeros(3), atol=TOL_EXACT)
        np.testing.assert_allclose(dOmega_d, np.zeros(3), atol=TOL_EXACT)


# ─── Every ValueError ─────────────────────────────────────────────────


class TestParameterValidation:
    """One case per ``raise ValueError`` in the controller.

    Cheap tests, but the alternative to each of them is a run that looks
    plausible and is not: a mis-shaped ``attachment_points`` broadcast into the
    wrong grasp, or a controller re-synced from a team of a different size.
    """

    def test_zero_quadrotors(self):
        with pytest.raises(ValueError, match="num_quadrotors must be >= 1"):
            MultiQuadRigidbodyController(num_quadrotors=0)

    @pytest.mark.parametrize("name", ["kp", "kd"])
    def test_malformed_position_gain(self, name):
        with pytest.raises(ValueError, match=f"{name} must be a scalar or a 3-vector"):
            MultiQuadRigidbodyController(num_quadrotors=3, **{name: [1.0, 2.0]})

    @pytest.mark.parametrize("name", ["masses", "cable_lengths"])
    def test_wrong_length_per_quadrotor_array(self, name):
        with pytest.raises(ValueError, match=f"{name} must have 3 entries"):
            MultiQuadRigidbodyController(num_quadrotors=3, **{name: [1.0, 1.0]})

    def test_non_positive_masses(self):
        with pytest.raises(ValueError, match="masses must be positive"):
            MultiQuadRigidbodyController(num_quadrotors=3, masses=[1.0, 0.0, 1.0])

    def test_non_positive_cable_lengths(self):
        with pytest.raises(ValueError, match="cable_lengths must be positive"):
            MultiQuadRigidbodyController(num_quadrotors=3, cable_lengths=[1.0, -1.0, 1.0])

    def test_non_positive_payload_mass(self):
        with pytest.raises(ValueError, match="payload_mass must be positive"):
            MultiQuadRigidbodyController(num_quadrotors=3, payload_mass=0.0)

    def test_malformed_payload_inertia(self):
        with pytest.raises(ValueError, match=r"payload_inertia must be \(3,\) or \(3, 3\)"):
            MultiQuadRigidbodyController(num_quadrotors=3, payload_inertia=np.zeros((2, 2)))

    def test_malformed_attachment_points(self):
        with pytest.raises(ValueError, match=r"attachment_points must be \(3, 3\)"):
            MultiQuadRigidbodyController(num_quadrotors=3, attachment_points=np.zeros((4, 3)))

    def test_malformed_hover_cable_attitudes(self):
        with pytest.raises(ValueError, match=r"hover_cable_attitudes must be \(3, 3\)"):
            MultiQuadRigidbodyController(num_quadrotors=3, hover_cable_attitudes=np.zeros((2, 3)))

    def test_policy_built_for_a_different_team_size(self):
        policy = FixedTensionDistribution(np.tile([0.0, 0.0, -1.0], (4, 1)))
        with pytest.raises(ValueError, match="built for 4 quadrotors"):
            MultiQuadRigidbodyController(num_quadrotors=3, tension_policy=policy)

    def test_assigned_policy_built_for_a_different_team_size(self):
        ctrl = MultiQuadRigidbodyController(num_quadrotors=3)
        policy = FixedTensionDistribution(np.tile([0.0, 0.0, -1.0], (5, 1)))
        with pytest.raises(ValueError, match="built for 5 quadrotors"):
            ctrl.tension_policy = policy

    def test_a_geometry_with_a_zero_hover_tension_is_refused(self):
        """A cable carrying nothing leaves the null-space bias undefined.

        The allocation needs a non-zero ``τ̄_i`` per cable: it is what fixes
        cable ``i``'s reference direction *and* the sign test that keeps the
        split on the taut side of slack. A trim that loads a cable with nothing
        therefore has to be refused at construction rather than discovered as a
        ``nan`` cable direction mid-flight.

        A degenerate hover direction is the reachable way in — its column of
        the hover wrench map is zero, so the least-squares trim puts exactly
        zero tension on it.
        """
        directions = np.array(
            [
                [0.0, 0.0, -1.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        )
        with pytest.raises(ValueError, match="zero tension"):
            MultiQuadRigidbodyController(num_quadrotors=3, hover_cable_attitudes=directions)

    def test_sync_from_a_different_team_size(self):
        ctrl = MultiQuadRigidbodyController(num_quadrotors=3)
        other = MultiQuadRigidbodyBase(num_quadrotors=4)
        with pytest.raises(ValueError, match="cannot sync a 3-quadrotor controller"):
            ctrl.sync_from_model(other)


# ─── Hover ────────────────────────────────────────────────────────────


class TestHoverEquilibrium:
    @pytest.mark.parametrize("nQ", [1, 2, 3, 4, 5, 6])
    def test_allocation_is_exact_at_hover(self, nQ):
        """At the hover wrench the split returns the bias identically.

        Not approximately: ``Φ τ̄ = W̄`` for a feasible trim, so the null-space
        term of the split is exactly what is left. That is what makes the
        controller's reference formation the model's own ``reset()`` formation
        rather than a nearby one, and it is the reason the bias is the ring
        trim instead of the least-norm ``τ̄ = 0``.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        ctrl = mdl.payload_controller
        assert ctrl.hover_residual < 1e-12

        forces = ctrl.allocate(GRAVITY * _E3, np.zeros(3), np.eye(3))
        expected = -np.asarray(ctrl.tension_policy.bias_forces)
        np.testing.assert_allclose(forces, expected, atol=TOL_ANALYTIC)
        assert ctrl.unachievable_wrenches == 0

    @pytest.mark.parametrize("nQ", [1, 2, 3, 4, 5, 6])
    def test_hover_tensions_match_a_from_scratch_trim(self, nQ):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(
            mdl.payload_controller.hover_tensions, hover_trim_tensions(mdl), atol=TOL_ANALYTIC
        )

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_commanded_thrust_is_the_equilibrium_thrust(self, nQ):
        """``u_i = m_i g e₃ - 𝐓̄_i`` at the trim — the whole cascade folds up."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        target = np.array([0.0, 0.0, 1.0])
        mdl.reset(payload_position=target)
        ctrl = mdl.payload_controller
        ctrl.setpoint = hover_setpoint(target)

        tensions = hover_trim_tensions(mdl)
        q = np.array([np.asarray(x) for x in mdl.state.cable_attitudes])
        expected = np.array([mdl.masses[i] * GRAVITY * _E3 - tensions[i] * q[i] for i in range(nQ)])
        np.testing.assert_allclose(ctrl.compute(0.0, mdl.state), expected, atol=TOL_ANALYTIC)


# ─── Gains ────────────────────────────────────────────────────────────


class TestGains:
    def test_gains_are_the_shipped_defaults(self):
        ctrl = MultiQuadRigidbodyController(num_quadrotors=4)
        np.testing.assert_allclose(ctrl.position_gains.kp, DEFAULT_MULTI_PAYLOAD_POS_KP)
        np.testing.assert_allclose(ctrl.position_gains.kd, DEFAULT_MULTI_PAYLOAD_POS_KD)
        np.testing.assert_allclose(ctrl.attitude_gains.kp, DEFAULT_RIGID_PAYLOAD_ATT_KP)
        np.testing.assert_allclose(ctrl.attitude_gains.kd, DEFAULT_RIGID_PAYLOAD_ATT_KD)
        np.testing.assert_allclose(ctrl.cable_gains.kp, DEFAULT_MULTI_PAYLOAD_CABLE_KP)
        np.testing.assert_allclose(ctrl.cable_gains.kd, DEFAULT_MULTI_PAYLOAD_CABLE_KD)

    def test_gains_are_copies(self):
        """Tuning one controller must not move the package defaults."""
        ctrl = MultiQuadRigidbodyController(num_quadrotors=4)
        ctrl.attitude_gains.kp[0] += 5.0
        ctrl.cable_gains.kd[1] += 5.0
        assert DEFAULT_RIGID_PAYLOAD_ATT_KP[0] != ctrl.attitude_gains.kp[0]
        assert DEFAULT_MULTI_PAYLOAD_CABLE_KD[1] != ctrl.cable_gains.kd[1]

    def test_kp_and_kd_reach_the_position_loop(self):
        """The documented constructor arguments drive the command.

        ``PDController.__init__`` parks ``kp``/``kd`` in ``self._gains``, which
        ``compute()`` never reads; the controller routes them into the position
        loop instead, and this is the test that the routing is live.
        """
        ctrl = MultiQuadRigidbodyController(num_quadrotors=4, kp=7.0, kd=2.0)
        np.testing.assert_allclose(ctrl.position_gains.kp, np.full(3, 7.0))
        np.testing.assert_allclose(ctrl.position_gains.kd, np.full(3, 2.0))
        assert ctrl._gains is ctrl.position_gains

    def test_tuning_the_attitude_gains_moves_the_moment(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        mdl.reset(
            payload_position=np.array([0.0, 0.0, 1.0]),
            payload_orientation=SO3(rodrigues_expm(np.array([0.1, -0.08, 0.25]))),
        )
        ctrl = mdl.payload_controller
        ctrl.setpoint = hover_setpoint(np.array([0.0, 0.0, 1.0]))
        _, moment_before, _ = ctrl.desired_wrench(0.0, mdl.state)
        ctrl.attitude_gains.kp[:] = 2.0 * np.asarray(ctrl.attitude_gains.kp)
        _, moment_after, _ = ctrl.desired_wrench(0.0, mdl.state)
        assert float(np.linalg.norm(moment_after - moment_before)) > 1e-3


# ─── Closed loop ──────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
class TestPayloadPoseTracking:
    @pytest.mark.parametrize("nQ", CONTROLLABLE_COUNTS)
    def test_position_and_attitude_converge(self, nQ):
        """Both halves of the SE(3) pose, from an off-target, off-level start."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        target = np.array([1.0, -0.5, 1.5])
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.simulate(
            tf=12.0,
            payload_position=np.array([0.0, 0.0, 1.0]),
            payload_orientation=SO3(rodrigues_expm(np.array([0.1, -0.08, 0.25]))),
        )
        assert float(np.linalg.norm(mdl.state.payload_position - target)) < TOL_PHYSICAL
        assert attitude_angle_deg(mdl.state.payload_orientation, np.eye(3)) < 1.0
        assert float(np.linalg.norm(np.asarray(mdl.state.payload_angular_velocity))) < TOL_PHYSICAL

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_tracks_a_moving_trajectory(self, nQ):
        traj = brisk_spiral()
        result = fly(MultiQuadRigidbodyBase(num_quadrotors=nQ), traj, traj.duration, True)
        assert result["rms"] < 0.5


@pytest.mark.slow
@pytest.mark.integration
class TestAttitudeSetpoint:
    """The cascade's headline feature, flown rather than merely read back.

    Before this file ``attitude_setpoint`` was never set to a non-default value
    anywhere in the repository, so every closed-loop test commanded ``R_d = I``
    and the whole reference branch of the attitude loop was dead code as far as
    the suite was concerned.
    """

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_holds_a_commanded_yaw(self, nQ):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        target = np.array([0.0, 0.0, 1.2])
        R_d = rodrigues_expm(0.4 * _E3)
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.payload_controller.attitude_setpoint = fixed_attitude(R_d)
        mdl.simulate(tf=15.0, payload_position=target)

        R = np.asarray(mdl.state.payload_orientation)
        # It really went there, and it really is not level: without the second
        # assertion "converged" would be satisfied by never moving.
        assert attitude_angle_deg(R, R_d) < 2.0
        assert attitude_angle_deg(R, np.eye(3)) > 15.0
        assert float(np.linalg.norm(mdl.state.payload_position - target)) < 5 * TOL_PHYSICAL

    def test_holds_a_commanded_tilt(self):
        """An off-axis target, so the moment command is not a single component."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        target = np.array([0.0, 0.0, 1.2])
        R_d = rodrigues_expm(np.array([0.12, -0.09, 0.2]))
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.payload_controller.attitude_setpoint = fixed_attitude(R_d)
        mdl.simulate(tf=15.0, payload_position=target)

        R = np.asarray(mdl.state.payload_orientation)
        assert attitude_angle_deg(R, R_d) < 2.0
        assert attitude_angle_deg(R, np.eye(3)) > 8.0

    def test_tracks_a_moving_attitude_reference(self):
        """A yaw ramp, so ``Ω_d`` and ``Ω̇_d`` carry content.

        A step in ``R_d`` leaves both reference rates at zero and exercises only
        the proportional channel; the quintic here does not, so the
        ``-J_L(Ω̂_L R_Lᵀ R_d Ω_d - R_Lᵀ R_d Ω̇_d)`` term of
        :meth:`desired_wrench` and the ``R_L(Ω̇_d × r_i)`` feedforward of the
        thrust law are both live.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        target = np.array([0.0, 0.0, 1.2])
        ramp = YawRamp(angle=0.6, duration=6.0)
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.payload_controller.attitude_setpoint = ramp.get
        mdl.simulate(tf=14.0, payload_position=target)

        R = np.asarray(mdl.state.payload_orientation)
        assert attitude_angle_deg(R, ramp.get(mdl.t)[0]) < 2.0
        assert attitude_angle_deg(R, np.eye(3)) > 25.0

    def test_the_reference_angular_acceleration_is_transported_into_the_body_frame(self):
        """``desired_wrench``'s third output is ``R_Lᵀ R_d Ω̇_d``, not ``Ω̇_d``.

        The module docstring spends a paragraph on why the transport matters —
        ``Ω̇_d`` is expressed in the reference frame ``R_d`` while its only
        consumer crosses it with the body-frame ``r_i`` — but dropping the two
        rotations survived every other test in this file: the closed-loop
        trajectories all drive ``R_L`` towards ``R_d``, where the two agree,
        and the discrepancy along the way is a feedforward-sized fraction of a
        converging run.

        Pinned open-loop instead, at an attitude error the transport cannot be
        confused with round-off at. The asserted value is the *definition*, so
        the test states what the code should compute rather than what it does;
        the second half checks the mutant is actually a different number.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        target = np.array([0.0, 0.0, 1.2])
        mdl.reset(payload_position=target)
        # 0.4 rad of yaw between the payload and its reference, the case the
        # docstring costs out at 0.03 m/s² of misdirected acceleration.
        mdl.state.payload_orientation = rodrigues_expm(0.4 * _E3)
        R_d = rodrigues_expm(np.array([0.15, -0.25, 0.1]))
        dOmega_d = np.array([0.37, -0.21, 0.44])

        ctrl = mdl.payload_controller
        ctrl.setpoint = hover_setpoint(target)
        ctrl.attitude_setpoint = lambda t: (R_d, np.zeros(3), dOmega_d)

        _, _, ref_ang_accel = ctrl.desired_wrench(0.0, mdl.state)
        R_L = np.asarray(mdl.state.payload_orientation)
        np.testing.assert_allclose(ref_ang_accel, R_L.T @ R_d @ dOmega_d, atol=TOL_ANALYTIC)
        # The untransported vector is a materially different command, so the
        # assertion above is not satisfied by both.
        assert float(np.linalg.norm(ref_ang_accel - dOmega_d)) > 0.1

    def test_the_transport_reaches_the_thrust_command(self):
        """... and the transported vector is what ``compute`` actually crosses with ``r_i``.

        The assertion above pins ``desired_wrench``; this one pins that its
        third output is the vector the thrust law uses, by reconstructing the
        commanded thrust from the published pieces. Without it the transport
        could be correct and then discarded.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        target = np.array([0.0, 0.0, 1.2])
        mdl.reset(payload_position=target)
        mdl.state.payload_orientation = rodrigues_expm(0.4 * _E3)
        R_d = rodrigues_expm(np.array([0.15, -0.25, 0.1]))
        dOmega_d = np.array([0.37, -0.21, 0.44])

        ctrl = mdl.payload_controller
        ctrl.setpoint = hover_setpoint(target)
        ctrl.attitude_setpoint = lambda t: (R_d, np.zeros(3), dOmega_d)

        out = ctrl.compute(0.0, mdl.state)
        accel_plus_g, moment, ref_ang_accel = ctrl.desired_wrench(0.0, mdl.state)
        R_L = np.asarray(mdl.state.payload_orientation)
        cable_forces = ctrl.allocate(accel_plus_g, moment, R_L)

        for i in range(mdl.num_quadrotors):
            m_i = float(mdl.masses[i])
            r_i = ctrl.attachment_points[i]
            acc_i = accel_plus_g + R_L @ np.cross(ref_ang_accel, r_i)
            q = np.asarray(mdl.state.cable_attitudes[i])
            q_hat = hat(q)
            qd = cable_forces[i] / np.linalg.norm(cable_forces[i])
            u_perp = (m_i * float(mdl.cable_lengths[i])) * (
                ctrl.cable_gains.kp * (q_hat @ q_hat @ qd)
            ) - m_i * (q_hat @ q_hat @ acc_i)
            expected = np.dot(m_i * acc_i - cable_forces[i], q) * q + u_perp
            np.testing.assert_allclose(out[i], expected, atol=TOL_ANALYTIC)

        # Cross-check that the term is doing something: the untransported
        # vector would give a visibly different thrust.
        for i in range(mdl.num_quadrotors):
            r_i = ctrl.attachment_points[i]
            assert (
                float(
                    np.linalg.norm(
                        R_L @ np.cross(ref_ang_accel, r_i) - R_L @ np.cross(dOmega_d, r_i)
                    )
                )
                > 1e-2
            )


@pytest.mark.slow
@pytest.mark.integration
class TestUnderactuatedGrasp:
    """``nQ <= 2``: the moment is structurally unavailable, and says so."""

    @pytest.mark.parametrize("nQ", UNDERACTUATED_COUNTS)
    def test_rank_deficiency_is_warned_at_construction(self, nQ, caplog):
        with caplog.at_level(logging.WARNING, logger=_CONTROL_LOGGER):
            ctrl = MultiQuadRigidbodyController(num_quadrotors=nQ)
        assert ctrl.grasp_rank < 6
        assert "structurally uncontrollable" in caplog.text

    @pytest.mark.parametrize("nQ", UNDERACTUATED_COUNTS)
    def test_an_attitude_command_is_dropped_and_counted(self, nQ, caplog):
        """Position still converges; the attitude does not, and it is reported.

        This is the shape of the failure that made it invisible — the obvious
        number looks fine. The counter and the warning are what has to keep
        working.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        target = np.array([0.0, 0.0, 1.2])
        R_d = rodrigues_expm(np.array([0.2, -0.15, 0.3]))
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.payload_controller.attitude_setpoint = fixed_attitude(R_d)

        with caplog.at_level(logging.WARNING, logger=_CONTROL_LOGGER):
            mdl.simulate(tf=10.0, payload_position=target)

        assert float(np.linalg.norm(mdl.state.payload_position - target)) < 5 * TOL_PHYSICAL
        assert attitude_angle_deg(mdl.state.payload_orientation, R_d) > 5.0
        assert mdl.payload_controller.unachievable_wrenches > 0
        assert "outside the range of the grasp map" in caplog.text

    def test_the_range_warning_fires_once_per_controller(self, caplog):
        """A per-step warning would bury the run; the counter carries the total."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=2)
        target = np.array([0.0, 0.0, 1.2])
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.payload_controller.attitude_setpoint = fixed_attitude(
            rodrigues_expm(np.array([0.2, -0.15, 0.3]))
        )
        mdl.reset(payload_position=target)
        with caplog.at_level(logging.WARNING, logger=_CONTROL_LOGGER):
            for _ in range(200):
                mdl.step(mdl.compute_default_input())
        messages = [r for r in caplog.records if "outside the range" in r.getMessage()]
        assert len(messages) == 1
        assert mdl.payload_controller.unachievable_wrenches == 200


# ─── The tension policy ───────────────────────────────────────────────


class TestTensionPolicy:
    def test_default_policy_is_the_hover_trim(self):
        """``τ̄_i = -t_i q̄_i`` for the model's own ``reset()`` formation."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        ctrl = mdl.payload_controller
        tensions = hover_trim_tensions(mdl)
        q = np.array([np.asarray(x) for x in mdl.state.cable_attitudes])
        np.testing.assert_allclose(
            ctrl.tension_policy.bias_forces, -tensions[:, None] * q, atol=TOL_ANALYTIC
        )

    def test_controller_and_flat_map_share_one_allocation(self):
        """The same ``Φ⁺`` and the same null-space bias on both sides.

        ``Φ⁺ W`` is only half the split. A feedforward built at a different
        bias is the reference for a different formation, and the two disagree
        without either of them being wrong on its own — which is exactly the
        kind of mismatch that shows up as a mediocre tracking number and
        nothing else.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        ctrl = mdl.payload_controller
        flat_map = MultiQuadRigidbody.from_model(mdl, tension_policy=ctrl.tension_policy)
        assert flat_map.tension_policy is ctrl.tension_policy
        np.testing.assert_allclose(flat_map.allocation.grasp, ctrl.grasp, atol=TOL_EXACT)
        np.testing.assert_allclose(
            flat_map.allocation.grasp_pinv, ctrl.allocation.grasp_pinv, atol=TOL_EXACT
        )

    def test_an_assigned_policy_moves_the_split(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        ctrl = mdl.payload_controller
        before = ctrl.allocate(GRAVITY * _E3, np.zeros(3), np.eye(3))

        # A tighter ring: a genuinely different null-space bias.
        ctrl.tension_policy = RingTensionDistribution(
            num_quadrotors=4,
            payload_mass=mdl.payload_mass,
            cable_lengths=mdl.cable_lengths,
            attachment_points=mdl.attachment_points,
            formation_radius=0.3,
        )
        after = ctrl.allocate(GRAVITY * _E3, np.zeros(3), np.eye(3))
        assert float(np.linalg.norm(after - before)) > 0.1

    def test_an_assigned_policy_survives_a_model_resync(self):
        """A caller-supplied policy is the caller's; a default one is ours."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        ctrl = mdl.payload_controller
        assigned = RingTensionDistribution(
            num_quadrotors=4,
            payload_mass=mdl.payload_mass,
            cable_lengths=mdl.cable_lengths,
            attachment_points=mdl.attachment_points,
            formation_radius=0.3,
        )
        ctrl.tension_policy = assigned
        mdl.payload_mass = 1.5
        assert ctrl.tension_policy is assigned

        default_ctrl = MultiQuadRigidbodyBase(num_quadrotors=4).payload_controller
        before = np.asarray(default_ctrl.tension_policy.bias_forces).copy()
        default_ctrl.sync_from_model(MultiQuadRigidbodyBase(num_quadrotors=4, payload_mass=2.0))
        assert float(np.linalg.norm(default_ctrl.tension_policy.bias_forces - before)) > 0.1


# ─── Flatness feedforward ─────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
class TestFlatnessFeedforward:
    """A trajectory that grows higher derivatives has to *measurably* help."""

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_feedforward_improves_tracking(self, nQ):
        traj = brisk_spiral()
        with_ff = fly(MultiQuadRigidbodyBase(num_quadrotors=nQ), traj, traj.duration, True)
        without = fly(MultiQuadRigidbodyBase(num_quadrotors=nQ), traj, traj.duration, False)

        # Two-sided: the feedback-only run has to be bad enough that there is
        # something to improve, and the feedforward run substantially better
        # rather than merely not worse.
        assert without["rms"] > 0.3, "trajectory too easy — the comparison proves nothing"
        assert with_ff["rms"] < 0.85 * without["rms"]
        assert with_ff["max"] < without["max"]

    def test_simulate_itself_runs_the_feedforward(self, monkeypatch):
        """The wiring inside :meth:`simulate`, not :func:`fly`'s copy of it.

        ``fly`` re-implements the stepping loop, so every other test in this
        file would stay green with the feedforward branch of ``simulate()``
        deleted. This one binds a bound ``get_full``, goes through
        ``simulate``, and demands both that the recovered attitudes reach every
        step and that the run tracks better than the feedback-only one.
        """
        traj = brisk_spiral()
        applied: list[int] = []
        original = MultiQuadRigidbodyBase._desired_attitudes

        def spy(self, flat_map, flat_fn, t):
            out = original(self, flat_map, flat_fn, t)
            applied.append(0 if out is None else len(out))
            return out

        monkeypatch.setattr(MultiQuadRigidbodyBase, "_desired_attitudes", spy)

        def run(feedforward):
            mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
            mdl.payload_controller.setpoint = (
                traj.get_full if feedforward else (lambda t: traj.get_full(t)[:3])
            )
            mdl.simulate(tf=traj.duration, payload_position=traj.get(0.0)[0])
            return mdl, float(np.linalg.norm(mdl.state.payload_position - traj.get(mdl.t)[0]))

        mdl_ff, err_ff = run(True)
        assert len(applied) > 100, "simulate() never asked for a feedforward attitude"
        assert set(applied) == {4}, "some step ran without the recovered attitudes"
        assert mdl_ff.feedforward_drops == 0

        applied.clear()
        _, err_fb = run(False)
        assert not applied, "the plain-closure setpoint must not unlock the feedforward"

        assert err_fb > 0.3, "trajectory too easy — the comparison proves nothing"
        assert err_ff < 0.85 * err_fb

    def test_feedforward_is_skipped_when_the_policy_is_not_published(self, caplog):
        """No ``tension_policy`` on the controller means no feedforward at all.

        Falling back on a default would build the reference at *this model's*
        ring while the loop allocates at whatever bias the controller chose —
        precisely the feedback/feedforward disagreement the shared policy
        exists to rule out, and silent when it happens.
        """

        class Opaque:
            """A duck-typed payload controller that publishes no split."""

            def __init__(self, inner):
                self._inner = inner

            @property
            def setpoint(self):
                return self._inner.setpoint

            @setpoint.setter
            def setpoint(self, fn):
                self._inner.setpoint = fn

            def compute(self, *args):
                return self._inner.compute(*args)

        traj = brisk_spiral()
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        inner = MultiQuadRigidbodyController.from_model(mdl)
        # Allocates on a tighter ring than the model's — the mismatch a default
        # policy would paper over.
        inner.tension_policy = RingTensionDistribution(
            num_quadrotors=4,
            payload_mass=mdl.payload_mass,
            cable_lengths=mdl.cable_lengths,
            attachment_points=mdl.attachment_points,
            formation_radius=0.3,
        )
        mdl._payload_controller = Opaque(inner)
        mdl.payload_controller.setpoint = traj.get_full

        # The sniffing still finds a ``get_full``; only the policy is missing.
        assert mdl._flatness_feedforward_fn() is not None
        with caplog.at_level(logging.WARNING, logger=_MODEL_LOGGER):
            mdl.simulate(tf=0.5, payload_position=traj.get(0.0)[0])

        assert "tension_policy" in caplog.text
        assert np.all(np.isfinite(mdl.state.payload_position))

    def test_an_underactuated_grasp_drops_the_feedforward_and_says_so(self, caplog):
        """At ``nQ = 2`` the flat map refuses part of the reference.

        The refusal is correct — the recovered reference would not solve the
        payload's equations of motion — and the right response is to run that
        step feedback-only rather than to abort. What is required is that it is
        not *silent*: one warning per run, then a running count. Silently
        degrading is how a mis-specified team gets mistaken for a badly tuned
        loop.
        """
        traj = brisk_spiral()
        mdl = MultiQuadRigidbodyBase(num_quadrotors=2)
        mdl.payload_controller.setpoint = traj.get_full
        mdl.payload_controller.attitude_setpoint = fixed_attitude(rodrigues_expm(0.3 * _E3))

        with caplog.at_level(logging.WARNING, logger=_MODEL_LOGGER):
            mdl.simulate(tf=1.0, payload_position=traj.get(0.0)[0])

        steps = int(round(mdl.t / mdl.dt))
        assert 0 < mdl.feedforward_drops <= steps
        warnings = [r for r in caplog.records if "feedforward dropped" in r.getMessage()]
        assert len(warnings) == 1, "the drop warning must fire once per run, not once per step"
        assert np.all(np.isfinite(mdl.state.payload_position))

        # A rank-6 grasp on the same trajectory drops nothing — so the count
        # above is the geometry, not the trajectory.
        healthy = MultiQuadRigidbodyBase(num_quadrotors=4)
        healthy.payload_controller.setpoint = traj.get_full
        healthy.simulate(tf=1.0, payload_position=traj.get(0.0)[0])
        assert healthy.feedforward_drops == 0

    def test_an_infeasible_reference_is_declined_rather_than_crashed(self):
        """A reference that would slacken a cable returns ``None``, not an error."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        flat_map = MultiQuadRigidbody.from_model(
            mdl, tension_policy=mdl.payload_controller.tension_policy
        )

        def violent(t):
            # Straight down, hard enough that the cables would have to push.
            return (np.zeros(3), np.zeros(3), -50.0 * _E3, np.zeros(3), np.zeros(3))

        assert mdl._desired_attitudes(flat_map, violent, 0.0) is None
        assert mdl.feedforward_drops == 1


# ─── Heterogeneous teams ──────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
class TestHeterogeneousTeam:
    def test_unequal_masses_and_cable_lengths_still_track(self):
        mdl = MultiQuadRigidbodyBase(
            num_quadrotors=4,
            masses=[0.6, 0.9, 1.1, 1.4],
            cable_lengths=[1.0, 1.2, 0.9, 1.1],
        )
        # The trim must genuinely come out unequal, or this is a homogeneous
        # test wearing a disguise.
        tensions = mdl.payload_controller.hover_tensions
        assert len(set(np.round(tensions, 6))) > 1

        target = np.array([0.8, 0.6, 1.6])
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.simulate(tf=12.0, payload_position=np.array([0.0, 0.0, 1.0]))
        assert float(np.linalg.norm(mdl.state.payload_position - target)) < TOL_PHYSICAL
        assert attitude_angle_deg(mdl.state.payload_orientation, np.eye(3)) < 2.0

    def test_an_asymmetric_grasp_still_holds_the_pose(self):
        """Attachment points off the ring — the grasp is no longer symmetric."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        mdl.attachment_points = np.array(
            [
                [0.45, 0.10, 0.05],
                [-0.05, 0.50, 0.05],
                [-0.50, -0.10, 0.05],
                [0.10, -0.45, 0.05],
            ]
        )
        target = np.array([0.3, -0.2, 1.4])
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.simulate(tf=12.0, payload_position=np.array([0.0, 0.0, 1.0]))
        assert float(np.linalg.norm(mdl.state.payload_position - target)) < 5 * TOL_PHYSICAL
        assert attitude_angle_deg(mdl.state.payload_orientation, np.eye(3)) < 5.0

    def test_a_heterogeneous_team_tracks_with_feedforward(self):
        traj = brisk_spiral()
        mdl = MultiQuadRigidbodyBase(
            num_quadrotors=4,
            masses=[0.8, 1.0, 0.9, 1.1],
            cable_lengths=[1.0, 1.1, 0.95, 1.05],
        )
        result = fly(mdl, traj, traj.duration, True)
        assert result["rms"] < 0.5
