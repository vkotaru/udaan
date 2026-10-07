"""Tests for :mod:`udaan.control.multi_quad_cspayload` — the M3 payload cascade.

The controller is the generic-``nQ`` sibling of M2's
:class:`~udaan.control.quadrotor_cspayload.QuadCSPayloadController`, and it is
wired into :meth:`MultiQuadCsPayloadBase.simulate` as the default. What is
worth testing therefore splits in two:

* the **algebra at hover** — the hover identity below is exact, and it is the
  check that pins the one term M2 does not have. With one cable the *plant's*
  payload acceleration is always parallel to it, so ``-m_i q̂_i²(a_L + g e₃)``
  is zero at the reference and M2 may omit it; with ``nQ > 1`` the cables are
  tilted and it is exactly the lateral force each quadrotor needs to hold its
  own weight. Assert the commanded thrust equals ``m_i g e₃ - 𝐓̄_i`` and the
  term cannot be dropped without the test going red. (The term is evaluated at
  the *measured* ``q_i``, so it does not vanish identically at ``nQ = 1`` —
  only where the cable-attitude error is zero.)
* the **algebra off it** — four of the five terms are zero at the ring
  equilibrium, so :class:`TestThrustLawTerms` re-checks the law from a state
  with cable error, cable rates and reference jerk/snap, where all of them are
  live.
* the **closed loop** — convergence, tracking, cable tension and the flatness
  feedforward, all of which need a simulation. :meth:`simulate` is exercised
  directly as well as through :func:`fly`, which re-implements its loop.

Cable tension is not a state variable of the analytical model (the cable is a
rigid constraint), so :func:`cable_tensions` re-solves the model's own payload
equation to recover it. ``T_i > 0`` is taut; ``T_i ≤ 0`` means the constraint
is pushing, which a real cable cannot do.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_PHYSICAL
from udaan.control.multi_quad_cspayload import MultiQuadCsPayloadController
from udaan.control.multi_quad_cspayload import payload_controller as pc_module
from udaan.core.defaults import (
    DEFAULT_MULTI_PAYLOAD_CABLE_KD,
    DEFAULT_MULTI_PAYLOAD_CABLE_KP,
    DEFAULT_MULTI_PAYLOAD_POS_KD,
    DEFAULT_MULTI_PAYLOAD_POS_KP,
    GRAVITY,
)
from udaan.flatness import (
    AffineTensionSplit,
    MultiQuadCsPayload,
    RingTensionSplit,
    TensionSplitContractError,
    tension_direction_jet,
)
from udaan.manif import S2, TS2
from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadBase
from udaan.utils.trajectory import SpiralTrajectory

QUAD_COUNTS = [1, 2, 3, 4, 5]

_E3 = np.array([0.0, 0.0, 1.0])


# ─── Helpers ──────────────────────────────────────────────────────────


def hover_setpoint(x):
    """A stationary payload setpoint at ``x``."""
    target = np.asarray(x, dtype=float).copy()
    return lambda t: (target, np.zeros(3), np.zeros(3))


def gentle_spiral():
    """A helix the cascade tracks comfortably: ~0.7 m/s, ~1 m/s² peak."""
    return SpiralTrajectory(
        start=np.array([0.0, 0.0, 2.0]),
        radius=0.5,
        speed=0.7,
        n_loops=1,
        advance=1.0,
        hold_time=0.5,
    )


def brisk_spiral():
    """Fast enough that the attitude loop visibly lags without feedforward."""
    return SpiralTrajectory(
        start=np.array([0.0, 0.0, 2.0]),
        radius=0.6,
        speed=1.5,
        n_loops=2,
        advance=2.0,
        hold_time=0.5,
    )


def rich_setpoint(t):
    """A setpoint with everything switched on: error, rates, jerk and snap.

    Constant in ``t`` — the thrust law is memoryless, so a frozen sample is
    enough, and it keeps the expected value in :class:`TestThrustLawTerms`
    hand-checkable.
    """
    return (
        np.array([0.0, 0.0, 1.0]),
        np.array([0.1, -0.2, 0.05]),
        np.array([0.4, 0.3, -0.2]),
        np.array([16.0, -12.0, 10.0]),
        np.array([-40.0, 60.0, 24.0]),
    )


def off_equilibrium_model():
    """A three-quadrotor team well away from the ring equilibrium.

    Unequal masses and cable lengths (so ``m_i l_i`` is neither 1 nor shared),
    a displaced and moving payload, cables tilted ~25° off their stations and
    turning at ~1.5 rad/s. Every term of the thrust law is then non-zero.
    """
    mdl = MultiQuadCsPayloadBase(
        num_quadrotors=3,
        masses=[0.7, 1.1, 0.5],
        cable_lengths=[1.3, 0.8, 1.5],
        payload_mass=0.3,
    )
    mdl.reset(payload_position=np.array([0.2, -0.1, 1.4]))
    mdl.state.payload_velocity = np.array([0.3, 0.2, -0.15])
    tilts = [
        np.array([0.45, -0.30, 0.15]),
        np.array([-0.15, 0.60, 0.30]),
        np.array([0.30, 0.30, -0.55]),
    ]
    rates = [
        np.array([1.0, -1.5, 0.5]),
        np.array([-0.75, 1.25, -1.0]),
        np.array([1.5, 0.5, 1.0]),
    ]
    for i in range(3):
        q = np.asarray(mdl.state.cable_attitudes[i], dtype=float) + tilts[i]
        q /= np.linalg.norm(q)
        mdl.state.cable_attitudes[i] = S2(q)
        # Keep ω tangent to the sphere, as the integrator does.
        mdl.state.cable_angular_velocities[i] = TS2(rates[i] - np.dot(rates[i], q) * q)
    mdl._sync_quad_states()
    return mdl


class VerticalSinusoid:
    """A payload trajectory that leaves and re-enters the taut-cable certificate.

    ``x_L(t) = start + A sin(ωt) e₃`` with ``A ω² = factor · bound``, so the
    peak commanded acceleration is ``factor`` times the flat map's certified
    bound while every derivative stays exactly consistent. The excursions are
    around the *downward* peaks, which is the only way to leave the certificate
    on a real ring: the bound is ``g ℓ/Δz > g``, and cables can pull but not
    push.
    """

    def __init__(self, start, bound: float, factor: float = 1.6, omega: float = 8.0):
        self.start = np.asarray(start, dtype=float)
        self.omega = float(omega)
        self.amplitude = factor * float(bound) / self.omega**2

    def get(self, t):
        return self.get_full(t)[:3]

    def get_full(self, t):
        w, a = self.omega, self.amplitude
        sin, cos = np.sin(w * t), np.cos(w * t)
        return (
            self.start + a * sin * _E3,
            a * w * cos * _E3,
            -a * w**2 * sin * _E3,
            -a * w**3 * cos * _E3,
            a * w**4 * sin * _E3,
        )


class LeakyTensionSplit(AffineTensionSplit):
    """A policy with a bug: its first-order split does not sum to ``Ḟ_net``."""

    def split(self, net_force):
        out = super().split(net_force)
        if len(out[0]) > 1:
            out[0][1] = out[0][1] + np.array([0.5, 0.0, 0.0])
        return out


def cable_tensions(mdl, wrench: np.ndarray) -> np.ndarray:
    """Per-cable tension ``T_i = 𝐓_i · q_i`` at the model's current state.

    The cable force is a constraint force, so it is recovered rather than read:
    solve the model's own payload equation for ``a_L + g e₃``, then use Newton
    on quadrotor ``i`` (with ``q_i · q̈_i = -‖q̇_i‖²``) to isolate

        ``T_i = m_i (a_L + g e₃)·q_i + m_i l_i ‖q̇_i‖² - u_i · q_i``.

    Positive means the cable pulls the quadrotor towards the payload — taut.
    """
    nQ = mdl.num_quadrotors
    q = [np.asarray(mdl.state.cable_attitudes[i], dtype=float) for i in range(nQ)]
    mass_matrix = mdl.payload_mass * np.eye(3)
    rhs = np.zeros(3)
    thrust_vectors = []
    for i in range(nQ):
        qqT = np.outer(q[i], q[i])
        mass_matrix += mdl.masses[i] * qqT
        u_i = wrench[i, 0] * (mdl.state.orientations[i] @ _E3)
        thrust_vectors.append(u_i)
        dq = mdl.state.dq(i)
        rhs += qqT @ u_i - mdl.masses[i] * mdl.cable_lengths[i] * np.dot(dq, dq) * q[i]
    accel_plus_g = np.linalg.solve(mass_matrix, rhs)

    out = np.zeros(nQ)
    for i in range(nQ):
        dq = mdl.state.dq(i)
        out[i] = (
            mdl.masses[i] * np.dot(accel_plus_g, q[i])
            + mdl.masses[i] * mdl.cable_lengths[i] * np.dot(dq, dq)
            - np.dot(thrust_vectors[i], q[i])
        )
    return out


def fly(mdl, trajectory, tf, feedforward: bool):
    """Run the cascade against ``trajectory``, returning error and tension stats.

    ``feedforward=True`` binds the trajectory's ``get_full`` — which both hands
    the cable loop its jerk/snap terms *and* trips the flat-map sniffing in
    :meth:`MultiQuadCsPayloadBase.simulate`. ``False`` binds a plain closure
    returning only ``(pos, vel, acc)``, which has no ``__self__`` and so
    switches both off. That is the whole switch, and it is deliberately the
    same one a user flips.

    Note this re-implements ``simulate``'s stepping loop rather than calling it,
    so that the per-step tension can be sampled; nothing here covers the wiring
    *inside* ``simulate``. ``test_simulate_itself_runs_the_feedforward`` does.
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
        MultiQuadCsPayload.from_model(mdl, tension_policy=ctrl.tension_policy)
        if flat_fn is not None
        else None
    )

    errors = []
    min_tension = np.inf
    while mdl.t < tf:
        u = mdl.compute_default_input()
        min_tension = min(min_tension, float(cable_tensions(mdl, mdl._repackage_input(u)).min()))
        desired_att = None if flat_map is None else mdl._desired_attitudes(flat_map, flat_fn, mdl.t)
        mdl.step(u, desired_att=desired_att)
        errors.append(float(np.linalg.norm(mdl.state.payload_position - trajectory.get(mdl.t)[0])))

    errors_arr = np.asarray(errors)
    return {
        "rms": float(np.sqrt(np.mean(errors_arr**2))),
        "max": float(errors_arr.max()),
        "min_tension": min_tension,
    }


# ─── Construction and shape ───────────────────────────────────────────


class TestConstruction:
    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_output_shape(self, nQ):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        u = mdl.payload_controller.compute(0.0, mdl.state)
        assert u.shape == (nQ, 3)
        assert np.all(np.isfinite(u))

    def test_from_model_adopts_model_parameters(self):
        mdl = MultiQuadCsPayloadBase(
            num_quadrotors=3, masses=[0.7, 0.9, 1.2], cable_lengths=[0.9, 1.1, 1.0]
        )
        ctrl = MultiQuadCsPayloadController.from_model(mdl)
        assert ctrl.num_quadrotors == 3
        np.testing.assert_allclose(ctrl._masses, mdl.masses)
        np.testing.assert_allclose(ctrl._cable_lengths, mdl.cable_lengths)
        assert ctrl._payload_mass == mdl.payload_mass

    def test_model_default_controller_is_the_cascade(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        assert isinstance(mdl.payload_controller, MultiQuadCsPayloadController)
        assert mdl.payload_controller.num_quadrotors == 3

    def test_rejects_policy_built_for_a_different_team_size(self):
        policy = RingTensionSplit(
            num_quadrotors=2, payload_mass=0.2, masses=[0.9, 0.9], cable_lengths=[1.0, 1.0]
        )
        with pytest.raises(ValueError, match="built for 2 quadrotors"):
            MultiQuadCsPayloadController(num_quadrotors=3, tension_policy=policy)

    def test_rejects_degenerate_parameters(self):
        with pytest.raises(ValueError, match="num_quadrotors"):
            MultiQuadCsPayloadController(num_quadrotors=0)
        with pytest.raises(ValueError, match="masses must be positive"):
            MultiQuadCsPayloadController(num_quadrotors=2, masses=[1.0, -1.0])
        with pytest.raises(ValueError, match="payload_mass must be positive"):
            MultiQuadCsPayloadController(num_quadrotors=2, payload_mass=0.0)


# ─── Hover algebra ────────────────────────────────────────────────────


class TestHoverEquilibrium:
    """At the ring formation with the setpoint on the payload, the commanded
    thrust must be *exactly* the equilibrium thrust ``m_i g e₃ - 𝐓̄_i``."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_commanded_thrust_is_the_equilibrium_thrust(self, nQ):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        start = np.array([0.3, -0.4, 1.5])
        mdl.reset(payload_position=start)
        ctrl = mdl.payload_controller
        ctrl.setpoint = hover_setpoint(start)

        anchors = ctrl.tension_policy.anchors
        expected = np.array([mdl.masses[i] * GRAVITY * _E3 - anchors[i] for i in range(nQ)])
        np.testing.assert_allclose(ctrl.compute(0.0, mdl.state), expected, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("nQ", [2, 3, 4, 5])
    def test_equilibrium_thrust_is_not_along_the_cable(self, nQ):
        """The lateral-share term is load-bearing, not decorative.

        For ``nQ ≥ 2`` the ring tilts every cable, so the equilibrium thrust has
        a genuine component across it — precisely the ``-m_i q̂_i²(g e₃)`` term.
        A cascade that kept only the parallel channel (as M2's law does, legally
        for one cable) would be short of it, so assert it is large enough that
        its absence could not hide inside numerical noise.
        """
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        start = np.array([0.0, 0.0, 1.5])
        mdl.reset(payload_position=start)
        ctrl = mdl.payload_controller
        ctrl.setpoint = hover_setpoint(start)

        u = ctrl.compute(0.0, mdl.state)
        for i in range(nQ):
            q = np.asarray(mdl.state.cable_attitudes[i], dtype=float)
            perpendicular = u[i] - np.dot(u[i], q) * q
            # m_i * perp_q(g e3) — as a *vector*: a magnitude check alone is
            # sign-blind (a flipped lateral share has the same norm), and the
            # only other assertion that would catch the flip is
            # ``test_commanded_thrust_is_the_equilibrium_thrust``.
            expected = mdl.masses[i] * GRAVITY * (_E3 - np.dot(_E3, q) * q)
            np.testing.assert_allclose(perpendicular, expected, atol=TOL_ANALYTIC)
            assert np.linalg.norm(perpendicular) > 0.5  # newtons — well clear of noise

    def test_single_quadrotor_hovers_straight_up(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=1)
        start = np.array([0.0, 0.0, 1.0])
        mdl.reset(payload_position=start)
        mdl.payload_controller.setpoint = hover_setpoint(start)
        u = mdl.payload_controller.compute(0.0, mdl.state)
        expected = (mdl.masses[0] + mdl.payload_mass) * GRAVITY * _E3
        np.testing.assert_allclose(u[0], expected, atol=TOL_ANALYTIC)


# ─── Off-equilibrium algebra ──────────────────────────────────────────


class TestThrustLawTerms:
    """Every term of the thrust law, pinned away from the ring equilibrium.

    :class:`TestHoverEquilibrium` is exact but blind to anything that is zero
    at hover, which is four of the five terms; the closed-loop tolerances have
    enough headroom to swallow them too. Measured: dropping the ``m_i l_i``
    factor from the cable PD, or deleting the ``q̇^d``, ``q̈^d`` or centripetal
    terms outright, leaves every other test in this file green.

    So fix a state with cable-attitude error, cable rates, payload error *and*
    reference jerk/snap, and check the command against a hand-written expansion
    of the two channels. The expansion re-derives the tension split from the
    policy's ``anchors``/``shares`` and writes ``q̂²`` as ``q qᵀ - I`` rather
    than as ``hat(q) @ hat(q)``, so it is not a transcription of ``compute``;
    what it borrows is :func:`~udaan.flatness.tension_direction_jet`, which has
    its own tests.
    """

    @staticmethod
    def _terms(ctrl, mdl, i, t=0.0):
        """``(parallel, pd, gyro, ffwd, lateral, centripetal)`` for cable ``i``."""
        xd, vd, ad, jerk, snap = ctrl.setpoint(t)
        mL = mdl.payload_mass
        accel = (
            ad
            - ctrl.position_gains.kp * (mdl.state.payload_position - xd)
            - ctrl.position_gains.kd * (mdl.state.payload_velocity - vd)
        )
        accel_plus_g = accel + GRAVITY * _E3
        net = [-mL * accel_plus_g, -mL * jerk, -mL * snap]

        policy = ctrl.tension_policy
        share = policy.shares[i]
        force = [policy.anchors[i] + share * (net[0] - policy.reference_net_force)]
        force += [share * net[k] for k in (1, 2)]
        _, jet = tension_direction_jet(force)
        qd, dqd, d2qd = jet[0], jet[1], jet[2]

        m_i = float(mdl.masses[i])
        l_i = float(mdl.cable_lengths[i])
        q = np.asarray(mdl.state.cable_attitudes[i], dtype=float)
        dq = mdl.state.dq(i)
        q_hat2 = np.outer(q, q) - np.eye(3)  # q̂² for unit q

        err_q = q_hat2 @ qd
        err_dq = dq - np.cross(np.cross(qd, dqd), q)
        centripetal = m_i * l_i * np.dot(dq, dq)
        return {
            "parallel": (m_i * np.dot(accel_plus_g, q) - np.dot(force[0], q) + centripetal) * q,
            "pd": (m_i * l_i) * (ctrl.cable_gains.kp * err_q + ctrl.cable_gains.kd * err_dq),
            "gyro": -(m_i * l_i) * np.dot(q, np.cross(qd, dqd)) * np.cross(q, dq),
            "ffwd": -(m_i * l_i) * np.cross(np.cross(qd, d2qd), q),
            "lateral": -m_i * (q_hat2 @ accel_plus_g),
            "centripetal": centripetal,
        }

    def _controller(self, mdl):
        ctrl = MultiQuadCsPayloadController.from_model(mdl)
        ctrl.setpoint = rich_setpoint
        return ctrl

    def test_command_is_the_sum_of_its_terms(self):
        mdl = off_equilibrium_model()
        ctrl = self._controller(mdl)
        u = ctrl.compute(0.0, mdl.state)
        for i in range(mdl.num_quadrotors):
            terms = self._terms(ctrl, mdl, i)
            expected = terms["parallel"] + terms["pd"] + terms["gyro"] + terms["ffwd"]
            expected = expected + terms["lateral"]
            np.testing.assert_allclose(u[i], expected, atol=TOL_ANALYTIC)

    def test_the_two_channels_are_separately_right(self):
        """Parallel and perpendicular are checked apart, not just in sum.

        Four of the five terms are perpendicular to ``q_i`` by construction and
        the fifth is parallel to it, so the split is a genuine decomposition:
        an error that leaked from one channel into the other would survive a
        check on the total but not this one.
        """
        mdl = off_equilibrium_model()
        ctrl = self._controller(mdl)
        u = ctrl.compute(0.0, mdl.state)
        for i in range(mdl.num_quadrotors):
            q = np.asarray(mdl.state.cable_attitudes[i], dtype=float)
            terms = self._terms(ctrl, mdl, i)

            assert np.dot(u[i], q) == pytest.approx(np.dot(terms["parallel"], q), abs=TOL_ANALYTIC)
            perpendicular = u[i] - np.dot(u[i], q) * q
            expected = terms["pd"] + terms["gyro"] + terms["ffwd"] + terms["lateral"]
            np.testing.assert_allclose(perpendicular, expected, atol=TOL_ANALYTIC)

    def test_every_term_is_live_in_this_state(self):
        """The state is discriminating: no term is quietly zero here.

        Without this the two tests above would pass on a law missing whichever
        term happened to vanish, which is exactly how the equilibrium tests
        miss them.
        """
        mdl = off_equilibrium_model()
        ctrl = self._controller(mdl)
        peaks = dict.fromkeys(("pd", "gyro", "ffwd", "lateral", "centripetal"), 0.0)
        for i in range(mdl.num_quadrotors):
            terms = self._terms(ctrl, mdl, i)
            for name in peaks:
                size = float(np.linalg.norm(terms[name]))
                # 0.02 N is eight orders of magnitude above the tolerance the
                # sum is checked at, so deleting any term goes red.
                assert size > 0.02, f"{name} vanishes on cable {i} — state not discriminating"
                peaks[name] = max(peaks[name], size)
        for name, size in peaks.items():
            assert size > 1.0, f"{name} never exceeds 1 N — the check is close to vacuous"

    def test_dropping_the_m_l_factor_would_be_caught_here(self):
        """The cable PD is ``m_i l_i (k_p e_q + k_d e_q̇)``, not ``(k_p e_q + …)``.

        ``m_i l_i`` sits within 25 % of 1 for the shipped parameters, which is
        why a law that omits it tracks well enough to pass every closed-loop
        test. This states the margin the checks above have against that
        particular mutation: the residual is ``|1 - 1/m_i l_i|`` of the PD
        term, and it must be far outside their tolerance.
        """
        mdl = off_equilibrium_model()
        ctrl = self._controller(mdl)
        for i in range(mdl.num_quadrotors):
            m_l = float(mdl.masses[i] * mdl.cable_lengths[i])
            assert not np.isclose(m_l, 1.0, atol=0.05), "m_i l_i == 1 hides the factor"
            residual = np.linalg.norm(self._terms(ctrl, mdl, i)["pd"] * (1.0 - 1.0 / m_l))
            assert residual > 0.5


# ─── Gains live in defaults.py ────────────────────────────────────────


class TestGainsComeFromDefaults:
    def test_gains_are_the_shipped_defaults(self):
        ctrl = MultiQuadCsPayloadController(num_quadrotors=3)
        np.testing.assert_allclose(ctrl._gain_pos.kp, DEFAULT_MULTI_PAYLOAD_POS_KP)
        np.testing.assert_allclose(ctrl._gain_pos.kd, DEFAULT_MULTI_PAYLOAD_POS_KD)
        np.testing.assert_allclose(ctrl._gain_cable.kp, DEFAULT_MULTI_PAYLOAD_CABLE_KP)
        np.testing.assert_allclose(ctrl._gain_cable.kd, DEFAULT_MULTI_PAYLOAD_CABLE_KD)

    def test_gains_are_copies(self):
        """Tuning one controller must not rewrite the package defaults."""
        ctrl = MultiQuadCsPayloadController(num_quadrotors=2)
        ctrl._gain_pos.kp[0] = 999.0
        ctrl._gain_cable.kp[0] = 999.0
        assert DEFAULT_MULTI_PAYLOAD_POS_KP[0] != 999.0
        assert DEFAULT_MULTI_PAYLOAD_CABLE_KP[0] != 999.0

    @pytest.mark.parametrize(
        "name",
        [
            "DEFAULT_MULTI_PAYLOAD_POS_KP",
            "DEFAULT_MULTI_PAYLOAD_POS_KD",
            "DEFAULT_MULTI_PAYLOAD_CABLE_KP",
            "DEFAULT_MULTI_PAYLOAD_CABLE_KD",
        ],
    )
    def test_changing_a_default_moves_the_command(self, monkeypatch, name):
        """No gain is inlined: perturbing the constant must change the output.

        The perturbation is applied off-equilibrium (position *and* velocity
        error, tilted cables) so that every one of the four gains has something
        to act on.
        """

        def commanded(mdl):
            ctrl = MultiQuadCsPayloadController.from_model(mdl)
            ctrl.setpoint = hover_setpoint(np.array([0.0, 0.0, 1.0]))
            return ctrl.compute(0.0, mdl.state)

        def perturbed_model():
            mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
            mdl.reset(payload_position=np.array([0.2, -0.1, 1.3]))
            mdl.state.payload_velocity = np.array([0.3, 0.2, -0.1])
            for i in range(3):
                mdl.state.cable_angular_velocities[i] += type(
                    mdl.state.cable_angular_velocities[i]
                )(np.array([0.2, -0.3, 0.1]))
            mdl._sync_quad_states()
            return mdl

        base = commanded(perturbed_model())
        monkeypatch.setattr(pc_module, name, getattr(pc_module, name) * 1.5)
        bumped = commanded(perturbed_model())
        assert np.linalg.norm(bumped - base) > 1e-3, (
            f"{name} does not reach the commanded thrust — is it inlined?"
        )

    def test_kp_and_kd_tune_the_payload_loop(self):
        """The documented ``kp``/``kd`` arguments must actually be read.

        They used to land in ``PDController._gains``, which ``compute()`` never
        touches, so anyone tuning through them silently got the defaults. They
        now *are* the payload-loop gains.
        """
        ctrl = MultiQuadCsPayloadController(num_quadrotors=2, kp=[100.0, 90.0, 80.0], kd=2.0)
        np.testing.assert_allclose(ctrl.position_gains.kp, [100.0, 90.0, 80.0])
        np.testing.assert_allclose(ctrl.position_gains.kd, [2.0, 2.0, 2.0])
        # No second, dead gain object left behind to disagree with this one.
        assert ctrl._gains is ctrl.position_gains
        # The cable loop is a separate PD and keeps its own defaults.
        np.testing.assert_allclose(ctrl.cable_gains.kp, DEFAULT_MULTI_PAYLOAD_CABLE_KP)

    def test_tuning_kp_moves_the_command(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.reset(payload_position=np.array([0.2, -0.1, 1.3]))

        def commanded(**gains):
            ctrl = MultiQuadCsPayloadController.from_model(mdl, **gains)
            ctrl.setpoint = hover_setpoint(np.array([0.0, 0.0, 1.0]))
            return ctrl.compute(0.0, mdl.state)

        assert np.linalg.norm(commanded(kp=40.0) - commanded()) > 1.0

    def test_a_malformed_gain_is_rejected(self):
        with pytest.raises(ValueError, match="kp must be a scalar or a 3-vector"):
            MultiQuadCsPayloadController(num_quadrotors=2, kp=[1.0, 2.0])


# ─── Closed loop ──────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
class TestPayloadTracking:
    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_step_setpoint_converges(self, nQ):
        """Two-sided: the payload must reach the target *and* actually travel.

        A cascade that silently did nothing would sit at the start and fail the
        distance check; one that diverged would fail the error check.
        """
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        start = np.array([0.0, 0.0, 1.0])
        target = np.array([1.0, -0.5, 1.8])
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.simulate(tf=10.0, payload_position=start)

        assert np.linalg.norm(mdl.state.payload_position - target) < TOL_PHYSICAL
        assert np.linalg.norm(mdl.state.payload_velocity) < TOL_PHYSICAL
        assert np.linalg.norm(mdl.state.payload_position - start) > 1.0

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_tracks_a_moving_trajectory(self, nQ):
        traj = gentle_spiral()
        result = fly(MultiQuadCsPayloadBase(num_quadrotors=nQ), traj, traj.duration, True)
        assert result["rms"] < 0.15
        assert result["max"] < 0.4
        # The trajectory is not a hover in disguise.
        assert np.linalg.norm(traj.end_position - traj.get(0.0)[0]) > 0.9

    def test_formation_holds_its_shape_while_tracking(self):
        """Each quadrotor stays roughly on its own ring station, not stacked."""
        nQ = 4
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        traj = gentle_spiral()
        fly(mdl, traj, traj.duration, True)
        radii = [
            float(np.linalg.norm(mdl.state.positions[i][:2] - mdl.state.payload_position[:2]))
            for i in range(nQ)
        ]
        for r in radii:
            assert r == pytest.approx(mdl.formation_radius, abs=0.15)


@pytest.mark.slow
@pytest.mark.integration
class TestFlatnessFeedforward:
    """A trajectory that grows higher derivatives has to *measurably* help."""

    @pytest.mark.parametrize("nQ", [1, 2, 3, 4])
    def test_feedforward_improves_tracking(self, nQ):
        traj = brisk_spiral()
        with_ff = fly(MultiQuadCsPayloadBase(num_quadrotors=nQ), traj, traj.duration, True)
        without = fly(MultiQuadCsPayloadBase(num_quadrotors=nQ), traj, traj.duration, False)

        # Discriminating in both directions: the feedback-only run has to be
        # bad enough that there is something to improve, and the feedforward
        # run has to be substantially better rather than merely not worse.
        assert without["rms"] > 0.2, "trajectory too easy — the comparison proves nothing"
        assert with_ff["rms"] < 0.8 * without["rms"]
        assert with_ff["max"] < without["max"]

    def test_simulate_itself_runs_the_feedforward(self, monkeypatch):
        """The wiring inside :meth:`simulate`, not :func:`fly`'s copy of it.

        ``fly`` re-implements the stepping loop, so every other test in this
        file stays green with the feedforward branch of ``simulate()`` deleted
        — and the only two tests that do call ``simulate`` bind a plain lambda,
        which has no ``__self__`` and never trips the sniffing. This one binds
        a bound ``get_full``, goes through ``simulate``, and demands that the
        recovered attitudes reach every step *and* that the run tracks better
        than the same trajectory flown feedback-only.
        """
        traj = brisk_spiral()
        applied: list[int] = []
        original = MultiQuadCsPayloadBase._desired_attitudes

        def spy(self, flat_map, flat_fn, t):
            out = original(self, flat_map, flat_fn, t)
            applied.append(0 if out is None else len(out))
            return out

        monkeypatch.setattr(MultiQuadCsPayloadBase, "_desired_attitudes", spy)

        def run(feedforward):
            mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
            mdl.payload_controller.setpoint = (
                traj.get_full if feedforward else (lambda t: traj.get_full(t)[:3])
            )
            mdl.simulate(tf=traj.duration, payload_position=traj.get(0.0)[0])
            return mdl, float(np.linalg.norm(mdl.state.payload_position - traj.get(mdl.t)[0]))

        mdl_ff, err_ff = run(True)
        steps = len(applied)
        assert steps > 100, "simulate() never asked for a feedforward attitude"
        assert set(applied) == {3}, "some step ran without the recovered attitudes"
        assert mdl_ff.feedforward_drops == 0

        applied.clear()
        _, err_fb = run(False)
        assert not applied, "the plain-closure setpoint must not unlock the feedforward"

        assert err_fb > 0.3, "trajectory too easy — the comparison proves nothing"
        assert err_ff < 0.8 * err_fb

    def test_feedforward_is_skipped_when_the_policy_is_not_published(self, caplog):
        """No ``tension_policy`` on the controller means no feedforward at all.

        Falling back on a default ring would build the reference at *this
        model's* formation radius while the loop allocates at whatever the
        controller chose — precisely the feedback/feedforward disagreement the
        shared policy exists to rule out, and silent when it happens. The run
        must go feedback-only and say so.
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

        traj = gentle_spiral()
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        # Allocates on a tighter ring than the model's — the mismatch a default
        # policy would paper over.
        inner = MultiQuadCsPayloadController.from_model(mdl)
        inner.tension_policy = RingTensionSplit(
            num_quadrotors=3,
            payload_mass=mdl.payload_mass,
            masses=mdl.masses,
            cable_lengths=mdl.cable_lengths,
            radius=0.35,
        )
        mdl._payload_controller = Opaque(inner)
        mdl.payload_controller.setpoint = traj.get_full

        # The sniffing still finds a ``get_full``; only the policy is missing.
        assert mdl._flatness_feedforward_fn() is not None
        with caplog.at_level(logging.WARNING, logger="udaan.models.multi_quad_cspayload.base"):
            mdl.simulate(tf=0.5, payload_position=traj.get(0.0)[0])

        assert "tension_policy" in caplog.text
        assert np.all(np.isfinite(mdl.state.payload_position))

    def test_an_infeasible_reference_is_counted_and_warned_once(self, caplog):
        """Dropped feedforward is visible: a warning, then a running count.

        A vertical sinusoid at 1.6× the certified acceleration bound leaves the
        taut-cable certificate around every downward peak and re-enters it in
        between — no payload on cables can fall faster than ``g`` — so the map
        refuses part of the run and accepts the rest. Silently degrading to
        feedback-only there is how a mis-specified trajectory gets mistaken for
        a badly tuned loop.
        """
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        flat_map = MultiQuadCsPayload.from_model(
            mdl, tension_policy=mdl.payload_controller.tension_policy
        )
        traj = VerticalSinusoid(np.array([0.0, 0.0, 2.0]), flat_map.payload_acceleration_bound())
        mdl.payload_controller.setpoint = traj.get_full

        with caplog.at_level(logging.WARNING, logger="udaan.models.multi_quad_cspayload.base"):
            mdl.simulate(tf=1.0, payload_position=traj.get(0.0)[0])

        steps = int(round(mdl.t / mdl.dt))
        assert 0 < mdl.feedforward_drops < steps, "the reference must be intermittent, not constant"
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1, "the drop warning must fire once per run, not once per step"
        assert "feedforward" in warnings[0].getMessage()

    def test_a_policy_that_does_not_add_up_is_not_absorbed(self):
        """A broken split is a bug, not an aggressive trajectory.

        ``recover`` raises :class:`ValueError` both for a reference outside the
        taut-cable certificate — which the model legitimately absorbs by
        dropping the feedforward — and for a policy that violates
        ``Σ_i 𝐓_i^(k) = F_net^(k)``. Absorbing the second would turn a
        programming error into a slightly worse tracking number.
        """
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        ring = mdl.payload_controller.tension_policy
        broken = LeakyTensionSplit(ring.anchors, ring.shares)
        mdl._payload_controller = MultiQuadCsPayloadController.from_model(
            mdl, tension_policy=broken
        )
        traj = gentle_spiral()
        mdl.payload_controller.setpoint = traj.get_full

        with pytest.raises(TensionSplitContractError, match="violates the split contract"):
            mdl.simulate(tf=0.1, payload_position=traj.get(0.0)[0])

    def test_flat_map_declines_gracefully_when_a_cable_would_slacken(self):
        """A too-aggressive reference drops the feedforward, it does not crash."""
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        flat_map = MultiQuadCsPayload.from_model(
            mdl, tension_policy=mdl.payload_controller.tension_policy
        )
        bound = flat_map.payload_acceleration_bound()
        assert np.isfinite(bound) and bound > 0.0

        def violent(t):
            # Straight down at several times the certified acceleration bound.
            accel = -3.0 * bound * _E3
            return (np.zeros(3), np.zeros(3), accel, np.zeros(3), np.zeros(3))

        assert mdl._desired_attitudes(flat_map, violent, 0.0) is None


@pytest.mark.slow
@pytest.mark.integration
class TestCablesStayTaut:
    @pytest.mark.parametrize("nQ", [1, 2, 3, 4])
    def test_tension_stays_positive_through_a_manoeuvre(self, nQ):
        """Every cable pulls throughout — a slack cable would show ``T_i ≤ 0``.

        The payload is heavier than the default here so the hover tension is
        comfortably clear of zero at large ``nQ``, where ``m_L g`` is shared
        across more cables; the point of the test is the *manoeuvre*, not how
        thin the static margin can be made.
        """
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ, payload_mass=0.6)
        traj = gentle_spiral()
        result = fly(mdl, traj, traj.duration, True)
        assert result["min_tension"] > 0.05
        assert result["rms"] < 0.2

    def test_trajectory_respects_the_policy_acceleration_bound(self):
        """The manoeuvre above is inside the taut-cable certificate."""
        mdl = MultiQuadCsPayloadBase(num_quadrotors=4, payload_mass=0.6)
        flat_map = MultiQuadCsPayload.from_model(
            mdl, tension_policy=mdl.payload_controller.tension_policy
        )
        traj = gentle_spiral()
        peak = max(
            float(np.linalg.norm(traj.get_full(t)[2])) for t in np.linspace(0.0, traj.duration, 400)
        )
        assert peak < flat_map.payload_acceleration_bound()


@pytest.mark.slow
@pytest.mark.integration
class TestHeterogeneousTeam:
    def test_unequal_masses_and_cable_lengths_still_track(self):
        mdl = MultiQuadCsPayloadBase(
            num_quadrotors=3,
            masses=[0.7, 0.9, 1.2],
            cable_lengths=[0.9, 1.1, 1.0],
        )
        # The ring policy is length-weighted and mass-weighted, so the anchors
        # and the dynamic shares must both come out unequal — otherwise this
        # test is a homogeneous one wearing a disguise.
        policy = mdl.payload_controller.tension_policy
        assert len(set(np.round(policy.shares, 6))) == 3
        assert len(set(np.round(np.linalg.norm(policy.anchors, axis=1), 6))) == 3

        traj = gentle_spiral()
        result = fly(mdl, traj, traj.duration, True)
        assert result["rms"] < 0.15
        assert result["min_tension"] > 0.0

    def test_unequal_team_reaches_a_step_setpoint(self):
        mdl = MultiQuadCsPayloadBase(
            num_quadrotors=4,
            masses=[0.6, 0.9, 1.1, 1.4],
            cable_lengths=[1.0, 1.2, 0.9, 1.1],
        )
        target = np.array([0.8, 0.6, 1.6])
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.simulate(tf=10.0, payload_position=np.array([0.0, 0.0, 1.0]))
        assert np.linalg.norm(mdl.state.payload_position - target) < TOL_PHYSICAL
        assert np.linalg.norm(mdl.state.payload_velocity) < TOL_PHYSICAL
