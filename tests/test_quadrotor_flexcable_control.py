"""Tests for :mod:`udaan.control.quadrotor_flexcable` — the M5 payload LQR.

M5's controller is the first here that is not a geometric cascade, so this file
looks different from its M2/M3/M4 siblings. There is no cable loop to check the
algebra of; what there is instead is a *linearisation*, a *Riccati solve* and a
set of claims about what the resulting gain can and cannot do. Every one of
those claims is asserted rather than asserted-about:

* :class:`TestLinearization` pins the two structural facts the design rests on
  — that the tangent subspace is exactly invariant (so the reduced Riccati
  solve loses nothing), and that the *embedded* pair is genuinely
  unstabilisable (so the reduction is necessary, not tidiness);
* :class:`TestControllability` is this file's answer to M4's grasp-rank tests.
  The system is controllable at every ``n``, and that is nearly meaningless on
  its own: the PBH margin collapses with the link count, the slowest
  closed-loop mode is the chain's *fastest* natural mode, and its damping is
  independent of the cost weights. All three are measured;
* :class:`TestFlatnessFeedforward` runs through :meth:`simulate` itself rather
  than a hand-rolled loop, so the branch it is testing is the one that ships,
  and it also separates the two feedforward qualities M5 makes possible — a
  jet padded past snap with zeros, and one exact to the full ``2n + 4``.

Every chain flown here is inside the ZOH-Euler stable regime;
:func:`assert_stable_regime` checks that rather than assuming it.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest
from scipy.linalg import expm

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_PHYSICAL, TOL_TIGHT
from udaan.control.quadrotor_flexcable import QuadrotorFlexCableController
from udaan.core.defaults import (
    DEFAULT_FLEXCABLE_LQR_Q_LINK,
    DEFAULT_FLEXCABLE_LQR_Q_LINK_RATE,
    DEFAULT_FLEXCABLE_LQR_Q_POSITION,
    DEFAULT_FLEXCABLE_LQR_R_MOMENT,
    DEFAULT_FLEXCABLE_LQR_R_THRUST,
    GRAVITY,
)
from udaan.flatness.chain import chain_jet_order
from udaan.flatness.jet import Jet
from udaan.flatness.quadrotor_flexcable import (
    QuadrotorFlexCableFlats,
    QuadrotorFlexCableInputs,
    QuadrotorFlexCableRefState,
)
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.quadrotor_flexcable import QuadrotorFlexCableBase
from udaan.utils import hat
from udaan.utils.trajectory import CrazyTrajectory

_E3 = np.array([0.0, 0.0, 1.0])

_MODEL_LOGGER = "udaan.models.quadrotor_flexcable.base"

#: Link counts whose default chain stays inside the ZOH-Euler stable region
#: (growth exponent 0.008 / 0.14 / 0.45 at n = 1 / 2 / 3 — see
#: ``QuadrotorFlexCableBase.stability_growth_rate``).
STABLE_COUNTS = [1, 2, 3]


# ─── Helpers ──────────────────────────────────────────────────────────


def assert_stable_regime(mdl):
    """Refuse to draw conclusions from a chain the integrator cannot hold.

    M5's ZOH-Euler loop is only *conditionally* stable, and a run that winds up
    would fail these tests for a reason that has nothing to do with the
    controller. Rather than pick link counts by hand and hope, every model
    flown here is checked against the model's own criterion.
    """
    rate = mdl.stability_growth_rate()
    assert rate < 1.0, (
        f"a {mdl.num_links}-link chain at dt={mdl.dt} has growth exponent {rate:.3f}; "
        "outside the ZOH-Euler stable region, so a closed-loop assertion on it "
        "would be measuring the integrator"
    )


def hover_setpoint(x):
    """A stationary payload setpoint at ``x``."""
    target = np.asarray(x, dtype=float).copy()
    return lambda t: (target, np.zeros(3), np.zeros(3))


def gentle_lissajous(center=None):
    """A payload trajectory the flat map accepts on every sample.

    Amplitudes and frequencies are sized so that no link ever leaves the
    hanging branch: the chain cascade refuses a reference whose links swing past
    horizontal, and a test trajectory that trips that would be measuring the
    drop path rather than the tracking.
    """
    if center is None:
        center = np.array([0.0, 0.0, 1.5])
    return CrazyTrajectory(
        center=np.asarray(center, dtype=float),
        ax=0.4,
        ay=0.35,
        az=0.25,
        f1=0.18,
        f2=0.13,
        f3=0.09,
    )


class SnapOnlyTrajectory:
    """A trajectory that publishes ``get_full`` but not ``get_jet``.

    Exactly what every trajectory in :mod:`udaan.utils.trajectory` except
    :class:`~udaan.utils.trajectory.CrazyTrajectory` looks like to the
    feedforward sniffer, and the reason M5 needed the deeper protocol at all:
    the chain wants ``2n + 4`` derivatives and this supplies four.
    """

    def __init__(self, traj):
        self._traj = traj

    def get(self, t):
        return self._traj.get(t)

    def get_full(self, t):
        return self._traj.get_full(t)


def fly(mdl, traj, tf, source):
    """Run ``mdl`` under :meth:`simulate` and return the payload tracking error.

    ``source`` selects how the setpoint is bound, which is the *only* switch the
    feedforward has:

    * ``"off"``   — a plain closure, so the sniffer finds no owner at all;
    * ``"snap"``  — a bound method whose owner stops at snap (zero-padded jet);
    * ``"jet"``   — a bound method whose owner supplies every derivative.

    Errors are sampled inside the stepping loop rather than only at the end, so
    a run that tracks well on average and badly in the middle cannot pass.
    """
    if source == "off":
        mdl.payload_controller.setpoint = lambda t: traj.get(t)
    elif source == "snap":
        mdl.payload_controller.setpoint = SnapOnlyTrajectory(traj).get
    elif source == "jet":
        mdl.payload_controller.setpoint = traj.get
    else:  # pragma: no cover - programming error in a test
        raise AssertionError(f"unknown source {source!r}")

    errors: list[float] = []
    times: list[float] = []
    original_step = mdl.step

    def sampling_step(u, desired_att=None):
        errors.append(float(np.linalg.norm(mdl.state.payload_position - traj.get(mdl.t)[0])))
        times.append(mdl.t)
        original_step(u, desired_att)

    mdl.step = sampling_step
    mdl.simulate(tf, payload_position=traj.get(0.0)[0])

    error = np.asarray(errors)
    time = np.asarray(times)
    tail = time > 0.5 * tf
    return {
        "rms": float(np.sqrt((error**2).mean())),
        "max": float(error.max()),
        # Steady-state tracking, past the transient the initial hanging chain
        # has to work off; this is where the feedforward's quality actually
        # shows.
        "tail_rms": float(np.sqrt((error[tail] ** 2).mean())),
        "drops": mdl.feedforward_drops,
    }


def recovered_reference(ctrl, traj, t):
    """``(RefState, Inputs)`` from the flat map at ``t``, at full jet depth."""
    flats = QuadrotorFlexCableFlats(x_L=Jet(traj.get_jet(t, ctrl.jet_order)), psi=Jet(np.zeros(3)))
    return ctrl.flat_map.recover(flats)


def aggressive_lissajous(center=None):
    """A payload trajectory the flat map still accepts on every sample — barely.

    Admissibility and *gentleness* are not the same thing, and the difference is
    the whole content of
    :meth:`TestLinearization.test_the_linearisation_moves_with_the_reference`.
    This one tilts the reference chain past 20° while the chain cascade still
    recovers a reference at every sample.
    """
    if center is None:
        center = np.array([0.0, 0.0, 1.5])
    return CrazyTrajectory(
        center=np.asarray(center, dtype=float),
        ax=0.6,
        ay=0.35,
        az=0.25,
        f1=0.4,
        f2=0.13,
        f3=0.09,
    )


def moving_reference(num_links, seed=3):
    """A **non-hover** reference, and a model posed at it.

    Almost everything reference-dependent in :meth:`linearize` is identically
    zero at the hanging equilibrium — every ``ω_i^d`` term, every ``ω̇_i^d``
    term, every ``Ω_d`` term, the whole ``R_d ≠ I`` thrust rotation. The gain
    ships from that operating point, so a suite that only ever linearises there
    exercises none of it; that is exactly where the reference MATLAB's
    scalar-broadcast defect lives, and reinstating it passes such a suite
    untouched.

    Returns ``(model, pose, ref, inputs)`` where ``pose(link_dirs, link_rates,
    R, Omega)`` writes a state into the model. ``ref``'s link angular
    *accelerations* are read back off the model's own nonlinear dynamics rather
    than invented, so the reference is one the plant could actually be on —
    which is what makes a finite-difference comparison against it meaningful.
    """
    rng = np.random.default_rng(seed)
    link_dirs = []
    link_rates = []
    for _ in range(num_links):
        direction = -_E3 + 0.35 * rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        rate = rng.normal(size=3)
        rate -= np.dot(rate, direction) * direction
        link_dirs.append(direction)
        link_rates.append(1.3 * rate)

    R_d = expm(hat(np.array([0.12, -0.19, 0.07])))
    Omega_d = np.array([0.3, -0.2, 0.45])
    thrust_d = 13.0
    moment_d = np.array([0.02, -0.03, 0.01])

    mdl = QuadrotorFlexCableBase(num_links=num_links)

    def pose(dirs, rates, R, Omega):
        mdl.state.orientation = SO3(np.asarray(R, dtype=float))
        mdl.state.angular_velocity = TSO3(np.asarray(Omega, dtype=float))
        mdl.state.link_attitudes = [S2(np.asarray(q, dtype=float)) for q in dirs]
        mdl.state.link_angular_velocities = [TS2(np.asarray(w, dtype=float)) for w in rates]

    pose(link_dirs, link_rates, R_d, Omega_d)
    accel = mdl.accelerations(thrust_d, moment_d)

    ref = QuadrotorFlexCableRefState(
        acceleration=accel["quad_accel"].copy(),
        link_attitudes=[q.copy() for q in link_dirs],
        link_angular_velocities=[w.copy() for w in link_rates],
        link_angular_accelerations=[row.copy() for row in accel["link_ang_accels"]],
        orientation=SO3(R_d),
        angular_velocity=TSO3(Omega_d),
    )
    inputs = QuadrotorFlexCableInputs(
        thrust=thrust_d, moment=moment_d, tensions=np.zeros(num_links)
    )
    return mdl, pose, ref, inputs


# ─── Construction ─────────────────────────────────────────────────────


class TestConstruction:
    @pytest.mark.parametrize("n", STABLE_COUNTS)
    def test_output_is_a_wrench(self, n):
        mdl = QuadrotorFlexCableBase(num_links=n)
        wrench = mdl.payload_controller.compute(0.0, mdl.state)
        assert wrench.shape == (4,)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_gain_shape_is_the_embedded_state(self, n):
        ctrl = QuadrotorFlexCableController(num_links=n)
        assert ctrl.state_dim == 12 + 6 * n
        assert ctrl.gain.shape == (4, 12 + 6 * n)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_jet_order_is_two_n_plus_four(self, n):
        assert (
            QuadrotorFlexCableController(num_links=n).jet_order == chain_jet_order(n) == 2 * n + 4
        )

    def test_from_model_adopts_model_parameters(self):
        mdl = QuadrotorFlexCableBase(num_links=3)
        mdl.link_masses = [0.1, 0.2, 0.7]
        mdl.link_lengths = [0.2, 0.3, 0.5]
        ctrl = QuadrotorFlexCableController.from_model(mdl)
        np.testing.assert_allclose(ctrl.link_masses, mdl.link_masses, atol=TOL_EXACT)
        np.testing.assert_allclose(ctrl.link_lengths, mdl.link_lengths, atol=TOL_EXACT)
        assert ctrl.mass == pytest.approx(mdl.mass)
        assert ctrl.total_mass == pytest.approx(mdl.total_mass)

    def test_model_default_controller_is_the_lqr(self):
        mdl = QuadrotorFlexCableBase(num_links=2)
        assert isinstance(mdl.payload_controller, QuadrotorFlexCableController)
        # And the model steps its output as a wrench, not a thrust vector.
        assert mdl._input_type is mdl.INPUT_TYPE.WRENCH

    def test_the_controller_owns_a_flat_map_at_its_own_parameters(self):
        """``simulate`` reads the map off the controller, so it has to be the
        controller's — not a copy of the model's."""
        mdl = QuadrotorFlexCableBase(num_links=2)
        ctrl = mdl.payload_controller
        ctrl.sync_from_model(mdl)
        np.testing.assert_allclose(ctrl.flat_map.link_masses, ctrl.link_masses, atol=TOL_EXACT)
        np.testing.assert_allclose(ctrl.flat_map.link_lengths, ctrl.link_lengths, atol=TOL_EXACT)

    def test_cost_weights_are_the_shipped_defaults(self):
        weights = QuadrotorFlexCableController(num_links=2).cost_weights
        assert weights["q_position"] == pytest.approx(DEFAULT_FLEXCABLE_LQR_Q_POSITION)
        assert weights["q_link"] == pytest.approx(DEFAULT_FLEXCABLE_LQR_Q_LINK)
        assert weights["r_thrust"] == pytest.approx(DEFAULT_FLEXCABLE_LQR_R_THRUST)
        assert weights["r_moment"] == pytest.approx(DEFAULT_FLEXCABLE_LQR_R_MOMENT)

    def test_cost_weights_snapshot_cannot_mutate_the_controller(self):
        ctrl = QuadrotorFlexCableController(num_links=2)
        ctrl.cost_weights["q_position"] = 1e6
        assert ctrl.cost_weights["q_position"] == pytest.approx(DEFAULT_FLEXCABLE_LQR_Q_POSITION)


# ─── Parameter validation ─────────────────────────────────────────────


class TestParameterValidation:
    """Every ``ValueError`` this controller raises, with the message pinned.

    Cheap, and the difference between a mis-specified chain failing where it is
    written and failing as a wrench that quietly regulates the wrong machine.
    """

    def test_zero_links(self):
        with pytest.raises(ValueError, match="num_links must be >= 1"):
            QuadrotorFlexCableController(num_links=0)

    @pytest.mark.parametrize("name", ["kp", "kd", "ki"])
    def test_pd_gains_are_refused(self, name):
        """``PDController`` would park these where nothing reads them."""
        with pytest.raises(ValueError, match=f"'{name}' has nothing to tune"):
            QuadrotorFlexCableController(num_links=2, **{name: np.ones(3)})

    def test_non_positive_mass(self):
        with pytest.raises(ValueError, match="mass must be positive"):
            QuadrotorFlexCableController(num_links=2, mass=0.0)

    def test_malformed_inertia(self):
        with pytest.raises(ValueError, match=r"inertia must be \(3,\) or \(3, 3\)"):
            QuadrotorFlexCableController(num_links=2, inertia=np.zeros((2, 2)))

    @pytest.mark.parametrize("name", ["link_masses", "link_lengths"])
    def test_wrong_length_per_link_array(self, name):
        with pytest.raises(ValueError, match=f"{name} must be a scalar or have 3 entries"):
            QuadrotorFlexCableController(num_links=3, **{name: [1.0, 2.0]})

    @pytest.mark.parametrize("name", ["link_masses", "link_lengths"])
    def test_non_positive_per_link_array(self, name):
        with pytest.raises(ValueError, match=f"{name} must be strictly positive"):
            QuadrotorFlexCableController(num_links=2, **{name: [1.0, -1.0]})

    @pytest.mark.parametrize(
        "name",
        ["q_position", "q_velocity", "q_attitude", "q_angular_rate", "q_link", "q_link_rate"],
    )
    def test_non_positive_state_weight(self, name):
        with pytest.raises(ValueError, match=f"{name} must be strictly positive"):
            QuadrotorFlexCableController(num_links=2, **{name: 0.0})

    @pytest.mark.parametrize("name", ["r_thrust", "r_moment"])
    def test_non_positive_input_weight(self, name):
        with pytest.raises(ValueError, match=f"{name} must be strictly positive"):
            QuadrotorFlexCableController(num_links=2, **{name: -1.0})

    def test_sync_from_a_different_chain_length(self):
        ctrl = QuadrotorFlexCableController(num_links=2)
        with pytest.raises(ValueError, match="cannot sync a 2-link controller from a 4-link"):
            ctrl.sync_from_model(QuadrotorFlexCableBase(num_links=4))

    def test_error_state_against_a_foreign_state(self):
        ctrl = QuadrotorFlexCableController(num_links=2)
        ref, _ = ctrl.equilibrium_reference(0.0)
        foreign = QuadrotorFlexCableBase(num_links=3).state
        with pytest.raises(ValueError, match="state has 3 links, but this controller has 2"):
            ctrl.error_state(foreign, ref)

    def test_error_state_against_a_foreign_reference(self):
        ctrl = QuadrotorFlexCableController(num_links=2)
        other, _ = QuadrotorFlexCableController(num_links=3).equilibrium_reference(0.0)
        state = QuadrotorFlexCableBase(num_links=2).state
        with pytest.raises(ValueError, match="reference has 3 links, but this controller has 2"):
            ctrl.error_state(state, other)

    def test_linearize_against_a_foreign_reference(self):
        ctrl = QuadrotorFlexCableController(num_links=2)
        other, _ = QuadrotorFlexCableController(num_links=3).equilibrium_reference(0.0)
        with pytest.raises(ValueError, match="reference describes 3 links"):
            ctrl.linearize(other)

    def test_an_unsolvable_riccati_is_reported_with_its_parameters(self):
        """A chain the LQR cannot be solved for is refused *when flown*.

        Six orders of magnitude of mass ratio on a millimetre lever leaves the
        dynamics perfectly well posed — ``mass_matrix`` stays positive definite,
        which ``tests/test_quadrotor_flexcable_base.py`` asserts — but the
        Riccati pencil far too ill-conditioned to reorder. The gain is
        therefore solved lazily: building such a model, and integrating it open
        loop, must still work.
        """
        mdl = QuadrotorFlexCableBase(num_links=5)
        mdl.link_masses = [1e-6, 1e-6, 1e-6, 1e-6, 5.0]
        mdl.link_lengths = [1e-3, 1e-3, 1e-3, 1e-3, 2.0]
        with pytest.raises(ValueError, match="no stabilising solution to the algebraic Riccati"):
            _ = mdl.payload_controller.gain

    def test_a_badly_scaled_chain_is_refused_rather_than_silently_regulated(self):
        """The other half of the same failure, and the one that used to slip through.

        The test above is the case :func:`scipy.linalg.solve_continuous_are`
        *reports*. It does not report every case it cannot usefully solve: on
        these parameters — all legal, all accepted by the constructor — it
        returns without complaint a ``P`` of norm ``1e10`` whose residual is
        hundreds of times ``\\|Q\\|``, and the caller used to get a gain of
        matching size with no signal at all.
        """
        from scipy.linalg import solve_continuous_are

        ctrl = QuadrotorFlexCableController(
            num_links=3,
            link_masses=[5.6535141258, 0.0005257727, 5.5366521072],
            link_lengths=[0.0861974377, 0.1861998586, 3.041639794],
            payload_mass=0.0433307839,
            mass=0.4454316403,
        )
        # Not a caught exception being re-raised: the solver returns happily.
        A, B = ctrl.linearize()
        basis = ctrl.tangent_projection()
        Q_r, R = ctrl.cost_matrices()
        assert np.linalg.norm(solve_continuous_are(basis.T @ A @ basis, basis.T @ B, Q_r, R)) > 1e8

        with pytest.raises(ValueError, match="does not solve the posed problem"):
            _ = ctrl.gain

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_the_residual_check_has_room_to_spare_on_the_shipped_chain(self, n):
        """The threshold has to clear the shipped chain at every usable ``n``.

        M5's residual *grows* with the link count — the pencil gets stiffer —
        so this is the direction in which the check could plausibly bite a
        legitimate configuration, and it does not: ``n = 8`` is already past
        the zero-order-hold envelope and still four orders under the warning.
        """
        from scipy.linalg import solve_continuous_are

        from udaan.control import ARE_RESIDUAL_TOLERANCE, ARE_RESIDUAL_WARNING

        ctrl = QuadrotorFlexCableController(num_links=n)
        A, B = ctrl.linearize()
        basis = ctrl.tangent_projection()
        A_r, B_r = basis.T @ A @ basis, basis.T @ B
        Q_r, R = ctrl.cost_matrices()
        P = solve_continuous_are(A_r, B_r, Q_r, R)
        gain_r = -np.linalg.solve(R, B_r.T @ P)
        residual = A_r.T @ P + P @ A_r + P @ B_r @ gain_r + Q_r
        relative = float(np.linalg.norm(residual)) / float(np.linalg.norm(Q_r))
        assert relative < ARE_RESIDUAL_WARNING < ARE_RESIDUAL_TOLERANCE

    def test_get_jet_refuses_a_negative_order(self):
        with pytest.raises(ValueError, match="order must be >= 0"):
            gentle_lissajous().get_jet(0.0, -1)


# ─── The hanging equilibrium ──────────────────────────────────────────


class TestHoverEquilibrium:
    """At the trim the correction has to vanish exactly, not approximately."""

    @pytest.mark.parametrize("n", STABLE_COUNTS)
    def test_error_state_vanishes_at_the_equilibrium(self, n):
        mdl = QuadrotorFlexCableBase(num_links=n)
        mdl.reset(payload_position=np.array([0.4, -0.2, 1.0]))
        ctrl = mdl.payload_controller
        ctrl.setpoint = hover_setpoint([0.4, -0.2, 1.0])
        ref, _ = ctrl.equilibrium_reference(0.0)
        np.testing.assert_allclose(
            ctrl.error_state(mdl.state, ref), np.zeros(ctrl.state_dim), atol=TOL_TIGHT
        )

    @pytest.mark.parametrize("n", STABLE_COUNTS)
    def test_commanded_wrench_is_the_static_trim(self, n):
        """Straight out of the model's own "Static trim": ``f = M₀₀ g``, ``M = 0``."""
        mdl = QuadrotorFlexCableBase(num_links=n)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        mdl.payload_controller.setpoint = hover_setpoint([0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            mdl.compute_default_input(), mdl.hover_input(), atol=TOL_ANALYTIC
        )

    def test_the_equilibrium_reference_ignores_the_setpoint_acceleration(self):
        """An accelerating reference is not an equilibrium.

        Folding the setpoint's acceleration in here would smuggle half a
        feedforward into the "feedback-only" law and make every with/without
        comparison in this file meaningless.
        """
        ctrl = QuadrotorFlexCableController(num_links=2)
        ctrl.setpoint = lambda t: (np.zeros(3), np.zeros(3), np.array([0.0, 5.0, 0.0]))
        ref, inputs = ctrl.equilibrium_reference(0.0)
        np.testing.assert_allclose(ref.acceleration, np.zeros(3), atol=TOL_EXACT)
        assert inputs.thrust == pytest.approx(ctrl.total_mass * GRAVITY)
        np.testing.assert_allclose(inputs.moment, np.zeros(3), atol=TOL_EXACT)

    def test_the_equilibrium_quadrotor_sits_a_cable_above_the_payload(self):
        ctrl = QuadrotorFlexCableController(num_links=3)
        ctrl.setpoint = hover_setpoint([1.0, 2.0, 0.5])
        ref, _ = ctrl.equilibrium_reference(0.0)
        np.testing.assert_allclose(
            ref.position - ref.payload_position,
            float(np.sum(ctrl.link_lengths)) * _E3,
            atol=TOL_EXACT,
        )


# ─── The linearisation ────────────────────────────────────────────────


class TestLinearization:
    """The two structural facts the Riccati solve rests on."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_the_tangent_subspace_is_exactly_invariant(self, n):
        """``W`` spans an invariant subspace containing the whole range of ``B``.

        This is what makes ``A_r = WᵀAW``, ``B_r = WᵀB`` a *reduction* rather
        than an approximation: nothing the input can reach, and nothing the
        dynamics can produce from inside the subspace, leaves it.
        """
        ctrl = QuadrotorFlexCableController(num_links=n)
        A, B = ctrl.linearize()
        W = ctrl.tangent_projection()
        complement = np.eye(A.shape[0]) - W @ W.T
        np.testing.assert_allclose(complement @ A @ W, 0.0, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(complement @ B, 0.0, atol=TOL_EXACT)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_embedded_pair_is_not_stabilisable(self, n):
        """The reduction is necessary, not cosmetic.

        ``2n`` coordinates of the embedded error state are the ``S²`` gauge:
        ``ξ_i`` is a cross product with ``q_i^d`` and ``ω_i`` is tangent to
        ``q_i``, so their components along ``q_i^d`` are integrators no input
        touches. Handing the embedded pair to the Riccati solver asks it to
        stabilise them, and it cannot.
        """
        from scipy.linalg import solve_continuous_are

        ctrl = QuadrotorFlexCableController(num_links=n)
        A, B = ctrl.linearize()
        with pytest.raises(np.linalg.LinAlgError):
            solve_continuous_are(A, B, np.eye(A.shape[0]), np.eye(4))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_gain_never_acts_on_a_gauge_direction(self, n):
        ctrl = QuadrotorFlexCableController(num_links=n)
        W = ctrl.tangent_projection()
        gauge = np.eye(ctrl.state_dim) - W @ W.T
        np.testing.assert_allclose(ctrl.gain @ gauge, 0.0, atol=TOL_ANALYTIC)

    def test_the_hover_linearisation_is_the_reference_one_at_rest(self):
        """``linearize(ref)`` at a *stationary* recovered reference reproduces
        the no-argument (hover) form, which is what lets one code path serve
        both."""
        ctrl = QuadrotorFlexCableController(num_links=2)
        ref, inputs = ctrl.equilibrium_reference(0.0)
        A_ref, B_ref = ctrl.linearize(ref, inputs)
        A_hover, B_hover = ctrl.linearize()
        np.testing.assert_allclose(A_ref, A_hover, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(B_ref, B_hover, atol=TOL_ANALYTIC)

    @staticmethod
    def _walk(ctrl, traj, times=None):
        """``(worst relative |A - A_hover|, worst link tilt in degrees, drops)``."""
        if times is None:
            times = np.linspace(0.0, 12.0, 40)
        A_hover, _ = ctrl.linearize()
        worst_A = 0.0
        worst_tilt = 0.0
        drops = 0
        for t in times:
            try:
                ref, inputs = recovered_reference(ctrl, traj, t)
            except ValueError:
                drops += 1
                continue
            A_ref, _ = ctrl.linearize(ref, inputs)
            worst_A = max(worst_A, np.linalg.norm(A_ref - A_hover) / np.linalg.norm(A_hover))
            worst_tilt = max(
                worst_tilt,
                max(
                    float(np.degrees(np.arccos(np.clip(-np.asarray(q)[2], -1.0, 1.0))))
                    for q in ref.link_attitudes
                ),
            )
        return worst_A, worst_tilt, drops

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_linearisation_barely_moves_over_a_gentle_trajectory(self, n):
        """The measurement behind shipping a *constant* gain — for gentle references.

        Walking the gentle Lissajous, the reference-dependent ``A`` stays within
        ~6% of the hover one and no link tilts past a few degrees, which is why
        the finite-horizon schedule the MATLAB reference caches buys little
        here.

        Two-sided: the trajectory also has to be *lively* enough that the
        reference links move at all, or the bound proves nothing. And see
        :meth:`test_the_linearisation_moves_a_long_way_on_an_aggressive_reference`
        for why this number must not be read as a bound over the whole
        admissible set.
        """
        ctrl = QuadrotorFlexCableController(num_links=n)
        worst_A, worst_tilt, drops = self._walk(ctrl, gentle_lissajous())
        assert drops == 0
        assert worst_tilt > 0.5, "reference chain never moved — the bound proves nothing"
        assert worst_A < 0.10, f"linearisation moved {worst_A:.3f} — a schedule would be needed"

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_linearisation_moves_a_long_way_on_an_aggressive_reference(self, n):
        """Admissible is *not* the same as gentle, and the docs used to conflate them.

        The flat map's refusal criterion — a link leaving the hanging branch —
        keeps every ``q_i^d`` inside 90° of ``-e_3``, and that bound is far too
        loose to justify a constant gain. This trajectory is accepted on *every*
        sample (``drops == 0``, so nothing here is measuring the drop path) and
        still moves ``A`` by more than a third, with the chain tilting past 20°.

        The assertion is deliberately two-sided: an upper bound would let the
        aggressive case quietly become gentle again under a future retune, and
        the point of this test is that the constant gain has a *domain*.
        """
        ctrl = QuadrotorFlexCableController(num_links=n)
        gentle_A, _, _ = self._walk(ctrl, gentle_lissajous())
        worst_A, worst_tilt, drops = self._walk(ctrl, aggressive_lissajous())
        assert drops == 0, "this trajectory is meant to be fully admissible"
        assert worst_tilt > 15.0, f"reference chain only tilted {worst_tilt:.1f}° — not aggressive"
        assert worst_A > 4.0 * gentle_A, (
            f"aggressive reference moved A by {worst_A:.3f} against the gentle "
            f"{gentle_A:.3f}; the two are supposed to differ by roughly 6x"
        )
        assert worst_A < 0.60, (
            f"linearisation moved {worst_A:.3f}; above ~50% the constant gain is "
            "no longer defensible even as an approximation"
        )


# ─── The linearisation away from hover ────────────────────────────────


class TestNonHoverLinearization:
    r"""Everything in :meth:`linearize` that hover cannot see.

    At the hanging equilibrium ``ω_i^d = ω̇_i^d = Ω_d = 0`` and ``R_d = I``, so
    most of the reference-dependent assembly evaluates to zero or to identity
    and never runs against a number. That is precisely where the reference
    MATLAB's ``\hat{\dot\omega} - \|\omega\|^2`` scalar-broadcast defect hides:
    reinstating it at *both* sites leaves the whole rest of this suite green.

    The load-bearing test here is a finite difference of the model's own
    nonlinear dynamics. :meth:`linearize` claims to be the variational Jacobian
    of :meth:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase.accelerations`
    about the reference; that is checkable directly, without re-deriving any of
    the algebra in the test, and it fails on any single wrong term.
    """

    @staticmethod
    def _error_direction(ctrl, rng):
        """A random error-state direction, tangent where the manifold demands it."""
        n = ctrl.num_links
        return {
            "eta": rng.normal(size=3),
            "delta_Omega": rng.normal(size=3),
            "xi": [rng.normal(size=3) for _ in range(n)],
            "delta_omega": [rng.normal(size=3) for _ in range(n)],
            "delta_f": 0.7,
        }

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_a_is_the_jacobian_of_the_models_own_dynamics(self, n):
        r"""Central-difference ``(δα, δω̇)`` against ``A s + B δu`` off hover.

        The reference is one the plant could actually be on — its ``ω̇_i^d``
        comes from the model — so the comparison is between two descriptions of
        the same system rather than between the linearisation and an invented
        operating point.

        This is the test that kills the errata-(a) mutants. Reverting either
        ``speed_sq * eye`` to the MATLAB ``speed_sq * ones((3, 3))`` broadcast
        moves ``δα`` (the ``[0:3]`` site) or ``δω̇`` (the ``i != j`` site) by
        order 10%, while every hover-only assertion stays green.
        """
        mdl, pose, ref, inputs = moving_reference(n)
        ctrl = QuadrotorFlexCableController.from_model(mdl)
        A, B = ctrl.linearize(ref, inputs)

        rng = np.random.default_rng(11)
        q_d = [np.asarray(q, dtype=float) for q in ref.link_attitudes]
        w_d = [np.asarray(w, dtype=float) for w in ref.link_angular_velocities]
        R_d = np.asarray(ref.orientation, dtype=float)
        Omega_d = np.asarray(ref.angular_velocity, dtype=float)
        direction = self._error_direction(ctrl, rng)
        # ξ_i and δω_i live in q_i^d's tangent plane; a component along q_i^d
        # is the gauge, which is not a perturbation of anything.
        xi = [v - np.dot(v, q_d[i]) * q_d[i] for i, v in enumerate(direction["xi"])]
        d_omega = [v - np.dot(v, q_d[i]) * q_d[i] for i, v in enumerate(direction["delta_omega"])]

        s = np.zeros(ctrl.state_dim)
        s[6:9] = direction["eta"]
        s[9:12] = direction["delta_Omega"]
        for i in range(n):
            s[12 + 3 * i : 12 + 3 * i + 3] = xi[i]
            s[12 + 3 * n + 3 * i : 12 + 3 * n + 3 * i + 3] = d_omega[i]
        du = np.array([direction["delta_f"], 0.0, 0.0, 0.0])

        def sample(step):
            """The plant's ``(α, ω̇)`` at the reference displaced by ``step * s``."""
            dirs = []
            rates = []
            for i in range(n):
                # q_i^d × q_i = step ξ_i, to first order, by construction.
                q = q_d[i] + step * np.cross(xi[i], q_d[i])
                q /= np.linalg.norm(q)
                # ω_i + q_i × (q_i × ω_i^d) = step δω_i, likewise.
                w = w_d[i] + step * d_omega[i]
                dirs.append(q)
                rates.append(w - np.dot(w, q) * q)
            R = R_d @ expm(step * hat(direction["eta"]))
            Omega = R.T @ R_d @ Omega_d + step * direction["delta_Omega"]
            pose(dirs, rates, R, Omega)
            accel = mdl.accelerations(
                float(inputs.thrust) + step * direction["delta_f"], inputs.moment
            )
            return accel["quad_accel"].copy(), accel["link_ang_accels"].copy()

        eps = 1e-6
        (alpha_p, omegadot_p), (alpha_m, omegadot_m) = sample(eps), sample(-eps)
        fd_alpha = (alpha_p - alpha_m) / (2.0 * eps)
        fd_omegadot = (omegadot_p - omegadot_m) / (2.0 * eps)

        predicted_alpha = A[3:6, :] @ s + B[3:6, :] @ du
        predicted_omegadot = (A[12 + 3 * n :, :] @ s + B[12 + 3 * n :, :] @ du).reshape(n, 3)

        # The reference is genuinely off hover, or the comparison is vacuous.
        assert np.linalg.norm(predicted_alpha) > 1.0
        assert np.linalg.norm(predicted_omegadot) > 1.0
        np.testing.assert_allclose(
            fd_alpha, predicted_alpha, rtol=1e-5, atol=1e-5 * np.linalg.norm(predicted_alpha)
        )
        np.testing.assert_allclose(
            fd_omegadot,
            predicted_omegadot,
            rtol=1e-5,
            atol=1e-5 * np.linalg.norm(predicted_omegadot),
        )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_xi_block_only_ever_feeds_the_gauge_direction(self, n):
        r"""``ξ̇_i``'s dependence on ``ξ_i`` is ``q_i^d (q_i^d)^\top \hat\omega_i^d``.

        The outer-product projector is the whole content of that block: without
        it the row would be ``\hat\omega_i^d``, which rotates ``ξ_i`` inside the
        tangent plane, and the tangent reduction would lose something real
        rather than only leaking along the gauge. At hover ``ω_i^d = 0`` and the
        block is zero either way, which is why this needs a moving reference.
        """
        _, _, ref, inputs = moving_reference(n)
        ctrl = QuadrotorFlexCableController(num_links=n)
        A, _ = ctrl.linearize(ref, inputs)
        for i in range(n):
            q_d = np.asarray(ref.link_attitudes[i], dtype=float)
            row = 12 + 3 * i
            block = A[row : row + 3, row : row + 3]
            assert np.linalg.norm(block) > TOL_PHYSICAL, "reference link is not turning"
            tangential = (np.eye(3) - np.outer(q_d, q_d)) @ block
            np.testing.assert_allclose(tangential, 0.0, atol=TOL_ANALYTIC)

    @staticmethod
    def _gauge_leak(ctrl, A, link_directions=None):
        r"""``\|(I - WW^\top) A W\| / \|A\|`` — how far ``A`` leaves the subspace."""
        W = ctrl.tangent_projection(link_directions)
        gauge = np.eye(A.shape[0]) - W @ W.T
        return float(np.linalg.norm(gauge @ A @ W) / np.linalg.norm(A))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_tangent_reduction_is_exact_at_hover_and_approximate_off_it(self, n):
        r"""The honest form of "``W`` spans an invariant subspace".

        ``B``'s range lies inside the subspace at every reference — no input
        reaches a gauge direction, ever. ``A``'s does not. Dotting ``ξ̇_i`` with
        ``q_i^d`` leaves ``ξ_i \cdot (q_i^d \times \omega_i^d)``, and dotting
        ``δ\dot\omega_i`` with it leaves a term in ``\dot\omega_i^d``; both
        vanish only where the reference chain is at rest and unaccelerated.

        So the shipped gain — solved at the hanging equilibrium — rests on an
        exact reduction, and :meth:`set_gain_from` used as a *scheduling* hook
        does not. On the admissible trajectories the flat map actually produces
        the residual stays around ``1e-4``–``1e-3``, which is what makes the
        approximation a usable one; it is not zero, and the docstrings that
        called the reduction exact everywhere were wrong.
        """
        ctrl = QuadrotorFlexCableController(num_links=n)

        A_hover, _ = ctrl.linearize()
        assert self._gauge_leak(ctrl, A_hover) < TOL_ANALYTIC

        # A reference the flat map really produced, not a hand-built one: the
        # magnitude quoted in the docs is only meaningful on those.
        traj = gentle_lissajous()
        worst = 0.0
        for t in np.linspace(0.0, 12.0, 40):
            ref, inputs = recovered_reference(ctrl, traj, t)
            A, B = ctrl.linearize(ref, inputs)
            W = ctrl.tangent_projection(ref.link_attitudes)
            np.testing.assert_allclose((np.eye(A.shape[0]) - W @ W.T) @ B, 0.0, atol=TOL_EXACT)
            worst = max(worst, self._gauge_leak(ctrl, A, ref.link_attitudes))

        assert worst > 1e-6, (
            "no leak anywhere on a moving reference — either the trajectory is "
            "not turning the chain or the xi block lost its omega_d term"
        )
        assert worst < 5e-3, f"gauge leak {worst:.2e}; too large to call the schedule usable"

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_gauge_leak_is_exactly_the_two_reference_motion_terms(self, n):
        r"""Not a magnitude bound — the closed form of the leak, term for term.

        Differentiating the two gauge conditions along the reference gives

        .. math::

            q_i^d \cdot \dot\xi_i = \xi_i \cdot (q_i^d \times \omega_i^d),
            \qquad
            q_i^d \cdot \delta\dot\omega_i = -\,\dot\omega_i^d \cdot \delta q_i,
            \qquad \delta q_i = -\hat q_i^d\, \xi_i ,

        both of which vanish exactly when the reference chain is at rest — the
        hanging equilibrium, where the shipped gain is solved — and neither of
        which vanishes on a moving reference. This is the correction to the
        docs' old claim that dotting ``ξ̇_i`` with ``q_i^d`` "gives 0 on both
        sides".

        The reference here is on-shell: its ``ω̇_i^d`` comes from the plant, so
        the second identity is a statement about the same system the first is.
        """
        mdl, _, ref, inputs = moving_reference(n)
        ctrl = QuadrotorFlexCableController.from_model(mdl)
        A, _ = ctrl.linearize(ref, inputs)

        rng = np.random.default_rng(2)
        s = np.zeros(ctrl.state_dim)
        xi = []
        for i in range(n):
            q_d = np.asarray(ref.link_attitudes[i], dtype=float)
            v = rng.normal(size=3)
            v -= np.dot(v, q_d) * q_d
            xi.append(v)
            s[12 + 3 * i : 12 + 3 * i + 3] = v

        rates = A @ s
        for i in range(n):
            q_d = np.asarray(ref.link_attitudes[i], dtype=float)
            omega_d = np.asarray(ref.link_angular_velocities[i], dtype=float)
            omegadot_d = np.asarray(ref.link_angular_accelerations[i], dtype=float)

            xi_dot = rates[12 + 3 * i : 12 + 3 * i + 3]
            expected = float(np.dot(xi[i], np.cross(q_d, omega_d)))
            assert abs(expected) > TOL_PHYSICAL, "reference link is not turning"
            assert float(np.dot(q_d, xi_dot)) == pytest.approx(expected, rel=1e-9)

            omegadot = rates[12 + 3 * n + 3 * i : 12 + 3 * n + 3 * i + 3]
            delta_q = -hat(q_d) @ xi[i]
            expected = -float(np.dot(omegadot_d, delta_q))
            assert abs(expected) > TOL_PHYSICAL
            assert float(np.dot(q_d, omegadot)) == pytest.approx(expected, rel=1e-9)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_gauge_leak_grows_with_how_hard_the_reference_works(self, n):
        """Two-sided on the magnitude: gentler reference, smaller leak.

        A residual that did not shrink with the reference's own motion would
        not be the term above.
        """
        ctrl = QuadrotorFlexCableController(num_links=n)

        def worst(traj):
            out = 0.0
            for t in np.linspace(0.0, 12.0, 40):
                ref, inputs = recovered_reference(ctrl, traj, t)
                A, _ = ctrl.linearize(ref, inputs)
                out = max(out, self._gauge_leak(ctrl, A, ref.link_attitudes))
            return out

        assert worst(aggressive_lissajous()) > 5.0 * worst(gentle_lissajous())

    @pytest.mark.parametrize("n", [1, 2])
    def test_set_gain_from_uses_the_pair_it_is_handed(self, n):
        """The public gain-scheduling hook, exercised at a non-hover linearisation.

        Its whole purpose is to accept an ``(A, B)`` the caller computed
        somewhere else; a version that quietly re-derived the hover pair would
        satisfy every other test in this file, since every other test schedules
        at hover or not at all.

        The two solves below share the *same* tangent basis and differ only in
        the pair handed in, so nothing but the ``(A, B)`` arguments can account
        for the difference.
        """
        _, _, ref, inputs = moving_reference(n)
        ctrl = QuadrotorFlexCableController(num_links=n)
        hover_gain = ctrl.gain.copy()
        directions = ref.link_attitudes

        ctrl.set_gain_from(*ctrl.linearize(ref, inputs), directions)
        scheduled = ctrl.gain.copy()
        scheduled_poles = ctrl.closed_loop_poles.copy()

        ctrl.set_gain_from(*ctrl.linearize(), directions)
        frozen = ctrl.gain.copy()

        assert np.linalg.norm(scheduled - frozen) > 1e-2 * np.linalg.norm(frozen), (
            "scheduling at a non-hover reference gave the same gain as the "
            "hover pair on the same basis; set_gain_from is ignoring its (A, B)"
        )
        assert np.linalg.norm(scheduled - hover_gain) > 1e-2 * np.linalg.norm(hover_gain)
        # Still a stabilising solution, and still blind to the gauge.
        assert scheduled_poles.real.max() < 0.0
        W = ctrl.tangent_projection(directions)
        gauge = np.eye(ctrl.state_dim) - W @ W.T
        np.testing.assert_allclose(scheduled @ gauge, 0.0, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("n", [1, 2])
    def test_the_link_rate_error_is_not_a_plain_difference(self, n):
        r"""``δω_i = ω_i + q_i \times (q_i \times ω_i^d)``, and the third term matters.

        It differs from the naive ``ω_i - ω_i^d`` by ``q_i (q_i \cdot ω_i^d)``,
        which is zero exactly when ``q_i = q_i^d`` — i.e. everywhere the rest of
        this suite looks. The projected form is the one that lives in ``q_i``'s
        tangent plane, which is where the linearisation's ``δω`` columns act.
        """
        _, _, ref, _ = moving_reference(n)
        ctrl = QuadrotorFlexCableController(num_links=n)
        state = ref.as_state()
        # Tip each link off its reference direction, leaving the rates alone.
        rng = np.random.default_rng(5)
        for i in range(n):
            q_d = np.asarray(ref.link_attitudes[i], dtype=float)
            tilted = q_d + 0.25 * rng.normal(size=3)
            state.link_attitudes[i] = S2(tilted / np.linalg.norm(tilted))

        error = ctrl.error_state(state, ref)
        for i in range(n):
            q = np.asarray(state.link_attitudes[i], dtype=float)
            omega = np.asarray(state.link_angular_velocities[i], dtype=float)
            omega_d = np.asarray(ref.link_angular_velocities[i], dtype=float)
            block = error[12 + 3 * n + 3 * i : 12 + 3 * n + 3 * i + 3]
            np.testing.assert_allclose(
                block, omega + np.cross(q, np.cross(q, omega_d)), atol=TOL_EXACT
            )
            naive = omega - omega_d
            # The two differ by exactly q_i (q_i · ω_i^d) — zero when the link
            # sits on its reference direction, which is everywhere else this
            # suite looks, and order 0.3 rad/s here.
            np.testing.assert_allclose(block - naive, q * float(np.dot(q, omega_d)), atol=TOL_EXACT)
            assert np.linalg.norm(block - naive) > TOL_PHYSICAL, (
                "the tilted link left q . omega_d at zero — this configuration "
                "cannot tell the two conventions apart"
            )


# ─── Controllability, and its limits ──────────────────────────────────


class TestControllability:
    """What this cascade stabilises, and what it does not.

    M4's analogue is its grasp rank: a structural limit made explicit and
    tested rather than worked around. M5's is subtler, because nothing here is
    *un*controllable in the rank sense — it is the *authority* that runs out.
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_the_reduced_pair_is_controllable_and_the_loop_is_stable(self, n):
        ctrl = QuadrotorFlexCableController(num_links=n)
        assert ctrl.controllability_margin > 0.0
        assert ctrl.closed_loop_poles.real.max() < 0.0
        assert ctrl.closed_loop_poles.shape == (12 + 4 * n,)

    def test_the_controllability_margin_collapses_with_the_link_count(self):
        """Controllable at every ``n``, and less so at each one.

        The claim the docstring makes quantitative: roughly an order of
        magnitude lost every two links. It is why "controllable" on its own is
        not a useful answer for this system.
        """
        margins = [
            QuadrotorFlexCableController(num_links=n).controllability_margin for n in (1, 2, 3, 5)
        ]
        assert all(m > 0.0 for m in margins)
        assert margins == sorted(margins, reverse=True)
        assert margins[0] / margins[-1] > 50.0

    @pytest.mark.parametrize("n", [2, 3])
    def test_the_slowest_closed_loop_mode_is_the_chains_fastest_natural_one(self, n):
        """Not the payload swing — the top of the chain's own spectrum.

        Which is also the mode the ZOH-Euler integrator is conditionally stable
        against, so the two limits of this system are the *same* frequency.
        """
        mdl = QuadrotorFlexCableBase(num_links=n)
        slowest = mdl.payload_controller.closed_loop_poles[0]
        assert abs(slowest.imag) == pytest.approx(mdl.natural_frequencies().max(), rel=2e-2)

    @staticmethod
    def _bending_mode(ctrl, model):
        """The closed-loop pole sitting on the chain's top natural frequency."""
        top = model.natural_frequencies().max()
        poles = ctrl.closed_loop_poles
        return poles[int(np.argmin(np.abs(np.abs(poles.imag) - top)))]

    @staticmethod
    def _damping(pole):
        return float(-pole.real / abs(pole))

    def test_uniform_weight_scaling_cannot_move_the_slowest_mode(self):
        """Sixty times the gain buys 5 % of damping — and proves nothing.

        This is the experiment the docs used to rest "no choice of weights
        fixes it" on, and it is the one perturbation that *cannot* work: for an
        LQR, scaling the whole of ``Q`` is identical to scaling ``R^{-1}``, so
        it slides along the cheap-control locus, where the finite closed-loop
        poles converge to the plant's own stable modes. The result below is
        real; the conclusion people drew from it was not, and
        :meth:`test_targeted_link_rate_weighting_damps_the_top_bending_mode`
        is the counterexample.
        """
        default = QuadrotorFlexCableController(num_links=3)
        heavy = QuadrotorFlexCableController(
            num_links=3,
            q_position=500.0,
            q_velocity=500.0,
            q_attitude=750.0,
            q_angular_rate=750.0,
            q_link=1000.0,
            q_link_rate=750.0,
        )
        # Uniform in the sense that matters: every block scaled by 1000.
        ratios = {
            heavy.cost_weights[name] / default.cost_weights[name]
            for name in default.cost_weights
            if name.startswith("q_")
        }
        assert ratios == {1000.0}, "this test is only meaningful for a uniform scaling"

        # The gain really did move — otherwise the comparison is vacuous.
        assert np.abs(heavy.gain).max() > 3.0 * np.abs(default.gain).max()
        slow_default = default.closed_loop_poles[0].real
        slow_heavy = heavy.closed_loop_poles[0].real
        assert abs(slow_heavy - slow_default) < 0.10 * abs(slow_default)

    @pytest.mark.parametrize("n", [2, 3])
    def test_targeted_link_rate_weighting_damps_the_top_bending_mode(self, n):
        """The claim "no choice of weights fixes it" is false, and this is why.

        Raising ``q_link_rate`` *alone* by 50x — 3.3x the gain norm at ``n = 3``,
        an order of magnitude less than the uniform scaling above — takes the
        chain's top bending mode from 1.3 % damping to 8 %, and the slowest pole
        overall from -0.34 to -0.51. What is structural here is the PBH margin,
        not the damping.
        """
        model = QuadrotorFlexCableBase(num_links=n)
        default = QuadrotorFlexCableController.from_model(model)
        targeted = QuadrotorFlexCableController.from_model(
            model, q_link_rate=50.0 * DEFAULT_FLEXCABLE_LQR_Q_LINK_RATE
        )

        base_mode = self._bending_mode(default, model)
        damped_mode = self._bending_mode(targeted, model)
        assert self._damping(base_mode) < 0.05
        assert self._damping(damped_mode) > 3.0 * self._damping(base_mode)

        # And it is not bought by brute gain: the uniform 1000x scaling above
        # moves the gain norm by ~60x for nothing.
        gain_growth = np.linalg.norm(targeted.gain) / np.linalg.norm(default.gain)
        assert gain_growth < 5.0, f"gain grew {gain_growth:.1f}x — not a targeted change"

        # The slowest mode overall is better *damped* too, not just the bending
        # one. (Its real part is not always more negative: at n = 2 a different,
        # much better damped mode takes over as slowest.)
        assert self._damping(targeted.closed_loop_poles[0]) > 3.0 * self._damping(
            default.closed_loop_poles[0]
        )

    @pytest.mark.slow
    @pytest.mark.integration
    def test_damping_the_chain_costs_payload_tracking(self):
        """The other half of the trade-off, which is why it is not the default.

        Same 50x on ``q_link_rate``: the tail RMS payload error on the gentle
        Lissajous goes from sub-millimetre to over a centimetre. The shipped
        defaults buy tracking; a caller who wants a quiet cable pays for it
        here.
        """
        traj = gentle_lissajous()
        errors = {}
        for label, kwargs in (
            ("default", {}),
            ("damped", {"q_link_rate": 50.0 * DEFAULT_FLEXCABLE_LQR_Q_LINK_RATE}),
        ):
            mdl = QuadrotorFlexCableBase(num_links=3)
            assert_stable_regime(mdl)
            mdl.payload_controller = QuadrotorFlexCableController.from_model(mdl, **kwargs)
            errors[label] = fly(mdl, traj, 12.0, "jet")["tail_rms"]

        assert errors["default"] < 2e-3
        assert errors["damped"] > 5.0 * errors["default"]

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_the_moment_weight_costs_chain_damping_and_the_cost_grows_with_n(self, n):
        """The price of the ``R`` deviation, which is not "under 2 %".

        ``r_moment`` 0.2 -> 20 is an *input* weight, so it barely touches the
        chain — but "barely" is n-dependent, and from ``n = 2`` up it is always
        in the worse direction: 1.5 % at ``n = 2``, 4.6 % at ``n = 3``, 13 % at
        ``n = 5``. (At ``n = 1`` it is 0.02 % and marginally the other way,
        which is the only reading that supports the old "under 2 %".) The
        deviation is still right — see
        :data:`~udaan.core.defaults.DEFAULT_FLEXCABLE_LQR_R_MOMENT` for the
        steady-state errors at 0.2 — but it is not free.
        """
        cheap = QuadrotorFlexCableController(num_links=n, r_moment=0.2)
        shipped = QuadrotorFlexCableController(num_links=n, r_moment=20.0)
        slow_cheap = cheap.closed_loop_poles[0].real
        slow_shipped = shipped.closed_loop_poles[0].real

        if n > 1:
            assert slow_shipped > slow_cheap, "the chain got *more* damped — check the sign"
        shift = abs(slow_shipped - slow_cheap) / abs(slow_cheap)
        expected = {1: 0.0002, 2: 0.0154, 3: 0.0461, 5: 0.1310}[n]
        assert shift == pytest.approx(expected, abs=2e-3)

        # And what it buys: the fastest pole out of the hold's phase-loss band.
        assert abs(cheap.closed_loop_poles[-1].real) > 800.0
        assert abs(shipped.closed_loop_poles[-1].real) < 100.0


# ─── Payload tracking ─────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
class TestPayloadTracking:
    @pytest.mark.parametrize("n", STABLE_COUNTS)
    def test_step_response_converges(self, n):
        target = np.array([1.0, 0.5, 2.0])
        mdl = QuadrotorFlexCableBase(num_links=n)
        assert_stable_regime(mdl)
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.simulate(8.0, payload_position=np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(mdl.state.payload_position, target, atol=TOL_PHYSICAL)
        # And it got there hanging, not swinging: the quadrotor is a full cable
        # above the load.
        assert np.linalg.norm(mdl.state.position - mdl.state.payload_position) == pytest.approx(
            mdl.cable_length, abs=TOL_PHYSICAL
        )

    @pytest.mark.parametrize("n", STABLE_COUNTS)
    def test_tracks_a_moving_trajectory(self, n):
        traj = gentle_lissajous()
        mdl = QuadrotorFlexCableBase(num_links=n)
        assert_stable_regime(mdl)
        result = fly(mdl, traj, 12.0, "jet")
        assert result["tail_rms"] < TOL_PHYSICAL
        assert result["drops"] == 0

    def test_a_step_is_regulated_without_any_feedforward(self):
        """The feedback-only law on its own has to be a working controller.

        A stationary setpoint needs no flat map at all — the hanging
        equilibrium *is* the reference — so this is the one tracking assertion
        that isolates the gain from the feedforward entirely.
        """
        target = np.array([-0.6, 0.8, 1.4])
        mdl = QuadrotorFlexCableBase(num_links=3)
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(8.0, payload_position=np.array([0.0, 0.0, 1.4]))
        assert mdl.feedforward_drops == 0
        np.testing.assert_allclose(mdl.state.payload_position, target, atol=TOL_PHYSICAL)


# ─── Flatness feedforward ─────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
class TestFlatnessFeedforward:
    """Two-sided, discriminating, and run through :meth:`simulate` itself.

    M3's review found the equivalent branch dead because its tests hand-rolled
    the stepping loop and bound a plain lambda. :func:`fly` calls
    ``mdl.simulate`` and wraps ``mdl.step`` only to sample the error, so
    deleting the feedforward branch of ``simulate`` fails these outright.
    """

    @pytest.mark.parametrize("n", STABLE_COUNTS)
    def test_feedforward_improves_tracking(self, n):
        traj = gentle_lissajous()
        without = fly(QuadrotorFlexCableBase(num_links=n), traj, 12.0, "off")
        with_ff = fly(QuadrotorFlexCableBase(num_links=n), traj, 12.0, "jet")

        # Two-sided: the feedback-only run has to be bad enough that there is
        # something to improve.
        assert without["rms"] > 0.15, "trajectory too easy — the comparison proves nothing"
        assert with_ff["rms"] < 0.5 * without["rms"]
        assert with_ff["tail_rms"] < 0.05 * without["tail_rms"]
        assert with_ff["drops"] == 0

    @pytest.mark.parametrize("n", [2, 3])
    def test_the_full_jet_beats_a_jet_padded_past_snap(self, n):
        """What ``2n + 4`` actually buys.

        A trajectory that stops at snap can still be fed forward — the padded
        rows only enter the recovered ``Ω``, ``Ω̇`` and moment, not the
        positions — so it is already most of the benefit. Supplying the
        remaining ``2n`` derivatives in closed form is worth about another
        order of magnitude on steady-state tracking, which is the whole reason
        :meth:`~udaan.utils.trajectory.CrazyTrajectory.get_jet` exists.
        """
        traj = gentle_lissajous()
        feedback_only = fly(QuadrotorFlexCableBase(num_links=n), traj, 12.0, "off")
        padded = fly(QuadrotorFlexCableBase(num_links=n), traj, 12.0, "snap")
        exact = fly(QuadrotorFlexCableBase(num_links=n), traj, 12.0, "jet")
        assert exact["tail_rms"] < 0.5 * padded["tail_rms"]
        assert padded["drops"] == exact["drops"] == 0

        # "Worse than exact" on its own would be satisfied by a padded path
        # that had silently stopped feeding anything forward. Bound it from
        # below as well: the padded jet is most of the benefit, not a token
        # one, and it tracks to under a centimetre in absolute terms.
        assert padded["tail_rms"] < 0.05 * feedback_only["tail_rms"]
        assert padded["tail_rms"] < 1e-2

    def test_simulate_asks_for_a_reference_on_every_step(self, monkeypatch):
        """The wiring, pinned: every step of a feedforward run gets a reference,
        and a plain closure gets none at all."""
        traj = gentle_lissajous()
        seen: list[bool] = []
        original = QuadrotorFlexCableBase._flat_reference

        def spy(self, flat_map, jet_fn, t):
            out = original(self, flat_map, jet_fn, t)
            seen.append(out is not None)
            return out

        monkeypatch.setattr(QuadrotorFlexCableBase, "_flat_reference", spy)

        fly(QuadrotorFlexCableBase(num_links=2), traj, 6.0, "jet")
        assert len(seen) > 100, "simulate() never asked for a flat reference"
        assert all(seen), "some step ran without the recovered reference"

        seen.clear()
        fly(QuadrotorFlexCableBase(num_links=2), traj, 6.0, "off")
        assert not seen, "a plain-closure setpoint must not unlock the feedforward"

    def test_the_reference_wrench_reaches_the_command(self):
        """The feedforward is not merely fetched — it is *used*.

        At zero error the commanded wrench must be exactly the flat map's
        ``(f_d, M_d)``, which no feedback-only law can produce for a moving
        reference.
        """
        mdl = QuadrotorFlexCableBase(num_links=2)
        ctrl = mdl.payload_controller
        traj = gentle_lissajous()
        ctrl.setpoint = traj.get
        reference = recovered_reference(ctrl, traj, 3.0)
        ref, inputs = reference
        mdl.reset(
            payload_position=ref.payload_position,
            payload_velocity=ref.payload_velocity,
            orientation=np.asarray(ref.orientation),
            angular_velocity=np.asarray(ref.angular_velocity),
            link_attitudes=ref.link_attitudes,
            link_angular_velocities=ref.link_angular_velocities,
        )
        wrench = ctrl.compute(3.0, mdl.state, reference)
        np.testing.assert_allclose(
            wrench, np.concatenate([[inputs.thrust], inputs.moment]), atol=TOL_ANALYTIC
        )
        # ... and the moment is genuinely non-trivial, so this is not a test
        # that zero equals zero.
        assert np.linalg.norm(inputs.moment) > 1e-6

    def test_a_controller_without_a_flat_map_gets_no_feedforward(self, caplog):
        """Falling back on a map built from the *model's* parameters would
        regulate one machine against another's reference, silently."""
        traj = gentle_lissajous()
        mdl = QuadrotorFlexCableBase(num_links=2)
        real = mdl.payload_controller

        class Opaque:
            """Publishes a setpoint and a wrench, but not its configuration."""

            def __init__(self, inner):
                self._inner = inner
                self.setpoint = traj.get

            def compute(self, *args):
                return self._inner.compute(args[0], args[1])

        mdl.payload_controller = Opaque(real)
        with caplog.at_level(logging.WARNING, logger=_MODEL_LOGGER):
            mdl.simulate(2.0, payload_position=traj.get(0.0)[0])
        assert "exposes no flat_map" in caplog.text
        assert mdl.feedforward_drops == 0

    def test_a_snap_only_trajectory_is_warned_about_once(self, caplog):
        """The padded rows are fabricated; say so rather than absorb it."""
        traj = gentle_lissajous()
        mdl = QuadrotorFlexCableBase(num_links=3)
        mdl.payload_controller.setpoint = SnapOnlyTrajectory(traj).get
        with caplog.at_level(logging.WARNING, logger=_MODEL_LOGGER):
            mdl.simulate(1.0, payload_position=traj.get(0.0)[0])
        assert caplog.text.count("supplies derivatives only through snap") == 1
        assert "10-th derivative" in caplog.text

    def test_an_infeasible_reference_is_dropped_counted_and_warned(self, caplog):
        """A trajectory that dives past free fall unloads the chain.

        The flat map refuses those samples. The right response is to run them
        feedback-only against the hanging equilibrium and *count* it, not to
        abort the simulation and not to bury it at debug level.
        """

        class FreeFall:
            """Straight down at well past ``g`` — every link goes slack."""

            def get(self, t):
                accel = np.array([0.0, 0.0, -3.0 * GRAVITY])
                return np.array([0.0, 0.0, 3.0]) + 0.5 * accel * t**2, accel * t, accel

            def get_jet(self, t, order):
                jet = np.zeros((order + 1, 3))
                jet[0] = (
                    np.array([0.0, 0.0, 3.0]) + 0.5 * np.array([0.0, 0.0, -3.0 * GRAVITY]) * t**2
                )
                jet[1] = np.array([0.0, 0.0, -3.0 * GRAVITY]) * t
                if order >= 2:
                    jet[2] = np.array([0.0, 0.0, -3.0 * GRAVITY])
                return jet

        mdl = QuadrotorFlexCableBase(num_links=2)
        mdl.payload_controller.setpoint = FreeFall().get
        with caplog.at_level(logging.WARNING, logger=_MODEL_LOGGER):
            mdl.simulate(0.5, payload_position=np.array([0.0, 0.0, 3.0]))
        assert mdl.feedforward_drops > 0
        assert caplog.text.count("flatness feedforward dropped") == 1

    def test_a_malformed_jet_is_a_programming_error_and_propagates(self):
        """Not every refusal is a physical one.

        A trajectory that returns the wrong *shape* is a bug in the trajectory,
        and degrading it to feedback-only would hide it forever.
        """

        class WrongShape:
            def get(self, t):
                return np.zeros(3), np.zeros(3), np.zeros(3)

            def get_jet(self, t, order):
                return np.zeros((order + 1, 2))

        mdl = QuadrotorFlexCableBase(num_links=2)
        mdl.payload_controller.setpoint = WrongShape().get
        with pytest.raises(ValueError, match="dim"):
            mdl.simulate(0.1, payload_position=np.zeros(3))

    def test_the_feedforward_drop_counter_resets_per_simulate(self):
        traj = gentle_lissajous()
        mdl = QuadrotorFlexCableBase(num_links=2)
        mdl.payload_controller.setpoint = traj.get
        mdl.simulate(1.0, payload_position=traj.get(0.0)[0])
        mdl._feedforward_drops = 7
        mdl.simulate(1.0, payload_position=traj.get(0.0)[0])
        assert mdl.feedforward_drops == 0


# ─── Heterogeneous chains ─────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.integration
class TestHeterogeneousChain:
    """Nothing above assumes equal links; this makes sure of it."""

    def test_unequal_masses_and_lengths_still_track(self):
        target = np.array([0.7, -0.4, 1.6])
        mdl = QuadrotorFlexCableBase(num_links=3)
        mdl.link_lengths = [0.5, 0.3, 0.2]
        mdl.link_masses = [0.05, 0.03, 0.25]
        assert_stable_regime(mdl)
        # The setters had to reach the controller, or the gain below regulates
        # a chain that no longer exists.
        np.testing.assert_allclose(
            mdl.payload_controller.link_lengths, mdl.link_lengths, atol=TOL_EXACT
        )
        mdl.payload_controller.setpoint = hover_setpoint(target)
        mdl.simulate(8.0, payload_position=np.array([0.0, 0.0, 1.6]))
        np.testing.assert_allclose(mdl.state.payload_position, target, atol=TOL_PHYSICAL)

    def test_a_heterogeneous_chain_tracks_with_feedforward(self):
        traj = gentle_lissajous()
        mdl = QuadrotorFlexCableBase(num_links=3)
        mdl.link_lengths = [0.5, 0.3, 0.2]
        mdl.link_masses = [0.05, 0.03, 0.25]
        assert_stable_regime(mdl)
        result = fly(mdl, traj, 12.0, "jet")
        assert result["tail_rms"] < TOL_PHYSICAL
        assert result["drops"] == 0

    def test_a_stale_gain_cannot_survive_a_parameter_change(self):
        """The failure mode :meth:`sync_from_model` exists to prevent.

        Every cached quantity — the flat map *and* the gain — is rebuilt when
        the model's parameters move, so a controller re-synced after the fact
        is identical to one freshly built.
        """
        mdl = QuadrotorFlexCableBase(num_links=2)
        before = mdl.payload_controller.gain.copy()
        mdl.link_masses = [0.4, 0.9]
        after = mdl.payload_controller.gain
        assert not np.allclose(before, after)
        fresh = QuadrotorFlexCableController.from_model(mdl)
        np.testing.assert_allclose(after, fresh.gain, atol=TOL_ANALYTIC)
