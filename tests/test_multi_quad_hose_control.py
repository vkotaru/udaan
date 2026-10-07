r"""Tests for the M6 hose LQR — :mod:`udaan.control.multi_quad_hose`.

Four things are worth the machinery here, and they are what the file is
organised around.

1. **The linearisation is checked against the plant, not against itself.**
   :class:`TestLinearization` finite-differences the *nonlinear* error dynamics
   driven by
   :meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.accelerations` and
   compares the Jacobian with
   :meth:`~udaan.control.multi_quad_hose.MultiQuadHoseController.linearize`.
   That catches a transcription slip anywhere in the block assembly, and it
   also pins the one place the published derivation and the exact error
   dynamics part company (see :func:`_error_rate`).

2. **The gauge claim is measured, not asserted.**  ``B``'s rows on the ``2n``
   ``S^2`` gauge directions must vanish at *every* reference and ``A``'s at
   every *static* one; both are checked, and so is the failure the reduction
   exists to avoid — handing the embedded pair to the Riccati solver.

3. **The feedforward is exercised through** :meth:`simulate` **itself.**  M3
   and M5 both had that branch dead because their tests hand-rolled the loop.
   Every tracking number below comes out of ``mdl.simulate(...)``.

4. **Every configuration is inside the ZOH envelope.**  The model's usable
   region is ``n <= 4`` at any ``nQ``
   (:meth:`~udaan.models.multi_quad_hose.MultiQuadHoseBase.stability_growth_rate`),
   and nothing here leaves it.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_INTEGRATION, TOL_PHYSICAL
from udaan.control.multi_quad_hose import MultiQuadHoseController
from udaan.core.defaults import GRAVITY
from udaan.flatness.jet import Jet
from udaan.flatness.multi_quad_hose import NotFlatError
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.multi_quad_hose import MultiQuadHoseBase
from udaan.utils import hat, rodrigues_expm, vee
from udaan.utils.trajectory import CrazyTrajectory

_E3 = np.array([0.0, 0.0, 1.0])

#: ``(num_links, attachments)`` pairs the whole file sweeps.  All are inside
#: the ``n <= 4`` ZOH envelope; the spread covers a two-vehicle span, an
#: interior vehicle, a vehicle at every vertex, a hose with a free tail above
#: its first vehicle, and the M5 configuration (one vehicle, at the end).
CONFIGS = [
    (2, (0, 2)),
    (3, (0, 3)),
    (4, (0, 4)),
    (4, (0, 2, 4)),
    (4, (0, 1, 2, 3, 4)),
    (3, (1, 3)),
    (2, (2,)),
]

#: Patterns where *every* inter-vehicle segment is a single rigid link.  The
#: trim's shape does not depend on the horizontal pull for these, so the sag
#: criterion has no root and the pre-tension fallback is what runs.  They are
#: split out rather than folded into :data:`CONFIGS` because the whole point is
#: that they take a different branch of :meth:`_solve_trim_tensions`.
SINGLE_LINK_CONFIGS = [
    (1, (0, 1)),
    (2, (0, 1, 2)),
    (3, (0, 1, 2, 3)),
    (4, (0, 1, 2, 3, 4)),
]


def build(n, attachments, **kwargs):
    """A ``(model, controller)`` pair that agree on every parameter."""
    model = MultiQuadHoseBase(num_links=n, attachments=attachments, input="wrench")
    return model, MultiQuadHoseController.from_model(model, **kwargs)


def gentle_trajectory(center=None, ax=0.3, f1=0.25):
    return CrazyTrajectory(
        ax=ax,
        ay=0.8 * ax,
        az=0.5 * ax,
        f1=f1,
        f2=0.8 * f1,
        f3=0.6 * f1,
        center=np.zeros(3) if center is None else np.asarray(center, dtype=float),
    )


# ── Finite-difference machinery for TestLinearization ──────────────────


def _apply_perturbation(model, ctrl, ref, reduced, basis, *, variational):
    """Write the perturbed state onto ``model.state`` and return it.

    ``variational=True`` adds the reference rates (``\\omega = \\omega^d +
    \\delta\\omega``), which is the convention
    :meth:`MultiQuadHoseController.linearize` implements and every geometric
    paper in this lineage uses.  ``False`` inverts the *exact* error definition
    of :meth:`MultiQuadHoseController.error_state`, which transports the
    reference rate to the current configuration.  The two differ at first order
    in the reference rates and agree identically where those vanish.
    """
    n, nQ = ctrl.num_links, ctrl.num_quadrotors
    s = basis @ reduced
    state = model.state
    state.root_position = np.asarray(ref.root_position, dtype=float) + s[0:3]
    state.root_velocity = np.asarray(ref.root_velocity, dtype=float) + s[3:6]
    for i in range(n):
        q_d = np.asarray(ref.link_attitudes[i], dtype=float)
        omega_d = np.asarray(ref.link_angular_velocities[i], dtype=float)
        xi = s[6 + 3 * i : 9 + 3 * i]
        delta = s[6 + 3 * n + 3 * i : 9 + 3 * n + 3 * i]
        q = rodrigues_expm(xi) @ q_d
        q = q / np.linalg.norm(q)
        omega = omega_d + delta if variational else delta - np.cross(q, np.cross(q, omega_d))
        omega = omega - np.dot(omega, q) * q
        state.link_attitudes[i] = S2(q)
        state.link_angular_velocities[i] = TS2(omega)
    base = 6 + 6 * n
    for k in range(nQ):
        R_d = np.asarray(ref.orientations[k], dtype=float)
        Omega_d = np.asarray(ref.angular_velocities[k], dtype=float)
        eta = s[base + 3 * k : base + 3 * k + 3]
        delta = s[base + 3 * nQ + 3 * k : base + 3 * nQ + 3 * k + 3]
        R = R_d @ rodrigues_expm(eta)
        state.orientations[k] = SO3(R)
        state.angular_velocities[k] = TSO3(
            Omega_d + delta if variational else R.T @ R_d @ Omega_d + delta
        )
    model._sync_quad_states()
    return state


def _error_rate(model, ctrl, ref, reduced, delta_u, basis, nominal_u, *, variational):
    """``d/dt`` of the error state at a perturbed state and input."""
    n, nQ = ctrl.num_links, ctrl.num_quadrotors
    state = _apply_perturbation(model, ctrl, ref, reduced, basis, variational=variational)
    wrench = nominal_u + np.asarray(delta_u, dtype=float)
    acc = model.accelerations(wrench[:nQ], wrench[nQ:].reshape(nQ, 3))

    out = np.zeros(ctrl.state_dim)
    out[0:3] = np.asarray(state.root_velocity) - np.asarray(ref.root_velocity)
    out[3:6] = acc["root_accel"] - np.asarray(ref.root_acceleration)
    for i in range(n):
        q = np.asarray(state.link_attitudes[i], dtype=float)
        omega = np.asarray(state.link_angular_velocities[i], dtype=float)
        dq = np.cross(omega, q)
        q_d = np.asarray(ref.link_attitudes[i], dtype=float)
        omega_d = np.asarray(ref.link_angular_velocities[i], dtype=float)
        alpha_d = np.asarray(ref.link_angular_accelerations[i], dtype=float)
        out[6 + 3 * i : 9 + 3 * i] = np.cross(np.cross(omega_d, q_d), q) + np.cross(q_d, dq)
        if variational:
            out[6 + 3 * n + 3 * i : 9 + 3 * n + 3 * i] = acc["link_ang_accels"][i] - alpha_d
        else:
            out[6 + 3 * n + 3 * i : 9 + 3 * n + 3 * i] = (
                acc["link_ang_accels"][i]
                + np.cross(dq, np.cross(q, omega_d))
                + np.cross(q, np.cross(dq, omega_d))
                + np.cross(q, np.cross(q, alpha_d))
            )
    base = 6 + 6 * n
    for k in range(nQ):
        R = np.asarray(state.orientations[k], dtype=float)
        Omega = np.asarray(state.angular_velocities[k], dtype=float)
        R_d = np.asarray(ref.orientations[k], dtype=float)
        Omega_d = np.asarray(ref.angular_velocities[k], dtype=float)
        dOmega_d = np.asarray(ref.angular_accelerations[k], dtype=float)
        relative = R_d.T @ R
        drelative = -hat(Omega_d) @ relative + relative @ hat(Omega)
        out[base + 3 * k : base + 3 * k + 3] = 0.5 * vee(drelative - drelative.T)
        rate = acc["quad_ang_accels"][k]
        out[base + 3 * nQ + 3 * k : base + 3 * nQ + 3 * k + 3] = (
            rate - dOmega_d
            if variational
            else rate + hat(Omega) @ R.T @ R_d @ Omega_d - R.T @ R_d @ dOmega_d
        )
    return out


def numerical_jacobians(model, ctrl, ref, inputs, *, variational=True, step=1e-6):
    """``(dA, dB, residual)`` — finite differences of the nonlinear error dynamics.

    ``dA`` is ``(state_dim, reduced_dim)``: only the tangent directions are
    realisable as state perturbations, so the comparison is against ``A @ W``.
    """
    basis = ctrl.tangent_projection(ref.link_attitudes)
    reduced_dim = basis.shape[1]
    nominal = np.concatenate(
        [np.asarray(inputs.thrusts, dtype=float), np.asarray(inputs.moments, dtype=float).ravel()]
    )
    zero_state = np.zeros(reduced_dim)
    zero_input = np.zeros(ctrl.input_dim)

    def rate(z, du):
        return _error_rate(model, ctrl, ref, z, du, basis, nominal, variational=variational)

    residual = rate(zero_state, zero_input)
    dA = np.zeros((ctrl.state_dim, reduced_dim))
    for j in range(reduced_dim):
        bump = np.zeros(reduced_dim)
        bump[j] = step
        dA[:, j] = (rate(bump, zero_input) - rate(-bump, zero_input)) / (2 * step)
    dB = np.zeros((ctrl.state_dim, ctrl.input_dim))
    for j in range(ctrl.input_dim):
        bump = np.zeros(ctrl.input_dim)
        bump[j] = step
        dB[:, j] = (rate(zero_state, bump) - rate(zero_state, -bump)) / (2 * step)
    return dA, dB, residual


# ── Construction and validation ────────────────────────────────────────


class TestConstruction:
    def test_from_model_copies_every_parameter(self):
        model = MultiQuadHoseBase(num_links=4, attachments=(0, 2, 4))
        ctrl = MultiQuadHoseController.from_model(model)
        assert ctrl.num_links == 4
        assert ctrl.attachments == (0, 2, 4)
        np.testing.assert_allclose(ctrl.masses, model.masses, atol=TOL_EXACT)
        np.testing.assert_allclose(ctrl.vertex_masses, model.vertex_masses, atol=TOL_EXACT)
        np.testing.assert_allclose(ctrl.link_lengths, model.link_lengths, atol=TOL_EXACT)
        np.testing.assert_allclose(
            ctrl.vertex_total_masses, model.vertex_total_masses, atol=TOL_EXACT
        )
        assert ctrl.total_mass == pytest.approx(model.total_mass)

    def test_state_and_input_dimensions(self):
        _, ctrl = build(4, (0, 2, 4))
        assert ctrl.state_dim == 6 + 6 * 4 + 6 * 3
        assert ctrl.input_dim == 4 * 3
        assert ctrl.jet_order == 2 * 4 + 4

    def test_no_vehicle_at_last_vertex_is_not_flat(self):
        # The structural refusal: vertex n has no link below it, so its force
        # balance constrains the flat output rather than being determined by it.
        with pytest.raises(NotFlatError, match="not differentially flat"):
            MultiQuadHoseController(num_links=4, attachments=(0, 2))

    def test_pd_gains_are_refused(self):
        for name in ("kp", "kd", "ki"):
            with pytest.raises(ValueError, match=f"'{name}' has nothing to tune"):
                MultiQuadHoseController(num_links=2, **{name: np.ones(3)})

    @pytest.mark.parametrize(
        "weight",
        [
            "q_root_position",
            "q_root_velocity",
            "q_link",
            "q_link_rate",
            "q_attitude",
            "q_angular_rate",
            "r_thrust",
            "r_moment",
        ],
    )
    def test_non_positive_weights_are_refused(self, weight):
        with pytest.raises(ValueError, match=f"{weight} must be strictly positive"):
            MultiQuadHoseController(num_links=2, **{weight: 0.0})

    def test_bad_num_links(self):
        with pytest.raises(ValueError, match="num_links must be >= 1"):
            MultiQuadHoseController(num_links=0)

    def test_bad_attachments(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            MultiQuadHoseController(num_links=4, attachments=(4, 2))
        with pytest.raises(ValueError, match=r"must lie in \[0, 4\]"):
            MultiQuadHoseController(num_links=4, attachments=(0, 5))

    def test_bad_masses(self):
        with pytest.raises(ValueError, match="masses must be strictly positive"):
            MultiQuadHoseController(num_links=2, masses=[1.0, 0.0])
        with pytest.raises(ValueError, match="masses must be a scalar or have 2 entries"):
            MultiQuadHoseController(num_links=2, masses=[1.0, 1.0, 1.0])

    def test_bad_vertex_masses(self):
        with pytest.raises(ValueError, match="vertex_masses must be non-negative"):
            MultiQuadHoseController(num_links=2, vertex_masses=[0.1, -0.1, 0.1])
        with pytest.raises(ValueError, match="vertex_masses must be a scalar or have 3 entries"):
            MultiQuadHoseController(num_links=2, vertex_masses=[0.1, 0.1])

    def test_massless_interior_vertex_is_refused(self):
        # Vertices 0 and 2 carry vehicles; vertex 1 carries nothing, so with a
        # zero hose mass there the two links meeting at it hinge freely.
        with pytest.raises(ValueError, match="strictly positive total mass"):
            MultiQuadHoseController(num_links=2, attachments=(0, 2), vertex_masses=[0.1, 0.0, 0.1])

    def test_bad_link_lengths(self):
        with pytest.raises(ValueError, match="link_lengths must be strictly positive"):
            MultiQuadHoseController(num_links=2, link_lengths=[1.0, 0.0])

    def test_bad_inertias(self):
        with pytest.raises(ValueError, match=r"inertias must be \(3,\), \(3, 3\) or \(2, 3, 3\)"):
            MultiQuadHoseController(num_links=2, attachments=(0, 2), inertias=np.zeros((4, 3, 3)))

    def test_bad_sag_and_tension(self):
        with pytest.raises(ValueError, match=r"sag_ratio must lie in \(0, 1\)"):
            MultiQuadHoseController(num_links=2, sag_ratio=1.0)
        with pytest.raises(ValueError, match="horizontal_tension must be strictly positive"):
            MultiQuadHoseController(num_links=2, horizontal_tension=0.0)
        with pytest.raises(ValueError, match="min_tension must be non-negative"):
            MultiQuadHoseController(num_links=2, min_tension=-1.0)

    def test_trim_tensions_setter_checks_shape(self):
        _, ctrl = build(4, (0, 2, 4))
        with pytest.raises(ValueError, match=r"trim_tensions must have shape \(2, 3\)"):
            ctrl.trim_tensions = np.zeros((3, 3))

    def test_sync_from_model_tracks_parameter_changes(self):
        model, ctrl = build(3, (0, 3))
        first = ctrl.gain.copy()
        model.vertex_masses = np.array([0.2, 0.3, 0.15, 0.25])
        ctrl.sync_from_model(model)
        np.testing.assert_allclose(ctrl.vertex_masses, model.vertex_masses, atol=TOL_EXACT)
        # The flat map is rebuilt at the new parameters, not left stale.
        np.testing.assert_allclose(ctrl.flat_map.vertex_masses, model.vertex_masses, atol=TOL_EXACT)
        assert not np.allclose(ctrl.gain, first)

    def test_sync_refuses_a_different_hose(self):
        _, ctrl = build(3, (0, 3))
        other = MultiQuadHoseBase(num_links=4, attachments=(0, 4))
        with pytest.raises(ValueError, match="cannot sync a 3-link controller"):
            ctrl.sync_from_model(other)
        other = MultiQuadHoseBase(num_links=3, attachments=(1, 3))
        with pytest.raises(ValueError, match="cannot sync a controller for attachments"):
            ctrl.sync_from_model(other)

    def test_model_setter_keeps_the_lqr_in_step(self):
        # The defect M4's review found: a setter that writes only self._... .
        model, _ = build(3, (0, 3))
        model.masses = np.array([0.9, 1.1])
        np.testing.assert_allclose(model.hose_controller.masses, model.masses, atol=TOL_EXACT)

    def test_heterogeneous_parameters(self):
        model = MultiQuadHoseBase(
            num_links=4,
            attachments=(0, 2, 4),
            input="wrench",
            link_lengths=[0.4, 0.6, 0.35, 0.55],
            vertex_masses=[0.05, 0.12, 0.03, 0.09, 0.06],
            masses=[0.8, 1.3, 0.95],
        )
        model.inertias = np.stack(
            [np.diag(v) for v in ([0.002, 0.002, 0.004], [0.003, 0.0025, 0.005], [0.0018] * 3)]
        )
        ctrl = MultiQuadHoseController.from_model(model)
        ref, inputs = ctrl.equilibrium_reference(0.0)
        # A genuine trim of the heterogeneous machine: zero acceleration.
        model.reset(
            root_position=ref.root_position,
            link_attitudes=ref.link_attitudes,
            orientations=[np.asarray(R) for R in ref.orientations],
        )
        acc = model.accelerations(inputs.thrusts, inputs.moments)
        assert np.max(np.abs(acc["root_accel"])) < TOL_ANALYTIC
        assert np.max(np.abs(acc["link_ang_accels"])) < TOL_ANALYTIC
        # And the gain solves.
        assert ctrl.gain.shape == (12, ctrl.state_dim)
        assert ctrl.closed_loop_poles.real.max() < 0.0


# ── The equilibrium trim ───────────────────────────────────────────────


class TestEquilibriumTrim:
    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_trim_is_an_equilibrium_of_the_plant(self, n, attachments):
        model, ctrl = build(n, attachments)
        ref, inputs = ctrl.equilibrium_reference(0.0)
        model.reset(
            root_position=ref.root_position,
            link_attitudes=ref.link_attitudes,
            orientations=[np.asarray(R) for R in ref.orientations],
        )
        acc = model.accelerations(inputs.thrusts, inputs.moments)
        assert np.max(np.abs(acc["root_accel"])) < TOL_ANALYTIC
        assert np.max(np.abs(acc["link_ang_accels"])) < TOL_ANALYTIC
        assert np.max(np.abs(acc["quad_ang_accels"])) < TOL_ANALYTIC

    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_no_link_is_slack_and_thrusts_carry_the_weight(self, n, attachments):
        _, ctrl = build(n, attachments)
        _, inputs = ctrl.equilibrium_reference(0.0)
        # The headline invariant, asserted as the invariant rather than as
        # "positive": every derived tension is bounded below by the horizontal
        # pull H. A bare ``> 0.0`` passed at 1.4e-9 N on the degenerate trims
        # the sag search used to return for all-single-link patterns.
        if ctrl.num_quadrotors > 1:
            horizontal = float(ctrl.trim_tensions[0, 0])
            assert horizontal > 0.5
            assert np.min(inputs.tensions) >= horizontal - TOL_ANALYTIC
        assert np.min(inputs.tensions) > 0.5
        # The team as a whole carries the system weight; individually nobody is
        # asked for more than the shipped 20 N limit on these parameters.
        assert np.sum(inputs.thrusts) >= ctrl.total_mass * GRAVITY - TOL_ANALYTIC
        assert np.max(inputs.thrusts) < 20.0

    @pytest.mark.parametrize(("n", "attachments"), SINGLE_LINK_CONFIGS)
    def test_all_single_link_patterns_get_a_usable_trim(self, n, attachments):
        """The sag criterion has no root here, and the fallback has to be taken.

        Every inter-vehicle segment is one rigid link, so the trim spans its own
        full length at *any* horizontal pull and ``sag_ratio`` — 0.8, less than
        1 — asks for less than the span can ever be.  The bisection therefore
        has no root to find.  Bracketing only from above, it used to walk
        ``high`` down to ~1e-9 N and hand back a trim of essentially zero
        tension: a slack hose, which the flat map rejects outright at
        ``n = 2``, and which silently made the published tables for
        ``(0, 1, 2, 3, 4)`` measurements of a degenerate horizontal trim rather
        than the documented catenary.
        """
        _, ctrl = build(n, attachments)
        pretension = GRAVITY * float(ctrl.vertex_masses.sum())
        assert float(ctrl.trim_tensions[0, 0]) == pytest.approx(pretension, rel=1e-12)
        ref, inputs = ctrl.equilibrium_reference(0.0)
        assert np.min(inputs.tensions) >= pretension - TOL_ANALYTIC
        assert np.max(inputs.thrusts) < 20.0
        # Straight and level whatever the pull — which is exactly why the sag
        # criterion cannot pick one.
        vertices = np.asarray(ref.vertex_positions)
        np.testing.assert_allclose(vertices[:, 2], vertices[0, 2], atol=TOL_ANALYTIC)
        assert ctrl.closed_loop_poles.real.max() < 0.0

    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_every_segment_comes_back_level(self, n, attachments):
        """What the half in ``-½ w`` buys: each segment hangs symmetrically.

        The vertical component at the top of a segment is minus *half* that
        segment's interior weight, so the segment descends and returns to the
        height it started at.  Any other fraction leaves consecutive vehicles
        at different heights.
        """
        _, ctrl = build(n, attachments)
        ref, _ = ctrl.equilibrium_reference(0.0)
        heights = np.asarray(ref.vertex_positions)[list(ctrl.attachments), 2]
        np.testing.assert_allclose(heights, heights[0], atol=TOL_ANALYTIC)

    def test_trim_spans_the_requested_sag(self):
        _, ctrl = build(4, (0, 4), sag_ratio=0.6)
        ref, _ = ctrl.equilibrium_reference(0.0)
        vertices = ref.vertex_positions
        span = float(np.linalg.norm(vertices[4, :2] - vertices[0, :2]))
        assert span == pytest.approx(0.6 * float(ctrl.link_lengths.sum()), rel=1e-6)

    def test_interior_vehicles_hover_level(self):
        # Only the outermost two vehicles pull horizontally; the horizontal
        # components cancel at every vehicle in between.
        _, ctrl = build(4, (0, 2, 4))
        ref, _ = ctrl.equilibrium_reference(0.0)
        thrust_axis = np.asarray(ref.orientations[1]) @ _E3
        np.testing.assert_allclose(thrust_axis, _E3, atol=TOL_ANALYTIC)

    def test_setpoint_velocity_is_a_uniform_translation(self):
        # A trim moving at constant velocity is still an exact solution, so the
        # feedback-only reference costs nothing in consistency.
        model, ctrl = build(3, (0, 3))
        velocity = np.array([0.3, -0.2, 0.1])
        ctrl.setpoint = lambda t: (np.array([0.0, 0.0, 1.0]), velocity, np.zeros(3))
        ref, inputs = ctrl.equilibrium_reference(0.0)
        np.testing.assert_allclose(ref.root_velocity, velocity, atol=TOL_EXACT)
        np.testing.assert_allclose(ref.root_acceleration, np.zeros(3), atol=TOL_ANALYTIC)
        for k in range(ctrl.num_quadrotors):
            np.testing.assert_allclose(ref.velocities[k], velocity, atol=TOL_ANALYTIC)

    def test_explicit_horizontal_tension_overrides_the_sag_search(self):
        _, ctrl = build(4, (0, 4), horizontal_tension=3.0)
        np.testing.assert_allclose(ctrl.trim_tensions[:, 0], 3.0, atol=TOL_EXACT)

    def test_single_vehicle_hose_has_no_flat_tensions(self):
        _, ctrl = build(2, (2,))
        assert ctrl.trim_tensions.shape == (0, 3)
        ref, _ = ctrl.equilibrium_reference(0.0)
        # Nothing prescribed: the hose simply hangs below its one vehicle.
        for q in ref.link_attitudes:
            np.testing.assert_allclose(np.asarray(q), _E3, atol=TOL_ANALYTIC)


# ── Linearisation ──────────────────────────────────────────────────────


class TestLinearization:
    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_matches_finite_differences_at_the_trim(self, n, attachments):
        model, ctrl = build(n, attachments)
        ref, inputs = ctrl.equilibrium_reference(0.0)
        A, B = ctrl.linearize(ref, inputs)
        basis = ctrl.tangent_projection(ref.link_attitudes)
        dA, dB, residual = numerical_jacobians(model, ctrl, ref, inputs)
        # The reference is an exact solution: the error rate vanishes at s = 0.
        assert np.max(np.abs(residual)) < TOL_ANALYTIC
        scale = max(1.0, float(np.max(np.abs(A))))
        assert np.max(np.abs(A @ basis - dA)) / scale < 1e-8
        assert np.max(np.abs(B - dB)) / max(1.0, float(np.max(np.abs(B)))) < 1e-8

    @pytest.mark.parametrize(("n", "attachments"), [(3, (0, 3)), (4, (0, 2, 4)), (2, (2,))])
    def test_matches_finite_differences_at_a_moving_reference(self, n, attachments):
        model, ctrl = build(n, attachments)
        traj = gentle_trajectory()
        ref, inputs = ctrl.flat_map.recover(ctrl.build_flats(traj.get_jet(0.7, ctrl.jet_order)))
        A, B = ctrl.linearize(ref, inputs)
        basis = ctrl.tangent_projection(ref.link_attitudes)
        dA, dB, residual = numerical_jacobians(model, ctrl, ref, inputs)
        # The flat map's reference is an exact solution here too.
        assert np.max(np.abs(residual)) < TOL_ANALYTIC
        scale = max(1.0, float(np.max(np.abs(A))))
        assert np.max(np.abs(A @ basis - dA)) / scale < 1e-8
        assert np.max(np.abs(B - dB)) / max(1.0, float(np.max(np.abs(B)))) < 1e-8

    def test_variational_and_exact_rate_conventions_part_company_off_a_trim(self):
        """The published linearisation is *variational*, and that is a choice.

        ``\\delta\\omega`` and ``\\delta\\Omega`` in :meth:`error_state` are the
        reference rates transported to the current configuration; the published
        blocks treat them as if they simply added.  The two agree exactly where
        the reference rates vanish, and differ at first order in them
        elsewhere.  Both halves are asserted, because a reader meeting only the
        first would conclude more than is true.
        """
        model, ctrl = build(3, (0, 3))
        ref, inputs = ctrl.equilibrium_reference(0.0)
        basis = ctrl.tangent_projection(ref.link_attitudes)
        A, _ = ctrl.linearize(ref, inputs)
        exact, _, _ = numerical_jacobians(model, ctrl, ref, inputs, variational=False)
        assert np.max(np.abs(A @ basis - exact)) / np.max(np.abs(A)) < 1e-8

        traj = gentle_trajectory()
        ref, inputs = ctrl.flat_map.recover(ctrl.build_flats(traj.get_jet(0.7, ctrl.jet_order)))
        basis = ctrl.tangent_projection(ref.link_attitudes)
        A, _ = ctrl.linearize(ref, inputs)
        exact, _, _ = numerical_jacobians(model, ctrl, ref, inputs, variational=False)
        gap = np.max(np.abs(A @ basis - exact)) / np.max(np.abs(A))
        assert 1e-4 < gap < 1e-1

    def test_needs_reference_and_inputs_together(self):
        _, ctrl = build(3, (0, 3))
        ref, inputs = ctrl.equilibrium_reference(0.0)
        with pytest.raises(ValueError, match="needs the reference state and its inputs together"):
            ctrl.linearize(ref, None)
        with pytest.raises(ValueError, match="needs the reference state and its inputs together"):
            ctrl.linearize(None, inputs)

    def test_rejects_a_reference_for_a_different_hose(self):
        _, ctrl = build(3, (0, 3))
        _, other = build(4, (0, 4))
        ref, inputs = other.equilibrium_reference(0.0)
        with pytest.raises(ValueError, match="reference describes 4 links"):
            ctrl.linearize(ref, inputs)

    @pytest.mark.parametrize(
        ("amplitude", "frequency", "lower", "upper"),
        [(0.15, 0.15, 0.0, 0.05), (0.4, 0.3, 0.05, 0.5), (0.8, 0.45, 1.0, 100.0)],
    )
    def test_drift_over_trajectory_aggressiveness(self, amplitude, frequency, lower, upper):
        """The measurement the constant-gain choice rests on.

        The gain is solved once at the trim, so the claim being made is that
        ``A`` barely moves over the references it will be flown on.  That claim
        holds only for gentle ones: the drift runs ~2 %, ~18 % and ~300 % over
        the three rows here, and the flat map accepts all of them — at
        ``nQ >= 2`` its hanging-branch test is off by construction, so nothing
        in the feedforward path flags that the reference has left the gain's
        domain.
        """
        _, ctrl = build(3, (0, 3))
        ref0, inputs0 = ctrl.equilibrium_reference(0.0)
        A0, _ = ctrl.linearize(ref0, inputs0)
        norm0 = np.linalg.norm(A0)
        traj = gentle_trajectory(ax=amplitude, f1=frequency)
        worst = 0.0
        for t in np.linspace(0.0, 6.0, 25):
            ref, inputs = ctrl.flat_map.recover(ctrl.build_flats(traj.get_jet(t, ctrl.jet_order)))
            A, _ = ctrl.linearize(ref, inputs)
            worst = max(worst, float(np.linalg.norm(A - A0) / norm0))
        assert lower <= worst <= upper


# ── Gauge directions and the Riccati solve ─────────────────────────────


class TestGaugeAndGain:
    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_input_matrix_has_no_gauge_rows(self, n, attachments):
        """``B``'s rows on the ``2n`` gauge directions vanish at every reference."""
        _, ctrl = build(n, attachments)
        traj = gentle_trajectory()
        for source in ("trim", "moving"):
            if source == "trim":
                ref, inputs = ctrl.equilibrium_reference(0.0)
            else:
                ref, inputs = ctrl.flat_map.recover(
                    ctrl.build_flats(traj.get_jet(0.4, ctrl.jet_order))
                )
            _, B = ctrl.linearize(ref, inputs)
            basis = ctrl.tangent_projection(ref.link_attitudes)
            leak = np.eye(ctrl.state_dim) - basis @ basis.T
            assert np.linalg.norm(leak @ B) / np.linalg.norm(B) < TOL_EXACT * ctrl.state_dim

    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_tangent_reduction_is_exact_at_a_static_trim(self, n, attachments):
        """Stronger than M5's claim, which is about a *hanging* equilibrium.

        At any trim the link row's own ``xi`` block is ``-hat(v_i) hat(q_i)``
        with ``v_i`` parallel to ``q_i`` by the trim condition itself, and
        ``-lambda hat(q) hat(q) = lambda (I - q q^T)`` has a zero ``q`` row.
        """
        _, ctrl = build(n, attachments)
        ref, inputs = ctrl.equilibrium_reference(0.0)
        A, _ = ctrl.linearize(ref, inputs)
        basis = ctrl.tangent_projection(ref.link_attitudes)
        leak = np.eye(ctrl.state_dim) - basis @ basis.T
        assert np.linalg.norm(leak @ A @ basis) / np.linalg.norm(A) < TOL_ANALYTIC

    def test_reduction_is_only_approximate_at_a_moving_reference(self):
        _, ctrl = build(3, (0, 3))
        traj = gentle_trajectory()
        ref, inputs = ctrl.flat_map.recover(ctrl.build_flats(traj.get_jet(0.7, ctrl.jet_order)))
        A, _ = ctrl.linearize(ref, inputs)
        basis = ctrl.tangent_projection(ref.link_attitudes)
        leak = np.eye(ctrl.state_dim) - basis @ basis.T
        residual = np.linalg.norm(leak @ A @ basis) / np.linalg.norm(A)
        assert 1e-5 < residual < 1e-1

    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_embedded_pair_is_unstabilisable(self, n, attachments):
        """What the reduction exists to avoid — and how it actually shows up.

        The ``2n`` gauge coordinates sit at eigenvalue zero with no input
        reaching them, so no stabilising solution exists.  M5 reports that
        :func:`~scipy.linalg.solve_continuous_are` *raises*, and on M5's own
        pair it does at every ``n`` from 1 to 8.  Here it raises on five of the
        seven configurations; on ``(0, 2)`` and ``(0, 4)`` it returns instead a
        ``P`` of norm ``1e9`` and ``1e12`` whose Riccati residuals are 2 and 51
        and whose closed loops still have their eigenvalues on the imaginary
        axis.  Which configurations land on which side is a property of an
        extremely ill-conditioned pencil and moves with the solver version, so
        the claim under test is the weaker and version-independent one: **no
        strictly stabilising solution comes back**, however the solver reports
        it.
        """
        from scipy.linalg import solve_continuous_are

        _, ctrl = build(n, attachments)
        ref, inputs = ctrl.equilibrium_reference(0.0)
        A, B = ctrl.linearize(ref, inputs)
        R = np.eye(ctrl.input_dim)
        try:
            P = solve_continuous_are(A, B, np.eye(ctrl.state_dim), R)
        except (ValueError, np.linalg.LinAlgError):
            return
        closed = A - B @ np.linalg.solve(R, B.T @ P)
        assert np.linalg.eigvals(closed).real.max() > -1e-9

    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_gain_stabilises_and_ignores_the_gauge(self, n, attachments):
        _, ctrl = build(n, attachments)
        assert ctrl.closed_loop_poles.real.max() < 0.0
        assert ctrl.controllability_margin > 0.0
        # K acts only on the tangent subspace, so a pure gauge perturbation of
        # the error produces no control at all.
        ref, _ = ctrl.equilibrium_reference(0.0)
        gauge = np.zeros(ctrl.state_dim)
        for i in range(n):
            gauge[6 + 3 * i : 9 + 3 * i] = np.asarray(ref.link_attitudes[i], dtype=float)
        np.testing.assert_allclose(ctrl.gain @ gauge, 0.0, atol=TOL_ANALYTIC)

    def test_spreading_the_vehicles_recovers_authority(self):
        """The controllability limitation, and where it comes from.

        ``(4,)`` is M5's configuration — one vehicle at the top of a four-link
        chain — and it reproduces M5's collapse.  Putting a vehicle at each end
        helps, and one in the middle helps much more: the limitation is about
        *where* the vehicles are, not how many there are.
        """
        margins = {
            att: build(4, att)[1].controllability_margin for att in [(4,), (0, 4), (0, 2, 4)]
        }
        assert margins[(4,)] < margins[(0, 4)] < margins[(0, 2, 4)]
        assert margins[(4,)] < 0.15
        assert margins[(0, 2, 4)] > 0.3

    def test_gain_is_solved_lazily_and_invalidated_by_a_parameter_change(self):
        _, ctrl = build(3, (0, 3))
        assert ctrl._gain is None
        first = ctrl.gain
        assert ctrl._gain is not None
        ctrl.trim_tensions = ctrl.trim_tensions * 1.5
        assert ctrl._gain is None
        assert not np.allclose(ctrl.gain, first)

    def test_riccati_failure_is_reported_with_the_hose_in_the_message(self):
        _, ctrl = build(2, (0, 2))
        ref, _ = ctrl.equilibrium_reference(0.0)
        dim = ctrl.state_dim
        with pytest.raises(ValueError, match="no stabilising solution"):
            # An input matrix that reaches nothing is unstabilisable on any
            # subspace the projection leaves.
            ctrl.set_gain_from(np.eye(dim), np.zeros((dim, ctrl.input_dim)), ref.link_attitudes)

    def test_a_badly_scaled_hose_is_refused_rather_than_silently_regulated(self):
        """The failure mode the solver does *not* report.

        These are legal parameters — every one of them passes
        ``_set_parameters`` — and :func:`scipy.linalg.solve_continuous_are`
        returns for them without raising.  What it returns is a ``P`` of norm
        ``1e12`` whose residual is thousands of times ``\\|Q\\|`` and whose
        closed loop sits a hair off the imaginary axis; before
        :func:`~udaan.control.solve_lqr_gain` the caller got a gain of norm
        ``1e5`` with no signal at all, and ``controllability_margin`` and
        ``closed_loop_poles`` reported on it as though it were sound.
        """
        from scipy.linalg import solve_continuous_are

        ctrl = MultiQuadHoseController(
            num_links=3,
            attachments=(3,),
            vertex_masses=[0.3752061014, 4.11619e-05, 2.3129e-06, 0.0006789892],
            link_lengths=[0.0394038932, 0.0187179868, 0.5508040919],
            masses=[0.031151622],
        )
        # The solver itself is perfectly happy: this is not a caught exception
        # being re-raised, it is a check on a returned answer.
        A, B = ctrl.linearize()
        ref, _ = ctrl.equilibrium_reference(0.0)
        basis = ctrl.tangent_projection(ref.link_attitudes)
        Q_r, R = ctrl.cost_matrices()
        P = solve_continuous_are(basis.T @ A @ basis, basis.T @ B, Q_r, R)
        assert np.linalg.norm(P) > 1e10

        with pytest.raises(ValueError, match="does not solve the posed problem"):
            _ = ctrl.gain

    @pytest.mark.parametrize(("n", "attachments"), CONFIGS + SINGLE_LINK_CONFIGS)
    def test_the_residual_check_has_room_to_spare_on_the_shipped_hoses(self, n, attachments):
        """The threshold is not load-bearing on anything that ships.

        A check that sat just above the configurations it has to pass would be
        a tuning knob; this one clears them by ten orders of magnitude, which
        is the claim worth pinning.
        """
        from scipy.linalg import solve_continuous_are

        from udaan.control import ARE_RESIDUAL_TOLERANCE, ARE_RESIDUAL_WARNING

        _, ctrl = build(n, attachments)
        A, B = ctrl.linearize()
        ref, _ = ctrl.equilibrium_reference(0.0)
        basis = ctrl.tangent_projection(ref.link_attitudes)
        A_r, B_r = basis.T @ A @ basis, basis.T @ B
        Q_r, R = ctrl.cost_matrices()
        P = solve_continuous_are(A_r, B_r, Q_r, R)
        gain_r = -np.linalg.solve(R, B_r.T @ P)
        residual = A_r.T @ P + P @ A_r + P @ B_r @ gain_r + Q_r
        relative = float(np.linalg.norm(residual)) / float(np.linalg.norm(Q_r))
        assert relative < 1e-10 < ARE_RESIDUAL_WARNING < ARE_RESIDUAL_TOLERANCE


# ── Weights ────────────────────────────────────────────────────────────


class TestWeights:
    def test_uniform_scaling_says_nothing(self):
        """For an LQR, scaling ``Q`` uniformly is scaling ``R^-1``.

        It slides along the cheap-control locus, so it is the one perturbation
        guaranteed to be uninformative — which is exactly the experiment that
        misled M5's first write-up.
        """
        _, base = build(4, (0, 2, 4))
        _, scaled = build(
            4,
            (0, 2, 4),
            q_root_position=500.0,
            q_root_velocity=500.0,
            q_link=750.0,
            q_link_rate=750.0,
            q_attitude=1000.0,
            q_angular_rate=750.0,
        )
        assert abs(scaled.closed_loop_poles[0].real - base.closed_loop_poles[0].real) < 0.3

    def test_targeted_position_weights_help_and_rate_weights_hurt(self):
        """Both halves, because the second is the surprising one.

        M5's lesson was that *targeted* weighting reaches a mode uniform
        scaling cannot.  It does here too — but in the opposite direction from
        M5: weighting the link **rates** moves M6's slowest pole the wrong way,
        and weighting the positions is what helps.
        """
        _, base = build(4, (0, 2, 4))
        slowest = base.closed_loop_poles[0].real
        _, position = build(4, (0, 2, 4), q_root_position=25.0)
        _, link_rate = build(4, (0, 2, 4), q_link_rate=37.5)
        assert position.closed_loop_poles[0].real < slowest
        assert link_rate.closed_loop_poles[0].real > slowest

    def test_published_input_weights_leave_an_unholdable_attitude_pole(self):
        """The M6 half of M5's input-weighting erratum, and its price.

        ``R = 0.2 I`` weights a thrust and a moment equally though their
        columns of ``B`` differ by ~500x.  The shipped ``r_moment = 20``
        slows the attitude loop to something the 5 ms hold can carry, and —
        unlike M5, where it cost up to 13 % of the chain's slowest real part —
        here it is nearly free.
        """
        model, published = build(4, (0, 4), r_moment=0.2)
        _, shipped = build(4, (0, 4))
        fast_published = abs(published.closed_loop_poles[-1].real)
        fast_shipped = abs(shipped.closed_loop_poles[-1].real)
        assert fast_published > 500.0
        assert fast_shipped < 150.0
        # Half a sample of hold lag at the closed-loop frequency.
        assert np.degrees(fast_published * model.dt / 2.0) > 90.0
        assert np.degrees(fast_shipped * model.dt / 2.0) < 20.0
        # And the hose modes barely notice.
        assert shipped.closed_loop_poles[0].real == pytest.approx(
            published.closed_loop_poles[0].real, rel=0.02
        )


# ── Error state ────────────────────────────────────────────────────────


class TestErrorState:
    def test_vanishes_at_the_reference(self):
        model, ctrl = build(4, (0, 2, 4))
        ref, _ = ctrl.equilibrium_reference(0.0)
        model.reset(
            root_position=ref.root_position,
            link_attitudes=ref.link_attitudes,
            orientations=[np.asarray(R) for R in ref.orientations],
        )
        np.testing.assert_allclose(ctrl.error_state(model.state, ref), 0.0, atol=TOL_ANALYTIC)

    def test_blocks_are_where_the_docstring_says(self):
        model, ctrl = build(2, (0, 2))
        ref, _ = ctrl.equilibrium_reference(0.0)
        model.reset(
            root_position=np.asarray(ref.root_position) + np.array([0.1, -0.2, 0.3]),
            link_attitudes=ref.link_attitudes,
            orientations=[np.asarray(R) for R in ref.orientations],
        )
        error = ctrl.error_state(model.state, ref)
        np.testing.assert_allclose(error[0:3], [0.1, -0.2, 0.3], atol=TOL_ANALYTIC)
        assert np.max(np.abs(error[3:])) < TOL_ANALYTIC

    def test_conventions_hold_at_a_moving_reference(self):
        r"""Each block evaluated *away* from the trim, at a nonzero value.

        The two tests above pin the translational block and pin everything at
        zero; neither ever evaluates ``\xi``, ``\delta\omega``, ``\eta`` or
        ``\delta\Omega`` anywhere but the origin, and a trim has
        ``\omega^d = \Omega^d = 0`` so the two rate blocks' transport terms are
        never exercised at all.  Swapping ``cross(q_d, q)`` for ``cross(q, q_d)``
        passes both, and every test in ``TestLinearization`` besides.

        Two perturbations, each with a closed form that does not restate the
        implementation:

        * rotate the configuration and put the rates *on* the transported
          reference.  ``q_i^d \times \exp(\hat{\theta n}) q_i^d = \sin\theta\, n``
          for ``n \perp q_i^d`` fixes ``\xi``'s sign, the same identity fixes
          ``\eta``'s, and both rate blocks must vanish — the transported rate
          is written here as the tangential projection
          ``\omega^d - (\omega^d\cdot q) q``, which is
          ``-q \times (q \times \omega^d)`` by an identity the implementation
          uses in the other form.
        * hold the configuration on the reference and offset the rates.  Then
          each rate block is exactly the offset.
        """
        model, ctrl = build(3, (0, 3))
        traj = gentle_trajectory(center=np.array([0.0, 0.0, 1.5]), ax=0.4, f1=0.3)
        ref, _ = ctrl.flat_map.recover(ctrl.build_flats(traj.get_jet(0.7, ctrl.jet_order)))
        n, nQ = ctrl.num_links, ctrl.num_quadrotors
        base = 6 + 6 * n

        # Vacuous unless the reference really is moving.  Link 0's direction is
        # the constant flat tension's, so its rate is identically zero and the
        # others carry the check.
        assert sum(np.linalg.norm(w) > 1e-3 for w in ref.link_angular_velocities) >= n - 1
        assert min(np.linalg.norm(w) for w in ref.angular_velocities) > 1e-3

        angles = [0.13, -0.21, 0.07]
        attitude_angles = [0.17, -0.09]
        axes, attitude_axes, expected_xi, expected_eta = [], [], [], []
        link_attitudes, link_rates = [], []
        for i in range(n):
            q_d = np.asarray(ref.link_attitudes[i], dtype=float)
            axis = np.cross(q_d, np.array([0.3, 0.8, 0.5]))
            axis /= np.linalg.norm(axis)
            axes.append(axis)
            q = rodrigues_expm(angles[i] * axis) @ q_d
            link_attitudes.append(q)
            expected_xi.append(np.sin(angles[i]) * axis)
            omega_d = np.asarray(ref.link_angular_velocities[i], dtype=float)
            link_rates.append(omega_d - np.dot(omega_d, q) * q)
        orientations, body_rates = [], []
        for k in range(nQ):
            R_d = np.asarray(ref.orientations[k], dtype=float)
            axis = np.array([0.6, -0.48, 0.64])
            attitude_axes.append(axis)
            R = R_d @ rodrigues_expm(attitude_angles[k] * axis)
            orientations.append(R)
            expected_eta.append(np.sin(attitude_angles[k]) * axis)
            body_rates.append(R.T @ R_d @ np.asarray(ref.angular_velocities[k], dtype=float))

        model.reset(
            root_position=np.asarray(ref.root_position, dtype=float),
            root_velocity=np.asarray(ref.root_velocity, dtype=float),
            link_attitudes=link_attitudes,
            link_angular_velocities=link_rates,
            orientations=orientations,
            angular_velocities=body_rates,
        )
        error = ctrl.error_state(model.state, ref)
        for i in range(n):
            np.testing.assert_allclose(
                error[6 + 3 * i : 9 + 3 * i], expected_xi[i], atol=TOL_ANALYTIC
            )
            np.testing.assert_allclose(
                error[6 + 3 * n + 3 * i : 9 + 3 * n + 3 * i], 0.0, atol=TOL_ANALYTIC
            )
        for k in range(nQ):
            np.testing.assert_allclose(
                error[base + 3 * k : base + 3 * k + 3], expected_eta[k], atol=TOL_ANALYTIC
            )
            np.testing.assert_allclose(
                error[base + 3 * nQ + 3 * k : base + 3 * nQ + 3 * k + 3], 0.0, atol=TOL_ANALYTIC
            )

        # Second perturbation: on the reference configuration, off its rates.
        link_offsets, body_offsets = [], []
        rates, rotations = [], []
        for i in range(n):
            q_d = np.asarray(ref.link_attitudes[i], dtype=float)
            offset = np.cross(q_d, np.array([0.2, -0.7, 0.4])) * 0.35
            link_offsets.append(offset)
            rates.append(np.asarray(ref.link_angular_velocities[i], dtype=float) + offset)
        for k in range(nQ):
            offset = np.array([0.11, -0.23, 0.17]) * (k + 1)
            body_offsets.append(offset)
            rotations.append(np.asarray(ref.angular_velocities[k], dtype=float) + offset)
        model.reset(
            root_position=np.asarray(ref.root_position, dtype=float),
            root_velocity=np.asarray(ref.root_velocity, dtype=float),
            link_attitudes=[np.asarray(q, dtype=float) for q in ref.link_attitudes],
            link_angular_velocities=rates,
            orientations=[np.asarray(R, dtype=float) for R in ref.orientations],
            angular_velocities=rotations,
        )
        error = ctrl.error_state(model.state, ref)
        np.testing.assert_allclose(error[6 : 6 + 3 * n], 0.0, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(error[base : base + 3 * nQ], 0.0, atol=TOL_ANALYTIC)
        for i in range(n):
            np.testing.assert_allclose(
                error[6 + 3 * n + 3 * i : 9 + 3 * n + 3 * i], link_offsets[i], atol=TOL_ANALYTIC
            )
        for k in range(nQ):
            np.testing.assert_allclose(
                error[base + 3 * nQ + 3 * k : base + 3 * nQ + 3 * k + 3],
                body_offsets[k],
                atol=TOL_ANALYTIC,
            )

    def test_rejects_a_mismatched_state_or_reference(self):
        _, ctrl = build(3, (0, 3))
        other_model, other = build(4, (0, 4))
        ref, _ = ctrl.equilibrium_reference(0.0)
        with pytest.raises(ValueError, match="state has 4 links"):
            ctrl.error_state(other_model.state, ref)
        other_ref, _ = other.equilibrium_reference(0.0)
        model, _ = build(3, (0, 3))
        with pytest.raises(ValueError, match="reference has 4 links"):
            ctrl.error_state(model.state, other_ref)

    def test_rejects_a_mismatched_vehicle_count(self):
        _, ctrl = build(4, (0, 4))
        model, other = build(4, (0, 2, 4))
        ref, _ = ctrl.equilibrium_reference(0.0)
        with pytest.raises(ValueError, match="state has 3 vehicles"):
            ctrl.error_state(model.state, ref)


# ── Flat output plumbing ───────────────────────────────────────────────


class TestFlatOutput:
    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_build_flats_reaches_every_required_depth(self, n, attachments):
        _, ctrl = build(n, attachments)
        traj = gentle_trajectory()
        flats = ctrl.build_flats(traj.get_jet(0.3, ctrl.jet_order))
        assert flats.x_0.order == ctrl.jet_order
        assert len(flats.tensions) == ctrl.num_quadrotors - 1
        for jet, order in zip(flats.tensions, ctrl.flat_map.tension_jet_orders, strict=True):
            assert jet.order >= order
        assert len(flats.psi) == ctrl.num_quadrotors
        # And the map accepts it.
        ctrl.flat_map.recover(flats)

    def test_a_shallow_jet_is_a_programming_error(self):
        _, ctrl = build(3, (0, 3))
        shallow = np.zeros((7, 3))
        shallow[0] = [0.0, 0.0, 1.0]
        with pytest.raises(ValueError, match="needs x_0 through its 10-th derivative"):
            ctrl.flat_map.recover(ctrl.build_flats(shallow))

    def test_trim_tensions_are_carried_into_the_flat_output(self):
        _, ctrl = build(4, (0, 2, 4))
        rows = np.zeros((ctrl.jet_order + 1, 3))
        rows[0] = [0.0, 0.0, 1.0]
        flats = ctrl.build_flats(rows)
        for k, jet in enumerate(flats.tensions):
            np.testing.assert_allclose(jet[0], ctrl.trim_tensions[k], atol=TOL_EXACT)
            np.testing.assert_allclose(jet.data[1:], 0.0, atol=TOL_EXACT)


# ── Closed loop ────────────────────────────────────────────────────────


def track(n, attachments, *, feedforward, trajectory=None, tf=6.0, offset=None, **kwargs):
    """Run ``simulate`` and return ``(model, tail RMS root error, final error, drops)``.

    The recorder wraps the controller's ``compute`` rather than replacing the
    loop, so every sample comes out of :meth:`simulate` itself — the branch M3
    and M5 both left dead by hand-rolling the loop in their tests.

    With a trajectory the run **starts on the reference**, using the flat map's
    own state at ``t = 0``.  Starting at the spawn shape instead measures the
    initial hose transient, which decays at the slowest closed-loop pole
    (~1.3 s) and would swamp the tracking error the comparison is about.  With
    no trajectory the run starts at the spawn shape on purpose: that *is* the
    step response.
    """
    model = MultiQuadHoseBase(num_links=n, attachments=attachments, input="wrench", **kwargs)
    ctrl = model.hose_controller
    start = {"root_position": np.array([0.0, 0.0, 1.0])}
    if trajectory is None:
        target = start["root_position"] + (np.array([0.3, 0.2, 0.25]) if offset is None else offset)

        def reference(t):
            return target, np.zeros(3), np.zeros(3)

        ctrl.setpoint = reference
    else:
        # A bound method exposes ``get_jet`` through ``__self__``; a lambda
        # hides it, which is exactly the two-sided switch under test.
        ctrl.setpoint = trajectory.get if feedforward else (lambda t: trajectory.get(t))
        ref, _ = ctrl.flat_map.recover(ctrl.build_flats(trajectory.get_jet(0.0, ctrl.jet_order)))
        start = {
            "root_position": np.asarray(ref.root_position, dtype=float),
            "root_velocity": np.asarray(ref.root_velocity, dtype=float),
            "link_attitudes": [np.asarray(q, dtype=float) for q in ref.link_attitudes],
            "link_angular_velocities": [
                np.asarray(w, dtype=float) for w in ref.link_angular_velocities
            ],
            "orientations": [np.asarray(R, dtype=float) for R in ref.orientations],
            "angular_velocities": [np.asarray(w, dtype=float) for w in ref.angular_velocities],
        }

    samples: list[float] = []
    inner = ctrl.compute

    def recording(*args):
        if args[0] >= 0.5 * tf:
            desired = ctrl.setpoint(args[0])[0]
            samples.append(float(np.linalg.norm(np.asarray(model.state.root_position) - desired)))
        return inner(*args)

    ctrl.compute = recording
    model.simulate(tf=tf, **start)
    final = float(np.linalg.norm(np.asarray(model.state.root_position) - ctrl.setpoint(model.t)[0]))
    return model, float(np.sqrt(np.mean(np.square(samples)))), final, model.feedforward_drops


@pytest.mark.slow
@pytest.mark.integration
class TestTracking:
    """Closed-loop runs of 6-8 s each; the slowest class in the M6 suite.

    Marked ``slow``/``integration`` so ``-m 'not slow'`` is a usable inner
    loop. CI still runs the whole suite — the marker deselects nothing by
    default.
    """

    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_step_response_settles(self, n, attachments):
        """A 0.44 m step on the root, from the spawn shape, against the trim.

        Scored on the *final* error rather than a tail RMS: the slowest
        closed-loop pole is around -0.75, so a tail window opened halfway
        through an 8 s run is still measuring the transient.
        """
        model, _, final, drops = track(n, attachments, feedforward=False, tf=8.0)
        assert final < TOL_PHYSICAL
        assert drops == 0
        # The hose ends in the trim's shape, not merely somewhere.
        ref, _ = model.hose_controller.equilibrium_reference(model.t)
        for i in range(n):
            cosine = float(
                np.dot(np.asarray(model.state.link_attitudes[i]), np.asarray(ref.link_attitudes[i]))
            )
            assert cosine > 0.999

    @pytest.mark.parametrize(("n", "attachments"), CONFIGS)
    def test_feedforward_improves_tracking_through_simulate(self, n, attachments):
        """Two-sided, and through :meth:`simulate`.

        The only difference between the runs is whether the bound setpoint is
        the trajectory's own method — which exposes ``get_jet`` — or a lambda
        that hides it.  Everything else, the initial condition and the gain
        included, is identical.
        """
        traj = gentle_trajectory(center=np.array([0.0, 0.0, 1.5]))
        _, without, _, drops_without = track(
            n, attachments, feedforward=False, trajectory=traj, tf=6.0
        )
        _, with_ff, _, drops_with = track(n, attachments, feedforward=True, trajectory=traj, tf=6.0)
        assert drops_without == 0
        assert drops_with == 0
        assert with_ff < 0.2 * without
        assert with_ff < TOL_PHYSICAL

    def test_heterogeneous_hose_tracks(self):
        _, rms, _, drops = track(
            4,
            (0, 2, 4),
            feedforward=True,
            trajectory=gentle_trajectory(center=np.array([0.0, 0.0, 1.5]), ax=0.2, f1=0.2),
            tf=6.0,
            link_lengths=[0.4, 0.6, 0.35, 0.55],
            vertex_masses=[0.05, 0.12, 0.03, 0.09, 0.06],
            masses=[0.8, 1.3, 0.95],
        )
        assert drops == 0
        assert rms < TOL_PHYSICAL

    @pytest.mark.slow
    def test_scheduling_the_gain_is_available_and_changes_the_loop(self):
        """The hook the constant-gain choice leaves open.

        Not automatic — a Riccati solve is milliseconds against a 5 ms control
        period — and not a rescue at the aggressive end of the drift table, but
        it is two lines and it works.
        """
        model = MultiQuadHoseBase(num_links=3, attachments=(0, 3), input="wrench")
        ctrl = model.hose_controller
        traj = gentle_trajectory(center=np.array([0.0, 0.0, 1.5]), ax=0.4, f1=0.3)
        ctrl.setpoint = traj.get
        inner = ctrl.compute
        gains = []

        def scheduling(*args):
            if len(args) > 2 and args[2] is not None:
                ref, inputs = args[2]
                ctrl.set_gain_from(*ctrl.linearize(ref, inputs), ref.link_attitudes)
                gains.append(np.linalg.norm(ctrl.gain))
            return inner(*args)

        ctrl.compute = scheduling
        model.simulate(tf=1.0, root_position=traj.get(0.0)[0].copy())
        assert len(gains) > 10
        assert np.std(gains) > 0.0


# ── Model wiring ───────────────────────────────────────────────────────


class TestModelWiring:
    def test_wrench_pipeline_drives_the_lqr(self):
        model = MultiQuadHoseBase(num_links=3, attachments=(0, 3), input="wrench")
        assert model.hose_controller is not None
        wrench = model.compute_hose_input()
        assert wrench.shape == (2, 4)

    def test_acceleration_pipeline_still_station_keeps(self):
        # The default pipeline is untouched: it emits thrust *vectors*.
        model = MultiQuadHoseBase(num_links=3, attachments=(0, 3))
        model.reset(root_position=np.array([0.0, 0.0, 2.0]))
        assert model.compute_default_input().shape == (2, 3)
        model.simulate(tf=0.2, root_position=np.array([0.0, 0.0, 2.0]))
        assert model.t >= 0.2

    def test_no_controller_when_the_last_vertex_is_free(self):
        model = MultiQuadHoseBase(num_links=4, attachments=(0, 2))
        assert model.hose_controller is None
        with pytest.raises(RuntimeError, match="no vehicle at its last vertex"):
            model.compute_hose_input()

    def test_simulate_refuses_the_wrench_pipeline_without_a_controller(self):
        model = MultiQuadHoseBase(num_links=4, attachments=(0, 2), input="wrench")
        with pytest.raises(RuntimeError, match="no vehicle at its last vertex"):
            model.simulate(tf=0.1)

    def test_feedforward_needs_a_bound_trajectory(self):
        model = MultiQuadHoseBase(num_links=2, attachments=(0, 2), input="wrench")
        target = np.array([0.0, 0.0, 1.0])
        model.hose_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        assert model._flatness_feedforward_fn() == (None, False)

    def test_get_full_only_trajectory_is_zero_padded_and_warned(self, caplog):
        class SnapOnly:
            def __init__(self, inner):
                self._inner = inner

            def get(self, t):
                return self._inner.get(t)

            def get_full(self, t):
                position, velocity, acceleration = self._inner.get(t)
                return position, velocity, acceleration, np.zeros(3), np.zeros(3)

        model = MultiQuadHoseBase(num_links=2, attachments=(0, 2), input="wrench")
        traj = SnapOnly(gentle_trajectory(center=np.array([0.0, 0.0, 1.5])))
        model.hose_controller.setpoint = traj.get
        jet_fn, exact = model._flatness_feedforward_fn()
        assert not exact
        rows = jet_fn(0.3, model.hose_controller.jet_order)
        assert rows.shape == (model.hose_controller.jet_order + 1, 3)
        np.testing.assert_allclose(rows[5:], 0.0, atol=TOL_EXACT)
        with caplog.at_level("WARNING"):
            model.simulate(tf=0.1, root_position=traj.get(0.0)[0].copy())
        assert "derivatives only through snap" in caplog.text

    def test_feedforward_drops_are_counted_and_warned_once(self, caplog):
        """A reference the map refuses is a fallback, not an abort."""
        model = MultiQuadHoseBase(num_links=2, attachments=(0, 2), input="wrench")
        ctrl = model.hose_controller
        # A tension the map will refuse as slack partway through: the flat
        # tension is handed over as a vector, so setting it to (almost) nothing
        # leaves the link with no defined direction.
        ctrl.trim_tensions = np.zeros((1, 3))
        target = np.array([0.0, 0.0, 1.5])
        traj = gentle_trajectory(center=target)
        ctrl.setpoint = traj.get
        with caplog.at_level("WARNING"), pytest.raises(ValueError):
            # Feedback-only cannot build its own reference either, so this
            # surfaces rather than silently regulating against nothing.
            model.simulate(tf=0.1, root_position=target.copy())
        assert model.feedforward_drops >= 1
        assert "flatness feedforward dropped" in caplog.text

    def test_replacing_the_controller_is_supported(self):
        model = MultiQuadHoseBase(num_links=2, attachments=(0, 2), input="wrench")
        replacement = MultiQuadHoseController.from_model(model, q_root_position=5.0)
        model.hose_controller = replacement
        assert model.hose_controller is replacement
        assert model.compute_hose_input().shape == (2, 4)


# ── Wrench shape ───────────────────────────────────────────────────────


class TestCompute:
    def test_returns_a_per_vehicle_wrench_stack(self):
        model, ctrl = build(4, (0, 2, 4))
        model.reset(root_position=np.array([0.0, 0.0, 1.0]))
        wrench = ctrl.compute(0.0, model.state)
        assert wrench.shape == (3, 4)
        assert np.all(np.isfinite(wrench))

    def test_supplied_reference_replaces_the_trim(self):
        model, ctrl = build(3, (0, 3))
        model.reset(root_position=np.array([0.0, 0.0, 1.0]))
        traj = gentle_trajectory(center=np.array([0.0, 0.0, 1.0]))
        reference = ctrl.flat_map.recover(ctrl.build_flats(traj.get_jet(0.9, ctrl.jet_order)))
        assert not np.allclose(
            ctrl.compute(0.9, model.state), ctrl.compute(0.9, model.state, reference)
        )

    def test_at_the_reference_the_command_is_the_feedforward(self):
        model, ctrl = build(3, (0, 3))
        ref, inputs = ctrl.equilibrium_reference(0.0)
        model.reset(
            root_position=ref.root_position,
            link_attitudes=ref.link_attitudes,
            orientations=[np.asarray(R) for R in ref.orientations],
        )
        wrench = ctrl.compute(0.0, model.state)
        np.testing.assert_allclose(wrench[:, 0], inputs.thrusts, atol=TOL_INTEGRATION)
        np.testing.assert_allclose(wrench[:, 1:], inputs.moments, atol=TOL_INTEGRATION)

    def test_jet_convention_is_unscaled_rows(self):
        # Guards the one thing a caller building flats by hand gets wrong.
        _, ctrl = build(2, (0, 2))
        rows = np.arange(3 * (ctrl.jet_order + 1), dtype=float).reshape(-1, 3)
        flats = ctrl.build_flats(rows)
        assert isinstance(flats.x_0, Jet)
        np.testing.assert_allclose(flats.x_0.data, rows, atol=TOL_EXACT)
