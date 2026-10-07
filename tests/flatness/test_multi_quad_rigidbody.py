"""Unit tests for the M4 wrench→tension allocation primitive
(:mod:`udaan.flatness.allocation`) and the ``nQ``-quadrotor + cable-suspended
**rigid-body** payload flatness map (:class:`udaan.flatness.MultiQuadRigidbody`).

The open-loop round trip lives next door in
``test_multi_quad_rigidbody_feedforward.py``; this file pins the pieces the
round trip is built from — the null-space construction, the allocation
identity ``Φ τ = W`` at every derivative order, the tautness certificate, the
rotation jet, and the agreement of the recovered reference with the Step-6
model's own dynamics.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_TIGHT
from udaan.core.defaults import (
    DEFAULT_RIGID_PAYLOAD_MASS,
    GRAVITY,
)
from udaan.flatness import (
    FixedTensionDistribution,
    Jet,
    MultiQuadRigidbody,
    MultiQuadRigidbodyFlats,
    RingTensionDistribution,
    WrenchAllocation,
    WrenchAllocationError,
    null_space_basis,
    rotation_jet,
)
from udaan.manif import hat
from udaan.models.multi_quad_rigidbody.base import MultiQuadRigidbodyBase
from udaan.utils.grasp import grasp_map, ring_attachment_points

_E3 = np.array([0.0, 0.0, 1.0])


# ─── Helpers ──────────────────────────────────────────────────────────


def _max_abs(array) -> float:
    """``max |·|`` that survives an empty array (a trivial kernel at nQ = 1)."""
    return float(np.max(np.abs(array), initial=0.0))


def _hover_flats(nQ: int) -> MultiQuadRigidbodyFlats:
    """Level hover: payload pose, every rate and every yaw identically zero."""
    return MultiQuadRigidbodyFlats(
        x_L=Jet.zeros(order=6, dim=3),
        R_L=np.eye(3),
        Omega_L=Jet.zeros(order=5, dim=3),
        psi=[Jet.zeros(order=2, dim=1) for _ in range(nQ)],
    )


def _flats(nQ: int, accel=None, ang_accel=None, R_L=None) -> MultiQuadRigidbodyFlats:
    """Flats with a single non-zero rung, for probing one channel at a time."""
    rows = np.zeros((7, 3))
    if accel is not None:
        rows[2] = np.asarray(accel, dtype=float)
    omega = np.zeros((6, 3))
    if ang_accel is not None:
        omega[1] = np.asarray(ang_accel, dtype=float)
    return MultiQuadRigidbodyFlats(
        x_L=Jet(rows),
        R_L=np.eye(3) if R_L is None else R_L,
        Omega_L=Jet(omega),
        psi=[Jet.zeros(order=2, dim=1) for _ in range(nQ)],
    )


def _random_geometry(rng, nQ: int) -> np.ndarray:
    """Attachment points scattered off any plane, so nothing is accidentally degenerate."""
    return rng.uniform(-0.6, 0.6, size=(nQ, 3))


def _collinear_geometry(nQ: int) -> np.ndarray:
    """Attachments strung along one line — a rank-deficient grasp by construction."""
    return np.column_stack([np.linspace(-0.5, 0.5, nQ), np.zeros(nQ), np.full(nQ, 0.05)])


def _near_collinear_geometry(nQ: int, epsilon: float) -> np.ndarray:
    """:func:`_collinear_geometry` with one attachment nudged ``epsilon`` off the line.

    Sweeping ``epsilon`` walks ``σ_min/σ_max`` continuously from healthy to zero
    while nothing else about the geometry changes, which makes it the probe for
    the module's singular-value cutoff: the only thing that moves is which side
    of the tolerance the smallest singular value lands on.
    """
    points = _collinear_geometry(nQ)
    points[1, 1] += float(epsilon)
    return points


#: An off-ring grasp whose ring hover trim is *infeasible* — the residual is
#: 0.85 N — yet whose recovered hover reference is still an exact equilibrium.
_ASYMMETRIC_ATTACHMENTS = np.array(
    [[0.55, 0.05, 0.05], [-0.15, 0.60, 0.02], [-0.62, -0.20, 0.08], [0.20, -0.55, -0.03]]
)


def _pairwise_null_columns(points: np.ndarray) -> np.ndarray:
    """The published hand-built ``N``: one column per attachment *pair*.

    Ports the construction of
    ``phd_research/quadrotor-flexible-cables-payload/Multiple-Quad-Rigid-Load/
    main_multiquad_rigid_flexible.m`` (lines 33–52), which carries its own
    ``(FIX THIS - write code to automate ...)`` comment: column ``(i, j)`` puts
    ``u_ij = (r_j - r_i)/‖r_j - r_i‖`` on cable ``i`` and ``-u_ij`` on cable
    ``j``.
    """
    n = len(points)
    columns = []
    for i, j in itertools.combinations(range(n), 2):
        u = points[j] - points[i]
        u = u / np.linalg.norm(u)
        column = np.zeros(3 * n)
        column[3 * i : 3 * i + 3] = u
        column[3 * j : 3 * j + 3] = -u
        columns.append(column)
    return np.column_stack(columns)


def _rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


#: The reference's own ``nQ = 4`` attachment geometry (same file, line 28).
_MATLAB_ATTACHMENTS = np.array(
    [
        _rot_z(0.0) @ np.array([1.0, 0.0, 0.0]),
        _rot_z(np.deg2rad(-30.0)) @ np.array([2.0, 0.0, 0.0]),
        _rot_z(np.deg2rad(-165.0)) @ np.array([1.5, 0.0, 0.0]),
        _rot_z(np.deg2rad(45.0)) @ np.array([2.5, 0.0, 0.0]),
    ]
)


def _seed(model: MultiQuadRigidbodyBase, ref) -> None:
    """Place the model exactly at a recovered flat reference."""
    state = ref.as_state()
    model.reset(
        payload_position=np.asarray(state.payload_position).copy(),
        payload_velocity=np.asarray(state.payload_velocity).copy(),
        payload_orientation=np.asarray(state.payload_orientation),
        payload_angular_velocity=np.asarray(state.payload_angular_velocity),
        cable_attitudes=state.cable_attitudes,
        cable_angular_velocities=state.cable_angular_velocities,
        orientations=state.orientations,
        angular_velocities=state.angular_velocities,
    )


# ─── The null-space construction ──────────────────────────────────────


class TestNullSpaceBasis:
    """``null_space_basis`` must be a genuine basis of ``ker Φ``, at every nQ."""

    @pytest.mark.parametrize("nQ", [1, 2, 3, 4, 5, 6, 8])
    def test_ring_geometry_spans_the_kernel(self, nQ):
        Phi = grasp_map(ring_attachment_points(nQ))
        N = null_space_basis(Phi)

        rank = np.linalg.matrix_rank(Phi)
        # At nQ = 1 the grasp is injective and the kernel is trivial — a legal,
        # zero-column answer that the shape check has to survive.
        assert N.shape == (3 * nQ, 3 * nQ - rank)
        # Every column is in the kernel …
        assert _max_abs(Phi @ N) < TOL_TIGHT
        # … the columns are orthonormal …
        assert _max_abs(N.T @ N - np.eye(N.shape[1])) < TOL_TIGHT
        # … and they span it: the kernel projector agrees with N Nᵀ.
        projector = np.eye(3 * nQ) - np.linalg.pinv(Phi) @ Phi
        assert _max_abs(N @ N.T - projector) < TOL_ANALYTIC

    @pytest.mark.parametrize("nQ", [3, 4, 5, 7])
    def test_random_geometry_spans_the_kernel(self, rng, nQ):
        Phi = grasp_map(_random_geometry(rng, nQ))
        N = null_space_basis(Phi)
        assert N.shape[1] == 3 * nQ - 6
        assert np.max(np.abs(Phi @ N)) < TOL_TIGHT
        # Any vector the grasp cannot see is reproduced exactly by the basis.
        for _ in range(5):
            v = (np.eye(3 * nQ) - np.linalg.pinv(Phi) @ Phi) @ rng.normal(size=3 * nQ)
            assert np.linalg.norm(N @ (N.T @ v) - v) < TOL_ANALYTIC

    @pytest.mark.parametrize("nQ", [3, 4, 5])
    def test_collinear_attachments_enlarge_the_kernel(self, nQ):
        """A degenerate grasp is handled, not assumed away.

        Collinear attachments cannot produce a moment about their own axis, so
        ``rank Φ = 5`` and the kernel is ``3nQ - 5``, one dimension *larger*
        than the generic ``3nQ - 6``. The dimension is discovered from the SVD
        rather than hardcoded, which is precisely what the published
        ``3nQ - 6``-shaped ``N`` cannot do.
        """
        Phi = grasp_map(_collinear_geometry(nQ))
        N = null_space_basis(Phi)
        assert np.linalg.matrix_rank(Phi) == 5
        assert N.shape[1] == 3 * nQ - 5
        assert np.max(np.abs(Phi @ N)) < TOL_TIGHT

    def test_trivial_kernel(self):
        """No kernel at all is a legal answer, returned with the right shape."""
        N = null_space_basis(np.eye(4))
        assert N.shape == (4, 0)

    def test_rejects_non_matrix(self):
        with pytest.raises(ValueError, match="2-D array"):
            null_space_basis(np.zeros(6))


class TestPublishedNullSpaceIsNotABasis:
    """The hand-built ``N`` of the reference, measured against ``ker Φ``."""

    def test_pairwise_columns_are_in_the_kernel(self):
        """The recipe's *idea* is sound: each column really is a null vector."""
        Phi = grasp_map(_MATLAB_ATTACHMENTS)
        assert np.max(np.abs(Phi @ _pairwise_null_columns(_MATLAB_ATTACHMENTS))) < TOL_TIGHT

    def test_pairwise_columns_do_not_span_the_kernel_at_nQ_4(self):
        """…but at the reference's own geometry they are rank 5, not 6.

        ``ker Φ`` is 6-dimensional there, and the six hand-picked columns are
        linearly dependent, so one whole direction of the tension null space is
        unreachable by *any* ``LAMBDA``. Our SVD basis reaches all six.
        """
        Phi = grasp_map(_MATLAB_ATTACHMENTS)
        published = _pairwise_null_columns(_MATLAB_ATTACHMENTS)
        assert published.shape[1] == 6
        assert np.linalg.matrix_rank(published) == 5
        assert null_space_basis(Phi).shape[1] == 6

    @pytest.mark.parametrize("nQ", [5, 6, 7])
    def test_pairwise_column_count_is_wrong_beyond_nQ_4(self, nQ):
        """``C(nQ, 2)`` columns for a ``3nQ - 6`` dimensional kernel."""
        points = ring_attachment_points(nQ)
        assert _pairwise_null_columns(points).shape[1] != 3 * nQ - 6
        assert null_space_basis(grasp_map(points)).shape[1] == 3 * nQ - 6


# ─── The allocator ────────────────────────────────────────────────────


class TestWrenchAllocationGeometry:
    @pytest.mark.parametrize(("nQ", "rank"), [(1, 3), (2, 5), (3, 6), (4, 6), (6, 6)])
    def test_ring_grasp_rank(self, nQ, rank):
        alloc = WrenchAllocation(ring_attachment_points(nQ))
        assert alloc.rank == rank
        assert alloc.is_fully_actuated == (rank == 6)
        assert alloc.null_basis.shape == (3 * nQ, 3 * nQ - rank)

    def test_grasp_matches_the_shared_helper(self):
        points = ring_attachment_points(4)
        alloc = WrenchAllocation(points)
        assert np.allclose(alloc.grasp, grasp_map(points), atol=TOL_EXACT)

    def test_rejects_a_policy_of_the_wrong_size(self):
        policy = FixedTensionDistribution(np.tile([0.0, 0.0, 3.0], (3, 1)))
        with pytest.raises(ValueError, match="built for 3 quadrotors"):
            WrenchAllocation(ring_attachment_points(4), policy=policy)

    def test_rejects_bad_attachment_shape(self):
        with pytest.raises(ValueError, match=r"\(nQ, 3\)"):
            WrenchAllocation(np.zeros((4, 2)))


class TestAllocationIdentity:
    """``Φ τ = W`` — at the bias point, at every wrench, at every order."""

    @pytest.mark.parametrize("nQ", [3, 4, 5])
    def test_exact_at_the_reference_wrench(self, nQ):
        alloc = WrenchAllocation(ring_attachment_points(nQ))
        tau = alloc.allocate(alloc.reference_wrench)
        assert np.max(np.abs(tau - alloc.bias_forces)) < TOL_ANALYTIC

    @pytest.mark.parametrize("nQ", [3, 4, 6])
    def test_residual_is_round_off_for_any_wrench(self, rng, nQ):
        alloc = WrenchAllocation(ring_attachment_points(nQ))
        for _ in range(10):
            wrench = alloc.reference_wrench + rng.normal(scale=0.2, size=6)
            tau = alloc.allocate(wrench, check_taut=False)
            assert alloc.residual(tau, wrench) < TOL_ANALYTIC

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_jet_satisfies_the_balance_at_every_order(self, rng, nQ):
        alloc = WrenchAllocation(ring_attachment_points(nQ))
        wrench = [alloc.reference_wrench] + [rng.normal(scale=0.3, size=6) for _ in range(4)]
        per_cable = alloc.allocate_jet(wrench)
        assert len(per_cable) == nQ
        for k in range(5):
            stack = np.concatenate([per_cable[i][k] for i in range(nQ)])
            assert np.linalg.norm(alloc.grasp @ stack - wrench[k]) < TOL_ANALYTIC

    def test_null_coordinates_reproduce_the_null_term(self):
        """``λ = Nᵀ τ̄`` in newtons is a complete replacement for ``LAMBDA``."""
        alloc = WrenchAllocation(ring_attachment_points(4))
        lam = alloc.null_coordinates
        assert lam.shape == (alloc.null_basis.shape[1],)
        # τ(W) = Φ⁺W + N λ, with N λ the projected bias.
        wrench = alloc.reference_wrench
        rebuilt = alloc.grasp_pinv @ wrench + alloc.null_basis @ lam
        assert np.max(np.abs(rebuilt.reshape(4, 3) - alloc.allocate(wrench))) < TOL_ANALYTIC

    def test_zero_bias_collapses_the_formation(self):
        """The min-norm solution alone is the degenerate answer, as documented."""
        alloc = WrenchAllocation(
            ring_attachment_points(4),
            policy=FixedTensionDistribution(np.tile([0.0, 0.0, 1e-9], (4, 1))),
        )
        tau = alloc.allocate(np.concatenate([9.81 * _E3, np.zeros(3)]), check_taut=False)
        directions = tau / np.linalg.norm(tau, axis=1, keepdims=True)
        # Every cable force ends up (nearly) parallel: no formation left.
        assert np.min(directions @ directions[0]) > 1.0 - 1e-6

    def test_rejects_a_malformed_wrench(self):
        alloc = WrenchAllocation(ring_attachment_points(3))
        with pytest.raises(ValueError, match="6-vector"):
            alloc.allocate(np.zeros(5))
        with pytest.raises(ValueError, match="at least the zeroth"):
            alloc.allocate_jet([])


class TestTautnessCertificate:
    @pytest.mark.parametrize("nQ", [3, 4, 5])
    def test_margin_certifies_tautness(self, rng, nQ):
        """Every wrench inside the certified ball keeps every cable taut."""
        alloc = WrenchAllocation(ring_attachment_points(nQ))
        margin = alloc.wrench_margin()
        assert margin > 0.0
        for _ in range(25):
            delta = rng.normal(size=6)
            delta *= 0.99 * margin / np.linalg.norm(delta)
            tau = alloc.allocate(alloc.reference_wrench + delta)  # must not raise
            assert np.min(np.linalg.norm(tau, axis=1)) > alloc.min_tension
            # The sign test cannot fire inside the ball either.
            assert np.min(np.sum(tau * alloc.bias_forces, axis=1)) > 0.0

    def test_slackening_trajectory_raises(self):
        """A payload angular acceleration large enough to invert a cable is refused."""
        model = MultiQuadRigidbodyBase(num_quadrotors=3)
        f2s = MultiQuadRigidbody.from_model(model)
        # 60 rad/s² about x: the moment channel drives cable 2's force through
        # zero and out the far side.
        with pytest.raises(ValueError, match="crossed the slack boundary"):
            f2s.recover(_flats(3, ang_accel=[60.0, 0.0, 0.0]))

    def test_tension_floor_raises_before_the_sign_test(self):
        """Landing exactly on the boundary is caught by the magnitude floor."""
        alloc = WrenchAllocation(
            ring_attachment_points(3),
            policy=FixedTensionDistribution(np.tile([0.0, 0.0, 3.0], (3, 1))),
            min_tension=1e-3,
        )
        tau = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0], [0.0, 0.0, 1e-9]])
        with pytest.raises(ValueError, match="would go slack"):
            alloc.check_taut(tau)

    def test_sign_test_is_two_sided(self):
        """An inverted cable has a healthy *magnitude*; only the sign sees it."""
        alloc = WrenchAllocation(
            ring_attachment_points(3),
            policy=FixedTensionDistribution(np.tile([0.0, 0.0, 3.0], (3, 1))),
        )
        tau = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 3.0], [0.0, 0.0, -3.0]])
        assert np.min(np.linalg.norm(tau, axis=1)) > alloc.min_tension
        with pytest.raises(ValueError, match="crossed the slack boundary"):
            alloc.check_taut(tau)

    def test_margin_uses_the_spectral_norm(self):
        """``σ_max(Φ⁺_i)``, not ``‖Φ⁺_i‖_F``.

        Frobenius over-estimates the gain, so it would yield a *smaller* — still
        sound but needlessly conservative — ball; pin the exact constant so the
        two cannot be swapped silently.
        """
        alloc = WrenchAllocation(ring_attachment_points(4))
        bias = alloc.bias_forces
        blocks = [alloc.grasp_pinv[3 * i : 3 * i + 3, :] for i in range(4)]
        spectral = min(
            (np.linalg.norm(bias[i]) - alloc.min_tension) / np.linalg.norm(b, ord=2)
            for i, b in enumerate(blocks)
        )
        frobenius = min(
            (np.linalg.norm(bias[i]) - alloc.min_tension) / np.linalg.norm(b, ord="fro")
            for i, b in enumerate(blocks)
        )
        assert frobenius < spectral  # the two are genuinely distinguishable here
        assert abs(alloc.wrench_margin() - spectral) < TOL_EXACT


class TestTensionDistributionPolicies:
    def test_ring_bias_is_the_model_hover_trim(self):
        """The default policy reproduces the model's own ``reset()`` formation."""
        model = MultiQuadRigidbodyBase(num_quadrotors=4)
        model.reset()
        policy = RingTensionDistribution(
            num_quadrotors=4,
            payload_mass=model.payload_mass,
            cable_lengths=model.cable_lengths,
            attachment_points=model.attachment_points,
            formation_radius=model.formation_radius,
        )
        assert policy.hover_residual < TOL_ANALYTIC
        for i in range(4):
            expected = -policy.hover_tensions[i] * np.asarray(model.state.cable_attitudes[i])
            assert np.linalg.norm(policy.bias_forces[i] - expected) < TOL_ANALYTIC

    def test_an_infeasible_ring_trim_is_reported_for_what_it_is(self, caplog):
        """Both halves of the warning, pinned.

        An off-ring grasp makes the *ring formation's* trim infeasible, so the
        bias is no longer the model's ``reset()`` formation.  That is the whole
        content of a non-zero ``hover_residual``: ``rank Φ`` is still 6, the
        grasp holds the weight and zeroes the moment without difficulty, and the
        recovered hover reference is still an exact equilibrium of the model.
        The message used to claim the opposite on both counts and sent anyone
        with a non-ring grasp hunting a bug that is not there.
        """
        model = MultiQuadRigidbodyBase(num_quadrotors=4)
        model.attachment_points = _ASYMMETRIC_ATTACHMENTS
        with caplog.at_level("WARNING"):
            f2s = MultiQuadRigidbody.from_model(model)
        assert f2s.tension_policy.hover_residual > 0.1
        assert "no longer coincides with the model's reset() formation" in caplog.text
        assert "still an exact equilibrium" in caplog.text
        assert f2s.allocation.rank == 6

        # It really is an equilibrium: seed the model at the recovered hover
        # reference, apply the recovered feedforward, and nothing accelerates.
        ref, inputs = f2s.recover(_hover_flats(4))
        _seed(model, ref)
        accel = model.accelerations(
            thrusts=np.asarray(inputs.thrusts), torques=np.asarray(inputs.moments)
        )
        assert float(np.linalg.norm(accel["payload_accel"])) < TOL_TIGHT
        assert float(np.linalg.norm(accel["payload_ang_accel"])) < TOL_TIGHT
        assert _max_abs(accel["cable_ang_accels"]) < TOL_TIGHT

        # What *is* lost, and all that is lost: reset() places the team somewhere
        # else, because it flies the ring the infeasible trim was solved from.
        model.reset()
        mismatch = max(
            float(np.linalg.norm(np.asarray(model.state.positions[i]) - ref.positions[i]))
            for i in range(4)
        )
        assert mismatch > 0.2

    @pytest.mark.parametrize("nQ", [1, 2, 3, 4, 5])
    def test_ring_bias_produces_the_hover_wrench(self, nQ):
        alloc = WrenchAllocation(ring_attachment_points(nQ))
        expected = np.concatenate([DEFAULT_RIGID_PAYLOAD_MASS * GRAVITY * _E3, np.zeros(3)])
        assert np.linalg.norm(alloc.reference_wrench - expected) < TOL_ANALYTIC

    def test_fixed_policy_rejects_a_zero_bias(self):
        with pytest.raises(ValueError, match="non-zero"):
            FixedTensionDistribution(np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))

    def test_fixed_policy_rejects_a_bad_shape(self):
        with pytest.raises(ValueError, match=r"\(nQ, 3\)"):
            FixedTensionDistribution(np.zeros(3))

    def test_ring_policy_validates_its_arguments(self):
        with pytest.raises(ValueError, match="num_quadrotors"):
            RingTensionDistribution(num_quadrotors=0)
        with pytest.raises(ValueError, match="payload_mass"):
            RingTensionDistribution(num_quadrotors=3, payload_mass=0.0)
        with pytest.raises(ValueError, match="cable_lengths"):
            RingTensionDistribution(num_quadrotors=3, cable_lengths=[1.0, 1.0])


# ─── The rotation jet ─────────────────────────────────────────────────


class TestRotationJet:
    def test_matches_a_fixed_axis_closed_form(self):
        """For ``R(t) = exp(n̂ ω t)``, ``R^(k) = R n̂^k ω^k`` in closed form."""
        axis = np.array([0.3, -0.5, 0.8])
        axis /= np.linalg.norm(axis)
        rate = 1.7
        R0 = _rot_x(0.4) @ _rot_z(-0.9)
        omega = [axis * rate] + [np.zeros(3)] * 5

        jet = rotation_jet(R0, omega, order=4)
        n_hat = hat(axis) * rate
        power = np.eye(3)
        for k in range(5):
            assert np.max(np.abs(jet[k] - R0 @ power)) < TOL_ANALYTIC
            power = power @ n_hat

    def test_matches_finite_differences(self):
        """A genuinely non-fixed-axis motion, checked numerically."""

        def pose(t):
            phi = 0.25 * np.sin(1.3 * t)
            return _rot_x(phi) @ _rot_z(0.7 * t)

        def omega_jet(t, order=5):
            # Ω = φ̇ [cos ψ, -sin ψ, 0] + ψ̇ e₃ for R = Rx(φ) Rz(ψ).
            dphi = [0.25 * 1.3 ** (k + 1) * np.sin(1.3 * t + (k + 1) * np.pi / 2) for k in range(6)]
            cos = [0.7**k * np.cos(0.7 * t + k * np.pi / 2) for k in range(6)]
            sin = [0.7**k * np.sin(0.7 * t + k * np.pi / 2) for k in range(6)]
            from udaan.flatness import leibniz_product

            ox = leibniz_product(dphi, cos)
            oy = [-v for v in leibniz_product(dphi, sin)]
            return [np.array([ox[k], oy[k], 0.7 if k == 0 else 0.0]) for k in range(order + 1)]

        t, h = 0.37, 1e-5
        jet = rotation_jet(pose(t), omega_jet(t), order=3)
        for k in range(3):
            numeric = (
                np.asarray(rotation_jet(pose(t + h), omega_jet(t + h), order=3)[k])
                - np.asarray(rotation_jet(pose(t - h), omega_jet(t - h), order=3)[k])
            ) / (2.0 * h)
            assert np.max(np.abs(numeric - jet[k + 1])) < 1e-6

    def test_needs_enough_angular_velocity_levels(self):
        with pytest.raises(ValueError, match="through derivative"):
            rotation_jet(np.eye(3), [np.zeros(3)], order=3)

    def test_rejects_negative_order(self):
        with pytest.raises(ValueError, match="order must be >= 0"):
            rotation_jet(np.eye(3), [], order=-1)


# ─── Flat-output validation ───────────────────────────────────────────


class TestFlatsValidation:
    def test_requires_position_through_pop(self):
        with pytest.raises(ValueError, match="x_L.order must be >= 6"):
            MultiQuadRigidbodyFlats(
                x_L=Jet.zeros(order=5, dim=3),
                R_L=np.eye(3),
                Omega_L=Jet.zeros(order=5, dim=3),
                psi=[Jet.zeros(order=2, dim=1)],
            )

    def test_requires_body_rate_through_the_fifth_derivative(self):
        """The rung M3 has no analogue of — ``W_m^(4)`` needs ``Ω_L^(5)``."""
        with pytest.raises(ValueError, match="Omega_L.order must be >= 5"):
            MultiQuadRigidbodyFlats(
                x_L=Jet.zeros(order=6, dim=3),
                R_L=np.eye(3),
                Omega_L=Jet.zeros(order=4, dim=3),
                psi=[Jet.zeros(order=2, dim=1)],
            )

    def test_requires_a_rotation_matrix(self):
        with pytest.raises(ValueError, match="orthonormal"):
            MultiQuadRigidbodyFlats(
                x_L=Jet.zeros(order=6, dim=3),
                R_L=np.diag([1.0, 1.0, 2.0]),
                Omega_L=Jet.zeros(order=5, dim=3),
                psi=[Jet.zeros(order=2, dim=1)],
            )

    def test_rejects_a_reflection(self):
        with pytest.raises(ValueError, match=r"det = \+1"):
            MultiQuadRigidbodyFlats(
                x_L=Jet.zeros(order=6, dim=3),
                R_L=np.diag([1.0, 1.0, -1.0]),
                Omega_L=Jet.zeros(order=5, dim=3),
                psi=[Jet.zeros(order=2, dim=1)],
            )

    def test_requires_yaw_through_acceleration(self):
        with pytest.raises(ValueError, match=r"psi\[0\].order must be >= 2"):
            MultiQuadRigidbodyFlats(
                x_L=Jet.zeros(order=6, dim=3),
                R_L=np.eye(3),
                Omega_L=Jet.zeros(order=5, dim=3),
                psi=[Jet.zeros(order=1, dim=1)],
            )

    def test_yaw_count_must_match_the_map(self):
        f2s = MultiQuadRigidbody(num_quadrotors=3)
        with pytest.raises(ValueError, match="but the map has 3 quadrotors"):
            f2s.recover(_hover_flats(4))


class TestJetDepth:
    """The declared depth is real: the top rung of each channel reaches the moment."""

    def test_payload_pop_reaches_the_moment(self):
        f2s = MultiQuadRigidbody(num_quadrotors=4)
        base = _hover_flats(4)
        bumped = MultiQuadRigidbodyFlats(
            x_L=Jet(np.vstack([np.zeros((6, 3)), np.array([[1.0, 0.0, 0.0]])])),
            R_L=np.eye(3),
            Omega_L=Jet.zeros(order=5, dim=3),
            psi=[Jet.zeros(order=2, dim=1) for _ in range(4)],
        )
        _, a = f2s.recover(base)
        _, b = f2s.recover(bumped)
        assert np.max(np.abs(np.array(a.moments) - np.array(b.moments))) > TOL_ANALYTIC

    def test_fifth_body_rate_derivative_reaches_the_moment(self):
        """``Ω_L^(5)`` is not decoration: perturbing it moves ``M_i``."""
        f2s = MultiQuadRigidbody(num_quadrotors=4)
        omega = np.zeros((6, 3))
        omega[5] = np.array([1.0, 0.0, 0.0])
        bumped = MultiQuadRigidbodyFlats(
            x_L=Jet.zeros(order=6, dim=3),
            R_L=np.eye(3),
            Omega_L=Jet(omega),
            psi=[Jet.zeros(order=2, dim=1) for _ in range(4)],
        )
        _, a = f2s.recover(_hover_flats(4))
        _, b = f2s.recover(bumped)
        assert np.max(np.abs(np.array(a.moments) - np.array(b.moments))) > TOL_ANALYTIC


# ─── Hover, and agreement with the Step-6 model ───────────────────────


class TestHover:
    @pytest.mark.parametrize("nQ", [1, 2, 3, 4, 5])
    def test_reference_is_the_models_reset_configuration(self, nQ):
        model = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        model.reset()
        f2s = MultiQuadRigidbody.from_model(model)
        ref, _ = f2s.recover(_hover_flats(nQ))
        for i in range(nQ):
            assert (
                np.linalg.norm(
                    np.asarray(ref.cable_attitudes[i]) - np.asarray(model.state.cable_attitudes[i])
                )
                < TOL_ANALYTIC
            )
            assert (
                np.linalg.norm(np.asarray(ref.positions[i]) - np.asarray(model.state.positions[i]))
                < TOL_ANALYTIC
            )

    @pytest.mark.parametrize("nQ", [1, 3, 4])
    def test_hover_wrench_is_weight_and_no_moment(self, nQ):
        f2s = MultiQuadRigidbody(num_quadrotors=nQ)
        wrench = f2s.payload_wrench_jet(_hover_flats(nQ))
        expected = np.concatenate([f2s.payload_mass * GRAVITY * _E3, np.zeros(3)])
        assert np.linalg.norm(wrench[0] - expected) < TOL_ANALYTIC
        for k in range(1, 5):
            assert np.linalg.norm(wrench[k]) < TOL_EXACT

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_hover_is_an_equilibrium_of_the_model(self, nQ):
        model = MultiQuadRigidbodyBase(num_quadrotors=nQ, input="wrench")
        f2s = MultiQuadRigidbody.from_model(model)
        ref, inputs = f2s.recover(_hover_flats(nQ))
        _seed(model, ref)
        accel = model.accelerations(np.array(inputs.thrusts), np.array(inputs.moments))
        assert np.linalg.norm(accel["payload_accel"]) < TOL_ANALYTIC
        assert np.linalg.norm(accel["payload_ang_accel"]) < TOL_ANALYTIC


class TestModelCrossCheck:
    """The recovered reference solves the Step-6 model's own equations."""

    @staticmethod
    def _dynamic_flats(nQ: int, t: float = 0.41) -> MultiQuadRigidbodyFlats:
        amps = np.array([0.35, 0.30, 0.20])
        rates = np.array([1.1, 0.7, 0.9])
        phases = np.array([0.0, 0.6, 1.3])
        rows = [amps * rates**k * np.sin(rates * t + phases + k * np.pi / 2) for k in range(7)]
        omega = np.zeros((6, 3))
        omega[0] = np.array([0.12, -0.08, 0.30])
        omega[1] = np.array([-0.20, 0.15, 0.05])
        return MultiQuadRigidbodyFlats(
            x_L=Jet(np.stack(rows)),
            R_L=_rot_x(0.12) @ _rot_z(0.35),
            Omega_L=Jet(omega),
            psi=[Jet(np.array([0.3 * i, 0.4, 0.0])) for i in range(nQ)],
        )

    @pytest.mark.parametrize("nQ", [3, 4, 5])
    def test_cable_tensions_agree_with_the_model(self, nQ):
        """``𝐓_i = -R_L τ_i`` matches what the model's 6×6 solve produces.

        The allocation and ``MultiQuadRigidbodyBase.cable_tensions`` share no
        code: one inverts the grasp map, the other eliminates the tensions out
        of the coupled system. Agreement is evidence, not tautology.
        """
        model = MultiQuadRigidbodyBase(num_quadrotors=nQ, input="wrench")
        f2s = MultiQuadRigidbody.from_model(model)
        flats = self._dynamic_flats(nQ)
        ref, inputs = f2s.recover(flats)
        _seed(model, ref)

        R_L = np.asarray(ref.payload_orientation)
        from_flat = np.array([-(R_L @ inputs.cable_forces[i]) for i in range(nQ)])
        from_model = model.cable_tensions(np.array(inputs.thrusts))
        assert np.max(np.abs(from_flat - from_model)) < TOL_ANALYTIC

    @pytest.mark.parametrize("nQ", [3, 4, 5])
    def test_accelerations_agree_with_the_flat_output(self, nQ):
        model = MultiQuadRigidbodyBase(num_quadrotors=nQ, input="wrench")
        f2s = MultiQuadRigidbody.from_model(model)
        flats = self._dynamic_flats(nQ)
        ref, inputs = f2s.recover(flats)
        _seed(model, ref)

        accel = model.accelerations(np.array(inputs.thrusts), np.array(inputs.moments))
        assert np.linalg.norm(accel["payload_accel"] - np.asarray(flats.x_L[2])) < TOL_ANALYTIC
        assert (
            np.linalg.norm(accel["payload_ang_accel"] - np.asarray(flats.Omega_L[1])) < TOL_ANALYTIC
        )
        assert model.compressive_cable_steps == 0

    @pytest.mark.parametrize("nQ", [3, 4])
    def test_allocation_residual_is_round_off(self, nQ):
        model = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        f2s = MultiQuadRigidbody.from_model(model)
        flats = self._dynamic_flats(nQ)
        wrench = f2s.payload_wrench_jet(flats)
        per_cable = f2s.allocation.allocate_jet(wrench)
        for k in range(len(wrench)):
            stack = np.concatenate([per_cable[i][k] for i in range(nQ)])
            assert np.linalg.norm(f2s.allocation.grasp @ stack - wrench[k]) < TOL_ANALYTIC


# ─── Under-actuated geometries ────────────────────────────────────────


class TestLowRankGeometries:
    """At ``nQ ≤ 2`` the payload attitude is simply not a flat output."""

    @pytest.mark.parametrize("nQ", [1, 2])
    def test_hover_still_recovers(self, nQ):
        """The hover wrench happens to lie inside ``range Φ``, so it works."""
        f2s = MultiQuadRigidbody(num_quadrotors=nQ)
        ref, inputs = f2s.recover(_hover_flats(nQ))
        assert len(inputs.thrusts) == nQ
        assert np.min(inputs.tensions) > 0.0
        assert ref.num_quadrotors == nQ

    @pytest.mark.parametrize("nQ", [1, 2])
    def test_lateral_acceleration_is_refused(self, nQ):
        """A level payload cannot be accelerated sideways by an under-actuated
        grasp without also tilting, and the map says so instead of
        least-squaring the command away.

        Which lateral direction fails is geometry, not accident: the ring's
        co-kernel at ``nQ = 2`` is ``W_m,x + 0.05 W_f,y = 0``, so it is a
        *y*-acceleration that leaves the range — the two attachments lie along
        *x*, and a moment about that line is the one wrench the pair cannot
        produce. At ``nQ = 1`` every lateral direction fails.
        """
        f2s = MultiQuadRigidbody(num_quadrotors=nQ)
        with pytest.raises(WrenchAllocationError, match="outside the range"):
            f2s.recover(_flats(nQ, accel=[0.0, 1.5, 0.0]))

    @pytest.mark.parametrize("nQ", [1, 2])
    def test_out_of_range_wrench_is_refused_at_every_order(self, nQ):
        """Built straight from the co-kernel, so it cannot possibly be reachable."""
        alloc = WrenchAllocation(ring_attachment_points(nQ))
        unreachable = null_space_basis(alloc.grasp.T)[:, 0]
        with pytest.raises(WrenchAllocationError, match="outside the range"):
            alloc.allocate(alloc.reference_wrench + unreachable, check_taut=False)
        with pytest.raises(WrenchAllocationError, match="order 2"):
            alloc.allocate_jet(
                [alloc.reference_wrench, np.zeros(6), unreachable, np.zeros(6), np.zeros(6)]
            )

    @pytest.mark.parametrize("nQ", [1, 2])
    def test_payload_attitude_manoeuvre_is_refused(self, nQ):
        f2s = MultiQuadRigidbody(num_quadrotors=nQ)
        with pytest.raises(WrenchAllocationError, match="rank"):
            f2s.recover(_flats(nQ, ang_accel=[0.4, 0.0, 0.0]))

    def test_single_quadrotor_at_the_centre_of_mass_reduces_to_M3(self):
        """With ``r = 0`` the moment channel vanishes and translation is free."""
        f2s = MultiQuadRigidbody(num_quadrotors=1, attachment_points=np.zeros((1, 3)))
        ref, inputs = f2s.recover(_flats(1, accel=[1.5, 0.0, 0.0]))
        assert np.linalg.norm(np.asarray(ref.payload_acceleration) - [1.5, 0.0, 0.0]) < TOL_EXACT
        # A lone cable carries the whole net force: 𝐓 = -m_L(a_L + g e₃).
        expected = f2s.payload_mass * np.linalg.norm(np.array([1.5, 0.0, GRAVITY]))
        assert abs(inputs.tensions[0] - expected) < TOL_ANALYTIC

    def test_collinear_attachments_are_reported(self, caplog):
        with caplog.at_level("WARNING"):
            alloc = WrenchAllocation(_collinear_geometry(4))
        assert alloc.rank == 5
        assert "rank 5 < 6" in caplog.text


class TestNearDegenerateGrasps:
    """One singular-value cutoff, and a warning when ``Φ`` is only just invertible.

    ``rank``, ``is_fully_actuated`` and the :class:`WrenchAllocationError` text
    all read the cutoff of :func:`null_space_basis`; ``allocate`` reads whatever
    rank ``Φ⁺`` actually inverts.  Left to numpy's own ``rcond`` default the two
    rules differ by two decades, and a grasp landing between them is *reported*
    rank-deficient while ``Φ⁺`` silently inverts the direction the report says
    is missing.
    """

    @pytest.mark.parametrize(
        ("epsilon", "expected_rank"), [(1e-6, 6), (1e-9, 6), (1e-12, 5), (0.0, 5)]
    )
    def test_one_cutoff_decides_both_the_reported_and_the_inverted_rank(
        self, epsilon, expected_rank
    ):
        """``tr(Φ⁺Φ) = rank Φ``, because ``Φ⁺Φ`` is the projector onto ``row Φ``.

        The equality is the whole point: it fails the moment the reported rank
        and the pseudo-inverse's effective rank are decided by different
        tolerances.  The ``ε = 1e-6`` row doubles as the pin on the cutoff's
        *value* — anything looser than ``1e-6`` absorbs a genuinely invertible
        direction into the kernel and reports rank 5 here.
        """
        alloc = WrenchAllocation(_near_collinear_geometry(4, epsilon))
        assert alloc.rank == expected_rank
        assert alloc.is_fully_actuated == (expected_rank == 6)
        assert abs(float(np.trace(alloc.grasp_pinv @ alloc.grasp)) - expected_rank) < 1e-6

    @pytest.mark.parametrize("epsilon", [1e-6, 1e-9])
    def test_a_full_rank_but_near_singular_grasp_is_flagged(self, epsilon, caplog):
        """Full rank is the honest answer; ill conditioning is said out loud."""
        with caplog.at_level("WARNING"):
            alloc = WrenchAllocation(_near_collinear_geometry(4, epsilon))
        assert alloc.rank == 6
        assert alloc.condition_number > 1e6
        assert "near-degenerate" in caplog.text

    # ``ε = 1e-9`` is deliberately absent: at ``cond Φ ≈ 1e9`` the range-check
    # residual is rounding noise straddling ``_RANGE_TOL`` (8.8e-9 locally,
    # 5.4e-8 on CI's BLAS), so whether it raises is platform-dependent.
    @pytest.mark.parametrize("epsilon", [1e-6])
    def test_the_flagged_grasp_really_does_return_an_absurd_reference(self, epsilon):
        """Why the flag earns its keep: 0.1 N·m of extra moment buys ≥ 10 kN of
        cable force, the range check passes, and ``check_taut`` calls it taut."""
        alloc = WrenchAllocation(_near_collinear_geometry(4, epsilon))
        wrench = alloc.reference_wrench + np.array([0.0, 0.0, 0.0, 0.1, 0.0, 0.0])
        tau = alloc.allocate(wrench, check_taut=False)
        assert np.max(np.linalg.norm(tau, axis=1)) > 1e4

    def test_a_healthy_grasp_is_not_flagged(self, caplog):
        """The ring is ``cond Φ = 2.8`` at every ``nQ`` — decades of headroom."""
        with caplog.at_level("WARNING"):
            alloc = WrenchAllocation(ring_attachment_points(4))
        assert alloc.condition_number < 10.0
        assert caplog.text == ""


# ─── Construction plumbing ────────────────────────────────────────────


class TestConstruction:
    def test_from_model_copies_every_parameter(self):
        model = MultiQuadRigidbodyBase(num_quadrotors=4)
        model.masses = [0.7, 0.9, 1.1, 1.3]
        model.cable_lengths = [0.9, 1.1, 1.2, 1.3]
        model.payload_mass = 1.4
        f2s = MultiQuadRigidbody.from_model(model)

        assert f2s.num_quadrotors == 4
        assert np.allclose(f2s.masses, model.masses)
        assert np.allclose(f2s.cable_lengths, model.cable_lengths)
        assert f2s.payload_mass == model.payload_mass
        assert np.allclose(f2s.attachment_points, model.attachment_points)
        assert np.allclose(f2s.payload_inertia, model.payload_inertia)

    def test_broadcasts_scalar_parameters(self):
        f2s = MultiQuadRigidbody(num_quadrotors=3, masses=0.8, cable_lengths=1.2)
        assert np.allclose(f2s.masses, 0.8)
        assert np.allclose(f2s.cable_lengths, 1.2)
        assert f2s.inertias.shape == (3, 3, 3)

    def test_rejects_bad_parameters(self):
        with pytest.raises(ValueError, match="num_quadrotors"):
            MultiQuadRigidbody(num_quadrotors=0)
        with pytest.raises(ValueError, match="masses must be positive"):
            MultiQuadRigidbody(num_quadrotors=2, masses=[1.0, -1.0])
        with pytest.raises(ValueError, match="cable_lengths must be positive"):
            MultiQuadRigidbody(num_quadrotors=2, cable_lengths=[1.0, 0.0])
        with pytest.raises(ValueError, match="payload_inertia must be positive definite"):
            MultiQuadRigidbody(num_quadrotors=3, payload_inertia=np.diag([1.0, 1.0, -1.0]))
        with pytest.raises(ValueError, match="attachment_points"):
            MultiQuadRigidbody(num_quadrotors=3, attachment_points=np.zeros((2, 3)))

    def test_explicit_policy_is_used(self):
        bias = np.tile([0.0, 0.0, 4.0], (3, 1))
        f2s = MultiQuadRigidbody(num_quadrotors=3, tension_policy=FixedTensionDistribution(bias))
        assert np.allclose(f2s.tension_policy.bias_forces, bias)

    def test_reference_state_projects_onto_the_model_state(self):
        model = MultiQuadRigidbodyBase(num_quadrotors=3)
        f2s = MultiQuadRigidbody.from_model(model)
        ref, _ = f2s.recover(_hover_flats(3))
        state = ref.as_state()
        assert state.num_quadrotors == 3
        assert len(state.cable_attitudes) == 3
        assert np.allclose(np.asarray(state.payload_orientation), np.eye(3), atol=TOL_EXACT)
