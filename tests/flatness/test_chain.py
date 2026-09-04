"""Tests for the cable-chain cascade primitive.

`udaan.flatness.chain` is exercised here **without the M5 flat map** — the
inputs are hand-built tip-position jets and the outputs are checked against
statics, against finite differences of the cascade's own zeroth-order
quantities, and against `cable_direction_jet` in the one-link case.

The finite-difference block is the load-bearing one: it verifies that the
derivative tuples the recursion produces really are the time derivatives of
the directions and tensions it produces, which is the whole content of the
primitive and the one thing statics cannot check.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_TIGHT
from udaan.core.defaults import GRAVITY
from udaan.flatness import cable_direction_jet
from udaan.flatness.chain import cable_chain_cascade, chain_jet_order

_E3 = np.array([0.0, 0.0, 1.0])


# ─── Analytic tip trajectories ────────────────────────────────────────


def sine_tip(t: float, order: int, amps, omegas, phases, offset) -> list[np.ndarray]:
    """Derivative tuple of ``x_j(t) = offset_j + A_j cos(ω_j t + φ_j)``.

    The k-th derivative of a cosine is the cosine phase-shifted by ``kπ/2``
    and scaled by ``ω^k``, so every row is closed form at any order — which is
    what makes an arbitrary-depth jet available for the growing ``2n + 4``.
    """
    rows = []
    for k in range(order + 1):
        row = np.array(
            [
                amps[j] * omegas[j] ** k * np.cos(omegas[j] * t + phases[j] + k * np.pi / 2)
                for j in range(3)
            ]
        )
        if k == 0:
            row = row + np.asarray(offset, dtype=float)
        rows.append(row)
    return rows


def _swing_tip(order: int):
    """A gentle, taut 3-D payload trajectory as a function of t."""
    return lambda t: sine_tip(
        t, order, [0.3, 0.3, 0.12], [1.2, 1.2, 0.9], [0.0, -np.pi / 2, 0.4], [0.0, 0.0, 0.0]
    )


def _static_tip(order: int, position=(0.0, 0.0, 0.0)) -> list[np.ndarray]:
    rows = [np.zeros(3) for _ in range(order + 1)]
    rows[0] = np.asarray(position, dtype=float)
    return rows


# ─── Jet-order bookkeeping ────────────────────────────────────────────


class TestChainJetOrder:
    @pytest.mark.parametrize("n,expected", [(1, 6), (2, 8), (3, 10), (5, 14), (10, 24)])
    def test_law_is_two_n_plus_four(self, n, expected):
        assert chain_jet_order(n) == expected

    def test_rejects_empty_chain(self):
        with pytest.raises(ValueError, match="num_links"):
            chain_jet_order(0)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_shallow_tip_jet_is_rejected(self, n):
        K = chain_jet_order(n)
        tip = _static_tip(K - 1, (0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match=r"2n \+ 4"):
            cable_chain_cascade(tip, np.full(n, 0.1), np.full(n, 0.2))

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_available_depth_drops_by_two_per_rung(self, n):
        """The documented depth table: link ``i`` comes out at ``K - 2(n-i)``."""
        K = chain_jet_order(n)
        chain = cable_chain_cascade(
            _swing_tip(K)(0.3), np.full(n, 0.1), np.full(n, 1.0 / n), gravity=GRAVITY
        )
        for i in range(n):
            expected_order = K - 2 * (n - i)
            assert len(chain.directions[i]) - 1 == expected_order
            assert len(chain.forces[i]) - 1 == expected_order
            assert len(chain.tensions[i]) - 1 == expected_order
        # Node positions: the tip keeps its full depth, node i inherits q_i's.
        assert len(chain.node_positions[n]) - 1 == K
        for i in range(n):
            assert len(chain.node_positions[i]) - 1 == K - 2 * (n - i)
        # In particular the top node comes out through its snap, at every n.
        assert len(chain.top_position) - 1 == 4

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_extra_depth_is_truncated_not_read(self, n):
        """A deeper tip jet gives bit-identical output.

        This is a **truncation regression guard**, not evidence that ``2n + 4``
        suffices. ``cable_chain_cascade`` slices ``tip[: K + 1]`` before doing
        any arithmetic, so surplus rows cannot move the answer no matter what
        the recursion does with the rows it keeps — any slicing implementation
        passes this. What it does catch is the truncation drifting: a mutant
        that keeps one row *fewer* is killed here. Sufficiency itself is an
        analytic statement, derived in the "Derivative depth" section of
        :mod:`udaan.flatness.chain`.
        """
        K = chain_jet_order(n)
        tip = _swing_tip(K + 3)(0.3)
        masses, lengths = np.full(n, 0.1), np.full(n, 1.0 / n)
        exact = cable_chain_cascade(tip[: K + 1], masses, lengths)
        deeper = cable_chain_cascade(tip, masses, lengths)
        # Poison every row above K; nothing may move.
        poisoned = list(tip)
        for k in range(K + 1, len(poisoned)):
            poisoned[k] = poisoned[k] + 1e6
        junk = cable_chain_cascade(poisoned, masses, lengths)
        for other in (deeper, junk):
            for i in range(n):
                assert np.array_equal(
                    np.asarray(exact.directions[i]), np.asarray(other.directions[i])
                )
                assert np.array_equal(np.asarray(exact.forces[i]), np.asarray(other.forces[i]))
            assert np.array_equal(np.asarray(exact.top_position), np.asarray(other.top_position))


# ─── Statics ──────────────────────────────────────────────────────────


class TestHangingChain:
    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_every_link_hangs_along_minus_e3(self, n):
        masses = np.linspace(0.05, 0.3, n)
        lengths = np.linspace(0.1, 0.4, n)
        chain = cable_chain_cascade(_static_tip(chain_jet_order(n)), masses, lengths)
        for i in range(n):
            assert np.allclose(chain.directions[i][0], -_E3, atol=TOL_EXACT)
            # A chain at rest has no rates at any order.
            for k in range(1, len(chain.directions[i])):
                assert np.allclose(chain.directions[i][k], 0.0, atol=TOL_EXACT)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_tension_is_the_weight_hanging_below(self, n):
        """``T_i = g Σ_{k ≥ i} m_k`` — the statics the flat map must reproduce."""
        masses = np.linspace(0.05, 0.3, n)
        lengths = np.full(n, 0.2)
        chain = cable_chain_cascade(_static_tip(chain_jet_order(n)), masses, lengths)
        for i in range(n):
            assert chain.tensions[i][0] == pytest.approx(GRAVITY * masses[i:].sum(), abs=TOL_TIGHT)

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_top_node_sits_one_cable_length_above_the_tip(self, n):
        lengths = np.linspace(0.1, 0.4, n)
        chain = cable_chain_cascade(
            _static_tip(chain_jet_order(n), (0.4, -0.2, 1.0)), np.full(n, 0.1), lengths
        )
        expected = np.array([0.4, -0.2, 1.0 + lengths.sum()])
        assert np.allclose(chain.top_position[0], expected, atol=TOL_TIGHT)
        # Nodes are strung out evenly along -e3 between the two.
        for i in range(n):
            below = 1.0 + lengths[i + 1 :].sum()
            assert chain.node_positions[i + 1][0][2] == pytest.approx(below, abs=TOL_TIGHT)

    def test_top_force_carries_the_whole_chain_weight(self):
        n, masses = 4, np.array([0.05, 0.07, 0.09, 0.4])
        chain = cable_chain_cascade(_static_tip(chain_jet_order(n)), masses, np.full(n, 0.2))
        # F_0 is the force the chain pulls the airframe down with.
        assert np.allclose(chain.top_force[0], -masses.sum() * GRAVITY * _E3, atol=TOL_TIGHT)


# ─── Internal consistency of one sweep ────────────────────────────────


class TestCascadeInvariants:
    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_node_positions_close_the_geometry(self, n):
        """``x_{i+1} = x_i + ℓ_i q_i`` at every derivative level both support."""
        K = chain_jet_order(n)
        lengths = np.linspace(0.1, 0.35, n)
        chain = cable_chain_cascade(_swing_tip(K)(0.7), np.linspace(0.05, 0.3, n), lengths)
        for i in range(n):
            above, below, q = (
                chain.node_positions[i],
                chain.node_positions[i + 1],
                chain.directions[i],
            )
            for k in range(min(len(above), len(below), len(q))):
                assert np.allclose(below[k], above[k] + lengths[i] * q[k], atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_force_recursion_closes(self, n):
        """``F_i − F_{i+1} = −m_i (ẍ_{i+1} + g e₃)`` at every level."""
        K = chain_jet_order(n)
        masses = np.linspace(0.05, 0.3, n)
        chain = cable_chain_cascade(_swing_tip(K)(0.7), masses, np.full(n, 1.0 / n))
        for i in range(n - 1):
            accel = chain.node_positions[i + 1][2:]
            depth = min(len(chain.forces[i]), len(chain.forces[i + 1]), len(accel))
            for k in range(depth):
                gravity_term = GRAVITY * _E3 if k == 0 else 0.0
                expected = -masses[i] * (accel[k] + gravity_term)
                assert np.allclose(
                    chain.forces[i][k] - chain.forces[i + 1][k], expected, atol=TOL_ANALYTIC
                )

    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_directions_are_unit_and_forces_decompose(self, n):
        K = chain_jet_order(n)
        chain = cable_chain_cascade(
            _swing_tip(K)(1.1), np.linspace(0.05, 0.3, n), np.full(n, 1.0 / n)
        )
        for i in range(n):
            q, T, F = chain.directions[i], chain.tensions[i], chain.forces[i]
            assert np.linalg.norm(q[0]) == pytest.approx(1.0, abs=TOL_TIGHT)
            assert T[0] > 0.0
            assert np.allclose(F[0], T[0] * q[0], atol=TOL_ANALYTIC)
            # q ⊥ q̇, so ω = q × q̇ is the genuine T_q S² rate.
            assert float(np.dot(q[0], q[1])) == pytest.approx(0.0, abs=TOL_TIGHT)
            omega = chain.angular_velocity(i)
            assert np.allclose(np.cross(omega, q[0]), q[1], atol=TOL_ANALYTIC)
            assert float(np.dot(omega, q[0])) == pytest.approx(0.0, abs=TOL_TIGHT)

    def test_single_link_matches_the_cable_primitive(self):
        """At ``n = 1`` the cascade *is* ``cable_direction_jet`` — the primitive
        it is built out of. Shared code, so this checks the plumbing (sign,
        truncation, gravity), not the physics."""
        K = chain_jet_order(1)
        tip = _swing_tip(K)(0.55)
        m, ell = 0.35, 0.8
        chain = cable_chain_cascade(tip, [m], [ell])
        T, q = cable_direction_jet(tip[2:], m, GRAVITY)
        assert np.allclose(np.asarray(chain.directions[0]), np.asarray(q), atol=TOL_EXACT)
        assert np.allclose(np.asarray(chain.tensions[0]), np.asarray(T), atol=TOL_EXACT)
        for k in range(len(chain.top_position)):
            assert np.allclose(chain.top_position[k], tip[k] - ell * q[k], atol=TOL_EXACT)


# ─── Finite-difference verification of the derivative tuples ──────────


def _fd(func, t: float, order: int, h: float = 2e-3) -> np.ndarray:
    """``order``-th derivative of ``func`` at ``t`` by a 5-point central stencil."""
    stencils = {
        1: ([-2, -1, 1, 2], [1 / 12, -2 / 3, 2 / 3, -1 / 12]),
        2: ([-2, -1, 0, 1, 2], [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
        3: ([-2, -1, 1, 2], [-1 / 2, 1.0, -1.0, 1 / 2]),
    }
    offsets, weights = stencils[order]
    total = None
    for offset, weight in zip(offsets, weights, strict=True):
        term = weight * np.asarray(func(t + offset * h), dtype=float)
        total = term if total is None else total + term
    return total / h**order


class TestDerivativesAgainstFiniteDifferences:
    """The recursion's derivative tuples really are time derivatives.

    Every quantity is re-evaluated as a plain function of ``t`` (order-0 output
    of an independent cascade call), then differentiated numerically. Nothing
    in this comparison reuses the analytic derivative machinery.
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_link_direction_derivatives(self, n, order):
        K = chain_jet_order(n)
        masses = np.linspace(0.06, 0.3, n)
        lengths = np.linspace(0.15, 0.35, n)
        tip = _swing_tip(K)

        def cascade(t):
            return cable_chain_cascade(tip(t), masses, lengths)

        t0 = 0.63
        analytic = cascade(t0)
        for i in range(n):
            numeric = _fd(lambda t, i=i: cascade(t).directions[i][0], t0, order)
            assert np.allclose(analytic.directions[i][order], numeric, atol=1e-7), (
                f"link {i}, d^{order}q/dt^{order}"
            )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_tension_and_force_derivatives(self, n):
        K = chain_jet_order(n)
        masses = np.linspace(0.06, 0.3, n)
        lengths = np.linspace(0.15, 0.35, n)
        tip = _swing_tip(K)

        def cascade(t):
            return cable_chain_cascade(tip(t), masses, lengths)

        t0 = 0.41
        analytic = cascade(t0)
        for i in range(n):
            for order in (1, 2):
                numeric_T = _fd(lambda t, i=i: cascade(t).tensions[i][0], t0, order)
                assert analytic.tensions[i][order] == pytest.approx(float(numeric_T), abs=1e-7)
                numeric_F = _fd(lambda t, i=i: cascade(t).forces[i][0], t0, order)
                assert np.allclose(analytic.forces[i][order], numeric_F, atol=1e-7)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_top_node_derivatives_through_snap(self, n):
        """The top node is what a vehicle map differentiates twice more, so its
        depth is the one that matters — check it all the way to jerk."""
        K = chain_jet_order(n)
        masses = np.linspace(0.06, 0.3, n)
        lengths = np.linspace(0.15, 0.35, n)
        tip = _swing_tip(K)

        def top(t):
            return cable_chain_cascade(tip(t), masses, lengths).top_position[0]

        t0 = 0.87
        analytic = cable_chain_cascade(tip(t0), masses, lengths).top_position
        for order in (1, 2, 3):
            assert np.allclose(analytic[order], _fd(top, t0, order), atol=1e-7)


# ─── External forces (the M6 hook) ────────────────────────────────────


class TestNodeForces:
    def test_upward_force_at_the_tip_unloads_the_last_link(self):
        """A force injected at a node enters that node's balance exactly once."""
        n, m, ell = 3, 0.2, 0.3
        K = chain_jet_order(n)
        lift = 0.4  # N, upwards
        forces = [None, None, [lift * _E3] + [np.zeros(3)] * K]
        plain = cable_chain_cascade(_static_tip(K), np.full(n, m), np.full(n, ell))
        lifted = cable_chain_cascade(
            _static_tip(K), np.full(n, m), np.full(n, ell), node_forces=forces
        )
        assert plain.tensions[2][0] == pytest.approx(m * GRAVITY, abs=TOL_TIGHT)
        assert lifted.tensions[2][0] == pytest.approx(m * GRAVITY - lift, abs=TOL_TIGHT)
        # Everything above it is unloaded by the same amount, and the chain
        # still hangs straight down.
        for i in range(n):
            assert lifted.tensions[i][0] == pytest.approx(
                plain.tensions[i][0] - lift, abs=TOL_TIGHT
            )
            assert np.allclose(lifted.directions[i][0], -_E3, atol=TOL_TIGHT)

    def test_force_at_an_intermediate_node_leaves_links_below_untouched(self):
        n, m, ell = 4, 0.15, 0.25
        K = chain_jet_order(n)
        push = [np.array([0.0, 0.3, 0.5])] + [np.zeros(3)] * K
        plain = cable_chain_cascade(_static_tip(K), np.full(n, m), np.full(n, ell))
        pushed = cable_chain_cascade(
            _static_tip(K),
            np.full(n, m),
            np.full(n, ell),
            node_forces=[None, push, None, None],
        )
        # Link 1's mass sits at node 2; links 2 and 3 are strictly below it.
        for i in (2, 3):
            assert np.allclose(
                np.asarray(pushed.forces[i]), np.asarray(plain.forces[i]), atol=TOL_EXACT
            )
        for i in (0, 1):
            assert not np.allclose(pushed.forces[i][0], plain.forces[i][0], atol=1e-6)

    def test_wrong_length_node_forces_is_rejected(self):
        n = 3
        with pytest.raises(ValueError, match="node_forces must have 3 entries"):
            cable_chain_cascade(
                _static_tip(chain_jet_order(n)),
                np.full(n, 0.1),
                np.full(n, 0.2),
                node_forces=[None, None],
            )

    @pytest.mark.parametrize(
        "entry,message",
        [
            ([], r"node_forces\[1\] must reach the 6-th derivative"),
            ([np.zeros(3)], r"node_forces\[1\] must reach the 6-th derivative"),
            ([np.zeros(3)] * 6, r"node_forces\[1\] must reach the 6-th derivative"),
            ([np.zeros(2)] * 11, r"node_forces\[1\] rows must be 3-vectors"),
        ],
    )
    def test_malformed_node_force_jets_are_named(self, entry, message):
        """The M6 hook validates to the same standard as the tip jet.

        A short or wrongly-shaped external jet used to collapse the chain
        silently and surface as a bare ``IndexError`` (or a numpy broadcast
        error) from somewhere much further up. Link 1 of a 4-link chain carries
        its force through the ``K - 2(n - i) = 12 - 6 = 6``-th derivative, so
        that is the depth an external force entering there must supply.
        """
        n = 4
        with pytest.raises(ValueError, match=message):
            cable_chain_cascade(
                _static_tip(chain_jet_order(n)),
                np.full(n, 0.1),
                np.full(n, 0.2),
                node_forces=[None, entry, None, None],
            )

    def test_a_node_force_at_the_required_depth_is_accepted(self):
        """Exactly ``K - 2(n - i)`` rows is enough — the boundary of the check."""
        n = 4
        K = chain_jet_order(n)
        entry = [np.array([0.0, 0.2, 0.1])] + [np.zeros(3)] * (K - 2 * (n - 1))
        chain = cable_chain_cascade(
            _static_tip(K),
            np.full(n, 0.1),
            np.full(n, 0.2),
            node_forces=[None, entry, None, None],
        )
        assert len(chain.top_position) - 1 == 4


# ─── Slack / singularity ──────────────────────────────────────────────


class TestSlack:
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_free_fall_slackens_the_last_link_first(self, n):
        """In free fall every link's force vanishes; the sweep reaches the
        bottom one first, so that is the index reported."""
        K = chain_jet_order(n)
        tip = _static_tip(K)
        tip[2] = -GRAVITY * _E3
        with pytest.raises(ValueError, match=rf"link {n - 1} of {n} is slack"):
            cable_chain_cascade(tip, np.full(n, 0.1), np.full(n, 0.2))

    def test_an_upper_link_can_slacken_on_its_own(self):
        """A node force that exactly cancels the weight below it slackens the
        link *above* that node while the ones below stay taut."""
        n, m, ell = 2, 0.2, 0.3
        K = chain_jet_order(n)
        # Lift node 1 (link 0's mass) with the full weight of everything below.
        lift = 2 * m * GRAVITY
        with pytest.raises(ValueError, match="link 0 of 2 is slack"):
            cable_chain_cascade(
                _static_tip(K),
                np.full(n, m),
                np.full(n, ell),
                node_forces=[[lift * _E3] + [np.zeros(3)] * K, None],
            )


class TestHangingBranch:
    """Crossing the slack boundary is refused, not silently followed.

    ``T_i = ‖F_i‖`` is a norm, so it is positive again on the far side of
    ``F_i = 0``; the slack check alone therefore only refuses to *land* on the
    boundary, which a sampled trajectory never does. The hanging-branch test is
    what makes the refusal two-sided.
    """

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_past_free_fall_the_flipped_branch_is_rejected(self, n):
        """``ẍ_L = -2g e₃`` puts every link at ``q_i = +e₃`` — the payload above
        the airframe. Well posed for a rigid link, meaningless for a cable."""
        K = chain_jet_order(n)
        tip = _static_tip(K)
        tip[2] = -2.0 * GRAVITY * _E3
        with pytest.raises(ValueError, match=rf"link {n - 1} of {n} has left the hanging branch"):
            cable_chain_cascade(tip, np.full(n, 0.1), np.full(n, 0.2))

    def test_a_link_swung_past_horizontal_is_rejected(self):
        """The check is conservative by design: 90° from straight down is the
        boundary, reached here by a purely horizontal specific force."""
        n = 1
        K = chain_jet_order(n)
        tip = _static_tip(K)
        # ẍ + g e₃ = (1, 0, -0.01): holding the payload needs a slightly
        # *downward* pull, so q tips just past horizontal into the upper half.
        tip[2] = np.array([1.0, 0.0, -GRAVITY - 0.01])
        with pytest.raises(ValueError, match="has left the hanging branch"):
            cable_chain_cascade(tip, np.full(n, 0.1), np.full(n, 0.2))
        # Just on the other side of horizontal it is accepted.
        tip[2] = np.array([1.0, 0.0, -GRAVITY + 0.01])
        chain = cable_chain_cascade(tip, np.full(n, 0.1), np.full(n, 0.2))
        assert chain.directions[0][0][2] < 0.0


# ─── Argument validation ──────────────────────────────────────────────


class TestValidation:
    def test_mismatched_masses_and_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            cable_chain_cascade(_static_tip(10), [0.1, 0.2, 0.3], [0.2, 0.2])

    @pytest.mark.parametrize(
        "masses,lengths,message",
        [
            ([0.1, -0.2], [0.2, 0.2], "masses must be strictly positive"),
            ([0.1, 0.2], [0.2, 0.0], "lengths must be strictly positive"),
        ],
    )
    def test_non_positive_parameters(self, masses, lengths, message):
        with pytest.raises(ValueError, match=message):
            cable_chain_cascade(_static_tip(chain_jet_order(2)), masses, lengths)

    def test_non_vector_rows(self):
        tip = [np.zeros(2) for _ in range(chain_jet_order(1) + 1)]
        with pytest.raises(ValueError, match="3-vectors"):
            cable_chain_cascade(tip, [0.2], [0.5])
