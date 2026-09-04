"""Unit tests for the M6 flat-to-state map — ``nQ`` quadrotors on one hose.

The open-loop round trip in ``test_multi_quad_hose_feedforward.py`` is the gate
on the whole map; this file covers the pieces that a round trip cannot isolate:
the structural not-flat guard, the ``2n + 4`` jet-depth law and its per-tension
companion, the cascade's geometry and force invariants, derivative levels
checked against finite differences, the tautness boundary and the wrong-branch
guards that stand behind it, the observability exponents ``ℓ^d / T^(d+1)``, the
reduction to M5 where the geometry allows it, and the reversal identity that
connects :meth:`~udaan.flatness.MultiQuadHose.cascade` to the raw
:func:`~udaan.flatness.cable_chain_cascade` primitive it delegates to.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_TIGHT
from udaan.flatness import (
    DEFAULT_MIN_TENSION,
    Jet,
    MultiQuadHose,
    MultiQuadHoseFlats,
    NotFlatError,
    QuadrotorFlexCable,
    QuadrotorFlexCableFlats,
    cable_chain_cascade,
)
from udaan.models.multi_quad_hose.base import MultiQuadHoseBase

_E3 = np.array([0.0, 0.0, 1.0])


# ─── Flat-output helpers ──────────────────────────────────────────────


def sine_rows(t: float, order: int, amps, omegas, phases) -> np.ndarray:
    """Exact derivative rows of ``x_j(t) = A_j cos(ω_j t + φ_j)``."""
    data = np.zeros((order + 1, 3))
    for k in range(order + 1):
        for j in range(3):
            data[k, j] = (
                amps[j] * omegas[j] ** k * np.cos(omegas[j] * t + phases[j] + k * np.pi / 2)
            )
    return data


def tension_jets(f2s: MultiQuadHose, t: float = 0.0, pull=3.0, spread=1.0, wiggle=0.0):
    """Flat tension jets: a constant taut pull, optionally breathing in x."""
    jets = []
    for k, order in enumerate(f2s.tension_jet_orders):
        rows = np.zeros((order + 1, 3))
        rows[0] = np.array([spread * (1.0 + 0.3 * k), 0.0, -pull])
        if wiggle:
            w = 0.6 + 0.11 * k
            for d in range(order + 1):
                rows[d, 0] += wiggle * w**d * np.cos(w * t + d * np.pi / 2)
        jets.append(Jet(rows))
    return jets


def yaw_jets(f2s: MultiQuadHose, t: float = 0.0, moving: bool = False):
    if not moving:
        return [Jet(np.zeros(3)) for _ in range(f2s.num_quadrotors)]
    return [
        Jet(
            np.array(
                [
                    0.18 * np.sin((0.5 + 0.13 * k) * t),
                    0.18 * (0.5 + 0.13 * k) * np.cos((0.5 + 0.13 * k) * t),
                    -0.18 * (0.5 + 0.13 * k) ** 2 * np.sin((0.5 + 0.13 * k) * t),
                ]
            )
        )
        for k in range(f2s.num_quadrotors)
    ]


def static_flats(f2s: MultiQuadHose, **kwargs) -> MultiQuadHoseFlats:
    return MultiQuadHoseFlats(
        x_0=Jet(np.zeros((f2s.jet_order + 1, 3))),
        tensions=tension_jets(f2s, **kwargs),
        psi=yaw_jets(f2s),
    )


def dynamic_flats(f2s: MultiQuadHose, t: float, wiggle=0.3) -> MultiQuadHoseFlats:
    return MultiQuadHoseFlats(
        x_0=Jet(sine_rows(t, f2s.jet_order, [0.25, 0.25, 0.10], [1.1, 1.1, 0.8], [0.0, -1.4, 0.4])),
        tensions=tension_jets(f2s, t, wiggle=wiggle),
        psi=yaw_jets(f2s, t, moving=True),
    )


def make(n: int, attachments, **kwargs) -> MultiQuadHose:
    return MultiQuadHose(num_links=n, attachments=attachments, **kwargs)


CONFIGURATIONS = [(1, (0, 1)), (2, (2,)), (3, (0, 3)), (4, (0, 2, 4)), (4, (2, 4)), (5, (0, 3, 5))]


# ─── Contract and construction ────────────────────────────────────────


class TestContract:
    def test_declares_the_flat2state_triple(self):
        assert MultiQuadHose.Flats is MultiQuadHoseFlats
        assert MultiQuadHose.RefState.__name__ == "MultiQuadHoseRefState"
        assert MultiQuadHose.Inputs.__name__ == "MultiQuadHoseInputs"

    def test_from_model_copies_every_parameter(self):
        model = MultiQuadHoseBase(
            num_links=4,
            attachments=(0, 2, 4),
            vertex_masses=np.linspace(0.04, 0.19, 5),
            link_lengths=np.linspace(0.3, 0.6, 4),
            masses=np.array([0.8, 0.9, 1.0]),
        )
        f2s = MultiQuadHose.from_model(model)
        assert f2s.attachments == model.attachments
        assert np.allclose(f2s.link_lengths, model.link_lengths)
        assert np.allclose(f2s.vertex_masses, model.vertex_masses)
        assert np.allclose(f2s.masses, model.masses)
        assert np.allclose(f2s.vertex_total_masses, model.vertex_total_masses)
        assert f2s.total_mass == pytest.approx(model.total_mass)

    def test_from_model_does_not_alias_the_model_arrays(self):
        model = MultiQuadHoseBase(num_links=3, attachments=(0, 3))
        f2s = MultiQuadHose.from_model(model)
        f2s.link_lengths[0] = 99.0
        assert model.link_lengths[0] != 99.0

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_flat_output_dimension_equals_input_dimension(self, n, attachments):
        """``3 + 3(nQ − 1) + nQ = 4nQ`` — differential flatness needs it."""
        f2s = make(n, attachments)
        nQ = f2s.num_quadrotors
        flat_dim = 3 + 3 * len(f2s.tension_jet_orders) + nQ
        assert flat_dim == 4 * nQ
        assert len(f2s.tension_jet_orders) == nQ - 1


class TestNotFlatGuard:
    """A hose with no vehicle at its last vertex is not flat in these coordinates."""

    @pytest.mark.parametrize("attachments", [(0,), (0, 2), (1, 3), (0, 1, 2)])
    def test_missing_vehicle_at_the_last_vertex_is_refused(self, attachments):
        with pytest.raises(NotFlatError, match="not differentially flat"):
            make(4, attachments)

    def test_the_message_names_the_vertex_to_add(self):
        with pytest.raises(NotFlatError, match=r"add vertex 4"):
            make(4, (0, 2))

    def test_it_is_a_valueerror_so_broad_catchers_still_see_it(self):
        assert issubclass(NotFlatError, ValueError)

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_a_vehicle_at_the_last_vertex_is_accepted(self, n, attachments):
        assert make(n, attachments).attachments[-1] == n

    def test_from_model_propagates_the_guard(self):
        model = MultiQuadHoseBase(num_links=3, attachments=(0, 2))
        with pytest.raises(NotFlatError):
            MultiQuadHose.from_model(model)

    def test_unsorted_attachments_are_refused_by_the_shared_validator(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            make(4, (2, 0, 4))

    def test_a_massless_vertex_is_refused(self):
        with pytest.raises(ValueError, match="strictly positive total mass"):
            make(3, (0, 3), vertex_masses=0.0)


# ─── Jet depth ────────────────────────────────────────────────────────


class TestJetDepth:
    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
    def test_root_jet_order_is_two_n_plus_four(self, n):
        assert make(n, (0, n)).jet_order == 2 * n + 4

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_tension_orders_are_two_n_minus_a_plus_two(self, n, attachments):
        f2s = make(n, attachments)
        expected = tuple(2 * (n - a) + 2 for a in attachments[:-1])
        assert f2s.tension_jet_orders == expected

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_one_order_short_is_refused(self, n, attachments):
        """``2n + 4 − 1`` is not enough: the last vehicle's moment needs the
        top vertex through its snap, and that is what the last two rows buy."""
        f2s = make(n, attachments)
        flats = static_flats(f2s)
        with pytest.raises(ValueError, match="order"):
            # At n = 1 the ``n``-independent floor in ``Flats.__post_init__``
            # catches it; above that it is ``recover``'s ``n``-aware check.
            f2s.recover(
                MultiQuadHoseFlats(
                    x_0=flats.x_0.truncate(f2s.jet_order - 1),
                    tensions=flats.tensions,
                    psi=flats.psi,
                )
            )

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_extra_depth_changes_nothing(self, n, attachments):
        """Rows above ``2n + 4`` are never read, so a deeper jet is identical."""
        f2s = make(n, attachments)
        base = dynamic_flats(f2s, 0.37)
        deeper = MultiQuadHoseFlats(
            x_0=Jet(
                sine_rows(
                    0.37, f2s.jet_order + 3, [0.25, 0.25, 0.10], [1.1, 1.1, 0.8], [0.0, -1.4, 0.4]
                )
            ),
            tensions=base.tensions,
            psi=base.psi,
        )
        ref_a, in_a = f2s.recover(base)
        ref_b, in_b = f2s.recover(deeper)
        assert np.allclose(in_a.thrusts, in_b.thrusts, atol=TOL_EXACT)
        assert np.allclose(in_a.moments, in_b.moments, atol=TOL_EXACT)
        assert np.allclose(ref_a.vertex_positions, ref_b.vertex_positions, atol=TOL_EXACT)

    def test_a_shallow_tension_jet_is_named(self):
        f2s = make(4, (0, 2, 4))
        flats = static_flats(f2s)
        short = MultiQuadHoseFlats(
            x_0=flats.x_0,
            tensions=[flats.tensions[0].truncate(2), flats.tensions[1]],
            psi=flats.psi,
        )
        with pytest.raises(ValueError, match=r"tensions\[0\]"):
            f2s.recover(short)

    def test_the_wrong_number_of_tension_jets_is_refused(self):
        f2s = make(4, (0, 2, 4))
        flats = static_flats(f2s)
        with pytest.raises(ValueError, match="expected 2 flat tension jets"):
            f2s.recover(
                MultiQuadHoseFlats(x_0=flats.x_0, tensions=flats.tensions[:1], psi=flats.psi)
            )

    def test_the_wrong_number_of_yaw_jets_is_refused(self):
        f2s = make(3, (0, 3))
        flats = static_flats(f2s)
        with pytest.raises(ValueError, match="expected 2 yaw jets"):
            f2s.recover(
                MultiQuadHoseFlats(x_0=flats.x_0, tensions=flats.tensions, psi=flats.psi[:1])
            )


# ─── Cascade invariants ───────────────────────────────────────────────


class TestCascade:
    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_geometry_closes(self, n, attachments):
        """``x_{j+1} = x_j + ℓ_j q_j`` at every vertex and every derivative."""
        f2s = make(n, attachments)
        chain = f2s.cascade(dynamic_flats(f2s, 0.21))
        for i in range(n):
            above = chain.vertex_positions[i]
            below = chain.vertex_positions[i + 1]
            q = chain.directions[i]
            depth = min(len(above), len(below), len(q))
            for k in range(depth):
                got = above[k] + f2s.link_lengths[i] * q[k]
                assert np.allclose(got, below[k], atol=TOL_TIGHT)

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_directions_are_unit_and_forces_decompose(self, n, attachments):
        f2s = make(n, attachments)
        chain = f2s.cascade(dynamic_flats(f2s, 0.21))
        for i in range(n):
            assert float(np.linalg.norm(chain.directions[i][0])) == pytest.approx(
                1.0, abs=TOL_TIGHT
            )
            # q · q̇ = 0 — the direction stays on S².
            assert float(np.dot(chain.directions[i][0], chain.directions[i][1])) == pytest.approx(
                0.0, abs=TOL_ANALYTIC
            )
            assert chain.tensions[i][0] > 0.0
            assert np.allclose(
                chain.forces[i][0],
                chain.tensions[i][0] * chain.directions[i][0],
                atol=TOL_TIGHT,
            )

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_newton_closes_at_every_vertex(self, n, attachments):
        """``m̄_j(ẍ_j + g e₃) = u_j − 𝛕_{j-1} + 𝛕_j`` — the recursion's own
        equation, re-checked on the *un-reversed* outputs, which is where a
        sign or index slip in the reversal would show up."""
        f2s = make(n, attachments)
        chain = f2s.cascade(dynamic_flats(f2s, 0.21))
        mbar = f2s.vertex_total_masses
        for j in range(n + 1):
            accel = chain.vertex_positions[j][2]
            lhs = mbar[j] * (accel + f2s.gravity * _E3)
            rhs = np.zeros(3)
            if chain.vertex_thrusts[j] is not None:
                rhs = rhs + chain.vertex_thrusts[j][0]
            if j > 0:
                rhs = rhs - chain.forces[j - 1][0]
            if j < n:
                rhs = rhs + chain.forces[j][0]
            assert np.allclose(lhs, rhs, atol=TOL_ANALYTIC), f"vertex {j}"

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_only_attachment_vertices_carry_a_thrust(self, n, attachments):
        f2s = make(n, attachments)
        chain = f2s.cascade(static_flats(f2s))
        carrying = {j for j in range(n + 1) if chain.vertex_thrusts[j] is not None}
        assert carrying == set(attachments)

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_the_flat_tensions_come_back_out_verbatim(self, n, attachments):
        """A flat tension is the link force by definition, not an approximation
        of one — the prescribed-force hook must reproduce it exactly."""
        f2s = make(n, attachments)
        flats = dynamic_flats(f2s, 0.21)
        chain = f2s.cascade(flats)
        for k, vertex in enumerate(attachments[:-1]):
            supplied = np.asarray(flats.tensions[k])
            got = np.stack(chain.forces[vertex][: len(supplied)])
            assert np.allclose(got, supplied[: len(got)], atol=TOL_EXACT)

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_available_depth_drops_by_two_per_rung(self, n, attachments):
        f2s = make(n, attachments)
        chain = f2s.cascade(static_flats(f2s))
        for j in range(n + 1):
            assert len(chain.vertex_positions[j]) == f2s.jet_order + 1 - 2 * j


class TestReversalIdentity:
    """:meth:`MultiQuadHose.cascade` is the raw primitive on the reversed chain.

    Asserting it here — rather than only inside the map — is what keeps the
    reversal honest: if the index or sign bookkeeping drifts, this test names
    the reversal rather than surfacing as a mysterious dynamics residual.
    """

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_matches_a_hand_reversed_cable_chain_cascade(self, n, attachments):
        f2s = make(n, attachments)
        flats = dynamic_flats(f2s, 0.21)
        got = f2s.cascade(flats)

        mbar = f2s.vertex_total_masses
        prescribed = [None] * n
        for k, vertex in enumerate(attachments[:-1]):
            jet = flats.tensions[k]
            prescribed[n - 1 - vertex] = [-np.asarray(jet[d]) for d in range(len(jet))]
        raw = cable_chain_cascade(
            [np.asarray(flats.x_0[d]) for d in range(f2s.jet_order + 1)],
            mbar[n - 1 :: -1],
            f2s.link_lengths[::-1],
            gravity=f2s.gravity,
            link_forces=prescribed,
            require_hanging_branch=False,
        )
        for j in range(n + 1):
            assert np.allclose(
                np.stack(got.vertex_positions[j]),
                np.stack(raw.node_positions[n - j]),
                atol=TOL_EXACT,
            )
        for i in range(n):
            assert np.allclose(
                np.stack(got.directions[i]), -np.stack(raw.directions[n - 1 - i]), atol=TOL_EXACT
            )
            assert np.allclose(
                np.stack(got.forces[i]), -np.stack(raw.forces[n - 1 - i]), atol=TOL_EXACT
            )
            assert np.allclose(got.tensions[i], raw.tensions[n - 1 - i], atol=TOL_EXACT)

    @staticmethod
    def _raw_args(f2s, flats):
        n = f2s.num_links
        prescribed = [None] * n
        for k, vertex in enumerate(f2s.attachments[:-1]):
            jet = flats.tensions[k]
            prescribed[n - 1 - vertex] = [-np.asarray(jet[d]) for d in range(len(jet))]
        args = (
            [np.asarray(flats.x_0[d]) for d in range(f2s.jet_order + 1)],
            f2s.vertex_total_masses[n - 1 :: -1],
            f2s.link_lengths[::-1],
        )
        return args, {"gravity": f2s.gravity, "link_forces": prescribed}

    @pytest.mark.parametrize(("n", "attachments"), [(2, (0, 2)), (4, (0, 4)), (4, (0, 2, 4))])
    def test_the_branch_check_rejects_a_multi_vehicle_hose(self, n, attachments):
        """The removal is load-bearing, not cosmetic: with M5's branch test on,
        a hose spread between two or more vehicles is refused outright."""
        f2s = make(n, attachments)
        args, kwargs = self._raw_args(f2s, static_flats(f2s))
        # Off: fine. On: refused.
        cable_chain_cascade(*args, require_hanging_branch=False, **kwargs)
        with pytest.raises(ValueError, match="hanging branch"):
            cable_chain_cascade(*args, require_hanging_branch=True, **kwargs)

    @pytest.mark.parametrize(("n", "attachments"), [(2, (2,)), (3, (3,)), (4, (4,))])
    def test_but_a_single_vehicle_hose_passes_it(self, n, attachments):
        """It is *not* every catenary the check rejects — only a chain whose own
        sweep runs uphill. A hose with its one vehicle at the last vertex hangs
        off that vehicle exactly as M5's cable does and passes unchanged, which
        is why the map leaves the check on there."""
        f2s = make(n, attachments)
        args, kwargs = self._raw_args(f2s, static_flats(f2s))
        chain = cable_chain_cascade(*args, require_hanging_branch=True, **kwargs)
        assert chain.num_links == n


# ─── Derivative levels against finite differences ─────────────────────


def _central(fn, t: float, h: float = 1e-4):
    return (fn(t + h) - fn(t - h)) / (2.0 * h)


class TestDerivativesAgainstFiniteDifferences:
    """Every derivative row is produced by the Leibniz engine; nothing in this
    file's algebra would notice a wrong one. Finite differences would."""

    @pytest.mark.parametrize(("n", "attachments"), [(2, (0, 2)), (3, (0, 3)), (4, (0, 2, 4))])
    def test_link_direction_derivatives(self, n, attachments):
        f2s = make(n, attachments)
        t0 = 0.31
        for i in range(n):
            for order in (1, 2):
                analytic = f2s.cascade(dynamic_flats(f2s, t0)).directions[i][order]
                numeric = _central(
                    lambda t, i=i, order=order: f2s.cascade(dynamic_flats(f2s, t)).directions[i][
                        order - 1
                    ],
                    t0,
                )
                assert np.allclose(analytic, numeric, atol=1e-6), f"link {i} d^{order} q"

    @pytest.mark.parametrize(("n", "attachments"), [(2, (0, 2)), (4, (0, 2, 4))])
    def test_thrust_vector_derivatives(self, n, attachments):
        """The rows that become ``Ω`` and ``M``."""
        f2s = make(n, attachments)
        t0 = 0.31
        for vertex in attachments:
            for order in (1, 2):
                analytic = f2s.cascade(dynamic_flats(f2s, t0)).vertex_thrusts[vertex][order]
                numeric = _central(
                    lambda t, v=vertex, o=order: f2s.cascade(dynamic_flats(f2s, t)).vertex_thrusts[
                        v
                    ][o - 1],
                    t0,
                )
                assert np.allclose(analytic, numeric, atol=1e-5), f"vertex {vertex} d^{order} u"

    @pytest.mark.parametrize(("n", "attachments"), [(3, (0, 3)), (4, (2, 4))])
    def test_body_rate_matches_the_attitude_derivative(self, n, attachments):
        """``Ω̂ = Rᵀ Ṙ`` against a numerically differentiated ``R(t)``."""
        from udaan.manif import vee

        f2s = make(n, attachments)
        t0 = 0.31
        ref = f2s.recover(dynamic_flats(f2s, t0))[0]
        for k in range(f2s.num_quadrotors):
            dR = _central(
                lambda t, k=k: np.asarray(f2s.recover(dynamic_flats(f2s, t))[0].orientations[k]),
                t0,
            )
            R = np.asarray(ref.orientations[k])
            Omega = vee(0.5 * (R.T @ dR - (R.T @ dR).T))
            assert np.allclose(Omega, np.asarray(ref.angular_velocities[k]), atol=1e-6)


# ─── Statics ──────────────────────────────────────────────────────────


class TestStatics:
    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_the_team_carries_exactly_the_system_weight(self, n, attachments):
        f2s = make(n, attachments)
        ref, inputs = f2s.recover(static_flats(f2s))
        total = sum(
            inputs.thrusts[k] * np.asarray(ref.orientations[k])[:, 2]
            for k in range(f2s.num_quadrotors)
        )
        assert np.allclose(total, f2s.total_mass * f2s.gravity * _E3, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_a_static_reference_has_no_rates_and_no_moments(self, n, attachments):
        f2s = make(n, attachments)
        ref, inputs = f2s.recover(static_flats(f2s))
        assert np.allclose(inputs.moments, 0.0, atol=TOL_ANALYTIC)
        for k in range(f2s.num_quadrotors):
            assert np.allclose(np.asarray(ref.angular_velocities[k]), 0.0, atol=TOL_ANALYTIC)
        for i in range(n):
            assert np.allclose(ref.link_angular_velocities[i], 0.0, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_the_model_agrees_it_is_an_equilibrium(self, n, attachments):
        """Independent of the map: hand the recovered state and wrench to the
        Step-12 model and ask *it* for the accelerations. They must vanish."""
        model = MultiQuadHoseBase(num_links=n, attachments=attachments)
        f2s = MultiQuadHose.from_model(model)
        ref, inputs = f2s.recover(static_flats(f2s))
        model.reset(
            root_position=np.asarray(ref.root_position).copy(),
            link_attitudes=[np.asarray(q).copy() for q in ref.link_attitudes],
            orientations=[np.asarray(R).copy() for R in ref.orientations],
        )
        acc = model.accelerations(inputs.thrusts, inputs.moments)
        assert np.max(np.abs(acc["vertex_accels"])) < 1e-9
        assert np.max(np.abs(acc["link_ang_accels"])) < 1e-9
        assert np.max(np.abs(acc["quad_ang_accels"])) < 1e-9

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_as_state_projects_onto_the_model_state(self, n, attachments):
        f2s = make(n, attachments)
        ref, _ = f2s.recover(static_flats(f2s))
        state = ref.as_state(attachments=attachments)
        assert state.num_links == n
        assert state.attachments == attachments
        assert len(state.link_attitudes) == n
        assert len(state.orientations) == f2s.num_quadrotors
        assert np.allclose(np.asarray(state.root_position), ref.root_position)


# ─── Tautness and slack ───────────────────────────────────────────────


class TestTautness:
    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_reported_tensions_are_the_link_force_norms(self, n, attachments):
        f2s = make(n, attachments)
        chain = f2s.cascade(dynamic_flats(f2s, 0.21))
        _, inputs = f2s.recover(dynamic_flats(f2s, 0.21))
        for i in range(n):
            assert inputs.tensions[i] == pytest.approx(
                float(np.linalg.norm(chain.forces[i][0])), abs=TOL_TIGHT
            )
            assert inputs.tensions[i] > 0.0

    def test_a_vanishing_flat_tension_slackens_its_own_link(self):
        """Set the flat tension to zero and the link it names goes limp — the
        map refuses, and the message says which link."""
        f2s = make(4, (0, 4))
        flats = MultiQuadHoseFlats(
            x_0=Jet(np.zeros((f2s.jet_order + 1, 3))),
            tensions=[Jet(np.zeros((f2s.tension_jet_orders[0] + 1, 3)))],
            psi=yaw_jets(f2s),
        )
        with pytest.raises(ValueError, match="slack"):
            f2s.recover(flats)

    def test_tension_falls_continuously_towards_the_slack_boundary(self):
        """The margin is genuinely two-sided here: ``𝛕_0`` is *given*, so
        shrinking it walks the link force to zero without any flipped branch to
        land on — unlike M5, where the tension is a norm."""
        f2s = make(3, (0, 3))
        seen = []
        for pull in (3.0, 1.0, 0.3, 0.1, 0.03):
            _, inputs = f2s.recover(static_flats(f2s, pull=pull, spread=0.0))
            seen.append(inputs.tensions[0])
        assert seen == sorted(seen, reverse=True)
        assert seen[-1] == pytest.approx(0.03, abs=TOL_ANALYTIC)

    def test_a_vehicle_in_free_fall_is_refused(self):
        """Choose the flat tension so the first vehicle's thrust vanishes."""
        f2s = make(3, (0, 3))
        mbar = f2s.vertex_total_masses
        # u_0 = m̄_0 (0 + g e₃) − 𝛕_0, so 𝛕_0 = m̄_0 g e₃ leaves nothing to lift.
        rows = np.zeros((f2s.tension_jet_orders[0] + 1, 3))
        rows[0] = mbar[0] * f2s.gravity * _E3
        flats = MultiQuadHoseFlats(
            x_0=Jet(np.zeros((f2s.jet_order + 1, 3))),
            tensions=[Jet(rows)],
            psi=yaw_jets(f2s),
        )
        with pytest.raises(ValueError, match="free fall"):
            f2s.recover(flats)


# ─── The wrong branch ─────────────────────────────────────────────────


def diving_flats(f2s, t, amp, omega, x_amp=0.0, **kwargs):
    """Root vertex on ``A cos(ωt)`` in ``z`` (and optionally ``x``).

    With ``A ω² > g`` the root spends part of every period accelerating
    downwards harder than gravity, which is where a derived link force passes
    through zero.
    """
    return MultiQuadHoseFlats(
        x_0=Jet(
            sine_rows(t, f2s.jet_order, [x_amp, 0.0, amp], [omega, 0.0, omega], [0.0, 0.0, 0.0])
        ),
        tensions=tension_jets(f2s, **kwargs),
        psi=yaw_jets(f2s),
    )


class TestWrongBranchProtection:
    """Only the links hanging off a vehicle have their force prescribed.

    The other ``n − (nQ − 1)`` are *derived* from
    ``𝛕_j = m̄_j(ẍ_j + g e₃) + 𝛕_{j-1}`` and their directions come out of a
    norm, exactly as M5's do — so they fold through zero and come back out with
    the tension positive and rising. Two guards and one gap:

    * ``nQ = 1``: no prescribed rungs, the hose *is* M5, and the map keeps M5's
      hanging-branch test — full two-sided protection, for free.
    * ``nQ ≥ 2``: the branch test has to go (a catenary's far links point up on
      purpose), and no per-rung direction test replaces it. What does is a
      ``min_tension`` floor on the *derived* rungs, which refuses the
      neighbourhood of the crossing but is a margin, not a branch test.
    """

    # ── nQ = 1: the hanging-branch test is back on ──────────────────
    #
    # A root diving at A = 0.6 m, ω = 4.5 rad/s exceeds free fall for part of
    # every period. Without the check the whole hose folds to the other side
    # of the root — vertex 2 jumping from −1.5 m to +2.5 m — with the tensions
    # positive and rising, the thrust positive, the attitude well defined and
    # no exception raised at all.
    DIVE = {"amp": 0.6, "omega": 4.5}

    def test_a_diving_root_is_refused_at_one_vehicle(self):
        f2s = make(2, (2,))
        assert f2s.num_quadrotors == 1
        for t in (0.10, 0.12, 0.14):  # ẍ·e₃ < −g: the folded branch
            with pytest.raises(ValueError, match="hanging branch"):
                f2s.recover(diving_flats(f2s, t, **self.DIVE))

    def test_the_same_trajectory_is_accepted_where_it_is_well_posed(self):
        """Not a blanket refusal: past the crossing the root is decelerating
        again, the hose hangs the right way up, and the map answers."""
        f2s = make(2, (2,))
        for t in (0.18, 0.20, 0.22):
            ref, inputs = f2s.recover(diving_flats(f2s, t, **self.DIVE))
            assert ref.vertex_positions[2][2] > ref.vertex_positions[0][2]
            assert inputs.thrusts[0] > 0.0

    def test_the_refusal_names_the_hose_link(self):
        f2s = make(2, (2,))
        with pytest.raises(ValueError, match=r"cascade link k is hose link 2 - 1 - k"):
            f2s.recover(diving_flats(f2s, 0.12, **self.DIVE))

    @pytest.mark.parametrize(("n", "attachments"), CONFIGURATIONS)
    def test_the_branch_test_is_on_exactly_where_there_are_no_prescribed_rungs(
        self, n, attachments
    ):
        """The switch itself: prescribed rungs exist iff nQ > 1, and it is
        their absence that lets M5's test stay on."""
        f2s = make(n, attachments)
        assert bool(f2s.tension_vehicles) == (len(attachments) > 1)
        # And with rungs prescribed the catenary is accepted, not refused.
        f2s.recover(static_flats(f2s))

    # ── nQ ≥ 2: the tension floor ───────────────────────────────────
    #
    # The canonical config. A derived link's tension walks through zero at
    # t ≈ 0.262 and the link folds; with no floor the map returns the folded
    # hose without complaint.
    FOLD = {"amp": 1.0, "omega": 4.5, "x_amp": 1.0, "pull": 0.3, "spread": 1.0}
    FOLD_WINDOW = np.arange(0.20, 0.34, 0.004)

    def _fold_scan(self, min_tension):
        f2s = make(3, (0, 3), min_tension=min_tension)
        previous, folds, refusals = None, 0, 0
        for t in self.FOLD_WINDOW:
            try:
                ref, _ = f2s.recover(diving_flats(f2s, t, **self.FOLD))
            except ValueError:
                refusals += 1
                previous = None
                continue
            current = np.stack(ref.link_attitudes)
            if previous is not None and np.any(np.einsum("ij,ij->i", previous, current) < 0.0):
                folds += 1
            previous = current
        return folds, refusals

    def test_without_a_floor_a_derived_link_folds_silently(self):
        """The gap, pinned so it cannot be mistaken for protection: at
        ``min_tension = 0`` the map hands back a folded hose and says nothing."""
        folds, refusals = self._fold_scan(0.0)
        assert refusals == 0
        assert folds == 1

    def test_a_physical_floor_refuses_the_fold(self):
        """0.1 N — a tenth of the prescribed ‖𝛕_0‖ — catches it."""
        folds, refusals = self._fold_scan(0.1)
        assert folds == 0
        assert refusals > 0

    def test_the_floor_message_names_the_link_and_the_threshold(self):
        f2s = make(3, (0, 3), min_tension=0.1)
        with pytest.raises(ValueError, match="below the tension floor"):
            f2s.recover(diving_flats(f2s, 0.26, **self.FOLD))
        with pytest.raises(ValueError, match=r"min_tension = 1\.000e-01 N"):
            f2s.recover(diving_flats(f2s, 0.26, **self.FOLD))

    def test_the_floor_leaves_a_taut_hose_alone(self):
        for n, attachments in CONFIGURATIONS:
            f2s = make(n, attachments, min_tension=0.1)
            _, inputs = f2s.recover(dynamic_flats(f2s, 0.21))
            assert np.min(inputs.tensions) > 0.1

    def test_a_prescribed_tension_is_exempt_from_the_floor(self):
        """``𝛕_0`` is handed over as a vector, so its magnitude may be as small
        as the planner likes — there is no branch for it to flip onto. Only the
        *derived* links are floored."""
        f2s = make(3, (0, 3), min_tension=0.1)
        _, inputs = f2s.recover(static_flats(f2s, pull=0.03, spread=0.0))
        assert inputs.tensions[0] == pytest.approx(0.03, abs=TOL_ANALYTIC)
        assert np.min(inputs.tensions[1:]) > 0.1

    def test_min_tension_defaults_to_the_shared_slack_floor(self):
        assert make(3, (0, 3)).min_tension == pytest.approx(DEFAULT_MIN_TENSION)

    def test_a_negative_floor_is_refused(self):
        with pytest.raises(ValueError, match="min_tension must be non-negative"):
            make(3, (0, 3), min_tension=-1.0)


# ─── Reduction to M5 ──────────────────────────────────────────────────


class TestReductionToM5:
    """One vehicle, at the last vertex: the hose *is* M5's flexible cable, read
    from the payload end.

    M5 hangs an ``n``-link chain off a single airframe at its head and takes the
    payload (the chain's tip) as its flat output. M6 with ``attachments = (n,)``
    is the same mechanism: its root vertex is M5's payload, its last vertex is
    M5's airframe, and no flat tension is consumed because the only vehicle is
    the mandatory one. Reversing the per-index parameters must then reproduce
    M5 exactly — thrust, moment, attitude and every link direction, up to a sign
    that is the two conventions pointing opposite ways along the same links.
    """

    @staticmethod
    def _pair(n: int, link_masses, link_lengths, mass_q=0.9):
        m5 = QuadrotorFlexCable(
            num_links=n, mass_q=mass_q, link_masses=link_masses, link_lengths=link_lengths
        )
        m6 = MultiQuadHose(
            num_links=n,
            attachments=(n,),
            vertex_masses=np.concatenate([np.asarray(link_masses)[::-1], [0.0]]),
            link_lengths=np.asarray(link_lengths)[::-1],
            masses=np.array([mass_q]),
            inertias=m5.inertia,
        )
        return m5, m6

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_hover_agrees_with_m5(self, n):
        m5, m6 = self._pair(
            n, np.full(n, 0.06) + np.array([0.0] * (n - 1) + [0.2]), np.full(n, 0.2)
        )
        order = m5.jet_order
        x = Jet(np.zeros((order + 1, 3)))
        psi = Jet(np.zeros(3))
        ref5, in5 = m5.recover(QuadrotorFlexCableFlats(x_L=x, psi=psi))
        ref6, in6 = m6.recover(MultiQuadHoseFlats(x_0=x, tensions=[], psi=[psi]))

        assert in6.thrusts[0] == pytest.approx(in5.thrust, abs=TOL_ANALYTIC)
        assert np.allclose(in6.moments[0], in5.moment, atol=TOL_ANALYTIC)
        assert np.allclose(np.asarray(ref6.orientations[0]), np.asarray(ref5.orientation))

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_a_dynamic_trajectory_agrees_with_m5(self, n):
        masses = np.linspace(0.04, 0.22, n)
        lengths = np.linspace(0.14, 0.30, n)
        m5, m6 = self._pair(n, masses, lengths)
        order = m5.jet_order
        x = Jet(sine_rows(0.41, order, [0.3, 0.3, 0.12], [1.2, 1.2, 0.9], [0.0, -1.6, 0.4]))
        psi = Jet(np.array([0.11, 0.07, -0.05]))
        ref5, in5 = m5.recover(QuadrotorFlexCableFlats(x_L=x, psi=psi))
        ref6, in6 = m6.recover(MultiQuadHoseFlats(x_0=x, tensions=[], psi=[psi]))

        assert in6.thrusts[0] == pytest.approx(in5.thrust, abs=TOL_ANALYTIC)
        assert np.allclose(in6.moments[0], in5.moment, atol=TOL_ANALYTIC)
        assert np.allclose(np.asarray(ref6.orientations[0]), np.asarray(ref5.orientation))
        assert np.allclose(
            np.asarray(ref6.angular_velocities[0]), np.asarray(ref5.angular_velocity)
        )
        # Geometry: M6 vertex j is M5 node n − j, and the directions point the
        # opposite way along the same link.
        assert np.allclose(ref6.root_position, ref5.payload_position, atol=TOL_ANALYTIC)
        assert np.allclose(ref6.vertex_positions[n], ref5.position, atol=TOL_ANALYTIC)
        for i in range(n):
            assert np.allclose(
                ref6.link_attitudes[i], -np.asarray(ref5.link_attitudes[n - 1 - i]), atol=TOL_TIGHT
            )
        assert np.allclose(np.asarray(in6.tensions), np.asarray(in5.tensions)[::-1], atol=TOL_TIGHT)


# ─── Observability of the deep rungs ──────────────────────────────────


def row_sensitivity(f2s: MultiQuadHose, row: int, epsilon: float = 1.0, **kwargs) -> float:
    """How much the last vehicle's moment moves when row ``row`` of the root
    jet is perturbed by ``epsilon``, about a static configuration."""
    base = static_flats(f2s, **kwargs)
    m0 = f2s.recover(base)[1].moments[-1]
    rows = np.asarray(base.x_0).copy()
    rows[row] += epsilon
    bumped = MultiQuadHoseFlats(x_0=Jet(rows), tensions=base.tensions, psi=base.psi)
    m1 = f2s.recover(bumped)[1].moments[-1]
    return float(np.linalg.norm(m1 - m0)) / epsilon


class TestWhichRowsAreActuallyRead:
    """``2n + 4`` is the *contract*; it is tight only at ``nQ = 1``.

    The recursion reaches ``x_0``'s deepest row by climbing the chain through
    the accelerations, two rows per rung. A **prescribed** flat tension cuts
    that climb: ``𝛕_a`` does not depend on ``x_0`` at all, so the path through
    rung ``a`` terminates and only the *direct* ``x_j = x_0 + Σ ℓ_i q_i`` term
    survives it. Every vehicle below the last therefore buys two rows back, and
    the top rows of a multi-vehicle hose's root jet are dead weight.

    The map still demands ``2n + 4`` uniformly — it is what
    :func:`~udaan.flatness.chain.chain_jet_order` contracts for, extra rows are
    free, and the alternative is a depth law that depends on the attachment
    pattern in a way no caller wants to reason about. But the fact is worth
    pinning down, because it is also what makes the ``2n + 4`` *experiment*
    below come out the way it does.
    """

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_at_one_vehicle_the_deepest_row_is_read(self, n):
        """``nQ = 1``: no tension is prescribed, nothing cuts the climb, and
        row ``2n + 4`` genuinely reaches the moment."""
        f2s = MultiQuadHose(
            num_links=n,
            attachments=(n,),
            vertex_masses=np.concatenate([np.full(n, 0.1), [0.0]]),
        )
        assert row_sensitivity(f2s, 2 * n + 4) > 0.0

    @pytest.mark.parametrize(("n", "attachments"), [(2, (0, 2)), (3, (0, 3)), (4, (2, 4))])
    def test_one_prescribed_tension_buys_two_rows_back(self, n, attachments):
        f2s = make(n, attachments)
        assert row_sensitivity(f2s, 2 * n + 2) > 0.0
        assert row_sensitivity(f2s, 2 * n + 3) == 0.0
        assert row_sensitivity(f2s, 2 * n + 4) == 0.0

    def test_two_prescribed_tensions_cut_deeper_still(self):
        f2s = make(4, (0, 2, 4))
        assert row_sensitivity(f2s, 8) > 0.0
        for row in range(9, f2s.jet_order + 1):
            assert row_sensitivity(f2s, row) == 0.0

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
    def test_the_deepest_row_read_is_four_plus_two_derived_rungs(self, n):
        """The closed form: ``d_max = 4 + 2(n − nQ + 1)``, over every attachment
        pattern. It depends on the *number* of vehicles, not on where they sit,
        and reduces to the contracted ``2n + 4`` exactly at ``nQ = 1``."""
        for sub in itertools.chain.from_iterable(
            itertools.combinations(range(n), r) for r in range(n + 1)
        ):
            attachments = (*sub, n)
            f2s = make(n, attachments)
            deepest = max(
                (row for row in range(f2s.jet_order + 1) if row_sensitivity(f2s, row) > 0.0),
                default=-1,
            )
            assert deepest == 4 + 2 * (n - len(attachments) + 1), attachments


class TestDeepRungObservability:
    """Are the deep jet rows more observable on a hose than on M5's cable?

    Step 10 found the per-rung attenuation to be governed by ``m_i ℓ_i / T_i``
    rather than by the link count. Both halves of that show up here, and the
    answer to the comparison is more interesting than "the hose is heavier":

    * At ``nQ = 1`` a hose *is* an M5 chain (see :class:`TestReductionToM5`),
      and the mass **cancels** — the tension in link ``i`` is the weight it
      carries, so ``m ℓ / T`` is scale-free in ``m``. Making a cable heavier
      does essentially nothing to the deep rungs.
    * What M6 has and M5 does not is a tension that is a *flat output*, set
      independently of the mass. Holding a heavy hose at low tension is a
      configuration no M5 chain can reach, and it is exactly the one in which
      the deep rows matter most.

    The law itself is ``ℓ^d / T^(d+1)`` with ``d = n − (nQ − 1)`` the number of
    *derived* rungs: one factor of ``ℓ/T`` per derived rung, and one further
    ``1/T`` because the last vehicle's own thrust has to react ``𝛕_{n-1}``, so
    recovering its attitude divides by a norm that grows with ``T`` too. Both
    exponents are pinned below — the ``ℓ`` one exactly, the ``T`` one
    asymptotically, since at low tension the vertex weights rather than the
    flat pull set the link tensions and there is no power law to measure.
    """

    N = 4

    @staticmethod
    def _deepest_row(f2s) -> int:
        return 4 + 2 * (f2s.num_links - f2s.num_quadrotors + 1)

    @staticmethod
    def _slope(f2s, row, low, high, key):
        a = row_sensitivity(f2s, row, **{key: low})
        b = row_sensitivity(f2s, row, **{key: high})
        return float(np.log(b / a) / np.log(high / low))

    def test_lower_tension_is_far_more_observable(self):
        f2s = make(self.N, (0, self.N))
        slack = row_sensitivity(f2s, 2 * self.N + 2, pull=1.0)
        taut = row_sensitivity(f2s, 2 * self.N + 2, pull=30.0)
        assert slack > 1e3 * taut

    @pytest.mark.parametrize(
        ("n", "attachments"), [(4, (0, 4)), (4, (0, 2, 4)), (4, (0, 1, 2, 4)), (2, (0, 2))]
    )
    def test_the_tension_exponent_is_one_more_than_the_derived_rung_count(self, n, attachments):
        """``T^-(d+1)``, not ``T^-d``.

        Measured between 1000 N and 10000 N, which is where the law is: the
        table in the theory page runs from 1 N, and its first few columns are
        not on the power law at all — there the flat pull is comparable to the
        vertex weights, so gravity sets the tensions and one link sits near
        slack. A slope read off that end (1 N → 30 N gives −2.67) means nothing.
        """
        f2s = make(n, attachments)
        derived = n - len(attachments) + 1
        slope = self._slope(f2s, self._deepest_row(f2s), 1000.0, 10000.0, "pull")
        assert slope == pytest.approx(-(derived + 1), abs=0.05)

    def test_the_tension_slope_is_still_settling_over_the_documented_range(self):
        """Guards the reason the exponent is measured out at 10³–10⁴ N: over
        1 N–30 N the slope has not settled, and it is *steeper* than its limit
        at 10–30 N before relaxing onto −4 from above."""
        f2s = make(self.N, (0, self.N))
        row = self._deepest_row(f2s)
        assert self._slope(f2s, row, 1.0, 3.0, "pull") == pytest.approx(-0.04, abs=0.02)
        assert self._slope(f2s, row, 10.0, 30.0, "pull") < -4.5
        assert self._slope(f2s, row, 100.0, 300.0, "pull") == pytest.approx(-4.12, abs=0.05)

    def test_heavier_vertices_are_more_observable_at_fixed_tension(self):
        light = row_sensitivity(make(self.N, (0, self.N), vertex_masses=0.01), 2 * self.N + 2)
        heavy = row_sensitivity(make(self.N, (0, self.N), vertex_masses=0.15), 2 * self.N + 2)
        assert heavy > 1e3 * light

    @pytest.mark.parametrize(
        ("n", "attachments"), [(4, (0, 4)), (4, (0, 2, 4)), (4, (0, 1, 2, 4)), (2, (0, 2))]
    )
    def test_longer_links_scale_as_one_factor_per_derived_rung(self, n, attachments):
        """``ℓ`` enters once per rung whose tension the recursion *derives*, and
        unlike the tension law this one is exact rather than asymptotic:
        doubling the hose multiplies the sensitivity by ``2^d`` to five decimal
        places."""
        f2s = make(n, attachments)
        derived = n - len(attachments) + 1
        row = self._deepest_row(f2s)
        base = row_sensitivity(make(n, attachments, hose_length=1.0), row)
        doubled = row_sensitivity(make(n, attachments, hose_length=2.0), row)
        assert doubled / base == pytest.approx(2.0**derived, rel=1e-5)

    def test_the_length_law_holds_over_four_doublings(self):
        f2s = make(self.N, (0, self.N))
        row = self._deepest_row(f2s)
        lengths = [0.5, 1.0, 2.0, 4.0, 8.0]
        values = [row_sensitivity(make(self.N, (0, self.N), hose_length=L), row) for L in lengths]
        for a, b in zip(values[:-1], values[1:], strict=True):
            assert b / a == pytest.approx(8.0, rel=1e-5)

    def test_at_one_vehicle_the_mass_scale_cancels(self):
        """The M5 regime: heavier links raise the tension by the same factor."""

        def sensitivity(mass):
            f2s = MultiQuadHose(
                num_links=self.N,
                attachments=(self.N,),
                vertex_masses=np.concatenate([np.full(self.N, mass), [0.0]]),
            )
            return row_sensitivity(f2s, 2 * self.N + 4)

        assert sensitivity(0.15) / sensitivity(0.01) == pytest.approx(1.0, rel=0.5)


# ─── Diagnostics — run with `pytest -s` to print the numbers ──────────


@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_print_row_sensitivity_profile(n):
    single = MultiQuadHose(
        num_links=n, attachments=(n,), vertex_masses=np.concatenate([np.full(n, 0.1), [0.0]])
    )
    pair = make(n, (0, n))
    rows = range(2 * n, 2 * n + 5)
    a = "  ".join(f"{r}:{row_sensitivity(single, r):.2e}" for r in rows)
    b = "  ".join(f"{r}:{row_sensitivity(pair, r):.2e}" for r in rows)
    print(f"\n  [n={n}] ∂‖M_last‖/∂x₀^(row)\n    nQ=1 (n,)   {a}\n    nQ=2 (0,n)  {b}")


@pytest.mark.parametrize("pull", [0.5, 1.0, 3.0, 10.0, 30.0])
def test_print_tension_vs_observability(pull):
    f2s = make(4, (0, 4))
    _, inputs = f2s.recover(static_flats(f2s, pull=pull))
    print(
        f"\n  [n=4, (0,4), pull={pull:5.2f} N] tensions "
        f"{np.array2string(inputs.tensions, precision=2)}   "
        f"∂‖M‖/∂x₀^(10) = {row_sensitivity(f2s, 10, pull=pull):.3e}"
    )
