"""Tests for the M5 flat-to-state map — quadrotor with an ``n``-link cable.

Six lines of evidence, weakest to strongest:

1. **Contract and bookkeeping** — the ``Flat2State`` triple, the flat-output
   validation, and the ``2n + 4`` jet-order law.
2. **Statics** — hover trim derived from Newton's laws without reading the map.
3. **Reduction to M2 at n = 1** — the strongest available comparison against
   independently verified code, *with the caveat below*.
4. **Instantaneous Newton-Euler residual** — the recovered state and wrench,
   pushed back through the Step-9 model's own equations of motion, must
   reproduce the flat output's accelerations. Algebraic; no integration.
5. **Finite differences of the recovered attitude** — ``Ω``, ``Ω̇`` and hence
   the moment, obtained by differentiating ``R(t)`` numerically. The half of
   the map (4) cannot see; see the note below.
6. **Open-loop round trip** — in ``test_quadrotor_flexcable_feedforward.py``.

On (3): M5 at ``n = 1`` and M2 are *not* independent implementations. Both
route through :func:`udaan.flatness.cable.tension_direction_jet` and
:func:`udaan.flatness.quadrotor._attitude_from_thrust_vector`; what differs is
only the surrounding assembly (a one-rung cascade versus M2's hand-written
chain). So the comparison checks the cascade's sign convention, gravity
handling and truncation against a long-verified assembly of the *same*
primitives — real evidence, but not an independent derivation. Item (4) is the
one that is independent of the flatness package entirely; item (5) is
independent of the map's own derivative chain.

On (4): the residual on ``quad_ang_accel`` alone is **tautological** and proves
nothing about the moment. The map sets ``M = J Ω̇ + Ω × J Ω`` and the model
computes ``Ω̇ = J⁻¹(M − Ω × J Ω)``, so feeding ``M`` back in returns the ``Ω̇``
it was built from whatever the attitude chain did. Two mutants of
``quadrotor_flexcable.py`` — dropping ``top_force[2]`` from ``d2B``, and
swapping ``x_Q[3]`` / ``x_Q[4]`` between ``dB`` and ``d2B`` — survive that
assertion untouched. What actually reaches the moment is
``TestAttitudeDerivatives``: it finite-differences the recovered attitude
``R(t)``, which depends on ``B`` alone, and so obtains ``Ω`` and ``Ω̇``
independently of the ``dB`` / ``d2B`` chain that produced them analytically.
Both mutants die there.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_TIGHT
from udaan.core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_CABLE_MASS,
    DEFAULT_NUM_CABLE_LINKS,
    DEFAULT_PAYLOAD_MASS,
    DEFAULT_QUAD_INERTIA,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from udaan.flatness import (
    Flat2State,
    Jet,
    QuadrotorCsPayload,
    QuadrotorCsPayloadFlats,
    QuadrotorFlexCable,
    QuadrotorFlexCableFlats,
    QuadrotorFlexCableInputs,
    QuadrotorFlexCableRefState,
)
from udaan.manif import SO3, TSO3, vee
from udaan.models.quadrotor_flexcable.base import QuadrotorFlexCableBase

_E3 = np.array([0.0, 0.0, 1.0])


# ─── Flat-output builders ─────────────────────────────────────────────


def sine_payload_jet(t: float, order: int, amps, omegas, phases, offset) -> Jet:
    """Payload jet of ``x_j(t) = offset_j + A_j cos(ω_j t + φ_j)``, closed form
    at any order — which is what makes the growing ``2n + 4`` reachable."""
    data = np.zeros((order + 1, 3))
    for k in range(order + 1):
        for j in range(3):
            data[k, j] = (
                amps[j] * omegas[j] ** k * np.cos(omegas[j] * t + phases[j] + k * np.pi / 2)
            )
    data[0] += np.asarray(offset, dtype=float)
    return Jet(data)


def hover_flats(f2s: QuadrotorFlexCable, position=(0.0, 0.0, 0.0)) -> QuadrotorFlexCableFlats:
    data = np.zeros((f2s.jet_order + 1, 3))
    data[0] = np.asarray(position, dtype=float)
    return QuadrotorFlexCableFlats(x_L=Jet(data), psi=Jet(np.zeros(3)))


def swing_flats(f2s: QuadrotorFlexCable, t: float = 0.6) -> QuadrotorFlexCableFlats:
    """A gentle, comfortably taut 3-D payload trajectory with moving yaw."""
    x_L = sine_payload_jet(
        t, f2s.jet_order, [0.3, 0.3, 0.12], [1.2, 1.2, 0.9], [0.0, -np.pi / 2, 0.4], [0.0, 0.0, 0.0]
    )
    psi = Jet(np.array([0.2 * np.sin(0.7 * t), 0.14 * np.cos(0.7 * t), -0.098 * np.sin(0.7 * t)]))
    return QuadrotorFlexCableFlats(x_L=x_L, psi=psi)


def _dive(num_links: int = 3, omega: float = 2.0):
    """A vertical dive ``z_L(t) = A sin(ωt)`` with ``A ω² = 2g``.

    So ``z̈_L(t) = -2g sin(ωt)``: at rest and taut at ``t = 0``, crossing the
    slack point ``z̈_L = -g`` at ``ωt = π/6``, and on the flipped branch for the
    two thirds of the half period where ``sin(ωt) > 1/2``.

    Returns ``(f2s, omega, flats_of_t)``.
    """
    f2s = QuadrotorFlexCable(num_links=num_links)
    K = f2s.jet_order
    amplitude = 2.0 * GRAVITY / omega**2

    def flats(t: float) -> QuadrotorFlexCableFlats:
        data = np.zeros((K + 1, 3))
        for k in range(K + 1):
            data[k, 2] = amplitude * omega**k * np.sin(omega * t + k * np.pi / 2)
        return QuadrotorFlexCableFlats(x_L=Jet(data), psi=Jet(np.zeros(3)))

    return f2s, omega, flats


def heavy_map(n: int) -> QuadrotorFlexCable:
    """A heavy hose: ``m_i = 1 kg``, ``ℓ_i = 1 m`` per link under a 20 kg airframe.

    The default chain divides a *fixed* 0.1 kg / 1.0 m cable into ``n`` links,
    so ``m_i ℓ_i`` falls as ``1/n²`` while the tension stays roughly constant —
    which is why the per-rung attenuation ``m_i ℓ_i / T_i`` makes the deepest
    jet row numerically invisible at the shipped defaults. Holding ``m_i`` and
    ``ℓ_i`` fixed as ``n`` grows keeps it observable, and shows that the
    attenuation is a statement about the *parameters*, not about ``n``.
    """
    return QuadrotorFlexCable(
        num_links=n,
        mass_q=20.0,
        inertia=DEFAULT_QUAD_INERTIA,
        link_masses=np.full(n, 1.0),
        link_lengths=np.full(n, 1.0),
    )


def het_map(n: int) -> QuadrotorFlexCable:
    """A deliberately heterogeneous chain: unequal masses *and* lengths."""
    masses = np.linspace(0.04, 0.34, n)
    lengths = np.linspace(0.12, 0.36, n)
    return QuadrotorFlexCable(
        num_links=n,
        mass_q=0.85,
        inertia=DEFAULT_QUAD_INERTIA,
        link_masses=masses,
        link_lengths=lengths,
    )


# ─── Contract ─────────────────────────────────────────────────────────


class TestContract:
    def test_declares_the_flat2state_triple(self):
        assert issubclass(QuadrotorFlexCable, Flat2State)
        assert QuadrotorFlexCable.Flats is QuadrotorFlexCableFlats
        assert QuadrotorFlexCable.RefState is QuadrotorFlexCableRefState
        assert QuadrotorFlexCable.Inputs is QuadrotorFlexCableInputs

    @pytest.mark.parametrize(
        "x_L,psi,message",
        [
            (Jet(np.zeros((7, 2))), Jet(np.zeros(3)), "x_L must be a 3-D jet"),
            (Jet(np.zeros((6, 3))), Jet(np.zeros(3)), r"x_L.order must be >= 6"),
            (Jet(np.zeros((7, 3))), Jet(np.zeros((3, 3))), "psi must be a scalar jet"),
            (Jet(np.zeros((7, 3))), Jet(np.zeros(2)), r"psi.order must be >= 2"),
        ],
    )
    def test_flats_validation(self, x_L, psi, message):
        with pytest.raises(ValueError, match=message):
            QuadrotorFlexCableFlats(x_L=x_L, psi=psi)

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"num_links": 0}, "num_links must be >= 1"),
            ({"mass_q": 0.0}, "mass_q must be positive"),
            ({"inertia": np.eye(2)}, "inertia must be 3×3"),
            ({"link_masses": [0.1, 0.2]}, "link_masses must be a scalar or have 3 entries"),
            ({"link_lengths": [-0.1, 0.2, 0.3]}, "link_lengths must be strictly positive"),
        ],
    )
    def test_constructor_validation(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            QuadrotorFlexCable(**{"num_links": 3, **kwargs})

    def test_defaults_mirror_the_model(self):
        """Same constructor semantics as the model: cable mass spread evenly,
        payload added to the last link, cable length divided equally."""
        f2s = QuadrotorFlexCable()
        model = QuadrotorFlexCableBase()
        assert f2s.num_links == DEFAULT_NUM_CABLE_LINKS
        assert np.allclose(f2s.link_masses, model.link_masses, atol=TOL_EXACT)
        assert np.allclose(f2s.link_lengths, model.link_lengths, atol=TOL_EXACT)
        assert f2s.cable_length == pytest.approx(DEFAULT_CABLE_LENGTH, abs=TOL_EXACT)
        assert f2s.total_mass == pytest.approx(
            DEFAULT_QUAD_MASS + DEFAULT_CABLE_MASS + DEFAULT_PAYLOAD_MASS, abs=TOL_TIGHT
        )

    @pytest.mark.parametrize("n", [1, 2, 4])
    def test_from_model_copies_the_parameters(self, n):
        model = QuadrotorFlexCableBase(num_links=n, mass=1.1)
        model.link_masses = np.linspace(0.05, 0.3, n)
        model.link_lengths = np.linspace(0.1, 0.4, n)
        f2s = QuadrotorFlexCable.from_model(model)
        assert f2s.num_links == n
        assert f2s.mass_q == pytest.approx(1.1)
        assert np.allclose(f2s.link_masses, model.link_masses, atol=TOL_EXACT)
        assert np.allclose(f2s.link_lengths, model.link_lengths, atol=TOL_EXACT)
        assert np.allclose(f2s.inertia, model.inertia, atol=TOL_EXACT)
        # A copy, not a view: mutating the model must not move the map.
        model.link_masses = np.full(n, 0.5)
        assert not np.allclose(f2s.link_masses, model.link_masses)


# ─── Jet-order law ────────────────────────────────────────────────────


class TestJetOrder:
    @pytest.mark.parametrize("n,expected", [(1, 6), (2, 8), (3, 10), (4, 12), (5, 14), (8, 20)])
    def test_law_is_two_n_plus_four(self, n, expected):
        assert QuadrotorFlexCable(num_links=n).jet_order == expected

    def test_n_equals_one_matches_the_m2_requirement(self):
        """At one link the requirement collapses onto M2's pop."""
        assert QuadrotorFlexCable(num_links=1).jet_order == 6

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_one_order_short_raises(self, n):
        """Necessity, as an error. ``n = 1`` is excluded only because its
        requirement coincides with the ``Flats`` floor, which rejects the short
        jet one step earlier — see ``TestContract.test_flats_validation``."""
        f2s = QuadrotorFlexCable(num_links=n)
        K = f2s.jet_order
        short = QuadrotorFlexCableFlats(
            x_L=Jet(np.asarray(swing_flats(f2s).x_L)[:K]), psi=Jet(np.zeros(3))
        )
        with pytest.raises(ValueError, match=rf"through its {K}-th derivative"):
            f2s.recover(short)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_exactly_2n_plus_4_succeeds(self, n):
        f2s = QuadrotorFlexCable(num_links=n)
        flats = swing_flats(f2s)
        assert flats.x_L.order == f2s.jet_order
        ref, inputs = f2s.recover(flats)
        assert inputs.thrust > 0.0
        assert np.all(inputs.tensions > 0.0)
        assert len(ref.link_attitudes) == n

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_rows_above_2n_plus_4_are_truncated(self, n):
        """A **truncation regression guard**, not a sufficiency proof.

        ``QuadrotorFlexCable.chain`` reads ``range(required + 1)`` rows and the
        cascade slices ``tip[: K + 1]``, so surplus rows are dropped before any
        arithmetic happens — every slicing implementation passes this, whatever
        it then does with the rows it kept. What it does catch is the
        truncation drifting: a mutant that keeps one row *fewer* dies here.
        Sufficiency of ``K = 2n + 4`` is the analytic content of the depth
        table in :mod:`udaan.flatness.chain`, not something a poisoning test
        can establish.
        """
        f2s = QuadrotorFlexCable(num_links=n)
        K = f2s.jet_order
        base = np.asarray(swing_flats(f2s, t=0.4).x_L)
        deep = np.vstack([base, np.full((3, 3), 1e6)])
        psi = swing_flats(f2s, t=0.4).psi
        ref_a, in_a = f2s.recover(QuadrotorFlexCableFlats(x_L=Jet(base), psi=psi))
        ref_b, in_b = f2s.recover(QuadrotorFlexCableFlats(x_L=Jet(deep), psi=psi))
        assert in_a.thrust == in_b.thrust
        assert np.array_equal(in_a.moment, in_b.moment)
        assert np.array_equal(np.asarray(ref_a.orientation), np.asarray(ref_b.orientation))
        assert np.array_equal(ref_a.position, ref_b.position)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_the_2n_plus_4_row_reaches_the_bottom_link(self, n):
        """Necessity, measured where it bites. The deepest row of the payload
        jet is the top derivative of the *last* link's direction — an O(1)
        dependence at every ``n``."""
        f2s = QuadrotorFlexCable(num_links=n)
        K = f2s.jet_order
        base = np.asarray(swing_flats(f2s, t=0.4).x_L).copy()
        psi = swing_flats(f2s, t=0.4).psi
        chain_a = f2s.chain(QuadrotorFlexCableFlats(x_L=Jet(base), psi=psi))
        bumped = base.copy()
        bumped[K] += 1.0
        chain_b = f2s.chain(QuadrotorFlexCableFlats(x_L=Jet(bumped), psi=psi))
        top = K - 2  # the last link's own top derivative
        delta = np.linalg.norm(chain_b.directions[n - 1][top] - chain_a.directions[n - 1][top])
        assert delta > 1e-3, f"row {K} did not move the bottom link's q^({top})"
        # ... and nothing shallower than that row moved.
        assert np.allclose(
            chain_b.directions[n - 1][top - 1], chain_a.directions[n - 1][top - 1], atol=TOL_EXACT
        )

    @staticmethod
    def _moment_shift(f2s: QuadrotorFlexCable, bump: float) -> float:
        """‖ΔM‖ from adding ``bump`` to row ``2n + 4`` of the payload jet."""
        K = f2s.jet_order
        flats = swing_flats(f2s, t=0.4)
        base = np.asarray(flats.x_L).copy()
        _, in_a = f2s.recover(QuadrotorFlexCableFlats(x_L=Jet(base), psi=flats.psi))
        bumped = base.copy()
        bumped[K] += bump
        _, in_b = f2s.recover(QuadrotorFlexCableFlats(x_L=Jet(bumped), psi=flats.psi))
        return float(np.linalg.norm(in_b.moment - in_a.moment))

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_the_2n_plus_4_row_reaches_the_moment(self, n):
        """Necessity at the map level, on the shipped default chain.

        The dependence survives all the way to the feedforward moment, but is
        attenuated by roughly ``m_i ℓ_i / T_i`` per rung. On the *defaults* a
        fixed 0.1 kg / 1.0 m cable is divided into ``n`` links, so ``m_i ℓ_i``
        falls as ``1/n²`` while ``T_i`` barely moves: measured ‖ΔM‖ for a unit
        bump is 2.6e-5 / 1.4e-10 / 2.9e-17 / exactly 0 at ``n`` = 1 / 3 / 5 / 7.
        That is a property of the *parameterisation*, not of ``n`` — see
        ``test_the_2n_plus_4_row_reaches_the_moment_on_a_heavy_chain``, which
        runs the same assertion at ``n = 5`` and ``7``.
        """
        assert self._moment_shift(QuadrotorFlexCable(num_links=n), 1.0) > 1e-13

    @pytest.mark.parametrize("n", [1, 3, 5, 7])
    def test_the_2n_plus_4_row_reaches_the_moment_on_a_heavy_chain(self, n):
        """The same necessity, at every ``n``, once the attenuation is undone.

        With ``m_i ℓ_i`` held fixed as the chain grows the deepest row is
        plainly observable in the feedforward moment out to ``n = 7``: the
        measured ‖ΔM‖ for a unit bump is 3.4e-5 / 5.4e-8 / 2.6e-11 / 6.0e-15 at
        ``n`` = 1 / 3 / 5 / 7. So ``K = 2n + 4`` is not over-specified and the
        effective depth is not lower — it is only *numerically* unobservable on
        the light default cable.

        The linear scaling is what separates the 1e-15 signal at ``n = 7`` from
        round-off: doubling the bump doubles ‖ΔM‖ to a part in 1e3.
        """
        f2s = heavy_map(n)
        delta = self._moment_shift(f2s, 1.0)
        assert delta > 1e-16, f"n={n}: ‖ΔM‖ = {delta:.3e}"
        assert self._moment_shift(f2s, 2.0) == pytest.approx(2.0 * delta, rel=1e-3)


# ─── Statics ──────────────────────────────────────────────────────────


class TestHoverTrim:
    """Hover trim, re-derived from Newton's laws rather than read off the map.

    A stationary payload means every body is stationary, so the cable hangs
    straight down, link ``i`` carries the weight of everything below it, the
    airframe sits one cable length above the payload, and the thrust carries
    the whole system with zero moment at ``R = I``.
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_uniform_chain(self, n):
        f2s = QuadrotorFlexCable(num_links=n)
        ref, inputs = f2s.recover(hover_flats(f2s, (0.4, -0.3, 0.5)))

        assert inputs.thrust == pytest.approx(f2s.total_mass * GRAVITY, abs=TOL_ANALYTIC)
        assert np.allclose(inputs.moment, 0.0, atol=TOL_ANALYTIC)
        assert np.allclose(np.asarray(ref.orientation), np.eye(3), atol=TOL_ANALYTIC)
        assert np.allclose(np.asarray(ref.angular_velocity), 0.0, atol=TOL_ANALYTIC)
        assert np.allclose(ref.position, [0.4, -0.3, 0.5 + f2s.cable_length], atol=TOL_ANALYTIC)
        for i in range(n):
            assert np.allclose(ref.link_attitudes[i], -_E3, atol=TOL_ANALYTIC)
            assert np.allclose(ref.link_angular_velocities[i], 0.0, atol=TOL_ANALYTIC)
            expected = GRAVITY * f2s.link_masses[i:].sum()
            assert inputs.tensions[i] == pytest.approx(expected, abs=TOL_ANALYTIC)

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_heterogeneous_chain(self, n):
        f2s = het_map(n)
        ref, inputs = f2s.recover(hover_flats(f2s))
        assert inputs.thrust == pytest.approx(f2s.total_mass * GRAVITY, abs=TOL_ANALYTIC)
        assert ref.position[2] == pytest.approx(f2s.cable_length, abs=TOL_ANALYTIC)
        # Nodes are strung out at the cumulative link lengths above the payload.
        for i in range(n):
            below = f2s.link_lengths[i + 1 :].sum()
            assert ref.link_positions[i][2] == pytest.approx(below, abs=TOL_ANALYTIC)

    def test_hover_reference_matches_the_model_reset_configuration(self):
        """The flat hover state *is* the model's spawn configuration."""
        model = QuadrotorFlexCableBase(num_links=4)
        f2s = QuadrotorFlexCable.from_model(model)
        ref, inputs = f2s.recover(hover_flats(f2s))
        model.reset(payload_position=np.zeros(3))
        assert np.allclose(ref.position, model.state.position, atol=TOL_ANALYTIC)
        for i in range(4):
            assert np.allclose(
                ref.link_attitudes[i], np.asarray(model.state.link_attitudes[i]), atol=TOL_ANALYTIC
            )
        assert np.allclose(np.asarray(model.hover_input()), [inputs.thrust, 0, 0, 0], atol=1e-9)


# ─── Reduction to M2 ──────────────────────────────────────────────────


class TestReductionToM2:
    """At ``n = 1`` M5 must reproduce M2 exactly.

    See the module docstring: the two share
    :func:`~udaan.flatness.cable.tension_direction_jet` and
    :func:`~udaan.flatness.quadrotor._attitude_from_thrust_vector`, so this is
    a check of the cascade's *assembly* — sign, gravity, truncation, ordering —
    against a long-verified assembly of the same primitives, not an independent
    derivation.
    """

    @staticmethod
    def _pair():
        f2s = QuadrotorFlexCable(
            num_links=1,
            mass_q=DEFAULT_QUAD_MASS,
            inertia=DEFAULT_QUAD_INERTIA,
            link_masses=[DEFAULT_PAYLOAD_MASS],
            link_lengths=[DEFAULT_CABLE_LENGTH],
        )
        m2 = QuadrotorCsPayload(
            mass_q=DEFAULT_QUAD_MASS,
            mass_l=DEFAULT_PAYLOAD_MASS,
            inertia=DEFAULT_QUAD_INERTIA,
            cable_length=DEFAULT_CABLE_LENGTH,
        )
        return f2s, m2

    @pytest.mark.parametrize("seed", range(6))
    def test_agrees_to_round_off_on_random_jets(self, seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(7, 3)) * 0.25
        x[2] += np.array([0.0, 0.0, -2.0])  # keep the cable taut
        psi = rng.normal(size=3) * 0.3

        f2s, m2 = self._pair()
        ref5, in5 = f2s.recover(QuadrotorFlexCableFlats(x_L=Jet(x), psi=Jet(psi)))
        ref2, in2 = m2.recover(QuadrotorCsPayloadFlats(x_L=Jet(x), psi=Jet(psi)))

        assert in5.thrust == pytest.approx(in2.thrust, abs=TOL_EXACT)
        assert np.allclose(in5.moment, in2.moment, atol=TOL_EXACT)
        assert in5.tensions[0] == pytest.approx(in2.tension, abs=TOL_EXACT)
        assert np.allclose(ref5.position, ref2.position, atol=TOL_EXACT)
        assert np.allclose(ref5.velocity, ref2.velocity, atol=TOL_EXACT)
        assert np.allclose(ref5.acceleration, ref2.acceleration, atol=TOL_EXACT)
        assert np.allclose(ref5.link_attitudes[0], ref2.cable_attitude, atol=TOL_EXACT)
        assert np.allclose(
            ref5.link_angular_velocities[0], ref2.cable_angular_velocity, atol=TOL_EXACT
        )
        assert np.allclose(
            ref5.link_angular_accelerations[0], ref2.cable_angular_acceleration, atol=TOL_EXACT
        )
        assert np.allclose(
            np.asarray(ref5.orientation), np.asarray(ref2.orientation), atol=TOL_EXACT
        )
        assert np.allclose(
            np.asarray(ref5.angular_velocity), np.asarray(ref2.angular_velocity), atol=TOL_EXACT
        )
        assert np.allclose(ref5.angular_acceleration, ref2.angular_acceleration, atol=TOL_EXACT)

    def test_splitting_one_link_into_two_conserves_the_hover_solution(self):
        """A second, independent reduction: refining the discretisation must
        not move the *static* answer, since the lumped weight is unchanged."""
        one = QuadrotorFlexCable(num_links=1, link_masses=[0.3], link_lengths=[1.0])
        _, in1 = one.recover(hover_flats(one))
        # A massless link is rejected as unphysical, so make the upper one
        # negligible and check convergence rather than exact equality.
        two = QuadrotorFlexCable(num_links=2, link_masses=[1e-9, 0.3], link_lengths=[0.5, 0.5])
        ref2, in2 = two.recover(hover_flats(two))
        assert in2.thrust == pytest.approx(in1.thrust, abs=1e-7)
        assert ref2.position[2] == pytest.approx(1.0, abs=TOL_ANALYTIC)


# ─── Newton-Euler residual: the model's own equations ─────────────────


def _seed_from_ref(model: QuadrotorFlexCableBase, ref) -> None:
    model.reset(
        payload_position=np.asarray(ref.payload_position).copy(),
        payload_velocity=np.asarray(ref.payload_velocity).copy(),
        link_attitudes=[np.asarray(q).copy() for q in ref.link_attitudes],
        link_angular_velocities=[np.asarray(w).copy() for w in ref.link_angular_velocities],
        orientation=np.asarray(ref.orientation).copy(),
        angular_velocity=np.asarray(ref.angular_velocity).copy(),
    )


class TestNewtonEulerResidual:
    """The recovered state and wrench, fed to the Step-9 model's equations of
    motion, must reproduce the flat output's own accelerations.

    This does not go through any other part of ``udaan.flatness``: the model was
    verified independently (a from-scratch Newton-Euler solver agrees to
    1.3e-13), so a non-zero residual here is a flat-map bug and nothing else. It
    is also *instantaneous* — no integrator is involved, so there is no
    round-off budget to hide in.

    **What it does not cover.** The ``quad_ang_accel`` comparison is
    algebraically empty: the map computes ``M = J Ω̇ + Ω × J Ω`` and the model
    inverts exactly that (``base.py``, ``accelerations``), so it round-trips by
    construction, and mutants of the ``(B, Ḃ, B̈) → (Ω, Ω̇, M)`` half survive it
    untouched. This class therefore covers the cascade's statics and its
    kinematics through second derivatives; the attitude half is
    ``TestAttitudeDerivatives`` below. The ``ref.angular_acceleration``
    assertion is kept as a consistency check on the packaging (that the ``Ω̇``
    handed out is the one the moment was built from), not as evidence about the
    moment.
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 7])
    @pytest.mark.parametrize("t", [0.0, 0.37, 1.4])
    def test_uniform_chain(self, n, t):
        model = QuadrotorFlexCableBase(num_links=n)
        f2s = QuadrotorFlexCable.from_model(model)
        flats = swing_flats(f2s, t=t)
        ref, inputs = f2s.recover(flats)
        _seed_from_ref(model, ref)

        acc = model.accelerations(inputs.thrust, inputs.moment)
        assert np.allclose(acc["payload_accel"], flats.x_L[2], atol=TOL_ANALYTIC)
        assert np.allclose(acc["quad_accel"], ref.acceleration, atol=TOL_ANALYTIC)
        # Tautological — see the class docstring; kept as a packaging check.
        assert np.allclose(acc["quad_ang_accel"], ref.angular_acceleration, atol=TOL_ANALYTIC)
        for i in range(n):
            assert np.allclose(
                acc["link_ang_accels"][i], ref.link_angular_accelerations[i], atol=TOL_ANALYTIC
            ), f"link {i} angular acceleration"

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_heterogeneous_chain(self, n):
        f2s = het_map(n)
        model = QuadrotorFlexCableBase(
            num_links=n,
            mass=f2s.mass_q,
            link_masses=f2s.link_masses,
            link_lengths=f2s.link_lengths,
        )
        flats = swing_flats(f2s, t=0.9)
        ref, inputs = f2s.recover(flats)
        _seed_from_ref(model, ref)

        acc = model.accelerations(inputs.thrust, inputs.moment)
        assert np.allclose(acc["payload_accel"], flats.x_L[2], atol=TOL_ANALYTIC)
        for i in range(n):
            assert np.allclose(
                acc["link_ang_accels"][i], ref.link_angular_accelerations[i], atol=TOL_ANALYTIC
            )

    @pytest.mark.parametrize("n", [1, 3])
    def test_thrust_vector_closes_the_airframe_equation(self, n):
        """``m_Q(ẍ_Q + g e₃) = f R e₃ + F_0`` — the definition the map inverts."""
        f2s = QuadrotorFlexCable(num_links=n)
        flats = swing_flats(f2s, t=0.55)
        ref, inputs = f2s.recover(flats)
        chain = f2s.chain(flats)
        lhs = f2s.mass_q * (ref.acceleration + GRAVITY * _E3)
        rhs = inputs.thrust * (np.asarray(ref.orientation) @ _E3) + chain.top_force[0]
        assert np.allclose(lhs, rhs, atol=TOL_ANALYTIC)


# ─── The attitude half: Ω, Ω̇ and M against finite differences ─────────


def _fd(func, t: float, order: int, h: float = 2e-3) -> np.ndarray:
    """``order``-th derivative of ``func`` at ``t`` by a 5-point central stencil."""
    stencils = {
        1: ([-2, -1, 1, 2], [1 / 12, -2 / 3, 2 / 3, -1 / 12]),
        2: ([-2, -1, 0, 1, 2], [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12]),
    }
    offsets, weights = stencils[order]
    total = None
    for offset, weight in zip(offsets, weights, strict=True):
        term = weight * np.asarray(func(t + offset * h), dtype=float)
        total = term if total is None else total + term
    return total / h**order


def _skew_part(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix - matrix.T)


class TestAttitudeDerivatives:
    """``Ω``, ``Ω̇`` and hence ``M`` really are derivatives of the recovered
    attitude — the half of the map the Newton-Euler residual cannot see.

    The recovered ``R(t)`` is a function of ``B(t)`` and ``ψ(t)`` **only**: no
    derivative of the thrust vector enters it. So re-evaluating the map at
    neighbouring times and differentiating ``R`` numerically produces ``Ω`` and
    ``Ω̇`` by a route that never touches the analytic ``(Ḃ, B̈)`` chain — which
    is exactly where the moment comes from. That makes it an independent check
    on ``M``, and it kills the two mutants ``TestNewtonEulerResidual`` does not:
    dropping ``top_force[2]`` from ``d2B``, and swapping ``x_Q[3]`` / ``x_Q[4]``
    between ``dB`` and ``d2B``.

    The 5-point stencil at ``h = 2e-3`` is good to ~1e-9 on these quantities;
    the mutants move them by ``O(1e-2)``.
    """

    @staticmethod
    def _rates(f2s: QuadrotorFlexCable, t0: float):
        """``(Ω, Ω̇)`` at ``t0`` from finite differences of ``R(t)``."""

        def rotation(t: float) -> np.ndarray:
            ref, _ = f2s.recover(swing_flats(f2s, t=t))
            return np.asarray(ref.orientation)

        R = rotation(t0)
        # Ω̂ = RᵀṘ, and Rᵀ R̈ = Ω̂̇ + Ω̂² with Ω̂² symmetric — so the skew part
        # of Rᵀ R̈ is Ω̂̇.  (This is just the product rule on Ω̂ = RᵀṘ.)
        omega = vee(_skew_part(R.T @ _fd(rotation, t0, 1)))
        domega = vee(_skew_part(R.T @ _fd(rotation, t0, 2)))
        return omega, domega

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    @pytest.mark.parametrize("t", [0.37, 1.4])
    def test_body_rates_are_the_attitude_derivatives(self, n, t):
        f2s = QuadrotorFlexCable(num_links=n)
        ref, _ = f2s.recover(swing_flats(f2s, t=t))
        omega_fd, domega_fd = self._rates(f2s, t)
        assert np.allclose(omega_fd, np.asarray(ref.angular_velocity), atol=1e-7)
        assert np.allclose(domega_fd, ref.angular_acceleration, atol=1e-6)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_the_moment_closes_eulers_equation_on_the_differentiated_rates(self, n):
        """``M = J Ω̇ + Ω × J Ω`` with ``Ω``, ``Ω̇`` from finite differences.

        Pushed through the model's own Euler equation rather than re-written
        here: seeded with the *numerically differentiated* body rate, the model
        must return the *numerically differentiated* angular acceleration when
        driven by the recovered moment. Nothing in that loop comes from the
        analytic thrust-vector derivatives.
        """
        t = 0.63
        model = QuadrotorFlexCableBase(num_links=n)
        f2s = QuadrotorFlexCable.from_model(model)
        flats = swing_flats(f2s, t=t)
        ref, inputs = f2s.recover(flats)
        omega_fd, domega_fd = self._rates(f2s, t)

        _seed_from_ref(model, ref)
        model.state.angular_velocity = TSO3(omega_fd)
        acc = model.accelerations(inputs.thrust, inputs.moment)
        assert np.allclose(acc["quad_ang_accel"], domega_fd, atol=1e-6)

    @pytest.mark.parametrize("n", [1, 3])
    def test_the_top_node_jets_are_the_derivatives_of_b(self, n):
        """One level lower, on the *cascade's* outputs rather than the map's.

        The ingredients ``recover`` assembles ``(Ḃ, B̈)`` from — ``x_Q^{(3)}``,
        ``x_Q^{(4)}``, ``Ḟ_0``, ``F̈_0`` — must really be time derivatives, which
        finite differences of ``B(t) = f R e₃`` check without touching the
        cascade's Leibniz recursions. (This re-forms the combination locally,
        so it constrains the chain, not ``recover``'s use of it — that is what
        the two tests above are for.)
        """
        f2s = QuadrotorFlexCable(num_links=n)
        t = 0.63

        def thrust_vector(t_: float) -> np.ndarray:
            ref, inputs = f2s.recover(swing_flats(f2s, t=t_))
            return inputs.thrust * (np.asarray(ref.orientation) @ _E3)

        chain = f2s.chain(swing_flats(f2s, t=t))
        x_Q = chain.top_position
        for order, analytic in (
            (1, f2s.mass_q * x_Q[3] - chain.top_force[1]),
            (2, f2s.mass_q * x_Q[4] - chain.top_force[2]),
        ):
            assert np.allclose(analytic, _fd(thrust_vector, t, order), atol=1e-7), (
                f"d^{order}B/dt^{order}"
            )


# ─── Recovered state shape and geometry ───────────────────────────────


class TestRefState:
    @pytest.mark.parametrize("n", [1, 3, 5])
    def test_geometry_closes(self, n):
        f2s = het_map(n)
        ref, _ = f2s.recover(swing_flats(f2s, t=0.8))
        # x_L = x_Q + Σ ℓ_i q_i, and the same for the velocities.
        offset = sum(f2s.link_lengths[i] * np.asarray(ref.link_attitudes[i]) for i in range(n))
        rate = sum(
            f2s.link_lengths[i] * np.cross(ref.link_angular_velocities[i], ref.link_attitudes[i])
            for i in range(n)
        )
        assert np.allclose(ref.position + offset, ref.payload_position, atol=TOL_ANALYTIC)
        assert np.allclose(ref.velocity + rate, ref.payload_velocity, atol=TOL_ANALYTIC)
        # The last node is the payload.
        assert np.allclose(ref.link_positions[-1], ref.payload_position, atol=TOL_ANALYTIC)
        assert np.allclose(ref.link_velocities[-1], ref.payload_velocity, atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("n", [1, 3])
    def test_link_rates_are_tangent_and_unit(self, n):
        f2s = QuadrotorFlexCable(num_links=n)
        ref, _ = f2s.recover(swing_flats(f2s, t=1.1))
        for i in range(n):
            q = np.asarray(ref.link_attitudes[i])
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=TOL_TIGHT)
            assert float(np.dot(q, ref.link_angular_velocities[i])) == pytest.approx(
                0.0, abs=TOL_TIGHT
            )

    def test_as_state_projects_onto_the_model_struct(self):
        f2s = QuadrotorFlexCable(num_links=4)
        ref, _ = f2s.recover(swing_flats(f2s, t=0.3))
        state = ref.as_state()
        assert state.num_links == 4
        assert len(state.link_attitudes) == 4
        assert np.allclose(np.asarray(state.link_attitudes[2]), ref.link_attitudes[2])
        assert np.allclose(state.payload_position, ref.payload_position)
        # The model accepts it as its own state type.
        model = QuadrotorFlexCableBase(num_links=4)
        model.state = state
        assert model.state.num_links == 4

    def test_thrust_direction_is_the_body_z_axis(self):
        f2s = QuadrotorFlexCable(num_links=3)
        flats = swing_flats(f2s, t=0.45)
        ref, inputs = f2s.recover(flats)
        R = np.asarray(ref.orientation)
        assert np.allclose(R.T @ R, np.eye(3), atol=TOL_TIGHT)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=TOL_TIGHT)
        # b1 lies in the plane containing the yaw hint.
        psi = flats.psi.value
        b1d = np.array([np.cos(psi), np.sin(psi), 0.0])
        assert float(np.dot(np.cross(R[:, 2], b1d), R[:, 0])) == pytest.approx(0.0, abs=TOL_TIGHT)


# ─── Tautness and singularities ───────────────────────────────────────


class TestTautness:
    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_free_fall_is_rejected(self, n):
        """A payload accelerating downwards at ``g`` unloads the cable
        completely; the map has no taut solution and must say so."""
        f2s = QuadrotorFlexCable(num_links=n)
        data = np.zeros((f2s.jet_order + 1, 3))
        data[2] = -GRAVITY * _E3
        flats = QuadrotorFlexCableFlats(x_L=Jet(data), psi=Jet(np.zeros(3)))
        with pytest.raises(ValueError, match=rf"link {n - 1} of {n} is slack"):
            f2s.recover(flats)

    @pytest.mark.parametrize("n", [1, 3])
    def test_the_slack_boundary_is_approached_continuously(self, n):
        """Tension falls to zero as the downward acceleration approaches ``g``
        — so ``inputs.tensions`` is a usable slack margin on the approach, not
        just a flag. It is one-sided: past the crossing the norm grows again,
        which is what the hanging-branch check is for."""
        f2s = QuadrotorFlexCable(num_links=n)
        previous = np.inf
        for fraction in (0.0, 0.5, 0.9, 0.99):
            data = np.zeros((f2s.jet_order + 1, 3))
            data[2] = -fraction * GRAVITY * _E3
            _, inputs = f2s.recover(QuadrotorFlexCableFlats(x_L=Jet(data), psi=Jet(np.zeros(3))))
            tension = inputs.tensions[-1]
            assert 0.0 < tension < previous
            previous = tension
        assert previous < 0.05 * GRAVITY * f2s.link_masses[-1] * 1.1

    def test_a_dive_through_the_singularity_is_rejected(self):
        """A vertical dive that starts at rest and accelerates through ``-g``.

        ``z_L(t) = A sin(ωt)`` with ``Aω² = 2g``, so ``z̈_L = -2g sin(ωt)``: at
        ``t = 0`` the payload is genuinely at rest and the chain hangs taut,
        ``z̈_L = -g`` is crossed at ``ωt = π/6``, and past that the recovery is
        on the flipped branch and must be refused.
        """
        f2s, omega, flats = _dive(num_links=3)

        ref, inputs = f2s.recover(flats(0.0))  # at rest: taut, hanging
        assert np.allclose(ref.link_attitudes[-1], -_E3, atol=TOL_ANALYTIC)
        assert inputs.tensions[-1] == pytest.approx(GRAVITY * f2s.link_masses[-1], abs=TOL_ANALYTIC)

        # z̈(t) = -2g sin(ωt) equals -g at ωt = π/6: the last link unloads.
        t_singular = (np.pi / 6.0) / omega
        assert flats(t_singular).x_L[2][2] == pytest.approx(-GRAVITY, abs=1e-12)
        with pytest.raises(ValueError, match="link 2 of 3 is slack"):
            f2s.recover(flats(t_singular))

        # And just past it, on the flipped branch.
        with pytest.raises(ValueError, match="link 2 of 3 has left the hanging branch"):
            f2s.recover(flats(t_singular + 1e-3))

    def test_the_dive_is_refused_at_every_sample_past_the_crossing(self):
        """The guard has to work on a *sampled* trajectory, which is the whole
        point: landing exactly on ``F_i = 0`` is measure-zero, so a slack check
        alone lets a dive through and returns a silently flipped reference.

        Before the fix this sweep produced 0 raises and 524 of 2001 samples
        (26%) with the payload recovered *above* the airframe, at tensions
        (0.005 - 2.29 N) indistinguishable from the taut side's.
        """
        f2s, omega, flats = _dive(num_links=3)
        hanging = refused = 0
        for t in np.linspace(0.0, np.pi / omega, 2001):
            try:
                ref, _ = f2s.recover(flats(t))
            except ValueError as exc:
                assert "slack" in str(exc) or "hanging branch" in str(exc)
                refused += 1
                continue
            # Whatever comes back is on the hanging branch, always.
            for q in ref.link_attitudes:
                assert float(np.dot(np.asarray(q), _E3)) <= 0.0
            hanging += 1
        # sin(ωt) > 1/2 on two thirds of the half period — every one refused.
        assert refused == pytest.approx(2001 * 2 / 3, rel=0.02)
        assert hanging > 0

    def test_beyond_free_fall_the_flipped_branch_is_refused(self):
        """Past the crossing the chain does not "go slack" — it inverts.

        M5's cable is a chain of **rigid links**, which carry compression as
        happily as tension, so ``T_i = ‖F_i‖ ≥ 0`` never changes sign and the
        only singularity of the *link* dynamics is ``‖F_i‖ = 0``. What flips is
        the direction: a payload driven downwards faster than ``g`` ends up
        *above* the airframe, at a perfectly healthy tension. A real
        (unilateral) cable would have gone slack at the crossing, so that
        branch is outside the physical validity of the cable interpretation and
        the map refuses it — ``inputs.tensions`` alone cannot, being a norm.
        """
        f2s = QuadrotorFlexCable(num_links=2)
        data = np.zeros((f2s.jet_order + 1, 3))
        data[2] = np.array([0.0, 0.0, -2.0 * GRAVITY])
        flats = QuadrotorFlexCableFlats(x_L=Jet(data), psi=Jet(np.zeros(3)))
        with pytest.raises(ValueError, match="link 1 of 2 has left the hanging branch"):
            f2s.recover(flats)

    def test_yaw_aligned_thrust_is_reported(self):
        """The yaw-hint singularity of the shared attitude primitive still
        surfaces through this map.

        Constant acceleration ``(a_x, 0, -g)`` puts the whole system in
        horizontal free fall: every link points along ``-e_1`` and the thrust
        vector is exactly ``M₀₀ a_x e_1``. With ``ψ = 0`` the yaw hint is also
        ``e_1``, and no attitude is determined.
        """
        f2s = QuadrotorFlexCable(num_links=2)
        data = np.zeros((f2s.jet_order + 1, 3))
        data[2] = np.array([1.0, 0.0, -GRAVITY])

        with pytest.raises(ValueError, match="aligned with yaw hint"):
            f2s.recover(QuadrotorFlexCableFlats(x_L=Jet(data), psi=Jet(np.zeros(3))))

        # Offsetting the yaw resolves it, exactly as the primitive's message says.
        ref, inputs = f2s.recover(
            QuadrotorFlexCableFlats(x_L=Jet(data), psi=Jet(np.array([np.pi / 2, 0.0, 0.0])))
        )
        assert inputs.thrust == pytest.approx(f2s.total_mass * 1.0, abs=TOL_ANALYTIC)
        assert np.allclose(np.asarray(ref.orientation)[:, 2], [1.0, 0.0, 0.0], atol=TOL_ANALYTIC)


# ─── Diagnostics ──────────────────────────────────────────────────────


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_print_jet_order_growth(n):
    f2s = QuadrotorFlexCable(num_links=n)
    ref, inputs = f2s.recover(swing_flats(f2s, t=0.5))
    tilt = float(SO3(np.eye(3)).config_error(ref.orientation))
    print(
        f"\n  n={n:2d}  jet order {f2s.jet_order:2d}  thrust {inputs.thrust:7.4f} N  "
        f"T_top {inputs.tensions[0]:6.4f} N  attitude tilt {tilt:.4f}"
    )
