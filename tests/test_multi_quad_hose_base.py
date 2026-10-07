r"""Tests for the M6 analytical model: ``nQ`` quadrotors carrying one flexible hose.

Six independent checks, the same six M5 carries and in the same order, plus the
genericity M6 adds (``n``, ``nQ`` and the attachment pattern are independent):

1. **Reduction to M5** at ``nQ = 1`` with the vehicle at vertex 0 — the
   strongest of them, because
   :class:`~udaan.models.quadrotor_flexcable.QuadrotorFlexCableBase` was itself
   verified against a from-scratch Newton-Euler derivation to 1.3e-13. The two
   share **no code**: M6 assembles its own ``(3+3n)`` system from its own mass
   coefficients, and this asserts the two arrive at the same numbers.
2. **Static trim** derived from first principles here, fed open loop.
3. **Energy** under zero input, converging at ``O(h)``.
4. **Momentum**: linear under gravity, angular about the system centre of mass.
5. **Conditioning** of :meth:`MultiQuadHoseBase.mass_matrix` on degenerate
   geometries, over a range of ``n``, ``nQ`` and attachment patterns.
6. **Conditional stability**: the ZOH-Euler growth exponent and where the
   usable envelope is.

The instantaneous Newton residual (item 4's undiscretised form) is the workhorse
here: ``Σ_j m̄_j ẍ_j = Σ_j F_j − M₀₀ g e₃`` and its moment about the centre of
mass hold to round-off at *any* state, so they check every coefficient of the
assembled system rather than a trajectory that happens not to diverge.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.conftest import TOL_EXACT, TOL_INTEGRATION, TOL_PHYSICAL, TOL_TIGHT
from udaan.core.defaults import (
    DEFAULT_HOSE_LENGTH,
    DEFAULT_HOSE_MASS,
    DEFAULT_NUM_HOSE_LINKS,
    DEFAULT_QUAD_MASS,
    GRAVITY,
)
from udaan.manif import S2, SO3, TS2, TSO3, rodrigues_expm
from udaan.models.multi_quad_hose import MultiQuadHoseBase, MultiQuadHoseState
from udaan.models.quadrotor_flexcable import QuadrotorFlexCableBase
from udaan.utils.hose import (
    hose_spawn_directions,
    hose_vertex_positions,
    validate_attachments,
)

E3 = np.array([0.0, 0.0, 1.0])

#: ``(num_links, attachments)`` pairs covering the shapes that differ
#: structurally: one vehicle at the head (the M5 case), one in the middle, both
#: ends, an interior pair leaving free ends at both ends, and one per vertex.
PATTERNS = [
    (1, (0,)),
    (2, (0, 2)),
    (3, (1,)),
    (4, (0, 4)),
    (4, (0, 2, 4)),
    (4, (1, 3)),
    (4, (0, 1, 2, 3, 4)),
]

#: A light team: two 0.4 kg airframes on a 0.4 kg hose weigh 11.8 N in total,
#: comfortably under the 20 N per-vehicle actuator limit, so an open-loop trim
#: that puts the whole weight on one vehicle is not clipped.
LIGHT = {"masses": 0.4, "hose_mass": 0.4}


def build(num_links=4, attachments=(0, 4), **kwargs):
    """A light model at ``(num_links, attachments)``."""
    return MultiQuadHoseBase(num_links=num_links, attachments=attachments, **{**LIGHT, **kwargs})


def random_state(model, rng, rate_scale=1.0):
    """Random link directions/rates and vehicle attitudes/rates, on-manifold."""
    directions, rates = [], []
    for _ in range(model.num_links):
        q = rng.normal(size=3)
        q /= np.linalg.norm(q)
        omega = rng.normal(size=3)
        omega -= float(omega @ q) * q
        directions.append(q)
        rates.append(rate_scale * omega)
    orientations = [
        np.asarray(rodrigues_expm(rng.normal(size=3) * 0.4)) for _ in range(model.num_quadrotors)
    ]
    body_rates = [rate_scale * rng.normal(size=3) for _ in range(model.num_quadrotors)]
    return {
        "link_attitudes": directions,
        "link_angular_velocities": rates,
        "orientations": orientations,
        "angular_velocities": body_rates,
    }


def total_energy(model) -> float:
    """Kinetic + potential energy of the whole system, in joules."""
    mbar = model.vertex_total_masses
    positions = model.vertex_positions()
    velocities = model.vertex_velocities()
    energy = 0.5 * float(np.sum(mbar * np.sum(velocities * velocities, axis=1)))
    energy += GRAVITY * float(np.sum(mbar * positions[:, 2]))
    for k in range(model.num_quadrotors):
        Omega = np.asarray(model.state.angular_velocities[k])
        energy += 0.5 * float(Omega @ (model.inertias[k] @ Omega))
    return energy


def linear_momentum(model) -> np.ndarray:
    mbar = model.vertex_total_masses
    return np.asarray(np.sum(mbar[:, None] * model.vertex_velocities(), axis=0))


def angular_momentum(model) -> np.ndarray:
    """Angular momentum about the system centre of mass, in its own frame."""
    mbar = model.vertex_total_masses
    positions = model.vertex_positions()
    velocities = model.vertex_velocities()
    centre = np.sum(mbar[:, None] * positions, axis=0) / mbar.sum()
    centre_velocity = np.sum(mbar[:, None] * velocities, axis=0) / mbar.sum()
    momentum = np.sum(
        np.cross(positions - centre, mbar[:, None] * (velocities - centre_velocity)), axis=0
    )
    for k in range(model.num_quadrotors):
        R = np.asarray(model.state.orientations[k])
        Omega = np.asarray(model.state.angular_velocities[k])
        momentum = momentum + R @ (model.inertias[k] @ Omega)
    return np.asarray(momentum)


# ── State ─────────────────────────────────────────────────────────────


class TestState:
    """State container: shape, typing, reset."""

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_state_shapes(self, n, attachments):
        state = MultiQuadHoseState(num_links=n, attachments=attachments)
        nQ = len(attachments)
        assert len(state.link_attitudes) == n
        assert len(state.link_angular_velocities) == n
        assert len(state.orientations) == nQ
        assert len(state.angular_velocities) == nQ
        assert len(state.positions) == nQ
        assert len(state.velocities) == nQ
        assert state.root_position.shape == (3,)
        assert state.root_velocity.shape == (3,)

    def test_stored_float_count_matches_the_reference(self):
        """``6 + 6n + 12 nQ``, the reference's own ``nvert*6 + nQ*12``."""
        for n, attachments in PATTERNS:
            state = MultiQuadHoseState(num_links=n, attachments=attachments)
            nQ = len(attachments)
            stored = (
                state.root_position.size
                + state.root_velocity.size
                + sum(np.asarray(q).size for q in state.link_attitudes)
                + sum(np.asarray(w).size for w in state.link_angular_velocities)
                + sum(np.asarray(R).size for R in state.orientations)
                + sum(np.asarray(O).size for O in state.angular_velocities)
            )
            assert stored == 6 + 6 * n + 12 * nQ

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_state_types(self, n, attachments):
        model = build(n, attachments)
        assert all(isinstance(q, S2) for q in model.state.link_attitudes)
        assert all(isinstance(w, TS2) for w in model.state.link_angular_velocities)
        assert all(isinstance(R, SO3) for R in model.state.orientations)
        assert all(isinstance(O, TSO3) for O in model.state.angular_velocities)

    def test_default_attachments_are_the_two_ends(self):
        state = MultiQuadHoseState(num_links=7)
        assert state.attachments == (0, 7)

    def test_zero_links_rejected(self):
        with pytest.raises(ValueError, match="num_links"):
            MultiQuadHoseBase(num_links=0)

    @pytest.mark.parametrize(
        "attachments,match",
        [
            ((), "at least one vertex"),
            ((0, 0), "strictly increasing"),
            ((4, 0), "strictly increasing"),
            ((-1, 2), r"must lie in"),
            ((0, 5), r"must lie in"),
        ],
    )
    def test_bad_attachments_rejected(self, attachments, match):
        with pytest.raises(ValueError, match=match):
            MultiQuadHoseBase(num_links=4, attachments=attachments)

    def test_reset_resets_every_field(self):
        model = build()
        rng = np.random.default_rng(1)
        model.reset(
            root_position=np.array([1.0, 2.0, 3.0]),
            root_velocity=np.array([0.4, -0.2, 0.1]),
            **random_state(model, rng),
        )
        model.t = 7.5
        model.reset()
        assert model.t == 0.0
        assert np.allclose(model.state.root_position, 0.0)
        assert np.allclose(model.state.root_velocity, 0.0)
        assert all(np.allclose(np.asarray(w), 0.0) for w in model.state.link_angular_velocities)
        assert all(np.allclose(np.asarray(R), np.eye(3)) for R in model.state.orientations)
        assert all(np.allclose(np.asarray(O), 0.0) for O in model.state.angular_velocities)
        # The link directions go back to the taut spawn, not to anything random.
        spawn = hose_spawn_directions(model.link_lengths, model.attachments, model.sag_ratio)
        assert np.allclose([np.asarray(q) for q in model.state.link_attitudes], spawn)

    def test_reset_does_not_alias_caller_arrays(self):
        model = build()
        root = np.array([1.0, 0.0, 2.0])
        model.reset(root_position=root)
        model.step(np.zeros((model.num_quadrotors, 3)))
        assert np.allclose(root, [1.0, 0.0, 2.0])

    @pytest.mark.parametrize(
        "key,size", [("link_attitudes", 4), ("link_angular_velocities", 4), ("orientations", 2)]
    )
    def test_wrong_per_index_length_rejected(self, key, size):
        model = build()
        with pytest.raises(ValueError, match=f"{key} must have {size} entries"):
            model.reset(**{key: [np.zeros(3)] * (size + 1)})

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_reset_repairs_off_manifold_per_link_input(self, n, attachments):
        """Unnormalised directions and non-tangent rates are repaired, not refused."""
        model = build(n, attachments)
        rng = np.random.default_rng(3)
        directions = [rng.normal(size=3) * rng.uniform(0.2, 5.0) for _ in range(n)]
        rates = [rng.normal(size=3) for _ in range(n)]
        model.reset(link_attitudes=directions, link_angular_velocities=rates)
        for i in range(n):
            q = np.asarray(model.state.link_attitudes[i])
            omega = np.asarray(model.state.link_angular_velocities[i])
            assert abs(np.linalg.norm(q) - 1.0) < TOL_EXACT
            assert abs(float(omega @ q)) < TOL_TIGHT
            # Direction preserved, only the length removed.
            assert np.allclose(q, directions[i] / np.linalg.norm(directions[i]))

    def test_reset_rejects_a_zero_link_direction(self):
        model = build()
        directions = [np.array([0.0, 0.0, -1.0])] * model.num_links
        directions[2] = np.zeros(3)
        with pytest.raises(ValueError, match="does not define a direction"):
            model.reset(link_attitudes=directions)

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_quad_positions_follow_the_chain(self, n, attachments):
        model = build(n, attachments)
        rng = np.random.default_rng(5)
        root = np.array([0.3, -0.2, 2.0])
        model.reset(root_position=root, **random_state(model, rng))
        vertices = hose_vertex_positions(
            root, model.link_lengths, [np.asarray(q) for q in model.state.link_attitudes]
        )
        for k, vertex in enumerate(attachments):
            assert np.allclose(model.state.positions[k], vertices[vertex], atol=TOL_EXACT)

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_the_spawn_is_taut(self, n, attachments):
        """Every link at exactly its own length — no stretch, no slack."""
        model = build(n, attachments)
        model.reset(root_position=np.array([0.0, 0.0, 2.0]))
        vertices = model.vertex_positions()
        spans = np.linalg.norm(np.diff(vertices, axis=0), axis=1)
        assert np.allclose(spans, model.link_lengths, atol=TOL_TIGHT)


# ── Parameters ────────────────────────────────────────────────────────


class TestParameters:
    """Physical parameters, their broadcasting and their guards."""

    def test_defaults(self):
        model = MultiQuadHoseBase()
        assert model.num_links == DEFAULT_NUM_HOSE_LINKS
        assert model.attachments == (0, DEFAULT_NUM_HOSE_LINKS)
        assert model.num_quadrotors == 2
        assert model.num_vertices == DEFAULT_NUM_HOSE_LINKS + 1
        assert model.hose_length == pytest.approx(DEFAULT_HOSE_LENGTH)
        assert model.vertex_masses.sum() == pytest.approx(DEFAULT_HOSE_MASS)
        assert model.masses == pytest.approx(DEFAULT_QUAD_MASS)
        assert model.total_mass == pytest.approx(DEFAULT_HOSE_MASS + 2 * DEFAULT_QUAD_MASS)

    @pytest.mark.parametrize("n", [1, 2, 4, 7])
    def test_hose_mass_and_length_split_evenly(self, n):
        model = MultiQuadHoseBase(num_links=n, attachments=(0, n), hose_mass=0.6, hose_length=1.5)
        assert np.allclose(model.link_lengths, 1.5 / n)
        assert np.allclose(model.vertex_masses, 0.6 / (n + 1))

    def test_vertex_masses_kwarg_overrides_the_convenience_split(self):
        masses = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        model = MultiQuadHoseBase(num_links=4, hose_mass=99.0, vertex_masses=masses)
        assert np.allclose(model.vertex_masses, masses)

    def test_scalar_broadcast(self):
        model = build()
        model.link_lengths = 0.7
        model.vertex_masses = 0.05
        assert np.allclose(model.link_lengths, 0.7)
        assert np.allclose(model.vertex_masses, 0.05)

    def test_heterogeneous_parameters(self):
        model = MultiQuadHoseBase(
            num_links=4,
            attachments=(0, 2, 4),
            link_lengths=[0.3, 0.5, 0.4, 0.6],
            vertex_masses=[0.02, 0.05, 0.01, 0.08, 0.03],
            masses=[0.4, 0.7, 0.5],
        )
        assert model.hose_length == pytest.approx(1.8)
        assert np.allclose(model.vertex_total_masses, [0.42, 0.05, 0.71, 0.08, 0.53])

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"link_lengths": [0.5, 0.5, 0.5]}, "4 entries"),
            ({"vertex_masses": [0.1] * 4}, "5 entries"),
            ({"masses": [0.4]}, "2 entries"),
        ],
    )
    def test_wrong_length_rejected(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            MultiQuadHoseBase(num_links=4, attachments=(0, 4), **kwargs)

    def test_non_positive_link_length_rejected(self):
        with pytest.raises(ValueError, match="link_lengths"):
            MultiQuadHoseBase(num_links=2, link_lengths=[0.5, -0.1])

    def test_negative_vertex_mass_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            MultiQuadHoseBase(num_links=2, vertex_masses=[0.1, -0.1, 0.1])

    def test_a_bare_vertex_may_be_massless_when_a_vehicle_sits_on_it(self):
        """``m_j = 0`` is fine; ``m̄_j = 0`` is not. This is the M5 reduction case."""
        model = MultiQuadHoseBase(
            num_links=2, attachments=(0,), vertex_masses=[0.0, 0.1, 0.1], masses=[0.9]
        )
        assert model.vertex_total_masses[0] == pytest.approx(0.9)

    def test_a_genuinely_massless_vertex_is_rejected(self):
        with pytest.raises(ValueError, match="strictly positive total mass"):
            MultiQuadHoseBase(
                num_links=2, attachments=(0,), vertex_masses=[0.1, 0.0, 0.1], masses=[0.9]
            )

    def test_a_rejected_mass_assignment_leaves_the_model_unchanged(self):
        model = MultiQuadHoseBase(num_links=2, attachments=(0,), vertex_masses=[0.0, 0.1, 0.1])
        before = model.vertex_masses.copy()
        with pytest.raises(ValueError, match="strictly positive total mass"):
            model.vertex_masses = [0.1, 0.0, 0.1]
        assert np.allclose(model.vertex_masses, before)

    def test_inertias_accept_diagonal_and_tensor(self):
        model = build()
        model.inertias = np.array([0.01, 0.02, 0.03])
        assert model.inertias.shape == (2, 3, 3)
        assert np.allclose(np.diagonal(model.inertias, axis1=1, axis2=2), [0.01, 0.02, 0.03])
        stack = np.stack([np.diag([0.01, 0.02, 0.03]), np.diag([0.04, 0.05, 0.06])])
        model.inertias = stack
        assert np.allclose(model.inertias, stack)

    def test_asymmetric_inertia_rejected(self):
        model = build()
        with pytest.raises(ValueError, match="symmetric"):
            model.inertias = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    def test_unknown_input_type_rejected(self):
        with pytest.raises(ValueError, match="unknown input type"):
            MultiQuadHoseBase(input="thrust_vector")

    def test_unknown_force_type_rejected(self):
        with pytest.raises(ValueError, match="unknown force type"):
            MultiQuadHoseBase(force="rotor_speeds")


class TestParameterSetterSync:
    """Every parameter setter must reach the cached controllers (M4's review)."""

    def test_masses_setter_reaches_the_position_controllers(self):
        model = build()
        model.masses = [1.5, 2.5]
        share = model.vertex_masses.sum() / 2
        assert model.position_controllers[0].mass == pytest.approx(1.5 + share)
        assert model.position_controllers[1].mass == pytest.approx(2.5 + share)

    def test_vertex_masses_setter_reaches_the_position_controllers(self):
        model = build()
        before = model.position_controllers[0].mass
        model.vertex_masses = 0.4
        assert model.position_controllers[0].mass > before

    def test_inertias_setter_reaches_the_attitude_controllers(self):
        model = build()
        model.inertias = np.array([0.05, 0.06, 0.07])
        for controller in model.attitude_controllers:
            assert np.allclose(np.diag(controller.inertia), [0.05, 0.06, 0.07])

    def test_a_controller_without_the_attribute_is_left_alone(self):
        class Bare:
            pass

        model = build()
        model.position_controllers = [Bare(), Bare()]
        model.masses = [1.0, 1.0]  # must not raise
        assert not hasattr(model.position_controllers[0], "mass")

    def test_sync_from_model_takes_precedence(self):
        seen = {}

        class Hook:
            mass = 0.0

            def sync_from_model(self, model):
                seen["mass"] = model.masses[0]

        model = build()
        hook = Hook()
        model.position_controllers = [hook, hook]
        model.masses = [3.25, 3.25]
        assert seen["mass"] == pytest.approx(3.25)
        assert hook.mass == 0.0  # the attribute was *not* written directly


# ── Dynamics ──────────────────────────────────────────────────────────


class TestReductionToM5:
    """Validation 1: ``nQ = 1`` at vertex 0 must reproduce M5 exactly.

    The two implementations share no code. M6 assembles its own ``(3+3n)``
    system from ``(M₀₀, M₀ᵢ, Mᵢⱼ)`` computed off *vertex* masses and an
    attachment indicator; M5 computes its own off *link* masses and a single
    airframe. The mapping is ``m̄ = (m_Q, m_1, …, m_n)``: M6's bare vertex mass
    at the head is zero and the airframe supplies it, which is why
    ``vertex_masses[0] = 0`` has to be legal.
    """

    @staticmethod
    def _pair(n, rng):
        lengths = rng.uniform(0.15, 0.6, n)
        link_masses = rng.uniform(0.05, 0.3, n)
        quad_mass = 0.9
        m5 = QuadrotorFlexCableBase(
            num_links=n, link_lengths=lengths, link_masses=link_masses, mass=quad_mass
        )
        m6 = MultiQuadHoseBase(
            num_links=n,
            attachments=(0,),
            link_lengths=lengths,
            vertex_masses=np.concatenate(([0.0], link_masses)),
            masses=[quad_mass],
            inertias=m5.inertia,
        )
        return m5, m6

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_mass_coefficients_agree(self, n):
        rng = np.random.default_rng(20 + n)
        m5, m6 = self._pair(n, rng)
        for a, b in zip(m5._mass_coefficients(), m6._mass_coefficients(), strict=True):
            assert np.allclose(a, b, rtol=0.0, atol=TOL_EXACT)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_every_acceleration_agrees_to_round_off(self, n):
        rng = np.random.default_rng(30 + n)
        m5, m6 = self._pair(n, rng)
        state = random_state(m6, rng)
        m5.reset(
            link_attitudes=state["link_attitudes"],
            link_angular_velocities=state["link_angular_velocities"],
            orientation=state["orientations"][0],
            angular_velocity=state["angular_velocities"][0],
            position=np.zeros(3),
        )
        m6.reset(root_position=np.zeros(3), **state)

        thrust = 6.5
        torque = np.array([0.12, -0.2, 0.05])
        a5 = m5.accelerations(thrust, torque)
        a6 = m6.accelerations([thrust], [torque])

        assert np.allclose(a5["quad_accel"], a6["root_accel"], rtol=0.0, atol=1e-13)
        assert np.allclose(a5["link_ang_accels"], a6["link_ang_accels"], rtol=0.0, atol=1e-13)
        assert np.allclose(a5["link_accels"], a6["link_accels"], rtol=0.0, atol=1e-13)
        assert np.allclose(a5["payload_accel"], a6["vertex_accels"][-1], rtol=0.0, atol=1e-13)
        assert np.allclose(a5["quad_ang_accel"], a6["quad_ang_accels"][0], rtol=0.0, atol=1e-13)

    @pytest.mark.parametrize("n", [1, 3])
    def test_mass_matrices_agree(self, n):
        rng = np.random.default_rng(40 + n)
        m5, m6 = self._pair(n, rng)
        state = random_state(m6, rng)
        m5.reset(
            link_attitudes=state["link_attitudes"],
            link_angular_velocities=state["link_angular_velocities"],
        )
        m6.reset(**state)
        assert np.allclose(m5.mass_matrix(), m6.mass_matrix(), rtol=0.0, atol=TOL_TIGHT)

    @pytest.mark.parametrize("n", [1, 3])
    def test_trajectories_agree_to_the_integrator_s_own_order(self, n):
        """The two models integrate the *same* ODE from opposite ends.

        M5 advances the payload and derives the airframe; M6 advances the root
        and derives everything below it. Both are ZOH Euler on the identical
        right-hand side, so the trajectories differ only by the ``O(h)`` the
        integrator itself carries — and the gap halves when ``h`` does, which is
        what this asserts rather than any absolute number.
        """
        gaps = []
        for dt in (1 / 100.0, 1 / 200.0, 1 / 400.0, 1 / 800.0):
            rng = np.random.default_rng(50 + n)
            m5, m6 = self._pair(n, rng)
            m5.dt = m6.dt = dt
            m5.h = m5.dt / m5._inner_loop_steps
            m6.h = m6.dt / m6._inner_loop_steps
            state = random_state(m6, rng, rate_scale=0.3)
            # ``velocity`` has to be given explicitly: M5's reset derives the
            # payload's velocity from the airframe's, and *omitting* it means
            # "the payload is at rest", which with non-zero link rates leaves
            # the airframe moving. M6's root at rest is the airframe at rest, so
            # the two are only the same initial condition when M5 is told so.
            m5.reset(
                link_attitudes=state["link_attitudes"],
                link_angular_velocities=state["link_angular_velocities"],
                orientation=state["orientations"][0],
                angular_velocity=state["angular_velocities"][0],
                position=np.array([0.0, 0.0, 3.0]),
                velocity=np.zeros(3),
            )
            m6.reset(root_position=np.array([0.0, 0.0, 3.0]), **state)
            assert np.allclose(m5.state.velocity, m6.state.root_velocity, atol=TOL_EXACT)
            m5._input_type = m5.INPUT_TYPE.WRENCH
            m6._input_type = m6.INPUT_TYPE.WRENCH
            wrench = np.array([m5.total_mass * GRAVITY, 0.0, 0.0, 0.0])
            while m6.t < 0.5:
                m5.step(wrench)
                m6.step(wrench.reshape(1, 4))
            gaps.append(float(np.max(np.abs(m5.state.position - m6.state.root_position))))
        assert gaps == sorted(gaps, reverse=True)
        # First order, so each halving of the step halves the gap.
        assert all(1.7 < a / b < 2.4 for a, b in zip(gaps, gaps[1:], strict=False))
        assert gaps[-1] < TOL_PHYSICAL


class TestDynamics:
    """The equations of motion, checked against Newton and against the trim."""

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    @pytest.mark.parametrize("loaded", [False, True])
    def test_newton_residuals_vanish(self, n, attachments, loaded):
        """Validation 4, undiscretised: force and moment balance at any state.

        ``Σ_j m̄_j ẍ_j = Σ_j F_j − M₀₀ g e₃`` and its moment about the centre of
        mass. Every internal constraint force cancels in pairs, so what is left
        checks the whole assembled system — the mass coefficients, the
        cumulative force sum ``G_i``, the gauge substitution and every sign.
        """
        rng = np.random.default_rng(60 + n + len(attachments))
        model = build(n, attachments)
        model.reset(root_position=rng.normal(size=3), **random_state(model, rng))
        nQ = model.num_quadrotors
        thrusts = rng.uniform(1.0, 8.0, nQ) if loaded else np.zeros(nQ)
        torques = rng.normal(size=(nQ, 3)) * 0.1 if loaded else np.zeros((nQ, 3))

        acc = model.accelerations(thrusts, torques)
        mbar = model.vertex_total_masses
        forces = model.vertex_forces(thrusts)
        positions = model.vertex_positions()
        centre = np.sum(mbar[:, None] * positions, axis=0) / mbar.sum()

        residual = np.sum(mbar[:, None] * acc["vertex_accels"], axis=0) - (
            forces.sum(axis=0) - mbar.sum() * GRAVITY * E3
        )
        assert np.max(np.abs(residual)) < 1e-13

        moment = np.sum(np.cross(positions - centre, mbar[:, None] * acc["vertex_accels"]), axis=0)
        moment -= np.sum(np.cross(positions - centre, forces), axis=0)
        assert np.max(np.abs(moment)) < 1e-13

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_the_mass_matrix_ignores_the_inputs(self, n, attachments):
        """Only the *right-hand side* sees the attachment pattern."""
        rng = np.random.default_rng(70)
        model = build(n, attachments)
        model.reset(**random_state(model, rng))
        nQ = model.num_quadrotors
        loaded = model._assemble(rng.uniform(1.0, 9.0, nQ))[0]
        assert np.allclose(loaded, model.mass_matrix(), rtol=0.0, atol=TOL_EXACT)

    def test_the_mass_matrix_ignores_the_attachment_pattern(self):
        """Two hoses with the same vertex masses have the same mass matrix.

        M6's matrix is M5's: it depends on ``m̄`` and the geometry only, never on
        which vertices carry a vehicle.
        """
        rng = np.random.default_rng(71)
        directions = []
        for _ in range(4):
            q = rng.normal(size=3)
            directions.append(q / np.linalg.norm(q))
        common = {"num_links": 4, "link_lengths": [0.3, 0.4, 0.5, 0.6], "masses": 0.5}
        a = MultiQuadHoseBase(attachments=(0, 4), vertex_masses=[0.1, 0.7, 0.3, 0.7, 0.1], **common)
        b = MultiQuadHoseBase(attachments=(1, 3), vertex_masses=[0.6, 0.2, 0.3, 0.2, 0.6], **common)
        # Same m̄ = (0.6, 0.7, 0.3, 0.7, 0.6) by construction.
        assert np.allclose(a.vertex_total_masses, b.vertex_total_masses)
        a.reset(link_attitudes=directions)
        b.reset(link_attitudes=directions)
        assert np.allclose(a.mass_matrix(), b.mass_matrix(), rtol=0.0, atol=TOL_EXACT)

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_hover_trim_holds_open_loop(self, n, attachments):
        """Validation 2: the derived equilibrium, fed open loop, does not move."""
        model = build(n, attachments)
        # The default trim concentrates the whole system weight on one vehicle,
        # which at five airframes exceeds the shipped 20 N limit. The claim
        # under test is about the equations of motion, not about the actuator,
        # so lift the clip rather than pick a weight that happens to fit.
        model._max_thrust = 1e3
        trim = model.hover_trim()
        model.reset(
            root_position=np.array([0.0, 0.0, 3.0]),
            link_attitudes=list(trim["link_attitudes"]),
            orientations=list(trim["orientations"]),
        )
        model._input_type = model.INPUT_TYPE.WRENCH
        root = model.state.root_position.copy()
        directions = np.array([np.asarray(q) for q in model.state.link_attitudes])
        for _ in range(200):
            model.step(trim["wrench"])
        assert np.max(np.abs(model.state.root_position - root)) < TOL_INTEGRATION
        after = np.array([np.asarray(q) for q in model.state.link_attitudes])
        assert np.max(np.abs(after - directions)) < TOL_INTEGRATION

    @pytest.mark.parametrize("n,attachments", [(4, (0, 4)), (4, (0, 2, 4)), (6, (0, 3, 6))])
    def test_an_equal_thrust_split_also_trims(self, n, attachments):
        """A spread split folds the hose but is just as much an equilibrium."""
        model = build(n, attachments)
        nQ = model.num_quadrotors
        split = np.full(nQ, model.total_mass * GRAVITY / nQ)
        trim = model.hover_trim(split)
        # The hose folds: some links point up, some down.
        vertical = trim["link_attitudes"][:, 2]
        assert np.any(vertical > 0) and np.any(vertical < 0)
        model.reset(
            root_position=np.array([0.0, 0.0, 3.0]),
            link_attitudes=list(trim["link_attitudes"]),
            orientations=list(trim["orientations"]),
        )
        model._input_type = model.INPUT_TYPE.WRENCH
        root = model.state.root_position.copy()
        for _ in range(200):
            model.step(trim["wrench"])
        assert np.max(np.abs(model.state.root_position - root)) < TOL_INTEGRATION

    def test_hover_trim_with_heterogeneous_parameters(self):
        model = MultiQuadHoseBase(
            num_links=5,
            attachments=(0, 2, 5),
            link_lengths=[0.3, 0.5, 0.4, 0.6, 0.35],
            vertex_masses=[0.02, 0.05, 0.01, 0.08, 0.03, 0.04],
            masses=[0.4, 0.7, 0.5],
        )
        trim = model.hover_trim()
        model.reset(
            root_position=np.array([0.0, 0.0, 3.0]),
            link_attitudes=list(trim["link_attitudes"]),
            orientations=list(trim["orientations"]),
        )
        model._input_type = model.INPUT_TYPE.WRENCH
        root = model.state.root_position.copy()
        for _ in range(200):
            model.step(trim["wrench"])
        assert np.max(np.abs(model.state.root_position - root)) < TOL_INTEGRATION

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_hover_thrust_is_the_system_weight(self, n, attachments):
        model = build(n, attachments)
        model._max_thrust = 1e3
        trim = model.hover_trim()
        assert trim["thrusts"].sum() == pytest.approx(model.total_mass * GRAVITY)
        assert trim["thrusts"][0] == pytest.approx(model.total_mass * GRAVITY)
        assert np.allclose(trim["orientations"], np.eye(3))

        # Newton on each sub-chain, derived without reading the model. Below the
        # topmost vehicle every link carries the weight hanging under it and
        # points down; *above* it there is nothing holding the free end up, so
        # each of those links carries the weight above it and points up — the
        # segment simply hangs back over the vehicle it is tied to.
        mbar = model.vertex_total_masses
        top = attachments[0]
        for i in range(n):
            if i < top:
                expected = float(mbar[: i + 1].sum()) * GRAVITY * E3
            else:
                expected = -float(mbar[i + 1 :].sum()) * GRAVITY * E3
            assert np.allclose(trim["tensions"][i], expected, atol=TOL_TIGHT)
            assert np.allclose(trim["link_attitudes"][i], np.sign(expected[2]) * E3)

    def test_a_split_that_zeroes_a_tension_is_refused(self):
        """One vehicle per vertex, each carrying its own weight: no vertical trim."""
        model = build(4, (0, 1, 2, 3, 4))
        mbar = model.vertex_total_masses
        with pytest.raises(ValueError, match="zero tension"):
            model.hover_trim(mbar * GRAVITY)

    def test_a_split_that_does_not_carry_the_weight_is_refused(self):
        model = build()
        with pytest.raises(ValueError, match="carry exactly the system weight"):
            model.hover_trim(np.array([1.0, 1.0]))

    def test_an_over_limit_trim_warns(self, caplog):
        model = MultiQuadHoseBase(num_links=4, attachments=(0, 4), masses=1.5, hose_mass=0.5)
        with caplog.at_level(logging.WARNING):
            model.hover_trim()
        assert "actuator limit" in caplog.text

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    @pytest.mark.parametrize("scale", [0.85, 1.15])
    def test_off_trim_thrust_accelerates_the_whole_system(self, n, attachments, scale):
        """A uniform thrust scaling moves the centre of mass the expected way."""
        model = build(n, attachments)
        trim = model.hover_trim()
        model.reset(
            root_position=np.array([0.0, 0.0, 3.0]),
            link_attitudes=list(trim["link_attitudes"]),
            orientations=list(trim["orientations"]),
        )
        wrench = trim["wrench"].copy()
        wrench[:, 0] *= scale
        acc = model.accelerations(wrench[:, 0], wrench[:, 1:])
        mbar = model.vertex_total_masses
        centre_accel = np.sum(mbar[:, None] * acc["vertex_accels"], axis=0) / mbar.sum()
        expected = (scale - 1.0) * GRAVITY
        assert centre_accel[2] == pytest.approx(expected, abs=TOL_TIGHT)

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_zero_input_is_rigid_free_fall(self, n, attachments):
        """No thrust: the whole system falls, and the shape never changes."""
        model = build(n, attachments)
        rng = np.random.default_rng(80)
        directions = [
            q / np.linalg.norm(q) for q in (rng.normal(size=3) for _ in range(model.num_links))
        ]
        model.reset(root_position=np.array([0.0, 0.0, 5.0]), link_attitudes=directions)
        model._input_type = model.INPUT_TYPE.WRENCH
        for _ in range(100):
            model.step(np.zeros((model.num_quadrotors, 4)))
        expected = 5.0 - 0.5 * GRAVITY * model.t**2
        assert model.state.root_position[2] == pytest.approx(expected, abs=TOL_INTEGRATION)
        assert np.allclose(
            [np.asarray(q) for q in model.state.link_attitudes], directions, atol=TOL_INTEGRATION
        )

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_link_directions_stay_unit_norm(self, n, attachments):
        model = build(n, attachments)
        rng = np.random.default_rng(90)
        model.reset(root_position=np.array([0.0, 0.0, 3.0]), **random_state(model, rng, 0.5))
        for _ in range(200):
            model.step(model.compute_default_input())
        for q in model.state.link_attitudes:
            assert abs(np.linalg.norm(np.asarray(q)) - 1.0) < TOL_TIGHT

    @pytest.mark.parametrize("n,attachments", [(2, (0, 2)), (4, (0, 4))])
    def test_link_rates_stay_tangent(self, n, attachments):
        """``q_i · ω_i`` is exactly zero in the continuous dynamics, and now in ZOH too.

        The gauge row ``M_ii I`` makes ``ω̇_i · q_i = 0`` an equation of the
        solve, so any residual was a pure ZOH-Euler artefact. It used to fall
        with the substep but never vanish — the worst here reached 1.0 at
        ``n = 4``, ``dt = 1/200``, the largest of any model in the suite,
        because a hose has the most links to accumulate it over.
        :func:`udaan.manif.S2.zoh_step` reprojects the updated rate onto
        ``T_q S²``, which removes a gauge component the dynamics never read —
        the same fix M2-M5 now share — so the invariant holds outright.
        """
        drifts = []
        for dt in (1 / 100.0, 1 / 200.0, 1 / 400.0):
            model = build(n, attachments, dt=dt)
            rng = np.random.default_rng(100)
            model.reset(root_position=np.array([0.0, 0.0, 3.0]), **random_state(model, rng, 0.5))
            for _ in range(int(1.0 / dt)):
                model.step(model.compute_default_input())
            drifts.append(
                max(
                    abs(
                        float(
                            np.asarray(model.state.link_angular_velocities[i])
                            @ np.asarray(model.state.link_attitudes[i])
                        )
                    )
                    for i in range(n)
                )
            )
        assert max(drifts) < 1e-12, f"q·ω drifted to {max(drifts):.3e}"

    @pytest.mark.slow
    @pytest.mark.parametrize("n,attachments", [(2, (0, 2)), (4, (0, 4)), (4, (0, 2, 4))])
    def test_energy_conserved_under_zero_input(self, n, attachments):
        """Validation 3: the O(h¹) energy drift this integrator family carries."""
        errors = []
        for dt in (1 / 200.0, 1 / 400.0, 1 / 800.0):
            model = build(n, attachments, dt=dt)
            rng = np.random.default_rng(110)
            model.reset(root_position=np.array([0.0, 0.0, 3.0]), **random_state(model, rng, 0.5))
            model._input_type = model.INPUT_TYPE.WRENCH
            start = total_energy(model)
            while model.t < 1.0:
                model.step(np.zeros((model.num_quadrotors, 4)))
            errors.append(abs(total_energy(model) - start))
        assert errors[0] > errors[1] > errors[2]
        # First order: halving the step halves the accumulated error.
        assert 1.7 < errors[0] / errors[1] < 2.3
        assert 1.7 < errors[1] / errors[2] < 2.3

    @pytest.mark.slow
    @pytest.mark.parametrize("n,attachments", [(2, (0, 2)), (4, (0, 4)), (4, (1, 3))])
    def test_momentum_conservation_under_gravity(self, n, attachments):
        """Validation 4, integrated: linear momentum tracks ``-M₀₀ g t e₃``.

        Angular momentum about the centre of mass is conserved outright: with
        no thrust the only external force is gravity, whose moment about the
        centre of mass is zero.
        """
        model = build(n, attachments, dt=1 / 800.0)
        rng = np.random.default_rng(120)
        model.reset(root_position=np.array([0.0, 0.0, 3.0]), **random_state(model, rng, 0.5))
        model._input_type = model.INPUT_TYPE.WRENCH
        mbar_total = model.vertex_total_masses.sum()
        p0 = linear_momentum(model)
        L0 = angular_momentum(model)
        while model.t < 1.0:
            model.step(np.zeros((model.num_quadrotors, 4)))
        expected = p0 - mbar_total * GRAVITY * model.t * E3
        assert np.max(np.abs(linear_momentum(model) - expected)) < 1e-4
        assert np.max(np.abs(angular_momentum(model) - L0)) < 1e-4


class TestInputPipeline:
    """``_repackage_input`` and the three input types."""

    def test_acceleration_is_the_default(self):
        model = build()
        wrench = model._repackage_input(np.tile(model.total_mass * GRAVITY * E3 / 2, (2, 1)))
        assert wrench.shape == (2, 4)
        assert np.all(wrench[:, 0] > 0.0)

    def test_wrench_passes_through_and_clips(self):
        model = build(input="wrench")
        wrench = model._repackage_input(np.array([[100.0, 0.0, 0.0, 0.0], [3.0, 9.0, 0.0, 0.0]]))
        assert wrench[0, 0] == pytest.approx(model._max_thrust)
        assert wrench[1, 1] == pytest.approx(model._max_torque[0])

    def test_prop_forces_round_trip(self):
        model = build(input="prop_forces")
        props = np.full((2, 4), 2.0)
        wrench = model._repackage_input(props)
        assert np.allclose(wrench[:, 0], 8.0)

    def test_flat_input_is_reshaped(self):
        model = build()
        flat = np.tile(model.total_mass * GRAVITY * E3 / 2, 2)
        assert np.allclose(model._repackage_input(flat), model._repackage_input(flat.reshape(2, 3)))


class TestConditioning:
    """Validation 5: :meth:`mass_matrix` on degenerate geometries."""

    @staticmethod
    def _assert_definite(model, cond_limit=1e8):
        matrix = model.mass_matrix()
        assert np.allclose(matrix, matrix.T, rtol=0.0, atol=TOL_EXACT)
        assert np.min(np.linalg.eigvalsh(matrix)) > 0.0
        assert np.linalg.cond(matrix) < cond_limit

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_the_spawn_shape_is_positive_definite(self, n, attachments):
        model = build(n, attachments)
        model.reset()
        self._assert_definite(model)

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_all_links_parallel_do_not_make_it_singular(self, n, attachments):
        model = build(n, attachments)
        model.reset(link_attitudes=[np.array([0.0, 0.0, -1.0])] * n)
        self._assert_definite(model)

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_random_configurations_are_positive_definite(self, n, attachments, rng):
        model = build(n, attachments)
        for _ in range(20):
            directions = []
            for _ in range(n):
                q = rng.normal(size=3)
                directions.append(q / np.linalg.norm(q))
            model.reset(link_attitudes=directions)
            self._assert_definite(model)

    def test_very_unequal_masses_and_lengths_stay_definite(self):
        """A 2000:1 length spread and a 50000:1 mass spread stay invertible.

        The condition number follows the *square* of the length spread — the
        ``M_ij`` blocks carry ``l_i l_j`` — so 1.9e9 here is the geometry, not
        a defect: the smallest eigenvalue is 1e-8 and strictly positive, which
        is the claim.
        """
        model = MultiQuadHoseBase(
            num_links=4,
            attachments=(0, 4),
            link_lengths=[1e-3, 2.0, 1e-2, 1.5],
            vertex_masses=[1e-4, 5.0, 1e-4, 3.0, 1e-4],
            masses=[0.4, 0.4],
        )
        model.reset()
        self._assert_definite(model, cond_limit=1e10)

    def test_a_massless_bare_vertex_under_a_vehicle_stays_definite(self):
        model = MultiQuadHoseBase(
            num_links=3,
            attachments=(0, 3),
            vertex_masses=[0.0, 0.05, 0.05, 0.0],
            masses=[0.5, 0.5],
        )
        model.reset()
        self._assert_definite(model)


class TestStability:
    """Validation 6: the ZOH-Euler envelope."""

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_frequencies_have_one_rigid_translation_mode(self, n, attachments):
        model = build(n, attachments)
        frequencies = model.natural_frequencies()
        assert frequencies.shape == (n + 1,)
        assert frequencies[0] == pytest.approx(0.0, abs=1e-8)
        assert np.all(frequencies[1:] > 0.0)

    def test_single_link_matches_the_derived_pendulum_frequency(self):
        r"""``\omega^2 = (g/\ell)(1 + \bar m_1/\bar m_0)``, derived by hand."""
        model = MultiQuadHoseBase(
            num_links=1,
            attachments=(0,),
            link_lengths=[0.8],
            vertex_masses=[0.3, 0.2],
            masses=[0.7],
        )
        mbar = model.vertex_total_masses
        expected = np.sqrt(GRAVITY / 0.8 * (1.0 + mbar[1] / mbar[0]))
        assert model.natural_frequencies()[1] == pytest.approx(expected, rel=1e-10)

    def test_growth_rate_is_the_euler_amplification_exponent(self):
        model = build()
        omega = float(model.natural_frequencies().max())
        assert model.stability_growth_rate() == pytest.approx(omega**2 * model.h / 2.0)

    def test_the_default_configuration_is_inside_the_envelope(self):
        assert MultiQuadHoseBase().stability_growth_rate() < 1.0

    def test_a_long_hose_warns_out_of_the_constructor(self, caplog):
        with caplog.at_level(logging.WARNING):
            MultiQuadHoseBase(num_links=12, attachments=(0, 12))
        assert "stiff for the ZOH-Euler integrator" in caplog.text

    def test_more_links_are_stiffer(self):
        rates = [
            MultiQuadHoseBase(num_links=n, attachments=(0, n)).stability_growth_rate()
            for n in (1, 2, 4, 8)
        ]
        assert rates == sorted(rates)


class TestSimulate:
    """The station-keeping cascade :meth:`simulate` drives."""

    @pytest.mark.parametrize("n,attachments", [(2, (0, 2)), (4, (0, 4)), (4, (0, 2, 4))])
    def test_station_keeping_settles_with_the_expected_sag_offset(self, n, attachments):
        """The loop is stable, and its steady-state offset is the hose's pull.

        The spawn is taut but *not* an equilibrium, so the hose sags and pulls
        the vehicles towards each other. The station-keeping PD has no integral
        action and feeds forward only an equal share of the hose's weight, so it
        settles at a genuine offset rather than on the target: roughly
        ``|horizontal pull| / k_p``, ~0.15 m at these gains. What this asserts is
        that the run *settles* — bounded offset, essentially zero residual speed
        — not that it has no offset, which would be false.
        """
        model = build(n, attachments)
        model.simulate(tf=5.0, root_position=np.array([0.0, 0.0, 2.0]))
        targets = model._station_targets
        for k in range(model.num_quadrotors):
            assert np.max(np.abs(model.state.positions[k] - targets[k])) < 0.3
            assert np.max(np.abs(model.state.velocities[k])) < 0.1
        assert np.all(np.isfinite(model.state.root_position))

    def test_simulate_resets_first(self):
        model = build()
        model.t = 9.0
        model.simulate(tf=0.05, root_position=np.array([0.0, 0.0, 2.0]))
        assert model.t == pytest.approx(0.05, abs=model.h)


# ── Spawn geometry ────────────────────────────────────────────────────


class TestSpawnGeometry:
    """:mod:`udaan.utils.hose` — the taut spawn and its feasibility guard."""

    @pytest.mark.parametrize("n,attachments", PATTERNS)
    def test_directions_are_unit(self, n, attachments):
        directions = hose_spawn_directions(np.full(n, 0.5), attachments)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=TOL_EXACT)

    @pytest.mark.parametrize("sag", [0.5, 0.8, 0.95])
    def test_each_segment_spans_the_requested_fraction_and_stays_level(self, sag):
        lengths = np.array([0.3, 0.5, 0.4, 0.6, 0.35, 0.45])
        attachments = (0, 3, 6)
        directions = hose_spawn_directions(lengths, attachments, sag)
        vertices = hose_vertex_positions(np.zeros(3), lengths, directions)
        for first, second in zip(attachments, attachments[1:], strict=False):
            span = vertices[second] - vertices[first]
            assert span[2] == pytest.approx(0.0, abs=TOL_TIGHT)
            assert span[1] == pytest.approx(0.0, abs=TOL_TIGHT)
            assert span[0] == pytest.approx(sag * float(lengths[first:second].sum()), abs=TOL_TIGHT)

    def test_a_one_link_segment_is_laid_horizontally(self):
        directions = hose_spawn_directions([0.4, 0.4], (0, 1, 2), sag_ratio=0.5)
        assert np.allclose(directions, [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    def test_free_ends_hang_from_their_nearest_vehicle(self):
        directions = hose_spawn_directions(np.full(5, 0.4), (1, 4))
        # Above the first vehicle the chain hangs *below* it, so q points up.
        assert np.allclose(directions[0], [0.0, 0.0, 1.0])
        assert np.allclose(directions[4], [0.0, 0.0, -1.0])

    def test_a_sag_ratio_of_one_is_a_straight_segment(self):
        directions = hose_spawn_directions(np.full(4, 0.5), (0, 4), sag_ratio=1.0)
        assert np.allclose(directions, [1.0, 0.0, 0.0])

    def test_an_unreachable_fold_is_refused(self):
        """A three-link segment cannot fold below one link's length."""
        with pytest.raises(ValueError, match="cannot fold down"):
            hose_spawn_directions(np.full(3, 0.5), (0, 3), sag_ratio=0.2)

    @pytest.mark.parametrize("sag", [0.0, -0.1, 1.5])
    def test_an_out_of_range_sag_ratio_is_refused(self, sag):
        with pytest.raises(ValueError, match="sag_ratio"):
            hose_spawn_directions(np.full(4, 0.5), (0, 4), sag_ratio=sag)

    def test_validate_attachments_returns_ints(self):
        assert validate_attachments(np.array([0, 3]), 3) == (0, 3)
