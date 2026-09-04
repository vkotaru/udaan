"""Tests for MultiQuadCsPayloadBase — state, invariants and the coupled EOM.

The dynamics under test are, with ``u_i = f_i R_i e₃``::

    (m_L I + Σ m_i q_i q_iᵀ)(a_L + g e₃) = Σ (q_i q_iᵀ u_i - m_i l_i ‖q̇_i‖² q_i)
    ω̇_i = (1/l_i) q̂_i (a_L + g e₃) - (1/(m_i l_i)) q̂_i (I - q_i q_iᵀ) u_i

The equilibrium checks below (statically-trimmed hover, free fall) pin signs
and the parallel/perpendicular thrust split, but they are *blind* to ``l_i``:
at hover both cable terms cancel for any length, and free fall zeroes the
right-hand side outright. The per-cable ``1/l_i`` and the centripetal
coefficient therefore need checks that leave equilibrium —
``test_newton_euler_residuals_pin_the_per_cable_length`` and
``test_single_quadrotor_reproduces_the_trusted_m2_model``.
"""

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_INTEGRATION, TOL_PHYSICAL, TOL_TIGHT
from udaan.core.defaults import DEFAULT_FORMATION_RADIUS, GRAVITY
from udaan.flatness import RingTensionSplit
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.multi_quad_cspayload import (
    MultiQuadCsPayloadBase,
    MultiQuadCsPayloadState,
    cable_state_from_geometry,
    formation_cable_attitudes,
)
from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadBase

QUAD_COUNTS = [1, 2, 3, 4, 6]

_E3 = np.array([0.0, 0.0, 1.0])


def hover_trim(mdl):
    """Static hover input for the model's current cable configuration.

    Force balance on quadrotor ``i`` with cable tension ``T_i`` pulling it
    toward the load along ``q_i`` gives ``u_i = m_i g e₃ - T_i q_i``; the
    payload needs ``Σ T_i q_i = -m_L g e₃`` (least-norm solution, since a team
    of three or more cables is over-actuated).

    Returns ``(thrusts, orientations, tensions)``.
    """
    nQ = mdl.num_quadrotors
    Q = np.column_stack([np.asarray(q) for q in mdl.state.cable_attitudes])
    tensions = np.linalg.pinv(Q) @ (-mdl.payload_mass * GRAVITY * _E3)

    thrusts = np.zeros(nQ)
    orientations = []
    for i in range(nQ):
        u = mdl.masses[i] * GRAVITY * _E3 - tensions[i] * np.asarray(mdl.state.cable_attitudes[i])
        thrusts[i] = np.linalg.norm(u)
        orientations.append(SO3.from_two_vectors(u / thrusts[i], np.array([1.0, 0.0, 0.0])))
    return thrusts, orientations, tensions


def total_energy(mdl):
    """Kinetic + potential energy of payload, quadrotors and their attitudes."""
    energy = (
        0.5
        * mdl.payload_mass
        * float(np.dot(mdl.state.payload_velocity, mdl.state.payload_velocity))
    )
    energy += mdl.payload_mass * GRAVITY * mdl.state.payload_position[2]
    for i in range(mdl.num_quadrotors):
        v = mdl.state.velocities[i]
        Omega = np.asarray(mdl.state.angular_velocities[i])
        energy += 0.5 * mdl.masses[i] * float(np.dot(v, v))
        energy += 0.5 * float(Omega @ mdl.inertias[i] @ Omega)
        energy += mdl.masses[i] * GRAVITY * mdl.state.positions[i][2]
    return energy


class TestState:
    """State shape and manifold typing across quadrotor counts."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_state_shapes(self, nQ):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        mdl.reset()
        assert isinstance(mdl.state, MultiQuadCsPayloadState)
        assert mdl.num_quadrotors == nQ
        for entries in (
            mdl.state.positions,
            mdl.state.velocities,
            mdl.state.orientations,
            mdl.state.angular_velocities,
            mdl.state.cable_attitudes,
            mdl.state.cable_angular_velocities,
        ):
            assert len(entries) == nQ
        assert mdl.state.payload_position.shape == (3,)
        assert mdl.state.payload_velocity.shape == (3,)

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_state_types(self, nQ):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        mdl.reset()
        for i in range(nQ):
            assert isinstance(mdl.state.orientations[i], SO3)
            assert isinstance(mdl.state.angular_velocities[i], TSO3)
            assert isinstance(mdl.state.cable_attitudes[i], S2)
            assert isinstance(mdl.state.cable_angular_velocities[i], TS2)

    def test_reset_resets_cables(self):
        # Regression: the retired model's State.reset() reset the quadrotors and
        # the load but silently left the cables at their previous values.
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.reset()
        pristine = [np.asarray(q).copy() for q in mdl.state.cable_attitudes]

        mdl.state.cable_attitudes = [S2(np.array([1.0, 0.0, 0.0]))] * 3
        mdl.state.cable_angular_velocities = [TS2(np.array([0.0, 0.0, 1.0]))] * 3
        mdl.reset()

        for i in range(3):
            np.testing.assert_allclose(
                np.asarray(mdl.state.cable_attitudes[i]), pristine[i], atol=TOL_TIGHT
            )
            np.testing.assert_allclose(
                np.asarray(mdl.state.cable_angular_velocities[i]), np.zeros(3), atol=TOL_TIGHT
            )

    def test_reset_does_not_alias_caller_arrays(self):
        start = np.array([1.0, 1.0, 2.0])
        snapshot = start.copy()
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.reset(payload_position=start)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        for _ in range(50):
            mdl.step(np.zeros((3, 4)))
        assert not np.allclose(mdl.state.payload_position, snapshot)
        np.testing.assert_array_equal(start, snapshot)

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_quad_positions_follow_cable_geometry(self, nQ):
        # x_Q_i = x_L - l_i q_i, i.e. q points from the quadrotor to the load.
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.3, -0.4, 1.0]))
        for i in range(nQ):
            expected = mdl.state.payload_position - mdl.cable_lengths[i] * np.asarray(
                mdl.state.cable_attitudes[i]
            )
            np.testing.assert_allclose(mdl.state.positions[i], expected, atol=TOL_ANALYTIC)
            assert mdl.state.positions[i][2] > mdl.state.payload_position[2]


class TestParameters:
    """Per-index parameters with scalar broadcast."""

    def test_scalar_broadcast(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=4)
        mdl.masses = 1.25
        mdl.cable_lengths = 1.5
        np.testing.assert_allclose(mdl.masses, np.full(4, 1.25))
        np.testing.assert_allclose(mdl.cable_lengths, np.full(4, 1.5))

    def test_per_index_values(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.masses = [0.8, 1.0, 1.2]
        mdl.cable_lengths = [1.0, 1.2, 1.4]
        np.testing.assert_allclose(mdl.masses, [0.8, 1.0, 1.2])
        np.testing.assert_allclose(mdl.cable_lengths, [1.0, 1.2, 1.4])
        mdl.reset()
        for i in range(3):
            expected = mdl.state.payload_position - mdl.cable_lengths[i] * np.asarray(
                mdl.state.cable_attitudes[i]
            )
            np.testing.assert_allclose(mdl.state.positions[i], expected, atol=TOL_ANALYTIC)

    def test_inertia_shapes(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.inertias = np.array([0.01, 0.02, 0.03])
        assert mdl.inertias.shape == (3, 3, 3)
        np.testing.assert_allclose(np.diag(mdl.inertias[1]), [0.01, 0.02, 0.03])

        per_index = np.stack([np.diag([0.01, 0.02, 0.03])] * 3)
        per_index[2] *= 2.0
        mdl.inertias = per_index
        np.testing.assert_allclose(np.diag(mdl.inertias[2]), [0.02, 0.04, 0.06])
        np.testing.assert_allclose(
            mdl.inertia_invs[2], np.linalg.inv(mdl.inertias[2]), atol=TOL_ANALYTIC
        )

        # Any other rank is rejected outright.
        with pytest.raises(ValueError):
            mdl.inertias = np.zeros((3, 3, 3, 3))

    def test_asymmetric_inertia_rejected(self):
        # An (nQ, 3) stack of per-quadrotor diagonals has the same shape as a
        # single (3, 3) tensor at nQ == 3, and used to be silently tiled to
        # every quadrotor as one asymmetric tensor. Requiring symmetry is both
        # physically right and what disambiguates the two.
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        stack_of_diagonals = np.array(
            [
                [0.010, 0.010, 0.020],
                [0.011, 0.012, 0.021],
                [0.012, 0.013, 0.022],
            ]
        )
        with pytest.raises(ValueError, match="symmetric"):
            mdl.inertias = stack_of_diagonals

        # An asymmetric entry inside an (nQ, 3, 3) stack is rejected too.
        bad_stack = np.stack([np.diag([0.01, 0.02, 0.03])] * 3)
        bad_stack[1, 0, 1] = 0.005
        with pytest.raises(ValueError, match="symmetric"):
            mdl.inertias = bad_stack

        # Symmetric but indefinite is rejected as well.
        with pytest.raises(ValueError, match="positive definite"):
            mdl.inertias = np.diag([0.01, -0.02, 0.03])
        with pytest.raises(ValueError, match="positive definite"):
            mdl.inertias = np.array([0.01, 0.0, 0.03])

        # Off-diagonal but symmetric products of inertia stay legal.
        symmetric = np.array([[0.02, 0.001, 0.0], [0.001, 0.02, 0.0], [0.0, 0.0, 0.04]])
        mdl.inertias = symmetric
        np.testing.assert_allclose(mdl.inertias[2], symmetric, atol=TOL_ANALYTIC)

    def test_wrong_length_rejected(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        with pytest.raises(ValueError):
            mdl.masses = [1.0, 1.0]
        with pytest.raises(ValueError):
            MultiQuadCsPayloadBase(num_quadrotors=0)

    def test_non_positive_formation_radius_rejected(self):
        """Both doors to the ring radius validate, and on the same domain as
        :class:`~udaan.flatness.RingTensionSplit`.

        ``formation_cable_attitudes`` guards ``lengths <= radius``, which a
        *negative* radius passes: before validation, ``mdl.formation_radius =
        -3.0`` followed by ``reset()`` produced ``cable_attitudes = [3, 0, nan]``
        behind nothing louder than a ``RuntimeWarning`` from ``sqrt``.
        """
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        for bad in (-3.0, 0.0):
            with pytest.raises(ValueError, match="formation_radius must be strictly positive"):
                mdl.formation_radius = bad
            with pytest.raises(ValueError, match="formation_radius must be strictly positive"):
                MultiQuadCsPayloadBase(num_quadrotors=3, formation_radius=bad)
        # The valid radius survived the rejected assignments.
        assert mdl.formation_radius == pytest.approx(DEFAULT_FORMATION_RADIUS)

        # And the ring re-derivation inside udaan.flatness agrees on the domain.
        with pytest.raises(ValueError, match="strictly positive"):
            RingTensionSplit(3, mdl.payload_mass, mdl.masses, mdl.cable_lengths, radius=-3.0)

    def test_cable_shorter_than_the_radius_is_rejected_at_construction(self):
        """A cable that cannot reach its ring station now fails at ``__init__``.

        It used to survive construction and surface at ``reset()`` as a ``nan``
        cable direction out of ``sqrt(l² - r²)``. Since the M3 payload cascade
        became the default controller, ``_init_default_controllers()`` builds a
        :class:`~udaan.flatness.RingTensionSplit` inside ``__init__`` and its
        guard refuses first — earlier and with a message that names both
        numbers. Pinned here because it is a change in *when* the model
        refuses, which no other test would notice.
        """
        with pytest.raises(
            ValueError, match=r"formation radius 0\.5 must be strictly below every cable length"
        ):
            MultiQuadCsPayloadBase(num_quadrotors=3, cable_lengths=0.4, formation_radius=0.5)

        # One quadrotor hovers on the axis, so the radius does not constrain it.
        MultiQuadCsPayloadBase(num_quadrotors=1, cable_lengths=0.4, formation_radius=0.5)

    @pytest.mark.slow
    def test_parameter_setters_reach_the_payload_cascade(self):
        """Assigning a parameter must be the same as constructing with it.

        The payload cascade caches every mass and builds its tension-split
        policy at construction, so a setter that wrote only ``self._…`` left
        the closed loop splitting the *old* load's weight: setting
        ``payload_mass = 3.0`` on a default model used to leave the payload
        1.64 m below its setpoint, with nothing else to show for it. The
        attitude and station-keeping loops cache ``J_i`` and the mass share the
        same way.
        """
        target = np.array([1.0, -0.5, 1.5])

        def run(assign_after_construction):
            if assign_after_construction:
                mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
                mdl.payload_mass = 3.0
                mdl.cable_lengths = 1.4
            else:
                mdl = MultiQuadCsPayloadBase(num_quadrotors=3, payload_mass=3.0, cable_lengths=1.4)
            mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
            mdl.simulate(tf=12.0, payload_position=np.array([0.0, 0.0, 1.0]))
            return mdl

        assigned, constructed = run(True), run(False)
        np.testing.assert_allclose(
            assigned.state.payload_position, constructed.state.payload_position, atol=1e-12
        )
        # ... and it is the *right* trajectory: the payload reaches the target,
        # where the stale cascade left it 1.6 m short.
        assert float(np.linalg.norm(assigned.state.payload_position - target)) < TOL_PHYSICAL

        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.masses = 1.5
        assert mdl.position_controllers[0].mass == pytest.approx(
            1.5 + mdl.payload_mass / mdl.num_quadrotors
        )
        mdl.inertias = np.diag([0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            mdl.attitude_controllers[0].inertia, np.diag([0.1, 0.2, 0.3]), atol=TOL_TIGHT
        )

    def test_an_explicit_tension_policy_is_not_rebuilt_by_a_setter(self):
        """The caller's policy is the caller's to keep current."""
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        policy = RingTensionSplit(
            num_quadrotors=3,
            payload_mass=0.2,
            masses=np.full(3, 0.9),
            cable_lengths=np.full(3, 1.0),
        )
        mdl.payload_controller.tension_policy = policy
        mdl.payload_mass = 3.0
        assert mdl.payload_controller.tension_policy is policy

    def test_reset_stays_finite_at_a_valid_radius(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3, formation_radius=0.35)
        mdl.reset()
        for q in mdl.state.cable_attitudes:
            assert np.all(np.isfinite(np.asarray(q)))


class TestCableStateFromGeometry:
    """Regression cover for the dropped 1/length factor."""

    def test_omega_carries_the_inverse_length_factor(self):
        # The retired model computed omega = q x (v_L - v_Q) with no 1/l, which
        # is only right for a unit-length cable. Use l = 2.5 so the two formulas
        # differ by a factor of 2.5 and the old one cannot pass.
        length = 2.5
        q_true = np.array([0.0, 0.6, -0.8])
        omega_true = np.array([0.4, -0.3, 0.0])
        omega_true = omega_true - np.dot(omega_true, q_true) * q_true  # make it tangent

        quad_position = np.array([1.0, -2.0, 3.0])
        quad_velocity = np.array([0.2, 0.1, -0.05])
        payload_position = quad_position + length * q_true
        # q̇ = ω × q, and x_L - x_Q = l q  ⇒  v_L - v_Q = l (ω × q)
        payload_velocity = quad_velocity + length * np.cross(omega_true, q_true)

        q, omega, recovered_length = cable_state_from_geometry(
            payload_position, payload_velocity, quad_position, quad_velocity
        )

        assert recovered_length == pytest.approx(length, abs=TOL_ANALYTIC)
        np.testing.assert_allclose(np.asarray(q), q_true, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(np.asarray(omega), omega_true, atol=TOL_ANALYTIC)

        # The old formula (no 1/length) is off by exactly the cable length.
        old_formula = np.cross(np.asarray(q), payload_velocity - quad_velocity)
        np.testing.assert_allclose(old_formula, length * omega_true, atol=TOL_ANALYTIC)
        assert not np.allclose(old_formula, omega_true, atol=1e-3)

    def test_coincident_endpoints_rejected(self):
        p = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            cable_state_from_geometry(p, np.zeros(3), p.copy(), np.zeros(3))


class TestDynamics:
    """Invariants and equilibria of the coupled EOM."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_hover_trim_holds(self, nQ):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        start = np.array([0.0, 0.0, 1.0])
        mdl.reset(payload_position=start)
        thrusts, orientations, tensions = hover_trim(mdl)
        assert np.all(tensions > 0.0), "cables must be in tension at hover"

        mdl.reset(payload_position=start, orientations=orientations)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        wrench = np.zeros((nQ, 4))
        wrench[:, 0] = thrusts
        for _ in range(1000):  # 5 s
            mdl.step(wrench)

        np.testing.assert_allclose(mdl.state.payload_position, start, atol=TOL_INTEGRATION)
        np.testing.assert_allclose(mdl.state.payload_velocity, np.zeros(3), atol=TOL_INTEGRATION)
        for i in range(nQ):
            np.testing.assert_allclose(
                np.asarray(mdl.state.cable_angular_velocities[i]),
                np.zeros(3),
                atol=TOL_INTEGRATION,
            )

    def test_hover_trim_holds_with_heterogeneous_quadrotors(self):
        # Distinct m_i and l_i per index. NOTE: this does *not* pin the 1/l_i
        # factor — at hover ω̇_i = (g/l_i) q̂_i e₃ - (m_i g/(m_i l_i)) q̂_i e₃ ≡ 0
        # for *any* l_i, and the centripetal term vanishes with ω_i = 0, so the
        # trim is structurally blind to cable length. The per-cable 1/l_i is
        # covered by test_newton_euler_residuals_pin_the_per_cable_length.
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.masses = [0.7, 1.0, 1.3]
        mdl.cable_lengths = [0.9, 1.1, 1.4]
        mdl.payload_mass = 0.35

        start = np.array([0.2, -0.3, 1.5])
        mdl.reset(payload_position=start)
        thrusts, orientations, tensions = hover_trim(mdl)
        assert np.all(tensions > 0.0)
        assert len(set(np.round(thrusts, 6))) == 3, "trim should differ per quadrotor"

        mdl.reset(payload_position=start, orientations=orientations)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        wrench = np.zeros((3, 4))
        wrench[:, 0] = thrusts
        for _ in range(1000):
            mdl.step(wrench)

        np.testing.assert_allclose(mdl.state.payload_position, start, atol=TOL_INTEGRATION)
        np.testing.assert_allclose(mdl.state.payload_velocity, np.zeros(3), atol=TOL_INTEGRATION)

    def test_newton_euler_residuals_pin_the_per_cable_length(self):
        """Swinging, heterogeneous cables checked against Newton, not the EOM.

        The model's own formulas are never evaluated here. Instead the
        integrated state is fed through two statements that hold for *any*
        cable-suspended assembly:

        1. **Cable force is along the cable.** For quadrotor ``i``,
           ``r_i := m_i a_{Q_i} - u_i + m_i g e₃`` is the force the cable
           exerts on it, so ``q_i × r_i = 0``.
        2. **Newton's third law.** The cable forces are internal, so
           ``Σ_i r_i = -m_L (a_L + g e₃)``.

        Both use only kinematics — ``a_{Q_i} = a_L - l_i q̈_i`` with
        ``q̈_i = ω̇_i × q_i + ω_i × q̇_i`` — evaluated with the *true* ``l_i``.
        The residual of (1) works out to ``(1 - l_i/l̃_i)(m_i a_g^⊥ - u_i^⊥)``
        when the model integrates with a wrong length ``l̃_i``, so a ``1/l_i``
        that is hoisted out of the loop, or dropped entirely, shows up
        directly. Heterogeneous ``l_i``, cables displaced from vertical with
        nonzero ``ω_i``, and nonzero thrust are all required: with any of them
        removed the residual degenerates to zero for every ``l̃_i``.

        Confirmed to fail against the 1e-4 bound for both mutations: 4.46 N
        for ``l_i = self._cable_lengths[0]`` and 1.77 N for a dropped
        ``1/l_i``, against 1.3e-6 N unmutated.
        """
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3, input="wrench", dt=1e-4, inner_loop_steps=1)
        mdl.masses = [0.7, 1.0, 1.3]
        mdl.cable_lengths = [0.7, 1.0, 1.4]  # heterogeneous: 1/l_i cannot hide
        mdl.payload_mass = 0.35

        cables = formation_cable_attitudes(3, mdl.cable_lengths)
        swing_axis = np.array([0.4, -0.3, 0.2])
        orientations = [
            SO3.from_two_vectors(b3 / np.linalg.norm(b3), np.array([1.0, 0.0, 0.0]))
            for b3 in (
                np.array([0.2, 0.1, 1.0]),
                np.array([-0.15, 0.25, 1.0]),
                np.array([0.05, -0.2, 1.0]),
            )
        ]
        mdl.reset(
            payload_position=np.array([0.0, 0.0, 5.0]),
            cable_attitudes=cables,
            cable_angular_velocities=[np.cross(np.asarray(q), swing_axis) for q in cables],
            orientations=orientations,
        )

        thrusts = np.array([9.0, 11.0, 13.0])  # off-trim, so u_i^⊥ ≠ 0
        wrench = np.zeros((3, 4))
        wrench[:, 0] = thrusts
        h = mdl.h

        worst_along_cable = 0.0
        worst_momentum = 0.0
        discriminant = 0.0
        for _ in range(200):
            q0 = [np.asarray(mdl.state.cable_attitudes[i]).copy() for i in range(3)]
            w0 = [np.asarray(mdl.state.cable_angular_velocities[i]).copy() for i in range(3)]
            vL0 = mdl.state.payload_velocity.copy()
            u = [thrusts[i] * (mdl.state.orientations[i] @ _E3) for i in range(3)]

            mdl.step(wrench)

            # ZOH-Euler updates velocities as v += a·h, so these are the exact
            # accelerations the model used — no finite-difference truncation.
            accel_payload = (mdl.state.payload_velocity - vL0) / h
            forces = np.zeros(3)
            for i in range(3):
                domega = (np.asarray(mdl.state.cable_angular_velocities[i]) - w0[i]) / h
                dq = np.cross(w0[i], q0[i])
                ddq = np.cross(domega, q0[i]) + np.cross(w0[i], dq)
                accel_quad = accel_payload - mdl.cable_lengths[i] * ddq
                r = mdl.masses[i] * accel_quad - u[i] + mdl.masses[i] * GRAVITY * _E3
                forces += r
                worst_along_cable = max(
                    worst_along_cable, float(np.linalg.norm(np.cross(q0[i], r)))
                )
                # Scale of the term a wrong l_i would multiply by (1 - l_i/l̃_i).
                discriminant = max(
                    discriminant,
                    float(
                        np.linalg.norm(
                            np.cross(q0[i], mdl.masses[i] * (accel_payload + GRAVITY * _E3) - u[i])
                        )
                    ),
                )
            worst_momentum = max(
                worst_momentum,
                float(np.linalg.norm(forces + mdl.payload_mass * (accel_payload + GRAVITY * _E3))),
            )

        assert discriminant > 1.0, "trajectory is too close to trim to discriminate 1/l_i"
        assert worst_along_cable < 1e-4, f"cable force off-axis by {worst_along_cable:.3e} N"
        assert worst_momentum < 1e-4, f"momentum balance off by {worst_momentum:.3e} N"

    def test_single_quadrotor_reproduces_the_trusted_m2_model(self):
        """At nQ = 1 this model must be bit-for-bit ``QuadrotorCsPayloadBase``.

        With one cable the mass matrix inverse acts along ``q``, so the payload
        equation collapses to M2's scalar form and ``q̂_i (a_L + g e₃)`` vanishes
        — the two integrators are algebraically identical. Any divergence is a
        transcription error in this model.

        This is what pins the centripetal coefficient at ``‖q̇_i‖²``: M2 uses
        ``np.dot(dq, dq)``, and swapping it for ``‖ω_i‖²`` here leaves the two
        agreeing to 1e-15 in ``q``/``ω`` while the payload position separates by
        2.1e-05 m over 2 s, because ZOH-Euler lets ``q·ω`` drift off zero.
        """
        cable = np.array([0.3, -0.2, -0.9327379053088815])
        cable /= np.linalg.norm(cable)
        rate = np.cross(cable, np.array([0.4, 0.25, -0.3]))
        b3 = np.array([0.1, -0.05, 1.0])
        orientation = SO3.from_two_vectors(b3 / np.linalg.norm(b3), np.array([1.0, 0.0, 0.0]))
        body_rate = np.array([0.2, -0.3, 0.15])
        payload_position = np.array([0.4, -0.2, 3.0])
        payload_velocity = np.array([0.1, 0.2, -0.05])
        wrench = np.array([11.0, 0.05, -0.03, 0.01])

        m2 = QuadrotorCsPayloadBase(input="wrench")
        m2.reset(
            payload_position=payload_position.copy(),
            payload_velocity=payload_velocity.copy(),
            cable_attitude=S2(cable),
            cable_angular_velocity=TS2(rate),
            orientation=orientation,
            angular_velocity=TSO3(body_rate),
        )
        m3 = MultiQuadCsPayloadBase(num_quadrotors=1, input="wrench")
        m3.masses = m2.mass
        m3.inertias = m2.inertia
        m3.cable_lengths = m2.cable_length
        m3.payload_mass = m2.payload_mass
        m3.reset(
            payload_position=payload_position.copy(),
            payload_velocity=payload_velocity.copy(),
            cable_attitudes=[cable],
            cable_angular_velocities=[rate],
            orientations=[np.asarray(orientation)],
            angular_velocities=[body_rate],
        )

        for _ in range(400):  # 2 s
            m2.step(wrench)
            m3.step(wrench.reshape(1, 4))

        np.testing.assert_allclose(
            m3.state.payload_position, m2.state.payload_position, atol=TOL_TIGHT
        )
        np.testing.assert_allclose(
            m3.state.payload_velocity, m2.state.payload_velocity, atol=TOL_TIGHT
        )
        np.testing.assert_allclose(m3.state.positions[0], m2.state.position, atol=TOL_TIGHT)
        np.testing.assert_allclose(
            np.asarray(m3.state.cable_attitudes[0]),
            np.asarray(m2.state.cable_attitude),
            atol=TOL_TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(m3.state.cable_angular_velocities[0]),
            np.asarray(m2.state.cable_angular_velocity),
            atol=TOL_TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(m3.state.orientations[0]), np.asarray(m2.state.orientation), atol=TOL_TIGHT
        )

    def test_gyroscopic_term_precesses_the_body_rates(self):
        """Torque-free asymmetric bodies must precess *and* conserve ``R J Ω``.

        Dropping ``Ω × J Ω`` from ``Ω̇ = J⁻¹(M - Ω × J Ω)`` leaves ``Ω``
        constant under zero torque, so the body-frame rate alone is not enough:
        the discriminating statement is conservation of angular momentum in the
        **world** frame, ``R_i J_i Ω_i``, which the frozen-``Ω`` mutant violates
        at O(1) because ``R_i`` keeps rotating. Every other test in this file
        holds ``Ω = 0`` or runs through MuJoCo, which never calls ``_zoh``.

        Confirmed with the ``Ω × J Ω`` term removed: ``Ω`` freezes exactly
        (the vacuity guard trips first) and the world momentum drifts 6.1e-1
        against the 1e-3 bound, versus 1.4e-4 unmutated.
        """
        mdl = MultiQuadCsPayloadBase(num_quadrotors=2, input="wrench", dt=1.0 / 2000.0)
        mdl.inertias = np.stack([np.diag([0.010, 0.020, 0.035])] * 2)  # J1 < J2 < J3
        Omega0 = np.array([2.0, -3.0, 1.5])  # not an eigenvector of J
        mdl.reset(
            payload_position=np.array([0.0, 0.0, 5.0]),
            angular_velocities=[Omega0.copy(), Omega0.copy()],
        )

        def world_momentum(i):
            return np.asarray(mdl.state.orientations[i]) @ (
                mdl.inertias[i] @ np.asarray(mdl.state.angular_velocities[i])
            )

        L0 = [world_momentum(i) for i in range(2)]
        worst = 0.0
        for _ in range(1000):  # 0.5 s
            mdl.step(np.zeros((2, 4)))
            for i in range(2):
                worst = max(
                    worst, float(np.linalg.norm(world_momentum(i) - L0[i]) / np.linalg.norm(L0[i]))
                )

        # Ω must actually move — otherwise the conservation check is vacuous.
        drift = np.linalg.norm(np.asarray(mdl.state.angular_velocities[0]) - Omega0)
        assert drift > 1.0, f"body rates barely precessed ({drift:.3e}); test is vacuous"
        # Residual is the O(h) ZOH splitting error (R stepped before Ω).
        assert worst < 1e-3, f"world angular momentum drifted by {worst:.3e}"

    @pytest.mark.parametrize("nQ", [2, 3])
    def test_hover_thrust_sums_to_system_weight(self, nQ):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        thrusts, orientations, _ = hover_trim(mdl)
        vertical = sum(thrusts[i] * float((orientations[i] @ _E3)[2]) for i in range(nQ))
        weight = (float(np.sum(mdl.masses)) + mdl.payload_mass) * GRAVITY
        assert vertical == pytest.approx(weight, abs=TOL_ANALYTIC)

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_zero_input_is_rigid_free_fall(self, nQ):
        # With u_i = 0 and ω_i = 0 the RHS of both the payload and the cable
        # equations vanishes, so the whole assembly falls at exactly -g e3.
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        start = np.array([0.0, 0.0, 20.0])
        mdl.reset(payload_position=start)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        cables0 = [np.asarray(q).copy() for q in mdl.state.cable_attitudes]

        for _ in range(200):  # 1 s
            mdl.step(np.zeros((nQ, 4)))

        expected = start - np.array([0.0, 0.0, 0.5 * GRAVITY * mdl.t**2])
        np.testing.assert_allclose(mdl.state.payload_position, expected, atol=TOL_INTEGRATION)
        for i in range(nQ):
            np.testing.assert_allclose(
                np.asarray(mdl.state.cable_attitudes[i]), cables0[i], atol=TOL_TIGHT
            )

    @pytest.mark.parametrize("nQ", [2, 3, 4])
    def test_cable_directions_stay_unit_norm(self, nQ):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        mdl.reset(
            payload_position=np.array([0.0, 0.0, 10.0]),
            cable_angular_velocities=self._swinging_rates(nQ, mdl),
        )
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        worst = 0.0
        for _ in range(400):  # 2 s, ~1600 substeps
            mdl.step(np.zeros((nQ, 4)))
            for i in range(nQ):
                norm = float(np.linalg.norm(np.asarray(mdl.state.cable_attitudes[i])))
                worst = max(worst, abs(norm - 1.0))
        # The S2 step is a true rotation, so this is pure float round-off.
        assert worst < TOL_TIGHT, f"‖q‖ drifted by {worst:.3e}"

    @pytest.mark.parametrize("nQ", [2, 3, 4])
    def test_cable_rates_stay_tangent(self, nQ):
        """``q·ω`` stays at machine precision, at any step size.

        ``ω̇`` is exactly tangent, so ``q·ω`` is conserved in continuous time.
        The old ZOH update stepped ``q`` first and then added ``h·ω̇(q_old)``,
        which left an ``O(h)`` residual along the *new* ``q``; it accumulated
        at first order, reaching 1.03e-2 here at ``nQ = 3``, ``dt = 1/200``
        over 2 s of thrusting flight (7.2e-2 at ``nQ = 2``), and 2.1e-1 in M5
        and 1.0 in M6.

        :func:`udaan.manif.zoh_step` now projects the updated rate back onto
        ``T_q S²`` at the ``q`` the same substep produced. Hence a **flat**
        bound rather than a convergence rate: there is no longer an order to
        measure, and this test now pins the invariant itself.

        The projected component is gauge in the *continuous* dynamics —
        ``q̇ = ω × q`` annihilates it — but not in the discrete step, because
        ``q_next = expm(hat(ω h)) q`` reads the whole of ``ω``. Every model's
        closed-loop trajectory therefore moves a little: 9.97e-05 m of payload
        position over a 3 s brisk spiral here, against 2.81e-04 m (M2),
        5.83e-04 m (M4) and 4.58e-05 m (M5). Every tolerance in this suite
        absorbs that, and no agreement or conservation assertion changes. See
        :func:`udaan.manif.zoh_step` for the full accounting, including the
        second path M5 and M6 have through their LQR error states.
        """
        worst = {dt: self._worst_tangency_drift(nQ, dt) for dt in (1.0 / 200.0, 1.0 / 400.0)}
        for dt, value in worst.items():
            assert value < 1e-12, f"q·ω drifted to {value:.3e} at dt={dt}"

    @staticmethod
    def _worst_tangency_drift(nQ, dt):
        """Worst ``|q·ω|`` over 2 s of swinging, thrusting, tilted flight."""
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ, input="wrench", dt=dt)
        tilt = 0.35
        orientations = [
            SO3.from_two_vectors(
                np.array(
                    [
                        np.sin(tilt) * np.cos(2.1 * i),
                        np.sin(tilt) * np.sin(2.1 * i),
                        np.cos(tilt),
                    ]
                ),
                np.array([1.0, 0.0, 0.0]),
            )
            for i in range(nQ)
        ]
        mdl.reset(
            payload_position=np.array([0.0, 0.0, 10.0]),
            cable_angular_velocities=TestDynamics._swinging_rates(nQ, mdl),
            orientations=orientations,
        )
        wrench = np.zeros((nQ, 4))
        wrench[:, 0] = 1.5 * (mdl.masses + mdl.payload_mass / nQ) * GRAVITY
        worst = 0.0
        for _ in range(int(round(2.0 / dt))):
            mdl.step(wrench)
            for i in range(nQ):
                q = np.asarray(mdl.state.cable_attitudes[i])
                omega = np.asarray(mdl.state.cable_angular_velocities[i])
                worst = max(worst, abs(float(np.dot(q, omega))))
        return worst

    @pytest.mark.parametrize("nQ", [2, 3])
    def test_energy_conserved_under_zero_input(self, nQ):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        rates = self._swinging_rates(nQ, mdl)
        mdl.reset(payload_position=np.array([0.0, 0.0, 10.0]), cable_angular_velocities=rates)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        e0 = total_energy(mdl)
        for _ in range(400):  # 2 s
            mdl.step(np.zeros((nQ, 4)))
        drift = abs(total_energy(mdl) - e0) / abs(e0)
        assert drift < 1e-4, f"energy drifted by {drift:.3e} with no input"

    @staticmethod
    def _swinging_rates(nQ, mdl):
        """Tangent cable rates that make every cable swing."""
        axis = np.array([0.3, -0.2, 0.15])
        return [
            np.cross(np.asarray(q), axis) for q in formation_cable_attitudes(nQ, mdl.cable_lengths)
        ]


class TestInputPipeline:
    """InputType / ForceType repackaging."""

    def test_acceleration_input_produces_wrench_stack(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.reset()
        thrust_vectors = np.tile(np.array([0.0, 0.0, 10.0]), (3, 1))
        wrench = mdl._repackage_input(thrust_vectors)
        assert wrench.shape == (3, 4)
        np.testing.assert_allclose(wrench[:, 0], 10.0, atol=TOL_ANALYTIC)

    def test_wrench_input_is_clipped(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=2, input="wrench")
        mdl.reset()
        wrench = mdl._repackage_input(np.array([[1e3, 0.0, 0.0, 0.0], [-1e3, 0.0, 0.0, 0.0]]))
        assert wrench[0, 0] == pytest.approx(mdl._max_thrust)
        assert wrench[1, 0] == pytest.approx(mdl._min_thrust)

    def test_prop_force_roundtrip(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        wrench = np.array([[8.0, 0.1, -0.2, 0.05]] * 3)
        props = mdl._wrench_to_propforces(wrench)
        assert props.shape == (3, 4)
        np.testing.assert_allclose(mdl._propforces_to_wrench(props), wrench, atol=TOL_ANALYTIC)

    def test_flat_input_is_reshaped(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=2)
        mdl.reset()
        flat = np.tile(np.array([0.0, 0.0, 9.0]), 2)
        np.testing.assert_allclose(
            mdl._repackage_input(flat),
            mdl._repackage_input(flat.reshape(2, 3)),
            atol=TOL_ANALYTIC,
        )


@pytest.mark.integration
class TestStationKeeping:
    """The per-quadrotor station-keeping PDs hold the formation.

    These drive ``compute_formation_input()``, *not* ``simulate()``: since the
    M3 payload controller landed, ``simulate()`` runs the payload cascade
    instead (see ``tests/test_multi_quad_cspayload_control.py``). The
    station-keeping PDs remain the baseline every quadrotor falls back on when
    it is told only where to hover, so they keep their own test.
    """

    @pytest.mark.parametrize("nQ", [2, 3, 4])
    def test_station_keeping_pds_hold_formation(self, nQ):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=nQ)
        start = np.array([0.0, 0.0, 1.0])
        mdl.reset(payload_position=start)
        while mdl.t < 5.0:
            mdl.step(mdl.compute_formation_input())
        assert np.linalg.norm(mdl.state.payload_position - start) < 0.15
        assert np.linalg.norm(mdl.state.payload_velocity) < 0.15


class TestParameterValidation:
    """Non-physical parameters are refused at assignment, not inside ``_zoh``.

    The payload mass matrix is ``m_L I + Σ_i m_i q_i q_iᵀ`` (module docstring).
    The rank-one sum alone is singular whenever the cables are parallel — which
    is every ``nQ = 1`` model, and the ring in the limit of a small radius — so
    ``m_L ≤ 0`` was not a merely inaccurate model: ``np.linalg.solve`` raised a
    ``LinAlgError`` from inside the integrator, several frames from the
    assignment that caused it. ``m_L < 0`` is worse, because the solve
    *succeeds* and returns an acceleration for an indefinite system.
    """

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_non_positive_payload_mass_is_refused(self, value):
        with pytest.raises(ValueError, match="strictly positive"):
            MultiQuadCsPayloadBase(num_quadrotors=3, payload_mass=value)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_non_positive_payload_mass_is_refused_on_assignment(self, value):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        with pytest.raises(ValueError, match="strictly positive"):
            mdl.payload_mass = value

    @pytest.mark.parametrize("name", ["masses", "cable_lengths"])
    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_non_positive_per_index_parameters_are_refused(self, name, value):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        with pytest.raises(ValueError, match="strictly positive"):
            setattr(mdl, name, value)

    def test_one_bad_entry_in_a_per_index_array_is_enough(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        with pytest.raises(ValueError, match="strictly positive"):
            mdl.masses = [1.0, -0.5, 1.0]

    @pytest.mark.parametrize("value", [np.nan, np.inf])
    def test_non_finite_payload_mass_is_refused(self, value):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=2)
        with pytest.raises(ValueError, match="finite"):
            mdl.payload_mass = value

    def test_a_refused_assignment_leaves_the_model_untouched(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        before = mdl.masses.copy()
        with pytest.raises(ValueError):
            mdl.masses = [1.0, -0.5, 1.0]
        np.testing.assert_allclose(mdl.masses, before)

    def test_the_singular_solve_this_prevents(self):
        """Show the failure mode, so the guard cannot be removed as cosmetic.

        With ``m_L = 0`` and one quadrotor the mass matrix is ``m q qᵀ``, rank
        one, and the solve that ``_zoh`` performs every substep is singular.
        """
        mdl = MultiQuadCsPayloadBase(num_quadrotors=1)
        q = np.asarray(mdl.state.cable_attitudes[0])
        singular = 0.0 * np.eye(3) + mdl.masses[0] * np.outer(q, q)
        assert np.linalg.matrix_rank(singular) == 1
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.solve(singular, np.ones(3))

    def test_valid_assignments_still_work(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.payload_mass = 0.4
        mdl.masses = [0.9, 1.0, 1.1]
        mdl.cable_lengths = 1.2
        assert mdl.payload_mass == 0.4
        np.testing.assert_allclose(mdl.masses, [0.9, 1.0, 1.1])
        np.testing.assert_allclose(mdl.cable_lengths, [1.2, 1.2, 1.2])


class TestYawSetpoint:
    r"""``yaw_setpoint`` is what makes the recovered ``R_i`` carry a heading.

    The feedforward path hardcoded ψ ≡ 0 before, which made the attitude
    composition in
    :meth:`~udaan.control.quadrotor.GeometricAttitudeController.compute` a near
    no-op and left the capability unreachable through the public API. The
    default is still ψ ≡ 0, so binding nothing changes nothing.
    """

    @staticmethod
    def _flat_map(mdl):
        from udaan.flatness import MultiQuadCsPayload

        return MultiQuadCsPayload.from_model(
            mdl, tension_policy=mdl.payload_controller.tension_policy
        )

    @staticmethod
    def _hover_flat_fn(mdl):
        """A ``get_full``-shaped callable holding the formation at its spawn."""
        target = np.asarray(mdl.state.payload_position, dtype=float).copy()
        return lambda t: (target, *(np.zeros(3) for _ in range(4)))

    def test_the_default_is_unbound(self):
        assert MultiQuadCsPayloadBase(num_quadrotors=3).yaw_setpoint is None

    def test_a_non_callable_is_refused(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        with pytest.raises(TypeError):
            mdl.yaw_setpoint = 0.5

    def test_it_reaches_the_recovered_attitudes(self):
        """Bound and unbound must not recover the same ``R_i``."""
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.reset(payload_position=np.array([0.0, 0.0, 2.0]))
        flat_map, flat_fn = self._flat_map(mdl), self._hover_flat_fn(mdl)

        unbound = mdl._desired_attitudes(flat_map, flat_fn, 0.0)
        mdl.yaw_setpoint = lambda t: (0.9, 0.0, 0.0)
        bound = mdl._desired_attitudes(flat_map, flat_fn, 0.0)

        assert unbound is not None and bound is not None
        for (R_a, _, _), (R_b, _, _) in zip(unbound, bound, strict=True):
            assert np.linalg.norm(np.asarray(R_a) - np.asarray(R_b)) > 0.5

    def test_unbound_is_exactly_psi_zero(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.reset(payload_position=np.array([0.0, 0.0, 2.0]))
        flat_map, flat_fn = self._flat_map(mdl), self._hover_flat_fn(mdl)

        unbound = mdl._desired_attitudes(flat_map, flat_fn, 0.0)
        mdl.yaw_setpoint = lambda t: (0.0, 0.0, 0.0)
        explicit = mdl._desired_attitudes(flat_map, flat_fn, 0.0)
        for (R_a, _, _), (R_b, _, _) in zip(unbound, explicit, strict=True):
            np.testing.assert_array_equal(np.asarray(R_a), np.asarray(R_b))

    def test_a_malformed_setpoint_is_refused_where_it_is_read(self):
        mdl = MultiQuadCsPayloadBase(num_quadrotors=3)
        mdl.reset(payload_position=np.array([0.0, 0.0, 2.0]))
        mdl.yaw_setpoint = lambda t: (0.0, 0.0)
        with pytest.raises(ValueError, match="psi, dpsi, ddpsi"):
            mdl._desired_attitudes(self._flat_map(mdl), self._hover_flat_fn(mdl), 0.0)
