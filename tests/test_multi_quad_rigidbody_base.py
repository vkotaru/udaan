"""Tests for MultiQuadRigidbodyBase — state, grasp geometry and the coupled EOM.

M4 has no trustworthy reference implementation (see the module docstring of
:mod:`udaan.models.multi_quad_rigidbody.base`), so the dynamics are pinned here
by six independent checks rather than by a port comparison:

1. the free-body Newton-Euler equations, as residuals on
   :meth:`~udaan.models.multi_quad_rigidbody.MultiQuadRigidbodyBase.accelerations`,
   and the payload-only oracle integrated from the extracted tensions;
2. **reduction to M3** — with every ``r_i = 0`` the translation must reproduce
   the independently verified :class:`MultiQuadCsPayloadBase` to round-off (not
   bitwise: M4 solves a ``6 x 6`` where M3 solves a ``3 x 3``, so the two agree
   only to rounding, and the cable directions do differ by ~1e-17);
3. the static hover trim, derived here from first principles and applied open
   loop, holding *position and attitude*;
4. energy under zero input;
5. linear and angular momentum under zero input;
6. conditioning of the ``6 x 6`` mass matrix on degenerate geometries.

Check 2 is the strongest of them: it is the only one that compares against
code that was verified some other way.
"""

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_INTEGRATION, TOL_PHYSICAL, TOL_TIGHT
from udaan.core.defaults import (
    DEFAULT_RIGID_PAYLOAD_INERTIA,
    DEFAULT_RIGID_PAYLOAD_MASS,
    GRAVITY,
)
from udaan.manif import S2, SO3, TS2, TSO3, rodrigues_expm
from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadBase
from udaan.models.multi_quad_rigidbody import (
    MultiQuadRigidbodyBase,
    MultiQuadRigidbodyState,
    grasp_map,
    hover_wrench_map,
    ring_attachment_points,
)

QUAD_COUNTS = [1, 2, 3, 4, 5, 6]

#: Counts whose default ring grasp has full rank 6, hence a controllable
#: payload attitude. See TestGraspGeometry.
CONTROLLABLE_COUNTS = [3, 4, 6]

_E3 = np.array([0.0, 0.0, 1.0])


# ── helpers ───────────────────────────────────────────────────────────


def hover_trim(mdl):
    """Static hover input for the model's current configuration, from scratch.

    Derived from the equilibrium conditions of the module docstring, *not* from
    the model: with ``𝐓_i = t_i q_i`` the force the cable applies to quadrotor
    ``i``, zeroing every derivative in Newton-Euler leaves

        Σ_i t_i q_i = -m_L g e₃            (payload force)
        Σ_i t_i r_i × (R_Lᵀ q_i) = 0       (payload moment)
        u_i = m_i g e₃ - t_i q_i           (quadrotor force)

    The moment row is the constraint a point-mass payload does not impose. A
    team of three or more cables is over-actuated, so the tensions come from
    the least-squares solution; the residual says whether the geometry admits a
    trim at all.

    Returns ``(thrust_vectors, thrusts, orientations, tensions, residual)``.
    """
    nQ = mdl.num_quadrotors
    q = np.array([np.asarray(x) for x in mdl.state.cable_attitudes])
    wrench_map = hover_wrench_map(q, mdl.attachment_points, mdl.state.payload_orientation)
    target = np.concatenate([-mdl.payload_mass * GRAVITY * _E3, np.zeros(3)])
    tensions, *_ = np.linalg.lstsq(wrench_map, target, rcond=None)
    residual = float(np.linalg.norm(wrench_map @ tensions - target))

    forces = np.array([mdl.masses[i] * GRAVITY * _E3 - tensions[i] * q[i] for i in range(nQ)])
    thrusts = np.linalg.norm(forces, axis=1)
    orientations = [
        SO3.from_two_vectors(forces[i] / thrusts[i], np.array([1.0, 0.0, 0.0])) for i in range(nQ)
    ]
    return forces, thrusts, orientations, tensions, residual


def attitude_error_deg(mdl):
    """Angle of the payload's attitude off level, in degrees."""
    R = np.asarray(mdl.state.payload_orientation)
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))


def total_energy(mdl):
    """Kinetic + potential energy of payload (including its spin) and quadrotors."""
    state = mdl.state
    energy = 0.5 * mdl.payload_mass * float(np.dot(state.payload_velocity, state.payload_velocity))
    energy += mdl.payload_mass * GRAVITY * state.payload_position[2]
    Omega_L = np.asarray(state.payload_angular_velocity)
    energy += 0.5 * float(Omega_L @ mdl.payload_inertia @ Omega_L)
    for i in range(mdl.num_quadrotors):
        v = state.velocities[i]
        Omega = np.asarray(state.angular_velocities[i])
        energy += 0.5 * mdl.masses[i] * float(np.dot(v, v))
        energy += 0.5 * float(Omega @ mdl.inertias[i] @ Omega)
        energy += mdl.masses[i] * GRAVITY * state.positions[i][2]
    return energy


def momenta(mdl):
    """``(p, L, M_total)``: linear momentum, angular momentum about the system COM."""
    state = mdl.state
    masses = [mdl.payload_mass, *list(mdl.masses)]
    positions = [state.payload_position, *state.positions]
    velocities = [state.payload_velocity, *state.velocities]

    total_mass = float(sum(masses))
    p = sum(m * v for m, v in zip(masses, velocities, strict=True))
    x_cm = sum(m * x for m, x in zip(masses, positions, strict=True)) / total_mass
    v_cm = p / total_mass

    # Spin: R J Ω in world coordinates, one term per rigid body.
    L = np.asarray(state.payload_orientation) @ (
        mdl.payload_inertia @ np.asarray(state.payload_angular_velocity)
    )
    for i in range(mdl.num_quadrotors):
        L = L + np.asarray(state.orientations[i]) @ (
            mdl.inertias[i] @ np.asarray(state.angular_velocities[i])
        )
    # Orbital: about the system centre of mass, where gravity exerts no torque.
    for m, x, v in zip(masses, positions, velocities, strict=True):
        L = L + m * np.cross(x - x_cm, v - v_cm)
    return p, L, total_mass


def swinging(nQ, seed=3, **kwargs):
    """A model in genuine motion: tumbling payload, spinning quadrotors, swinging cables.

    The cable rates are built **after** ``reset()`` has placed the ring, so
    that ``ω_i`` is tangent to the formation's ``q_i`` and not to the ``-e₃``
    the bare state starts with. Getting that wrong seeds a tangency violation
    the continuous dynamics then conserve, and every conservation law fails by
    an amount that does not shrink with the step size.
    """
    rng = np.random.default_rng(seed)
    mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ, payload_mass=1.2, **kwargs)
    mdl.reset(
        payload_position=np.array([0.0, 0.0, 2.0]),
        payload_velocity=np.array([0.1, 0.05, -0.03]),
        payload_orientation=SO3(rodrigues_expm(np.array([0.1, -0.05, 0.2]))),
        payload_angular_velocity=TSO3(np.array([0.2, -0.15, 0.3])),
        orientations=[SO3(rodrigues_expm(rng.normal(scale=0.2, size=3))) for _ in range(nQ)],
        angular_velocities=[TSO3(rng.normal(scale=0.5, size=3)) for _ in range(nQ)],
    )
    mdl.state.cable_angular_velocities = [
        TS2(np.cross(np.asarray(mdl.state.cable_attitudes[i]), rng.normal(scale=0.5, size=3)))
        for i in range(nQ)
    ]
    mdl._sync_quad_states()
    return mdl


# ── tests ─────────────────────────────────────────────────────────────


class TestState:
    """State shape and manifold typing across quadrotor counts."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_state_shapes(self, nQ):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset()
        assert isinstance(mdl.state, MultiQuadRigidbodyState)
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
        assert mdl.attachment_points.shape == (nQ, 3)

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_state_types(self, nQ):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset()
        assert isinstance(mdl.state.payload_orientation, SO3)
        assert isinstance(mdl.state.payload_angular_velocity, TSO3)
        for i in range(nQ):
            assert isinstance(mdl.state.orientations[i], SO3)
            assert isinstance(mdl.state.angular_velocities[i], TSO3)
            assert isinstance(mdl.state.cable_attitudes[i], S2)
            assert isinstance(mdl.state.cable_angular_velocities[i], TS2)

    def test_reset_resets_every_field(self):
        """Every state field, the payload pose included, goes back to default."""
        nQ = 3
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([1.0, 2.0, 3.0]))
        # Dirty every field.
        mdl.state.payload_velocity = np.array([1.0, 1.0, 1.0])
        mdl.state.payload_orientation = SO3(rodrigues_expm(np.array([0.3, 0.2, 0.1])))
        mdl.state.payload_angular_velocity = TSO3(np.array([1.0, 2.0, 3.0]))
        for i in range(nQ):
            mdl.state.orientations[i] = SO3(rodrigues_expm(np.array([0.1, 0.2, 0.3])))
            mdl.state.angular_velocities[i] = TSO3(np.ones(3))
            mdl.state.cable_attitudes[i] = S2(np.array([1.0, 0.0, 0.0]))
            mdl.state.cable_angular_velocities[i] = TS2(np.ones(3))

        mdl.reset()
        np.testing.assert_allclose(mdl.state.payload_position, np.zeros(3), atol=TOL_EXACT)
        np.testing.assert_allclose(mdl.state.payload_velocity, np.zeros(3), atol=TOL_EXACT)
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_orientation), np.eye(3), atol=TOL_EXACT
        )
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_angular_velocity), np.zeros(3), atol=TOL_EXACT
        )
        assert mdl.t == 0.0
        for i in range(nQ):
            np.testing.assert_allclose(np.asarray(mdl.state.orientations[i]), np.eye(3))
            np.testing.assert_allclose(np.asarray(mdl.state.angular_velocities[i]), np.zeros(3))
            np.testing.assert_allclose(
                np.asarray(mdl.state.cable_angular_velocities[i]), np.zeros(3)
            )
            # Cables go back to the ring formation, not to the bare -e3.
            assert np.asarray(mdl.state.cable_attitudes[i])[2] < 0.0

    def test_reset_does_not_alias_caller_arrays(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=2)
        position = np.array([1.0, 2.0, 3.0])
        rates = [TSO3(np.array([0.1, 0.0, 0.0])) for _ in range(2)]
        mdl.reset(payload_position=position, angular_velocities=rates)
        mdl.step(np.zeros((2, 3)))
        np.testing.assert_allclose(position, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(np.asarray(rates[0]), np.array([0.1, 0.0, 0.0]))

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_quad_positions_follow_attachment_geometry(self, nQ):
        """``x_Q_i = x_L + R_L r_i - l_i q_i`` — the attachment offset is real."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset(
            payload_position=np.array([0.3, -0.2, 1.5]),
            payload_orientation=SO3(rodrigues_expm(np.array([0.0, 0.0, 0.7]))),
        )
        R_L = np.asarray(mdl.state.payload_orientation)
        for i in range(nQ):
            expected = (
                mdl.state.payload_position
                + R_L @ mdl.attachment_points[i]
                - mdl.cable_lengths[i] * np.asarray(mdl.state.cable_attitudes[i])
            )
            np.testing.assert_allclose(mdl.state.positions[i], expected, atol=TOL_EXACT)
            np.testing.assert_allclose(
                mdl.attachment_position(i),
                mdl.state.payload_position + R_L @ mdl.attachment_points[i],
                atol=TOL_EXACT,
            )


class TestParameters:
    """Parameter broadcasting, validation and the new payload-side parameters."""

    def test_scalar_broadcast(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4, masses=0.8, cable_lengths=1.5)
        np.testing.assert_allclose(mdl.masses, np.full(4, 0.8))
        np.testing.assert_allclose(mdl.cable_lengths, np.full(4, 1.5))

    def test_per_index_values(self):
        mdl = MultiQuadRigidbodyBase(
            num_quadrotors=3, masses=[0.8, 0.9, 1.0], cable_lengths=[1.0, 1.2, 1.4]
        )
        np.testing.assert_allclose(mdl.masses, [0.8, 0.9, 1.0])
        np.testing.assert_allclose(mdl.cable_lengths, [1.0, 1.2, 1.4])

    def test_payload_inertia_shapes(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=2)
        np.testing.assert_allclose(mdl.payload_inertia, DEFAULT_RIGID_PAYLOAD_INERTIA)
        mdl.payload_inertia = np.array([0.1, 0.2, 0.3])
        np.testing.assert_allclose(mdl.payload_inertia, np.diag([0.1, 0.2, 0.3]))
        np.testing.assert_allclose(mdl.payload_inertia_inv, np.diag([10.0, 5.0, 10.0 / 3.0]))

    def test_payload_inertia_must_be_positive_definite(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=2)
        with pytest.raises(ValueError, match="positive definite"):
            mdl.payload_inertia = np.diag([0.1, 0.0, 0.3])
        with pytest.raises(ValueError, match="symmetric"):
            mdl.payload_inertia = np.array([[0.1, 0.5, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
        with pytest.raises(ValueError, match=r"\(3,\) or \(3, 3\)"):
            mdl.payload_inertia = np.zeros((4, 4))

    def test_attachment_points_are_a_real_parameter(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=3)
        points = np.array([[0.4, 0.0, 0.1], [0.0, 0.4, 0.1], [-0.4, -0.4, 0.1]])
        mdl.attachment_points = points
        np.testing.assert_allclose(mdl.attachment_points, points)
        # Assignment copies: a later mutation of the caller's array must not
        # silently reshape the grasp.
        points[0, 0] = 99.0
        assert mdl.attachment_points[0, 0] == pytest.approx(0.4)

    def test_wrong_attachment_shape_rejected(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=3)
        with pytest.raises(ValueError, match=r"\(3, 3\)"):
            mdl.attachment_points = np.zeros((4, 3))

    def test_constructor_accepts_the_payload_parameters(self):
        mdl = MultiQuadRigidbodyBase(
            num_quadrotors=2,
            payload_mass=2.5,
            payload_inertia=[0.5, 0.5, 0.9],
            attachment_points=[[0.3, 0.0, 0.0], [-0.3, 0.0, 0.0]],
        )
        assert mdl.payload_mass == pytest.approx(2.5)
        np.testing.assert_allclose(np.diag(mdl.payload_inertia), [0.5, 0.5, 0.9])
        np.testing.assert_allclose(mdl.attachment_points[0], [0.3, 0.0, 0.0])

    def test_invalid_quadrotor_count_rejected(self):
        with pytest.raises(ValueError, match="num_quadrotors must be >= 1"):
            MultiQuadRigidbodyBase(num_quadrotors=0)

    def test_masses_must_be_positive(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=3)
        with pytest.raises(ValueError, match="masses must be strictly positive"):
            mdl.masses = [-1.0, 1.0, 1.0]
        with pytest.raises(ValueError, match="masses must be strictly positive"):
            mdl.masses = 0.0
        with pytest.raises(ValueError, match="masses must be finite"):
            mdl.masses = [np.nan, 1.0, 1.0]

    def test_payload_mass_must_be_positive(self):
        """The other half of the invariant the inertia setter already guards.

        ``m_L < 0`` makes the ``6 x 6`` mass matrix indefinite — measured
        ``λ_min = -1.0`` at ``m_L = -1`` — and ``accelerations()`` happily
        solves it and returns a number.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=3)
        with pytest.raises(ValueError, match="payload_mass must be strictly positive"):
            mdl.payload_mass = -1.0
        with pytest.raises(ValueError, match="payload_mass must be strictly positive"):
            mdl.payload_mass = 0.0
        assert np.min(np.linalg.eigvalsh(mdl.mass_matrix())) > 0.0

    def test_unphysically_light_payload_warns(self, caplog):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=3)
        with caplog.at_level("WARNING"):
            mdl.payload_mass = 1e-30
        assert "below" in caplog.text

    def test_cable_lengths_must_be_positive(self):
        """A zero-length cable divides by zero in the ω̇ back-substitution."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=3)
        with pytest.raises(ValueError, match="cable_lengths must be strictly positive"):
            mdl.cable_lengths = [0.0, 1.0, 1.0]
        assert np.all(np.isfinite(mdl.accelerations()["cable_ang_accels"]))


class TestParameterSetterSync:
    """Assigning a parameter must reach the controllers that cached it.

    Every controller in the default cascade caches physical parameters at
    construction — the payload controller caches the whole grasp — so a setter
    that writes only ``self._…`` leaves the closed loop driving a different
    machine than the one being integrated. The failure is invisible from the
    position error, which is why it survived: it shows up only in the payload's
    *attitude*.
    """

    DOCSTRING_POINTS = np.array(
        [[0.5, 0.5, 0.05], [-0.5, 0.5, 0.05], [-0.5, -0.5, 0.05], [0.5, -0.5, 0.05]]
    )

    def test_attachment_points_reach_the_payload_controller(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        mdl.attachment_points = self.DOCSTRING_POINTS
        np.testing.assert_allclose(
            mdl.payload_controller.grasp, grasp_map(self.DOCSTRING_POINTS), atol=TOL_EXACT
        )
        np.testing.assert_allclose(
            mdl.payload_controller.attachment_points, mdl.attachment_points, atol=TOL_EXACT
        )

    @pytest.mark.slow
    def test_the_class_docstring_example_holds_the_payload_level(self):
        """The ``Usage`` block of :class:`MultiQuadRigidbodyBase`, run verbatim.

        It assigns the attachment points *after* construction, which used to
        leave the cascade allocating through the default ring's grasp: the
        payload ended 5 s later at 132.01° of attitude error and 1.48 rad/s,
        while its position was millimetre-perfect throughout. Assigning the
        same geometry through the constructor gave 0.24°. The two paths must
        agree.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        mdl.attachment_points = self.DOCSTRING_POINTS
        mdl.simulate(tf=5.0, payload_position=np.array([0.0, 0.0, 1.0]))

        assert attitude_error_deg(mdl) < 1.0
        assert float(np.linalg.norm(np.asarray(mdl.state.payload_angular_velocity))) < 0.2

        constructed = MultiQuadRigidbodyBase(
            num_quadrotors=4, attachment_points=self.DOCSTRING_POINTS
        )
        constructed.simulate(tf=5.0, payload_position=np.array([0.0, 0.0, 1.0]))
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_orientation),
            np.asarray(constructed.state.payload_orientation),
            atol=TOL_EXACT,
        )
        np.testing.assert_allclose(
            mdl.state.payload_position, constructed.state.payload_position, atol=TOL_EXACT
        )

    def test_mass_and_length_setters_reach_the_controllers(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        before = mdl.payload_controller.hover_tensions.copy()

        mdl.payload_mass = 4.0 * mdl.payload_mass
        # The hover bias is solved from m_L, so it must have quadrupled with it.
        np.testing.assert_allclose(mdl.payload_controller.hover_tensions, 4.0 * before, rtol=1e-12)

        mdl.masses = 1.5
        assert mdl.position_controllers[0].mass == pytest.approx(
            1.5 + mdl.payload_mass / mdl.num_quadrotors
        )
        mdl.inertias = np.diag([0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            mdl.attitude_controllers[0].inertia, np.diag([0.1, 0.2, 0.3]), atol=TOL_EXACT
        )

    def test_a_substituted_controller_is_left_alone(self):
        """A controller the caller supplied is the caller's to parameterise."""

        class Foreign:
            def compute(self, *args):
                return np.zeros((2, 3))

        mdl = MultiQuadRigidbodyBase(num_quadrotors=2)
        foreign = Foreign()
        mdl.payload_controller = foreign
        mdl.attachment_points = np.array([[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]])
        assert mdl.payload_controller is foreign


class TestGraspGeometry:
    """The grasp map, and the ranks that decide what the team can hold."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_grasp_map_maps_forces_to_a_wrench(self, nQ):
        rng = np.random.default_rng(11)
        points = ring_attachment_points(nQ)
        Phi = grasp_map(points)
        assert Phi.shape == (6, 3 * nQ)
        tau = rng.normal(size=(nQ, 3))
        wrench = Phi @ tau.reshape(-1)
        np.testing.assert_allclose(wrench[:3], tau.sum(axis=0), atol=TOL_EXACT)
        np.testing.assert_allclose(
            wrench[3:], sum(np.cross(points[i], tau[i]) for i in range(nQ)), atol=TOL_EXACT
        )

    @pytest.mark.parametrize(
        ("nQ", "rank"),
        [(1, 3), (2, 5), (3, 6), (4, 6), (5, 6), (6, 6)],
    )
    def test_ring_grasp_rank(self, nQ, rank):
        """One cable produces no moment at all; two leave one axis unactuated.

        This is what makes the payload attitude uncontrollable at ``nQ < 3``
        and is the reason ``TestClosedLoop`` only asserts attitude convergence
        for :data:`CONTROLLABLE_COUNTS`.
        """
        assert np.linalg.matrix_rank(grasp_map(ring_attachment_points(nQ))) == rank

    def test_collinear_attachments_lose_a_moment_axis(self):
        """Four cables on a line still cannot twist about that line."""
        points = np.array([[x, 0.0, 0.05] for x in (-0.6, -0.2, 0.2, 0.6)])
        assert np.linalg.matrix_rank(grasp_map(points)) == 5

    def test_single_attachment_is_on_the_payload_axis(self):
        """A lone cable must attach on the vertical axis or it cannot trim."""
        np.testing.assert_allclose(ring_attachment_points(1)[0][:2], np.zeros(2), atol=TOL_EXACT)

    def test_model_exposes_its_own_grasp_map(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        np.testing.assert_allclose(mdl.grasp_map, grasp_map(mdl.attachment_points))


class TestDynamics:
    """The coupled equations of motion."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_free_body_newton_euler_residuals(self, nQ):
        """Every free-body equation the derivation started from, as a residual.

        The ``6 x 6`` solve and the ``𝐓_i`` back-substitution are an exact block
        elimination of these, so if the elimination is right the residuals are
        at round-off — and if a sign or a transpose is wrong they are not.
        """
        rng = np.random.default_rng(5)
        mdl = swinging(nQ, seed=5)
        mdl.attachment_points = mdl.attachment_points + rng.normal(scale=0.05, size=(nQ, 3))
        thrusts = rng.uniform(3.0, 9.0, nQ)
        torques = rng.normal(scale=0.2, size=(nQ, 3))
        acc = mdl.accelerations(thrusts, torques)

        T = acc["tensions"]
        R_L = np.asarray(mdl.state.payload_orientation)
        Omega_L = np.asarray(mdl.state.payload_angular_velocity)

        # m_L a_L = -Σ 𝐓_i - m_L g e₃
        np.testing.assert_allclose(
            mdl.payload_mass * acc["payload_accel"],
            -T.sum(axis=0) - mdl.payload_mass * GRAVITY * _E3,
            atol=TOL_ANALYTIC,
        )
        # J_L Ω̇_L + Ω_L x J_L Ω_L = Σ r_i x (R_Lᵀ (-𝐓_i))
        moment = sum(np.cross(mdl.attachment_points[i], R_L.T @ (-T[i])) for i in range(nQ))
        np.testing.assert_allclose(
            mdl.payload_inertia @ acc["payload_ang_accel"]
            + np.cross(Omega_L, mdl.payload_inertia @ Omega_L),
            moment,
            atol=TOL_ANALYTIC,
        )
        for i in range(nQ):
            u_i = thrusts[i] * (np.asarray(mdl.state.orientations[i]) @ _E3)
            # m_i a_Q_i = u_i - m_i g e₃ + 𝐓_i
            np.testing.assert_allclose(
                mdl.masses[i] * acc["quad_accels"][i],
                u_i - mdl.masses[i] * GRAVITY * _E3 + T[i],
                atol=TOL_ANALYTIC,
            )
            # A massless taut cable carries force only along itself.
            q = np.asarray(mdl.state.cable_attitudes[i])
            assert np.linalg.norm(np.cross(T[i], q)) < TOL_ANALYTIC
            # ẍ_P_i = a_L + R_L(Ω̇_L x r_i) + R_L(Ω_L x (Ω_L x r_i))
            r_i = mdl.attachment_points[i]
            np.testing.assert_allclose(
                acc["attachment_accels"][i],
                acc["payload_accel"]
                + R_L @ np.cross(acc["payload_ang_accel"], r_i)
                + R_L @ np.cross(Omega_L, np.cross(Omega_L, r_i)),
                atol=TOL_ANALYTIC,
            )
            # ω̇_i stays on T_q S²
            assert abs(float(np.dot(acc["cable_ang_accels"][i], q))) < TOL_ANALYTIC

    @pytest.mark.parametrize("nQ", [2, 4])
    def test_payload_only_oracle_reproduces_the_payload_trajectory(self, nQ):
        """Validation 1: the decoupled payload SE(3) integrator agrees.

        This is the shape of ``verify_diff_flat.m::odefun_load_dynamics``, with
        its moment sign corrected — that file accumulates
        ``-r_i x (R_Lᵀ B_i)`` while its own grasp matrix ``[I; hat(r_i)]``, the
        thing that produced those tensions, says ``+``.
        """
        mdl = swinging(nQ, seed=9)
        thrusts = np.full(nQ, 5.0)
        torques = np.zeros((nQ, 3))

        x_L = mdl.state.payload_position.copy()
        v_L = mdl.state.payload_velocity.copy()
        R_L = SO3(np.asarray(mdl.state.payload_orientation).copy())
        Omega_L = np.asarray(mdl.state.payload_angular_velocity).copy()

        h = mdl.h
        for _ in range(800):
            T = mdl.cable_tensions(thrusts)
            accel = -T.sum(axis=0) / mdl.payload_mass - GRAVITY * _E3
            moment = sum(
                np.cross(mdl.attachment_points[i], np.asarray(R_L).T @ (-T[i])) for i in range(nQ)
            )
            ang_accel = np.linalg.solve(
                mdl.payload_inertia,
                -np.cross(Omega_L, mdl.payload_inertia @ Omega_L) + moment,
            )
            x_L = x_L + v_L * h + 0.5 * accel * h**2
            v_L = v_L + accel * h
            R_L = R_L.step(Omega_L * h)
            Omega_L = Omega_L + ang_accel * h
            mdl._zoh(thrusts, torques, h)

        np.testing.assert_allclose(x_L, mdl.state.payload_position, atol=TOL_ANALYTIC)
        np.testing.assert_allclose(
            Omega_L, np.asarray(mdl.state.payload_angular_velocity), atol=TOL_ANALYTIC
        )
        np.testing.assert_allclose(
            np.asarray(R_L), np.asarray(mdl.state.payload_orientation), atol=TOL_ANALYTIC
        )

    @pytest.mark.parametrize("nQ", [1, 2, 3, 4])
    def test_zero_attachment_offsets_reproduce_the_trusted_m3_model(self, nQ):
        """Validation 2, the strongest check available.

        With every ``r_i = 0`` the attachment points collapse onto the payload
        centre, the ``M_12`` coupling block vanishes and the payload rotation
        decouples. What is left must be
        :class:`~udaan.models.multi_quad_cspayload.MultiQuadCsPayloadBase`,
        which is independently verified.

        Note that ``J_L`` is left alone rather than driven to zero: at ``r_i = 0``
        the rotation decouples for *any* ``J_L ≻ 0``, and ``J_L → 0`` would make
        the ``M_22`` block singular for nothing.
        """
        rng = np.random.default_rng(7)
        params = {
            "num_quadrotors": nQ,
            "masses": [0.8 + 0.1 * i for i in range(nQ)],
            "cable_lengths": [1.0 + 0.1 * i for i in range(nQ)],
            "payload_mass": 0.7,
        }
        m4 = MultiQuadRigidbodyBase(**params)
        m3 = MultiQuadCsPayloadBase(**params)
        m4.attachment_points = np.zeros((nQ, 3))

        initial = {
            "payload_position": np.array([0.1, -0.2, 1.3]),
            "payload_velocity": np.array([0.05, 0.1, -0.02]),
            "orientations": [SO3(rodrigues_expm(rng.normal(scale=0.2, size=3))) for _ in range(nQ)],
            "angular_velocities": [TSO3(rng.normal(scale=0.3, size=3)) for _ in range(nQ)],
            "cable_angular_velocities": [TS2(rng.normal(scale=0.2, size=3)) for _ in range(nQ)],
        }
        m4.reset(**initial)
        m3.reset(**initial)

        u = rng.normal(scale=1.0, size=(nQ, 3)) + np.array([0.0, 0.0, 9.0])
        for _ in range(400):
            m4.step(u)
            m3.step(u)

        np.testing.assert_allclose(
            m4.state.payload_position, m3.state.payload_position, atol=TOL_EXACT
        )
        np.testing.assert_allclose(
            m4.state.payload_velocity, m3.state.payload_velocity, atol=TOL_TIGHT
        )
        for i in range(nQ):
            np.testing.assert_allclose(
                np.asarray(m4.state.cable_attitudes[i]),
                np.asarray(m3.state.cable_attitudes[i]),
                atol=TOL_EXACT,
            )
            np.testing.assert_allclose(m4.state.positions[i], m3.state.positions[i], atol=TOL_TIGHT)
        # The payload rotation is untouched: no attachment offset, no moment.
        np.testing.assert_allclose(
            np.asarray(m4.state.payload_angular_velocity), np.zeros(3), atol=TOL_EXACT
        )
        np.testing.assert_allclose(
            np.asarray(m4.state.payload_orientation), np.eye(3), atol=TOL_EXACT
        )

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_hover_trim_holds_position_and_attitude(self, nQ):
        """Validation 3: an open-loop trim derived from first principles."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        forces, thrusts, orientations, tensions, residual = hover_trim(mdl)
        # The ring geometry admits an exact trim at every nQ — the moment row
        # is not merely small, it is satisfied.
        assert residual < TOL_ANALYTIC
        assert np.all(tensions > 0.0), "a cable can only pull"

        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]), orientations=orientations)
        start = mdl.state.payload_position.copy()
        for _ in range(int(3.0 / mdl.dt)):
            mdl.step(forces)

        np.testing.assert_allclose(mdl.state.payload_position, start, atol=TOL_INTEGRATION)
        np.testing.assert_allclose(mdl.state.payload_velocity, np.zeros(3), atol=TOL_INTEGRATION)
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_orientation), np.eye(3), atol=TOL_INTEGRATION
        )
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_angular_velocity), np.zeros(3), atol=TOL_INTEGRATION
        )

    def test_hover_trim_holds_with_heterogeneous_parameters(self):
        """Different masses, cable lengths *and* attachment radii per cable."""
        nQ = 4
        angles = np.linspace(0.0, 2 * np.pi, nQ, endpoint=False)
        radii = [0.45, 0.55, 0.45, 0.55]
        points = np.array(
            [[radii[i] * np.cos(angles[i]), radii[i] * np.sin(angles[i]), 0.05] for i in range(nQ)]
        )
        mdl = MultiQuadRigidbodyBase(
            num_quadrotors=nQ,
            masses=[0.8, 0.9, 1.0, 1.1],
            cable_lengths=[1.0, 1.2, 1.0, 1.2],
            payload_mass=1.4,
            attachment_points=points,
        )
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        forces, _, orientations, tensions, residual = hover_trim(mdl)
        assert residual < TOL_ANALYTIC
        assert np.all(tensions > 0.0)

        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]), orientations=orientations)
        start = mdl.state.payload_position.copy()
        for _ in range(int(3.0 / mdl.dt)):
            mdl.step(forces)
        np.testing.assert_allclose(mdl.state.payload_position, start, atol=TOL_INTEGRATION)
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_orientation), np.eye(3), atol=TOL_INTEGRATION
        )

    def test_hover_thrust_sums_to_system_weight(self):
        nQ = 4
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        forces, _, _, _, _ = hover_trim(mdl)
        total_weight = (mdl.payload_mass + float(np.sum(mdl.masses))) * GRAVITY
        assert float(forces.sum(axis=0)[2]) == pytest.approx(total_weight, abs=TOL_ANALYTIC)
        np.testing.assert_allclose(forces.sum(axis=0)[:2], np.zeros(2), atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_zero_input_is_rigid_free_fall(self, nQ):
        """Nothing pulls, so the whole formation falls together at g."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        mdl.reset(payload_position=np.array([0.0, 0.0, 5.0]))
        start = mdl.state.payload_position.copy()
        offsets = [p - start for p in mdl.state.positions]
        for _ in range(int(0.5 / mdl.dt)):
            mdl.step(np.zeros((nQ, 4)))
        drop = 0.5 * GRAVITY * mdl.t**2
        np.testing.assert_allclose(
            mdl.state.payload_position, start - drop * _E3, atol=TOL_INTEGRATION
        )
        for i in range(nQ):
            np.testing.assert_allclose(
                mdl.state.positions[i] - mdl.state.payload_position, offsets[i], atol=TOL_TIGHT
            )
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_angular_velocity), np.zeros(3), atol=TOL_TIGHT
        )

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_cable_directions_stay_unit_norm(self, nQ):
        mdl = swinging(nQ)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        for _ in range(int(1.0 / mdl.dt)):
            mdl.step(np.zeros((nQ, 4)))
        for i in range(nQ):
            assert float(np.linalg.norm(np.asarray(mdl.state.cable_attitudes[i]))) == pytest.approx(
                1.0, abs=TOL_TIGHT
            )

    @pytest.mark.parametrize("nQ", [2, 4])
    def test_cable_rates_stay_tangent(self, nQ):
        """``q_i · ω_i`` is a conserved quantity, so the drift must vanish with h.

        The continuous dynamics preserve it exactly — ``ω̇_i`` is a cross
        product with ``q_i``, so ``d(q·ω)/dt = ω · (ω x q) = 0`` — which makes
        the residual pure integration error and its first-order convergence a
        real check on the integrator, not on the model.
        """
        drifts = []
        for dt in (1.0 / 200.0, 1.0 / 800.0):
            mdl = swinging(nQ, dt=dt)
            mdl._input_type = mdl.INPUT_TYPE.WRENCH
            for _ in range(int(1.0 / mdl.dt)):
                mdl.step(np.zeros((nQ, 4)))
            drifts.append(
                max(
                    abs(
                        float(
                            np.dot(
                                np.asarray(mdl.state.cable_angular_velocities[i]),
                                np.asarray(mdl.state.cable_attitudes[i]),
                            )
                        )
                    )
                    for i in range(nQ)
                )
            )
        assert drifts[0] < 1e-3
        # A 4x smaller step must give roughly 4x less drift (first order).
        assert drifts[1] < 0.35 * drifts[0]

    @pytest.mark.slow
    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_energy_conserved_under_zero_input(self, nQ):
        """Validation 4. Only gravity and workless constraint forces act."""
        errors = []
        for dt in (1.0 / 200.0, 1.0 / 800.0):
            mdl = swinging(nQ, dt=dt)
            mdl._input_type = mdl.INPUT_TYPE.WRENCH
            start = total_energy(mdl)
            for _ in range(int(1.0 / mdl.dt)):
                mdl.step(np.zeros((nQ, 4)))
            errors.append(abs(total_energy(mdl) - start) / abs(start))
        assert errors[0] < 1e-3
        assert errors[1] < 0.35 * errors[0]

    @pytest.mark.slow
    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_momentum_conservation_under_zero_input(self, nQ):
        """Validation 5.

        Linear momentum changes only under gravity; angular momentum about the
        system centre of mass — where gravity exerts no torque — is conserved.
        Both are exact invariants of the continuous flow, so the residual is
        integration error and has to shrink with the step.
        """
        linear, angular = [], []
        for dt in (1.0 / 200.0, 1.0 / 800.0):
            mdl = swinging(nQ, dt=dt)
            mdl._input_type = mdl.INPUT_TYPE.WRENCH
            p0, L0, total_mass = momenta(mdl)
            for _ in range(int(1.0 / mdl.dt)):
                mdl.step(np.zeros((nQ, 4)))
            p1, L1, _ = momenta(mdl)
            linear.append(float(np.linalg.norm(p1 - p0 + total_mass * GRAVITY * mdl.t * _E3)))
            angular.append(float(np.linalg.norm(L1 - L0)))
        assert linear[0] < 5e-3
        assert linear[1] < 0.35 * linear[0]
        assert angular[0] < 5e-3
        assert angular[1] < 0.35 * angular[0]

    @pytest.mark.slow
    def test_compressive_cable_tension_is_counted_and_warned_about(self, caplog):
        """The taut-cable precondition ``t_i ≥ 0`` is violated by ordinary steps.

        Nothing in the algebra enforces it — the ``6 x 6`` solve is exact for a
        strut too — so the only defence is to notice. A 4 m lateral step at
        ``nQ = 4`` puts a cable in compression on ~79 of its 1201 steps, down
        to -3.85 N, and the model used to return those vectors from
        :meth:`cable_tensions` without comment.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        target = np.array([4.0, 0.0, 1.0])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        with caplog.at_level("WARNING"):
            mdl.simulate(tf=6.0, payload_position=np.array([0.0, 0.0, 1.0]))
        assert mdl.compressive_cable_steps > 0
        assert "compressive" in caplog.text

        # A hover holds the precondition, and reset() clears the count.
        mdl.payload_controller.setpoint = lambda t: (
            np.array([0.0, 0.0, 1.0]),
            np.zeros(3),
            np.zeros(3),
        )
        mdl.simulate(tf=2.0, payload_position=np.array([0.0, 0.0, 1.0]))
        assert mdl.compressive_cable_steps == 0

    def test_payload_spins_freely_without_cable_moments(self):
        """Euler's equation, exercised through the payload rotation block."""
        # A fully anisotropic payload: the default slab is axisymmetric about
        # z, so Ω x J Ω vanishes for any rate in its equatorial plane and the
        # gyroscopic term would go untested.
        mdl = MultiQuadRigidbodyBase(num_quadrotors=2, payload_inertia=[0.05, 0.12, 0.16])
        mdl.attachment_points = np.zeros((2, 3))
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        Omega0 = np.array([1.0, 0.5, 0.7])
        mdl.reset(payload_position=np.array([0.0, 0.0, 3.0]), payload_angular_velocity=TSO3(Omega0))
        spin0 = mdl.payload_inertia @ Omega0
        for _ in range(int(1.0 / mdl.dt)):
            mdl.step(np.zeros((2, 4)))
        # With no external moment the body-frame spin magnitude is conserved
        # and the rate precesses because the inertia is not isotropic.
        Omega = np.asarray(mdl.state.payload_angular_velocity)
        spin = mdl.payload_inertia @ Omega
        # Relative, not TOL_INTEGRATION: ZOH-Euler on a free rigid body leaves
        # ~1e-4 relative drift in ‖J Ω‖ over a second at h = 1.25 ms. The
        # step-size convergence of the invariants is checked properly in
        # test_momentum_conservation_under_zero_input.
        assert float(np.linalg.norm(spin)) == pytest.approx(float(np.linalg.norm(spin0)), rel=1e-3)
        assert float(np.linalg.norm(Omega - Omega0)) > 0.1


class TestConditioning:
    """Validation 6: the ``6 x 6`` system is never singular."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_mass_matrix_is_positive_definite_in_the_ring(self, nQ):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        M = mdl.mass_matrix()
        assert M.shape == (6, 6)
        np.testing.assert_allclose(M, M.T, atol=TOL_ANALYTIC)
        assert np.min(np.linalg.eigvalsh(M)) > 0.0

    @pytest.mark.parametrize("nQ", [2, 4, 6])
    def test_parallel_cables_do_not_make_it_singular(self, nQ):
        """Every ``q_i`` identical kills the rank of ``Σ m_i P_i``, not of ``M``.

        ``m_L I`` and ``J_L`` carry the whole matrix on their own; the cable
        terms are positive *semi*-definite, so they can only add.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset(
            payload_position=np.array([0.0, 0.0, 1.0]),
            cable_attitudes=[np.array([0.0, 0.0, -1.0]) for _ in range(nQ)],
        )
        assert np.min(np.linalg.eigvalsh(mdl.mass_matrix())) > 0.0

    def test_collinear_attachments_do_not_make_it_singular(self):
        """A rank-5 *grasp* still gives a full-rank *mass matrix*.

        The two are different statements: the grasp map decides what wrench the
        team can produce statically; the mass matrix decides whether the ODE has
        a right-hand side at all.
        """
        nQ = 4
        points = np.array([[x, 0.0, 0.05] for x in (-0.6, -0.2, 0.2, 0.6)])
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ, attachment_points=points)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        assert np.linalg.matrix_rank(mdl.grasp_map) == 5
        assert np.min(np.linalg.eigvalsh(mdl.mass_matrix())) > 0.0

    def test_single_quadrotor_is_not_singular(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=1)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        assert np.min(np.linalg.eigvalsh(mdl.mass_matrix())) > 0.0

    def test_a_heavy_grasp_stays_conditioned(self):
        """Long lever arms inflate ``M_22`` but never flip its sign."""
        nQ = 4
        angles = np.linspace(0.0, 2 * np.pi, nQ, endpoint=False)
        points = np.column_stack([5.0 * np.cos(angles), 5.0 * np.sin(angles), np.zeros(nQ)])
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ, attachment_points=points)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        eig = np.linalg.eigvalsh(mdl.mass_matrix())
        assert np.min(eig) > 0.0
        assert np.isfinite(np.linalg.cond(mdl.mass_matrix()))


class TestInputPipeline:
    """The inherited ``_repackage_input`` funnel."""

    def test_acceleration_input_produces_wrench_stack(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=3)
        mdl.reset()
        wrench = mdl._repackage_input(np.tile(np.array([0.0, 0.0, 8.0]), (3, 1)))
        assert wrench.shape == (3, 4)
        assert np.all(wrench[:, 0] > 0.0)

    def test_wrench_input_is_clipped(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=2, input="wrench")
        mdl.reset()
        wrench = mdl._repackage_input(np.tile(np.array([1e3, 1e3, 1e3, 1e3]), (2, 1)))
        assert np.all(wrench[:, 0] <= mdl._max_thrust)
        assert np.all(wrench[:, 1:] <= mdl._max_torque)

    def test_prop_force_roundtrip(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=2, input="wrench", force="prop_forces")
        mdl.reset()
        requested = np.tile(np.array([6.0, 0.1, -0.1, 0.02]), (2, 1))
        np.testing.assert_allclose(mdl._repackage_input(requested), requested, atol=TOL_ANALYTIC)

    def test_flat_input_is_reshaped(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=3)
        mdl.reset()
        flat = mdl._repackage_input(np.tile(np.array([0.0, 0.0, 8.0]), 3))
        stacked = mdl._repackage_input(np.tile(np.array([0.0, 0.0, 8.0]), (3, 1)))
        np.testing.assert_allclose(flat, stacked, atol=TOL_EXACT)


class TestClosedLoop:
    """The default payload cascade, and where the grasp geometry stops it."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_controller_returns_the_first_principles_trim_at_hover(self, nQ):
        """At the trim state the cascade returns the open-loop trim exactly.

        This is a consistency check on the *cascade*, not two independent
        derivations of the trim: :func:`hover_trim` and the controller both
        solve the same :func:`hover_wrench_map` in least squares. What it pins
        is that the four stages the controller wraps around that solve — pose
        loop, grasp allocation, cable loop, thrust law — compose back to the
        identity at equilibrium, which they do only if every frame and sign in
        them is right.

        It passes at ``nQ = 1`` and ``2`` as well, and that is not evidence
        those configurations work: hovering level needs no moment, so an
        underactuated grasp is never asked for one. See
        :meth:`test_underactuated_grasp_cannot_hold_attitude`.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        target = np.array([0.0, 0.0, 1.0])
        mdl.reset(payload_position=target)
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        forces, _, _, _, _ = hover_trim(mdl)
        np.testing.assert_allclose(mdl.compute_default_input(), forces, atol=TOL_ANALYTIC)

    @pytest.mark.slow
    @pytest.mark.parametrize("nQ", CONTROLLABLE_COUNTS)
    def test_payload_pose_converges(self, nQ):
        """Position *and* attitude, from an off-level, off-target start."""
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        target = np.array([1.0, -0.5, 1.5])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(
            tf=12.0,
            payload_position=np.array([0.0, 0.0, 1.0]),
            payload_orientation=SO3(rodrigues_expm(np.array([0.1, -0.08, 0.25]))),
        )
        assert float(np.linalg.norm(mdl.state.payload_position - target)) < TOL_PHYSICAL
        R = np.asarray(mdl.state.payload_orientation)
        angle = float(np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)))
        assert np.degrees(angle) < 1.0
        assert float(np.linalg.norm(np.asarray(mdl.state.payload_angular_velocity))) < TOL_PHYSICAL

    @pytest.mark.slow
    @pytest.mark.parametrize("nQ", [1, 2])
    def test_underactuated_grasp_cannot_hold_attitude(self, nQ, caplog):
        """The negative of :meth:`test_payload_pose_converges`, pinned.

        At ``nQ = 1`` the ring grasp has rank 3 and at ``nQ = 2`` rank 5, so
        part of the commanded moment cannot be produced at all and ``Φ⁺``
        drops it. Measured over 40 s from an off-level start: 133.09° and
        25.63° of attitude error, with the payload still spinning at ~0.8
        rad/s, while *position* tracking converges normally in both.

        This is a structural limit of the mechanism, not a controller bug —
        which is exactly why it has to be asserted rather than fixed. What was
        a bug is that it happened in silence; the two counters below are the
        part that is required to keep working.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        target = np.array([1.0, -0.5, 1.5])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        assert mdl.payload_controller.grasp_rank < 6

        with caplog.at_level("WARNING"):
            mdl.simulate(
                tf=10.0,
                payload_position=np.array([0.0, 0.0, 1.0]),
                payload_orientation=SO3(rodrigues_expm(np.array([0.1, -0.08, 0.25]))),
            )

        # Position still converges — to a couple of centimetres, disturbed only
        # by the payload it is failing to level. That is what made this
        # invisible: the obvious number looks fine.
        assert float(np.linalg.norm(mdl.state.payload_position - target)) < 5 * TOL_PHYSICAL
        # The attitude does not, and says so.
        assert attitude_error_deg(mdl) > 5.0
        assert mdl.payload_controller.unachievable_wrenches > 0
        assert "outside the range of the grasp map" in caplog.text

    def test_unachievable_wrenches_stay_zero_on_a_full_rank_grasp(self):
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        target = np.array([0.3, 0.2, 1.2])
        mdl.payload_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        for _ in range(int(1.0 / mdl.dt)):
            mdl.step(mdl.compute_default_input())
        assert mdl.payload_controller.grasp_rank == 6
        assert mdl.payload_controller.unachievable_wrenches == 0

    def test_reference_angular_acceleration_is_taken_to_the_body_frame(self):
        """``Ω̇_d`` lives in the reference frame; ``r_i`` lives in the body frame.

        The feedforward term is ``R_L (Ω̇ × r_i)``, so the reference has to be
        transported by ``R_Lᵀ R_d`` first. Feeding the raw ``Ω̇_d`` in is a
        no-op while ``R_L = R_d = I`` — which is every setpoint in this
        repository, so nothing here caught it — and misdirects feedforward in
        proportion to ``‖Ω̇_d‖ ‖r_i‖`` as soon as it is not.
        """
        mdl = MultiQuadRigidbodyBase(num_quadrotors=4)
        R_d = rodrigues_expm(np.array([0.0, 0.0, 0.4]))
        dOmega_d = np.array([0.3, -0.2, 0.1])
        controller = mdl.payload_controller
        controller.attitude_setpoint = lambda t: (R_d, np.zeros(3), dOmega_d)
        mdl.reset(
            payload_position=np.array([0.0, 0.0, 1.0]),
            payload_orientation=SO3(rodrigues_expm(np.array([0.1, -0.08, 0.25]))),
        )
        R_L = np.asarray(mdl.state.payload_orientation)

        _, _, ref_ang_accel = controller.desired_wrench(0.0, mdl.state)
        np.testing.assert_allclose(ref_ang_accel, R_L.T @ R_d @ dOmega_d, atol=TOL_EXACT)
        # And it really is a different vector from the raw reference.
        assert float(np.linalg.norm(ref_ang_accel - dOmega_d)) > 0.05

    @pytest.mark.slow
    def test_station_keeping_pds_hold_the_formation(self):
        """The pre-cascade baseline still works."""
        nQ = 4
        mdl = MultiQuadRigidbodyBase(num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        start = [p.copy() for p in mdl.state.positions]
        for _ in range(int(5.0 / mdl.dt)):
            mdl.step(mdl.compute_formation_input())
        for i in range(nQ):
            assert float(np.linalg.norm(mdl.state.positions[i] - start[i])) < 0.35


def test_default_payload_inertia_is_the_slab_it_claims_to_be():
    """The default inertia is MuJoCo's box formula, not a hand-typed number.

    It has to be, because the MuJoCo backend reads its own value out of the
    MJCF's ``inertiafromgeom`` and the two must agree.
    """
    a, b, c = 0.5, 0.5, 0.05
    expected = (DEFAULT_RIGID_PAYLOAD_MASS / 3.0) * np.diag([b**2 + c**2, a**2 + c**2, a**2 + b**2])
    np.testing.assert_allclose(DEFAULT_RIGID_PAYLOAD_INERTIA, expected, atol=TOL_EXACT)
