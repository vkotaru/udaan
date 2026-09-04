"""Tests for QuadrotorFlexCableBase — state, the coupled EOM and its invariants.

M5's legacy reference (``main_quad_flexible.m::odefun_singleQuadFlex``) was
re-derived rather than transcribed, so the dynamics are pinned here by six
independent checks:

1. **Reduction to M2 at one link** — the strongest of them, and the only one
   that compares against code verified some other way.  ``M5(n=1)`` and
   :class:`~udaan.models.quadrotor_cspayload.QuadrotorCsPayloadBase` are not
   the same code path (M5 solves a 6x6, M2 evaluates a closed form), so they
   agree only to round-off; measured over 400 steps the worst component gap is
   2.3e-14.
2. the static hover trim, derived *here* from Newton's laws without reading the
   model — applied open loop, and (the half with any discriminating power)
   perturbed off trim, where the same first-principles derivation predicts a
   uniform ``eps * g`` on every body;
3. energy under zero input, converging at O(h);
4. linear and angular momentum under gravity alone;
5. conditioning of the ``(3+3n)`` mass matrix on degenerate geometries;
6. the manifold invariants ``||q_i|| = 1`` and ``q_i . omega_i = 0``.

A seventh, cheap and completely independent of the block solve, is
``test_total_momentum_matches_newton``: whatever the chain does internally,
``sum_i m_i a_i`` must equal ``f R e3 - M00 g e3``.

Note on step sizes.  Checks that excite the chain *under tension* have to
respect the integrator's stability limit (see "Numerical stability" in the
model's module docstring); checks at zero input do not, because a free-falling
chain carries no tension and has no fast modes at all.  That is why the energy
and momentum tests sweep every link count while the closed-loop ones do not.

Note on ``reset()``.  :func:`swinging` builds ``omega_i`` as ``q_i x r`` from an
already normalised ``q_i``, so everything it feeds is on the manifold and the
repair ``reset()`` performs is invisible to it.
``test_reset_repairs_off_manifold_per_link_input`` is the path that feeds a
stretched direction and a non-tangent rate deliberately; its cross-backend
counterpart lives in ``test_quadrotor_flexcable_mujoco.py``.
"""

import logging

import numpy as np
import pytest

from tests.conftest import TOL_EXACT, TOL_INTEGRATION, TOL_PHYSICAL, TOL_TIGHT
from udaan.core.defaults import (
    DEFAULT_CABLE_LENGTH,
    DEFAULT_NUM_CABLE_LINKS,
    DEFAULT_PAYLOAD_MASS,
    GRAVITY,
)
from udaan.manif import S2, SO3, TS2, TSO3, rodrigues_expm
from udaan.models.quadrotor_cspayload import QuadrotorCsPayloadBase
from udaan.models.quadrotor_flexcable import (
    QuadrotorFlexCableBase,
    QuadrotorFlexCableState,
)

LINK_COUNTS = [1, 2, 5, 10]

_E3 = np.array([0.0, 0.0, 1.0])


# ── helpers ───────────────────────────────────────────────────────────


def hover_trim(mdl):
    """Static hover input for the model's configuration, from first principles.

    Nothing here reads the equations of motion.  At rest the chain is a stack
    of point masses hanging on rigid links: each joint transmits the weight of
    everything below it, every link is therefore vertical, and the airframe has
    to carry the lot::

        f = (m_Q + sum_k m_k) g,    M = 0,    R = I,    q_i = -e3

    Returns the wrench ``[f, 0, 0, 0]``.
    """
    total = float(mdl.mass + np.sum(mdl.link_masses))
    return np.array([total * GRAVITY, 0.0, 0.0, 0.0])


def total_energy(mdl):
    """Kinetic + potential energy of the airframe (including spin) and every mass."""
    state = mdl.state
    energy = 0.5 * mdl.mass * float(np.dot(state.velocity, state.velocity))
    energy += mdl.mass * GRAVITY * state.position[2]
    Omega = np.asarray(state.angular_velocity)
    energy += 0.5 * float(Omega @ mdl.inertia @ Omega)
    positions = mdl.link_positions()
    velocities = mdl.link_velocities()
    for i in range(mdl.num_links):
        energy += 0.5 * mdl.link_masses[i] * float(np.dot(velocities[i], velocities[i]))
        energy += mdl.link_masses[i] * GRAVITY * positions[i][2]
    return energy


def momenta(mdl):
    """``(p, L, M_total)``: linear momentum, angular momentum about the system COM."""
    state = mdl.state
    masses = [mdl.mass, *list(mdl.link_masses)]
    positions = [np.asarray(state.position), *list(mdl.link_positions())]
    velocities = [np.asarray(state.velocity), *list(mdl.link_velocities())]

    total_mass = float(sum(masses))
    p = sum(m * v for m, v in zip(masses, velocities, strict=True))
    x_cm = sum(m * x for m, x in zip(masses, positions, strict=True)) / total_mass
    v_cm = p / total_mass

    # Spin of the only rigid body in the system; the links are point masses.
    L = np.asarray(state.orientation) @ (mdl.inertia @ np.asarray(state.angular_velocity))
    # Orbital: about the system centre of mass, where gravity exerts no torque.
    for m, x, v in zip(masses, positions, velocities, strict=True):
        L = L + m * np.cross(x - x_cm, v - v_cm)
    return p, L, total_mass


def swinging(n, seed=3, spread=0.25, **kwargs):
    """A model in genuine motion: tilted airframe, spinning, swinging chain.

    The link rates are built **after** the directions are chosen, as
    ``omega_i = q_i x r``, so ``omega_i`` is tangent to the link it belongs to.
    Getting that wrong seeds a tangency violation the continuous dynamics then
    conserve, and every conservation law fails by an amount that does not
    shrink with the step size.
    """
    rng = np.random.default_rng(seed)
    mdl = QuadrotorFlexCableBase(num_links=n, **kwargs)
    directions = []
    for _ in range(n):
        v = np.array([0.0, 0.0, -1.0]) + rng.normal(scale=spread, size=3)
        directions.append(S2(v / np.linalg.norm(v)))
    mdl.reset(
        payload_position=np.array([0.0, 0.0, 1.0]),
        payload_velocity=np.array([0.1, 0.05, -0.03]),
        orientation=SO3(rodrigues_expm(np.array([0.1, -0.05, 0.2]))),
        angular_velocity=TSO3(np.array([0.2, -0.15, 0.3])),
        link_attitudes=directions,
        link_angular_velocities=[
            TS2(np.cross(np.asarray(directions[i]), rng.normal(scale=0.5, size=3)))
            for i in range(n)
        ],
    )
    mdl._input_type = mdl.INPUT_TYPE.WRENCH
    return mdl


# ── tests ─────────────────────────────────────────────────────────────


class TestState:
    """State shape and manifold typing across link counts."""

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_state_shapes(self, n):
        state = QuadrotorFlexCableState(num_links=n)
        assert state.num_links == n
        assert len(state.link_attitudes) == n
        assert len(state.link_angular_velocities) == n
        assert state.payload_position.shape == (3,)
        assert state.payload_velocity.shape == (3,)
        assert state.position.shape == (3,)
        assert state.velocity.shape == (3,)

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_state_types(self, n):
        state = QuadrotorFlexCableState(num_links=n)
        assert isinstance(state.orientation, SO3)
        assert isinstance(state.angular_velocity, TSO3)
        for i in range(n):
            assert isinstance(state.link_attitudes[i], S2)
            assert isinstance(state.link_angular_velocities[i], TS2)

    def test_zero_links_rejected(self):
        with pytest.raises(ValueError, match="num_links"):
            QuadrotorFlexCableState(num_links=0)
        with pytest.raises(ValueError, match="num_links"):
            QuadrotorFlexCableBase(num_links=0)

    def test_reset_resets_every_field(self):
        """Every field, not just the ones the integrator happens to rebind."""
        n = 4
        mdl = QuadrotorFlexCableBase(num_links=n)
        state = mdl.state
        state.payload_position = np.array([1.0, 2.0, 3.0])
        state.payload_velocity = np.array([4.0, 5.0, 6.0])
        state.position = np.array([7.0, 8.0, 9.0])
        state.velocity = np.array([1.5, 2.5, 3.5])
        state.orientation = SO3(rodrigues_expm(np.array([0.3, 0.2, 0.1])))
        state.angular_velocity = TSO3(np.array([1.0, 1.0, 1.0]))
        state.link_attitudes = [S2(np.array([1.0, 0.0, 0.0])) for _ in range(n)]
        state.link_angular_velocities = [TS2(np.array([0.0, 1.0, 0.0])) for _ in range(n)]
        mdl.t = 12.0

        mdl.reset()

        assert mdl.t == 0.0
        np.testing.assert_allclose(state.payload_position, np.zeros(3), atol=TOL_EXACT)
        np.testing.assert_allclose(state.payload_velocity, np.zeros(3), atol=TOL_EXACT)
        np.testing.assert_allclose(state.velocity, np.zeros(3), atol=TOL_EXACT)
        np.testing.assert_allclose(np.asarray(state.orientation), np.eye(3), atol=TOL_EXACT)
        np.testing.assert_allclose(np.asarray(state.angular_velocity), np.zeros(3), atol=TOL_EXACT)
        # Default chain hangs straight down, so the airframe is one cable
        # length above the payload at the origin.
        np.testing.assert_allclose(
            state.position, np.array([0.0, 0.0, mdl.cable_length]), atol=TOL_EXACT
        )
        for i in range(n):
            np.testing.assert_allclose(np.asarray(state.link_attitudes[i]), -_E3, atol=TOL_EXACT)
            np.testing.assert_allclose(
                np.asarray(state.link_angular_velocities[i]), np.zeros(3), atol=TOL_EXACT
            )

    def test_reset_does_not_alias_caller_arrays(self):
        mdl = QuadrotorFlexCableBase(num_links=2)
        start = np.array([0.0, 0.0, 2.0])
        directions = [S2(np.array([0.0, 0.0, -1.0])) for _ in range(2)]
        rates = [TS2(np.array([0.1, 0.0, 0.0])) for _ in range(2)]
        mdl.reset(payload_position=start, link_attitudes=directions, link_angular_velocities=rates)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        for _ in range(20):
            mdl.step(np.zeros(4))
        np.testing.assert_allclose(start, np.array([0.0, 0.0, 2.0]), atol=TOL_EXACT)
        np.testing.assert_allclose(np.asarray(rates[0]), np.array([0.1, 0.0, 0.0]), atol=TOL_EXACT)

    def test_wrong_per_link_length_rejected(self):
        mdl = QuadrotorFlexCableBase(num_links=3)
        with pytest.raises(ValueError, match="link_attitudes"):
            mdl.reset(link_attitudes=[S2(-_E3), S2(-_E3)])

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_reset_repairs_off_manifold_per_link_input(self, n):
        """``q_i`` is normalised and ``omega_i`` projected onto its tangent plane.

        These are the values the MuJoCo backend has no choice but to repair (a
        ball joint cannot represent a stretched link, and a spin about the link
        axis is unobservable), so the base has to repair them the same way or
        the two backends start from different states.  The cross-backend half
        of this contract is
        ``test_quadrotor_flexcable_mujoco.py::TestResetRepairsOffManifoldInput``.
        """
        direction = np.array([0.3, 0.1, -1.0])  # norm 1.0488
        rate = np.array([0.2, 0.1, 0.3])  # direction . rate = -0.23
        assert abs(float(np.linalg.norm(direction)) - 1.0) > 1e-2
        assert abs(float(direction @ rate)) > 1e-2

        mdl = QuadrotorFlexCableBase(num_links=n)
        mdl.reset(
            position=np.array([0.0, 0.0, 2.5]),
            link_attitudes=[direction.copy() for _ in range(n)],
            link_angular_velocities=[rate.copy() for _ in range(n)],
        )

        unit = direction / np.linalg.norm(direction)
        tangent = rate - float(rate @ unit) * unit
        for i in range(n):
            q_i = np.asarray(mdl.state.link_attitudes[i])
            omega_i = np.asarray(mdl.state.link_angular_velocities[i])
            np.testing.assert_allclose(q_i, unit, atol=TOL_EXACT)
            np.testing.assert_allclose(omega_i, tangent, atol=TOL_EXACT)
            assert float(np.linalg.norm(q_i)) == pytest.approx(1.0, abs=TOL_TIGHT)
            assert float(q_i @ omega_i) == pytest.approx(0.0, abs=TOL_TIGHT)

        # The derived payload position uses the repaired directions, so the
        # chain is exactly ``cable_length`` long rather than 4.9% too long.
        assert float(
            np.linalg.norm(mdl.state.payload_position - mdl.state.position)
        ) == pytest.approx(mdl.cable_length, abs=TOL_TIGHT)

    def test_reset_rejects_a_zero_link_direction(self):
        mdl = QuadrotorFlexCableBase(num_links=2)
        with pytest.raises(ValueError, match="zero norm"):
            mdl.reset(link_attitudes=[np.zeros(3), -_E3])

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_quad_position_follows_the_chain(self, n):
        """``x_Q = x_L - sum l_i q_i`` and ``link_positions()[-1] == x_L``."""
        mdl = swinging(n)
        offset = sum(
            mdl.link_lengths[i] * np.asarray(mdl.state.link_attitudes[i]) for i in range(n)
        )
        np.testing.assert_allclose(
            mdl.state.position, mdl.state.payload_position - offset, atol=TOL_EXACT
        )
        np.testing.assert_allclose(
            mdl.link_positions()[-1], mdl.state.payload_position, atol=TOL_TIGHT
        )
        np.testing.assert_allclose(
            mdl.link_velocities()[-1], mdl.state.payload_velocity, atol=TOL_TIGHT
        )

    def test_payload_position_wins_over_position(self):
        mdl = QuadrotorFlexCableBase(num_links=3)
        mdl.reset(position=np.array([5.0, 5.0, 5.0]), payload_position=np.array([0.0, 0.0, 2.0]))
        np.testing.assert_allclose(
            mdl.state.payload_position, np.array([0.0, 0.0, 2.0]), atol=TOL_EXACT
        )
        np.testing.assert_allclose(
            mdl.state.position, np.array([0.0, 0.0, 2.0 + mdl.cable_length]), atol=TOL_EXACT
        )


class TestParameters:
    """Parameter validation, broadcasting and the mass bookkeeping."""

    def test_defaults(self):
        mdl = QuadrotorFlexCableBase()
        assert mdl.num_links == DEFAULT_NUM_CABLE_LINKS
        assert mdl.cable_length == pytest.approx(DEFAULT_CABLE_LENGTH)
        # The payload is the point mass at the distal end of the last link.
        assert mdl.link_masses[-1] > DEFAULT_PAYLOAD_MASS
        assert mdl.total_mass == pytest.approx(mdl.mass + float(np.sum(mdl.link_masses)))

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_convenience_masses_split_the_cable(self, n):
        mdl = QuadrotorFlexCableBase(num_links=n, cable_mass=0.4, payload_mass=1.0)
        np.testing.assert_allclose(mdl.link_masses[:-1], 0.4 / n, atol=TOL_TIGHT)
        assert mdl.link_masses[-1] == pytest.approx(0.4 / n + 1.0)
        assert float(np.sum(mdl.link_masses)) == pytest.approx(1.4)

    def test_link_masses_kwarg_overrides_the_convenience_split(self):
        mdl = QuadrotorFlexCableBase(num_links=3, cable_mass=9.0, link_masses=[0.1, 0.2, 0.3])
        np.testing.assert_allclose(mdl.link_masses, [0.1, 0.2, 0.3], atol=TOL_EXACT)

    def test_scalar_broadcast(self):
        mdl = QuadrotorFlexCableBase(num_links=4)
        mdl.link_lengths = 0.25
        np.testing.assert_allclose(mdl.link_lengths, np.full(4, 0.25), atol=TOL_EXACT)
        assert mdl.cable_length == pytest.approx(1.0)

    def test_per_index_values(self):
        mdl = QuadrotorFlexCableBase(num_links=3)
        mdl.link_lengths = [0.5, 0.3, 0.2]
        mdl.link_masses = [0.01, 0.02, 0.5]
        np.testing.assert_allclose(mdl.link_lengths, [0.5, 0.3, 0.2], atol=TOL_EXACT)
        assert mdl.cable_length == pytest.approx(1.0)
        assert mdl.total_mass == pytest.approx(mdl.mass + 0.53)

    def test_wrong_length_rejected(self):
        mdl = QuadrotorFlexCableBase(num_links=3)
        with pytest.raises(ValueError, match="link_lengths"):
            mdl.link_lengths = [1.0, 2.0]

    @pytest.mark.parametrize("name,value", [("link_masses", 0.0), ("link_lengths", -1.0)])
    def test_non_positive_rejected(self, name, value):
        mdl = QuadrotorFlexCableBase(num_links=2)
        with pytest.raises(ValueError, match=name):
            setattr(mdl, name, value)

    def test_mass_must_be_positive(self):
        with pytest.raises(ValueError, match="mass"):
            QuadrotorFlexCableBase(mass=0.0)

    def test_inertia_accepts_diagonal_and_tensor(self):
        mdl = QuadrotorFlexCableBase(num_links=2)
        mdl.inertia = np.array([0.01, 0.02, 0.03])
        np.testing.assert_allclose(np.diag(mdl.inertia), [0.01, 0.02, 0.03], atol=TOL_EXACT)
        np.testing.assert_allclose(mdl.inertia @ mdl.inertia_inv, np.eye(3), atol=TOL_TIGHT)
        with pytest.raises(ValueError, match="inertia"):
            mdl.inertia = np.zeros((2, 2))


class TestParameterSetterSync:
    """A setter that writes only ``self._…`` leaves the closed loop elsewhere."""

    def test_mass_setter_reaches_the_position_controller(self):
        mdl = QuadrotorFlexCableBase(num_links=2)
        mdl.mass = 3.0
        assert mdl.position_controller.mass == pytest.approx(mdl.total_mass)

    def test_link_mass_setter_reaches_the_position_controller(self):
        mdl = QuadrotorFlexCableBase(num_links=2)
        mdl.link_masses = [0.5, 2.0]
        assert mdl.position_controller.mass == pytest.approx(mdl.mass + 2.5)

    def test_inertia_setter_reaches_the_attitude_controller(self):
        mdl = QuadrotorFlexCableBase(num_links=2)
        mdl.inertia = np.diag([0.05, 0.06, 0.07])
        np.testing.assert_allclose(
            np.asarray(mdl.attitude_controller.inertia), mdl.inertia, atol=TOL_EXACT
        )

    def test_a_controller_without_the_attribute_is_left_alone(self):
        """The sync is duck-typed; a controller that exposes nothing is untouched."""

        class Foreign:
            def compute(self, *args):
                return np.zeros(3)

        mdl = QuadrotorFlexCableBase(num_links=2)
        foreign = Foreign()
        mdl.position_controller = foreign
        mdl.mass = 5.0
        assert mdl.position_controller is foreign
        assert not hasattr(foreign, "mass")

    def test_sync_from_model_takes_precedence(self):
        class Opting:
            mass = None
            seen = None

            def sync_from_model(self, model):
                self.seen = model.total_mass

        mdl = QuadrotorFlexCableBase(num_links=2)
        opting = Opting()
        mdl.position_controller = opting
        mdl.mass = 5.0
        assert opting.seen == pytest.approx(mdl.total_mass)
        # The hook owns the parameters; nothing is written behind its back.
        assert opting.mass is None


class TestDynamics:
    """The equations of motion themselves."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10])
    @pytest.mark.parametrize("thrust", [0.0, 7.5])
    def test_total_momentum_matches_newton(self, n, thrust):
        """Independent of the block solve: ``sum_i m_i a_i = f R e3 - M00 g e3``.

        Whatever the chain's internal constraint forces do, they are internal,
        so the system's total linear momentum obeys plain Newton.  Nothing in
        :meth:`accelerations` enforces this — it drops straight out of the
        assembled matrix if and only if the assembly is right.
        """
        mdl = swinging(n)
        acc = mdl.accelerations(thrust, np.array([0.1, -0.2, 0.05]))
        accel = acc["quad_accel"].copy()
        total = mdl.mass * accel
        for i in range(n):
            accel = accel + mdl.link_lengths[i] * acc["link_accels"][i]
            total = total + mdl.link_masses[i] * accel
        expected = thrust * (np.asarray(mdl.state.orientation) @ _E3)
        expected = expected - mdl.total_mass * GRAVITY * _E3
        np.testing.assert_allclose(total, expected, atol=1e-12)
        # The recursion is the same one `accelerations` reports as the payload.
        np.testing.assert_allclose(acc["payload_accel"], accel, atol=TOL_EXACT)

    def test_reduction_to_m2_at_one_link(self):
        """Validation 1, and the strongest check available.

        With one link, no cable mass and the same physical parameters, M5 must
        reproduce :class:`QuadrotorCsPayloadBase` — which is long verified —
        step for step.  The two are *not* the same code: M2 evaluates a closed
        form, M5 solves a 6x6 system, and the agreement is round-off rather
        than bitwise for that reason.
        """
        flex = QuadrotorFlexCableBase(
            num_links=1, cable_mass=0.0, payload_mass=0.2, cable_length=1.0
        )
        cspayload = QuadrotorCsPayloadBase()
        assert flex.mass == pytest.approx(cspayload.mass)
        assert flex.link_masses[0] == pytest.approx(cspayload.payload_mass)
        assert flex.link_lengths[0] == pytest.approx(cspayload.cable_length)

        direction = np.array([0.3, -0.2, -1.0])
        direction /= np.linalg.norm(direction)
        rate = np.cross(direction, np.array([0.4, -0.1, 0.2]))
        orientation = SO3(rodrigues_expm(np.array([0.1, -0.2, 0.05])))
        body_rate = TSO3(np.array([0.3, -0.1, 0.2]))
        payload_position = np.array([0.5, -0.3, 1.5])
        payload_velocity = np.array([0.1, 0.2, -0.05])

        flex.reset(
            payload_position=payload_position,
            payload_velocity=payload_velocity,
            orientation=orientation,
            angular_velocity=body_rate,
            link_attitudes=[S2(direction)],
            link_angular_velocities=[TS2(rate)],
        )
        cspayload.reset(
            payload_position=payload_position,
            payload_velocity=payload_velocity,
            orientation=orientation,
            angular_velocity=body_rate,
            cable_attitude=S2(direction),
            cable_angular_velocity=TS2(rate),
        )
        flex._input_type = flex.INPUT_TYPE.WRENCH
        cspayload._input_type = cspayload.INPUT_TYPE.WRENCH

        wrench = np.array([6.0, 0.05, -0.02, 0.01])
        for _ in range(400):
            flex.step(wrench.copy())
            cspayload.step(wrench.copy())

        np.testing.assert_allclose(
            flex.state.payload_position, cspayload.state.payload_position, atol=TOL_TIGHT
        )
        np.testing.assert_allclose(
            flex.state.payload_velocity, cspayload.state.payload_velocity, atol=TOL_TIGHT
        )
        np.testing.assert_allclose(flex.state.position, cspayload.state.position, atol=TOL_TIGHT)
        np.testing.assert_allclose(
            np.asarray(flex.state.link_attitudes[0]),
            np.asarray(cspayload.state.cable_attitude),
            atol=TOL_TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(flex.state.link_angular_velocities[0]),
            np.asarray(cspayload.state.cable_angular_velocity),
            atol=TOL_TIGHT,
        )
        np.testing.assert_allclose(
            np.asarray(flex.state.orientation),
            np.asarray(cspayload.state.orientation),
            atol=TOL_TIGHT,
        )

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_hover_trim_holds_open_loop(self, n):
        """Validation 2. The first-principles trim, fed open loop, holds.

        On its own this says very little: at the trim every acceleration the
        solve returns is bitwise zero on the first call, so the loop below is
        five thousand no-ops and it asserts only that an equilibrium is an
        equilibrium.  :meth:`test_off_trim_thrust_accelerates_the_whole_chain`
        is the discriminating half — it feeds a *wrong* thrust and checks the
        response against the same first-principles derivation.
        """
        mdl = QuadrotorFlexCableBase(num_links=n)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        mdl.reset(position=np.array([0.0, 0.0, 3.0]))
        start_quad = mdl.state.position.copy()
        start_payload = mdl.state.payload_position.copy()
        wrench = hover_trim(mdl)
        for _ in range(int(5.0 / mdl.dt)):
            mdl.step(wrench.copy())
        np.testing.assert_allclose(mdl.state.position, start_quad, atol=TOL_INTEGRATION)
        np.testing.assert_allclose(mdl.state.payload_position, start_payload, atol=TOL_INTEGRATION)
        for i in range(n):
            np.testing.assert_allclose(
                np.asarray(mdl.state.link_attitudes[i]), -_E3, atol=TOL_INTEGRATION
            )

    @pytest.mark.parametrize("n", LINK_COUNTS)
    @pytest.mark.parametrize("eps", [-0.15, -0.02, 0.02, 0.15])
    def test_off_trim_thrust_accelerates_the_whole_chain(self, n, eps):
        """Validation 2, made discriminating: perturb off trim and predict.

        Feed ``f = (1 + eps) * hover_trim`` with the chain hanging.  Newton on
        the whole system — no equation of motion read, exactly as
        :func:`hover_trim` does it — gives net force
        ``eps * M_00 * g * e3`` on total mass ``M_00``, so **every** body
        accelerates uniformly at ``eps * g`` and the chain stays rigid and
        vertical.  A sign error, a mislumped ``M_0i``, or a chain that does not
        transmit tension all break this; none of them break the trim itself.
        """
        mdl = QuadrotorFlexCableBase(num_links=n)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        mdl.reset(position=np.array([0.0, 0.0, 5.0]))
        start_quad = mdl.state.position.copy()
        start_payload = mdl.state.payload_position.copy()
        wrench = hover_trim(mdl)
        wrench[0] *= 1.0 + eps

        for _ in range(int(0.5 / mdl.dt)):
            mdl.step(wrench.copy())

        # Uniform acceleration eps*g. The ZOH-Euler substep is exact on a
        # constant acceleration (it advances position with the trapezoid of
        # the two substep velocities), so this is the continuous answer.
        rise = 0.5 * eps * GRAVITY * mdl.t**2
        np.testing.assert_allclose(
            mdl.state.position, start_quad + rise * _E3, atol=TOL_INTEGRATION
        )
        np.testing.assert_allclose(
            mdl.state.payload_position, start_payload + rise * _E3, atol=TOL_INTEGRATION
        )
        np.testing.assert_allclose(
            mdl.state.velocity, eps * GRAVITY * mdl.t * _E3, atol=TOL_INTEGRATION
        )
        for i in range(n):
            np.testing.assert_allclose(
                np.asarray(mdl.state.link_attitudes[i]), -_E3, atol=TOL_INTEGRATION
            )
        # And it really is off trim: at eps = 0 nothing would have moved.
        assert abs(rise) > 1e-3

    def test_hover_trim_holds_with_heterogeneous_parameters(self):
        mdl = QuadrotorFlexCableBase(num_links=4)
        mdl.link_lengths = [0.5, 0.25, 0.15, 0.1]
        mdl.link_masses = [0.05, 0.03, 0.02, 0.4]
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        mdl.reset(position=np.array([0.0, 0.0, 3.0]))
        start = mdl.state.payload_position.copy()
        wrench = hover_trim(mdl)
        for _ in range(int(5.0 / mdl.dt)):
            mdl.step(wrench.copy())
        np.testing.assert_allclose(mdl.state.payload_position, start, atol=TOL_INTEGRATION)

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_hover_thrust_is_the_system_weight(self, n):
        mdl = QuadrotorFlexCableBase(num_links=n)
        np.testing.assert_allclose(mdl.hover_input(), hover_trim(mdl), atol=TOL_EXACT)
        assert mdl.hover_input()[0] == pytest.approx(mdl.total_mass * GRAVITY)

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_zero_input_is_rigid_free_fall(self, n):
        """Nothing pulls, so the whole chain falls together at g."""
        mdl = QuadrotorFlexCableBase(num_links=n)
        mdl._input_type = mdl.INPUT_TYPE.WRENCH
        mdl.reset(position=np.array([0.0, 0.0, 5.0]))
        start = mdl.state.position.copy()
        offsets = mdl.link_positions() - start
        for _ in range(int(0.5 / mdl.dt)):
            mdl.step(np.zeros(4))
        drop = 0.5 * GRAVITY * mdl.t**2
        np.testing.assert_allclose(mdl.state.position, start - drop * _E3, atol=TOL_INTEGRATION)
        np.testing.assert_allclose(
            mdl.link_positions() - mdl.state.position, offsets, atol=TOL_TIGHT
        )

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_link_directions_stay_unit_norm(self, n):
        """Validation 6, first half: the retraction never leaves ``S^2``."""
        mdl = swinging(n)
        for _ in range(int(1.0 / mdl.dt)):
            mdl.step(np.zeros(4))
        for i in range(n):
            assert float(np.linalg.norm(np.asarray(mdl.state.link_attitudes[i]))) == pytest.approx(
                1.0, abs=TOL_TIGHT
            )

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_link_rates_stay_tangent(self, n):
        """Validation 6, second half: ``q_i . omega_i`` drifts at O(h), like M2-M4.

        The continuous dynamics conserve it exactly — the ``M_ii I`` diagonal
        makes ``omega_dot_i . q_i = 0`` an *equation of the solve* rather than
        an accident — so the residual is pure integration error and its
        first-order convergence is a check on the integrator, not the model.
        A nonzero thrust is essential: at zero input the whole system is in
        free fall, every ``omega_dot_i`` is zero and the drift never starts.
        Which is also why the horizon is short and the link counts small — a
        taut chain is where the stability limit lives, and past it the drift
        stops being integration error and becomes the instability instead.
        Measured at ``dt = 5 ms`` over 0.5 s: 2.8e-3 (n=1), 6.5e-2 (n=2),
        2.1e-1 (n=3), each falling by ~4x at ``dt = 1.25 ms``.
        """
        drifts = []
        for dt in (1.0 / 200.0, 1.0 / 800.0):
            mdl = swinging(n, dt=dt)
            wrench = hover_trim(mdl)
            for _ in range(int(0.5 / mdl.dt)):
                mdl.step(wrench.copy())
            drifts.append(
                max(
                    abs(
                        float(
                            np.dot(
                                np.asarray(mdl.state.link_angular_velocities[i]),
                                np.asarray(mdl.state.link_attitudes[i]),
                            )
                        )
                    )
                    for i in range(n)
                )
            )
        assert drifts[0] < 0.3
        # A 4x smaller step must give roughly 4x less drift (first order).
        assert drifts[1] < 0.35 * drifts[0]

    @pytest.mark.slow
    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_energy_conserved_under_zero_input(self, n):
        """Validation 3. Only gravity and workless constraint forces act."""
        errors = []
        for dt in (1.0 / 200.0, 1.0 / 800.0):
            mdl = swinging(n, dt=dt)
            start = total_energy(mdl)
            for _ in range(int(1.0 / mdl.dt)):
                mdl.step(np.zeros(4))
            errors.append(abs(total_energy(mdl) - start) / abs(start))
        assert errors[0] < 5e-3
        assert errors[1] < 0.35 * errors[0]

    @pytest.mark.slow
    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_momentum_conservation_under_zero_input(self, n):
        """Validation 4.

        Linear momentum changes only under gravity; angular momentum about the
        system centre of mass — where gravity exerts no torque — is conserved.
        Both are exact invariants of the continuous flow, so the residual is
        integration error and has to shrink with the step.
        """
        linear, angular = [], []
        for dt in (1.0 / 200.0, 1.0 / 800.0):
            mdl = swinging(n, dt=dt)
            p0, L0, total_mass = momenta(mdl)
            for _ in range(int(1.0 / mdl.dt)):
                mdl.step(np.zeros(4))
            p1, L1, _ = momenta(mdl)
            linear.append(float(np.linalg.norm(p1 - p0 + total_mass * GRAVITY * mdl.t * _E3)))
            angular.append(float(np.linalg.norm(L1 - L0)))
        assert linear[0] < 5e-2
        assert linear[1] < 0.35 * linear[0]
        assert angular[0] < 5e-2
        assert angular[1] < 0.35 * angular[0]


class TestConditioning:
    """Validation 5: the ``(3+3n)`` mass matrix on degenerate geometries.

    Positive definiteness is a property of the *masses* alone — the geometry
    enters only through ``hat(q_i)`` blocks whose contribution is a pullback of
    a positive semi-definite kinetic-energy Hessian — so no link configuration
    can make it singular. That claim is tested here directly rather than
    inferred from a simulation that happens not to blow up.
    """

    @staticmethod
    def _eigenvalues(mdl):
        matrix = mdl.mass_matrix()
        assert matrix.shape == (3 * (mdl.num_links + 1), 3 * (mdl.num_links + 1))
        # The assembly is symmetric by construction: hat(q)^T = -hat(q) maps
        # each off-diagonal block onto its transpose.
        np.testing.assert_allclose(matrix, matrix.T, atol=TOL_EXACT)
        return np.linalg.eigvalsh(matrix)

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_hanging_chain_is_positive_definite(self, n):
        mdl = QuadrotorFlexCableBase(num_links=n)
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        assert self._eigenvalues(mdl).min() > 0.0

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_all_links_parallel_do_not_make_it_singular(self, n):
        """The most degenerate configuration there is: a perfectly straight cable."""
        mdl = QuadrotorFlexCableBase(num_links=n)
        direction = np.array([1.0, 1.0, -1.0])
        direction /= np.linalg.norm(direction)
        mdl.reset(
            position=np.array([0.0, 0.0, 2.0]),
            link_attitudes=[S2(direction) for _ in range(n)],
        )
        assert self._eigenvalues(mdl).min() > 0.0

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_random_configurations_are_positive_definite(self, n, rng):
        mdl = QuadrotorFlexCableBase(num_links=n)
        for _ in range(5):
            directions = []
            for _ in range(n):
                v = rng.normal(size=3)
                directions.append(S2(v / np.linalg.norm(v)))
            mdl.reset(position=np.array([0.0, 0.0, 2.0]), link_attitudes=directions)
            assert self._eigenvalues(mdl).min() > 0.0

    def test_very_unequal_masses_and_lengths_stay_definite(self):
        """Six orders of magnitude of mass ratio: definite, but ill-conditioned.

        Conditioning is the honest word for what degenerate parameters cost
        here — the matrix never loses rank, but a 1e-6 kg link on a 1e-3 m
        lever against a 5 kg payload pushes the condition number to ~1e13, and
        the ZOH-Euler stability limit bites long before the solve does.
        """
        mdl = QuadrotorFlexCableBase(num_links=5)
        mdl.link_masses = [1e-6, 1e-6, 1e-6, 1e-6, 5.0]
        mdl.link_lengths = [1e-3, 1e-3, 1e-3, 1e-3, 2.0]
        mdl.reset(position=np.array([0.0, 0.0, 4.0]))
        eigenvalues = self._eigenvalues(mdl)
        assert eigenvalues.min() > 0.0
        assert eigenvalues.max() / eigenvalues.min() > 1e10

    def test_single_link_is_not_singular(self):
        mdl = QuadrotorFlexCableBase(num_links=1)
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        assert self._eigenvalues(mdl).min() > 0.0


class TestStability:
    """The small-oscillation analysis and the integrator's stability limit."""

    @pytest.mark.parametrize("n", LINK_COUNTS)
    def test_frequencies_have_one_rigid_translation_mode(self, n):
        mdl = QuadrotorFlexCableBase(num_links=n)
        frequencies = mdl.natural_frequencies()
        assert frequencies.shape == (n + 1,)
        assert frequencies[0] == pytest.approx(0.0, abs=1e-8)
        assert np.all(frequencies[1:] > 0.0)
        assert np.all(np.diff(frequencies) >= -1e-12)

    def test_single_link_matches_the_pendulum_frequency(self):
        """One link with a massless cable is a plain pendulum on a free cart.

        Its only nonzero mode is ``sqrt(g (m_Q + m_L) / (m_Q l))`` — the
        reduced-mass pendulum — which the general eigenproblem must reproduce.
        """
        mdl = QuadrotorFlexCableBase(
            num_links=1, cable_mass=0.0, payload_mass=0.2, cable_length=1.0
        )
        expected = np.sqrt(GRAVITY * (mdl.mass + mdl.link_masses[0]) / (mdl.mass * 1.0))
        assert mdl.natural_frequencies()[1] == pytest.approx(expected, rel=1e-9)

    def test_growth_rate_is_the_euler_amplification_exponent(self):
        mdl = QuadrotorFlexCableBase(num_links=3)
        expected = mdl.natural_frequencies().max() ** 2 * mdl.h / 2.0
        assert mdl.stability_growth_rate() == pytest.approx(expected)

    def test_defaults_sit_inside_the_stable_region(self):
        assert QuadrotorFlexCableBase().stability_growth_rate() < 1.0

    def test_a_stiff_chain_warns(self, caplog):
        with caplog.at_level("WARNING"):
            mdl = QuadrotorFlexCableBase(num_links=10, cable_mass=0.01)
        assert mdl.stability_growth_rate() > 1.0
        assert "stiff for the ZOH-Euler integrator" in caplog.text

    def test_a_smaller_step_recovers_the_stable_region(self):
        mdl = QuadrotorFlexCableBase(num_links=10, dt=1.0 / 2000.0)
        assert mdl.stability_growth_rate() < 1.0


class TestInputPipeline:
    """``_repackage_input`` — shared with M1/M2, exercised here for M5's state."""

    def test_acceleration_input_produces_a_wrench(self):
        # Explicit: M5 defaults to WRENCH, because its default controller is an
        # LQR that emits one. The thrust-vector pipeline is opt-in here.
        mdl = QuadrotorFlexCableBase(num_links=2, input="acceleration")
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        wrench = mdl._repackage_input(np.array([0.0, 0.0, mdl.total_mass * GRAVITY]))
        assert wrench.shape == (4,)
        assert wrench[0] == pytest.approx(mdl.total_mass * GRAVITY, abs=TOL_TIGHT)

    def test_wrench_input_is_clipped(self):
        mdl = QuadrotorFlexCableBase(num_links=2, input="wrench")
        wrench = mdl._repackage_input(np.array([1e6, 1e6, -1e6, 1e6]))
        assert wrench[0] == pytest.approx(mdl._max_thrust)
        np.testing.assert_allclose(wrench[1:], [5.0, -5.0, 2.0], atol=TOL_EXACT)

    def test_prop_force_roundtrip(self):
        mdl = QuadrotorFlexCableBase(num_links=2, input="prop_forces")
        props = np.array([2.0, 2.0, 2.0, 2.0])
        wrench = mdl._repackage_input(props)
        assert wrench[0] == pytest.approx(8.0)
        np.testing.assert_allclose(wrench[1:], np.zeros(3), atol=TOL_TIGHT)

    def test_input_does_not_mutate_the_caller_array(self):
        mdl = QuadrotorFlexCableBase(num_links=2, input="wrench")
        u = np.array([1e6, 0.0, 0.0, 0.0])
        mdl._repackage_input(u)
        assert u[0] == pytest.approx(1e6)

    @pytest.mark.parametrize("value", ["WRENCH", "accel", "", None, 3])
    def test_an_unrecognised_input_type_is_refused(self, value):
        """Not silently resolved to the default.

        The three pipelines take three incompatible input *shapes* — a ``(4,)``
        wrench, a ``(4,)`` propeller vector, a ``(3,)`` thrust vector — so a
        typo that fell through to ``WRENCH`` would not mistune the loop, it
        would run a different one and crash somewhere unrelated.
        """
        with pytest.raises(ValueError, match="unknown input type"):
            QuadrotorFlexCableBase(num_links=2, input=value)

    def test_an_unrecognised_force_type_is_refused(self):
        with pytest.raises(ValueError, match="unknown force type"):
            QuadrotorFlexCableBase(num_links=2, force="props")

    @pytest.mark.parametrize("value", ["wrench", "prop_forces", "acceleration"])
    def test_the_recognised_input_types_are_accepted(self, value):
        assert QuadrotorFlexCableBase(num_links=2, input=value) is not None


class TestClosedLoop:
    """The station-keeping baseline.

    These drive ``compute_station_keeping_input()``, *not* ``simulate()``:
    since the M5 payload LQR landed, ``simulate()`` runs that instead (see
    ``tests/test_quadrotor_flexcable_control.py``). The quadrotor position PD
    remains the baseline the airframe falls back on when it is told only where
    to hover, so it keeps its own test.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_quadrotor_converges_to_its_setpoint(self, n):
        target = np.array([1.0, 0.5, 2.0])
        mdl = QuadrotorFlexCableBase(num_links=n, input="acceleration")
        mdl.position_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.reset(position=np.array([0.0, 0.0, 2.0]))
        while mdl.t < 6.0:
            mdl.step(mdl.compute_station_keeping_input())
        np.testing.assert_allclose(mdl.state.position, target, atol=TOL_PHYSICAL)
        # The payload hangs below, at the chain's full length in steady state.
        assert np.linalg.norm(mdl.state.position - mdl.state.payload_position) == pytest.approx(
            mdl.cable_length, abs=TOL_PHYSICAL
        )

    @pytest.mark.slow
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_simulate_drives_the_station_keeping_loop_under_acceleration_input(self, n):
        """``simulate()`` has to work on the pipeline the docs advertise.

        The loop above is hand-rolled precisely because ``simulate()`` used to
        crash here: the payload LQR emits a ``(4,)`` wrench and the
        ``ACCELERATION`` pipeline hands its input to the geometric attitude
        controller as a *thrust vector*, so the two met inside
        ``_repackage_input`` and raised ``incompatible dimensions for cross
        product``. ``simulate()`` now routes ``compute_station_keeping_input()``
        under this input type, which is the loop that belongs on it.
        """
        target = np.array([1.0, 0.5, 2.0])
        mdl = QuadrotorFlexCableBase(num_links=n, input="acceleration")
        mdl.position_controller.setpoint = lambda t: (target, np.zeros(3), np.zeros(3))
        mdl.simulate(6.0, position=np.array([0.0, 0.0, 2.0]))

        np.testing.assert_allclose(mdl.state.position, target, atol=TOL_PHYSICAL)
        assert np.linalg.norm(mdl.state.position - mdl.state.payload_position) == pytest.approx(
            mdl.cable_length, abs=TOL_PHYSICAL
        )

    def test_simulate_under_acceleration_input_takes_no_flat_feedforward(self, caplog):
        """The station-keeping loop regulates the *airframe*, so the payload
        flat map has nothing to say to it — and must not be built, warned
        about, or counted against ``feedforward_drops``."""
        from udaan.utils.trajectory import CrazyTrajectory

        traj = CrazyTrajectory(center=np.array([0.0, 0.0, 1.5]), ax=0.4, ay=0.35, az=0.25)
        mdl = QuadrotorFlexCableBase(num_links=3, input="acceleration")
        mdl.payload_controller.setpoint = traj.get
        mdl.position_controller.setpoint = lambda t: (
            (np.array([0.0, 0.0, 2.0]),)
            + (
                np.zeros(3),
                np.zeros(3),
            )
        )
        with caplog.at_level(logging.WARNING, logger="udaan.models.quadrotor_flexcable.base"):
            mdl.simulate(1.0, position=np.array([0.0, 0.0, 2.0]))
        assert mdl.feedforward_drops == 0
        assert caplog.records == []

    def test_station_keeping_input_carries_the_whole_system_weight(self):
        mdl = QuadrotorFlexCableBase(num_links=3, input="acceleration")
        mdl.reset(position=np.array([0.0, 0.0, 0.0]))
        mdl.position_controller.setpoint = lambda t: (np.zeros(3), np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(
            mdl.compute_station_keeping_input(),
            np.array([0.0, 0.0, mdl.total_mass * GRAVITY]),
            atol=TOL_TIGHT,
        )
