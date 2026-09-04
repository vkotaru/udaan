"""Tests for MultiQuadRigidbodyMujoco — MJCF wiring, state sync, station keeping.

The backend loads a fixed, hand-written asset, so most of what is worth testing
here is the *correspondence* between that asset and the analytical model: the
quadrotor count, the attachment points, which tendon joins which pair of sites,
and the inertial properties. Each of those was silently wrong in the retired
``models/mujoco/multi_quad_rigidbody.py``, and each is now checked at load time.
"""

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_EXACT, TOL_TIGHT
from udaan.core.defaults import DEFAULT_RIGID_PAYLOAD_INERTIA, GRAVITY
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.multi_quad_rigidbody import MultiQuadRigidbodyMujoco
from udaan.models.multi_quad_rigidbody.mujoco import (
    MJCF_ATTACHMENT_POINTS,
    MJCF_NUM_QUADROTORS,
)

pytestmark = pytest.mark.mujoco


@pytest.fixture
def mdl():
    return MultiQuadRigidbodyMujoco(render=False)


class TestMujocoSetup:
    """Model loading, index resolution and parameters read back from the MJCF."""

    def test_instantiate(self, mdl):
        assert mdl.num_quadrotors == MJCF_NUM_QUADROTORS
        assert mdl._mjMdl.model.nu == 4 * MJCF_NUM_QUADROTORS
        assert len(mdl.state.positions) == MJCF_NUM_QUADROTORS

    def test_quadrotor_count_is_validated_against_the_asset(self):
        """``num_quadrotors=6`` used to silently yield four."""
        with pytest.raises(ValueError, match="defines exactly 4"):
            MultiQuadRigidbodyMujoco(render=False, num_quadrotors=6)
        with pytest.raises(ValueError, match="defines exactly 4"):
            MultiQuadRigidbodyMujoco(render=False, num_quadrotors=2)

    def test_attachment_points_are_validated_against_the_sites(self, mdl):
        """The grasp geometry must agree with the ``pyld_poc`` sites.

        The retired model's hardcoded grasp matrix had ``(0.5, -0.5)`` where
        ``pyld_poc2`` sits at ``(-0.5, -0.5)``, so two of its four columns were
        duplicates and one corner of the slab was never represented. Nothing
        compared the two.
        """
        np.testing.assert_allclose(mdl.attachment_points, MJCF_ATTACHMENT_POINTS, atol=TOL_EXACT)
        model = mdl._mjMdl.model
        sites = np.array([model.site_pos[i] for i in mdl._attachment_site_id])
        np.testing.assert_allclose(sites, MJCF_ATTACHMENT_POINTS, atol=1e-9)
        # All four corners distinct — the duplicate-column bug, as a property.
        assert len({tuple(np.round(p, 9)) for p in mdl.attachment_points}) == 4

    def test_mismatched_attachment_points_raise(self):
        with pytest.raises(RuntimeError, match="disagree with the MJCF"):
            MultiQuadRigidbodyMujoco(render=False, attachment_points=np.zeros((4, 3)))

    def test_quad_mass_is_the_whole_body_not_the_centre_box(self, mdl):
        """0.99 kg, not the 0.75 kg the retired model hardcoded.

        The MJCF's ``quad{i}_geom`` box carries 0.75 kg, but each quadrotor also
        has four 0.01 kg arms and four 0.05 kg propellers. Feeding 0.75 into a
        weight feedforward is 24% light.
        """
        np.testing.assert_allclose(mdl.masses, np.full(4, 0.99), atol=1e-9)
        model = mdl._mjMdl.model
        for i, body_id in enumerate(mdl._quad_body_id):
            assert model.body_mass[body_id] == pytest.approx(mdl.masses[i])

    def test_payload_inertia_matches_the_analytical_default(self, mdl):
        """MuJoCo's ``inertiafromgeom`` and ``DEFAULT_RIGID_PAYLOAD_INERTIA`` agree."""
        assert mdl.payload_mass == pytest.approx(1.0)
        np.testing.assert_allclose(
            mdl.payload_inertia, DEFAULT_RIGID_PAYLOAD_INERTIA, atol=TOL_ANALYTIC
        )

    def test_params_read_from_mjcf(self, mdl):
        nQ = mdl.num_quadrotors
        assert mdl.masses.shape == (nQ,)
        assert np.all(mdl.masses > 0.0)
        assert mdl.inertias.shape == (nQ, 3, 3)
        assert mdl.cable_lengths.shape == (nQ,)
        assert np.all(mdl.cable_lengths > 0.0)

    def test_inertias_cannot_be_passed_to_the_constructor(self):
        with pytest.raises(ValueError, match="derives inertias"):
            MultiQuadRigidbodyMujoco(render=False, inertias=np.eye(3))

    def test_tendon_wiring_is_checked(self, mdl):
        """Tendon ``i`` joins ``quad{i}_end1`` to ``pyld_poc{i}``.

        Nothing in the MJCF enforces the pairing, and it is what makes
        ``attachment_points[i]`` belong to cable ``i``. The check is
        constructive: swapping two tendons in a copy of the model must trip it.
        """
        model = mdl._mjMdl.model
        assert model.ntendon == mdl.num_quadrotors
        original = model.wrap_objid.copy()
        try:
            start_a = int(model.tendon_adr[0])
            start_b = int(model.tendon_adr[1])
            model.wrap_objid[start_a + 1], model.wrap_objid[start_b + 1] = (
                model.wrap_objid[start_b + 1],
                model.wrap_objid[start_a + 1],
            )
            with pytest.raises(RuntimeError, match="tendon 0 joins"):
                mdl._check_tendon_wiring()
        finally:
            model.wrap_objid[:] = original
        mdl._check_tendon_wiring()

    def test_grasp_map_has_full_rank(self, mdl):
        """Four corners of a slab can produce any wrench."""
        assert np.linalg.matrix_rank(mdl.grasp_map) == 6


class TestMujocoStateSync:
    """``_query_latest_state`` pulls the right quantities out of MuJoCo."""

    def test_state_types(self, mdl):
        mdl.reset()
        assert isinstance(mdl.state.payload_orientation, SO3)
        assert isinstance(mdl.state.payload_angular_velocity, TSO3)
        for i in range(mdl.num_quadrotors):
            assert isinstance(mdl.state.orientations[i], SO3)
            assert isinstance(mdl.state.angular_velocities[i], TSO3)
            assert isinstance(mdl.state.cable_attitudes[i], S2)
            assert isinstance(mdl.state.cable_angular_velocities[i], TS2)

    def test_reset_places_the_payload_absolutely(self, mdl):
        """``payload_position`` is where the payload ends up, not a displacement.

        The retired model added it to every body's ``qpos``, so the same flag
        under the same CLI help text meant an offset there and an absolute
        position for ``multi-quad``.
        """
        target = np.array([-1.0, 2.0, 1.5])
        mdl.reset(payload_position=target)
        np.testing.assert_allclose(mdl.state.payload_position, target, atol=TOL_ANALYTIC)
        assert mdl.t == 0.0
        # Idempotent: resetting to the same target twice must not drift.
        mdl.reset(payload_position=target)
        np.testing.assert_allclose(mdl.state.payload_position, target, atol=TOL_ANALYTIC)
        for i in range(mdl.num_quadrotors):
            assert mdl.state.positions[i][2] > mdl.state.payload_position[2]

    def test_reset_rejects_the_overrides_it_cannot_apply(self, mdl):
        """A documented base kwarg used to be accepted and then ignored.

        ``simulate(tf, payload_orientation=...)` ran the MJCF's own level
        formation and reported nothing, so the run answered a different
        question from the one asked. The initial pose here is fixed by the
        tendon constraints, so the honest answer is an error.
        """
        for kwargs in (
            {"payload_orientation": np.eye(3)},
            {"payload_velocity": np.zeros(3)},
            {"cable_attitudes": [np.array([0.0, 0.0, -1.0])] * MJCF_NUM_QUADROTORS},
            {"not_a_state_at_all": 1.0},
        ):
            with pytest.raises(ValueError, match="cannot honour"):
                mdl.reset(**kwargs)
        with pytest.raises(ValueError, match="cannot honour"):
            mdl.simulate(tf=0.05, payload_orientation=np.eye(3))
        # The one it can apply still works.
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.34]))

    def test_reset_translates_the_whole_formation(self, mdl):
        mdl.reset()
        base = [p.copy() for p in mdl.state.positions]
        base_payload = mdl.state.payload_position.copy()
        target = np.array([0.5, -0.5, 2.0])
        mdl.reset(payload_position=target)
        shift = target - base_payload
        for i in range(mdl.num_quadrotors):
            np.testing.assert_allclose(mdl.state.positions[i], base[i] + shift, atol=TOL_ANALYTIC)

    def test_cables_are_measured_from_the_attachment_not_the_payload_centre(self, mdl):
        """The whole difference from M3, and the thing the sites exist for."""
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.34]))
        for i in range(mdl.num_quadrotors):
            q = np.asarray(mdl.state.cable_attitudes[i])
            assert float(np.linalg.norm(q)) == pytest.approx(1.0, abs=TOL_TIGHT)
            direction = mdl.attachment_position(i) - mdl.state.positions[i]
            np.testing.assert_allclose(q, direction / np.linalg.norm(direction), atol=TOL_ANALYTIC)
            # ... and it is genuinely a different vector from the M3 answer.
            centre = mdl.state.payload_position - mdl.state.positions[i]
            assert float(np.linalg.norm(q - centre / np.linalg.norm(centre))) > 1e-3

    def test_payload_pose_is_read_back(self, mdl):
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.34]))
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_orientation), np.eye(3), atol=TOL_ANALYTIC
        )
        np.testing.assert_allclose(
            np.asarray(mdl.state.payload_angular_velocity), np.zeros(3), atol=TOL_ANALYTIC
        )

    def test_attachment_positions_match_the_mujoco_sites(self, mdl):
        """``x_L + R_L r_i`` and MuJoCo's own ``site_xpos`` are the same point."""
        mdl.reset(payload_position=np.array([0.3, -0.2, 1.5]))
        for i in range(mdl.num_quadrotors):
            np.testing.assert_allclose(
                mdl.attachment_position(i),
                mdl._mjMdl.data.site_xpos[mdl._attachment_site_id[i]],
                atol=TOL_ANALYTIC,
            )


class TestMujocoSetters:
    """Property setters write through into the MuJoCo model."""

    def test_mass_setters(self, mdl):
        mdl.masses = [1.1, 1.2, 1.3, 1.4]
        for i, body_id in enumerate(mdl._quad_body_id):
            assert mdl._mjMdl.model.body_mass[body_id] == pytest.approx(mdl.masses[i])
        mdl.payload_mass = 0.42
        assert mdl._mjMdl.model.body_mass[mdl._payload_body_id] == pytest.approx(0.42)

    def test_payload_inertia_setter(self, mdl):
        mdl.payload_inertia = np.array([0.2, 0.3, 0.4])
        np.testing.assert_allclose(
            mdl._mjMdl.model.body_inertia[mdl._payload_body_id], [0.2, 0.3, 0.4], atol=TOL_ANALYTIC
        )

    def test_cable_length_setter_moves_the_tendon_limit(self, mdl):
        mdl.cable_lengths = [0.9, 0.95, 1.0, 1.05]
        np.testing.assert_allclose(
            mdl._mjMdl.model.tendon_range[:, 1], [0.9, 0.95, 1.0, 1.05], atol=TOL_ANALYTIC
        )


class TestMujocoClosedLoop:
    """The cascade runs against the physics backend."""

    @pytest.mark.slow
    def test_simulate_keeps_the_formation_airborne(self, mdl):
        """The asset spawns slack, so the payload drops before the cables catch.

        That drop is a property of the hand-written MJCF (0.789 m of cable
        against a 1.0 m tendon limit), not of the controller, and the backend
        warns about it at load time. What is asserted here is only that the
        cascade recovers: the team stays above the ground, the payload settles
        and no state goes non-finite.
        """
        mdl.simulate(tf=6.0, payload_position=np.array([0.0, 0.0, 1.34]))
        assert np.all(np.isfinite(mdl.state.payload_position))
        assert np.all(np.isfinite(np.asarray(mdl.state.payload_orientation)))
        assert mdl.state.payload_position[2] > 0.2
        for i in range(mdl.num_quadrotors):
            assert np.all(np.isfinite(mdl.state.positions[i]))
            assert mdl.state.positions[i][2] > mdl.state.payload_position[2]

    def test_step_accepts_the_base_input_pipeline(self, mdl):
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.34]))
        weight = (mdl.payload_mass / mdl.num_quadrotors + mdl.masses[0]) * GRAVITY
        u = np.tile(np.array([0.0, 0.0, weight]), (mdl.num_quadrotors, 1))
        for _ in range(20):
            mdl.step(u)
        assert mdl.t > 0.0
        assert np.all(np.isfinite(mdl.state.payload_velocity))
