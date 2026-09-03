"""Tests for MultiQuadCsPayloadMujoco — MJCF wiring, state sync, station keeping."""

import numpy as np
import pytest

from tests.conftest import TOL_ANALYTIC, TOL_TIGHT
from udaan.manif import S2, SO3, TS2, TSO3
from udaan.models.multi_quad_cspayload import MultiQuadCsPayloadMujoco

pytestmark = pytest.mark.mujoco

# The packaged multi_quad_pointmass.xml has two quadrotors; 3 and 4 exercise the
# runtime MJCF generation path.
QUAD_COUNTS = [2, 3, 4]


class TestMujocoSetup:
    """Model loading, index resolution and parameters read back from the MJCF."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_instantiate(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        assert mdl.num_quadrotors == nQ
        assert mdl._mjMdl.model.nu == 4 * nQ
        assert len(mdl.state.positions) == nQ

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_params_read_from_mjcf(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        assert mdl.masses.shape == (nQ,)
        assert np.all(mdl.masses > 0.0)
        assert mdl.inertias.shape == (nQ, 3, 3)
        assert mdl.payload_mass > 0.0
        assert mdl.cable_lengths.shape == (nQ,)
        assert np.all(mdl.cable_lengths > 0.0)

    def test_state_types(self):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3)
        mdl.reset()
        for i in range(3):
            assert isinstance(mdl.state.orientations[i], SO3)
            assert isinstance(mdl.state.angular_velocities[i], TSO3)
            assert isinstance(mdl.state.cable_attitudes[i], S2)
            assert isinstance(mdl.state.cable_angular_velocities[i], TS2)

    def test_mass_setter_writes_into_mujoco(self):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3)
        mdl.masses = [1.1, 1.2, 1.3]
        for i, body_id in enumerate(mdl._quad_body_id):
            assert mdl._mjMdl.model.body_mass[body_id] == pytest.approx(mdl.masses[i])
        mdl.payload_mass = 0.42
        assert mdl._mjMdl.model.body_mass[mdl._payload_body_id] == pytest.approx(0.42)


class TestMujocoStateSync:
    """_query_latest_state pulls the right quantities out of MuJoCo."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_reset_places_payload(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        target = np.array([-1.0, 2.0, 0.5])
        mdl.reset(payload_position=target)
        np.testing.assert_allclose(mdl.state.payload_position, target, atol=TOL_ANALYTIC)
        assert mdl.t == 0.0
        for i in range(nQ):
            assert mdl.state.positions[i][2] > mdl.state.payload_position[2]

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_cable_attitudes_are_unit_and_point_at_the_load(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
        for i in range(nQ):
            q = np.asarray(mdl.state.cable_attitudes[i])
            assert np.linalg.norm(q) == pytest.approx(1.0, abs=TOL_TIGHT)
            direction = mdl.state.payload_position - mdl.state.positions[i]
            np.testing.assert_allclose(q, direction / np.linalg.norm(direction), atol=TOL_ANALYTIC)

    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_cable_omega_scales_with_a_longer_cable(self, k):
        """Same relative velocity, longer cable -> proportionally smaller omega.

        This is the structural cover for the ``1/length`` factor the retired
        model dropped. It compares two *different* cable lengths rather than
        recomputing ``q × Δv / l`` alongside the implementation, so it cannot
        be satisfied by any length-independent formula regardless of what the
        packaged MJCF happens to start at.

        (An earlier ``test_cable_omega_uses_the_inverse_length_factor`` was
        deleted: its ``expected`` restated the implementation's own expression,
        and its second assertion, ``l·(x/l) == x``, is an identity. It only
        discriminated because the shipped MJCF starts slack at ~0.83 m.)
        """
        payload_vel = np.array([0.0, 0.0, -1.0])
        recovered = {}
        for length in (1.0, 2.0):
            mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=3)
            mdl.reset(payload_position=np.array([0.0, 0.0, 1.0]))
            # Push quad k out to `length` along its own cable direction.
            qk = np.asarray(mdl.state.cable_attitudes[k])
            adr = mdl._quad_qposadr[k]
            mdl._mjMdl.data.qpos[adr : adr + 3] = mdl.state.payload_position - length * qk
            i = mdl._payload_qveladr
            mdl._mjMdl.data.qvel[i : i + 3] = payload_vel
            mdl._mjMdl.data.qvel[mdl._quad_qveladr[k] : mdl._quad_qveladr[k] + 3] = 0.0
            mdl._query_latest_state()
            omega = np.asarray(mdl.state.cable_angular_velocities[k])
            # ω must stay tangent to the sphere whatever the length.
            assert abs(float(np.dot(np.asarray(mdl.state.cable_attitudes[k]), omega))) < 1e-12
            recovered[length] = omega.copy()

        np.testing.assert_allclose(recovered[2.0], 0.5 * recovered[1.0], atol=TOL_ANALYTIC)
        assert np.linalg.norm(recovered[1.0]) > 1e-3


@pytest.mark.integration
class TestMujocoStationKeeping:
    """Closed-loop: the default per-quadrotor PDs hold the team steady."""

    @pytest.mark.parametrize("nQ", QUAD_COUNTS)
    def test_formation_is_held(self, nQ):
        mdl = MultiQuadCsPayloadMujoco(render=False, num_quadrotors=nQ)
        start = np.array([-1.0, 2.0, 0.5])
        mdl.simulate(tf=4.0, payload_position=start)

        assert mdl.t >= 4.0
        # The packaged MJCF starts with slack cables, so the payload sags as
        # they go taut; the horizontal position and every velocity must settle.
        np.testing.assert_allclose(mdl.state.payload_position[:2], start[:2], atol=0.1)
        assert np.linalg.norm(mdl.state.payload_velocity) < 0.1
        for i in range(nQ):
            np.testing.assert_allclose(mdl.state.positions[i], mdl._formation_targets[i], atol=0.1)
            assert np.linalg.norm(mdl.state.velocities[i]) < 0.1
